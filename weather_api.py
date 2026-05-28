import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone
from scipy.stats import norm
import pickle
import os

from market_filter import STATION_COORDS

# Open-Meteo public API - free, no API key required
# Documentation: https://open-meteo.com/en/docs
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# How many forecast days to request (today + next N days)
# Weather markets typically cover today and the next 1-2 days
FORECAST_DAYS = 3

RF_MODEL_PATH = "models/rf_weather.pkl"

def _load_rf_model():
    if os.path.exists(RF_MODEL_PATH):
        with open(RF_MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

# Fetch a URL with automatic retry on connection errors
# Waits delay seconds between attempts, doubling each time (exponential backoff)
def _fetch_with_retry(url: str, params: dict, retries: int = 3, delay: float = 2.0) -> dict:
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt < retries - 1:
                wait = delay * (2 ** attempt)
                print(f"[WeatherAPI] Retry {attempt + 1}/{retries} after {wait:.0f}s: {e}")
                time.sleep(wait)
            else:
                raise
    return {}

# Fetch the daily maximum temperature forecast for a specific location and date
# Uses Open-Meteo's daily temperature_2m_max variable, which is the closest equivalent to the Weather Underground "Daily High"
def fetch_forecast(lat: float, lon: float, target_date: str) -> dict:
    try:
        data = _fetch_with_retry(OPEN_METEO_URL, params={
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max",
            "forecast_days": FORECAST_DAYS,
            "timezone": "auto",
            "models": "best_match",
        })

        daily = data.get("daily", {})
        dates = daily.get("time", [])
        temps = daily.get("temperature_2m_max", [])

        if target_date not in dates:
            return {"date": target_date, "available": False}
        
        idx = dates.index(target_date)
        temp_max = temps[idx]

        if temp_max is None:
            return {"date": target_date, "available": False}
        
        # Open-Meteo does not provide percentile forecasts on the free tier
        # We approximate uncertainty using a simple heuristic based on forecast horizon:
        # Same day (T+0): +/- 1.0°C (model has strong observational constraint)
        # Next day (T+1): +/- 1.5°C
        # Two days (T+2): +/- 2.0°C
        today = datetime.now(timezone.utc).date()
        target = datetime.strptime(target_date, "%Y-%m-%d").date()
        horizon_days = (target - today).days
        uncertainty = 1.0 + max(0, horizon_days) * 0.5

        return {
            "date": target_date,
            "temp_max_c": round(temp_max, 1),
            "temp_max_c_low": round(temp_max - uncertainty, 1),
            "temp_max_c_high": round(temp_max + uncertainty, 1),
            "horizon_days": horizon_days,
            "available": True,
        }
    
    except requests.RequestException as e:
        print(f"[WeatherAPI] Error fetching forecast for ({lat}, {lon}): {e}")
        return {"date": target_date, "available": False}
    
# Fetch the current observed maximum temperature for today at a given location
# Uses Open-Meteo hourly data and takes the max temperature recorded so far today
def fetch_current_observation(lat: float, lon: float) -> dict:
    try:
        data = _fetch_with_retry(OPEN_METEO_URL, params={
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m",
            "forecast_days": 1,
            "timezone": "auto",
        })

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])

        if not times or not temps:
            return {"available": False}
        
        # Find current local hour at the station
        utc_offset_s = data.get("utc_offset_seconds", 0)
        now_local = datetime.now(timezone.utc) + timedelta(seconds=utc_offset_s)
        current_hour = now_local.hour

        # Take max temperature across all hours recorded so far today
        temps_so_far = [t for t, h in zip(temps, times) if t is not None and int(h[11:13]) <= current_hour]

        if not temps_so_far:
            return {"available": False}
        
        observed_max = max(temps_so_far)

        return {"observed_max_c": round(observed_max, 1), "hour_local": current_hour, "day_complete": current_hour >= 20, "available": True,}
    
    except requests.RequestException as e:
        print(f"[WeatherAPI] Observation error for ({lat}, {lon}): {e}")
        return {"available": False}
# Fetch temperature forecasts for all markets in the DataFrame
# For each unique combination (series_slug, event_date):
# makes one API call to Open-Meteo and attaches the forecast to all outcome for that event
def fetch_forecasts_for_markets(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    
    df = df.copy()
    df["forecast_temp_c"] = np.nan
    df["forecast_temp_c_low"] = np.nan
    df["forecast_temp_c_high"] = np.nan
    df["forecast_horizon_days"] = np.nan
    df["forecast_available"] = False
    df["observed_max_c"]  = np.nan
    df["observation_hour"] = np.nan
    df["day_complete"]    = False

    # Deduplicate: one API call per pair (station. date)
    unique_events = df[["series_slug", "event_date", "lat", "lon"]].drop_duplicates()

    print(f"[WeatherAPI] Fetching forecasts for {len(unique_events)} event(s)...")

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for _, event_row  in unique_events.iterrows():
        slug = event_row["series_slug"]
        target_date = str(event_row["event_date"])[:10]
        lat = event_row["lat"]
        lon = event_row["lon"]

        if pd.isna(lat) or pd.isna(lon):
            continue

        forecast = fetch_forecast(lat, lon, target_date)
        city = slug.replace("-daily-weather", "").replace("-", " ").title()

        if forecast["available"]:
            print(
                f"{city} {target_date}:"
                f"{forecast['temp_max_c']}°C"
                f"[{forecast['temp_max_c_low']} - {forecast['temp_max_c_high']}]"
                f"(T+{forecast['horizon_days']}d)"
            )
        else:
                print(f"{city} {target_date}: forecast not available")

        # Attach forecast to all outcomes for this event
        mask = (df["series_slug"] == slug) & (df["event_date"].astype(str).str[:10] == target_date)
        df.loc[mask, "forecast_temp_c"] = forecast.get("temp_max_c")
        df.loc[mask, "forecast_temp_c_low"] = forecast.get("temp_max_c_low")
        df.loc[mask, "forecast_temp_c_high"] = forecast.get("temp_max_c_high")
        df.loc[mask, "forecast_horizon_days"] = forecast.get("horizon_days")
        df.loc[mask, "forecast_available"] = forecast.get("available", False)

        # For today's markets, enrich with real observed max temperature
        if target_date == today_str:
            obs = fetch_current_observation(lat, lon)
            if obs["available"]:
                df.loc[mask, "observed_max_c"]  = obs["observed_max_c"]
                df.loc[mask, "observation_hour"] = obs["hour_local"]
                df.loc[mask, "day_complete"]     = obs["day_complete"]
    
    available = df["forecast_available"].sum()
    print(f"[WeatherAPI] Forecasts available for {available}/{len(df)} outcomes")
    return df

# Parse a temperature label from a Polymarket question into a (low, high) range in Celsius
def parse_temp_range(temp_str: str, unit: str = "C") -> tuple[float, float]:
    s = temp_str.strip().upper().replace("°", "").replace(" ", "")

    is_fahrenheit = "F" in s
    s = s.replace("C", "").replace("F", "")

    def to_celsius(t: float) -> float:
        return round((t - 32) * 5 / 9, 1) if is_fahrenheit else t
    
    if "ORHIGHER" in s:
        val = float(s.replace("ORHIGHER", ""))
        return (to_celsius(val), np.inf)
    
    if "ORBELOW" in s:
        val = float(s.replace("ORBELOW", ""))
        return(-np.inf, to_celsius(val))
    
    if "-" in s:
        parts = s.split("-")
        try:
            lo, hi = float(parts[0]), float(parts[1])
            return (to_celsius(lo), to_celsius(hi))
        except ValueError:
            pass

    try:
        val = float(s)
        return (to_celsius(val), to_celsius(val))
    except ValueError:
        return(np.nan, np.nan)
    
# Compute the model's probability that the actual temperature falls within the outcome's temperature range (given the forecast and its uncertainty)
def compute_model_probability(
        forecast_temp: float, 
        forecast_low: float, 
        forecast_high: float,
        temp_range_low: float,
        temp_range_high: float,
        ) -> float:
    
        sigma = max((forecast_high - forecast_low) / 3.29, 0.5) # Min 0.5°C to avoid overconfidence

        cdf_high = norm.cdf(temp_range_high + 0.5, loc=forecast_temp, scale=sigma) if not np.isinf(temp_range_high) else 1.0

        cdf_low = norm.cdf(temp_range_low - 0.5, loc=forecast_temp, scale=sigma) if not np.isinf(temp_range_low) else 0.0

        prob = cdf_high - cdf_low
        return float(np.clip(prob, 0.001, 0.999))

# Add model_prob column to the DataFrame
# For each outcome, computes the model's estimated probability using the forecast temperature and the outcome's temperature range
def add_model_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    
    df = df.copy()
    model_probs = []

    rf_model = _load_rf_model()

    for _, row in df.iterrows():
        # Use real observation for today if available, forecast otherwise
        if pd.notna(row.get("observed_max_c")) and not np.isnan(row.get("observed_max_c", np.nan)):
            obs_max = row["observed_max_c"]
            day_complete = row.get("day_complete", False)
            # If day is complete, the observed max is the final temperature
            # If day is ongoing, model uncertainty remains but observation constrains the lower bound
            effective_temp = obs_max
            effective_low = obs_max if day_complete else obs_max - 0.5
            effective_high = obs_max if day_complete else obs_max + 2.0
        elif not row.get("forecast_available", False) or pd.isna(row.get("forecast_temp_c")):
            model_probs.append(np.nan)
            continue
        else:
            effective_temp = row["forecast_temp_c"]
            effective_low = row["forecast_temp_c_low"]
            effective_high = row["forecast_temp_c_high"]

        temp_low, temp_high = parse_temp_range(row["temp_str"])

        if np.isnan(temp_low) and np.isnan(temp_high):
            model_probs.append(np.nan)
            continue

        prob = compute_model_probability(
            forecast_temp=effective_temp,
            forecast_low=effective_low,
            forecast_high=effective_high,
            temp_range_low=temp_low,
            temp_range_high=temp_high,
        )
        model_probs.append(prob)

    df["model_prob"] = model_probs

    if rf_model is not None:
        city_codes = df["series_slug"].astype("category").cat.codes
        X_live = pd.DataFrame({
            "entry_prob": df["prob_yes"],
            "model_prob": df["model_prob"],
            "forecast_horizon_days": df["forecast_horizon_days"].fillna(1),
            "is_buy_yes": 0,
            "city_code": city_codes,
            "is_observation": df["observed_max_c"].notna().astype(int),
            "forecast_temp_c": df["forecast_temp_c"].fillna(df["forecast_temp_c"].median()),
            "temp_diff": df["observed_max_c"] - df["forecast_temp_c"],
        })
        X_live = X_live.fillna(X_live.median())
        df["model_prob"] = rf_model.predict_proba(X_live)[:, 1]
        print(f"[WeatherAPI] RF model applied to {len(df)} outcomes")

    return df

if __name__ == "__main__":
    # Quick test with London coordinates
    from market_filter import fetch_active_weather_markets

    print("=== Testing WeatherAPI ===\n")

    df = fetch_active_weather_markets(series_slugs=["london-daily-weather"])
    if df.empty:
        print("No London markets found.")
    else:
        df = fetch_forecasts_for_markets(df)
        df = add_model_probabilities(df)

        print(f"\n{'Outcome':<22} {'Market':>8} {'Model':>8} {'Edge':>8}")
        print("-" * 52)
        for _, row in df.iterrows():
            if pd.isna(row.get("model_prob")):
                continue
            edge = row["model_prob"] - row["prob_yes"]
            print(
                f"{row['temp_str']:<22} "
                f"{row['prob_yes']:>8.3f} "
                f"{row['model_prob']:>8.3f} "
                f"{edge:>+8.3f}"
            )