import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone
from scipy.stats import norm
import pickle
import os

from market_filter import STATION_COORDS, RESOLUTION_STATIONS, STATION_IEM_NETWORKS

# Open-Meteo public API - free, no API key required
# Documentation: https://open-meteo.com/en/docs
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Historical reanalysis (ERA5) archive — independent of our forecast model.
# Available from ~event+1 day, so it is ready by the time markets resolve.
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Live METAR reports from the exact resolution station (free, no API key).
# This is the same thermometer Weather Underground publishes and Polymarket
# resolves on, unlike Open-Meteo hourly which is model-interpolated.
AVIATION_METAR_URL = "https://aviationweather.gov/api/data/metar"

# IEM daily summaries computed from station METARs (93% agreement with actual
# market resolutions over 507 historical trades, vs 66% for the ERA5 grid).
IEM_DAILY_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py"

# How many forecast days to request (today + next N days)
# Weather markets typically cover today and the next 1-2 days
FORECAST_DAYS = 3

RF_MODEL_PATH = "models/rf_weather.pkl"

_DEFAULT_FEATURES = ["entry_prob", "model_prob_gaussian", "forecast_horizon_days", "is_buy_yes",
                     "city_code", "is_observation", "forecast_temp_c", "temp_diff"]

def _load_rf_model() -> tuple:
    if os.path.exists(RF_MODEL_PATH):
        with open(RF_MODEL_PATH, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, tuple):
            return payload  # (model, feature_cols)
        return payload, _DEFAULT_FEATURES  # legacy single-object format
    return None, _DEFAULT_FEATURES

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
    
# Fetch the max temperature reported so far today (station local date) from the
# station's own METARs. Returns {"available": False} on any failure so the caller
# can fall back to Open-Meteo hourly data.
def fetch_metar_max_today(station: str, utc_offset_s: int) -> dict:
    try:
        resp = requests.get(AVIATION_METAR_URL, params={
            "ids": station, "format": "json", "hours": 30,
        }, timeout=15)
        resp.raise_for_status()
        reports = resp.json()
    except (requests.RequestException, ValueError):
        return {"available": False}

    if not isinstance(reports, list) or not reports:
        return {"available": False}

    today_local = (datetime.now(timezone.utc) + timedelta(seconds=utc_offset_s)).date()
    temps = []
    for rep in reports:
        obs_epoch = rep.get("obsTime")
        if obs_epoch is None:
            continue
        local_dt = datetime.fromtimestamp(obs_epoch, tz=timezone.utc) + timedelta(seconds=utc_offset_s)
        if local_dt.date() != today_local:
            continue
        if rep.get("temp") is not None:
            temps.append(float(rep["temp"]))
        # 6-hourly max group catches peaks between regular reports, but only
        # count it when its whole 6h window falls within today (local)
        if rep.get("maxT") is not None and (local_dt - timedelta(hours=6)).date() == today_local:
            temps.append(float(rep["maxT"]))

    if not temps:
        return {"available": False}
    return {"observed_max_c": round(max(temps), 1), "available": True}

# Fetch the current observed maximum temperature for today at a given location
# Prefers real METARs from the resolution station; falls back to Open-Meteo
# hourly data (model-interpolated) when no station is given or METAR fails
def fetch_current_observation(lat: float, lon: float, station: str = None) -> dict:
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
        source = "openmeteo"

        # Overlay real station METARs when available (same source the market resolves on)
        if station:
            metar = fetch_metar_max_today(station, utc_offset_s)
            if metar["available"]:
                observed_max = metar["observed_max_c"]
                source = "metar"

        return {"observed_max_c": round(observed_max, 1), "hour_local": current_hour, "day_complete": current_hour >= 20, "obs_source": source, "available": True,}
    
    except requests.RequestException as e:
        print(f"[WeatherAPI] Observation error for ({lat}, {lon}): {e}")
        return {"available": False}
# Fetch the IEM daily max (°C) for a station, computed from its own METARs.
# Matches the actual market resolution 93% of the time (vs 66% for ERA5).
def _fetch_iem_daily_max(station: str, event_date: str) -> float | None:
    net = STATION_IEM_NETWORKS.get(station)
    if net is None:
        return None
    network, iem_id = net
    y, m, d = event_date.split("-")
    try:
        resp = requests.get(IEM_DAILY_URL, params={
            "network": network, "stations": iem_id,
            "year1": y, "month1": int(m), "day1": int(d),
            "year2": y, "month2": int(m), "day2": int(d),
            "var": "max_temp_f", "format": "csv", "na": "blank",
        }, timeout=20)
        resp.raise_for_status()
        lines = resp.text.strip().splitlines()
        if len(lines) < 2:
            return None
        header = lines[0].split(",")
        val = lines[1].split(",")[header.index("max_temp_f")]
        if val:
            return round((float(val) - 32) * 5 / 9, 1)
    except (requests.RequestException, ValueError, IndexError):
        pass
    return None

# Fetch the actual observed daily max temperature (°C) for a resolved market.
# Used purely for instrumentation/auditing: stored in Position.resolved_temp so we can
# measure forecast bias per city (forecast_temp_c - resolved_temp) over time.
# Primary source: IEM daily summary from the exact resolution station's METARs.
# Fallback: ERA5 reanalysis at the station coords — a ~25km grid, NOT the exact
# station, so a residual 1-2°C gap can remain there. Returns None if unavailable.
def fetch_observed_max(series_slug: str, event_date: str) -> float | None:
    station = RESOLUTION_STATIONS.get(series_slug)
    lat, lon = STATION_COORDS.get(station, (None, None)) if station else (None, None)
    if lat is None or lon is None:
        return None

    try:
        target = datetime.strptime(event_date, "%Y-%m-%d").date()
    except ValueError:
        return None
    if target >= datetime.now(timezone.utc).date():
        return None  # event today/future: daily summaries not final yet

    iem_max = _fetch_iem_daily_max(station, event_date)
    if iem_max is not None:
        return iem_max

    try:
        data = _fetch_with_retry(OPEN_METEO_ARCHIVE_URL, params={
            "latitude": lat,
            "longitude": lon,
            "start_date": event_date,
            "end_date": event_date,
            "daily": "temperature_2m_max",
            "timezone": "auto",
        })
        temps = data.get("daily", {}).get("temperature_2m_max", [])
        if temps and temps[0] is not None:
            return round(float(temps[0]), 1)
    except requests.RequestException as e:
        print(f"[WeatherAPI] Observed-max fetch failed for {series_slug} {event_date}: {e}")
    return None

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
    df["observed_max_c"] = np.nan
    df["observation_hour"] = np.nan
    df["day_complete"] = False
    df["obs_source"] = ""

    # Deduplicate: one API call per pair (station. date)
    unique_events = df[["series_slug", "event_date", "lat", "lon", "station"]].drop_duplicates()

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
            obs = fetch_current_observation(lat, lon, station=event_row.get("station"))
            if obs["available"]:
                df.loc[mask, "observed_max_c"] = obs["observed_max_c"]
                df.loc[mask, "observation_hour"] = obs["hour_local"]
                df.loc[mask, "day_complete"] = obs["day_complete"]
                df.loc[mask, "obs_source"] = obs.get("obs_source", "")
    
    available = df["forecast_available"].sum()
    print(f"[WeatherAPI] Forecasts available for {available}/{len(df)} outcomes")
    return df

# Parse a temperature label from a Polymarket question into a (low, high) range in Celsius
# Returns (low_c, high_c, is_fahrenheit); is_fahrenheit tells the caller the market's
# native bin width (1°F vs 1°C) for the continuity correction
def parse_temp_range(temp_str: str) -> tuple[float, float, bool]:
    s = temp_str.strip().upper().replace("°", "").replace(" ", "")

    is_fahrenheit = "F" in s
    # US markets phrase ranges as "between 88-89°F"; strip the keyword before parsing numbers
    s = s.replace("BETWEEN", "")
    s = s.replace("C", "").replace("F", "")

    def to_celsius(t: float) -> float:
        return round((t - 32) * 5 / 9, 1) if is_fahrenheit else t
    
    if "ORHIGHER" in s:
        val = float(s.replace("ORHIGHER", ""))
        return (to_celsius(val), np.inf, is_fahrenheit)

    if "ORBELOW" in s:
        val = float(s.replace("ORBELOW", ""))
        return(-np.inf, to_celsius(val), is_fahrenheit)

    if "-" in s:
        parts = s.split("-")
        try:
            lo, hi = float(parts[0]), float(parts[1])
            return (to_celsius(lo), to_celsius(hi), is_fahrenheit)
        except ValueError:
            pass

    try:
        val = float(s)
        return (to_celsius(val), to_celsius(val), is_fahrenheit)
    except ValueError:
        return(np.nan, np.nan, is_fahrenheit)
    
# Compute the model's probability that the actual temperature falls within the outcome's temperature range (given the forecast and its uncertainty)
def compute_model_probability(
        forecast_temp: float,
        forecast_low: float,
        forecast_high: float,
        temp_range_low: float,
        temp_range_high: float,
        day_complete: bool = False,
        half_bin: float = 0.5,
        ) -> float:

        # When the day is complete, observed_max is final — use tight sigma (source diff ~0.1°C)
        # Otherwise keep minimum 0.5°C to avoid overconfidence on forecasts
        min_sigma = 0.1 if day_complete else 0.5
        sigma = max((forecast_high - forecast_low) / 3.29, min_sigma)

        # half_bin: half the market's bin width in °C — 0.5 for 1°C markets,
        # ~0.278 for 1°F markets. The resolution source rounds to the nearest
        # whole degree in the market's native unit, so the true range extends
        # half a bin beyond each labeled boundary.
        cdf_high = norm.cdf(temp_range_high + half_bin, loc=forecast_temp, scale=sigma) if not np.isinf(temp_range_high) else 1.0

        cdf_low = norm.cdf(temp_range_low - half_bin, loc=forecast_temp, scale=sigma) if not np.isinf(temp_range_low) else 0.0

        prob = cdf_high - cdf_low
        return float(np.clip(prob, 0.001, 0.999))

# Add model_prob column to the DataFrame
# For each outcome, computes the model's estimated probability using the forecast temperature and the outcome's temperature range
def add_model_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    
    df = df.copy()
    model_probs = []

    rf_model, rf_features = _load_rf_model()

    for _, row in df.iterrows():
        # Use real observation for today if available, forecast otherwise
        is_complete = False
        if pd.notna(row.get("observed_max_c")) and not np.isnan(row.get("observed_max_c", np.nan)):
            obs_max = row["observed_max_c"]
            is_complete = bool(row.get("day_complete", False))
            effective_temp = obs_max
            effective_low = obs_max if is_complete else obs_max - 0.5
            effective_high = obs_max if is_complete else obs_max + 2.0
        elif not row.get("forecast_available", False) or pd.isna(row.get("forecast_temp_c")):
            model_probs.append(np.nan)
            continue
        else:
            effective_temp = row["forecast_temp_c"]
            effective_low = row["forecast_temp_c_low"]
            effective_high = row["forecast_temp_c_high"]

        temp_low, temp_high, is_fahrenheit = parse_temp_range(row["temp_str"])

        if np.isnan(temp_low) and np.isnan(temp_high):
            model_probs.append(np.nan)
            continue

        prob = compute_model_probability(
            forecast_temp=effective_temp,
            forecast_low=effective_low,
            forecast_high=effective_high,
            temp_range_low=temp_low,
            temp_range_high=temp_high,
            day_complete=is_complete,
            half_bin=0.5 * 5 / 9 if is_fahrenheit else 0.5,
        )
        model_probs.append(prob)

    df["model_prob_gaussian"] = model_probs  # original gaussian, before RF refinement
    df["model_prob"] = model_probs

    if rf_model is not None:
        city_codes = df["series_slug"].astype("category").cat.codes
        all_cols = pd.DataFrame({
            "entry_prob": df["prob_yes"],
            "model_prob_gaussian": df["model_prob_gaussian"],
            "forecast_horizon_days": df["forecast_horizon_days"].fillna(1),
            "is_buy_yes": 0,
            "city_code": city_codes,
            "is_observation": df["observed_max_c"].notna().astype(int),
            "forecast_temp_c": df["forecast_temp_c"].fillna(df["forecast_temp_c"].median()),
            "temp_diff": df["observed_max_c"] - df["forecast_temp_c"],
        })
        X_live = all_cols[rf_features].fillna(all_cols[rf_features].median())
        df["model_prob"] = rf_model.predict_proba(X_live)[:, 1]
        print(f"[WeatherAPI] RF model applied ({len(rf_features)} features: {rf_features})")

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