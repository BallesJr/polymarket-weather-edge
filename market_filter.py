import requests
import pandas as pd
import json
import time
from datetime import datetime, timezone, timedelta

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"

# Known active weather series slugs discovered empirically
# The /series endpoint does not reliably filter by type, so we maintain a hardcoded list of confirmed daily temperature series
# Maps series slug to the Weather Underground station used for resolution
# Source: confirmed from Polymarket market descriptions (add new cities here as they are discovered)
RESOLUTION_STATIONS = {
    "london-daily-weather": "EGLC", # London City Airport
    "nyc-daily-weather": "KLGA", # NYC LaGuardia Airport
    "warsaw-daily-weather": "EPWA", # Warsaw Chopin Airport
    "tokyo-daily-weather": "RJTT", # Tokyo Haneda
    "seoul-daily-weather": "RKSI", # Seoul Incheon
    "mexico-city-daily-weather": "MMMX", # Mexico City Airport
    "wellington-daily-weather": "NZWN", # Wellington Airport
    "taipei-daily-weather": "RCSS", # Taipei Songshan
    "shanghai-daily-weather": "ZSPD", # Shanghai Pudong
    "singapore-daily-weather": "WSSS", # Singapore Changi
    "chicago-daily-weather": "KORD", # Chicago O'Hare
    "miami-daily-weather": "KMIA", # Miami Airport
    "milan-daily-weather": "LIMC", # Milan Malpensa
    "paris-daily-weather": "LFPG", # Paris CDG
    "austin-daily-weather": "KAUS", # Austin-Bergstrom
    "boston-daily-weather": "KBOS", # Boston Logan
    "houston-daily-weather": "KHOU", # Houston Hobby
    # TODO: Add Tel Aviv (LLBG) - uses NOAA instead of Weather Underground
    # TODO: Add Hong Kong (HKO) - uses HK Observatory instead of Weather Underground
}

# Coordinates for each station (used by weather_api.py to fetch Open-Meteo forecasts)
# Aligned to the exact Weather Underground station to minimize forecast vs resolutions divergence
STATION_COORDS = {
    "EGLC": (51.5048,   0.0495), # London City Airport
    "KLGA": (40.7772, -73.8726), # NYC LaGuardia
    "EPWA": (52.1657,  20.9671), # Warsaw Chopin
    "RJTT": (35.5494, 139.7798), # Tokyo Haneda
    "RKSI": (37.4691, 126.4505), # Seoul Incheon
    "MMMX": (19.4363, -99.0721), # Mexico City Airport
    "NZWN": (-41.3272, 174.8052), # Wellington Airport
    "RCSS": (25.0694, 121.5522), # Taipei Songshan
    "ZSPD": (31.1443, 121.8083), # Shanghai Pudong
    "WSSS": (1.3644,  103.9915), # Singapore Changi
    "KORD": (41.9742, -87.9073), # Chicago O'Hare
    "KMIA": (25.7959, -80.2870), # Miami Airport
    "LIMC": (45.6306,   8.7281), # Milan Malpensa
    "LFPG": (49.0097,   2.5479), # Paris CDG
    "KAUS": (30.1975, -97.6664), # Austin-Bergstrom
    "KHOU": (29.6454, -95.2789), # Houston Hobby
    "VILK": (26.7606,  80.8893), # Lucknow Airport
    "KBOS": (42.3656, -71.0096), # Boston Logan
}

# Fetch active events for a given series slug via Gamma API
def _fetch_events_by_series(series_slug: str) -> list[dict]:
    city = series_slug.replace("-daily-weather", "")
    events = []
    today = datetime.now(timezone.utc).date()

    for days_ahead in range(3): # Today, tomorrow, next tomorrow
        target = today + timedelta(days=days_ahead)
        date_str = target.strftime("%B-%-d-%Y").lower() # "april-16-2026"
        slug = f"highest-temperature-in-{city}-on-{date_str}"

        try:
            resp = requests.get(
                f"{GAMMA_URL}/events",
                params={"slug": slug},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15
            )
            resp.raise_for_status()
            data = resp.json()
            if data:
                events.extend(data)
        except requests.RequestException as e:
            print(f"[Warning] Could not fetch series {series_slug}: {e}")
        
    return events
    
# Parse all individual outcome markets from a neg_risk temperature event
# Each event has around 10/12 binary markets, one per temperature range
# Extract question text, token IDs, current market probability, and the temperature value implied by each outcome
def _parse_market_outcomes(event: dict, series_slug: str) -> list[dict]:
    rows = []
    event_title = event.get("title", "")
    event_date = event.get("eventDate") or event.get("endDate", "")[:10]
    neg_risk = event.get("negRisk", False)
    station = RESOLUTION_STATIONS.get(series_slug, "")
    coords = STATION_COORDS.get(station, (None, None))

    for m in event.get("markets", []):
        if not m.get("acceptingOrders", False):
            continue
        if not m.get("enableOrderBook", True):
            continue

        question = m.get("question", "")

        # Extract current market prices
        try:
            prices = json.loads(m.get("outcomePrices", "[]"))
            prob_yes = float(prices[0]) if prices else None
        except (json.JSONDecodeError, ValueError):
            prob_yes = None

        # Extract token IDs (yes and no)
        try:
            token_ids = json.loads(m.get("clobTokenIds", "[]"))
            token_yes =  token_ids[0] if token_ids else None
            token_no = token_ids[1] if len(token_ids) > 1 else None
        except (json.JSONDecodeError, ValueError):
            token_yes = token_no = None

        if prob_yes is None or token_yes is None:
            continue
 
        rows.append({
            "series_slug": series_slug,
            "event_id": event.get("id"),
            "event_title": event_title,
            "event_date": event_date,
            "market_id": m.get("id"),
            "condition_id": m.get("conditionId"),
            "question": question,
            "temp_str": _extract_temp_from_question(question), # e.g. "14°C", "84-85°F", "15°C or higher"         
            "prob_yes": prob_yes, # Current market implied probability
            "token_yes": token_yes,
            "token_no": token_no,
            "neg_risk": neg_risk,
            "liquidity": float(m.get("liquidityNum") or m.get("liquidity") or 0),
            "volume_24h": float(m.get("volume24hr") or 0),
            "station": station,
            "lat": coords[0],
            "lon": coords[1],
            "resolution_source": m.get("resolutionSource", ""),
        })
 
    return rows

# Extract the temperature label from a question string
def _extract_temp_from_question(question: str) -> str:
    q = question.lower()
    # Find the "be X" pattern: "Will the highest temperature in London 14°C on..."
    if "be" in q:
        after_be = question[q.index(" be ") + 4:]
        # Take everything up to " on "
        if " on " in after_be.lower():
            return after_be[:after_be.lower().index(" on ")].strip()
        return after_be.strip()
    return ""

# Main function: fetch all active temperature across all known cities
def fetch_active_weather_markets(series_slugs: list[str] = None, min_liquidity: float = 50.0, min_prob: float = 0.02, max_prob: float = 0.98,) -> pd.DataFrame:
    if series_slugs is None:
        series_slugs = list(RESOLUTION_STATIONS.keys())
    all_rows = []
    print(f"[MarketFilter] Fetching active temperature markets...")

    for slug in series_slugs:
        events = _fetch_events_by_series(slug)
        if not events:
            continue
        
        city_rows = []
        for event in events:
            city_rows.extend(_parse_market_outcomes(event, slug))

        if city_rows:
            city = slug.replace("-daily-weather", "").replace("-", " ").title()
            print (f"{city}: {len(events)} events, {len(city_rows)} outcomes")
            all_rows.extend(city_rows)

        time.sleep(0.2) # Soft rate limiting

    if not all_rows:
        print("[Market filter] No weather markets found.")
        return pd.DataFrame()
        
    df = pd.DataFrame(all_rows)

    # Filter to tradeable outcomes
    df = df[
        (df["prob_yes"] >= min_prob) &
        (df["prob_yes"] <= max_prob) &
        (df["liquidity"] >= min_liquidity) &
        (df["token_yes"].notna()) &
        (df["lat"].notna())
    ].copy()

    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.sort_values(["event_date", "series_slug", "prob_yes"], ascending=[True, True, False])
    df = df.reset_index(drop=True)

    n_cities = df["series_slug"].nunique()
    print(f"[MarketFilter] {len(df)} tradeable outcomes across {n_cities} cities")
    return df
    
# Enrich DataFrame with live midpoint prices from the CLOB API
# More accurate than outcomePrices from Gamma (which can be stale)
def fetch_current_midpoints(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    
    df = df.copy()
    midpoints = []

    for _, row in df.iterrows():
        try:
            resp = requests.get(
                f"{CLOB_URL}/midpoint",
                params={"token_id": row["token_yes"]},
                timeout=5,
            )
            mid = float(resp.json().get("mid", row["prob_yes"]))
        except Exception:
            mid = row["prob_yes"] # Fallback to Gamma price
        midpoints.append(mid)
        time.sleep(0.05)

    df["prob_yes_live"] = midpoints
    return df

if __name__ == "__main__":
    # Quick test: print all active temperature markets
    df = fetch_active_weather_markets()

    if df.empty:
        print("No markets found. Check API connectivity.")
    else:
        print(f"\n{'City':<18} {'Date':<12} {'Outcome':<22} {'Prob':>6} {'Liquidity':>10}")
        print("-" * 40)
        for _, row in df.iterrows():
            city = row["series_slug"].replace("-daily-weather", "").replace("-", " ").title()
            print(f"{city:<18} {str(row['event_date'])[:10]:<12} {row['temp_str']:<22} {row['prob_yes']:>6.2f} {row['liquidity']:>10.0f}")