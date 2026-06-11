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
    "paris-daily-weather": "LFPB", # Paris Le Bourget (NOT CDG: market description points to LFPB)
    "austin-daily-weather": "KAUS", # Austin-Bergstrom
    "boston-daily-weather": "KBOS", # Boston Logan
    "houston-daily-weather": "KHOU", # Houston Hobby
    "lucknow-daily-weather": "VILK", # Lucknow Airport
    "beijing-daily-weather": "ZBAA", # Beijing Capital International Airport
    "munich-daily-weather": "EDDM", # Munich Airport
    "denver-daily-weather": "KBKF", # Buckley Space Force Base (Fahrenheit)
    "shenzhen-daily-weather": "ZGSZ", # Shenzhen Bao'an International Airport
    "amsterdam-daily-weather": "EHAM", # Amsterdam Airport Schiphol
    "madrid-daily-weather": "LEMD", # Adolfo Suárez Madrid-Barajas Airport
    # Batch added 2026-06-11, stations taken from each market's own description and
    # validated against actual resolutions (IEM daily max vs winning bin, 06-02..06-09):
    "ankara-daily-weather": "LTAC", # Ankara Esenboga (8/8)
    "buenos-aires-daily-weather": "SAEZ", # Buenos Aires Ezeiza / Ministro Pistarini (8/8)
    "busan-daily-weather": "RKPK", # Busan Gimhae (8/8)
    "cape-town-daily-weather": "FACT", # Cape Town International (8/8)
    "chengdu-daily-weather": "ZUUU", # Chengdu Shuangliu (8/8)
    "chongqing-daily-weather": "ZUCK", # Chongqing Jiangbei (8/8)
    "guangzhou-daily-weather": "ZGGG", # Guangzhou Baiyun (8/8)
    "helsinki-daily-weather": "EFHK", # Helsinki Vantaa (8/8)
    "istanbul-daily-weather": "LTFM", # Istanbul Airport, NOAA-sourced (8/8)
    "jeddah-daily-weather": "OEJN", # Jeddah King Abdulaziz (8/8)
    "kuala-lumpur-daily-weather": "WMKK", # Kuala Lumpur International (8/8)
    "manila-daily-weather": "RPLL", # Manila Ninoy Aquino (8/8)
    "qingdao-daily-weather": "ZSQD", # Qingdao Jiaodong (8/8)
    "sao-paulo-daily-weather": "SBGR", # Sao Paulo Guarulhos (8/8)
    "seattle-daily-weather": "KSEA", # Seattle-Tacoma, Fahrenheit (7/8, like Seoul)
    "tel-aviv-daily-weather": "LLBG", # Tel Aviv Ben Gurion, NOAA-sourced (8/8)
    "toronto-daily-weather": "CYYZ", # Toronto Pearson (8/8)
    "wuhan-daily-weather": "ZHHH", # Wuhan Tianhe (8/8)
    # Skipped after the same validation: hong-kong (resolves on the HK Observatory,
    # not an airport METAR), atlanta KATL (4/8 agreement), dallas KDAL (6/8),
    # moscow UUWW (6/8), karachi OPMR (no METAR or IEM data)
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
    "LFPB": (48.9694,   2.4414), # Paris Le Bourget
    "KAUS": (30.1975, -97.6664), # Austin-Bergstrom
    "KHOU": (29.6454, -95.2789), # Houston Hobby
    "VILK": (26.7606,  80.8893), # Lucknow Airport
    "KBOS": (42.3656, -71.0096), # Boston Logan
    "ZBAA": (40.0799, 116.6031), # Beijing Capital International Airport
    "EDDM": (48.3538, 11.7861), # Munich Airport
    "KBKF": (39.7169, -104.7519), # Buckley Space Force Base, Denver
    "ZGSZ": (22.6397, 113.8105), # Shenzhen Bao'an International Airport
    "EHAM": (52.3105, 4.7683), # Amsterdam Airport Schiphol
    "LEMD": (40.4719, -3.5626), # Adolfo Suárez Madrid-Barajas Airport
    # Coordinates below come from each station's own METAR reports (aviationweather.gov)
    "LTAC": (40.128, 32.995), # Ankara Esenboga
    "SAEZ": (-34.822, -58.536), # Buenos Aires Ezeiza
    "RKPK": (35.179, 128.938), # Busan Gimhae
    "FACT": (-33.965, 18.602), # Cape Town International
    "ZUUU": (30.576, 103.950), # Chengdu Shuangliu
    "ZUCK": (29.718, 106.639), # Chongqing Jiangbei
    "ZGGG": (23.392, 113.307), # Guangzhou Baiyun
    "EFHK": (60.327, 24.957), # Helsinki Vantaa
    "LTFM": (41.262, 28.740), # Istanbul Airport
    "OEJN": (21.685, 39.166), # Jeddah King Abdulaziz
    "WMKK": (2.747, 101.714), # Kuala Lumpur International
    "RPLL": (14.507, 121.004), # Manila Ninoy Aquino
    "ZSQD": (36.362, 120.087), # Qingdao Jiaodong
    "SBGR": (-23.432, -46.469), # Sao Paulo Guarulhos
    "KSEA": (47.4447, -122.3144), # Seattle-Tacoma
    "LLBG": (32.011, 34.887), # Tel Aviv Ben Gurion
    "CYYZ": (43.679, -79.629), # Toronto Pearson
    "ZHHH": (30.783, 114.205), # Wuhan Tianhe
}

# IEM (Iowa Environmental Mesonet) network + station id for each resolution station
# Used to fetch daily max computed from the station's own METARs — the same data
# Weather Underground publishes and Polymarket resolves on (93% agreement vs 66% for ERA5)
# US networks use the 3-letter FAA id (no K prefix); international use COUNTRY__ASOS
# New Zealand is not in IEM: NZWN intentionally absent (falls back to ERA5)
STATION_IEM_NETWORKS = {
    "EGLC": ("GB__ASOS", "EGLC"), # London City
    "KLGA": ("NY_ASOS", "LGA"), # NYC LaGuardia
    "EPWA": ("PL__ASOS", "EPWA"), # Warsaw Chopin
    "RJTT": ("JP__ASOS", "RJTT"), # Tokyo Haneda
    "RKSI": ("KR__ASOS", "RKSI"), # Seoul Incheon
    "MMMX": ("MX__ASOS", "MMMX"), # Mexico City
    "RCSS": ("TW__ASOS", "RCSS"), # Taipei Songshan
    "ZSPD": ("CN__ASOS", "ZSPD"), # Shanghai Pudong
    "WSSS": ("SG__ASOS", "WSSS"), # Singapore Changi
    "KORD": ("IL_ASOS", "ORD"), # Chicago O'Hare
    "KMIA": ("FL_ASOS", "MIA"), # Miami
    "LIMC": ("IT__ASOS", "LIMC"), # Milan Malpensa
    "LFPB": ("FR__ASOS", "LFPB"), # Paris Le Bourget
    "KAUS": ("TX_ASOS", "AUS"), # Austin-Bergstrom
    "KBOS": ("MA_ASOS", "BOS"), # Boston Logan
    "KHOU": ("TX_ASOS", "HOU"), # Houston Hobby
    "VILK": ("IN__ASOS", "VILK"), # Lucknow
    "ZBAA": ("CN__ASOS", "ZBAA"), # Beijing Capital
    "EDDM": ("DE__ASOS", "EDDM"), # Munich
    "KBKF": ("CO_ASOS", "BKF"), # Buckley SFB, Denver
    "ZGSZ": ("CN__ASOS", "ZGSZ"), # Shenzhen Bao'an
    "EHAM": ("NL__ASOS", "EHAM"), # Amsterdam Schiphol
    "LEMD": ("ES__ASOS", "LEMD"), # Madrid Barajas
    "LTAC": ("TR__ASOS", "LTAC"), # Ankara Esenboga
    "SAEZ": ("AR__ASOS", "SAEZ"), # Buenos Aires Ezeiza
    "RKPK": ("KR__ASOS", "RKPK"), # Busan Gimhae
    "FACT": ("ZA__ASOS", "FACT"), # Cape Town International
    "ZUUU": ("CN__ASOS", "ZUUU"), # Chengdu Shuangliu
    "ZUCK": ("CN__ASOS", "ZUCK"), # Chongqing Jiangbei
    "ZGGG": ("CN__ASOS", "ZGGG"), # Guangzhou Baiyun
    "EFHK": ("FI__ASOS", "EFHK"), # Helsinki Vantaa
    "LTFM": ("TR__ASOS", "LTFM"), # Istanbul Airport
    "OEJN": ("SA__ASOS", "OEJN"), # Jeddah King Abdulaziz
    "WMKK": ("MY__ASOS", "WMKK"), # Kuala Lumpur International
    "RPLL": ("PH__ASOS", "RPLL"), # Manila Ninoy Aquino
    "ZSQD": ("CN__ASOS", "ZSQD"), # Qingdao Jiaodong
    "SBGR": ("BR__ASOS", "SBGR"), # Sao Paulo Guarulhos
    "KSEA": ("WA_ASOS", "SEA"), # Seattle-Tacoma
    "LLBG": ("IL__ASOS", "LLBG"), # Tel Aviv Ben Gurion
    "CYYZ": ("CA_ON_ASOS", "CYYZ"), # Toronto Pearson
    "ZHHH": ("CN__ASOS", "ZHHH"), # Wuhan Tianhe
}

# IANA timezone for each station, so local date/hour can be computed without
# depending on an Open-Meteo response (markets resolve on the station's local day)
STATION_TZ = {
    "EGLC": "Europe/London",
    "KLGA": "America/New_York",
    "EPWA": "Europe/Warsaw",
    "RJTT": "Asia/Tokyo",
    "RKSI": "Asia/Seoul",
    "MMMX": "America/Mexico_City",
    "NZWN": "Pacific/Auckland",
    "RCSS": "Asia/Taipei",
    "ZSPD": "Asia/Shanghai",
    "WSSS": "Asia/Singapore",
    "KORD": "America/Chicago",
    "KMIA": "America/New_York",
    "LIMC": "Europe/Rome",
    "LFPB": "Europe/Paris",
    "KAUS": "America/Chicago",
    "KHOU": "America/Chicago",
    "VILK": "Asia/Kolkata",
    "KBOS": "America/New_York",
    "ZBAA": "Asia/Shanghai",
    "EDDM": "Europe/Berlin",
    "KBKF": "America/Denver",
    "ZGSZ": "Asia/Shanghai",
    "EHAM": "Europe/Amsterdam",
    "LEMD": "Europe/Madrid",
    "LTAC": "Europe/Istanbul",
    "SAEZ": "America/Argentina/Buenos_Aires",
    "RKPK": "Asia/Seoul",
    "FACT": "Africa/Johannesburg",
    "ZUUU": "Asia/Shanghai",
    "ZUCK": "Asia/Shanghai",
    "ZGGG": "Asia/Shanghai",
    "EFHK": "Europe/Helsinki",
    "LTFM": "Europe/Istanbul",
    "OEJN": "Asia/Riyadh",
    "WMKK": "Asia/Kuala_Lumpur",
    "RPLL": "Asia/Manila",
    "ZSQD": "Asia/Shanghai",
    "SBGR": "America/Sao_Paulo",
    "KSEA": "America/Los_Angeles",
    "LLBG": "Asia/Jerusalem",
    "CYYZ": "America/Toronto",
    "ZHHH": "Asia/Shanghai",
}

# Fetch active events for a given series slug via Gamma API
def _fetch_events_by_series(series_slug: str) -> list[dict]:
    city = series_slug.replace("-daily-weather", "")
    events = []
    today = datetime.now(timezone.utc).date()

    for days_ahead in range(3): # Today, tomorrow, next tomorrow
        target = today + timedelta(days=days_ahead)
        date_str = f"{target.strftime('%B').lower()}-{target.day}-{target.year}"  # "april-16-2026"
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
    # Match " be " with spaces: a bare "be" also matches inside words like "Beijing"
    if " be " in q:
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