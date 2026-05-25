import json
import os
import requests
import pandas as pd
from datetime import datetime, timezone
from dataclasses import dataclass, asdict

from signal_engine import Signal, generate_signals
from market_filter import fetch_active_weather_markets
from weather_api import fetch_forecasts_for_markets, add_model_probabilities

# Portfolio state is persisted to disk so it survives bot restarts
PORTFOLIO_PATH = "data/paper_portfolio_weather.json"

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"
 
INITIAL_BANKROLL = 50.0

# --- Position dataclass ---

@dataclass
class Position:
    # Market identification
    series_slug: str
    event_date: str
    question: str
    condition_id: str # Specific condition ID for this outcome
    temp_str: str

    # Trade details
    direction: str # "BUY YES" or "BUY NO"
    token_id: str # Token bought (yes or no)
    token_yes: str  # Always the YES token, regardless of direction
    entry_prob: float # Market price at entry
    model_prob: float # Model probability at entry
    edge: float # Raw edge at entry
    net_edge: float # Fee-adjusted edge at entry
    size_usd: float # Amount invested in USDC

    # Forecast data at entry (for calibration training later)
    forecast_temp_c: float
    forecast_horizon_days: int
    data_source: str # "observation" or "forecast"
    observed_max_c: float # NaN if not today's market

    # Source
    source_model: str # "base" or "calibration_rf"

    # Status
    status: str # "OPEN", "WON", "LOST", "EXPIRED"
    opened_at: str # ISO timestamp
    closed_at: str = ""
    pnl_usd: float = 0.0
    resolved_temp: float = None # Actual temperature at resolution

# --- Portfolio persistence ---

# Load portfolio state from disk, or create a fresh one
def _load_portfolio(initial_bankroll: float = INITIAL_BANKROLL) -> dict:
    if os.path.exists(PORTFOLIO_PATH):
        with open(PORTFOLIO_PATH, "r") as f:
            return json.load(f)
    return {
        "bankroll": initial_bankroll,
        "initial_bankroll": initial_bankroll,
        "positions": [],
        "closed_trades": [],
        "total_pnl": 0.0,
        "n_won": 0,
        "n_lost": 0,
        "n_expired": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }

# Save portfolio state to disk
def _save_portfolio(portfolio: dict) -> None:
    os.makedirs(os.path.dirname(PORTFOLIO_PATH), exist_ok=True)
    portfolio["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(PORTFOLIO_PATH, "w") as f:
        json.dump(portfolio, f, indent=2)

# --- Position management ---

# Check if we already have an open position for this outcome
# Prevents double-entering the same market
def _is_duplicate(signal: Signal, portfolio: dict) -> bool:
    for pos in portfolio["positions"]:
        if (pos["series_slug"] == signal.series_slug and
            pos["event_date"] == signal.event_date and
            pos["temp_str"] == signal.temp_str and
            pos["direction"] == signal.direction and
            pos["status"] == "OPEN"):
            return True
    return False

# Open new paper position for qualifying signals
def open_positions(signals: list[Signal], portfolio: dict, df_enriched: pd.DataFrame) -> list[Position]:
    opened = []

    for signal in signals:
        if _is_duplicate(signal, portfolio):
            continue

        if portfolio["bankroll"] < signal.position_size:
            print(f"[PaperTrader] Insufficient bankroll for {signal.temp_str} in "
                  f"{signal.series_slug} (need ${signal.position_size:.2f}, "
                  f"have ${portfolio['bankroll']:.2f})")
            continue

        # Get enriched row for this signal to capture forecast data
        row = df_enriched[
            (df_enriched["series_slug"] == signal.series_slug) &
            (df_enriched["event_date"].astype(str).str[:10] == signal.event_date) &
            (df_enriched["temp_str"] == signal.temp_str)
        ]

        forecast_temp = float(row["forecast_temp_c"].iloc[0]) if not row.empty else float("nan")
        observed_max  = float(row["observed_max_c"].iloc[0]) if not row.empty and pd.notna(row["observed_max_c"].iloc[0]) else float("nan")

        # Determine which token was bought
        token_id = signal.token_yes if signal.direction == "BUY_YES" else signal.token_no

        position = Position(
            series_slug=signal.series_slug,
            event_date=signal.event_date,
            question=signal.question,
            condition_id=row["condition_id"].iloc[0] if not row.empty else "",
            temp_str=signal.temp_str,
            direction=signal.direction,
            token_id=token_id,
            token_yes=signal.token_yes,
            entry_prob=signal.market_prob if signal.direction == "BUY_YES" else 1 - signal.market_prob,
            model_prob=signal.model_prob,
            edge=signal.edge,
            net_edge=signal.net_edge,
            size_usd=signal.position_size,
            forecast_temp_c=forecast_temp,
            forecast_horizon_days=signal.horizon_days,
            data_source=signal.data_source,
            observed_max_c=observed_max,
            source_model="base",
            status="OPEN",
            opened_at=datetime.now(timezone.utc).isoformat()
        )
        
        portfolio["bankroll"] -= signal.position_size
        portfolio["positions"].append(asdict(position))
        opened.append(position)

        city = signal.series_slug.replace("-daily-weather", "").replace("-", " ").title()
        print(f"[PaperTrader] OPEN {signal.direction} | {signal.temp_str} in {city} "
              f"on {signal.event_date} | ${signal.position_size} @ {signal.market_prob:.3f} "
              f"| edge {signal.net_edge:+.3f}")
        
    return opened

# --- Resolution ---

# Check if a market has resolved and which outcome won
def _fetch_market_resolution(condition_id: str) -> dict:
    try:
        resp = requests.get(
            f"{CLOB_URL}/markets/{condition_id}",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        resp.raise_for_status()
        market = resp.json()

        if not market.get("closed", False):
            return {"resolved": False}
        
        tokens = market.get("tokens", [])
        for token in tokens:
            if token.get("winner"):
                return {
                    "resolved": True,
                    "winner": token.get("outcome"),
                    "winning_token": token.get("token_id"),
                }
            
        return {"resolved": False}
    
    except requests.RequestException:
        return {"resolved": False}

# Check all open positions for resolution, updates portfolio state in-place and returns list of resolved positions
def check_resolutions(portfolio: dict) -> list[dict]:
    resolved = []
    still_open = []

    for pos in portfolio["positions"]:
        if pos["status"] != "OPEN":
            still_open.append(pos)
            continue

        # Skip if event date is in the future
        event_date = datetime.strptime(pos["event_date"], "%Y-%m-%d").date()
        today = datetime.now(timezone.utc).date()
        if event_date >= today:
            still_open.append(pos)
            continue

        # Fetch condition ID and check resolution
        condition_id = pos.get("condition_id")
        if not condition_id:
            still_open.append(pos)
            continue

        resolution = _fetch_market_resolution(condition_id)
        if not resolution["resolved"]:
            # Mark as expired if more than 3 days past event date
            if (today - event_date).days > 3:
                pos["status"] = "EXPIRED"
                pos["closed_at"] = datetime.now(timezone.utc).isoformat()
                pos["pnl_usd"] = -pos["size_usd"]
                portfolio["bankroll"] += 0 # Expired = full loss
                portfolio["n_expired"] += 1
                portfolio["total_pnl"] += pos["pnl_usd"]
                portfolio["closed_trades"].append(pos)
                resolved.append(pos)
                print(f"[PaperTrader] EXPIRED | {pos['temp_str']} | ${pos['pnl_usd']:.2f}")
            else:
                still_open.append(pos)
            continue

        # Market resolved - determine win/loss
        winning_token = resolution["winning_token"]
        if pos["direction"] == "BUY_YES":
            won = (pos["token_id"] == winning_token)
        else:  # BUY_NO
            won = (pos["token_yes"] != winning_token)
    
        if won:
            # Winning token redeems at $1 per share
            # Shares bought = size_usd / entry_prob
            shares = pos["size_usd"] / pos["entry_prob"]
            gross_pnl = shares - pos["size_usd"] # Gain before fees
            pos["status"] = "WON"
            pos["pnl_usd"] = round(gross_pnl, 4)
            portfolio["n_won"] += 1
        else:
            pos["status"] = "LOST"
            pos["pnl_usd"] = -pos["size_usd"]
            portfolio["n_lost"] += 1

        pos["closed_at"] = datetime.now(timezone.utc).isoformat()
        portfolio["bankroll"] += pos["size_usd"] + pos["pnl_usd"]
        portfolio["total_pnl"] += pos["pnl_usd"]
        portfolio["closed_trades"].append(pos)
        resolved.append(pos)

        city = pos["series_slug"].replace("-daily-weather", "").replace("-", " ").title()
        result = "WON" if won else "LOST"
        print(f"[PaperTrader] {result} | {pos['temp_str']} in {city} | "
              f"${pos['pnl_usd']:+.2f} | bankroll ${portfolio['bankroll']:.2f}")
        
    portfolio["positions"] = still_open
    return resolved

# --- Summary ---

# Print a formatted portfolio summary
def print_summary(portfolio: dict) -> None:
    n_open = len(portfolio["positions"])
    n_closed = len(portfolio["closed_trades"])
    n_won = portfolio["n_won"]
    n_lost = portfolio["n_lost"]
    n_expired = portfolio["n_expired"]
    win_rate = n_won / max(n_won + n_lost, 1)
    roi = (portfolio["bankroll"] - portfolio["initial_bankroll"]) / portfolio["initial_bankroll"]

    print(f"\n{'='*50}")
    print(f"PAPER PORTFOLIO - Weather Edge Bot")
    print(f"{'='*50}")
    print(f"Bankroll: ${portfolio['bankroll']:.2f} (initial: ${portfolio['initial_bankroll']:.2f})")
    print(f"Total P&L: ${portfolio['total_pnl']:+.2f} (ROI: {roi:+.1%})")
    print(f"Open: {n_open} positions")
    print(f"Closed: {n_closed} trades  (Won: {n_won} | Lost: {n_lost} | Expired: {n_expired})")
    print(f"Win rate: {win_rate:.1%}  (excluding expired)")
    print(f"Last update: {portfolio['last_updated'][:19]} UTC")
    print(f"{'='*50}\n")

    if portfolio["positions"]:
        print("Open positions:")
        for pos in portfolio["positions"]:
            city = pos["series_slug"].replace("-daily-weather", "").replace("-", " ").title()
            print(f"{pos['direction']:<9} {pos['temp_str']:<20} {city} {pos['event_date']} "
                  f"@ {pos['entry_prob']:.3f}  ${pos['size_usd']:.2f}")
        print()

# --- Main cycle ---

# Run one full paper trading cycle:
# 1. Load portfolio
# 2. Check resolutions
# 3. Fetch markets and generate signals
# 4. Open new positions
# 5. Save portfolio
def run_cycle(bankroll: float = INITIAL_BANKROLL) -> dict:
    print(f"\n[PaperTrader] Starting cycle at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    # STEP 1: Load portfolio
    portfolio = _load_portfolio(bankroll)

    # STEP 2: Check resolutions
    resolved = check_resolutions(portfolio)
    if resolved:
        print(f"[PaperTrader] {len(resolved)} position(s) resolved")

    # STEP 3: Fetch markets and generate signals
    df = fetch_active_weather_markets()
    if df.empty:
        print(f"[PaperTrader] No markets found.")
        _save_portfolio(portfolio)
        return portfolio
    
    df = fetch_forecasts_for_markets(df)
    df = add_model_probabilities(df)
    signals = generate_signals(df, bankroll=portfolio["bankroll"])

    print(f"[PaperTrader] {len(signals)} signal(s) found")

    # STEP 4: Open new positions
    if signals:
        opened = open_positions(signals, portfolio, df)
        print(f"[PaperTrader] {len(opened)} new position(s) opened")

    # STEP 5: Save and summarize
    portfolio["total_pnl"] = sum(t["pnl_usd"] for t in portfolio["closed_trades"])
    _save_portfolio(portfolio)
    print_summary(portfolio)

    return portfolio

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Weather Edge Paper Trader")
    parser.add_argument("--status", action="store_true", help="Print portfolio status and exit")
    parser.add_argument("--resolve-only", action="store_true", help="Check resolutions only")
    args = parser.parse_args()

    if args.status:
        portfolio = _load_portfolio()
        print_summary(portfolio)

    elif args.resolve_only:
        portfolio = _load_portfolio()
        resolved = check_resolutions(portfolio)
        print(f"Resolved: {len(resolved)}")
        _save_portfolio(portfolio)
        print_summary(portfolio)

    else:
        run_cycle()