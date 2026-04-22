import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone

from market_filter import fetch_active_weather_markets
from weather_api import fetch_forecasts_for_markets, add_model_probabilities

# --- Configuration ---

# Minimum edge required to generate a signal
# Edge = model_prob - market_prob (positive = BUY YES, negative = BUY NO)
# A higher threshold means fewer but higher-confidence trades
# Threshold is adjusted by forecast horizon: further out = higher bar required
MIN_EDGE_TODAY = 0.08 # T+0: observation-based, high confidence
MIN_EDGE_TOMORROW = 0.12 # T+1: forecast, moderate confidence
MIN_EDGE_D2 = 0.18 # T+2: forecast, lower confidence, need bigger edge

# Minimum liquidity (USDC) required to trade an outcome
# Below this, market impact would be too large for small position sizes
MIN_LIQUIDITY = 200.0

# Fee rate for Weather category markets (confirmed from Polymarket docs)
# fee = feeRate * C * p * (1 - p), where C = shares, p = price
# As a taker, we pay this fee on entry and exit
WEATHER_FEE_RATE = 0.05
 
# Half-Kelly fraction: conservative position sizing to limit variance
KELLY_FRACTION = 0.5
 
# Maximum position size as fraction of bankroll per trade
MAX_POSITION_FRACTION = 0.10

# --- Signal dataclass ---

@dataclass
class Signal:
    # Market identification
    series_slug: str
    event_date: str
    question: str
    temp_str: str

    # Direction: "BUY_YES" or "BUY_NO"
    direction: str

    # Prices:
    market_prob: float # Current market implied probability for YES
    model_prob: float # Model's estimated probability for YES
    edge: float # model_prob - market_prob

    # Fee-adjusted edge: net edge after paying taker fees on entry + exit
    net_edge: float

    # Kelly position sizing
    kelly_fraction: float # Full Kelly fraction
    position_size: float # Recommended position size in USDC (Half-Kelly, capped)

    # Data source used for this signal
    data_source: str # "observation" (today) or "forecast" (future)
    horizon_days: int # 0 = today, 1 = tomorrow, 2 = day after tomorrow

    # Market metadata
    token_yes: str
    token_no: str
    neg_risk: bool
    liquidity: float

    # Confidence tier based on edge and horizon
    confidence: str # "HIGH", "MEDIUM", "LOW"

    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

# Compute the taker fee as a fraction of trade value for a given probability
# fee_fraction = feeRate * p * (1-p) (for a roundtrip (entry + exit), multiply by 2)
def _compute_fee(prob: float) -> float:
    return WEATHER_FEE_RATE * prob * (1 - prob)

# Adjust raw edge for taker fees
# For BUY YES at prob p:
# - Entry fee: feeRate * p * (1 - p)
# - Exit (sell or resolution): no fee on resolution, fee on early sell. We assume one roundtrip fee
def _compute_net_edge(edge: float, entry_prob: float) -> float:
    fee = _compute_fee(entry_prob)
    return edge - 2 * fee

# Compute the full Kelly fraction for a binary bet
def _compute_kelly(model_prob: float, market_prob: float, direction: str) -> float:
    if direction == "BUY_YES":
        b = (1 - market_prob) / market_prob # Net odds for YES
        q = model_prob # Model probability of YES
        p = 1 - q # Model probability of NO
    else: # BUY_NO
        no_price = 1 - market_prob
        b = (1 - no_price) / no_price # Net odds for NO
        q = 1 - model_prob # Model probability of NO
        p = model_prob # Model probability of YES

    kelly = (b * q - p) / b
    return float(max(kelly, 0.0)) # Never negative

# Return the minimum edge threshold for a given forecast horizon
def _get_threshold(horizon_days: int) -> float:
    if horizon_days == 0:
        return MIN_EDGE_TODAY
    elif horizon_days == 1:
        return MIN_EDGE_TOMORROW
    else:
        return MIN_EDGE_D2
    
# Classify signal confidence based on net edge and horizon
def _get_confidence(net_edge: float, horizon_days: int) -> str:
    threshold = _get_threshold(horizon_days)
    if net_edge >= threshold * 2:
        return "HIGH"
    elif net_edge >= threshold * 1.3:
        return "MEDIUM"
    else: 
        return "LOW"
    
# Generate trading signals from a DataFrame with model probabilities
# For each outcome, computes the edge, adjusts for fees, applies Kelly sizing, and returns a Signal if the net edge exceeds the threshold
def generate_signals(df: pd.DataFrame, bankroll: float=50.0,) -> list[Signal]:
    signals = []

    for _, row in df.iterrows():
        model_prob = row.get("model_prob")
        market_prob = row.get("prob_yes")

        if pd.isna(model_prob) or pd.isna(market_prob):
            continue

        raw = row.get("forecast_horizon_days", 1)
        horizon_days = int(raw) if pd.notna(raw) else 1
        
        threshold = _get_threshold(horizon_days)
        liquidity = row.get("liquidity", 0)

        if liquidity < MIN_LIQUIDITY:
            continue

        # Determine data source
        has_obs = pd.notna(row.get("observed_max_c"))
        data_source = "observation" if has_obs else "forecast"

        # Compute raw edge
        edge = model_prob - market_prob

        # Determine direction and entry price
        if edge >= threshold:
            direction = "BUY_YES"
            entry_prob = market_prob
        elif edge <= -threshold:
            direction = "BUY_NO"
            entry_prob = 1 - market_prob # NO token price
        else:
            continue # Edge too small

        # Fee adjusted net edge
        net_edge = _compute_net_edge(abs(edge), entry_prob)

        if net_edge <= 0:
            continue # Fees eat the entire edge

        # Kelly position sizing
        kelly = _compute_kelly(model_prob, market_prob, direction)
        half_kelly = kelly * KELLY_FRACTION

        # Cap at max fraction of bankroll
        position_fraction = min(half_kelly, MAX_POSITION_FRACTION)
        position_size = round(bankroll * position_fraction, 2)
        position_size = max(position_size, 1) # Minimum $1 trade

        confidence = _get_confidence(net_edge, horizon_days)

        signals.append(Signal(
            series_slug=row["series_slug"],
            event_date=str(row["event_date"])[:10],
            question=row["question"],
            temp_str=row["temp_str"],
            direction=direction,
            market_prob=round(market_prob, 4),
            model_prob=round(model_prob, 4),
            edge=round(edge, 4),
            net_edge=round(net_edge, 4),
            kelly_fraction=round(kelly, 4),
            position_size=position_size,
            data_source=data_source,
            horizon_days=horizon_days,
            token_yes=row["token_yes"],
            token_no=row.get("token_no", ""),
            neg_risk=bool(row.get("neg_risk", False)),
            liquidity=liquidity,
            confidence=confidence,
        ))

    # Sort by net edge descending (best opportunities first)
    signals.sort(key=lambda s: s.net_edge, reverse=True)
    return signals

# Print a formatted summary of all signals
def print_signals(signals: list[Signal]) -> None:
    if not signals:
        print("[SignalEngine] No signals generated.")
        return
    
    print(f"\n[SignalEngine] {len(signals)} signal(s) generated:\n")
    print(f"{'City':<16} {'Date':<12} {'Outcome':<22} {'Dir':<9} {'Market':>7} {'Model':>7} {'NetEdge':>8} {'Size':>6} {'Conf':<8}")
    print("-" * 105)

    for s in signals:
        city = s.series_slug.replace("-daily-weather", "").replace("-", " ").title()
        print(
            f"{city:<16} "
            f"{s.event_date:<12} "
            f"{s.temp_str:<22} "
            f"{s.direction:<9} "
            f"{s.market_prob:>7.3f} "
            f"{s.model_prob:>7.3f} "
            f"{s.net_edge:>+8.4f} "
            f"${s.position_size:>5.2f} "
            f"{s.confidence:<8}"
        )

if __name__ == "__main__":
    print("=== Signal Engine Test ===\n")

    # Fetch all active weather markets
    df = fetch_active_weather_markets()

    if df.empty:
        print("No markets found.")
    else:
        # Enrich with forecasts and model probabilities
        df = fetch_forecasts_for_markets(df)
        df = add_model_probabilities(df)

        # Generate signals with $50 bankroll
        signals = generate_signals(df, bankroll=50.0)
        print_signals(signals)

        if signals:
            print(f"\nTop signal:")
            s = signals[0]
            city = s.series_slug.replace("-daily-weather", "").replace("-", " ").title()
            print(f"  {s.direction} on '{s.temp_str}' in {city} on {s.event_date}")
            print(f"  Market: {s.market_prob:.1%} | Model: {s.model_prob:.1%} | Net edge: {s.net_edge:+.1%}")
            print(f"  Position: ${s.position_size} | Confidence: {s.confidence} | Source: {s.data_source}")