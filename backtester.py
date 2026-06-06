import json
import pandas as pd

PORTFOLIO_PATH = "data/paper_portfolio_weather.json"

def load_trades() -> pd.DataFrame:
    with open(PORTFOLIO_PATH) as f:
        data = json.load(f)

    trades = data["closed_trades"]
    df = pd.DataFrame(trades)

    # Only resolved trades (excludes EXPIRED)
    df = df[df["status"].isin(["WON", "LOST"])].copy()

    # Ensure correct types
    df["pnl_usd"] = pd.to_numeric(df["pnl_usd"])
    df["net_edge"] = pd.to_numeric(df["net_edge"])
    df["forecast_horizon_days"] = pd.to_numeric(df["forecast_horizon_days"])
    
    return df

def summary(df: pd.DataFrame, label: str = "") -> None:
    n = len(df)
    won = (df["status"] == "WON").sum()
    pnl = df["pnl_usd"].sum()
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}Trades: {n} | Won: {won} ({won/n:.1%}) | P&L: ${pnl:.2f}")

def by_horizon(df: pd.DataFrame) -> None:
    print("\n--- P&L by forecast horizon ---")
    g = df.groupby("forecast_horizon_days").agg(
        trades=("pnl_usd", "count"),
        won=("status", lambda x: (x == "WON").sum()),
        pnl=("pnl_usd", "sum"),
    )
    g["win_rate"] = g["won"] / g["trades"]
    g["avg_pnl"] = g["pnl"] / g["trades"]
    print(g.to_string())

def by_edge_bucket(df: pd.DataFrame) -> None:
    print("\n--- P&L by net edge bucket ---")
    bins = [0, 0.20, 0.30, 0.40, 0.50, 1.0]
    labels = ["<0.20", "0.20-0.30", "0.30-0.40", "0.40-0.50", ">0.50"]

    df = df.copy()
    df["edge_bucket"] = pd.cut(df["net_edge"], bins=bins, labels=labels)
    g = df.groupby("edge_bucket", observed=True).agg(
        trades=("pnl_usd", "count"),
        won=("status", lambda x: (x == "WON").sum()),
        pnl=("pnl_usd", "sum"),
    )
    g["win_rate"] = g["won"] / g["trades"]
    g["avg_pnl"] = g["pnl"] / g["trades"]
    print(g.to_string())

def by_direction(df: pd.DataFrame) -> None:
    print("\n--- P&L by direction ---")
    g = df.groupby("direction").agg(
        trades=("pnl_usd", "count"),
        won=("status", lambda x: (x == "WON").sum()),
        pnl=("pnl_usd", "sum"),
    )
    g["win_rate"] = g["won"] / g["trades"]
    g["avg_pnl"] = g["pnl"] / g["trades"]
    print(g.to_string())

# --- Simulation ---
# Return a filtered copy of df applying the requested trade filters
def simulate_filters(
        df: pd.DataFrame,
        disable_t1: bool = True,
        disable_t2: bool = True,
        disable_buy_yes: bool = True,
        min_edge: float = 0.20,
) -> pd.DataFrame:
    
    mask = pd.Series(True, index=df.index)

    if disable_t1:
        mask &= df["forecast_horizon_days"] != 1 
        
    if disable_t2:                                
        mask &= df["forecast_horizon_days"] != 2

    if disable_buy_yes:
        mask &= df["direction"] != "BUY_YES"

    mask &= df["net_edge"] >= min_edge

    return df[mask].copy()

# Print a side-by-side summary of baseline vs filtered dataset,
# followed by the full breakdown tables for the filtered subset.
def compare_filters(
    df: pd.DataFrame,
    disable_t1: bool = True,
    disable_t2: bool = True,
    disable_buy_yes: bool = True,
    min_edge: float = 0.20,
) -> None:
    
    filtered = simulate_filters(df, disable_t1=disable_t1, disable_t2=disable_t2, disable_buy_yes=disable_buy_yes, min_edge=min_edge)

    removed = len(df) - len(filtered)
    pnl_base = df["pnl_usd"].sum()
    pnl_filt = filtered["pnl_usd"].sum()
    delta = pnl_filt - pnl_base

    print("=" * 20)
    print("FILTER SIMULATION")
    print(f"disable_t1: {disable_t1}")
    print(f"disable_buy_yes: {disable_buy_yes}")
    print(f"min_edge: {min_edge}")
    print("=" * 20)
 
    summary(df, label="baseline")
    summary(filtered, label="filtered")
 
    print(f"\nTrades removed: {removed} ({removed / len(df):.1%} of baseline)")
    print(f"P&L delta: ${delta:+.2f}")
    print("=" * 55)

    # Detailed breakdowns on the filtered subset
    by_horizon(filtered)
    by_edge_bucket(filtered)
    by_direction(filtered)

def edge_calibration(df: pd.DataFrame) -> None:
    print("\n--- Edge calibration (predicted edge vs actual win rate) ---")
    bins = [0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 1.0]
    labels = ["<0.10","0.10-0.20","0.20-0.30","0.30-0.40","0.40-0.50","0.50-0.60","0.60-0.70",">0.70"]
    df = df.copy()
    df["eb"] = pd.cut(df["net_edge"], bins=bins, labels=labels)
    g = df.groupby("eb", observed=True).agg(
        trades=("pnl_usd","count"),
        won=("status", lambda x: (x=="WON").sum()),
        avg_edge=("net_edge","mean"),
        pnl=("pnl_usd","sum"),
    )
    g["win_rate"] = g["won"] / g["trades"]
    g["edge_vs_wr"] = g["avg_edge"] - g["win_rate"]
    print(g[["trades","avg_edge","win_rate","edge_vs_wr","pnl"]].to_string())

if __name__ == "__main__":
    df = load_trades()

    # Original analysis
    summary(df) 
    by_horizon(df)
    by_edge_bucket(df)
    by_direction(df)
    edge_calibration(df)

    # Simulation: all three filters active
    print("\n" + "=" * 55)
    compare_filters(df, disable_t1=True, disable_t2=True, disable_buy_yes=True, min_edge=0.20)
