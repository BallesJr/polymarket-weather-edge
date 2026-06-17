"""Realized-equity curves for the daily Telegram report.

Equity is reconstructed from closed trades as the running sum of pnl_usd ordered
by closed_at; there is no separate equity snapshot to read. Two views are built:

  - intraday: cumulative P&L of trades that resolved on a given UTC day
  - regime:   cumulative P&L since REGIME_START (the clean-data regime); this is
              the meaningful curve, since the old regime's losses drag the global
              bankroll but say nothing about the strategy that runs today.

Rendering uses the Agg backend so it works headless on GitHub Actions.
"""
import json
import os
import tempfile
from datetime import date, datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from signal_engine import REGIME_START  # single source of truth for the cutoff

PORTFOLIO_PATH = "data/paper_portfolio_weather.json"

_GREEN = "#2e7d32"
_RED = "#c62828"


def _parse_ts(s: str):
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def load_closed_trades(path: str = PORTFOLIO_PATH):
    """Return (portfolio_dict, trades) where trades is sorted by close time.

    Trades without a parseable closed_at or pnl are dropped (old/partial rows).
    EXPIRED trades are kept: their loss is real equity, even though they are
    excluded from win-rate stats.
    """
    with open(path) as f:
        data = json.load(f)

    trades = []
    for t in data.get("closed_trades", []):
        ts = _parse_ts(t.get("closed_at", ""))
        pnl = t.get("pnl_usd")
        if ts is None or pnl is None:
            continue
        trades.append({
            "closed_at": ts,
            "opened_at": t.get("opened_at", ""),
            "pnl": float(pnl),
            "status": t.get("status", ""),
        })
    trades.sort(key=lambda x: x["closed_at"])
    return data, trades


def _cumulative(points):
    """Turn a list of trades into (xs, ys) of close-time vs cumulative P&L."""
    xs, ys, cum = [], [], 0.0
    for t in points:
        cum += t["pnl"]
        xs.append(t["closed_at"])
        ys.append(round(cum, 2))
    return xs, ys


def _stats(points, xs, ys):
    wins = sum(1 for t in points if t["status"] == "WON")
    losses = sum(1 for t in points if t["status"] == "LOST")
    decided = wins + losses
    return {
        "pnl": ys[-1] if ys else 0.0,
        "n": len(points),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / decided if decided else 0.0,
    }


def intraday_points(trades, day: date):
    return [t for t in trades if t["closed_at"].astimezone(timezone.utc).date() == day]


def regime_points(trades, regime_start: str = REGIME_START):
    # opened_at is an ISO string; lexical compare matches the rest of the codebase
    return [t for t in trades if t["opened_at"] >= regime_start]


def _render(xs, ys, title: str, time_fmt: str, out_path: str) -> None:
    color = _GREEN if (not ys or ys[-1] >= 0) else _RED
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)
    ax.plot(xs, ys, marker="o", ms=4, lw=1.8, color=color)
    ax.fill_between(xs, ys, 0, alpha=0.12, color=color)
    ax.axhline(0, color="#888888", lw=0.8, ls="--")
    ax.set_title(title)
    ax.set_ylabel("Cumulative P&L ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter(time_fmt, tz=timezone.utc))
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def generate_charts(path: str = PORTFOLIO_PATH, day: date = None, out_dir: str = None) -> dict:
    """Build both charts and return their paths + caption stats.

    `day` defaults to today (UTC). The intraday entry is None when nothing
    resolved that day. Regime is None only if no clean-regime trade has closed.
    """
    if day is None:
        day = datetime.now(timezone.utc).date()
    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="equity_")
    os.makedirs(out_dir, exist_ok=True)

    data, trades = load_closed_trades(path)

    initial = float(data.get("initial_bankroll", 0.0))
    bankroll = float(data.get("bankroll", 0.0))
    roi = (bankroll - initial) / initial if initial else 0.0

    result = {
        "day": day,
        "bankroll": bankroll,
        "initial_bankroll": initial,
        "roi": roi,
        "intraday": None,
        "regime": None,
    }

    intra = intraday_points(trades, day)
    if intra:
        xs, ys = _cumulative(intra)
        out = os.path.join(out_dir, f"intraday_{day.isoformat()}.png")
        _render(xs, ys, f"Intraday P&L — {day.isoformat()} (UTC)", "%H:%M", out)
        result["intraday"] = {**_stats(intra, xs, ys), "path": out}

    reg = regime_points(trades)
    if reg:
        xs, ys = _cumulative(reg)
        out = os.path.join(out_dir, "regime.png")
        _render(xs, ys, f"New-regime P&L — since {REGIME_START} (UTC)", "%m-%d", out)
        result["regime"] = {**_stats(reg, xs, ys), "path": out}

    return result
