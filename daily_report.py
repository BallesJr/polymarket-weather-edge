"""End-of-day (UTC) Telegram report: two equity curves.

Triggered once a day by .github/workflows/daily_report.yml. Sends an intraday
P&L curve (the trades that resolved today) and the cumulative new-regime curve.
Read-only: it never writes portfolio state, so it runs in its own workflow
without a commit step.
"""
import argparse
from datetime import datetime, timezone

from equity_curve import generate_charts
from notifier import send_photo


def _intraday_caption(r: dict, intra: dict) -> str:
    return (
        f"<b>Intraday P&amp;L — {r['day'].isoformat()} (UTC)</b>\n"
        f"Today: <b>${intra['pnl']:+.2f}</b> over {intra['n']} resolutions\n"
        f"{intra['wins']}W / {intra['losses']}L ({intra['win_rate']:.0%})"
    )


def _regime_caption(r: dict, reg: dict) -> str:
    intra = r["intraday"]
    today = (f"Today: <b>${intra['pnl']:+.2f}</b> over {intra['n']} resolutions"
             if intra else "Today: no resolutions")
    return (
        f"<b>New-regime P&amp;L (since regime start)</b>\n"
        f"{today}\n"
        f"Regime: <b>${reg['pnl']:+.2f}</b> over {reg['n']} trades "
        f"({reg['wins']}W / {reg['losses']}L, {reg['win_rate']:.0%})\n"
        f"Bankroll: ${r['bankroll']:.2f} ({r['roi']:+.1%} ROI)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily equity-curve Telegram report")
    parser.add_argument("--date", help="UTC day YYYY-MM-DD (default: today)")
    parser.add_argument("--no-send", action="store_true",
                        help="Render charts but do not send to Telegram")
    args = parser.parse_args()

    day = (datetime.strptime(args.date, "%Y-%m-%d").date()
           if args.date else datetime.now(timezone.utc).date())

    r = generate_charts(day=day)
    intra, reg = r["intraday"], r["regime"]

    print(f"[DailyReport] {day.isoformat()} UTC")
    if intra:
        print(f"  intraday: ${intra['pnl']:+.2f} over {intra['n']} resolutions -> {intra['path']}")
    else:
        print("  intraday: no resolutions today")
    if reg:
        print(f"  regime:   ${reg['pnl']:+.2f} over {reg['n']} trades -> {reg['path']}")
    else:
        print("  regime:   no closed trades yet")

    if args.no_send:
        return

    sent = 0
    if intra:
        sent += send_photo(intra["path"], _intraday_caption(r, intra))
    if reg:
        sent += send_photo(reg["path"], _regime_caption(r, reg))
    print(f"[DailyReport] Sent {sent} photo(s)")


if __name__ == "__main__":
    main()
