import json
import os
import requests
from datetime import datetime, timezone

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Messages are queued here during the cycle and only sent by flush_pending()
# after the portfolio commit is pushed. A run that fails to push discards the
# queue along with its state, so the next cycle's re-resolution of the same
# position doesn't produce a duplicate notification.
PENDING_PATH = "data/pending_notifications.json"


def _queue(message: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    pending = []
    if os.path.exists(PENDING_PATH):
        try:
            with open(PENDING_PATH, "r") as f:
                pending = json.load(f)
        except (json.JSONDecodeError, OSError):
            pending = []
    pending.append(message)
    os.makedirs(os.path.dirname(PENDING_PATH), exist_ok=True)
    with open(PENDING_PATH, "w") as f:
        json.dump(pending, f)


def flush_pending() -> int:
    if not os.path.exists(PENDING_PATH):
        return 0
    try:
        with open(PENDING_PATH, "r") as f:
            pending = json.load(f)
    except (json.JSONDecodeError, OSError):
        pending = []
    sent = sum(1 for msg in pending if _send(msg))
    os.remove(PENDING_PATH)
    return sent


def _send(message: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        resp = requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML",
        }, timeout=10)
        return resp.ok
    except Exception:
        return False


def notify_cycle_summary(portfolio: dict, n_opened: int, n_resolved: int) -> None:
    # At 48 cycles/day a summary per cycle floods the chat. Quiet cycles only
    # notify on the :00 run of even UTC hours (~every 2h, stateless check);
    # cycles that opened or resolved something always notify
    now = datetime.now(timezone.utc)
    if n_opened == 0 and n_resolved == 0 and not (now.hour % 2 == 0 and now.minute < 30):
        return

    bankroll = portfolio["bankroll"]
    total_pnl = portfolio["total_pnl"]
    n_won = portfolio["n_won"]
    n_lost = portfolio["n_lost"]
    n_open = len(portfolio["positions"])
    win_rate = n_won / max(n_won + n_lost, 1)
    roi = (bankroll - portfolio["initial_bankroll"]) / portfolio["initial_bankroll"]

    msg = (
        f"<b>Weather Edge Bot</b>\n"
        f"Resolved: {n_resolved} | Opened: {n_opened}\n\n"
        f"Bankroll: <b>${bankroll:.2f}</b> ({roi:+.1%} ROI)\n"
        f"P&amp;L total: <b>${total_pnl:+.2f}</b>\n"
        f"Win rate: {win_rate:.1%} ({n_won}W / {n_lost}L)\n"
        f"Open positions: {n_open}"
    )
    _queue(msg)


def notify_high_edge_open(direction: str, temp_str: str, city: str,
                           event_date: str, net_edge: float, size: float) -> None:
    msg = (
        f"<b>High-edge trade opened</b>\n"
        f"{direction} | {temp_str} | {city}\n"
        f"Date: {event_date} | Edge: {net_edge:+.3f} | Size: ${size:.2f}"
    )
    _queue(msg)


def notify_resolution(direction: str, temp_str: str, city: str,
                       status: str, pnl: float, bankroll: float) -> None:
    icon = "✅" if status == "WON" else "❌"
    msg = (
        f"{icon} <b>{status}</b> | {direction} {temp_str} | {city}\n"
        f"P&amp;L: ${pnl:+.2f} | Bankroll: ${bankroll:.2f}"
    )
    _queue(msg)
