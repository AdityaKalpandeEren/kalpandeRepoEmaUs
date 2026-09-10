import requests
import config


def send_alert(message: str):
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    resp = requests.post(
        url,
        data={
            "chat_id": config.TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown",
        },
    )
    if resp.status_code != 200:
        print(f"Telegram send failed: {resp.status_code} {resp.text}")


def format_signal_message(signal) -> str:
    risk = signal.entry - signal.stop_loss
    reward = signal.target - signal.entry
    rr = reward / risk if risk else 0
    return (
        f"🟢 *{signal.symbol}* — Long setup\n"
        f"Entry: ${signal.entry}\n"
        f"Stop Loss: ${signal.stop_loss}\n"
        f"Target: ${signal.target}\n"
        f"R:R ≈ 1:{rr:.1f}\n"
        f"Reason: {signal.reason}\n"
        f"Candle time (ET): {signal.candle_time}\n\n"
        f"_Manual trade only — bot does not place orders._"
    )


def format_retest_message(signal) -> str:
    risk = signal.entry - signal.stop_loss
    reward = signal.target - signal.entry
    rr = reward / risk if risk else 0
    aggressor_emoji = "🟢" if signal.aggressor == "BUY" else "🔴" if signal.aggressor == "SELL" else "⚪"
    return (
        f"🔵 *{signal.symbol}* — VWAP Retest (Long)\n"
        f"Entry: ${signal.entry}\n"
        f"Stop Loss: ${signal.stop_loss}\n"
        f"Target: ${signal.target}\n"
        f"VWAP: ${signal.vwap}\n"
        f"R:R ≈ 1:{rr:.1f}\n"
        f"{aggressor_emoji} Last candle aggressor: *{signal.aggressor}* ({signal.buy_share_pct}% buy)\n"
        f"Buy vol: {signal.buy_volume:.2f} | Sell vol: {signal.sell_volume:.2f}\n"
        f"Candle time (ET): {signal.candle_time}\n\n"
        f"_Manual trade only — bot does not place orders._"
    )
