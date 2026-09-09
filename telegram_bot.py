#!/usr/bin/env python3
"""
Telegram On-Demand Analysis Bot
==================================
Listens for messages in your Telegram chat. Type a coin ticker (e.g. "BTC",
"ETH", "SOL") and it replies with a real-time technical analysis for that
coin: current price, RSI/MACD/trend, multi-timeframe confluence, detected
patterns, and a composite score - using the exact same analyze_coin() logic
as technical_screener.py, so this matches the daily digest's methodology.

IMPORTANT: This gives you a technical "lean" (bullish/bearish/neutral/mixed)
based on indicator scoring - it does NOT tell you to buy or sell. It's not
a financial advisor and isn't one. See the disclaimer included in every
reply. Use it to speed up your own research, not to replace your judgment.

This is a continuous, always-listening process (uses Telegram long-polling)
- deploy it as its own Railway service (NOT a cron job, since it needs to
stay running to catch messages as they arrive).

SETUP
-----
1. Install dependencies:
       pip install -r requirements.txt --break-system-packages
2. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (same as the other scripts).
3. Run:
       python3 telegram_bot.py
4. In Telegram, message your bot a ticker like "BTC" and wait a few seconds.
"""

import os
import sys
import time
import json
import logging

import requests

from technical_screener import (
    fetch_candles, analyze_coin, fmt_price, COINBASE_PRODUCTS_URL, HEADERS,
)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "PUT_YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "PUT_YOUR_CHAT_ID_HERE")

LONG_POLL_TIMEOUT_SECONDS = 30
VALID_PRODUCTS_REFRESH_SECONDS = 3600  # re-check Coinbase's tradable list hourly

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot_offset_state.json")

TELEGRAM_API_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
COINGECKO_SEARCH_URL = "https://api.coingecko.com/api/v3/search"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("telegram_bot")


# ---------------------------------------------------------------------------
# STATE (persist the Telegram update offset so a restart doesn't replay
# old messages)
# ---------------------------------------------------------------------------

def load_offset():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f).get("offset")
        except (json.JSONDecodeError, OSError):
            pass
    return None


def save_offset(offset):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump({"offset": offset}, f)
    except OSError as e:
        log.warning(f"Could not save offset: {e}")


# ---------------------------------------------------------------------------
# TELEGRAM
# ---------------------------------------------------------------------------

def get_updates(offset):
    params = {"timeout": LONG_POLL_TIMEOUT_SECONDS}
    if offset is not None:
        params["offset"] = offset
    resp = requests.get(
        f"{TELEGRAM_API_BASE}/getUpdates",
        params=params,
        timeout=LONG_POLL_TIMEOUT_SECONDS + 10,
    )
    resp.raise_for_status()
    return resp.json().get("result", [])


def send_message(chat_id, text):
    try:
        resp = requests.post(
            f"{TELEGRAM_API_BASE}/sendMessage",
            data={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        log.error(f"Failed to send message: {e}")


# ---------------------------------------------------------------------------
# COIN LOOKUP
# ---------------------------------------------------------------------------

_valid_products_cache = {"products": None, "ts": 0}


def get_valid_products():
    if _valid_products_cache["products"] is None or \
            time.time() - _valid_products_cache["ts"] > VALID_PRODUCTS_REFRESH_SECONDS:
        try:
            resp = requests.get(COINBASE_PRODUCTS_URL, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            _valid_products_cache["products"] = {
                p["id"] for p in resp.json()
                if p.get("quote_currency") == "USD" and not p.get("trading_disabled")
            }
            _valid_products_cache["ts"] = time.time()
        except requests.RequestException as e:
            log.warning(f"Could not refresh Coinbase product list: {e}")
            if _valid_products_cache["products"] is None:
                _valid_products_cache["products"] = set()
    return _valid_products_cache["products"]


def lookup_coin(symbol):
    """Resolve a ticker like 'BTC' to a coin's name and market cap rank via
    CoinGecko's search endpoint."""
    try:
        resp = requests.get(COINGECKO_SEARCH_URL, params={"query": symbol}, timeout=15)
        resp.raise_for_status()
        coins = resp.json().get("coins", [])
    except requests.RequestException as e:
        log.warning(f"CoinGecko search failed for {symbol}: {e}")
        return None

    matches = [c for c in coins if c.get("symbol", "").lower() == symbol.lower()]
    if not matches:
        return None
    matches.sort(key=lambda c: (c.get("market_cap_rank") is None, c.get("market_cap_rank") or 0))
    best = matches[0]
    return {
        "symbol": symbol.lower(),
        "name": best.get("name", symbol.upper()),
        "market_cap_rank": best.get("market_cap_rank"),
    }


# ---------------------------------------------------------------------------
# ANALYSIS REPLY
# ---------------------------------------------------------------------------

def lean_label(score):
    if score >= 3:
        return "🟢 Bullish lean"
    elif score >= 1:
        return "🟡 Mild bullish lean"
    elif score == 0:
        return "⚪ Neutral / mixed"
    elif score >= -2:
        return "🟠 Mild bearish lean"
    else:
        return "🔴 Bearish lean"


def format_coin_reply(analysis):
    lines = [
        "━━━━━━━━━━━━━━━━━━━━",
        f"🔍 *{analysis['name']} ({analysis['symbol'].upper()})*",
        "━━━━━━━━━━━━━━━━━━━━",
        f"💰 Price: {fmt_price(analysis['price'])}   📊 Rank #{analysis.get('rank', '?')}",
        f"📈 Score: {analysis['score']:+d}  —  {lean_label(analysis['score'])}",
        "",
        "*Signals:*",
    ]
    for s in analysis["signals"]:
        lines.append(f"• {s}")

    lines.append("")
    lines.append(
        "⚠️ _This is a technical lean based on indicator scoring, not a buy/sell "
        "instruction. Not investment advice — do your own research before trading._"
    )
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    return "\n".join(lines)


def handle_ticker(chat_id, symbol):
    send_message(chat_id, f"🔎 Analyzing {symbol}...")

    valid_products = get_valid_products()
    product_id = f"{symbol}-USD"
    if product_id not in valid_products:
        send_message(chat_id, f"'{symbol}' isn't tradable on Coinbase as a USD pair, "
                               f"so I can't pull live data for it.")
        return

    coin_meta = lookup_coin(symbol)
    if coin_meta is None:
        coin_meta = {"symbol": symbol.lower(), "name": symbol.upper(), "market_cap_rank": None}

    try:
        candles_1h = fetch_candles(product_id, 3600)
        time.sleep(0.3)
        candles_4h = fetch_candles(product_id, 14400)
        time.sleep(0.3)
        candles_1d = fetch_candles(product_id, 86400)
        time.sleep(0.3)
        daily_full = fetch_candles(product_id, 86400, limit=300)
        daily_closes_full = [c[4] for c in daily_full]
    except requests.RequestException as e:
        send_message(chat_id, f"Error fetching data for {symbol}: {e}")
        return

    coin_meta["product_id"] = product_id
    analysis = analyze_coin(coin_meta, candles_1h, candles_4h, candles_1d, daily_closes_full)
    if analysis is None:
        send_message(chat_id, f"Not enough historical data yet for {symbol} to analyze.")
        return

    send_message(chat_id, format_coin_reply(analysis))


# ---------------------------------------------------------------------------
# MESSAGE HANDLING
# ---------------------------------------------------------------------------

def looks_like_ticker(text):
    t = text.strip().lstrip("$").upper()
    return t.isalpha() and 2 <= len(t) <= 6


def handle_message(chat_id, text):
    text = text.strip()

    if text.startswith("/start") or text.startswith("/help"):
        send_message(chat_id, "Send me a coin ticker (e.g. *BTC*, *ETH*, *SOL*) and I'll pull "
                               "a real-time technical analysis for it.")
        return

    symbol = text.lstrip("$").strip().upper()
    if not looks_like_ticker(symbol):
        send_message(chat_id, "Send a coin ticker like *BTC*, *ETH*, or *SOL* to get an analysis.")
        return

    handle_ticker(chat_id, symbol)


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------

def main():
    if TELEGRAM_BOT_TOKEN.startswith("PUT_YOUR") or TELEGRAM_CHAT_ID.startswith("PUT_YOUR"):
        log.error("Telegram credentials not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
        return

    log.info("Telegram bot started - listening for ticker messages...")
    offset = load_offset()

    while True:
        try:
            updates = get_updates(offset)
        except requests.RequestException as e:
            log.warning(f"getUpdates failed: {e}")
            time.sleep(5)
            continue

        for update in updates:
            offset = update["update_id"] + 1
            message = update.get("message")
            if not message or "text" not in message:
                continue

            chat_id = str(message["chat"]["id"])
            if chat_id != str(TELEGRAM_CHAT_ID):
                log.info(f"Ignoring message from unauthorized chat_id {chat_id}")
                continue

            try:
                handle_message(chat_id, message["text"])
            except Exception as e:
                log.exception(f"Error handling message: {e}")
                send_message(chat_id, "Something went wrong processing that - try again in a moment.")

        if updates:
            save_offset(offset)


if __name__ == "__main__":
    main()
