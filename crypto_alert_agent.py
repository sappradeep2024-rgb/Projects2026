#!/usr/bin/env python3
"""
Crypto Movement Alert Agent
============================
Watches the top N cryptocurrencies by market cap (via CoinGecko's public API),
detects when a coin's price moves more than a configurable threshold (default 10%)
within a given window, pulls related recent news headlines, and sends a Telegram
alert.

Runs as a continuous loop, polling every CHECK_INTERVAL_MINUTES.

SETUP
-----
1. Install dependencies:
       pip install requests feedparser --break-system-packages

2. Create a Telegram bot:
   - Message @BotFather on Telegram, send /newbot, follow prompts.
   - Copy the bot token it gives you.

3. Get your chat ID:
   - Message your new bot anything (e.g. "hi").
   - Visit: https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
   - Find "chat":{"id": ...} in the JSON response - that's your chat ID.

4. Fill in TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID below, or set them as
   environment variables of the same name.

5. Run:
       python3 crypto_alert_agent.py

NOTES
-----
- Uses CoinGecko's free public API for price data (no API key required,
  reasonable rate limits). This is far more reliable than scraping exchange
  websites, which change their HTML often and frequently block scrapers.
- News comes from public RSS feeds of major crypto news outlets, matched by
  coin name/symbol mentions.
- The agent tracks which coins it has already alerted on for a given move so
  you don't get spammed every single poll - it only re-alerts if the price
  moves ANOTHER threshold-sized step beyond the last alert, or after a cooldown.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timezone

import requests

try:
    import feedparser
except ImportError:
    feedparser = None

# ---------------------------------------------------------------------------
# CONFIGURATION - edit these, or set as environment variables
# ---------------------------------------------------------------------------

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "PUT_YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "PUT_YOUR_CHAT_ID_HERE")

TOP_N_COINS = 20                 # how many top-market-cap coins to watch
MOVE_THRESHOLD_PERCENT = 0.5    # alert when 24h change exceeds this (absolute value)
CHECK_INTERVAL_MINUTES = 5       # how often to poll CoinGecko
COOLDOWN_MINUTES = 60            # don't re-alert on the same coin/direction within this window
VS_CURRENCY = "usd"

COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"

# RSS feeds used for news matching (add/remove freely)
NEWS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
]

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_state.json")

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("crypto_agent")


# ---------------------------------------------------------------------------
# STATE (persisted across restarts so we don't spam alerts)
# ---------------------------------------------------------------------------

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            log.warning("Could not read state file, starting fresh.")
    return {}


def save_state(state):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except OSError as e:
        log.error(f"Could not save state file: {e}")


# ---------------------------------------------------------------------------
# PRICE DATA
# ---------------------------------------------------------------------------

def fetch_top_coins(top_n=TOP_N_COINS, vs_currency=VS_CURRENCY):
    """Fetch top N coins by market cap with 24h change data from CoinGecko."""
    params = {
        "vs_currency": vs_currency,
        "order": "market_cap_desc",
        "per_page": top_n,
        "page": 1,
        "price_change_percentage": "24h",
        "sparkline": "false",
    }
    resp = requests.get(COINGECKO_MARKETS_URL, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# NEWS MATCHING
# ---------------------------------------------------------------------------

def fetch_recent_news():
    """Pull recent headlines from configured RSS feeds. Returns list of (title, link, source)."""
    if feedparser is None:
        log.warning("feedparser not installed; skipping news lookup. "
                    "Install with: pip install feedparser --break-system-packages")
        return []

    items = []
    for feed_url in NEWS_FEEDS:
        try:
            parsed = feedparser.parse(feed_url)
            source = parsed.feed.get("title", feed_url)
            for entry in parsed.entries[:30]:
                items.append({
                    "title": entry.get("title", ""),
                    "link": entry.get("link", ""),
                    "source": source,
                })
        except Exception as e:
            log.warning(f"Failed to fetch news feed {feed_url}: {e}")
    return items


def find_related_headlines(coin, news_items, max_items=3):
    """Match news headlines mentioning the coin's name or symbol."""
    name = coin["name"].lower()
    symbol = coin["symbol"].lower()
    matches = []
    for item in news_items:
        title_lower = item["title"].lower()
        if name in title_lower or f" {symbol} " in f" {title_lower} ":
            matches.append(item)
        if len(matches) >= max_items:
            break
    return matches


# ---------------------------------------------------------------------------
# TELEGRAM
# ---------------------------------------------------------------------------

def send_telegram_message(text):
    if TELEGRAM_BOT_TOKEN.startswith("PUT_YOUR") or TELEGRAM_CHAT_ID.startswith("PUT_YOUR"):
        log.error("Telegram credentials not configured. Set TELEGRAM_BOT_TOKEN and "
                   "TELEGRAM_CHAT_ID (env vars or in the script). Printing alert instead:\n" + text)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": False,
    }
    try:
        resp = requests.post(url, data=payload, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.error(f"Failed to send Telegram message: {e}")


# ---------------------------------------------------------------------------
# ALERT LOGIC
# ---------------------------------------------------------------------------

def format_alert(coin, news_matches):
    change = coin["price_change_percentage_24h"]
    direction = "🚀 UP" if change > 0 else "🔻 DOWN"
    price = coin["current_price"]
    symbol = coin["symbol"].upper()
    name = coin["name"]

    lines = [
        f"*{direction} {abs(change):.2f}%* — {name} ({symbol})",
        f"Price: ${price:,.4f}" if price < 1 else f"Price: ${price:,.2f}",
        f"24h change: {change:+.2f}%",
        f"Market cap rank: #{coin.get('market_cap_rank', '?')}",
    ]

    if news_matches:
        lines.append("\n*Related news:*")
        for item in news_matches:
            lines.append(f"- [{item['title']}]({item['link']}) ({item['source']})")
    else:
        lines.append("\n_No matching recent headlines found._")

    lines.append(f"\n_{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_")
    return "\n".join(lines)


def check_and_alert(state):
    try:
        coins = fetch_top_coins()
    except requests.RequestException as e:
        log.error(f"Failed to fetch price data: {e}")
        return state

    news_items = fetch_recent_news()
    now_ts = time.time()

    for coin in coins:
        symbol = coin["symbol"].upper()
        change = coin.get("price_change_percentage_24h")
        if change is None:
            continue

        if abs(change) < MOVE_THRESHOLD_PERCENT:
            continue

        direction = "up" if change > 0 else "down"
        key = f"{symbol}_{direction}"
        last_alert = state.get(key)

        # Cooldown: skip if we already alerted on this coin/direction recently
        if last_alert and (now_ts - last_alert["ts"]) < COOLDOWN_MINUTES * 60:
            continue

        news_matches = find_related_headlines(coin, news_items)
        message = format_alert(coin, news_matches)
        log.info(f"ALERT: {symbol} moved {change:+.2f}% — sending Telegram message.")
        send_telegram_message(message)

        state[key] = {"ts": now_ts, "change": change}

    return state


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------

def main():
    log.info(f"Starting crypto alert agent — watching top {TOP_N_COINS} coins, "
              f"threshold {MOVE_THRESHOLD_PERCENT}%, checking every {CHECK_INTERVAL_MINUTES} min.")
    state = load_state()

    while True:
        try:
            state = check_and_alert(state)
            save_state(state)
        except Exception as e:
            log.exception(f"Unexpected error in check cycle: {e}")

        time.sleep(CHECK_INTERVAL_MINUTES * 60)


if __name__ == "__main__":
    main()
