#!/usr/bin/env python3
"""
Crypto Movement Alert Agent v2
================================
Watches the top N cryptocurrencies by market cap and sends Telegram alerts on:
  - 1h / 24h / 7d price moves past configurable thresholds
  - Volume spikes (current volume vs recent trailing average)
  - Fast moves within a short rolling window (default 5 min), detected via a
    live Coinbase WebSocket feed, so you don't have to wait for the next poll

Data sources:
  - CoinMarketCap (if CMC_API_KEY is set) - richer data, one call gives
    1h/24h/7d % change and volume together.
  - CoinGecko (automatic fallback, no key needed) - used if no CMC key is set.
  - Coinbase public WebSocket - real-time price ticks for the fast-move layer.
    No key needed, this is Coinbase's public market data stream. (Binance's
    equivalent feed blocks connections from US-based servers, which is where
    most free hosts like Railway run, so Coinbase is used instead.)

SETUP
-----
1. Install dependencies:
       pip install -r requirements.txt --break-system-packages

2. (Recommended, optional) Get a free CoinMarketCap API key:
   - Sign up at https://coinmarketcap.com/api/ (free "Basic" plan)
   - Copy your API key from the dashboard
   - Set it as CMC_API_KEY below or as an environment variable
   - Without this, the agent automatically uses CoinGecko instead - still
     works, just slightly less rich data (no built-in volume-change field,
     which this script computes itself anyway, so functionally similar).

3. Telegram bot token + chat ID: see README.md (same as v1).

4. Run:
       python3 crypto_alert_agent.py
"""

import os
import sys
import time
import json
import logging
import threading
from collections import deque
from datetime import datetime, timezone

import requests

try:
    import feedparser
except ImportError:
    feedparser = None

try:
    import websocket  # websocket-client package
except ImportError:
    websocket = None

# ---------------------------------------------------------------------------
# CONFIGURATION - edit these, or set as environment variables
# ---------------------------------------------------------------------------

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "PUT_YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "PUT_YOUR_CHAT_ID_HERE")
CMC_API_KEY = os.environ.get("CMC_API_KEY", "")  # optional - leave blank to use CoinGecko

TOP_N_COINS = 20
VS_CURRENCY = "usd"

CHECK_INTERVAL_MINUTES = 5       # how often to poll for 1h/24h/7d + volume data
COOLDOWN_MINUTES = 60            # min gap before re-alerting same coin+timeframe+direction

# Percent-move thresholds per timeframe (polled data)
THRESHOLDS = {
    "1h": 5.0,
    "24h": 10.0,
    "7d": 20.0,
}

# Volume spike: alert if current volume is this many % above the trailing average
VOLUME_SPIKE_THRESHOLD_PERCENT = 100.0
VOLUME_HISTORY_SAMPLES = 12      # ~1 hour of history at a 5-min poll interval

# Fast-move layer (WebSocket, near-real-time)
FAST_MOVE_WINDOW_SECONDS = 300       # 5-minute rolling window
FAST_MOVE_THRESHOLD_PERCENT = 3.0    # alert if price moves this much within the window
FAST_MOVE_COOLDOWN_SECONDS = 900     # 15 min between fast-move alerts per coin/direction

# Coins to skip (stablecoins don't meaningfully "move")
STABLECOIN_SYMBOLS = {"usdt", "usdc", "dai", "tusd", "usde", "fdusd", "busd", "usds", "pyusd"}

NEWS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
]

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_state.json")

COINBASE_PRODUCTS_URL = "https://api.exchange.coinbase.com/products"
COINBASE_WS_URL = "wss://ws-feed.exchange.coinbase.com"

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("crypto_agent")

state_lock = threading.Lock()

# ---------------------------------------------------------------------------
# STATE (persisted so we don't spam alerts across restarts)
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
        with state_lock:
            with open(STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
    except OSError as e:
        log.error(f"Could not save state file: {e}")


def cooldown_ok(state, key, cooldown_seconds):
    with state_lock:
        last = state.get(key)
        now_ts = time.time()
        if last and (now_ts - last) < cooldown_seconds:
            return False
        state[key] = now_ts
        return True


# ---------------------------------------------------------------------------
# MARKET DATA (CoinMarketCap primary, CoinGecko fallback)
# ---------------------------------------------------------------------------

def fetch_market_data_cmc():
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    headers = {"X-CMC_PRO_API_KEY": CMC_API_KEY, "Accepts": "application/json"}
    params = {"start": 1, "limit": TOP_N_COINS, "convert": VS_CURRENCY.upper()}
    resp = requests.get(url, headers=headers, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()["data"]

    coins = []
    for c in data:
        quote = c["quote"][VS_CURRENCY.upper()]
        coins.append({
            "symbol": c["symbol"].lower(),
            "name": c["name"],
            "price": quote["price"],
            "rank": c.get("cmc_rank"),
            "pct_1h": quote.get("percent_change_1h"),
            "pct_24h": quote.get("percent_change_24h"),
            "pct_7d": quote.get("percent_change_7d"),
            "volume_24h": quote.get("volume_24h"),
        })
    return coins


def fetch_market_data_coingecko():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": VS_CURRENCY,
        "order": "market_cap_desc",
        "per_page": TOP_N_COINS,
        "page": 1,
        "price_change_percentage": "1h,24h,7d",
        "sparkline": "false",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    coins = []
    for c in data:
        coins.append({
            "symbol": c["symbol"].lower(),
            "name": c["name"],
            "price": c["current_price"],
            "rank": c.get("market_cap_rank"),
            "pct_1h": c.get("price_change_percentage_1h_in_currency"),
            "pct_24h": c.get("price_change_percentage_24h_in_currency"),
            "pct_7d": c.get("price_change_percentage_7d_in_currency"),
            "volume_24h": c.get("total_volume"),
        })
    return coins


def fetch_market_data():
    if CMC_API_KEY:
        try:
            return fetch_market_data_cmc()
        except requests.RequestException as e:
            log.warning(f"CoinMarketCap fetch failed ({e}), falling back to CoinGecko.")
    try:
        return fetch_market_data_coingecko()
    except requests.RequestException as e:
        log.error(f"CoinGecko fetch also failed: {e}")
        return []


# ---------------------------------------------------------------------------
# NEWS MATCHING
# ---------------------------------------------------------------------------

def fetch_recent_news():
    if feedparser is None:
        log.warning("feedparser not installed; skipping news lookup.")
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


def find_related_headlines(name, symbol, news_items, max_items=3):
    name_l = name.lower()
    symbol_l = symbol.lower()
    matches = []
    for item in news_items:
        title_lower = item["title"].lower()
        if name_l in title_lower or f" {symbol_l} " in f" {title_lower} ":
            matches.append(item)
        if len(matches) >= max_items:
            break
    return matches


# ---------------------------------------------------------------------------
# TELEGRAM
# ---------------------------------------------------------------------------

def send_telegram_message(text):
    if TELEGRAM_BOT_TOKEN.startswith("PUT_YOUR") or TELEGRAM_CHAT_ID.startswith("PUT_YOUR"):
        log.error("Telegram credentials not configured. Printing alert instead:\n" + text)
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


def fmt_price(p):
    return f"${p:,.4f}" if p < 1 else f"${p:,.2f}"


def send_alert(kind, name, symbol, detail_lines, news_items):
    matches = find_related_headlines(name, symbol, news_items)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "━━━━━━━━━━━━━━━━━━━━",
        f"🚨 *CRYPTO ALERT* — {now_str}",
        "━━━━━━━━━━━━━━━━━━━━",
        f"*{kind}*",
        f"*{name} ({symbol.upper()})*",
        "",
    ] + detail_lines

    if matches:
        lines.append("\n*Related news:*")
        for m in matches:
            lines.append(f"- [{m['title']}]({m['link']}) ({m['source']})")

    lines.append("━━━━━━━━━━━━━━━━━━━━")
    message = "\n".join(lines)
    log.info(f"ALERT [{kind}] {symbol.upper()}")
    send_telegram_message(message)


# ---------------------------------------------------------------------------
# PERIODIC CHECK: 1h / 24h / 7d moves + volume spikes
# ---------------------------------------------------------------------------

volume_history = {}  # symbol -> deque of recent volume readings

def check_periodic(state, news_items):
    coins = fetch_market_data()
    if not coins:
        return coins

    for coin in coins:
        symbol = coin["symbol"]
        if symbol in STABLECOIN_SYMBOLS:
            continue
        name = coin["name"]

        # --- price move checks across timeframes ---
        for timeframe, threshold in THRESHOLDS.items():
            change = coin.get(f"pct_{timeframe}")
            if change is None or abs(change) < threshold:
                continue
            direction = "up" if change > 0 else "down"
            key = f"{symbol}_{timeframe}_{direction}"
            if not cooldown_ok(state, key, COOLDOWN_MINUTES * 60):
                continue
            arrow = "🚀 UP" if change > 0 else "🔻 DOWN"
            send_alert(
                f"{arrow} {abs(change):.2f}% ({timeframe})",
                name, symbol,
                [f"Price: {fmt_price(coin['price'])}",
                 f"{timeframe} change: {change:+.2f}%",
                 f"Market cap rank: #{coin.get('rank', '?')}"],
                news_items,
            )

        # --- volume spike check ---
        vol = coin.get("volume_24h")
        if vol is not None:
            hist = volume_history.setdefault(symbol, deque(maxlen=VOLUME_HISTORY_SAMPLES))
            if len(hist) >= 3:  # need a little history before judging a "spike"
                avg = sum(hist) / len(hist)
                if avg > 0:
                    vol_change = (vol - avg) / avg * 100
                    if vol_change >= VOLUME_SPIKE_THRESHOLD_PERCENT:
                        key = f"{symbol}_volume"
                        if cooldown_ok(state, key, COOLDOWN_MINUTES * 60):
                            send_alert(
                                f"📊 VOLUME SPIKE +{vol_change:.0f}%",
                                name, symbol,
                                [f"Price: {fmt_price(coin['price'])}",
                                 f"24h volume: ${vol:,.0f} (avg: ${avg:,.0f})",
                                 f"Market cap rank: #{coin.get('rank', '?')}"],
                                news_items,
                            )
            hist.append(vol)

    return coins


# ---------------------------------------------------------------------------
# FAST-MOVE LAYER: live Coinbase WebSocket
# ---------------------------------------------------------------------------
# Uses Coinbase's public "ticker" feed instead of Binance - Binance.com blocks
# connections from US-based servers (incl. Railway's default region), while
# Coinbase, being a US exchange, does not.

class CoinbaseWatcher:
    """Maintains a live WebSocket connection to Coinbase for the current set
    of watched coins, and raises fast-move alerts within a short rolling
    window, independent of the slower periodic poll."""

    def __init__(self, state, news_items_getter):
        self.state = state
        self.news_items_getter = news_items_getter
        self.price_history = {}   # product_id -> deque[(ts, price)]
        self.symbol_meta = {}     # product_id (e.g. "BTC-USD") -> {"symbol":..., "name":...}
        self._valid_pairs = None
        self._ws = None
        self._ws_thread = None
        self._lock = threading.Lock()
        self._stop = False

    def _fetch_valid_pairs(self):
        try:
            resp = requests.get(
                COINBASE_PRODUCTS_URL,
                headers={"User-Agent": "crypto-alert-agent"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                p["id"] for p in data
                if p.get("quote_currency") == "USD" and not p.get("trading_disabled")
            }
        except requests.RequestException as e:
            log.warning(f"Could not fetch Coinbase products: {e}")
            return set()

    def update_watchlist(self, coins):
        """Call whenever the top-N coin list changes. Rebuilds the socket
        subscription if the set of tradable pairs changed."""
        if self._valid_pairs is None:
            self._valid_pairs = self._fetch_valid_pairs()

        new_meta = {}
        for c in coins:
            symbol = c["symbol"]
            if symbol in STABLECOIN_SYMBOLS:
                continue
            product_id = f"{symbol.upper()}-USD"
            if product_id in self._valid_pairs:
                new_meta[product_id] = {"symbol": symbol, "name": c["name"]}

        with self._lock:
            changed = set(new_meta.keys()) != set(self.symbol_meta.keys())
            self.symbol_meta = new_meta

        if changed:
            log.info(f"Fast-move watchlist updated: {sorted(m['symbol'].upper() for m in new_meta.values())}")
            self._restart_socket()

    def _restart_socket(self):
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
        if not self.symbol_meta:
            return
        if self._ws_thread is None or not self._ws_thread.is_alive():
            self._ws_thread = threading.Thread(target=self._run_forever, daemon=True)
            self._ws_thread.start()

    def _on_open(self, ws):
        with self._lock:
            product_ids = list(self.symbol_meta.keys())
        if not product_ids:
            return
        sub_msg = {
            "type": "subscribe",
            "product_ids": product_ids,
            "channels": ["ticker"],
        }
        try:
            ws.send(json.dumps(sub_msg))
        except Exception as e:
            log.warning(f"Coinbase WS subscribe failed: {e}")

    def _run_forever(self):
        if websocket is None:
            log.warning("websocket-client not installed; skipping fast-move layer. "
                        "Install with: pip install websocket-client --break-system-packages")
            return
        while not self._stop:
            with self._lock:
                has_pairs = bool(self.symbol_meta)
            if not has_pairs:
                time.sleep(5)
                continue
            try:
                self._ws = websocket.WebSocketApp(
                    COINBASE_WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=lambda ws, err: log.warning(f"Coinbase WS error: {err}"),
                )
                self._ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                log.warning(f"Coinbase WS connection failed: {e}")
            if not self._stop:
                time.sleep(5)  # brief backoff before reconnecting

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            if data.get("type") != "ticker":
                return
            pair = data.get("product_id", "")
            price = float(data.get("price", 0))
        except (ValueError, TypeError, json.JSONDecodeError):
            return
        if not pair or price <= 0:
            return

        with self._lock:
            meta = self.symbol_meta.get(pair)
        if meta is None:
            return

        now = time.time()
        hist = self.price_history.setdefault(pair, deque(maxlen=1000))
        hist.append((now, price))

        # prune old entries beyond 2x the window to bound memory
        cutoff = now - (FAST_MOVE_WINDOW_SECONDS * 2)
        while hist and hist[0][0] < cutoff:
            hist.popleft()

        # find the oldest sample within the window
        window_start = now - FAST_MOVE_WINDOW_SECONDS
        old_price = None
        for ts, p in hist:
            if ts >= window_start:
                old_price = p
                break
        if old_price is None or old_price <= 0:
            return

        change = (price - old_price) / old_price * 100
        if abs(change) < FAST_MOVE_THRESHOLD_PERCENT:
            return

        direction = "up" if change > 0 else "down"
        key = f"{meta['symbol']}_fast_{direction}"
        if not cooldown_ok(self.state, key, FAST_MOVE_COOLDOWN_SECONDS):
            return

        arrow = "⚡🚀 FAST MOVE UP" if change > 0 else "⚡🔻 FAST MOVE DOWN"
        news_items = self.news_items_getter()
        send_alert(
            f"{arrow} {abs(change):.2f}% (last {FAST_MOVE_WINDOW_SECONDS // 60} min)",
            meta["name"], meta["symbol"],
            [f"Price: {fmt_price(price)}",
             f"Move: {change:+.2f}% in ~{FAST_MOVE_WINDOW_SECONDS // 60} min"],
            news_items,
        )

    def stop(self):
        self._stop = True
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------

def main():
    source = "CoinMarketCap" if CMC_API_KEY else "CoinGecko"
    log.info(f"Starting crypto alert agent v2 — top {TOP_N_COINS} coins via {source}, "
              f"polling every {CHECK_INTERVAL_MINUTES} min, fast-move window "
              f"{FAST_MOVE_WINDOW_SECONDS}s @ {FAST_MOVE_THRESHOLD_PERCENT}%.")

    state = load_state()
    cached_news = {"items": [], "ts": 0}

    def get_news():
        # refresh news at most once every 10 minutes, shared across layers
        if time.time() - cached_news["ts"] > 600:
            cached_news["items"] = fetch_recent_news()
            cached_news["ts"] = time.time()
        return cached_news["items"]

    watcher = CoinbaseWatcher(state, get_news)

    while True:
        try:
            news_items = get_news()
            coins = check_periodic(state, news_items)
            if coins:
                watcher.update_watchlist(coins)
            save_state(state)
        except Exception as e:
            log.exception(f"Unexpected error in check cycle: {e}")

        time.sleep(CHECK_INTERVAL_MINUTES * 60)


if __name__ == "__main__":
    main()
