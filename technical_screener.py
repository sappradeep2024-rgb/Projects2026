#!/usr/bin/env python3
"""
Crypto Technical Screener
==========================
Pulls the top N coins by market cap, fetches 1-hour candles from Coinbase
(a legal, US-regulated exchange), computes standard technical indicators
(RSI, MACD, SMA trend, Bollinger Bands, volume ratio), and ranks coins by
a simple composite signal score.

IMPORTANT: This is a screener, not a trading signal generator. It surfaces
indicator data and highlights which coins currently show the most bullish
or bearish technical setups - it does NOT tell you what to buy. Technical
indicators are lagging/heuristic by nature, day trading crypto is high-risk
and highly volatile, and past patterns do not guarantee future moves. Use
this to speed up your own research, not to replace it.

Runs ONCE per execution (prints to console + sends a Telegram digest) and
exits - this is intentional so it works both:
  - Run manually whenever you want a report: python3 technical_screener.py
  - Deployed as a scheduled/cron job (e.g. Railway's Cron Schedule feature)
    that runs it automatically once a day.

SETUP
-----
1. Install dependencies:
       pip install -r requirements.txt --break-system-packages
2. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (same as the alert agent).
3. Run:
       python3 technical_screener.py
"""

import os
import sys
import time
import logging
from datetime import datetime, timezone

import requests

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "PUT_YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "PUT_YOUR_CHAT_ID_HERE")

UNIVERSE_SIZE = 30            # how many top-market-cap coins to consider
TOP_RESULTS_SHOWN = 10        # how many ranked coins to include in the report
CANDLE_GRANULARITY_SECONDS = 3600   # 1-hour candles
CANDLE_LOOKBACK = 100          # ~4 days of 1h candles - enough for MACD(12,26,9) to stabilize

STABLECOIN_SYMBOLS = {"usdt", "usdc", "dai", "tusd", "usde", "fdusd", "busd", "usds", "pyusd"}

REQUEST_DELAY_SECONDS = 0.35   # be polite to Coinbase's public API between calls

COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"
COINBASE_PRODUCTS_URL = "https://api.exchange.coinbase.com/products"
COINBASE_CANDLES_URL = "https://api.exchange.coinbase.com/products/{product_id}/candles"
HEADERS = {"User-Agent": "crypto-technical-screener"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("screener")


# ---------------------------------------------------------------------------
# DATA FETCHING
# ---------------------------------------------------------------------------

def fetch_universe():
    """Top-market-cap coins from CoinGecko, filtered to those tradable on
    Coinbase against USD, excluding stablecoins."""
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": UNIVERSE_SIZE,
        "page": 1,
        "sparkline": "false",
    }
    resp = requests.get(COINGECKO_MARKETS_URL, params=params, timeout=15)
    resp.raise_for_status()
    coins = resp.json()

    resp2 = requests.get(COINBASE_PRODUCTS_URL, headers=HEADERS, timeout=15)
    resp2.raise_for_status()
    valid_products = {
        p["id"] for p in resp2.json()
        if p.get("quote_currency") == "USD" and not p.get("trading_disabled")
    }

    universe = []
    for c in coins:
        symbol = c["symbol"].lower()
        if symbol in STABLECOIN_SYMBOLS:
            continue
        product_id = f"{symbol.upper()}-USD"
        if product_id in valid_products:
            universe.append({
                "symbol": symbol,
                "name": c["name"],
                "product_id": product_id,
                "market_cap_rank": c.get("market_cap_rank"),
            })
    return universe


def fetch_candles(product_id):
    url = COINBASE_CANDLES_URL.format(product_id=product_id)
    params = {"granularity": CANDLE_GRANULARITY_SECONDS}
    resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()  # [time, low, high, open, close, volume], newest first
    data.sort(key=lambda c: c[0])  # chronological order
    if len(data) > CANDLE_LOOKBACK:
        data = data[-CANDLE_LOOKBACK:]
    return data


# ---------------------------------------------------------------------------
# TECHNICAL INDICATORS (pure Python, no extra dependencies)
# ---------------------------------------------------------------------------

def sma(values, period):
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def ema_series(values, period):
    """Returns the EMA series, seeded with an SMA, aligned to values[period-1:]."""
    if len(values) < period:
        return []
    ema_vals = [sum(values[:period]) / period]
    k = 2 / (period + 1)
    for price in values[period:]:
        ema_vals.append(price * k + ema_vals[-1] * (1 - k))
    return ema_vals


def compute_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(closes, fast=12, slow=26, signal=9):
    if len(closes) < slow + signal:
        return None, None, None
    ema_fast = ema_series(closes, fast)
    ema_slow = ema_series(closes, slow)
    offset = (slow - 1) - (fast - 1)
    ema_fast_aligned = ema_fast[offset:]
    macd_line = [f - s for f, s in zip(ema_fast_aligned, ema_slow)]
    if len(macd_line) < signal:
        return None, None, None
    signal_line = ema_series(macd_line, signal)
    macd_current = macd_line[-1]
    signal_current = signal_line[-1]
    return macd_current, signal_current, macd_current - signal_current


def compute_bollinger(closes, period=20, num_std=2):
    if len(closes) < period:
        return None, None, None
    window = closes[-period:]
    mean = sum(window) / period
    variance = sum((x - mean) ** 2 for x in window) / period
    std = variance ** 0.5
    return mean - num_std * std, mean, mean + num_std * std


# ---------------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------------

def analyze_coin(coin, candles):
    if len(candles) < 30:
        return None  # not enough data for reliable indicators

    closes = [c[4] for c in candles]
    volumes = [c[5] for c in candles]
    price = closes[-1]

    rsi_val = compute_rsi(closes)
    macd_line, macd_signal, macd_hist = compute_macd(closes)
    sma20 = sma(closes, 20)
    sma50 = sma(closes, 50) if len(closes) >= 50 else None
    boll_lower, boll_mid, boll_upper = compute_bollinger(closes)
    avg_volume = sum(volumes[-20:]) / min(20, len(volumes))
    volume_ratio = (volumes[-1] / avg_volume) if avg_volume > 0 else None

    score = 0
    signals = []

    if rsi_val is not None:
        if rsi_val > 70:
            signals.append(f"RSI {rsi_val:.0f} — overbought")
            score -= 1
        elif rsi_val < 30:
            signals.append(f"RSI {rsi_val:.0f} — oversold")
        elif 50 <= rsi_val <= 70:
            signals.append(f"RSI {rsi_val:.0f} — bullish momentum")
            score += 1
        else:
            signals.append(f"RSI {rsi_val:.0f} — neutral/weak")

    if macd_line is not None:
        if macd_line > macd_signal:
            signals.append("MACD bullish (above signal line)")
            score += 1
        else:
            signals.append("MACD bearish (below signal line)")
            score -= 1

    if sma20 and sma50:
        if price > sma20 > sma50:
            signals.append("Uptrend (price > SMA20 > SMA50)")
            score += 1
        elif price < sma20 < sma50:
            signals.append("Downtrend (price < SMA20 < SMA50)")
            score -= 1
        else:
            signals.append("Mixed/no clear trend")

    if boll_upper and price >= boll_upper:
        signals.append("Price at/above upper Bollinger Band")
    elif boll_lower and price <= boll_lower:
        signals.append("Price at/below lower Bollinger Band")

    if volume_ratio is not None and volume_ratio > 1.5:
        signals.append(f"Volume {volume_ratio:.1f}x recent average")
        score += 1

    return {
        "symbol": coin["symbol"],
        "name": coin["name"],
        "rank": coin["market_cap_rank"],
        "price": price,
        "rsi": rsi_val,
        "macd_hist": macd_hist,
        "volume_ratio": volume_ratio,
        "score": score,
        "signals": signals,
    }


# ---------------------------------------------------------------------------
# TELEGRAM
# ---------------------------------------------------------------------------

def send_telegram_message(text):
    if TELEGRAM_BOT_TOKEN.startswith("PUT_YOUR") or TELEGRAM_CHAT_ID.startswith("PUT_YOUR"):
        log.error("Telegram credentials not configured. Printing report instead:\n" + text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    try:
        resp = requests.post(url, data=payload, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.error(f"Failed to send Telegram message: {e}")


def fmt_price(p):
    return f"${p:,.4f}" if p < 1 else f"${p:,.2f}"


def build_report(results):
    results_sorted = sorted(results, key=lambda r: (r["score"], r["volume_ratio"] or 0), reverse=True)
    top = results_sorted[:TOP_RESULTS_SHOWN]

    bullish = sum(1 for r in results_sorted if r["score"] >= 2)
    bearish = sum(1 for r in results_sorted if r["score"] <= -1)
    neutral = len(results_sorted) - bullish - bearish
    now_str = datetime.now(timezone.utc).strftime("%A, %B %d %Y — %H:%M UTC")

    header = (
        "━━━━━━━━━━━━━━━━━━━━\n"
        "📈 *DAILY TECHNICAL SCREENER*\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"🗓 {now_str}\n"
        f"🔎 Scanned: *{len(results_sorted)}* coins  |  Showing top *{len(top)}*\n"
        f"🟢 Bullish: {bullish}   ⚪ Neutral: {neutral}   🔴 Bearish: {bearish}\n"
        "⏱ Timeframe: 1h candles\n\n"
        "⚠️ _Informational only — not investment advice. Indicators are lagging "
        "heuristics, not signals to act on. Day trading crypto is high-risk. "
        "Do your own research._\n"
        "━━━━━━━━━━━━━━━━━━━━"
    )

    lines = [header]

    for i, r in enumerate(top, 1):
        score_label = "🟢" if r["score"] >= 2 else ("🟡" if r["score"] == 1 else ("⚪" if r["score"] == 0 else "🔴"))
        lines.append(
            f"{score_label} *#{i}  {r['name']} ({r['symbol'].upper()})*  —  score {r['score']:+d}\n"
            f"    💰 {fmt_price(r['price'])}   📊 Rank #{r['rank']}\n"
            f"    {' • '.join(r['signals'])}"
        )

    lines.append("━━━━━━━━━━━━━━━━━━━━")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    log.info(f"Building technical screener report — universe of top {UNIVERSE_SIZE}, "
              f"{CANDLE_GRANULARITY_SECONDS // 3600}h candles.")

    try:
        universe = fetch_universe()
    except requests.RequestException as e:
        log.error(f"Failed to build universe: {e}")
        return

    log.info(f"Analyzing {len(universe)} coins tradable on Coinbase...")
    results = []
    for coin in universe:
        try:
            candles = fetch_candles(coin["product_id"])
            analysis = analyze_coin(coin, candles)
            if analysis:
                results.append(analysis)
        except requests.RequestException as e:
            log.warning(f"Skipping {coin['symbol'].upper()}: {e}")
        time.sleep(REQUEST_DELAY_SECONDS)

    if not results:
        log.warning("No results to report.")
        return

    report = build_report(results)
    print("\n" + report + "\n")
    send_telegram_message(report)
    log.info("Report sent.")


if __name__ == "__main__":
    main()
