"""
Pattern Detection
===================
Pure-Python detectors for three families of technical patterns:

1. Candlestick patterns (single/double candle): Doji, Hammer, Shooting Star,
   Bullish/Bearish Engulfing
2. Chart patterns (multi-candle swing structure): Double Top/Bottom,
   Head & Shoulders (regular + inverse), Triangles (ascending/descending/
   symmetrical), Bull/Bear Flags
3. Crossovers & divergence: Golden Cross / Death Cross (SMA50 vs SMA200 on
   daily candles), RSI Divergence (bullish/bearish)

These are heuristic, rule-based approximations of patterns human chartists
look for - not a guarantee the "textbook" pattern is actually present. Chart
pattern detection in particular is inherently fuzzy; different traders would
draw the lines differently. Treat detected patterns as "worth a closer look",
not as confirmed signals.

Each detector returns (pattern_name, direction) tuples where direction is
"bullish", "bearish", or "neutral". Functions return None / [] when nothing
is detected, never a forced result.
"""


# ---------------------------------------------------------------------------
# SHARED HELPERS
# ---------------------------------------------------------------------------

def sma(values, period):
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def find_swing_points(closes, order=3):
    """Local maxima/minima: a point higher/lower than `order` neighbors on
    each side. Returns (highs, lows) as lists of (index, price)."""
    highs, lows = [], []
    for i in range(order, len(closes) - order):
        window = closes[i - order:i + order + 1]
        if closes[i] == max(window):
            highs.append((i, closes[i]))
        if closes[i] == min(window):
            lows.append((i, closes[i]))
    return highs, lows


def linear_regression_slope(points):
    """Simple least-squares slope for a list of (x, y) points."""
    n = len(points)
    if n < 2:
        return 0.0
    sum_x = sum(p[0] for p in points)
    sum_y = sum(p[1] for p in points)
    sum_xy = sum(p[0] * p[1] for p in points)
    sum_xx = sum(p[0] * p[0] for p in points)
    denom = n * sum_xx - sum_x * sum_x
    if denom == 0:
        return 0.0
    return (n * sum_xy - sum_x * sum_y) / denom


def compute_rsi_series(closes, period=14):
    """Rolling RSI values. Returned list is aligned to closes[period:]."""
    if len(closes) < period + 1:
        return []
    gains, losses = [], []
    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))
    rsi_vals = []
    for i in range(period, len(gains) + 1):
        avg_gain = sum(gains[i - period:i]) / period
        avg_loss = sum(losses[i - period:i]) / period
        if avg_loss == 0:
            rsi_vals.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_vals.append(100 - (100 / (1 + rs)))
    return rsi_vals


# ---------------------------------------------------------------------------
# 1. CANDLESTICK PATTERNS
# ---------------------------------------------------------------------------

def detect_candlestick_patterns(candles):
    """candles: list of [time, low, high, open, close, volume]. Looks at the
    most recent 1-2 candles only."""
    patterns = []
    if len(candles) < 2:
        return patterns

    _, low, high, open_, close, _ = candles[-1]
    body = abs(close - open_)
    rng = max(high - low, 1e-9)
    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low

    if body / rng < 0.1:
        patterns.append(("Doji", "neutral"))
    elif lower_wick > 2 * body and upper_wick < body:
        patterns.append(("Hammer", "bullish"))
    elif upper_wick > 2 * body and lower_wick < body:
        patterns.append(("Shooting Star", "bearish"))

    _, _, _, popen, pclose, _ = candles[-2]
    prev_top, prev_bottom = max(popen, pclose), min(popen, pclose)
    cur_top, cur_bottom = max(open_, close), min(open_, close)

    if close > open_ and pclose < popen and cur_top >= prev_top and cur_bottom <= prev_bottom:
        patterns.append(("Bullish Engulfing", "bullish"))
    elif close < open_ and pclose > popen and cur_top >= prev_top and cur_bottom <= prev_bottom:
        patterns.append(("Bearish Engulfing", "bearish"))

    return patterns


# ---------------------------------------------------------------------------
# 2. CHART PATTERNS
# ---------------------------------------------------------------------------

def detect_double_top_bottom(highs, lows, tolerance=0.02):
    patterns = []
    if len(highs) >= 2:
        (_, p1), (_, p2) = highs[-2], highs[-1]
        if abs(p1 - p2) / p1 < tolerance:
            patterns.append(("Double Top", "bearish"))
    if len(lows) >= 2:
        (_, p1), (_, p2) = lows[-2], lows[-1]
        if abs(p1 - p2) / p1 < tolerance:
            patterns.append(("Double Bottom", "bullish"))
    return patterns


def detect_head_and_shoulders(highs, lows, tolerance=0.03):
    patterns = []
    if len(highs) >= 3:
        (_, p1), (_, p2), (_, p3) = highs[-3], highs[-2], highs[-1]
        if p2 > p1 and p2 > p3 and abs(p1 - p3) / p1 < tolerance:
            patterns.append(("Head and Shoulders", "bearish"))
    if len(lows) >= 3:
        (_, p1), (_, p2), (_, p3) = lows[-3], lows[-2], lows[-1]
        if p2 < p1 and p2 < p3 and abs(p1 - p3) / p1 < tolerance:
            patterns.append(("Inverse Head and Shoulders", "bullish"))
    return patterns


def detect_triangle(highs, lows, min_points=3):
    if len(highs) < min_points or len(lows) < min_points:
        return None
    recent_highs = highs[-min_points:]
    recent_lows = lows[-min_points:]
    high_slope = linear_regression_slope(recent_highs)
    low_slope = linear_regression_slope(recent_lows)
    avg_price = sum(p for _, p in recent_highs + recent_lows) / (len(recent_highs) + len(recent_lows))
    flat_thresh = avg_price * 0.0005

    if abs(high_slope) < flat_thresh and low_slope > flat_thresh:
        return ("Ascending Triangle", "bullish")
    if abs(low_slope) < flat_thresh and high_slope < -flat_thresh:
        return ("Descending Triangle", "bearish")
    if high_slope < -flat_thresh and low_slope > flat_thresh:
        return ("Symmetrical Triangle", "neutral")
    return None


def detect_flag(closes, pole_window=10, flag_window=8, pole_thresh=0.05):
    if len(closes) < pole_window + flag_window:
        return None
    pole = closes[-(pole_window + flag_window):-flag_window]
    flag = closes[-flag_window:]
    if not pole or not flag:
        return None
    pole_change = (pole[-1] - pole[0]) / pole[0]
    flag_range = (max(flag) - min(flag)) / flag[0]

    if abs(pole_change) > pole_thresh and flag_range < abs(pole_change) * 0.4:
        return ("Bull Flag", "bullish") if pole_change > 0 else ("Bear Flag", "bearish")
    return None


def detect_chart_patterns(closes, swing_order=3):
    """Runs all chart-pattern detectors and returns a combined list."""
    highs, lows = find_swing_points(closes, order=swing_order)
    patterns = []
    patterns.extend(detect_double_top_bottom(highs, lows))
    patterns.extend(detect_head_and_shoulders(highs, lows))
    triangle = detect_triangle(highs, lows)
    if triangle:
        patterns.append(triangle)
    flag = detect_flag(closes)
    if flag:
        patterns.append(flag)
    return patterns


# ---------------------------------------------------------------------------
# 3. CROSSOVERS & DIVERGENCE
# ---------------------------------------------------------------------------

def detect_ma_cross(daily_closes):
    """Golden Cross / Death Cross using SMA50 vs SMA200 on DAILY closes.
    Needs at least 201 daily candles (~200 days) to detect the crossing
    moment; returns None if there isn't enough history."""
    if len(daily_closes) < 201:
        return None
    sma50_now = sma(daily_closes, 50)
    sma200_now = sma(daily_closes, 200)
    sma50_prev = sma(daily_closes[:-1], 50)
    sma200_prev = sma(daily_closes[:-1], 200)
    if None in (sma50_now, sma200_now, sma50_prev, sma200_prev):
        return None
    if sma50_prev <= sma200_prev and sma50_now > sma200_now:
        return ("Golden Cross", "bullish")
    if sma50_prev >= sma200_prev and sma50_now < sma200_now:
        return ("Death Cross", "bearish")
    return None


def detect_rsi_divergence(closes, lookback=40, swing_order=2):
    rsi_series = compute_rsi_series(closes)
    if len(rsi_series) < lookback:
        return None
    period_offset = len(closes) - len(rsi_series)
    recent_closes = closes[-lookback:]
    recent_rsi = rsi_series[-lookback:]

    highs, lows = find_swing_points(recent_closes, order=swing_order)

    if len(lows) >= 2:
        (i1, p1), (i2, p2) = lows[-2], lows[-1]
        if p2 < p1 and recent_rsi[i2] > recent_rsi[i1]:
            return ("Bullish RSI Divergence", "bullish")
    if len(highs) >= 2:
        (i1, p1), (i2, p2) = highs[-2], highs[-1]
        if p2 > p1 and recent_rsi[i2] < recent_rsi[i1]:
            return ("Bearish RSI Divergence", "bearish")
    return None


# ---------------------------------------------------------------------------
# COMBINED ENTRY POINT
# ---------------------------------------------------------------------------

def detect_all_patterns(candles_1h, daily_closes=None):
    """candles_1h: list of [time, low, high, open, close, volume] (1h).
    daily_closes: optional list of daily close prices for the MA-cross check.
    Returns a list of (pattern_name, direction) tuples."""
    closes = [c[4] for c in candles_1h]
    patterns = []
    patterns.extend(detect_candlestick_patterns(candles_1h))
    patterns.extend(detect_chart_patterns(closes))

    div = detect_rsi_divergence(closes)
    if div:
        patterns.append(div)

    if daily_closes:
        cross = detect_ma_cross(daily_closes)
        if cross:
            patterns.append(cross)

    return patterns
