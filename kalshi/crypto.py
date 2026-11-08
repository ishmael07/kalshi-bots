"""Fair value for Kalshi crypto strike markets.

Settlement = average of 60 one-second CF RTI prints in the final minute. Model the
log price as driftless Brownian motion with per-sqrt-second vol `sigma`.
Variance of the settlement average seen from time tau before close:
  tau >= W:  sigma^2 * (tau - W + W/3)
  tau <  W:  only the remaining tau seconds of the window are random, and they carry
             weight tau/W in the average -> sigma^2 * tau^3 / (3 W^2)
(W = averaging window). With W = 0 this reduces to a point-in-time settle.
Spot comes from Coinbase 1-minute candles, using only candles that had closed before
the trade (no lookahead).
"""
from __future__ import annotations

import math
from bisect import bisect_right
from typing import Optional

SQRT2 = math.sqrt(2.0)

PRODUCTS = {"KXBTC": "BTC-USD", "KXETH": "ETH-USD", "KXSOL": "SOL-USD",
            "KXXRP": "XRP-USD", "KXDOGE": "DOGE-USD"}


def product_for(series: str) -> Optional[str]:
    for pre, prod in PRODUCTS.items():
        if series.startswith(pre):
            return prod
    return None


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT2))


def settle_std(tau: float, sigma: float, avg_window: float = 60.0) -> float:
    tau = max(tau, 0.0)
    W = avg_window
    if W <= 0:
        var = sigma * sigma * tau
    elif tau >= W:
        var = sigma * sigma * (tau - W + W / 3.0)
    else:
        var = sigma * sigma * tau ** 3 / (3.0 * W * W)
    return math.sqrt(max(var, 1e-18))


def _p_above(spot: float, k: float, sd: float) -> float:
    return _phi(math.log(spot / k) / sd)


def prob_yes(strike_type: str, spot: float, floor: Optional[float], cap: Optional[float],
             tau: float, sigma: float, avg_window: float = 60.0) -> Optional[float]:
    sd = settle_std(tau, sigma, avg_window)
    st = strike_type or ""
    if st in ("greater", "greater_or_equal"):
        return _p_above(spot, floor, sd) if floor else None
    if st in ("less", "less_or_equal"):
        k = cap if cap else floor
        return 1.0 - _p_above(spot, k, sd) if k else None
    if st == "between":
        if not (floor and cap):
            return None
        return max(_p_above(spot, floor, sd) - _p_above(spot, cap, sd), 0.0)
    return None


def realized_sigma(closes: list) -> float:
    """Std of 1-minute log returns, scaled to per-sqrt-second."""
    rets = [math.log(b / a) for a, b in zip(closes, closes[1:]) if a > 0 and b > 0]
    if len(rets) < 5:
        return 0.0
    m = sum(rets) / len(rets)
    var = sum((r - m) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) / math.sqrt(60.0)


def spot_before(candles: dict, ts: float) -> Optional[float]:
    """Close of the latest 1-minute candle that closed at or before ts."""
    keys = candles.get("_sorted") if isinstance(candles, dict) and "_sorted" in candles else sorted(
        k for k in candles if k != "_sorted")
    i = bisect_right(keys, ts - 60) - 1
    if i < 0:
        return None
    return candles[keys[i]]
