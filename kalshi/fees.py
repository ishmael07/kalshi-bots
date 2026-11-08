"""Kalshi event-contract fee schedule.

fee = round_up_to_cent( coeff * multiplier * contracts * P * (1 - P) )
  taker coeff = 0.07 (all quadratic series)
  maker coeff = 0.0175, charged only on series whose fee_type includes maker fees
Prices are in dollars (0..1). Contracts may be fractional.
"""
from __future__ import annotations

import math

TAKER_COEFF = 0.07
MAKER_COEFF = 0.0175
_EPS = 1e-9  # guard against float noise turning 1.75 into 1.7500000001 -> 1.76


def _round_up_cent(x: float) -> float:
    return math.ceil(x * 100 - _EPS) / 100 if x > 0 else 0.0


def taker_fee(price: float, contracts: float, multiplier: float = 1.0) -> float:
    return _round_up_cent(TAKER_COEFF * multiplier * contracts * price * (1 - price))


def maker_fee(price: float, contracts: float, multiplier: float = 1.0) -> float:
    return _round_up_cent(MAKER_COEFF * multiplier * contracts * price * (1 - price))


def fee_for(role: str, price: float, contracts: float, fee_type: str, multiplier: float) -> float:
    if role == "taker":
        return taker_fee(price, contracts, multiplier)
    if role == "maker":
        if fee_type and "maker_fees" in fee_type:
            return maker_fee(price, contracts, multiplier)
        return 0.0
    raise ValueError(role)
