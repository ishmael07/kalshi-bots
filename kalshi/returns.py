"""Per-trade returns for both sides of a Kalshi print.

Every Kalshi trade is a resting (maker) order being hit by a taker. The taker
bought `taker_side` at that side's price; the maker bought the opposite side at
1 - that price. Holding to settlement, each side's return on stake is
(payout - price - fee_per_contract) / price.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .fees import fee_for


@dataclass(frozen=True)
class Trade:
    ticker: str
    event: str
    ts: float              # unix seconds
    yes_price: float       # dollars
    count: float           # contracts
    taker_side: str        # "yes" | "no"
    result: str            # "yes" | "no" | "" | "void" ...
    fee_type: str = "quadratic"
    fee_mult: float = 1.0


@dataclass(frozen=True)
class TradeReturns:
    taker_side: str
    taker_price: float
    taker_ret: float
    taker_fee: float
    maker_side: str
    maker_price: float
    maker_ret: float
    maker_fee: float


def _other(side: str) -> str:
    return "no" if side == "yes" else "yes"


def trade_returns(t: Trade) -> Optional[TradeReturns]:
    if t.result not in ("yes", "no") or t.count <= 0:
        return None
    tp = t.yes_price if t.taker_side == "yes" else 1.0 - t.yes_price
    mp = 1.0 - tp
    if not (0.0 < tp < 1.0):
        return None
    ms = _other(t.taker_side)
    tf = fee_for("taker", tp, t.count, t.fee_type, t.fee_mult)
    mf = fee_for("maker", mp, t.count, t.fee_type, t.fee_mult)
    t_pay = 1.0 if t.result == t.taker_side else 0.0
    m_pay = 1.0 - t_pay
    return TradeReturns(
        taker_side=t.taker_side, taker_price=tp,
        taker_ret=(t_pay - tp - tf / t.count) / tp, taker_fee=tf,
        maker_side=ms, maker_price=mp,
        maker_ret=(m_pay - mp - mf / t.count) / mp, maker_fee=mf,
    )
