"""Live engine pieces: learned cell model, maker quoter, shadow executor, breakers.

The quoter only ever posts post-only bids on the FAVORITE side (price >= 50c) of
a market, and only when the walk-forward-learned win rate for that cell beats
the price by `margin`. It improves the best bid by one tick when that stays
below the ask (so we are first in queue), otherwise it joins the best bid.

ShadowExecutor = forward paper trading on the PUBLIC trade tape: a virtual bid
fills only when a real trade prints through its price after it was posted.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from .sim import kelly_fraction
from .study import fine_bucket, ttc_bucket


class CellModel:
    """cell -> [sum of per-event win fractions, number of events]; q = mean over events."""

    def __init__(self, min_prior: float = 30, shrink: float = 10.0):
        self.table: dict = {}
        self.min_prior = min_prior                     # minimum distinct events
        self.shrink = shrink                           # pseudo-events at the market price

    def q(self, cell: str, price: Optional[float] = None) -> Optional[float]:
        v = self.table.get(cell)
        if not v or v[1] < self.min_prior:
            return None
        if price is None:
            lo = int(cell.split("|")[1].split("-")[0]) / 100
            price = lo + 0.025
        return (v[0] + self.shrink * price) / (v[1] + self.shrink)

    def windows(self) -> set:
        """(category, ttc bucket) pairs that have at least one trusted cell."""
        out = set()
        for cell, v in self.table.items():
            if v[1] >= self.min_prior:
                cat, _band, ttc = cell.split("|")
                out.add((cat, ttc))
        return out

    def save(self, path):
        json.dump({"min_prior": self.min_prior, "table": self.table}, open(path, "w"))

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        m = cls(d.get("min_prior", 30))
        m.table = d["table"]
        return m

    @classmethod
    def from_fills(cls, fills, min_prior=30):
        from .strategy import CellStats
        stats: dict = {}
        for f in fills:
            stats.setdefault(f.cell, CellStats()).add(f.event, f.count, f.won)
        m = cls(min_prior)
        for cell, st in stats.items():
            m.table[cell] = [st.q() * st.n_events, float(st.n_events)]
        return m


@dataclass
class Book:
    ticker: str
    event: str
    category: str
    close_ts: float
    yes_bids: list            # [(price, size)] best first
    no_bids: list
    tick: float = 0.01


@dataclass
class Want:
    ticker: str
    event: str
    side: str
    price: float
    count: float
    q: float
    improved: bool


def cell_key(category: str, price: float, ttc_s: float) -> str:
    return f"{category}|{fine_bucket(price)}|{ttc_bucket(ttc_s)}"


class Quoter:
    def __init__(self, model: CellModel, margin: float = 0.03, kelly_mult: float = 0.25,
                 max_frac: float = 0.10, min_price: float = 0.50, max_price: float = 0.97,
                 sizing: str = "kelly", fixed_frac: float = 0.01):
        self.sizing = sizing
        self.fixed_frac = fixed_frac
        self.model = model
        self.margin = margin
        self.kelly_mult = kelly_mult
        self.max_frac = max_frac
        self.min_price = min_price
        self.max_price = max_price

    def desired(self, b: Book, now: float, equity: float) -> list:
        out = []
        ttc = b.close_ts - now
        if ttc <= 0:
            return out
        for side, bids, opp in (("yes", b.yes_bids, b.no_bids), ("no", b.no_bids, b.yes_bids)):
            if not bids:
                continue
            best = bids[0][0]
            ask = 1.0 - opp[0][0] if opp else 1.0
            px, improved = round(best + b.tick, 4), True
            if px >= ask - 1e-9:
                px, improved = best, False
            if not (self.min_price <= px <= self.max_price):
                continue
            q = self.model.q(cell_key(b.category, px, ttc), px)
            if q is None or q - px < self.margin:
                continue
            f = min(self.kelly_mult * kelly_fraction(q, px), self.max_frac)
            if self.sizing == "fixed":
                f = min(f, self.fixed_frac)
            count = round(f * equity / px, 2)
            if count < 0.01:
                continue
            out.append(Want(b.ticker, b.event, side, px, count, q, improved))
        return out


def apply_view(quoter: "Quoter", b: Book, wants: list, view: dict, equity: float,
               locked_q: float = 0.99, max_price: float = 0.97) -> list:
    """Hard information beats statistics: drop wants on a side that observations have
    already killed, and bid the locked side (maker-only) up to max_price."""
    if not view:
        return wants
    out = [w for w in wants if w.side != view.get("dead")]
    side = view.get("locked")
    if side and not any(w.side == side for w in out):
        bids, opp = (b.yes_bids, b.no_bids) if side == "yes" else (b.no_bids, b.yes_bids)
        ask = 1.0 - opp[0][0] if opp else 1.0
        best = bids[0][0] if bids else 0.0
        px, improved = round(min(best + b.tick, max_price), 4), True
        if px >= ask - 1e-9:
            px, improved = round(min(best, max_price), 4), False
        if 0.0 < px < ask - 1e-9 and locked_q - px > 0:
            f = quoter.fixed_frac if quoter.sizing == "fixed" else min(
                quoter.kelly_mult * kelly_fraction(locked_q, px), quoter.max_frac)
            count = round(f * equity / px, 2)
            if count >= 0.01:
                out.append(Want(b.ticker, b.event, side, px, count, locked_q, improved))
    return out


@dataclass
class VirtualOrder:
    ticker: str
    event: str
    side: str
    price: float
    count: float
    placed_ts: float
    improved: bool
    filled: float = 0.0
    cancelled: bool = False


class ShadowExecutor:
    def __init__(self, fill_share: float = 0.25):
        self.fill_share = fill_share
        self.orders: dict = {}          # ticker -> [VirtualOrder]
        self.positions: dict = {}       # ticker -> [(side, price, contracts, event)]
        self.realized = 0.0

    def place(self, o: VirtualOrder):
        self.orders.setdefault(o.ticker, []).append(o)

    def cancel_all(self, ticker: Optional[str] = None):
        for tk, lst in self.orders.items():
            if ticker is None or tk == ticker:
                for o in lst:
                    o.cancelled = True
        self._gc()

    def resting(self, ticker: str) -> list:
        return [o for o in self.orders.get(ticker, []) if not o.cancelled and o.filled < o.count - 1e-9]

    def _gc(self):
        for tk in list(self.orders):
            self.orders[tk] = [o for o in self.orders[tk] if not o.cancelled and o.filled < o.count - 1e-9]
            if not self.orders[tk]:
                del self.orders[tk]

    def on_trades(self, ticker: str, trades: list) -> list:
        """trades: dicts with ts, yes_price, count, taker_side. Returns new fills."""
        fills = []
        for t in sorted(trades, key=lambda x: x["ts"]):
            for o in self.resting(ticker):
                if t["ts"] <= o.placed_ts:
                    continue
                if o.side == "yes":
                    hit = t["taker_side"] == "no" and t["yes_price"] <= o.price + 1e-9
                else:
                    hit = t["taker_side"] == "yes" and (1 - t["yes_price"]) <= o.price + 1e-9
                if not hit:
                    continue
                share = 1.0 if o.improved else self.fill_share
                n = min(t["count"] * share, o.count - o.filled)
                if n <= 0:
                    continue
                o.filled += n
                self.positions.setdefault(ticker, []).append((o.side, o.price, n, o.event))
                fills.append((o, n, t["ts"]))
        self._gc()
        return fills

    def save(self, path):
        """Atomic: write a temp file then rename, so a full disk can't truncate the book."""
        from pathlib import Path
        path = Path(path)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"positions": self.positions, "realized": self.realized}))
        tmp.replace(path)

    def load(self, path):
        from pathlib import Path
        if Path(path).exists():
            d = json.load(open(path))
            self.positions = {tk: [tuple(x) for x in lst] for tk, lst in d.get("positions", {}).items()}
            self.realized = float(d.get("realized", 0.0))

    def open_cost(self, event: Optional[str] = None) -> float:
        return sum(p * n for lst in self.positions.values() for (s, p, n, ev) in lst
                   if event is None or ev == event)

    def sell(self, ticker: str, side: str, price: float, n: float, fee: float = 0.0) -> float:
        """Sell up to n contracts of `side` into a bid (oldest lots first). Returns realized P&L."""
        keep, left, pnl = [], n, -fee
        for (s, p, k, ev) in self.positions.get(ticker, []):
            if s == side and left > 1e-9:
                q = min(k, left)
                left -= q
                pnl += q * (price - p)
                if k - q > 1e-9:
                    keep.append((s, p, k - q, ev))
            else:
                keep.append((s, p, k, ev))
        if keep:
            self.positions[ticker] = keep
        else:
            self.positions.pop(ticker, None)
        self.realized += pnl
        return pnl

    def settle(self, ticker: str, result: str) -> float:
        pnl = 0.0
        for side, price, n, _ev in self.positions.pop(ticker, []):
            pnl += n * ((1.0 if side == result else 0.0) - price)
        self.cancel_all(ticker)
        self.realized += pnl
        return pnl


class Breakers:
    """Category-level circuit breakers. '*' blocks everything."""

    def __init__(self, cooldown_s: float = 600.0):
        self.cooldown = cooldown_s
        self.until: dict = {}
        self.reasons: dict = {}

    def trip(self, category: str, reason: str, now: float, cooldown_s: Optional[float] = None):
        self.until[category] = max(self.until.get(category, 0), now + (cooldown_s or self.cooldown))
        self.reasons[category] = reason

    def blocked(self, category: str, now: float) -> bool:
        return now < self.until.get("*", 0) or now < self.until.get(category, 0)

    def active(self, now: float) -> dict:
        return {c: self.reasons[c] for c, u in self.until.items() if u > now}
