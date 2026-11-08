"""Bankroll simulator: what would happen to a real small account?

Replays candidate fills in time order. Cash is locked until each market settles.
Size = kelly_mult * f*(q, p) * equity, capped by available cash, by a share of the
historical print (we would have been somewhere in the queue), and by a per-event
exposure cap (all strikes of one event are one correlated bet).
"""
from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Candidate:
    ts: float
    close_ts: float
    event: str
    price: float          # price we pay per contract (our side)
    won: bool
    q: float              # our estimated probability that our side wins
    available: float      # contracts in the historical print
    fee_per_contract: float = 0.0
    group: str = ""       # correlated markets share a group; caps apply per group


@dataclass
class SimResult:
    start: float
    final: float = 0.0
    trades: int = 0
    wins: int = 0
    max_drawdown: float = 0.0
    max_exposure: float = 0.0
    curve: list = field(default_factory=list)          # (ts, equity)
    staked_by_event: dict = field(default_factory=dict)
    total_fees: float = 0.0
    log: list = field(default_factory=list)            # (candidate, contracts, cost) if record=True


def kelly_fraction(q: float, p: float) -> float:
    if p >= 1 or q <= p:
        return 0.0
    return (q - p) / (1 - p)


def simulate(cands, start: float = 75.0, kelly_mult: float = 0.25, fill_share: float = 0.25,
             event_cap: float = 0.10, max_frac: float = 0.10, min_stake: float = 0.0,
             curve_every: float = 3600.0, sizing: str = "kelly", fixed_frac: float = 0.01,
             record: bool = False) -> SimResult:
    """sizing="kelly": stake = kelly_mult * f*(q,p), capped at max_frac.
    sizing="fixed": stake = fixed_frac of equity whenever q - p > 0 (the paper's
    fixed-fractional rule; Kelly is only used as a ceiling)."""
    cash = start
    open_q = []                # (close_ts, seq, payout, cost, event)
    locked = 0.0
    ev_open = defaultdict(float)
    res = SimResult(start=start)
    peak = start
    seq = 0
    last_curve = None

    def settle_until(t):
        nonlocal cash, locked
        while open_q and open_q[0][0] <= t:
            _, _, payout, cost, ev = heapq.heappop(open_q)
            cash += payout
            locked -= cost
            ev_open[ev] -= cost

    for c in sorted(cands, key=lambda x: x.ts):
        settle_until(c.ts)
        equity = cash + locked                      # open positions marked at cost
        peak = max(peak, equity)
        if peak > 0:
            res.max_drawdown = max(res.max_drawdown, (peak - equity) / peak)
        if last_curve is None or c.ts - last_curve >= curve_every:
            res.curve.append((c.ts, round(equity, 4)))
            last_curve = c.ts
        f = kelly_mult * kelly_fraction(c.q, c.price)
        if f <= 0:
            continue
        if sizing == "fixed":
            f = min(fixed_frac, f)
        stake = min(f, max_frac) * equity
        g = c.group or c.event
        stake = min(stake, cash, event_cap * equity - ev_open[g])
        unit = c.price + c.fee_per_contract
        contracts = min(stake / unit, c.available * fill_share)
        cost = contracts * unit
        if contracts <= 0 or cost < max(min_stake, 1e-9):
            continue
        cash -= cost
        locked += cost
        ev_open[g] += cost
        res.staked_by_event[g] = res.staked_by_event.get(g, 0.0) + cost
        res.total_fees += contracts * c.fee_per_contract
        res.trades += 1
        res.wins += int(c.won)
        if record:
            res.log.append((c, contracts, cost))
        payout = contracts * (1.0 if c.won else 0.0)
        seq += 1
        heapq.heappush(open_q, (c.close_ts, seq, payout, cost, g))
        res.max_exposure = max(res.max_exposure, locked / max(cash + locked, 1e-9))
    settle_until(float("inf"))
    res.final = cash
    peak = max(peak, cash)
    res.max_drawdown = max(res.max_drawdown, (peak - cash) / peak if peak else 0)
    res.curve.append((float("inf"), round(cash, 4)))
    return res
