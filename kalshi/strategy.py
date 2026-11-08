"""Walk-forward 'favorite maker' strategy on real historical maker fills.

A cell is (category, 5c price band, time-to-close bucket). For each week, the
win rate of every cell is estimated ONLY from earlier weeks. A historical maker
fill in the current week is taken if (estimated win rate - price) >= margin.
The taken fills (with their estimated q) feed the bankroll simulator.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Fill:
    ts: float
    close_ts: float
    week: int
    event: str
    cell: str
    price: float
    won: bool
    count: float
    fee_per_contract: float = 0.0
    group: str = ""            # correlation group (one bet for sizing caps)


class CellStats:
    """Per-cell win stats where every EVENT counts once (one huge event can't dominate)."""

    __slots__ = ("ev",)

    def __init__(self):
        self.ev = {}                                   # event -> [won contracts, contracts]

    def add(self, event, count, won):
        e = self.ev.get(event)
        if e is None:
            e = self.ev[event] = [0.0, 0.0]
        e[0] += count * (1.0 if won else 0.0)
        e[1] += count

    @property
    def n_events(self):
        return len(self.ev)

    def q(self, prior_price=None, shrink: float = 0.0):
        if not self.ev:
            return None
        tot = sum(w / t for w, t in self.ev.values() if t > 0)
        n = len(self.ev)
        if prior_price is None or shrink <= 0:
            return tot / n
        return (tot + shrink * prior_price) / (n + shrink)


def walk_forward(fills, warmup_weeks: int = 8, margin: float = 0.02, min_prior: float = 30,
                 price_shift: float = 0.0, shrink: float = 10.0):
    """Returns [(fill, q_estimate)] for fills taken out of sample.

    min_prior = minimum number of distinct prior EVENTS in the cell.
    price_shift > 0 stresses the test: we assume we paid that much more per
    contract than the historical maker did (worse queue position, stale quote)."""
    by_week = defaultdict(list)
    for f in fills:
        by_week[f.week].append(f)
    weeks = sorted(by_week)
    if not weeks:
        return []
    first = weeks[0]
    prior = defaultdict(CellStats)
    taken = []
    for w in weeks:
        if w - first >= warmup_weeks:
            ok = {}
            for f in by_week[w]:
                if f.cell not in ok:
                    st = prior.get(f.cell)
                    ok[f.cell] = st if st is not None and st.n_events >= min_prior else None
                st = ok[f.cell]
                if st is None:
                    continue
                q = st.q(f.price, shrink)
                p = f.price + price_shift
                if p < 1 and q - p >= margin:
                    taken.append((f, q))
        for f in by_week[w]:                          # learn after trading the week
            prior[f.cell].add(f.event, f.count, f.won)
    return taken
