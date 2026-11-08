"""Would the bot's model have made money on cities it never saw?

The cell model is trained on 7 cities. This applies it unchanged, with the bot's live rules
(favorite-side maker, <= 24h before close, q - price >= margin, price <= 0.97), to other
temperature series stored in a separate DB, and reports return per position (one position =
event x cell, equal weight, same as the strategy gates), the +1c stress, and weekly consistency.
A city is worth adding only if this out-of-sample check is positive.
python3 -m kalshi citycheck --db data/kalshi_new.db
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from datetime import datetime, timezone

from .live import CellModel
from .run_strategy import load_fills
from .store import Store


def evaluate(db: str, model_path: str = "data/cell_model.json", margin: float = 0.03,
             since: str = "2025-05-01", window_s: float = 86400):
    model = CellModel.load(model_path)
    fills = load_fills(Store(db), datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    pos = defaultdict(lambda: [0.0, 0.0, 0.0, 0])     # (series, event, cell) -> [ret*cnt, ret1c*cnt, cnt, week]
    for f in fills:
        if f.close_ts - f.ts > window_s or f.price > 0.97:
            continue
        q = model.q(f.cell, f.price)
        if q is None or q - f.price < margin:
            continue
        ser = f.event.split("-")[0]
        win = 1.0 if f.won else 0.0
        r = (win - f.price - f.fee_per_contract) / f.price
        p1 = min(f.price + 0.01, 0.999)
        r1 = (win - p1 - f.fee_per_contract) / p1
        P = pos[(ser, f.event, f.cell)]
        P[0] += r * f.count
        P[1] += r1 * f.count
        P[2] += f.count
        P[3] = f.week
    by_series = defaultdict(list)
    for (ser, ev, _cell), (rc, r1c, c, wk) in pos.items():
        by_series[ser].append((rc / c, r1c / c, ev, wk))
    return by_series


def summarize(rows):
    n = len(rows)
    if not n:
        return None
    mean = sum(r[0] for r in rows) / n
    stressed = sum(r[1] for r in rows) / n
    ev = defaultdict(list)
    for r in rows:
        ev[r[2]].append(r[0])
    em = [sum(v) / len(v) for v in ev.values()]
    mu = sum(em) / len(em)
    se = math.sqrt(sum((x - mu) ** 2 for x in em) / max(1, len(em) - 1)) / math.sqrt(len(em)) if len(em) > 1 else float("nan")
    wk = defaultdict(list)
    for r in rows:
        wk[r[3]].append(r[0])
    weeks_up = sum(1 for v in wk.values() if sum(v) > 0)
    return dict(n=n, events=len(ev), mean=mean, stressed=stressed, se=se, weeks_up=weeks_up, weeks=len(wk))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kalshi citycheck")
    ap.add_argument("--db", default="data/kalshi_new.db")
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--since", default="2025-05-01")
    a = ap.parse_args(argv)
    by = evaluate(a.db, margin=a.margin, since=a.since)
    allrows = [r for v in by.values() for r in v]
    print(f"Out-of-sample check of the 7-city model on new series (margin {a.margin:.2f}, <= 24h, maker)")
    print(f"{'series':14s} {'positions':>9s} {'events':>6s} {'return':>8s} {'+/- se':>7s} {'at +1c':>8s} {'weeks up':>9s}")
    for ser in sorted(by, key=lambda s: -summarize(by[s])["mean"]):
        s = summarize(by[ser])
        print(f"{ser:14s} {s['n']:9,d} {s['events']:6d} {s['mean'] * 100:+7.2f}% {s['se'] * 100:6.2f}% "
              f"{s['stressed'] * 100:+7.2f}% {s['weeks_up']:4d}/{s['weeks']:<4d}")
    s = summarize(allrows)
    if s:
        print(f"{'ALL':14s} {s['n']:9,d} {s['events']:6d} {s['mean'] * 100:+7.2f}% {s['se'] * 100:6.2f}% "
              f"{s['stressed'] * 100:+7.2f}% {s['weeks_up']:4d}/{s['weeks']:<4d}")

