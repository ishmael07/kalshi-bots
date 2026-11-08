"""Edge study: who makes money on Kalshi, at what prices, in which markets.

Rows are position-level records: one per (trade, role) with
  role, price, stake (= price * count), pnl (= stake * ret), event, week,
  category, series.
Returns are dollar-weighted: sum(pnl) / sum(stake).
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable, Sequence


def bucket_of(price: float) -> str:
    b = min(int(price * 10 + 1e-9), 9)
    return f"{b * 10:02d}-{b * 10 + 10}"


def aggregate(rows: Iterable[dict], keys: Sequence[str]) -> dict:
    out: dict = defaultdict(lambda: {"stake": 0.0, "pnl": 0.0, "n": 0, "events": set()})
    for r in rows:
        k = tuple(r[x] for x in keys)
        g = out[k]
        g["stake"] += r["stake"]
        g["pnl"] += r["pnl"]
        g["n"] += 1
        g["events"].add(r.get("event"))
    res = {}
    for k, g in out.items():
        res[k] = {
            "stake": g["stake"], "pnl": g["pnl"], "n": g["n"],
            "n_events": len(g["events"]),
            "ret": g["pnl"] / g["stake"] if g["stake"] else 0.0,
        }
    return res


def cluster_bootstrap_ci(rows: Sequence[dict], n_boot: int = 1000, seed: int = 0,
                         alpha: float = 0.05) -> tuple[float, float]:
    """CI for dollar-weighted return, resampling whole events (outcomes within an
    event are correlated, so trades are not independent)."""
    by_ev: dict = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        e = by_ev[r["event"]]
        e[0] += r["stake"]
        e[1] += r["pnl"]
    evs = list(by_ev.values())
    if not evs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    m = len(evs)
    stats = []
    for _ in range(n_boot):
        s = p = 0.0
        for _ in range(m):
            st, pn = evs[rng.randrange(m)]
            s += st
            p += pn
        stats.append(p / s if s else 0.0)
    stats.sort()
    lo = stats[int(alpha / 2 * n_boot)]
    hi = stats[min(int((1 - alpha / 2) * n_boot), n_boot - 1)]
    return (lo, hi)


# ---------------------------------------------------------------- full study

def ttc_bucket(seconds: float) -> str:
    for lim, name in ((300, "a <5m"), (900, "b 5-15m"), (3600, "c 15-60m"),
                      (6 * 3600, "d 1-6h"), (86400, "e 6-24h")):
        if seconds < lim:
            return name
    return "f >1d"


def fine_bucket(price: float) -> str:
    b = min(int(price * 20 + 1e-9), 19)
    return f"{b * 5:02d}-{b * 5 + 5}"


class Acc:
    """Streaming accumulator: per cell totals plus per-event sums for bootstrap."""

    def __init__(self):
        self.cells = defaultdict(lambda: [0.0, 0.0, 0])
        self.events = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))

    def add(self, key, stake, pnl, event, track_events=False):
        c = self.cells[key]
        c[0] += stake
        c[1] += pnl
        c[2] += 1
        if track_events:
            e = self.events[key][event]
            e[0] += stake
            e[1] += pnl

    def table(self, ci_keys=(), n_boot=400):
        out = {}
        for k, (st, pn, n) in self.cells.items():
            row = {"stake": round(st, 2), "pnl": round(pn, 2), "n": n,
                   "ret": pn / st if st else 0.0}
            if k in self.events:
                evs = self.events[k]
                rows = [dict(stake=v[0], pnl=v[1], event=e) for e, v in evs.items()]
                row["n_events"] = len(rows)
                row["ci"] = cluster_bootstrap_ci(rows, n_boot=n_boot, seed=1)
            out[k] = row
        return out


def run_study(store, since_ts: float = 0, log=print) -> dict:
    from datetime import datetime, timezone
    from .returns import Trade, trade_returns

    acc = Acc()
    n = 0
    for r in store.trade_rows("m.close_ts >= ?", (since_ts,)):
        tr = trade_returns(Trade(r["ticker"], r["event"], r["ts"], r["yes_price"], r["count"],
                                 r["taker_side"], r["result"], r["fee_type"] or "quadratic",
                                 r["fee_mult"] if r["fee_mult"] is not None else 1.0))
        if tr is None:
            continue
        n += 1
        if n % 1_000_000 == 0:
            log(f"  {n:,} trades processed")
        cat, ser, ev = r["category"], r["series"], r["event"]
        ttc = ttc_bucket(r["close_ts"] - r["ts"])
        wk = datetime.fromtimestamp(r["ts"], timezone.utc).strftime("%G-W%V")
        for role, price, ret in (("taker", tr.taker_price, tr.taker_ret),
                                 ("maker", tr.maker_price, tr.maker_ret)):
            stake = price * r["count"]
            pnl = stake * ret
            fav = "fav>=50" if price >= 0.5 else "dog<50"
            acc.add(("role", role), stake, pnl, ev, True)
            acc.add(("role_bucket", role, bucket_of(price)), stake, pnl, ev, True)
            acc.add(("cat_role_side", cat, role, fav), stake, pnl, ev, True)
            acc.add(("series_role_side", ser, role, fav), stake, pnl, ev, True)
            if role == "maker" and price >= 0.5:
                acc.add(("maker_fav_fine", fine_bucket(price)), stake, pnl, ev, True)
                acc.add(("maker_fav_ttc", ttc), stake, pnl, ev, True)
                acc.add(("maker_fav_week", wk), stake, pnl, ev)
                acc.add(("maker_fav_cat_ttc", cat, ttc), stake, pnl, ev, True)
    log(f"  {n:,} trades total; computing confidence intervals...")
    return {"n_trades": n, "table": acc.table()}


def _fmt(k, row):
    ci = row.get("ci")
    cis = f"[{ci[0] * 100:+6.1f}%, {ci[1] * 100:+6.1f}%]" if ci else ""
    ev = row.get("n_events", "")
    return (f"{' | '.join(str(x) for x in k[1:]):38s} ret {row['ret'] * 100:+6.1f}% {cis:20s} "
            f"stake ${row['stake']:>13,.0f}  trades {row['n']:>9,}  events {ev}")


def report(res: dict) -> str:
    t = res["table"]
    lines = [f"Kalshi edge study — {res['n_trades']:,} trades (dollar-weighted returns after fees)"]

    def section(title, kind, sort=None):
        rows = [(k, v) for k, v in t.items() if k[0] == kind]
        rows.sort(key=sort or (lambda kv: kv[0]))
        lines.append("\n== " + title)
        lines.extend(_fmt(k, v) for k, v in rows)

    section("By role", "role")
    section("By role x price band", "role_bucket")
    section("Maker on favorite side, 5c bands", "maker_fav_fine")
    section("Maker favorites by time-to-close", "maker_fav_ttc")
    section("By category x role x side", "cat_role_side")
    section("Maker favorites by category x time-to-close", "maker_fav_cat_ttc")
    section("By series x role x side", "series_role_side")
    wk = sorted((k[1], v) for k, v in t.items() if k[0] == "maker_fav_week")
    pos = sum(1 for _, v in wk if v["ret"] > 0)
    lines.append(f"\n== Maker favorites week by week: {pos}/{len(wk)} weeks positive")
    lines.extend(f"{w}  {v['ret'] * 100:+6.2f}%  stake ${v['stake']:>12,.0f}" for w, v in wk)
    return "\n".join(lines)


def main(argv=None):
    import argparse
    import json
    from datetime import datetime, timezone
    from .store import Store
    ap = argparse.ArgumentParser(prog="kalshi study")
    ap.add_argument("--db", default="data/kalshi.db")
    ap.add_argument("--since", default="2025-05-01")
    ap.add_argument("--out", default="data/study")
    a = ap.parse_args(argv)
    since = datetime.strptime(a.since, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
    res = run_study(Store(a.db), since, log=lambda m: print(m, flush=True))
    txt = report(res)
    print(txt)
    with open(a.out + ".txt", "w") as f:
        f.write(txt)
    with open(a.out + ".json", "w") as f:
        json.dump({"n_trades": res["n_trades"],
                   "rows": [{"key": list(k), **v} for k, v in res["table"].items()]}, f)
