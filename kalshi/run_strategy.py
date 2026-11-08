"""python3 -m kalshi strategy — walk-forward favorite-maker test, gates, $75 sim."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone

from .fees import fee_for
from .gates import gate_report
from .sim import Candidate, simulate
from .strategy import Fill, walk_forward
from .study import fine_bucket, ttc_bucket

EPOCH_WEEK = 7 * 86400


def load_fills(store, since_ts: float, min_price: float = 0.5, cell_mode: str = "cat_band_ttc",
               categories=None):
    q = ("SELECT t.ts, t.yes_price, t.count, t.taker_yes, m.event, m.series, m.result, m.close_ts, "
         "s.category, s.fee_type, s.fee_mult FROM trades t JOIN markets m ON m.id=t.mid "
         "JOIN series s ON s.ticker=m.series JOIN fetched f ON f.ticker=m.ticker "
         "WHERE m.close_ts >= ? AND m.result IN ('yes','no')")
    out = []
    for ts, yp, cnt, ty, ev, ser, res, close_ts, cat, fee_type, fee_mult in store.db.execute(q, (since_ts,)):
        if categories and cat not in categories:
            continue
        maker_side = "no" if ty else "yes"
        price = yp if maker_side == "yes" else 1 - yp
        if price < min_price or price >= 1 or cnt <= 0:
            continue
        fee = fee_for("maker", price, cnt, fee_type or "quadratic", fee_mult or 1.0) / cnt
        ttc = ttc_bucket(close_ts - ts)
        if cell_mode == "cat_band_ttc":
            cell = f"{cat}|{fine_bucket(price)}|{ttc}"
        elif cell_mode == "series_band_ttc":
            cell = f"{ser}|{fine_bucket(price)}|{ttc}"
        else:
            cell = f"{cat}|{fine_bucket(price)}"
        out.append(Fill(ts=ts, close_ts=close_ts, week=int(ts // EPOCH_WEEK), event=ev, cell=cell,
                        price=price, won=(res == maker_side), count=cnt, fee_per_contract=fee,
                        group=corr_group(cat, ev, close_ts)))
    return out


def corr_group(category: str, event: str, close_ts: float) -> str:
    """Markets that move together are one bet: every crypto/financial/commodity market
    closing in the same hour, and every economic market settling on the same day
    (CPI and CPI y/y come from one release)."""
    if category in ("Crypto", "Financials", "Commodities"):
        return f"{category}|{int(close_ts // 3600)}"
    if category == "Economics":
        return f"{category}|{int(close_ts // 86400)}"
    return event


def weekly_returns(taken, price_shift: float = 0.0, until_week: float = float("inf")):
    """Per-week return per $ staked. Weeks that are not over yet are left out."""
    wk = defaultdict(lambda: [0.0, 0.0])
    for f, _q in taken:
        if f.week >= until_week:
            continue
        p = f.price + price_shift + f.fee_per_contract
        stake = p * f.count
        pnl = (1.0 if f.won else 0.0) * f.count - stake
        wk[f.week][0] += stake
        wk[f.week][1] += pnl
    weeks = sorted(wk)
    return weeks, [wk[w][1] / wk[w][0] for w in weeks], sum(v[1] for v in wk.values()) / max(
        sum(v[0] for v in wk.values()), 1e-9)


def weekly_position_returns(taken, price_shift: float = 0.0, until_week: float = float("inf")):
    """Equal weight per position (event x cell), the way a small fixed-fraction bot
    experiences returns. A position's week is the week of its first fill."""
    pos = {}
    for f, _q in taken:
        if f.week >= until_week:
            continue
        k = (f.event, f.cell)
        p = f.price + price_shift + f.fee_per_contract
        st = p * f.count
        v = pos.get(k)
        if v is None:
            v = pos[k] = [f.week, 0.0, 0.0]
        v[1] += st
        v[2] += (1.0 if f.won else 0.0) * f.count - st
    wk = defaultdict(list)
    for w, st, pn in pos.values():
        wk[w].append(pn / st)
    weeks = sorted(wk)
    allr = [r for w in weeks for r in wk[w]]
    return weeks, [sum(wk[w]) / len(wk[w]) for w in weeks], (sum(allr) / len(allr) if allr else 0.0), len(allr)


def sampling_ratio(store, since_ts):
    rows = store.db.execute(
        "SELECT m.series, COUNT(DISTINCT m.event), COUNT(DISTINCT CASE WHEN f.ticker IS NOT NULL "
        "THEN m.event END) FROM markets m LEFT JOIN fetched f ON f.ticker=m.ticker "
        "WHERE m.close_ts >= ? GROUP BY m.series", (since_ts,)).fetchall()
    return {r[0]: (r[2] / r[1] if r[1] else 0.0, r[1], r[2]) for r in rows}


FINAL_CATEGORIES = {"Climate and Weather", "Economics", "Financials", "Commodities"}
# Variants examined before this final test (2026-11-06): 10 (v1 grid) + 12 (v2 grid)
# + 5 window lengths + ~13 category/sizing explorations. Counted so DSR stays honest.
PRIOR_TRIALS = 40


def run(store, since="2025-05-01", start_cash=75.0, log=print, categories=None, now=None):
    """Final strategy: favorite-side maker, no crypto, only within `window` of close.
    Gates use equal-weight position returns; the bankroll sim uses fixed 1% sizing."""
    import time as _t
    since_ts = datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
    until_week = int((now or _t.time()) // EPOCH_WEEK)
    margins = [0.01, 0.02, 0.03]
    windows = [12, 24]
    n_trials = PRIOR_TRIALS + len(margins) * len(windows)
    log("loading fills...")
    fills = load_fills(store, since_ts, cell_mode="cat_band_ttc", categories=categories or FINAL_CATEGORIES)
    log(f"  {len(fills):,} favorite-side maker fills (no crypto)")
    results = {}
    for mg in margins:
        base = walk_forward(fills, warmup_weeks=8, margin=mg, min_prior=30)
        stressed_all = walk_forward(fills, 8, mg, 30, price_shift=0.01)
        for h in windows:
            taken = [(f, q) for f, q in base if f.close_ts - f.ts <= h * 3600]
            st_taken = [(f, q) for f, q in stressed_all if f.close_ts - f.ts <= h * 3600]
            weeks, wr, tot, npos = weekly_position_returns(taken, until_week=until_week)
            _, _, stressed, _ = weekly_position_returns(st_taken, 0.01, until_week)
            results[(f"<= {h}h", mg)] = dict(taken=taken, weeks=weeks, weekly=wr, total=tot,
                                             stressed=stressed, n=npos)
            log(f"  window <= {h}h margin {mg:.2f}: {npos:,} positions, mean return {tot * 100:+.2f}%, "
                f"stressed(+1c) {stressed * 100:+.2f}%, weeks+ {sum(1 for x in wr if x > 0)}/{len(wr)}")
    del fills
    gates = {}
    keys = list(results)
    for key in keys:
        r = results[key]
        neigh = [results[k]["total"] for k in keys if k != key and
                 ((k[0] == key[0] and abs(k[1] - key[1]) <= 0.011) or (k[1] == key[1]))]
        gates[key] = gate_report(r["weekly"], r["n"], n_trials, neigh, r["total"], r["stressed"])
    passing = [k for k in results if gates[k]["all_pass"]]
    best_key = max(passing or keys, key=lambda k: gates[k]["deflated_sharpe"]["value"])
    best = results[best_key]
    cands = [Candidate(ts=f.ts, close_ts=f.close_ts, event=f.event, price=f.price, won=f.won,
                       q=q, available=f.count, fee_per_contract=f.fee_per_contract, group=f.group)
             for f, q in best["taken"] if f.week < until_week]
    sims = {}
    for label, share, stress in (("fixed 1%, 25% of prints", 0.25, 0.0),
                                 ("fixed 1%, 10% of prints", 0.10, 0.0),
                                 ("fixed 1%, 10% of prints, +1c", 0.10, 0.01)):
        cc = [Candidate(c.ts, c.close_ts, c.event, min(c.price + stress, 0.999), c.won, c.q,
                        c.available, c.fee_per_contract, c.group) for c in cands]
        sims[label] = simulate(cc, start=start_cash, fill_share=share, event_cap=0.03, max_frac=0.10,
                               min_stake=0.01, sizing="fixed", fixed_frac=0.01)
    ratio = sampling_ratio(store, since_ts)
    return dict(best_key=best_key, results=results, gates=gates[best_key], all_gates=gates,
                sims=sims, ratio=ratio, n_trials=n_trials)


def report(out) -> str:
    L = []
    mode, mg = out["best_key"]
    b = out["results"][out["best_key"]]
    L.append(f"Favorite-maker walk-forward — best variant: markets={mode}, margin={mg:.2f} "
             f"(DSR corrects for {out['n_trials']} variants tried in total; current partial week excluded)")
    L.append(f"Out-of-sample positions: {b['n']:,} over {len(b['weeks'])} weeks; mean return per position "
             f"{b['total'] * 100:+.2f}%; paying 1c more: {b['stressed'] * 100:+.2f}%")
    L.append("\nGates:")
    for k, v in out["gates"].items():
        if k == "all_pass":
            continue
        val = v["value"]
        L.append(f"  {'PASS' if v['pass'] else 'FAIL'}  {k:18s} {val:.4f}" if isinstance(val, float)
                 else f"  {'PASS' if v['pass'] else 'FAIL'}  {k:18s} {val}")
    L.append(f"  => {'ALL GATES PASS' if out['gates']['all_pass'] else 'NOT PROVEN'}")
    L.append("\nAll variants:")
    for (md, m), r in sorted(out["results"].items()):
        g = out["all_gates"][(md, m)]
        fails = [k for k, v in g.items() if k != "all_pass" and not v["pass"]]
        L.append(f"  {md:10s} margin {m:.2f}: n={r['n']:>8,}  mean {r['total'] * 100:+6.2f}%  "
                 f"stressed {r['stressed'] * 100:+6.2f}%  weeks+ "
                 f"{sum(1 for x in r['weekly'] if x > 0)}/{len(r['weekly'])}  DSR "
                 f"{g['deflated_sharpe']['value']:.2f}  {'PASS' if g['all_pass'] else 'fails: ' + ','.join(fails)}")
    L.append("\n$75 bankroll on the sampled events (fixed 1% per bet, 3% cap per correlated group):")
    for label, r in out["sims"].items():
        L.append(f"  {label:26s} ${r.start:.2f} -> ${r.final:,.2f}  trades {r.trades:,}  "
                 f"win {r.wins / max(r.trades, 1) * 100:.0f}%  maxDD {r.max_drawdown * 100:.1f}%")
    L.append("\nSampling (events fetched / events settled in window) — the live market has this many "
             "times more opportunities:")
    for s, (ratio, tot, got) in sorted(out["ratio"].items()):
        if got:
            L.append(f"  {s:16s} {got:>5}/{tot:<6} -> x{1 / ratio:,.0f}")
    return "\n".join(L)


def main(argv=None):
    import argparse
    from .store import Store
    ap = argparse.ArgumentParser(prog="kalshi strategy")
    ap.add_argument("--db", default="data/kalshi.db")
    ap.add_argument("--since", default="2025-05-01")
    ap.add_argument("--start", type=float, default=75.0)
    ap.add_argument("--categories", default="", help="comma list (default: everything but crypto)")
    ap.add_argument("--out", default="data/strategy")
    a = ap.parse_args(argv)
    cats = set(a.categories.split(",")) if a.categories else None
    out = run(Store(a.db), a.since, a.start, log=lambda m: print(m, flush=True), categories=cats)
    txt = report(out)
    print(txt)
    open(a.out + ".txt", "w").write(txt)
    sims = {k: {"final": v.final, "trades": v.trades, "wins": v.wins, "max_drawdown": v.max_drawdown,
                "curve": [(t, e) for t, e in v.curve if t != float("inf")]} for k, v in out["sims"].items()}
    json.dump({"best": list(out["best_key"]), "n_trials": out["n_trials"], "gates": out["gates"],
               "variants": {f"{k[0]} m{k[1]}": {"mean": r["total"], "stressed": r["stressed"], "n": r["n"],
                                                "weekly": r["weekly"], "weeks": r["weeks"]}
                            for k, r in out["results"].items()},
               "sims": sims}, open(a.out + ".json", "w"), default=str)
