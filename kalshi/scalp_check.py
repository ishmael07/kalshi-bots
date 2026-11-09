"""Would the scalper make money with OUR reaction speed? Replays stored trade tapes.

Every `cycle` seconds the bot sees the last prices (bid = last taker-sell print, ask = last taker-buy print), then rests a
YES bid at the bid and a NO bid at the ask (= sell YES at the ask) until the next cycle. Quotes go stale between cycles, so
this includes being picked off. Two fill models: "touch" takes `share` of prints at our price; "through" fills only when a
print trades strictly past our price (we were certainly filled, and the worst adverse selection).
Inventory is capped per market; whatever is left at close settles at the result. Maker fees are charged on every fill.
python3 -m kalshi scalpcheck --db data/kalshi.db
"""
from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict


def group(s: str) -> str:
    if s.endswith("15M"): return "crypto 15-min"
    if s in ("KXBTC", "KXETH"): return "crypto hourly"
    if s in ("KXINXU", "KXINX"): return "S&P 500"
    if s in ("KXUSDJPYH", "KXEURUSDH"): return "FX hourly"
    if s in ("KXWTI", "KXGOLDD"): return "oil/gold"
    if s.startswith("KXHIGH"): return "weather"
    return "econ/TSA"


def replay(rows, close_ts, result, cycle=60.0, window=86400.0, min_spread=0.02, size=5.0, cap=15.0, share=0.25,
           through=False, stop_before=120.0):
    """rows: (ts, yes_price, count, taker_yes) sorted. Returns dict(pnl, matched, peak_cost, fills) or None."""
    fee = lambda p: 0.0175 * p * (1 - p)
    start = close_ts - window
    bid = ask = None; bid_ts = ask_ts = -1e18      # latest prints seen so far
    q_bid = q_ask = None; next_cycle = None
    b_left = a_left = 0.0
    long_lots, short_lots = [], []                 # [price, n]: YES bought, YES sold
    pnl = matched = fills = peak = 0.0

    def fill(is_buy, px, n):
        nonlocal pnl, matched, fills, peak
        pnl -= n * fee(px); fills += n
        opp, own = (short_lots, long_lots) if is_buy else (long_lots, short_lots)
        left = n
        while left > 1e-9 and opp:
            o = opp[0]; q = min(o[1], left)
            pnl += q * (o[0] - px) if is_buy else q * (px - o[0])
            matched += q; left -= q; o[1] -= q
            if o[1] <= 1e-9: opp.pop(0)
        if left > 1e-9:
            own.append([px, left])
        cost = sum(p * k for p, k in long_lots) + sum((1 - p) * k for p, k in short_lots)
        peak = max(peak, cost)

    for ts, p, c, ty in rows:
        if ts >= start and ts < close_ts - stop_before:
            if next_cycle is None:
                next_cycle = ts
            while ts >= next_cycle:                # requote using only what was known at that moment
                fresh = bid is not None and ask is not None and next_cycle - min(bid_ts, ask_ts) < 900
                net = sum(k for _, k in long_lots) - sum(k for _, k in short_lots)
                if fresh and ask - bid >= min_spread - 1e-9 and 0.05 <= bid and ask <= 0.95:
                    q_bid, q_ask = bid, ask
                    b_left, a_left = (size if net < cap else 0.0), (size if net > -cap else 0.0)
                else:
                    q_bid = q_ask = None
                next_cycle += cycle
            if q_bid is not None:
                if not ty and b_left > 0 and (p < q_bid - 1e-9 if through else p <= q_bid + 1e-9):
                    n = min(c * (1.0 if through else share), b_left); b_left -= n; fill(True, q_bid, n)
                if ty and a_left > 0 and (p > q_ask + 1e-9 if through else p >= q_ask - 1e-9):
                    n = min(c * (1.0 if through else share), a_left); a_left -= n; fill(False, q_ask, n)
        if ty: ask, ask_ts = p, ts
        else: bid, bid_ts = p, ts
    if not fills:
        return None
    win = 1.0 if result == "yes" else 0.0
    inv = sum(k * (win - p) for p, k in long_lots) + sum(k * (p - win) for p, k in short_lots)
    return dict(pnl=pnl + inv, spread=pnl, inv=inv, matched=matched, fills=fills, peak=peak)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kalshi scalpcheck")
    ap.add_argument("--db", default="data/kalshi.db")
    ap.add_argument("--cycles", default="60")
    ap.add_argument("--min-spread", type=float, default=0.02)
    ap.add_argument("--window", type=float, default=86400.0)
    a = ap.parse_args(argv)
    db = sqlite3.connect(a.db)
    mk = {r[0]: r[1:] for r in db.execute("select id, series, close_ts, result from markets where result in ('yes','no')")}
    variants = [(float(c), t) for c in a.cycles.split(",") for t in (False, True)]
    acc = defaultdict(lambda: dict(n=0, pnl=0.0, spread=0.0, inv=0.0, matched=0.0, fills=0.0, peak=0.0))
    days = defaultdict(float)
    rows, cur = [], None

    def flush(mid, rows):
        if mid not in mk or len(rows) < 5:
            return
        ser, close_ts, result = mk[mid]
        g = group(ser)
        for cyc, thr in variants:
            r = replay(rows, close_ts, result, cycle=cyc, through=thr, min_spread=a.min_spread, window=a.window)
            if not r:
                continue
            for key in ((g, cyc, thr), ("ALL", cyc, thr)):
                A = acc[key]; A["n"] += 1
                for k in ("pnl", "spread", "inv", "matched", "fills"):
                    A[k] += r[k]
                A["peak"] = max(A["peak"], r["peak"])
            days[(g, cyc, thr, int(close_ts // 604800))] += r["pnl"]

    for m_, t, p, c, ty in db.execute("select mid, ts, yes_price, count, taker_yes from trades order by mid, ts"):
        if m_ != cur:
            flush(cur, rows); cur, rows = m_, []
        rows.append((t, p, c, ty))
    flush(cur, rows)
    print("$ per market, quoting 5 contracts a side, cap 15 net, maker fee on every fill")
    print(f"{'group':14s} {'cycle':>5s} {'fills':8s} {'mkts':>7s} {'P/L $':>11s} {'spread $':>11s} {'leftover $':>11s} "
          f"{'c/contract':>10s} {'matched%':>9s} {'weeks up':>9s}")
    for key in sorted(acc, key=lambda k: (k[0] != "ALL", k[0], k[1], k[2])):
        A = acc[key]; g, cyc, thr = key
        wk = [v for (gg, c, t, _w), v in days.items() if gg == g and c == cyc and t == thr] if g != "ALL" else []
        up = f"{sum(v > 0 for v in wk)}/{len(wk)}" if wk else ""
        print(f"{g:14s} {cyc:5.0f} {'through' if thr else 'touch':8s} {A['n']:7,d} {A['pnl']:+11,.0f} {A['spread']:+11,.0f} "
              f"{A['inv']:+11,.0f} {A['pnl'] / max(A['fills'], 1) * 100:+10.2f} {A['matched'] / max(A['fills'], 1) * 100:8.0f}% {up:>9s}")
