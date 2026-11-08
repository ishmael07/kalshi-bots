"""Does the crypto pricing model tell good fills from bad ones?

For every historical trade in a crypto strike market we compute the model's
probability for each side, using only data available before the trade. Then we
ask: among REAL fills (makers who actually got filled, takers who actually crossed),
does a bigger model edge (model prob - price paid) mean better realized returns?
If yes, a bot that only accepts fills where edge >= threshold has an edge itself.
"""
from __future__ import annotations

import time
from bisect import bisect_right
from collections import defaultdict

import requests

from .crypto import prob_yes, product_for, realized_sigma
from .returns import Trade, trade_returns
from .study import Acc, cluster_bootstrap_ci

CANDLE_URL = "https://api.exchange.coinbase.com/products/{p}/candles"
VOL_LOOKBACK_MIN = 60
EDGE_BUCKETS = (-1, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 1)


def edge_bucket(e: float) -> str:
    for lo, hi in zip(EDGE_BUCKETS, EDGE_BUCKETS[1:]):
        if e < hi:
            return f"{lo:+.2f}..{hi:+.2f}"
    return "+0.10..+1.00"


class Candles:
    """1-minute Coinbase closes cached in the study DB."""

    def __init__(self, path="data/candles.db", session=None, sleep=time.sleep):
        import sqlite3
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), timeout=120)
        self.db.execute("CREATE TABLE IF NOT EXISTS candles(product TEXT, ts INTEGER, close REAL, "
                        "PRIMARY KEY(product, ts))")
        self.db.execute("CREATE TABLE IF NOT EXISTS candle_spans(product TEXT, a INTEGER, b INTEGER)")
        self.s = session or requests.Session()
        self.sleep = sleep

    def _covered(self, product, a, b) -> bool:
        return self.db.execute("SELECT 1 FROM candle_spans WHERE product=? AND a<=? AND b>=?",
                               (product, a, b)).fetchone() is not None

    def ensure(self, product: str, a: int, b: int):
        a, b = a - a % 60, b - b % 60
        if self._covered(product, a, b):
            return
        t = a
        while t <= b:
            end = min(t + 299 * 60, b)
            for attempt in range(5):
                r = self.s.get(CANDLE_URL.format(p=product), timeout=30, params={
                    "granularity": 60,
                    "start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t)),
                    "end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end))})
                if r.status_code == 200:
                    rows = r.json()
                    self.db.executemany("INSERT OR REPLACE INTO candles VALUES(?,?,?)",
                                        [(product, int(c[0]), float(c[4])) for c in rows])
                    break
                self.sleep(1.5 * (attempt + 1))
            self.sleep(0.12)  # stay well under Coinbase public limits
            t = end + 60
        self.db.execute("INSERT INTO candle_spans VALUES(?,?,?)", (product, a, b))
        self.db.commit()

    def series(self, product: str, a: int, b: int) -> tuple[list, list]:
        rows = self.db.execute("SELECT ts, close FROM candles WHERE product=? AND ts BETWEEN ? AND ? "
                               "ORDER BY ts", (product, a, b)).fetchall()
        return [r[0] for r in rows], [r[1] for r in rows]


def evaluate(store, series_list, log=print, max_window_h: float = 26.0, candles=None) -> dict:
    candles = candles or Candles()
    acc = Acc()
    n_used = n_skipped = 0
    for ser in series_list:
        prod = product_for(ser)
        if not prod:
            continue
        mk = store.db.execute(
            "SELECT m.ticker, m.event, m.open_ts, m.close_ts, m.result, m.strike_type, "
            "m.floor_strike, m.cap_strike, s.fee_type, s.fee_mult FROM markets m "
            "JOIN fetched f ON f.ticker=m.ticker JOIN series s ON s.ticker=m.series "
            "WHERE m.series=? AND m.result IN ('yes','no')", (ser,)).fetchall()
        log(f"[{ser}] {len(mk)} markets")
        for i, m in enumerate(mk, 1):
            start = max(m["open_ts"], m["close_ts"] - max_window_h * 3600)
            a = int(start) - (VOL_LOOKBACK_MIN + 2) * 60
            b = int(m["close_ts"])
            candles.ensure(prod, a, b)
            ts_list, closes = candles.series(prod, a, b)
            if len(ts_list) < VOL_LOOKBACK_MIN:
                n_skipped += 1
                continue
            sig_cache = {}
            trades = store.db.execute(
                "SELECT t.ts, t.yes_price, t.count, t.taker_yes FROM trades t "
                "JOIN markets mm ON mm.id=t.mid WHERE mm.ticker=? AND t.ts>=?",
                (m["ticker"], start)).fetchall()
            for ts, yp, cnt, ty in trades:
                j = bisect_right(ts_list, ts - 60) - 1   # last candle closed before trade
                if j < VOL_LOOKBACK_MIN:
                    n_skipped += 1
                    continue
                if j not in sig_cache:
                    sig_cache[j] = realized_sigma(closes[j - VOL_LOOKBACK_MIN:j + 1])
                sig = sig_cache[j]
                if sig <= 0:
                    continue
                spot = closes[j]
                tau = m["close_ts"] - ts
                q_yes = prob_yes(m["strike_type"], spot, m["floor_strike"], m["cap_strike"], tau, sig)
                if q_yes is None:
                    n_skipped += 1
                    continue
                taker_side = "yes" if ty else "no"
                tr = trade_returns(Trade(m["ticker"], m["event"], ts, yp, cnt, taker_side,
                                         m["result"], m["fee_type"] or "quadratic",
                                         m["fee_mult"] or 1.0))
                if tr is None:
                    continue
                n_used += 1
                for role, side, price, ret in (("taker", tr.taker_side, tr.taker_price, tr.taker_ret),
                                               ("maker", tr.maker_side, tr.maker_price, tr.maker_ret)):
                    q = q_yes if side == "yes" else 1 - q_yes
                    edge = q - price
                    stake = price * cnt
                    pnl = stake * ret
                    eb = edge_bucket(edge)
                    acc.add(("edge", role, eb), stake, pnl, m["event"], True)
                    acc.add(("edge_series", ser, role, eb), stake, pnl, m["event"], True)
                    if edge >= 0.05:
                        fav = "fav>=50" if price >= 0.5 else "dog<50"
                        acc.add(("edge5_side", role, fav), stake, pnl, m["event"], True)
            if i % 25 == 0:
                log(f"[{ser}] {i}/{len(mk)} markets, {n_used:,} trades scored")
    log(f"scored {n_used:,} trades, skipped {n_skipped:,}")
    return {"n": n_used, "table": acc.table()}


def report(res) -> str:
    from .study import _fmt
    t = res["table"]
    lines = [f"Crypto model study — {res['n']:,} real fills scored by model edge (after fees)"]
    for title, kind in (("Edge bucket by role (all crypto)", "edge"),
                        ("Edge >= 5c by role x side", "edge5_side"),
                        ("Edge bucket by series x role", "edge_series")):
        lines.append("\n== " + title)
        rows = sorted((k, v) for k, v in t.items() if k[0] == kind)
        lines.extend(_fmt(k, v) for k, v in rows)
    return "\n".join(lines)


def main(argv=None):
    import argparse
    from .store import Store
    ap = argparse.ArgumentParser(prog="kalshi crypto")
    ap.add_argument("--db", default="data/kalshi.db")
    ap.add_argument("--series", default="KXBTC15M,KXETH15M,KXSOL15M,KXXRP15M,KXBTCD,KXETHD,KXBTC,KXETH")
    ap.add_argument("--out", default="data/crypto")
    a = ap.parse_args(argv)
    res = evaluate(Store(a.db), a.series.split(","), log=lambda m: print(m, flush=True))
    txt = report(res)
    print(txt)
    open(a.out + ".txt", "w").write(txt)
