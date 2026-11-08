"""Resumable collector: settled markets + full trade tapes for a time-even
sample of events per series."""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from typing import Sequence

from .api import KalshiPublic
from .store import Store


def sample_evenly(events: Sequence[tuple], n: int) -> list:
    """events: [(event_ticker, close_ts)]. Pick n spread evenly through time."""
    ev = sorted(events, key=lambda e: e[1])
    if len(ev) <= n:
        return ev
    step = (len(ev) - 1) / (n - 1) if n > 1 else 0
    return [ev[round(i * step)] for i in range(n)]


def collect_series(client: KalshiPublic, store: Store, series: str, since_ts: int,
                   until_ts: int, n_events: int, log=print, max_trades_per_market: int = 0):
    meta = client.series(series)
    if not meta:
        log(f"[{series}] no such series, skipping")
        return
    store.upsert_series([meta])
    # Stream: some series have hundreds of thousands of markets. Keep only small tuples.
    by_ev = defaultdict(list)
    close = {}
    n_markets = 0
    batch = []
    for m in client.settled_markets(series, since_ts, until_ts):
        n_markets += 1
        batch.append(m)
        if m.get("result") in ("yes", "no"):
            by_ev[m["event_ticker"]].append(m["ticker"])
            close[m["event_ticker"]] = m.get("close_time", "")
        if len(batch) >= 1000:
            store.upsert_markets(series, batch)
            batch = []
    if batch:
        store.upsert_markets(series, batch)
    picked = sample_evenly([(e, close[e]) for e in by_ev], n_events)
    todo = [t for e, _ in picked for t in by_ev[e] if not store.is_fetched(t)]
    log(f"[{series}] {n_markets} settled markets, {len(by_ev)} events; "
        f"sampled {len(picked)} events -> {len(todo)} markets to fetch")
    t0 = time.time()
    for i, tk in enumerate(todo, 1):
        if store.is_fetched(tk):      # another collector got here first
            continue
        store.clear_trades(tk)
        store.db.commit()
        batch, n = [], 0
        for tr in client.market_trades(tk):
            batch.append(tr)
            n += 1
            if len(batch) >= 5000:
                store.insert_trades(batch)
                batch = []
            if max_trades_per_market and n >= max_trades_per_market:
                break
        if batch:
            store.insert_trades(batch)
        store.mark_fetched(tk, n)
        if i % 50 == 0 or i == len(todo):
            log(f"[{series}] {i}/{len(todo)} markets, {time.time() - t0:.0f}s, "
                f"{client.requests} requests")


def main(argv=None):
    import argparse
    from datetime import datetime, timezone
    ap = argparse.ArgumentParser(prog="kalshi collect")
    ap.add_argument("--series", required=True, help="comma-separated series tickers")
    ap.add_argument("--since", required=True, help="YYYY-MM-DD (close time)")
    ap.add_argument("--until", default=None)
    ap.add_argument("--events", type=int, default=300, help="events sampled per series")
    ap.add_argument("--db", default="data/kalshi.db")
    ap.add_argument("--max-trades", type=int, default=0)
    a = ap.parse_args(argv)
    ts = lambda d: int(datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    until = ts(a.until) if a.until else int(time.time())
    client = KalshiPublic()
    store = Store(a.db)
    for s in a.series.split(","):
        try:
            collect_series(client, store, s.strip(), ts(a.since), until, a.events,
                           log=lambda m: print(m, flush=True), max_trades_per_market=a.max_trades)
        except Exception as e:  # keep going on other series
            print(f"[{s}] FAILED: {e}", file=sys.stderr, flush=True)
