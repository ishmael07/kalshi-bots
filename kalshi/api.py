"""Public (no-auth) Kalshi REST client: markets, trades, series.

Kalshi splits history at a cutoff: older settled markets/trades live under
/historical/*, newer under /markets*. market_trades() reads both and dedupes.
"""
from __future__ import annotations

import time
from typing import Iterator, Optional

import requests

BASE = "https://external-api.kalshi.com/trade-api/v2"
DEMO_BASE = "https://external-api.demo.kalshi.co/trade-api/v2"


class KalshiPublic:
    def __init__(self, base: str = BASE, session=None, rps: float = 8.0,
                 sleep=time.sleep, clock=time.monotonic, timeout: float = 30.0):
        self.base = base.rstrip("/")
        self.session = session or requests.Session()
        if session is None:
            self.session.headers["Accept-Encoding"] = "gzip"
        self.min_gap = 1.0 / rps
        self.sleep = sleep
        self.clock = clock
        self.timeout = timeout
        self.max_retries = 10
        self._last = 0.0
        self.requests = 0

    def _throttle(self):
        wait = self.min_gap - (self.clock() - self._last)
        if wait > 0:
            self.sleep(wait)
        self._last = self.clock()

    def get(self, path: str, params: Optional[dict] = None) -> dict:
        url = self.base + path
        delay = 1.0
        last_err = None
        for _ in range(self.max_retries):
            self._throttle()
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                self.requests += 1
            except requests.RequestException as e:  # network blip
                last_err = e
                self.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return {}
            last_err = f"HTTP {r.status_code}"
            if r.status_code == 429 or r.status_code >= 500:
                self.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            raise RuntimeError(f"{url} {params}: HTTP {r.status_code}")
        raise RuntimeError(f"{url} {params}: gave up after retries ({last_err})")

    def paginate(self, path: str, key: str, params: Optional[dict] = None,
                 limit: int = 1000) -> Iterator[dict]:
        p = dict(params or {})
        p["limit"] = limit
        while True:
            d = self.get(path, p)
            for item in d.get(key, []) or []:
                yield item
            cur = d.get("cursor")
            if not cur:
                return
            p["cursor"] = cur

    # ---- convenience ----
    def series(self, ticker: str) -> dict:
        return self.get(f"/series/{ticker}").get("series", {})

    def settled_markets(self, series: str, min_close_ts: Optional[int] = None,
                        max_close_ts: Optional[int] = None) -> Iterator[dict]:
        """Kalshi ignores close-ts filters on /historical/markets, and both endpoints
        return newest first, so filter here and stop once a whole page is older
        than the window."""
        from .store import iso_ts
        lo = min_close_ts or 0
        hi = max_close_ts or float("inf")
        seen = set()
        for path, extra in (("/historical/markets", {}), ("/markets", {"status": "settled"})):
            p = {"series_ticker": series, "limit": 1000, **extra}
            while True:
                d = self.get(path, p)
                items = d.get("markets", []) or []
                older = 0
                for m in items:
                    ts = iso_ts(m.get("close_time", ""))
                    if ts < lo:
                        older += 1
                        continue
                    if ts > hi or m["ticker"] in seen:
                        continue
                    seen.add(m["ticker"])
                    yield m
                cur = d.get("cursor")
                if not cur or (items and older == len(items)):
                    break
                p["cursor"] = cur

    def market_trades(self, ticker: str) -> Iterator[dict]:
        seen = set()
        for path in ("/historical/trades", "/markets/trades"):
            for t in self.paginate(path, "trades", {"ticker": ticker}):
                if t["trade_id"] not in seen:
                    seen.add(t["trade_id"])
                    yield t

    def orderbook(self, ticker: str, depth: int = 0) -> dict:
        return self.get(f"/markets/{ticker}/orderbook", {"depth": depth} if depth else None)
