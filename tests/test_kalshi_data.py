"""API client pagination/retry, store round-trip, and even event sampling."""
import pytest

from kalshi.api import KalshiPublic
from kalshi.store import Store
from kalshi.collect import sample_evenly


class FakeResp:
    def __init__(self, status, payload):
        self.status_code = status
        self._p = payload

    def json(self):
        return self._p


class FakeSession:
    def __init__(self, script):
        self.script = script  # list of (path_contains, params_check, FakeResp)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        for i, (frag, resp) in enumerate(self.script):
            if frag in url:
                self.script.pop(i)
                return resp
        raise AssertionError(f"unexpected call {url} {params}")


def _client(script):
    return KalshiPublic(session=FakeSession(script), rps=1000, sleep=lambda s: None)


def test_paginate_follows_cursor_until_empty():
    c = _client([
        ("/markets", FakeResp(200, {"markets": [{"ticker": "A"}], "cursor": "c1"})),
        ("/markets", FakeResp(200, {"markets": [{"ticker": "B"}], "cursor": ""})),
    ])
    got = list(c.paginate("/markets", "markets", {"series_ticker": "S"}))
    assert [m["ticker"] for m in got] == ["A", "B"]
    assert c.session.calls[1][1]["cursor"] == "c1"


def test_get_retries_on_429_then_succeeds():
    c = _client([
        ("/series", FakeResp(429, {})),
        ("/series", FakeResp(200, {"series": []})),
    ])
    assert c.get("/series/X") == {"series": []}


def test_get_gives_up_after_max_retries():
    c = _client([("/series", FakeResp(500, {})) for _ in range(10)])
    c.max_retries = 3
    with pytest.raises(RuntimeError):
        c.get("/series/X")


def test_market_trades_merges_live_and_historical_and_dedupes():
    t1 = {"trade_id": "1", "ticker": "M", "created_time": "2026-09-01T00:00:00Z",
          "yes_price_dollars": "0.6", "count_fp": "1", "taker_side": "yes"}
    t2 = dict(t1, trade_id="2")
    c = _client([
        ("/historical/trades", FakeResp(200, {"trades": [t1, t2], "cursor": ""})),
        ("/markets/trades", FakeResp(200, {"trades": [t2], "cursor": ""})),
    ])
    got = list(c.market_trades("M"))
    assert sorted(t["trade_id"] for t in got) == ["1", "2"]


def test_store_roundtrip(tmp_path):
    s = Store(tmp_path / "k.db")
    s.upsert_series([{"ticker": "S", "category": "Crypto", "frequency": "fifteen_min",
                      "fee_type": "quadratic", "fee_multiplier": 1, "title": "t"}])
    s.upsert_markets("S", [{"ticker": "M", "event_ticker": "E", "close_time": "2026-09-01T06:00:00Z",
                            "result": "yes", "strike_type": "greater_or_equal", "floor_strike": 10.0,
                            "volume_fp": "5", "expiration_value": "11"}])
    s.insert_trades([{"trade_id": "1", "ticker": "M", "created_time": "2026-09-01T05:50:00Z",
                      "yes_price_dollars": "0.6000", "count_fp": "2.5", "taker_side": "no"}])
    s.mark_fetched("M")
    rows = list(s.trade_rows())
    assert len(rows) == 1
    r = rows[0]
    assert r["yes_price"] == pytest.approx(0.6) and r["count"] == pytest.approx(2.5)
    assert r["result"] == "yes" and r["category"] == "Crypto" and r["series"] == "S"
    assert s.is_fetched("M") and not s.is_fetched("N")
    assert r["taker_side"] == "no" and r["ticker"] == "M"
    # re-fetch of a market replaces its tape
    s.clear_trades("M")
    s.insert_trades([{"trade_id": "1", "ticker": "M", "created_time": "2026-09-01T05:50:00Z",
                      "yes_price_dollars": "0.6000", "count_fp": "2.5", "taker_side": "no"}])
    assert len(list(s.trade_rows())) == 1
    # market upsert keeps its id (trades stay attached)
    s.upsert_markets("S", [{"ticker": "M", "event_ticker": "E", "close_time": "2026-09-01T06:00:00Z",
                            "result": "yes"}])
    assert len(list(s.trade_rows())) == 1


def test_settled_markets_filters_window_client_side_and_stops_early():
    def m(tk, close):
        return {"ticker": tk, "close_time": close, "event_ticker": tk}
    page1 = [m("A", "2026-09-05T00:00:00Z"), m("B", "2026-08-01T00:00:00Z")]
    page2 = [m("C", "2025-01-01T00:00:00Z")]  # entire page older than window -> stop
    c = _client([
        ("/historical/markets", FakeResp(200, {"markets": page1, "cursor": "x"})),
        ("/historical/markets", FakeResp(200, {"markets": page2, "cursor": "y"})),
        ("/markets", FakeResp(200, {"markets": [m("D", "2026-10-01T00:00:00Z")], "cursor": ""})),
    ])
    from kalshi.store import iso_ts
    got = [x["ticker"] for x in c.settled_markets("S", iso_ts("2026-07-01T00:00:00Z"),
                                                   iso_ts("2026-09-30T00:00:00Z"))]
    assert got == ["A", "B"]


def test_sample_evenly_spreads_across_time():
    events = [(f"E{i}", float(i)) for i in range(100)]  # (event, close_ts)
    picked = sample_evenly(events, 10)
    assert len(picked) == 10
    ts = sorted(t for _, t in picked)
    assert ts[0] < 10 and ts[-1] > 89
    assert sample_evenly(events[:5], 10) == events[:5]
