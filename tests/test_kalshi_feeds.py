import pytest

from kalshi.feeds import classify, NewsHub, XSource, SpotWatch


def test_classify_headlines():
    assert classify("Fed unexpectedly cuts rates by 50bp")[0] >= {"Economics", "Financials"}
    cats, sev = classify("Bitcoin plunges 8% as exchange halts withdrawals")
    assert "Crypto" in cats and sev == "high"
    cats, sev = classify("Hurricane warning issued for Miami-Dade")
    assert "Climate and Weather" in cats
    assert classify("Local bakery wins award") == (set(), "none")


class FakeSrc:
    name = "fake"

    def __init__(self, batches):
        self.batches = batches

    def poll(self, now):
        return self.batches.pop(0) if self.batches else []


def test_newshub_dedupes_and_trips_breakers():
    tripped = []
    hub = NewsHub([FakeSrc([[{"id": "1", "ts": 10, "headline": "Fed announces emergency rate cut"},
                             {"id": "1", "ts": 10, "headline": "Fed announces emergency rate cut"}],
                            [{"id": "2", "ts": 20, "headline": "Fed announces emergency rate cut!"}]])],
                  on_trip=lambda cat, why, now: tripped.append(cat))
    hub.poll(now=11)
    hub.poll(now=21)
    assert len(hub.recent) == 1                 # same story twice = one item
    assert "Economics" in tripped and "Financials" in tripped


def test_x_source_respects_daily_budget():
    calls = []

    class S:
        def get(self, url, params=None, headers=None, timeout=None):
            calls.append(params)

            class R:
                status_code = 200

                def json(self_inner):
                    return {"data": [{"id": str(len(calls) * 10 + i), "text": "Bitcoin crash",
                                      "created_at": "2026-11-06T12:00:00Z"} for i in range(10)],
                            "meta": {"newest_id": str(len(calls) * 10 + 9)}}
            return R()

    x = XSource("TOKEN", query="bitcoin", max_reads_per_day=25, session=S(), min_interval_s=0)
    got = x.poll(now=0) + x.poll(now=1) + x.poll(now=2) + x.poll(now=3)
    assert len(got) <= 25
    assert x.reads_today <= 25
    # never asks for more than what's left of the budget, and stops once < 10 remain
    left = 25
    for c in calls:
        assert 10 <= c["max_results"] <= left
        left -= 10                               # fake returns 10 posts per call
    assert x.poll(now=10) == [] and len(calls) == 2


def test_x_source_disabled_without_token():
    assert XSource("", query="x").poll(now=0) == []


def test_spotwatch_trips_on_fast_move():
    tripped = []
    w = SpotWatch("BTC-USD", "Crypto", pct=0.01, window_s=300, on_trip=lambda c, why, now: tripped.append(c))
    w.update(100.0, now=0)
    w.update(100.5, now=100)
    assert not tripped
    w.update(101.6, now=200)
    assert tripped == ["Crypto"]
    assert w.age(now=200) == 0 and w.age(now=260) == 60


def test_old_headlines_do_not_trip_breakers():
    tripped = []
    hub = NewsHub([FakeSrc([[{"id": "1", "ts": 0, "headline": "Bitcoin crashes 10% on exchange hack"}]])],
                  on_trip=lambda c, w, n: tripped.append(c), max_age_s=600)
    hub.poll(now=5000)
    assert tripped == [] and len(hub.recent) == 1


def test_trade_record_high_is_not_weather():
    cats, _ = classify("US trade deficit narrows; imports hit record high")
    assert "Climate and Weather" not in cats
