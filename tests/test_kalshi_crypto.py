"""Crypto strike pricing: lognormal with settlement-averaging, no lookahead."""
import math

import pytest

from kalshi.crypto import (prob_yes, realized_sigma, spot_before, product_for)


def test_at_the_money_is_half():
    assert prob_yes("greater_or_equal", 100.0, 100.0, None, 600, 0.0005) == pytest.approx(0.5, abs=1e-6)


def test_deep_itm_and_otm():
    assert prob_yes("greater_or_equal", 110.0, 100.0, None, 600, 0.0005) > 0.99
    assert prob_yes("greater_or_equal", 90.0, 100.0, None, 600, 0.0005) < 0.01


def test_less_is_complement():
    p = prob_yes("greater", 101.0, 100.0, None, 900, 0.0005)
    q = prob_yes("less", 101.0, 100.0, None, 900, 0.0005)
    assert p + q == pytest.approx(1.0)


def test_between_range():
    p = prob_yes("between", 100.0, 99.0, 101.0, 900, 0.0005)
    assert 0 < p < 1
    lo = prob_yes("greater", 100.0, 99.0, None, 900, 0.0005)
    hi = prob_yes("greater", 100.0, 101.0, None, 900, 0.0005)
    assert p == pytest.approx(lo - hi)


def test_more_time_means_less_certain():
    near = prob_yes("greater", 100.5, 100.0, None, 60, 0.0005)
    far = prob_yes("greater", 100.5, 100.0, None, 3600, 0.0005)
    assert near > far > 0.5


def test_settlement_average_reduces_final_minute_variance():
    # With 30s left inside the 60s averaging window, part of the average is locked in,
    # so the result is MORE certain than a point-in-time settle would be.
    avg = prob_yes("greater", 100.2, 100.0, None, 30, 0.0005, avg_window=60)
    point = prob_yes("greater", 100.2, 100.0, None, 30, 0.0005, avg_window=0)
    assert avg > point


def test_realized_sigma_per_sqrt_second():
    closes = [100 * math.exp(0.001 * (1 if i % 2 else -1)) for i in range(61)]
    s = realized_sigma(closes)
    # 1-min log returns of +-0.002 alternate -> std ~0.002 per minute -> /sqrt(60) per sec
    assert s == pytest.approx(0.002 / math.sqrt(60), rel=0.05)


def test_spot_before_uses_only_closed_candles():
    candles = {0: 10.0, 60: 11.0, 120: 12.0}  # minute start -> close
    # at t=119 only the candle starting at 0 has closed (closes at 60); 60's closes at 120
    assert spot_before(candles, 119) == 10.0
    assert spot_before(candles, 120) == 11.0
    assert spot_before(candles, 5) is None


def test_product_for():
    assert product_for("KXBTC15M") == "BTC-USD"
    assert product_for("KXETHD") == "ETH-USD"
    assert product_for("KXSOL15M") == "SOL-USD"
    assert product_for("KXHIGHNY") is None


def test_edge_bucket_edges():
    from kalshi.crypto_eval import edge_bucket
    assert edge_bucket(-0.5) == "-1.00..-0.10"
    assert edge_bucket(0.0) == "+0.00..+0.02"
    assert edge_bucket(0.07) == "+0.05..+0.10"
    assert edge_bucket(0.5) == "+0.10..+1.00"


def test_evaluate_scores_fills_without_network(tmp_path):
    from kalshi.store import Store
    from kalshi.crypto_eval import evaluate, Candles
    s = Store(tmp_path / "k.db")
    s.upsert_series([{"ticker": "KXBTC15M", "category": "Crypto", "frequency": "fifteen_min",
                      "fee_type": "quadratic", "fee_multiplier": 1, "title": "t"}])
    s.upsert_markets("KXBTC15M", [{"ticker": "M", "event_ticker": "E", "open_time": "2026-09-01T05:45:00Z",
                                   "close_time": "2026-09-01T06:00:00Z", "result": "yes",
                                   "strike_type": "greater_or_equal", "floor_strike": 100.0}])
    s.insert_trades([{"trade_id": "1", "ticker": "M", "created_time": "2026-09-01T05:55:00Z",
                      "yes_price_dollars": "0.60", "count_fp": "10", "taker_side": "no"}])
    s.mark_fetched("M")
    c = Candles(tmp_path / "c.db")
    from kalshi.store import iso_ts
    close = int(iso_ts("2026-09-01T06:00:00Z"))
    a = close - 15 * 60 - 62 * 60
    import math
    rows = [("BTC-USD", t, 101.0 * math.exp(0.0005 * (1 if (t // 60) % 2 else -1)))
            for t in range(a - a % 60, close + 60, 60)]
    c.db.executemany("INSERT INTO candles VALUES(?,?,?)", rows)
    c.db.execute("INSERT INTO candle_spans VALUES(?,?,?)", ("BTC-USD", a - a % 60 - 600, close + 600))
    res = evaluate(s, ["KXBTC15M"], log=lambda m: None, candles=c)
    assert res["n"] == 1
    # spot ~101 vs strike 100 with 5 min left -> model says YES likely; maker held YES at 0.60
    maker_rows = [k for k in res["table"] if k[0] == "edge" and k[1] == "maker"]
    assert maker_rows and maker_rows[0][2].startswith("+0.10")
