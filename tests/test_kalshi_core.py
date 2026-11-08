"""Core Kalshi math: fees, per-trade returns for maker and taker, aggregation."""
import math

import pytest

from kalshi.fees import taker_fee, maker_fee, fee_for
from kalshi.returns import trade_returns, Trade
from kalshi.study import bucket_of, aggregate, cluster_bootstrap_ci


# ---------- fees ----------

def test_taker_fee_peaks_at_50c_and_rounds_up_to_cent():
    # 0.07 * 1 * 0.5 * 0.5 = 0.0175 -> rounds up to $0.02
    assert taker_fee(0.50, 1) == pytest.approx(0.02)
    # 100 contracts at 50c: 0.07*100*0.25 = 1.75 exactly
    assert taker_fee(0.50, 100) == pytest.approx(1.75)


def test_taker_fee_small_at_extremes():
    # 0.07 * 10 * 0.9 * 0.1 = 0.063 -> 0.07
    assert taker_fee(0.90, 10) == pytest.approx(0.07)


def test_fee_multiplier_scales():
    assert taker_fee(0.50, 100, multiplier=0.5) == pytest.approx(0.88)  # 0.875 rounds up


def test_maker_fee_quarter_rate():
    # 0.0175 * 100 * 0.25 = 0.4375 -> 0.44
    assert maker_fee(0.50, 100) == pytest.approx(0.44)


def test_fee_for_routes_by_fee_type():
    assert fee_for("taker", 0.5, 100, "quadratic", 1) == pytest.approx(1.75)
    assert fee_for("maker", 0.5, 100, "quadratic", 1) == 0.0
    assert fee_for("maker", 0.5, 100, "quadratic_with_maker_fees", 1) == pytest.approx(0.44)


def test_fractional_contracts_are_allowed():
    # 0.07 * 2.5 * .25 = 0.04375 -> 0.05
    assert taker_fee(0.50, 2.5) == pytest.approx(0.05)


# ---------- per-trade returns ----------

def _t(**kw):
    base = dict(ticker="X-1", event="X", ts=0.0, yes_price=0.80, count=10.0,
                taker_side="yes", result="yes", fee_type="quadratic", fee_mult=1.0)
    base.update(kw)
    return Trade(**base)


def test_taker_buys_yes_and_wins():
    r = trade_returns(_t())
    # taker paid 0.80 for yes, won 1.00; fee 0.07*10*.8*.2=0.112 -> 0.12 total -> 0.012/contract
    assert r.taker_side == "yes" and r.taker_price == pytest.approx(0.80)
    assert r.taker_ret == pytest.approx((1.0 - 0.80 - 0.012) / 0.80)
    # maker held the other side (no) at 0.20 and lost everything; no maker fee
    assert r.maker_side == "no" and r.maker_price == pytest.approx(0.20)
    assert r.maker_ret == pytest.approx(-1.0)


def test_taker_buys_no_maker_holds_yes():
    r = trade_returns(_t(taker_side="no", yes_price=0.90, result="yes"))
    assert r.taker_price == pytest.approx(0.10)
    # fee 0.07*10*.1*.9 = 0.063 -> 0.07 total -> 0.007/contract; lost the 0.10 stake too
    assert r.taker_ret == pytest.approx((0.0 - 0.10 - 0.007) / 0.10)
    assert r.maker_side == "yes" and r.maker_price == pytest.approx(0.90)
    assert r.maker_ret == pytest.approx((1.0 - 0.90) / 0.90)


def test_unsettled_or_voided_market_is_skipped():
    assert trade_returns(_t(result="")) is None
    assert trade_returns(_t(result="void")) is None


# ---------- aggregation ----------

def test_bucket_of_10c_bands():
    assert bucket_of(0.05) == "00-10"
    assert bucket_of(0.10) == "10-20"
    assert bucket_of(0.55) == "50-60"
    assert bucket_of(0.999) == "90-100"


def test_aggregate_is_dollar_weighted():
    # Two maker positions: $8 staked returning +25%, $2 staked returning -100%
    rows = [
        dict(role="maker", price=0.80, stake=8.0, pnl=2.0, event="A", week="w1", category="C", series="S"),
        dict(role="maker", price=0.80, stake=2.0, pnl=-2.0, event="B", week="w1", category="C", series="S"),
    ]
    agg = aggregate(rows, keys=("role",))
    g = agg[("maker",)]
    assert g["stake"] == pytest.approx(10.0)
    assert g["ret"] == pytest.approx(0.0)
    assert g["n"] == 2


def test_cluster_bootstrap_ci_brackets_mean_and_is_deterministic():
    import random as _r
    g = _r.Random(1)
    rows = [dict(stake=1.0, pnl=g.choice([0.3, -0.2, 0.05, -0.1]), event=f"E{i // 4}")
            for i in range(200)]
    lo, hi = cluster_bootstrap_ci(rows, n_boot=300, seed=7)
    mean = sum(r["pnl"] for r in rows) / sum(r["stake"] for r in rows)
    assert lo < mean < hi
    assert (lo, hi) == cluster_bootstrap_ci(rows, n_boot=300, seed=7)


def test_run_study_end_to_end(tmp_path):
    from kalshi.store import Store
    from kalshi.study import run_study, report
    s = Store(tmp_path / "k.db")
    s.upsert_series([{"ticker": "S", "category": "Crypto", "frequency": "fifteen_min",
                      "fee_type": "quadratic", "fee_multiplier": 1, "title": "t"}])
    s.upsert_markets("S", [
        {"ticker": "M1", "event_ticker": "E1", "close_time": "2026-09-01T06:00:00Z", "result": "yes"},
        {"ticker": "M2", "event_ticker": "E2", "close_time": "2026-09-02T06:00:00Z", "result": "no"},
    ])
    s.insert_trades([
        # taker buys NO at 0.20 on M1 (yes wins): maker holds YES at 0.80 -> wins
        {"trade_id": "a", "ticker": "M1", "created_time": "2026-09-01T05:58:00Z",
         "yes_price_dollars": "0.80", "count_fp": "10", "taker_side": "no"},
        # taker buys YES at 0.70 on M2 (no wins): taker loses, maker holds NO at 0.30 -> wins
        {"trade_id": "b", "ticker": "M2", "created_time": "2026-09-02T05:00:00Z",
         "yes_price_dollars": "0.70", "count_fp": "10", "taker_side": "yes"},
    ])
    s.mark_fetched("M1"); s.mark_fetched("M2")
    res = run_study(s, 0, log=lambda m: None)
    t = res["table"]
    assert res["n_trades"] == 2
    assert t[("role", "maker")]["pnl"] > 0 and t[("role", "taker")]["pnl"] < 0
    # maker fav only the M1 position (0.80); M2 maker was 0.30 (dog)
    assert t[("maker_fav_ttc", "a <5m")]["n"] == 1
    assert "By role" in report(res)
