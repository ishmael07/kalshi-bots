import math
import random

import pytest

from kalshi.gates import deflated_sharpe, gate_report
from kalshi.strategy import Fill, walk_forward


def test_dsr_high_for_strong_consistent_series():
    rng = random.Random(0)
    weekly = [0.02 + rng.gauss(0, 0.01) for _ in range(60)]
    assert deflated_sharpe(weekly, n_trials=20) > 0.95


def test_dsr_low_for_noise_with_many_trials():
    rng = random.Random(1)
    weekly = [rng.gauss(0.001, 0.02) for _ in range(30)]
    assert deflated_sharpe(weekly, n_trials=200) < 0.5


def test_more_trials_lower_dsr():
    rng = random.Random(2)
    weekly = [0.006 + rng.gauss(0, 0.02) for _ in range(52)]
    assert deflated_sharpe(weekly, n_trials=1) > deflated_sharpe(weekly, n_trials=500)


def _fill(week, price, won, cell="c", cnt=10.0, i=0):
    ts = week * 7 * 86400 + 1000
    return Fill(ts=ts, close_ts=ts + 600, week=week, event=f"E{week}-{cell}-{i}",
                cell=cell, price=price, won=won, count=cnt, fee_per_contract=0.0)


def test_walk_forward_uses_only_prior_weeks():
    fills = []
    # cell "good": priced 0.60 but wins 80% of the time in every week
    for w in range(12):
        for i in range(10):
            fills.append(_fill(w, 0.60, i < 8, cell="good", i=i))
            fills.append(_fill(w, 0.60, i < 5, cell="bad", i=i))   # wins 50%: overpriced
    taken = walk_forward(fills, warmup_weeks=4, margin=0.05, min_prior=20, shrink=0)
    assert taken, "should take some fills"
    assert all(c.cell == "good" for c, _ in taken)
    assert min(f.week for f, _ in taken) == 4               # nothing before warmup
    # q estimate for week 4 comes only from weeks 0-3
    f0, q0 = taken[0]
    assert q0 == pytest.approx(0.8)


def test_walk_forward_no_lookahead_when_regime_flips():
    fills = []
    for w in range(10):
        p_win = 8 if w < 6 else 2      # great for 6 weeks, then terrible
        for i in range(10):
            fills.append(_fill(w, 0.60, i < p_win, cell="x", i=i))
    taken = walk_forward(fills, warmup_weeks=4, margin=0.05, min_prior=20, shrink=0)
    weeks = sorted({f.week for f, _ in taken})
    # It keeps trading into the bad regime for a while (it cannot know the future)
    assert 6 in weeks


def test_gate_report_shape():
    weekly = [0.01] * 20 + [-0.005] * 5
    g = gate_report(weekly_returns=weekly, n_trades=500, n_trials=10,
                    neighbor_returns=[0.009, 0.011], base_return=0.01, stressed_return=0.002)
    assert set(g) >= {"trade_count", "oos_positive", "deflated_sharpe", "walk_forward",
                      "latest_period", "parameter_plateau", "noise_test", "all_pass"}
    assert g["walk_forward"]["value"] == pytest.approx(20 / 25)


def test_one_giant_event_cannot_dominate_a_cell():
    from kalshi.strategy import CellStats
    st = CellStats()
    st.add("BIG", 100000, True)          # one huge winning event
    for i in range(9):
        st.add(f"e{i}", 10, False)       # nine normal losing events
    assert st.q() == pytest.approx(0.1)
    assert st.n_events == 10
