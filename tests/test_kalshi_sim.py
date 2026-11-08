import pytest

from kalshi.sim import Candidate, simulate, kelly_fraction


def test_kelly_fraction_binary_contract():
    # buy at 0.60 with true prob 0.70: f* = (q - p) / (1 - p) = 0.25
    assert kelly_fraction(0.70, 0.60) == pytest.approx(0.25)
    assert kelly_fraction(0.55, 0.60) == 0.0


def _c(ts, close, price, won, q=0.9, avail=100.0, event="E", fee=0.0):
    return Candidate(ts=ts, close_ts=close, event=event, price=price, won=won, q=q,
                     available=avail, fee_per_contract=fee)


def test_win_grows_cash_after_settlement():
    r = simulate([_c(0, 10, 0.80, True)], start=100.0, kelly_mult=0.25, fill_share=1.0, max_frac=1.0,
                 event_cap=1.0)
    # f* = (0.9-0.8)/0.2 = 0.5 -> quarter = 0.125 -> $12.5 stake -> 15.625 contracts -> +$3.125
    assert r.final == pytest.approx(103.125)
    assert r.trades == 1 and r.wins == 1


def test_cash_is_locked_until_settlement():
    cands = [_c(0, 100, 0.80, True, event="A"), _c(1, 100, 0.80, True, event="B")]
    r = simulate(cands, start=100.0, kelly_mult=1.0, fill_share=1.0, event_cap=1.0, max_frac=1.0)
    # first bet uses 50% of equity; the second can only use remaining cash, sized on equity
    assert r.trades == 2
    assert r.max_exposure <= 1.0 + 1e-9


def test_event_cap_limits_correlated_exposure():
    cands = [_c(0, 10, 0.80, True, event="A"), _c(1, 10, 0.80, True, event="A")]
    r = simulate(cands, start=100.0, kelly_mult=1.0, fill_share=1.0, event_cap=0.10)
    assert r.staked_by_event["A"] <= 10.0 + 1e-9


def test_fill_share_limits_size():
    r = simulate([_c(0, 10, 0.80, True, avail=4.0)], start=100.0, kelly_mult=0.25,
                 fill_share=0.25, event_cap=1.0)
    # at most 1 contract (25% of 4)
    assert r.final == pytest.approx(100.2)


def test_loss_and_drawdown_and_fees():
    r = simulate([_c(0, 10, 0.80, False, fee=0.01)], start=100.0, kelly_mult=0.25,
                 fill_share=1.0, event_cap=1.0)
    assert r.final < 100.0
    assert r.max_drawdown > 0


def test_min_order_skips_dust():
    r = simulate([_c(0, 10, 0.80, True, q=0.801)], start=10.0, kelly_mult=0.25,
                 fill_share=1.0, event_cap=1.0, min_stake=1.0)
    assert r.trades == 0


def test_cap_applies_per_correlation_group():
    from kalshi.sim import Candidate
    cands = [Candidate(0, 10, "BTC-1", 0.80, True, 0.9, 100, 0.0, group="Crypto|1"),
             Candidate(1, 10, "ETH-1", 0.80, True, 0.9, 100, 0.0, group="Crypto|1")]
    r = simulate(cands, start=100.0, kelly_mult=1.0, fill_share=1.0, event_cap=0.10, max_frac=1.0)
    assert r.staked_by_event["Crypto|1"] <= 10.0 + 1e-9


def test_corr_group_rules():
    from kalshi.run_strategy import corr_group
    assert corr_group("Crypto", "KXBTC15M-X", 7200.5) == corr_group("Crypto", "KXETH15M-Y", 7300.0)
    assert corr_group("Economics", "KXCPI-26OCT", 86400 * 5 + 10) == corr_group("Economics", "KXCPIYOY-26OCT", 86400 * 5 + 99)
    assert corr_group("Climate and Weather", "KXHIGHNY-26NOV07", 0) == "KXHIGHNY-26NOV07"


def test_fixed_fractional_sizing_caps_each_bet():
    r = simulate([_c(0, 10, 0.95, False, q=0.99)], start=100.0, kelly_mult=0.25, fill_share=1.0,
                 event_cap=1.0, max_frac=1.0, sizing="fixed", fixed_frac=0.01)
    assert r.final == pytest.approx(99.0)        # lost exactly the 1% stake
