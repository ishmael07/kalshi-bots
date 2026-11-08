import pytest

from kalshi.live import CellModel, Quoter, Book, ShadowExecutor, VirtualOrder, Breakers


def _model():
    m = CellModel(min_prior=30, shrink=0)
    # weather favorites at 60-65c, 6-24h before close won 72% across 100 events
    m.table["Climate and Weather|60-65|e 6-24h"] = [72.0, 100.0]
    m.table["Climate and Weather|55-60|e 6-24h"] = [57.0, 100.0]   # fair, no edge
    return m


def _book(**kw):
    base = dict(ticker="KXHIGHNY-26NOV07-B55.5", event="KXHIGHNY-26NOV07", category="Climate and Weather",
                close_ts=10 * 3600, yes_bids=[(0.60, 50.0)], no_bids=[(0.35, 40.0)], tick=0.01)
    base.update(kw)
    return Book(**base)


def test_cell_model_q_needs_min_prior():
    m = _model()
    assert m.q("Climate and Weather|60-65|e 6-24h") == pytest.approx(0.72)
    assert m.q("Nope|60-65|e 6-24h") is None


def test_quoter_improves_best_bid_when_edge_exists():
    q = Quoter(_model(), margin=0.03, kelly_mult=0.25, max_frac=0.10)
    want = q.desired(_book(), now=0, equity=100.0)
    # yes best bid 0.60, yes ask = 1 - 0.35 = 0.65 -> improve to 0.61 (still maker), cell 60-65
    assert len(want) == 1
    o = want[0]
    assert o.side == "yes" and o.price == pytest.approx(0.61)
    # quarter Kelly: (0.72-0.61)/(0.39) * .25 = 0.0705 of equity -> $7.05 -> 11.5 contracts
    assert o.count == pytest.approx(7.05 / 0.61, rel=0.02)


def test_quoter_never_crosses_the_spread():
    q = Quoter(_model(), margin=0.03)
    # yes bid 0.60, no bid 0.39 -> yes ask 0.61; improving to 0.61 would cross -> join at 0.60
    want = q.desired(_book(no_bids=[(0.39, 10.0)]), now=0, equity=100.0)
    assert want and want[0].price == pytest.approx(0.60) and want[0].improved is False


def test_quoter_skips_without_edge_or_longshots():
    q = Quoter(_model(), margin=0.03)
    assert q.desired(_book(yes_bids=[(0.55, 5.0)], no_bids=[(0.40, 5.0)]), now=0, equity=100) == []
    # no side at 0.35 is a longshot -> never quoted
    assert all(o.side == "yes" for o in q.desired(_book(), now=0, equity=100))


def test_shadow_fill_requires_tape_through_our_price():
    ex = ShadowExecutor(fill_share=0.25)
    o = VirtualOrder(ticker="T", event="E", side="yes", price=0.61, count=10.0, placed_ts=100.0,
                     improved=True)
    ex.place(o)
    # trade before we posted: ignored. taker bought YES (lifts asks): doesn't hit our bid.
    ex.on_trades("T", [dict(ts=90.0, yes_price=0.60, count=5.0, taker_side="no"),
                       dict(ts=110.0, yes_price=0.62, count=5.0, taker_side="yes")])
    assert o.filled == 0
    # taker sold YES (bought NO) at yes 0.61 after we posted: we were top of book -> full fill
    ex.on_trades("T", [dict(ts=120.0, yes_price=0.61, count=4.0, taker_side="no")])
    assert o.filled == pytest.approx(4.0)


def test_shadow_join_gets_queue_haircut_and_no_side_mapping():
    ex = ShadowExecutor(fill_share=0.25)
    o = VirtualOrder(ticker="T", event="E", side="no", price=0.70, count=10.0, placed_ts=0.0,
                     improved=False)
    ex.place(o)
    # NO bid at 0.70 is hit when a taker buys YES at yes_price >= 0.30
    ex.on_trades("T", [dict(ts=5.0, yes_price=0.30, count=8.0, taker_side="yes")])
    assert o.filled == pytest.approx(2.0)


def test_shadow_settlement_pnl():
    ex = ShadowExecutor(fill_share=1.0)
    o = VirtualOrder(ticker="T", event="E", side="yes", price=0.60, count=10.0, placed_ts=0.0,
                     improved=True)
    ex.place(o)
    ex.on_trades("T", [dict(ts=1.0, yes_price=0.60, count=10.0, taker_side="no")])
    pnl = ex.settle("T", "yes")
    assert pnl == pytest.approx(10 * (1 - 0.60))


def test_breakers_cancel_category_after_headline_and_expire():
    b = Breakers(cooldown_s=600)
    b.trip("Financials", "Fed surprise cut", now=1000)
    assert b.blocked("Financials", now=1200)
    assert not b.blocked("Climate and Weather", now=1200)
    assert not b.blocked("Financials", now=1700)
    b.trip("*", "stale data", now=2000)
    assert b.blocked("Climate and Weather", now=2001)


def test_q_is_shrunk_toward_price_so_small_samples_cannot_claim_certainty():
    m = CellModel(min_prior=30, shrink=10)
    m.table["Economics|95-100|f >1d"] = [30.0, 30.0]       # 30 of 30 events won
    q = m.q("Economics|95-100|f >1d", 0.96)
    assert q < 1.0 and q == pytest.approx((30 + 10 * 0.96) / 40)


def test_fixed_fraction_sizing_and_book_persistence(tmp_path):
    q = Quoter(_model(), margin=0.03, sizing="fixed", fixed_frac=0.01)
    o = q.desired(_book(), now=0, equity=100.0)[0]
    assert o.count * o.price == pytest.approx(1.0, rel=0.02)        # 1% of equity
    ex = ShadowExecutor()
    ex.positions = {"T": [("yes", 0.6, 2.0, "E")]}
    ex.realized = 1.5
    ex.save(tmp_path / "b.json")
    ex2 = ShadowExecutor()
    ex2.load(tmp_path / "b.json")
    assert ex2.positions == {"T": [("yes", 0.6, 2.0, "E")]} and ex2.realized == 1.5


def test_executor_sell_realizes_pnl_oldest_lots_first():
    from kalshi.live import ShadowExecutor
    ex = ShadowExecutor()
    ex.positions["T"] = [("no", 0.80, 2.0, "E"), ("no", 0.70, 3.0, "E"), ("yes", 0.10, 1.0, "E")]
    pnl = ex.sell("T", "no", 0.05, 4.0, fee=0.01)
    assert pnl == pytest.approx(2 * (0.05 - 0.80) + 2 * (0.05 - 0.70) - 0.01)
    assert ex.positions["T"] == [("no", 0.70, 1.0, "E"), ("yes", 0.10, 1.0, "E")]
    assert ex.realized == pytest.approx(pnl)
