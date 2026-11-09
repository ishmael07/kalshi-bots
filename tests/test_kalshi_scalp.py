import pytest

from kalshi.bot import BotConfig
from kalshi.fees import maker_fee, taker_fee
from kalshi.scalp import ScalpBot, ScalpExecutor, ScalpParams

iso = lambda s: "1970-01-01T%02d:%02d:%02dZ" % (s // 3600, s % 3600 // 60, s % 60)


class Pub:
    def __init__(self):
        self.listing = [dict(ticker="T", event_ticker="SER-1", close_time=iso(20000), expected_expiration_time=iso(19000),
                             volume_24h_fp="5000", yes_bid_dollars="0.4000", yes_ask_dollars="0.4500", title="t",
                             yes_sub_title="s"),
                        dict(ticker="TIGHT", event_ticker="SER-2", close_time=iso(20000), volume_24h_fp="9000",
                             yes_bid_dollars="0.4000", yes_ask_dollars="0.4100"),
                        dict(ticker="THIN", event_ticker="SER-3", close_time=iso(20000), volume_24h_fp="10",
                             yes_bid_dollars="0.4000", yes_ask_dollars="0.5000")]
        self.book = dict(yes=[["0.4000", "50"]], no=[["0.5500", "50"]])
        self.tape = []
        self.results = {}

    def series(self, s):
        return {"category": "Sports"}

    def paginate(self, path, key, params=None, limit=1000):
        if path == "/markets":
            assert params["mve_filter"] == "exclude"
            yield from self.listing
        else:
            yield from self.tape

    def get(self, path, params=None):
        if path.endswith("/orderbook"):
            return {"orderbook_fp": {"yes_dollars": self.book["yes"], "no_dollars": self.book["no"]}}
        return {"market": {"result": self.results.get(path.split("/")[2], "")}}


def trade(s, yes_price, count, taker):
    return dict(created_time=iso(s), yes_price_dollars=str(yes_price), count_fp=str(count), taker_side=taker)


def make(tmp_path, pub, **kw):
    cfg = BotConfig(series=[], start_equity=75.0, fill_share=1.0, out_dir=str(tmp_path), discover_every_s=1e9)
    return ScalpBot(cfg, pub, ScalpExecutor(1.0), ScalpParams(**kw))


def test_picks_busy_wide_markets_and_quotes_both_sides(tmp_path):
    pub = Pub()
    b = make(tmp_path, pub)
    b.step(now=1000.0)
    assert set(b.markets) == {"T"}                                # tight spread and thin volume are skipped
    assert b.markets["T"]["stop_ts"] == 19000 - 600               # flat before expected resolution, not close_time
    rest = {o.side: o for o in b.executor.resting("T")}
    assert rest["yes"].price == 0.41 and rest["no"].price == 0.56            # 5c spread: step in front of the line
    assert rest["yes"].count == rest["no"].count == int(7.5 / 0.56)


def test_round_trip_locks_the_spread_and_flips_flat(tmp_path):
    pub = Pub()
    b = make(tmp_path, pub)
    b.step(now=1000.0)
    n = b.executor.resting("T")[0].count
    pub.tape = [trade(1005, 0.41, n, "no")]                       # someone sells to our YES bid
    b.step(now=1010.0)
    assert b.executor.held("T") == ("yes", n)
    assert [o.side for o in b.executor.resting("T")] == ["no"]    # only the exit is left up
    pub.tape = [trade(1015, 0.44, n, "yes")]                      # someone buys from our NO bid (YES at 44c)
    b.step(now=1020.0)
    assert b.executor.held("T") == (None, 0.0) and "T" not in b.executor.positions and b.executor.trips == {}
    fees = maker_fee(0.41, n) + maker_fee(0.56, n)
    assert b.executor.realized == pytest.approx(n * 0.03 - fees)
    assert len(b.executor.resting("T")) == 2                      # next trade straight away


def test_time_stop_sells_into_the_bid(tmp_path):
    pub = Pub()
    b = make(tmp_path, pub, max_hold_s=60)
    b.step(now=1000.0)
    n = b.executor.resting("T")[0].count
    pub.tape = [trade(1005, 0.41, n, "no")]
    b.step(now=1010.0)
    pub.tape = []
    pub.book = dict(yes=[["0.3500", "500"]], no=[["0.6000", "50"]])
    b.step(now=1100.0)
    assert b.executor.held("T") == (None, 0.0)
    assert b.executor.realized == pytest.approx(n * (0.35 - 0.41) - maker_fee(0.41, n) - taker_fee(0.35, n))


def test_flat_before_resolution_and_leftovers_settle(tmp_path):
    pub = Pub()
    b = make(tmp_path, pub)
    b.step(now=1000.0)
    b.step(now=18500.0)                                           # inside stop window
    assert b.executor.resting("T") == []
    b2 = make(tmp_path / "b", pub)
    b2.executor.positions["T"] = [("yes", 0.4, 5.0, "SER-1")]
    b2.executor.trips["T"] = dict(id=1, since=1.0)
    b2.markets["T"] = dict(close_ts=500.0, event="SER-1", tick=0.01, category="x", group="SER-1")
    pub.results["T"] = "yes"
    pub.listing = []                                              # a closed market is no longer listed as open
    b2.step(now=1000.0)
    assert b2.executor.realized == pytest.approx(5 * 0.6) and b2.executor.trips == {}


def test_book_and_trips_survive_a_restart(tmp_path):
    a = ScalpExecutor()
    a.positions["T"] = [("yes", 0.4, 5.0, "E")]
    a.trips["T"] = dict(id=7, since=7.0)
    a.save(tmp_path / "book.json")
    c = ScalpExecutor()
    c.load(tmp_path / "book.json")
    assert c.held("T") == ("yes", 5.0) and c.trips["T"]["id"] == 7


def test_joining_the_line_waits_behind_the_queue(tmp_path):
    pub = Pub()
    pub.book = dict(yes=[["0.4000", "50"]], no=[["0.5800", "70"]])    # 2c spread: join, 50 contracts ahead of our bid
    b = make(tmp_path, pub)
    b.step(now=1000.0)
    assert {o.side: o.price for o in b.executor.resting("T")} == {"yes": 0.40, "no": 0.58}
    pub.tape = [trade(1005, 0.40, 30, "no")]
    b.step(now=1010.0)
    assert b.executor.held("T") == (None, 0.0)                        # 30 of the 50 ahead traded
    pub.tape = [trade(1015, 0.40, 23, "no")]
    b.step(now=1020.0)
    assert b.executor.held("T") == ("yes", 3.0)                       # 20 more clear the line, 3 reach us


def test_a_print_through_our_price_clears_the_line(tmp_path):
    pub = Pub()
    pub.book = dict(yes=[["0.4000", "50"]], no=[["0.5800", "70"]])
    b = make(tmp_path, pub)
    b.step(now=1000.0)
    pub.tape = [trade(1005, 0.39, 4, "no")]
    b.step(now=1010.0)
    assert b.executor.held("T") == ("yes", 4.0)
