import json

import pytest

from kalshi.bot import Bot, BotConfig
from kalshi.live import CellModel, ShadowExecutor, Breakers


class FakePublic:
    """Minimal stand-in for KalshiPublic used by the bot."""

    def __init__(self):
        self.markets = {"KXHIGHNY": [dict(ticker="T1", event_ticker="E1", close_time="1970-01-01T10:00:00Z",
                                          status="open", price_level_structure="linear_cent",
                                          yes_bid_dollars="0.6000", no_bid_dollars="0.3500")]}
        self.books = {"T1": {"orderbook_fp": {"yes_dollars": [["0.5500", "10"], ["0.6000", "50"]],
                                              "no_dollars": [["0.3000", "5"], ["0.3500", "40"]]}}}
        self.tape = {"T1": []}
        self.results = {}
        self.series_meta = {"KXHIGHNY": {"ticker": "KXHIGHNY", "category": "Climate and Weather"}}

    def series(self, t):
        return self.series_meta[t]

    def paginate(self, path, key, params=None, limit=1000):
        if path == "/markets":
            yield from self.markets.get(params["series_ticker"], [])
        elif path == "/markets/trades":
            yield from [t for t in self.tape.get(params["ticker"], []) if t["_ts"] > params.get("min_ts", 0)]

    def get(self, path, params=None):
        if path.endswith("/orderbook"):
            return self.books[path.split("/")[2]]
        if path.startswith("/markets/"):
            tk = path.split("/")[2]
            return {"market": {"ticker": tk, "result": self.results.get(tk, ""),
                               "status": "finalized" if tk in self.results else "open"}}
        raise AssertionError(path)


def _bot(tmp_path, pub):
    m = CellModel(min_prior=30, shrink=0)
    m.table["Climate and Weather|60-65|d 1-6h"] = [80.0, 100.0]
    cfg = BotConfig(series=["KXHIGHNY"], start_equity=75.0, margin=0.03, out_dir=str(tmp_path),
                    min_ttc_s=60, max_ttc_s=86400, max_order_dollars=10, max_open_dollars=60,
                    max_event_dollars=10, daily_loss_halt=5)
    return Bot(cfg, pub, m, ShadowExecutor(fill_share=0.25), Breakers(), news=None, spots=[],
               notify=lambda *a, **k: None)


def test_bot_quotes_fills_and_settles(tmp_path):
    pub = FakePublic()
    bot = _bot(tmp_path, pub)
    bot.step(now=3 * 3600.0)                     # 7h before close = "e 6-24h": no trusted cell
    assert bot.executor.orders == {} and bot.markets == {}
    bot.last_discover = -1e18
    bot.step(now=6 * 3600.0)                     # 4h before close -> d 1-6h cell has edge
    rest = bot.executor.resting("T1")
    assert len(rest) == 1 and rest[0].side == "yes" and rest[0].price == pytest.approx(0.61)
    pub.tape["T1"] = [dict(_ts=6 * 3600.0 + 30, ts=6 * 3600.0 + 30, yes_price=0.61, count=3.0,
                           taker_side="no")]
    pub.tape["T1"][0].update({"created_time": "1970-01-01T06:00:30Z", "yes_price_dollars": "0.6100",
                              "count_fp": "3.00", "trade_id": "x"})
    bot.step(now=6 * 3600.0 + 60)
    assert bot.executor.positions["T1"][0][2] == pytest.approx(3.0)
    pub.results["T1"] = "yes"
    bot.step(now=10 * 3600.0 + 120)
    assert bot.executor.realized == pytest.approx(3.0 * (1 - 0.61))
    state = json.load(open(tmp_path / "state.json"))
    assert state["equity"] == pytest.approx(75.0 + 3.0 * 0.39)
    kinds = [json.loads(l)["kind"] for l in open(tmp_path / "journal.jsonl")]
    assert "quote" in kinds and "fill" in kinds and "settle" in kinds


def test_breaker_cancels_and_blocks_quotes(tmp_path):
    pub = FakePublic()
    bot = _bot(tmp_path, pub)
    bot.step(now=6 * 3600.0)
    assert bot.executor.resting("T1")
    bot.breakers.trip("Climate and Weather", "Hurricane warning", now=6 * 3600.0 + 10)
    bot.step(now=6 * 3600.0 + 20)
    assert not bot.executor.resting("T1")


def test_risk_guard_blocks_oversize(tmp_path):
    pub = FakePublic()
    bot = _bot(tmp_path, pub)
    bot.cfg.max_order_dollars = 0.5
    bot.guard.max_order = 0.5
    bot.step(now=6 * 3600.0)
    assert not bot.executor.resting("T1")


def test_disk_full_does_not_kill_the_loop(tmp_path, monkeypatch):
    pub = FakePublic()
    bot = _bot(tmp_path, pub)

    def boom(*a, **k):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(bot.executor, "save", boom)
    bot.step(now=6 * 3600.0)                 # must not raise
    assert bot.disk_errors >= 1
    assert bot.executor.resting("T1")        # trading logic still ran


class _Opinion:
    name = "fake"

    def __init__(self, view):
        self._v = view

    def poll(self, now):
        pass

    def status(self, now):
        return [dict(name="Fake feed", ok=True, detail="x", age=0)]

    def view(self, m, now):
        return self._v


def test_signal_disagreement_is_logged_not_enforced_by_default(tmp_path):
    pub = FakePublic()
    bot = _bot(tmp_path, pub)
    bot.signals = [_Opinion({"p_yes": 0.10, "why": "model says no"})]
    bot.step(now=6 * 3600.0)
    assert bot.executor.resting("T1")                       # still quoted
    assert bot.vetoes[0]["enforced"] is False
    kinds = [json.loads(l)["kind"] for l in open(tmp_path / "journal.jsonl")]
    assert "disagree" in kinds
    st = json.load(open(tmp_path / "state.json"))
    assert any(f["name"] == "Fake feed" for f in st["data_feeds"]) and st["vetoes"]


def test_enforced_signal_blocks_the_bid(tmp_path):
    pub = FakePublic()
    bot = _bot(tmp_path, pub)
    bot.cfg.veto_enforce = ["fake"]
    bot.signals = [_Opinion({"p_yes": 0.10, "why": "model says no"})]
    bot.step(now=6 * 3600.0)
    assert not bot.executor.resting("T1")


def test_series_breaker_blocks_only_that_series(tmp_path):
    pub = FakePublic()
    bot = _bot(tmp_path, pub)
    bot.breakers.trip("KXHIGHNY", "release", now=6 * 3600.0 - 1, cooldown_s=600)
    bot.step(now=6 * 3600.0)
    assert not bot.executor.resting("T1")
    bot.breakers.until.clear()
    bot.breakers.trip("KXFED", "release", now=6 * 3600.0, cooldown_s=600)
    bot.step(now=6 * 3600.0 + 30)
    assert bot.executor.resting("T1")


def test_data_lock_bids_the_decided_side(tmp_path):
    pub = FakePublic()
    bot = _bot(tmp_path, pub)
    bot.signals = [_Opinion({"locked": "no", "dead": "yes", "why": "published"})]
    bot.step(now=6 * 3600.0)
    rest = bot.executor.resting("T1")
    assert [o.side for o in rest] == ["no"]


def test_marked_equity_snapshots_for_the_live_chart(tmp_path):
    pub = FakePublic()
    bot = _bot(tmp_path, pub)
    bot.executor.positions["T1"] = [("yes", 0.60, 10.0, "E1")]
    bot.marks["T1"] = dict(yes=0.70, no=0.28, ts=0)
    assert bot.equity_marked() == pytest.approx(75.0 + 10 * 0.10)
    bot._snapshot(100.0)
    bot._snapshot(130.0)                         # < 60s later: skipped
    lines = open(tmp_path / "equity.jsonl").read().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["equity"] == pytest.approx(76.0)


def test_sells_a_position_that_live_data_proves_lost(tmp_path):
    pub = FakePublic()
    bot = _bot(tmp_path, pub)
    bot.step(now=6 * 3600.0)                                 # discover T1
    bot.executor.positions["T1"] = [("yes", 0.60, 5.0, "E1")]
    bot.signals = [_Opinion({"locked": "no", "dead": "yes", "why": "station hit 64F"})]
    bot.step(now=6 * 3600.0 + 30)                            # yes bid is 0.60 in the fake book
    assert "T1" not in bot.executor.positions or all(s != "yes" for s, *_ in bot.executor.positions["T1"])
    ex = [json.loads(l) for l in open(tmp_path / "journal.jsonl") if '"exit"' in l]
    assert ex and ex[0]["side"] == "yes" and ex[0]["price"] == pytest.approx(0.60)
    assert bot.executor.realized == pytest.approx(5 * (0.60 - 0.60) - ex[0]["fee"])


def test_holds_when_nothing_is_decided(tmp_path):
    pub = FakePublic()
    bot = _bot(tmp_path, pub)
    bot.step(now=6 * 3600.0)
    bot.executor.positions["T1"] = [("yes", 0.60, 5.0, "E1")]
    bot.signals = [_Opinion({"p_yes": 0.05, "why": "forecast says no"})]   # opinion, not proof
    bot.step(now=6 * 3600.0 + 30)
    assert bot.executor.positions["T1"][0][2] == pytest.approx(5.0)
