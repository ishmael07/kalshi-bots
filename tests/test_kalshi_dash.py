import json

from kalshi.dash import Titles, build, kalshi_url


def test_trades_join_fills_settles_marks_and_links(tmp_path):
    j = tmp_path / "journal.jsonl"
    rows = [dict(kind="quote", ts=1, ticker="KXHIGHNY-26NOV08-T68", side="no", price=0.9, count=2, q=0.95),
            dict(kind="fill", ts=2, ticker="KXHIGHNY-26NOV08-T68", side="no", price=0.9, contracts=2.0),
            dict(kind="fill", ts=3, ticker="KXHIGHNY-26NOV07-T61", side="yes", price=0.6, contracts=1.0),
            dict(kind="settle", ts=10, ticker="KXHIGHNY-26NOV07-T61", result="no", pnl=-0.6)]
    j.write_text("\n".join(json.dumps(r) for r in rows))
    state = dict(start_equity=75.0, equity=74.4, ts=20,
                 catalog={"KXHIGHNY-26NOV08-T68": dict(title="Will the **high** be >68°?", sub="69° or above",
                                                       series="KXHIGHNY", event="KXHIGHNY-26NOV08", close_ts=99)},
                 marks={"KXHIGHNY-26NOV08-T68": dict(yes=0.02, no=0.97)})
    t = Titles(tmp_path / "titles.json", per_call=0)
    t.c["series"]["KXHIGHNY"] = "Highest temperature in NYC"
    d = build(j, state, t)
    o = d["open"][0]
    assert o["title"] == "Will the high be >68°?" and o["q"] == 0.95
    assert o["pnl"] == round(2 * 0.97 - 1.8, 2)
    assert o["url"] == "https://kalshi.com/markets/kxhighny/highest-temperature-in-nyc/kxhighny-26nov08"
    s = d["settled"][0]
    assert s["status"] == "lost" and s["pnl"] == -0.6
    st = d["stats"]
    assert {k: st[k] for k in ("settled", "wins", "losses", "realized", "unrealized", "staked")} == \
        dict(settled=1, wins=0, losses=1, realized=-0.6, unrealized=0.14, staked=1.8)
    assert st["total"] == round(-0.6 + 0.14, 2) and st["equity_marked"] == round(75 - 0.6 + 0.14, 2)
    assert st["today"] == round(st["today_realized"] + 0.14, 2)
    assert [p["equity"] for p in d["curve"]] == [75.0, 74.4, 74.54] and d["curve"][-1]["live"]


def test_kalshi_url_falls_back_to_series_page():
    assert kalshi_url("KXCPI", "", "") == "https://kalshi.com/markets/kxcpi"


def test_sold_early_row(tmp_path):
    j = tmp_path / "journal.jsonl"
    rows = [dict(kind="fill", ts=2, ticker="KXHIGHNY-26NOV08-B61.5", side="yes", price=0.6, contracts=2.0),
            dict(kind="exit", ts=5, ticker="KXHIGHNY-26NOV08-B61.5", side="yes", price=0.05, contracts=2.0, fee=0.01)]
    j.write_text("\n".join(json.dumps(r) for r in rows))
    d = build(j, dict(start_equity=75.0, ts=9), Titles(tmp_path / "t.json", per_call=0))
    r = d["settled"][0]
    assert r["status"] == "sold" and r["pnl"] == round(2 * 0.05 - 0.01 - 1.2, 2) and not d["open"]
