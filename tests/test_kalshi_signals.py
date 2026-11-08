import json
from datetime import date, datetime, timezone

import pytest

from kalshi import signals as sg
from kalshi.feeds import BLSLatestSource, NewsHub, RSSSource


class Resp:
    def __init__(self, text="", js=None):
        self.text, self._js, self.status_code = text, js, 200
        self.content = text.encode("utf-8")

    def json(self):
        return self._js

    def raise_for_status(self):
        pass


class Sess:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append(url)
        for k, v in self.routes.items():
            if k in url:
                if isinstance(v, Exception):
                    raise v
                return v
        raise AssertionError(url)


ICS = """BEGIN:VCALENDAR\r
BEGIN:VEVENT\r
SUMMARY:Consumer Price Index\r
DTSTART;TZID=US-Eastern:20261110T083000\r
END:VEVENT\r
BEGIN:VEVENT\r
SUMMARY:Producer Price Index\r
DTSTART;TZID=US-Eastern:20261112T083000\r
END:VEVENT\r
BEGIN:VEVENT\r
SUMMARY:Employment Situation\r
DTSTART;TZID=US-Eastern:20261204T083000\r
END:VEVENT\r
END:VCALENDAR\r
"""
FOMC = ("<h4>2026 FOMC Meetings</h4><div>October</div><div>27-28</div><div>December</div><div>8-9*</div>"
        "<h4>2027 FOMC Meetings</h4><div>Jan/Feb</div><div>31-1</div>")


def test_bls_and_fomc_calendars_parse():
    ev = sg.parse_bls_ics(ICS)
    assert [k for k, _ in ev] == ["cpi", "jobs"]                     # PPI is ignored
    assert datetime.fromtimestamp(ev[0][1], sg.ET).strftime("%Y-%m-%d %H:%M") == "2026-11-10 08:30"
    f = sg.parse_fomc(FOMC)
    days = [datetime.fromtimestamp(t, sg.ET).strftime("%Y-%m-%d %H:%M") for _, t in f]
    assert "2026-12-09 14:00" in days and "2026-10-28 14:00" in days and "2027-02-01 14:00" in days


def test_calendar_pauses_fed_markets_after_cpi_but_never_cpi_itself():
    trips = []
    cal = sg.ReleaseCalendar(lambda *a: trips.append(a),
                             session=Sess({"bls.ics": Resp(ICS), "fomccalendars": Resp(FOMC)}))
    t = datetime(2026, 11, 10, 8, 30, tzinfo=sg.ET).timestamp()
    cal.poll(t - 600)                                   # before the print: nothing paused
    assert trips == []
    cal.poll(t + 60)
    keys = {a[0] for a in trips}
    assert keys == {"KXFED", "KXFEDDECISION"}
    assert all(a[3] == pytest.approx(1800 - 60) for a in trips)
    assert "CPI" in cal.status(t)[0]["detail"]


def test_calendar_learns_release_times_from_kalshi_markets():
    cal = sg.ReleaseCalendar(lambda *a: None)
    cal.note_markets({"T": dict(series="KXPAYROLLS", close_ts=1000.0)})
    cal.note_markets({"T": dict(series="KXPAYROLLS", close_ts=1000.0)})   # no duplicates
    assert list(cal.events) == [("jobs", 1060.0)]


def _clev(sub, nowcast, actual=None):
    ds = [{"seriesname": "CPI Inflation", "data": [{"value": ""}, {"value": str(nowcast)}]}]
    if actual is not None:
        ds.append({"seriesname": "Actual CPI Inflation", "data": [{"value": str(actual)}]})
    return {"chart": {"subcaption": sub}, "dataset": ds}


def test_nowcast_view_uses_one_decimal_rounding():
    s = Sess({"nowcast_month": Resp(js=[_clev("2026-9", 0.5, actual=0.6), _clev("2026-10", 0.24)]),
              "nowcast_year": Resp(js=[_clev("2026-10", 3.74)]),
              "fredgraph": Resp("observation_date,GDPNOW\n2026-07-01,3.27\n2026-10-01,3.44\n")})
    nc = sg.Nowcasts(session=s)
    nc.poll(0)
    assert (2026, 9) not in nc.cpi_mom                   # released months are dropped
    m = dict(series="KXCPI", event="KXCPI-26OCT", strike_type="greater", floor=0.2)
    # YES iff the one-decimal print > 0.2, i.e. raw >= 0.25: nowcast 0.24 is just below
    assert nc.view(m, 0)["p_yes"] == pytest.approx(1 - sg.ncdf((0.25 - 0.24) / 0.12))
    g = nc.view(dict(series="KXGDP", event="KXGDP-27JAN29", strike_type="greater", floor=3.0), 0)
    assert g["p_yes"] > 0.6 and "GDPNow" in g["why"]     # Jan release -> Q4 (Oct) nowcast 3.44
    assert sg.gdp_quarter((2027, 7, 29)) == (2027, 4) and sg.gdp_quarter((2027, 1, 29)) == (2026, 10)
    assert nc.view(dict(series="KXU3", event="KXU3-26NOV", strike_type="greater", floor=4.0), 0) == {}


def test_weather_thresholds_match_kalshi_whole_degree_rules():
    assert sg.wx_yes_threshold("greater", 68, None) == (68.5, None)          # "69 or above"
    assert sg.wx_yes_threshold("less", None, 61) == (None, 60.5)             # "60 or below"
    assert sg.wx_yes_threshold("between", 67, 68) == (66.5, 68.5)            # "67 to 68"
    assert sg.p_band([70.0], 68.5, None, 0.01) == pytest.approx(1.0)
    assert sg.p_band([67.0], 66.5, 68.5, 0.01) == pytest.approx(1.0)


def test_weather_forecast_view_floors_members_with_observations():
    from kalshi.weather import climate_day
    a, b = climate_day((2026, 11, 8), -5)
    times = [a + 3600 * i for i in range(24)]
    members = {f"temperature_2m_member{i:02d}": [50.0 + (i % 3)] * 24 for i in range(30)}
    s = Sess({"ensemble-api": Resp(js={"hourly": {"time": times, **members}}),
              "/points/": Resp(js={"properties": {"forecastHourly": "https://api.weather.gov/grid/hourly"}}),
              "grid/hourly": Resp(js={"properties": {"periods": [
                  {"startTime": datetime.fromtimestamp(t, timezone.utc).isoformat(), "temperature": 52,
                   "temperatureUnit": "F"} for t in times]}}),
              "alerts/active": Resp(js={"features": [{"properties": {"event": "Wind Advisory"}}]})})

    class Obs:
        def max_f(self, st, a_, b_):
            return 61.0                      # already hit 61 today
    wf = sg.WeatherForecast(Obs(), ["KXHIGHNY"], session=s)
    wf.poll(a)
    m = dict(series="KXHIGHNY", event="KXHIGHNY-26NOV08", strike_type="greater", floor=60, cap=None)
    v = wf.view(m, a + 6 * 3600)
    assert v["p_yes"] > 0.95                 # observed 61 beats every member's 50-52
    assert wf.alerts["KNYC"] == ["Wind Advisory"]
    assert "Wind Advisory" in wf.status(a)[1]["detail"]


AAA = """<table><tr><td>Current Avg.</td><td>$4.0371</td><td>$4.5663</td></tr>
<tr><td>Yesterday Avg.</td><td>$4.0688</td></tr><tr><td>Week Ago Avg.</td><td>$4.1837</td></tr></table>
<p class="price-text">Price as of 11/7/26 </p>"""


def test_gas_parse_and_view():
    asof, cur, yd, wk = sg.GasPrices.parse(AAA)
    assert asof == date(2026, 11, 7) and cur == 4.0371 and yd == 4.0688 and wk == 4.1837
    g = sg.GasPrices(session=Sess({"aaa": Resp(AAA)}))
    g.poll(0)
    hi = g.view(dict(series="KXAAAGASD", event="KXAAAGASD-26NOV08", strike_type="greater", floor=4.10), 0)
    lo = g.view(dict(series="KXAAAGASD", event="KXAAAGASD-26NOV08", strike_type="greater", floor=3.90), 0)
    assert hi["p_yes"] < 0.01 and lo["p_yes"] > 0.99
    same_day = g.view(dict(series="KXAAAGASD", event="KXAAAGASD-26NOV07", strike_type="greater", floor=4.03), 0)
    assert same_day["locked"] == "yes"


TSA = "".join(f"<tr><td>{d.month}/{d.day}/{d.year}</td><td>{n:,}</td></tr>" for d, n in [
    (date(2026, 11, 5), 2_600_000), (date(2026, 11, 4), 2_100_000), (date(2026, 11, 3), 2_000_000),
    (date(2026, 11, 2), 2_400_000), (date(2026, 11, 1), 2_700_000), (date(2026, 10, 31), 2_000_000),
    (date(2026, 10, 30), 2_600_000), (date(2026, 10, 25), 2_700_000), (date(2026, 10, 24), 2_000_000),
    (date(2026, 10, 23), 2_600_000)])


def test_tsa_week_estimate_and_lock():
    t = sg.TSAThroughput(session=Sess({"tsa.gov": Resp(TSA)}))
    t.poll(0)
    assert t.daily[date(2026, 11, 5)] == 2_600_000
    known, n_known, est, sd = t.week_estimate(date(2026, 11, 8))
    assert n_known == 4 and known == 9_100_000 and est == pytest.approx(2_600_000 + 2_000_000 + 2_700_000)
    v = t.view(dict(series="KXTSAW", event="KXTSAW-26NOV08", strike_type="greater", floor=2_400_000), 0)
    assert v["p_yes"] < 0.1                  # (9.1 + 7.3) / 7 = 2.34M
    lock = t.view(dict(series="KXTSAW", event="KXTSAW-26NOV08", strike_type="greater", floor=1_200_000), 0)
    assert lock["locked"] == "yes"           # 9.1M / 7 > 1.2M already, whatever happens next


def test_equity_watch_trips_fed_series_on_shock():
    trips = []
    q = {"trades": {"SPY": {"p": 700.0}, "QQQ": {"p": 600.0}}}
    s = Sess({"alpaca": Resp(js=q)})
    e = sg.EquityWatch("k", "s", lambda *a: trips.append(a), session=s)
    now = datetime(2026, 11, 9, 10, 0, tzinfo=sg.ET).timestamp()         # a Monday, market open
    e.poll(now)
    q["trades"]["SPY"]["p"] = 690.0                                       # -1.4% in 30s
    e.poll(now + 30)
    assert {a[0] for a in trips} == {"KXFED", "KXFEDDECISION"}
    assert not sg.EquityWatch("", "", lambda *a: None).status(now)[0]["ok"]


def test_failing_signal_records_error_and_retries_soon():
    cal = sg.ReleaseCalendar(lambda *a: None, session=Sess({"bls": RuntimeError("down"),
                                                            "fomc": RuntimeError("down")}))
    cal.poll(1000.0)
    assert cal.error and not cal.status(1000.0)[0]["ok"]
    assert 1000.0 + 300 - cal.every <= cal.last <= 1000.0


RSS = """<?xml version="1.0"?><rss><channel>
<item><title>Federal Reserve issues emergency statement</title><link>https://x/1</link>
<guid>1</guid><pubDate>Sat, 07 Nov 2026 21:00:00 GMT</pubDate></item>
<item><title>Old story</title><link>https://x/2</link></item></channel></rss>"""


def test_rss_parse_and_trip_flag():
    items = RSSSource.parse(RSS)
    assert items[0]["headline"].startswith("Federal Reserve") and items[0]["ts"] > 0
    assert items[1]["ts"] == 0.0                       # no date -> never fresh
    now = items[0]["ts"] + 60
    for trip, expect in ((True, 1), (False, 0)):
        trips = []
        src = RSSSource("fed", "u", trip=trip, session=Sess({"u": Resp(RSS)}))
        NewsHub([src], on_trip=lambda *a: trips.append(a)).poll(now)
        assert len(trips) == expect


def test_bls_latest_numbers_parse():
    xml = """<rss><channel><item><title>x</title><description><![CDATA[<p>Consumer Price Index (CPI): <br>
    <strong><span>+0.6%  in Sep 2026</span></strong><br><a>News Release</a></p>
    <p>Unemployment Rate: <br><strong><span>4.1% in Oct 2026</span></strong><br><a>News Release</a></p>]]>
    </description></item></channel></rss>"""
    lines = [i["headline"] for i in BLSLatestSource.parse(xml)]
    assert lines == ["Consumer Price Index (CPI): +0.6% in Sep 2026", "Unemployment Rate: 4.1% in Oct 2026"]


def test_observed_high_is_a_hard_floor():
    # band 60-61 (YES iff 59.5 <= H < 61.5); already observed 63 -> YES impossible
    assert sg.p_band([55.0], 59.5, 61.5, 2.0, floor=63.0) == 0.0
    # "69 or above" with 70 observed -> certain, whatever the forecast says
    assert sg.p_band([60.0], 68.5, None, 2.0, floor=70.0) == pytest.approx(1.0)


def test_bluesky_keeps_only_market_relevant_posts():
    from kalshi.feeds import BlueskySource
    feed = {"feed": [
        {"post": {"uri": "at://x/app.bsky.feed.post/1", "author": {"handle": "reuters.com"},
                  "record": {"text": "Fed signals rate hike as inflation runs hot", "createdAt": "2026-11-07T22:00:00Z"}}},
        {"post": {"uri": "at://x/app.bsky.feed.post/2", "author": {"handle": "reuters.com"},
                  "record": {"text": "NCAA wins court ruling", "createdAt": "2026-11-07T22:00:00Z"}}},
        {"reason": {"$type": "repost"}, "post": {"uri": "at://x/3", "record": {"text": "repost about CPI"}}}]}
    src = BlueskySource(session=Sess({"bsky": Resp(js=feed)}))
    hub = NewsHub([src], on_trip=lambda *a: None)
    got = hub.poll(datetime(2026, 11, 7, 22, 5, tzinfo=timezone.utc).timestamp())
    assert [g["headline"][:3] for g in got] == ["Fed"]
    assert got[0]["url"] == "https://bsky.app/profile/reuters.com/post/1"


def test_polymarket_maps_fed_outcomes_to_kalshi_tickers():
    d = {"events": [{"title": "Fed Decision in December?", "endDate": "2026-12-10T04:59:00Z", "closed": False,
                     "markets": [{"groupItemTitle": "No change", "bestBid": "0.21", "bestAsk": "0.22"},
                                 {"groupItemTitle": "25 bps increase", "bestBid": "0.76", "bestAsk": "0.77"}]}]}
    p = sg.PolymarketFed(session=Sess({"polymarket": Resp(js=d)}))
    p.poll(0)
    v = p.view(dict(series="KXFEDDECISION", event="KXFEDDECISION-26DEC", ticker="KXFEDDECISION-26DEC-H25"), 0)
    assert v["p_yes"] == pytest.approx(0.765)
    assert p.view(dict(series="KXFEDDECISION", event="KXFEDDECISION-27JAN", ticker="KXFEDDECISION-27JAN-H0"), 0) == {}
