"""Free data signals for the markets the bot trades.

Each signal has
  poll(now)        fetch on its own schedule (errors are kept, never raised)
  status(now)      rows for the dashboard's Data feeds card
  view(m, now)     opinion on one market (m = the bot's market dict):
                     {}                                   no opinion
                     {"p_yes": p, "why": ...}             probability -> the bot may only VETO a
                                                          favorite bid it disagrees with
                     {"locked": s, "dead": s, "why": ...} decided by published data

None of these was in the backtest, so they never ADD trades on a probability. The bot logs
every disagreement ("would veto"); a signal only blocks bids once it is listed in
bot.veto_enforce. Published numbers that decide a market (locks) are always used.

What the stored fills say (python3 -m kalshi wxcheck --pre-day, and the release checks):
  - Weather: before the climate day starts, fills the forecast disagrees with earned the same
    as the rest (+3..+6%/position at every gap/smear tried) -> the market already prices the
    forecast; a veto would just delete edge. So forecasts are log-only.
  - Econ: fills in the last 15 min before their own release earned +12%/position (the best
    window), so the calendar NEVER pauses a series ahead of its own release. It only pauses
    the Fed markets for a while after a CPI/jobs/GDP print reprices them.

  ReleaseCalendar   BLS schedule (CPI, jobs), FOMC dates, Kalshi econ close times -> pause
  Nowcasts          Cleveland Fed CPI m/m + y/y nowcast, Atlanta Fed GDPNow (via FRED)
  WeatherForecast   Open-Meteo GEFS+ECMWF ensembles (82 members), NWS hourly forecast, NWS alerts
  GasPrices         AAA national average (the settlement source for KXAAAGASD)
  TSAThroughput     TSA checkpoint counts (the settlement source for KXTSAW)
  EquityWatch       SPY/QQQ via Alpaca -> pause Fed markets on a sudden 1% move
"""
from __future__ import annotations

import math
import re
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

from . import weather as wx
from .feeds import SpotWatch

ET = ZoneInfo("America/New_York")
UA = {"User-Agent": "kalshi-research-bot/1.0"}
_MON = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july",
                                     "august", "september", "october", "november", "december"], 1)}


def ncdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def p_at_least(mu: float, sigma: float, t: float) -> float:
    """P(X >= t) for X ~ N(mu, sigma)."""
    if sigma <= 0:
        return 1.0 if mu >= t else 0.0
    return 1.0 - ncdf((t - mu) / sigma)


def event_month(event: str):
    m = re.search(r"-(\d{2})([A-Z]{3})$", event or "")
    if not m or m.group(2) not in wx._MONTHS:
        return None
    return 2000 + int(m.group(1)), wx._MONTHS[m.group(2)]


def _ago(ts, now):
    return None if ts is None else now - ts


class _Polled:
    name = "signal"
    every = 3600.0

    def __init__(self, session=None, every=None):
        self.s = session or requests.Session()
        if every is not None:
            self.every = every
        self.last = -1e18
        self.ok_ts = None
        self.error = None

    def poll(self, now: float):
        if now - self.last < self.every:
            return
        self.last = now
        try:
            self.fetch(now)
            self.ok_ts, self.error = now, None
        except Exception as e:                       # a dead feed never stops the bot
            self.error = str(e)[:160]
            self.last = now - self.every + min(300.0, self.every)     # retry in <= 5 min

    def fetch(self, now: float):
        raise NotImplementedError

    def _get(self, url, **kw):
        r = self.s.get(url, headers={**UA, **kw.pop("headers", {})}, timeout=20, **kw)
        r.raise_for_status()
        return r

    def view(self, m: dict, now: float) -> dict:
        return {}


# --------------------------------------------------------------------------- release calendar
# Which OTHER series a release reprices. A market's own release is not here: it closes a
# minute before the print, and its last-15-minute fills were the most profitable in the data.
RELEASE_SERIES = {
    "cpi": ["KXFED", "KXFEDDECISION"],
    "jobs": ["KXFED", "KXFEDDECISION"],
    "gdp": ["KXFED", "KXFEDDECISION"],
    "claims": [],
    "fomc": [],
}
SERIES_KIND = {"KXCPI": "cpi", "KXCPIYOY": "cpi", "KXPAYROLLS": "jobs", "KXU3": "jobs", "KXGDP": "gdp",
               "KXJOBLESSCLAIMS": "claims", "KXFED": "fomc", "KXFEDDECISION": "fomc"}
BLS_KIND = {"Consumer Price Index": "cpi", "Employment Situation": "jobs"}
KIND_LABEL = {"cpi": "CPI", "jobs": "Jobs report", "gdp": "GDP", "claims": "Jobless claims", "fomc": "FOMC"}


def parse_bls_ics(text: str) -> list:
    out = []
    for block in text.split("BEGIN:VEVENT")[1:]:
        summ = re.search(r"^SUMMARY:(.+?)\r?$", block, re.M)
        dt = re.search(r"^DTSTART;TZID=[^:]+:(\d{8}T\d{6})\r?$", block, re.M)
        if not summ or not dt or summ.group(1).strip() not in BLS_KIND:
            continue
        t = datetime.strptime(dt.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=ET).timestamp()
        out.append((BLS_KIND[summ.group(1).strip()], t))
    return out


def parse_fomc(html: str) -> list:
    """Statement days (2:00 pm ET on the meeting's last day) from the Fed's calendar page."""
    out = []
    for y in re.findall(r"(\d{4}) FOMC Meetings", html):
        i = html.find(f"{y} FOMC Meetings")
        j = html.find("FOMC Meetings", i + 20)
        txt = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html[i:j if j > 0 else None]))
        for mon, _d1, d2 in re.findall(r"\b([A-Z][a-z]+(?:/[A-Z][a-z]+)?)\s+(\d{1,2})-(\d{1,2})\b", txt):
            name = mon.split("/")[-1].lower()
            mo = next((v for k, v in _MON.items() if k.startswith(name[:3])), None)
            if mo is None:
                continue
            try:
                out.append(("fomc", datetime(int(y), mo, int(d2), 14, 0, tzinfo=ET).timestamp()))
            except ValueError:
                pass
    return out


class ReleaseCalendar(_Polled):
    """Tracks every release; pauses the series a print reprices from `before_s` ahead of it
    (default 0) to `after_s` after it."""
    name = "calendar"
    every = 6 * 3600.0
    BLS = "https://www.bls.gov/schedule/news_release/bls.ics"
    FOMC = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

    def __init__(self, on_trip, before_s: float = 0, after_s: float = 1800, session=None):
        super().__init__(session)
        self.on_trip = on_trip
        self.before, self.after = before_s, after_s
        self.events: dict = {}                       # (kind, ts) -> source

    def add(self, kind: str, ts: float, source: str):
        if not any(k == kind and abs(t - ts) < 3 * 3600 for (k, t) in self.events):
            self.events[(kind, ts)] = source

    def fetch(self, now: float):
        errs = []
        for url, parse, src in ((self.BLS, parse_bls_ics, "BLS"), (self.FOMC, parse_fomc, "Fed")):
            try:
                for kind, t in parse(self._get(url).text):
                    self.add(kind, t, src)
            except Exception as e:
                errs.append(f"{src}: {str(e)[:60]}")
        if errs and not self.events:
            raise RuntimeError("; ".join(errs))

    def note_markets(self, markets: dict):
        for m in markets.values():
            kind = SERIES_KIND.get(m.get("series"))
            if kind:
                self.add(kind, m["close_ts"] + 60, "Kalshi")

    def poll(self, now: float):
        super().poll(now)
        for (kind, t) in self.events:
            if t - self.before <= now <= t + self.after:
                why = f"{KIND_LABEL[kind]} release {datetime.fromtimestamp(t, ET):%a %-I:%M %p} ET"
                for ser in RELEASE_SERIES[kind]:
                    self.on_trip(ser, why, now, t + self.after - now)

    def upcoming(self, now: float, n: int = 3) -> list:
        return sorted((t, k) for (k, t) in self.events if t + self.after >= now)[:n]

    def status(self, now: float) -> list:
        nxt = self.upcoming(now, 1)
        detail = (f"next {KIND_LABEL[nxt[0][1]]} {datetime.fromtimestamp(nxt[0][0], ET):%b %-d %-I:%M %p}"
                  if nxt else (self.error or "waiting"))
        return [dict(name="Economic calendar", ok=bool(self.events) and not self.error, detail=detail,
                     age=_ago(self.ok_ts, now))]


# --------------------------------------------------------------------------- nowcasts
def _clev_latest(charts: list) -> dict:
    """{(year, month or quarter-start-month): latest 'CPI Inflation' nowcast} for unreleased periods."""
    out = {}
    for ch in charts:
        sub = (ch.get("chart") or {}).get("subcaption", "")
        m = re.match(r"(\d{4})-(\d{1,2})$", sub)
        if not m:
            continue
        ds = {d.get("seriesname"): [p.get("value") for p in d.get("data", []) if p.get("value") not in ("", None)]
              for d in ch.get("dataset", [])}
        if ds.get("Actual CPI Inflation"):
            continue                                  # already released
        vals = ds.get("CPI Inflation") or []
        if vals:
            out[(int(m.group(1)), int(m.group(2)))] = float(vals[-1])
    return out


def gdp_quarter(release_ymd):
    """Advance GDP released in month M covers the quarter that ended before M."""
    y, mo, _d = release_ymd
    pm, py = mo - 1, y
    if pm == 0:
        pm, py = 12, y - 1
    return py, ((pm - 1) // 3) * 3 + 1


class Nowcasts(_Polled):
    name = "nowcasts"
    every = 3 * 3600.0
    CLEV = "https://www.clevelandfed.org/-/media/files/webcharts/inflationnowcasting/nowcast_{}.json"
    FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GDPNOW"
    # Typical nowcast error (pct points) near release; deliberately on the wide side.
    SIGMA = {"KXCPI": 0.12, "KXCPIYOY": 0.10, "KXGDP": 1.2}

    def __init__(self, session=None):
        super().__init__(session)
        self.cpi_mom, self.cpi_yoy, self.gdp = {}, {}, {}

    def fetch(self, now: float):
        self.cpi_mom = _clev_latest(self._get(self.CLEV.format("month")).json())
        self.cpi_yoy = _clev_latest(self._get(self.CLEV.format("year")).json())
        gdp = {}
        for line in self._get(self.FRED).text.splitlines()[1:]:
            d, _, v = line.partition(",")
            try:
                gdp[(int(d[:4]), int(d[5:7]))] = float(v)
            except ValueError:
                continue
        self.gdp = gdp

    def estimate(self, m: dict):
        ser = m.get("series")
        if ser in ("KXCPI", "KXCPIYOY"):
            ym = event_month(m.get("event", ""))
            mu = (self.cpi_mom if ser == "KXCPI" else self.cpi_yoy).get(ym)
            return mu, "Cleveland Fed nowcast"
        if ser == "KXGDP":
            ymd = wx.market_date(m.get("event", ""))
            return (self.gdp.get(gdp_quarter(ymd)) if ymd else None), "GDPNow"
        return None, ""

    def view(self, m: dict, now: float) -> dict:
        mu, src = self.estimate(m)
        k = m.get("floor")
        if mu is None or k is None or m.get("strike_type") not in ("greater", "greater_or_equal"):
            return {}
        # BLS/BEA report one decimal: YES iff the rounded value > k, i.e. x >= k + 0.05
        p = p_at_least(mu, self.SIGMA[m["series"]], k + 0.05)
        return {"p_yes": p, "why": f"{src} {mu:.2f}%"}

    def status(self, now: float) -> list:
        def latest(d):
            return d[max(d)] if d else None
        mom, yoy, g = latest(self.cpi_mom), latest(self.cpi_yoy), latest(self.gdp)
        ok = not self.error and mom is not None
        return [
            dict(name="Cleveland Fed CPI nowcast", ok=ok,
                 detail=(f"{mom:.2f}% m/m · {yoy:.2f}% y/y" if mom is not None and yoy is not None
                         else self.error or "waiting"), age=_ago(self.ok_ts, now)),
            dict(name="Atlanta Fed GDPNow", ok=not self.error and g is not None,
                 detail=f"{g:.1f}%" if g is not None else (self.error or "waiting"), age=_ago(self.ok_ts, now)),
        ]


# --------------------------------------------------------------------------- weather forecasts
COORDS = {"KNYC": (40.7789, -73.9692), "KMDW": (41.7861, -87.7522), "KMIA": (25.7881, -80.3169),
          "KAUS": (30.1945, -97.6699), "KLAX": (33.9382, -118.3866), "KDEN": (39.8466, -104.6562),
          "KPHL": (39.8733, -75.2268), "KDCA": (38.8483, -77.0342), "KATL": (33.6301, -84.4418),
          "KBOS": (42.3606, -71.0106), "KMSP": (44.8848, -93.2223), "KDFW": (32.8998, -97.0403),
          "KHOU": (29.6454, -95.2789), "KOKC": (35.3931, -97.6007), "KSAT": (29.5337, -98.4698),
          "KPHX": (33.4278, -112.0037), "KSFO": (37.6196, -122.3656), "KLAS": (36.0719, -115.1634),
          "KSEA": (47.4447, -122.3144), "KMSY": (29.9934, -90.2580)}


def wx_yes_threshold(strike_type, floor, cap):
    """Kalshi reports whole degrees, so for a continuous high v:
    greater K -> YES iff v >= K+0.5 ; less C -> YES iff v < C-0.5 ; between a-b -> a-0.5 <= v < b+0.5"""
    if strike_type in ("greater", "greater_or_equal") and floor is not None:
        return floor + 0.5, None
    if strike_type in ("less", "less_or_equal") and cap is not None:
        return None, cap - 0.5
    if strike_type == "between" and floor is not None and cap is not None:
        return floor - 0.5, cap + 0.5
    return None


def p_band(values, lo, hi, smear: float, floor=None) -> float:
    """Mean over forecasts v of P(lo <= H < hi), H = max(floor, v + e), e ~ N(0, smear).
    `floor` is the high already observed today: a hard lower bound, not uncertain."""
    if not values:
        return float("nan")

    def below(x, v):                       # P(H < x)
        if x is None:
            return 1.0
        if floor is not None and floor >= x:
            return 0.0
        return 1.0 - p_at_least(v, smear, x)
    tot = 0.0
    for v in values:
        tot += max(0.0, below(hi, v) - (below(lo, v) if lo is not None else 0.0))
    return tot / len(values)


class WeatherForecast(_Polled):
    """Polls one station per call (stations rotate), so a slow API never stalls the loop."""
    name = "forecast"
    every = 3600.0
    ENS = "https://ensemble-api.open-meteo.com/v1/ensemble"
    BIAS = 0.5           # hourly samples under-read the true daily max
    SMEAR = 2.0          # station vs grid-point error, deg F

    def __init__(self, obs=None, series=(), session=None):
        super().__init__(session)
        self.obs = obs
        self.stations = sorted({wx.STATIONS[s][0] for s in series if s in wx.STATIONS and s not in wx.LOW_SERIES
                                and wx.STATIONS[s][0] in COORDS})
        self.members: dict = {}      # station -> (times, [member temps])
        self.nws: dict = {}          # station -> [(ts, tempF)]
        self.alerts: dict = {}       # station -> [event names]
        self.hourly_url: dict = {}
        self.st_last: dict = {}
        self.st_err: dict = {}
        self.st_ok: dict = {}

    def poll(self, now: float):
        due = [st for st in self.stations if now - self.st_last.get(st, -1e18) >= self.every]
        if not due:
            return
        st = due[0]
        self.st_last[st] = now
        try:
            self.fetch_station(st)
            self.st_ok[st] = now
            self.st_err.pop(st, None)
        except Exception as e:
            self.st_err[st] = str(e)[:120]
            self.st_last[st] = now - self.every + 300

    def fetch_station(self, st: str):
        lat, lon = COORDS[st]
        d = self._get(self.ENS, params=dict(latitude=lat, longitude=lon, hourly="temperature_2m",
                                            models="gfs025,ecmwf_ifs025", temperature_unit="fahrenheit",
                                            timezone="GMT", forecast_days=3, timeformat="unixtime")).json()
        h = d.get("hourly") or {}
        times = h.get("time") or []
        mem = [v for k, v in h.items() if k.startswith("temperature_2m") and isinstance(v, list)]
        if not times or not mem:
            raise RuntimeError(d.get("reason") or "no ensemble data")
        self.members[st] = (times, mem)
        geo = {"Accept": "application/geo+json"}
        if st not in self.hourly_url:
            p = self._get(f"https://api.weather.gov/points/{lat},{lon}", headers=geo).json()["properties"]
            self.hourly_url[st] = p["forecastHourly"]
        per = self._get(self.hourly_url[st], headers=geo).json()["properties"]["periods"]
        self.nws[st] = [(datetime.fromisoformat(x["startTime"]).timestamp(),
                         float(x["temperature"]) if x.get("temperatureUnit") == "F" else wx.f_from_c(x["temperature"]))
                        for x in per]
        al = self._get("https://api.weather.gov/alerts/active", headers=geo,
                       params={"point": f"{lat},{lon}"}).json().get("features", [])
        self.alerts[st] = sorted({(f.get("properties") or {}).get("event", "") for f in al} - {""})

    def day_max(self, st: str, a: float, b: float, now: float):
        """Per-member forecast of the climate day's high, floored by what has been observed."""
        obs_max = self.obs.max_f(st, a, b) if self.obs is not None else None
        ens = []
        if st in self.members:
            times, mem = self.members[st]
            idx = [i for i, t in enumerate(times) if a <= t < b and t >= now - 1800]
            if idx:
                for arr in mem:
                    vals = [arr[i] for i in idx if arr[i] is not None]
                    if vals:
                        ens.append(max(vals) + self.BIAS)
        nws = [v for t, v in self.nws.get(st, []) if a <= t < b and t >= now - 1800]
        return ens, (max(nws) if nws else None), obs_max

    def view(self, m: dict, now: float) -> dict:
        ser = m.get("series")
        if ser not in wx.STATIONS or ser in wx.LOW_SERIES:
            return {}
        st, off = wx.STATIONS[ser]
        ymd = wx.market_date(m.get("event", ""))
        th = wx_yes_threshold(m.get("strike_type"), m.get("floor"), m.get("cap"))
        if not ymd or th is None:
            return {}
        a, b = wx.climate_day(ymd, off)
        if now >= b:
            return {}
        ens, nws_max, obs_max = self.day_max(st, a, b, now)
        lo, hi = th
        ps, parts = [], []
        if len(ens) >= 20:
            ps.append(p_band(ens, lo, hi, self.SMEAR, floor=obs_max))
            parts.append(f"{len(ens)} ensemble members")
        if nws_max is not None:
            ps.append(p_band([nws_max], lo, hi, 2.5, floor=obs_max))
            parts.append(f"NWS high {max(nws_max, obs_max or -1e9):.0f}°F")
        if obs_max is not None:
            parts.append(f"so far {obs_max:.0f}°F")
        if not ps:
            return {}
        return {"p_yes": sum(ps) / len(ps), "why": ", ".join(parts)}

    def status(self, now: float) -> list:
        n = len(self.members)
        mem = max((len(v[1]) for v in self.members.values()), default=0)
        last = max(self.st_ok.values(), default=None)
        rows = [dict(name="Weather forecasts (GEFS + ECMWF + NWS)", ok=n > 0 and not self.st_err,
                     detail=(f"{n}/{len(self.stations)} cities · {mem} members" if n else
                             next(iter(self.st_err.values()), "waiting")) +
                            (f" · {len(self.st_err)} failing" if n and self.st_err else ""),
                     age=_ago(last, now))]
        al = [f"{st} {', '.join(v)}" for st, v in self.alerts.items() if v]
        rows.append(dict(name="NWS alerts", ok=bool(self.alerts), detail="; ".join(al) if al else
                         ("none active" if self.alerts else "waiting"), age=_ago(last, now)))
        return rows


# --------------------------------------------------------------------------- gas
class GasPrices(_Polled):
    name = "gas"
    every = 1800.0
    URL = "https://gasprices.aaa.com/"

    def __init__(self, session=None):
        super().__init__(session)
        self.asof = None             # date
        self.cur = self.yday = self.week = None

    @staticmethod
    def parse(html: str):
        d = re.search(r"Price as of (\d{1,2})/(\d{1,2})/(\d{2,4})", html)
        txt = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))
        grab = lambda lab: re.search(lab + r"\s+\$(\d+\.\d+)", txt)
        cur, yd, wk = grab(r"Current Avg\."), grab(r"Yesterday Avg\."), grab(r"Week Ago Avg\.")
        if not d or not cur:
            raise RuntimeError("AAA page layout changed")
        y = int(d.group(3))
        asof = date(y + 2000 if y < 100 else y, int(d.group(1)), int(d.group(2)))
        return asof, float(cur.group(1)), float(yd.group(1)) if yd else None, float(wk.group(1)) if wk else None

    def fetch(self, now: float):
        self.asof, self.cur, self.yday, self.week = self.parse(self._get(self.URL).text)

    def view(self, m: dict, now: float) -> dict:
        if m.get("series") != "KXAAAGASD" or self.cur is None or m.get("floor") is None:
            return {}
        ymd = wx.market_date(m.get("event", ""))
        if not ymd or m.get("strike_type") not in ("greater", "greater_or_equal"):
            return {}
        days = (date(*ymd) - self.asof).days
        if days < 0:
            return {}
        if days == 0:                                # AAA has already published the settling number
            yes = self.cur > m["floor"] + 1e-9
            return {"locked": "yes" if yes else "no", "dead": "no" if yes else "yes",
                    "why": f"AAA {self.cur:.4f} published"}
        drift = (self.cur - self.week) / 7.0 if self.week else 0.0
        sigma = (0.01 + 0.5 * abs(drift)) * math.sqrt(days)
        mu = self.cur + drift * days
        p = 1.0 - ncdf((m["floor"] - mu) / sigma)    # strictly greater than the strike
        return {"p_yes": p, "why": f"AAA {self.cur:.3f}, trend {drift * 100:+.1f}c/day"}

    def status(self, now: float) -> list:
        return [dict(name="AAA gas price", ok=self.cur is not None and not self.error,
                     detail=(f"${self.cur:.3f} ({self.asof:%b %-d})" if self.cur is not None else self.error or "waiting"),
                     age=_ago(self.ok_ts, now))]


# --------------------------------------------------------------------------- TSA
def _holiday_week(d: date) -> bool:
    return (d.month == 11 and d.day >= 20) or (d.month == 12 and d.day >= 18) or (d.month == 1 and d.day <= 4) \
        or (d.month == 7 and 1 <= d.day <= 8)


class TSAThroughput(_Polled):
    name = "tsa"
    every = 3600.0
    URL = "https://www.tsa.gov/travel/passenger-volumes"

    def __init__(self, session=None):
        super().__init__(session)
        self.daily: dict = {}

    @staticmethod
    def parse(html: str) -> dict:
        out = {}
        for mo, dd, yy, num in re.findall(r"(\d{1,2})/(\d{1,2})/(\d{4})\s*</td>\s*<td[^>]*>\s*([\d,]+)", html):
            out[date(int(yy), int(mo), int(dd))] = int(num.replace(",", ""))
        return out

    def fetch(self, now: float):
        d = self.parse(self._get(self.URL).text)
        if not d:
            raise RuntimeError("TSA table not found")
        self.daily = d

    def week_estimate(self, sunday: date):
        known, est, var = 0.0, 0.0, 0.0
        n_known = 0
        for i in range(6, -1, -1):
            d = sunday - timedelta(days=i)
            if d in self.daily:
                known += self.daily[d]
                n_known += 1
                continue
            prev = [self.daily[d - timedelta(days=7 * k)] for k in (1, 2) if d - timedelta(days=7 * k) in self.daily]
            if not prev:
                return None
            e = sum(prev) / len(prev)
            est += e
            var += (e * (0.15 if _holiday_week(d) else 0.06)) ** 2
        return known, n_known, est, math.sqrt(var)

    def view(self, m: dict, now: float) -> dict:
        if m.get("series") != "KXTSAW" or m.get("floor") is None or m.get("strike_type") != "greater":
            return {}
        ymd = wx.market_date(m.get("event", ""))
        w = self.week_estimate(date(*ymd)) if ymd else None
        if w is None:
            return {}
        known, n_known, est, sd = w
        k = m["floor"]
        if known / 7.0 > k or (n_known == 7 and known / 7.0 <= k):
            yes = known / 7.0 > k                    # counts can't go negative: this is decided
            return {"locked": "yes" if yes else "no", "dead": "no" if yes else "yes",
                    "why": f"TSA {n_known} of 7 days published"}
        mean = (known + est) / 7.0
        p = 1.0 - ncdf((k - mean) / (sd / 7.0)) if sd > 0 else float(mean > k)
        return {"p_yes": p, "why": f"TSA {n_known}/7 days known, est {mean / 1e6:.2f}M/day"}

    def status(self, now: float) -> list:
        last = max(self.daily) if self.daily else None
        return [dict(name="TSA checkpoint counts", ok=last is not None and not self.error,
                     detail=(f"{self.daily[last] / 1e6:.2f}M on {last:%b %-d}" if last else self.error or "waiting"),
                     age=_ago(self.ok_ts, now))]


# --------------------------------------------------------------------------- equities
FED_SERIES = ["KXFED", "KXFEDDECISION"]


def us_market_open(now: float) -> bool:
    t = datetime.fromtimestamp(now, ET)
    return t.weekday() < 5 and (9 * 60 + 30) <= t.hour * 60 + t.minute < 16 * 60


class EquityWatch(_Polled):
    """A 1% SPY/QQQ move inside 5 minutes is a macro shock: pause the Fed markets."""
    name = "equities"
    every = 30.0
    URL = "https://data.alpaca.markets/v2/stocks/trades/latest"

    def __init__(self, key_id: str, secret: str, on_trip, symbols=("SPY", "QQQ"), pct=0.01, window_s=300,
                 series=FED_SERIES, session=None):
        super().__init__(session)
        self.h = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret}
        self.enabled = bool(key_id and secret)

        def trip(_cat, why, now):
            for ser in series:
                on_trip(ser, why, now, 600)
        self.watch = {s: SpotWatch(s, "Financials", pct, window_s, on_trip=trip) for s in symbols}

    def fetch(self, now: float):
        if not self.enabled or (not us_market_open(now) and any(w.last_px for w in self.watch.values())):
            return
        d = self._get(self.URL, headers=self.h, params={"symbols": ",".join(self.watch), "feed": "iex"}).json()
        for sym, t in (d.get("trades") or {}).items():
            if sym in self.watch and t.get("p"):
                self.watch[sym].update(float(t["p"]), now)

    def status(self, now: float) -> list:
        if not self.enabled:
            return [dict(name="SPY / QQQ (Alpaca)", ok=False, detail="no Alpaca key", age=None)]
        px = " · ".join(f"{s} ${w.last_px:.2f}" for s, w in self.watch.items() if w.last_px)
        return [dict(name="SPY / QQQ (Alpaca)", ok=not self.error and bool(px),
                     detail=(px or self.error or "waiting") + ("" if us_market_open(now) else " · market closed"),
                     age=_ago(self.ok_ts, now))]


# --------------------------------------------------------------------------- Polymarket
POLY_OUTCOME = {"H0": "No change", "H25": "25 bps increase", "H26": "50+ bps increase",
                "C25": "25 bps decrease", "C26": "50+ bps decrease"}


class PolymarketFed(_Polled):
    """The same Fed decisions trade on Polymarket. A second, independent crowd: when it
    disagrees with Kalshi by 20c+ the bot logs it (never trades on it)."""
    name = "polymarket"
    every = 300.0
    URL = "https://gamma-api.polymarket.com/public-search"

    def __init__(self, session=None):
        super().__init__(session)
        self.odds: dict = {}          # (year, month) -> {outcome label: mid}

    @staticmethod
    def parse(d: dict) -> dict:
        out = {}
        for e in d.get("events", []):
            if e.get("closed") or not re.match(r"Fed Decision in \w+\?", e.get("title", "")):
                continue
            try:
                end = datetime.fromisoformat(e["endDate"].replace("Z", "+00:00")).astimezone(ET)
            except Exception:
                continue
            od = {}
            for m in e.get("markets", []):
                try:
                    bid, ask = float(m.get("bestBid") or 0), float(m.get("bestAsk") or 0)
                except ValueError:
                    continue
                if 0 < bid <= ask < 1:
                    od[m.get("groupItemTitle", "")] = (bid + ask) / 2
            if od:
                # endDate is the evening after the meeting (UTC), sometimes the next day local
                out[(end.year, end.month)] = od
        return out

    def fetch(self, now: float):
        self.odds = self.parse(self._get(self.URL, params={"q": "fed decision", "limit_per_type": 20}).json())

    def view(self, m: dict, now: float) -> dict:
        if m.get("series") != "KXFEDDECISION":
            return {}
        ym = event_month(m.get("event", ""))
        suffix = (m.get("ticker", "").rsplit("-", 1) + [""])[1]
        label = POLY_OUTCOME.get(suffix)
        p = (self.odds.get(ym) or {}).get(label) if ym and label else None
        if p is None:
            return {}
        return {"p_yes": p, "why": f"Polymarket {label} {p * 100:.0f}%"}

    def status(self, now: float) -> list:
        if not self.odds:
            return [dict(name="Polymarket Fed odds", ok=False, detail=self.error or "waiting", age=_ago(self.ok_ts, now))]
        ym = min(self.odds)
        top = max(self.odds[ym].items(), key=lambda kv: kv[1])
        return [dict(name="Polymarket Fed odds", ok=not self.error,
                     detail=f"{date(ym[0], ym[1], 1):%b}: {top[0]} {top[1] * 100:.0f}%", age=_ago(self.ok_ts, now))]
