"""Live weather at the exact stations Kalshi settles daily-high markets on.

Kalshi (via The Weather Company) settles on the NWS climate report stations:
CLINYC (Central Park), CLIMDW (Chicago Midway), CLIMIA, CLIAUS, CLILAX, CLIDEN, CLIPHL.
A climate day runs midnight-to-midnight LOCAL STANDARD time.

The observed max so far is a hard lower bound on the day's high. So once it
clears a strike by a safety margin (hourly readings are rounded), one side of
that market is decided:
  greater K:   max >= K + m  -> YES locked, NO dead
  less K:      max >= K + m  -> NO locked, YES dead
  between a-b: max >= b + m  -> NO locked, YES dead
"""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

import requests

# series -> (ICAO station, standard UTC offset hours)
STATIONS = {
    "KXHIGHNY": ("KNYC", -5), "KXHIGHPHIL": ("KPHL", -5), "KXHIGHMIA": ("KMIA", -5),
    "KXHIGHCHI": ("KMDW", -6), "KXHIGHAUS": ("KAUS", -6),
    "KXHIGHDEN": ("KDEN", -7), "KXHIGHLAX": ("KLAX", -8),
    # added 2026-11-07 (settlement stations from each series' rules: CLIDCA, CLIATL, ...)
    "KXHIGHTDC": ("KDCA", -5), "KXHIGHTATL": ("KATL", -5), "KXHIGHTBOS": ("KBOS", -5),
    "KXHIGHTMIN": ("KMSP", -6), "KXHIGHTDAL": ("KDFW", -6), "KXHIGHTHOU": ("KHOU", -6),
    "KXHIGHTOKC": ("KOKC", -6), "KXHIGHTSATX": ("KSAT", -6), "KXHIGHTPHX": ("KPHX", -7),
    "KXHIGHTSFO": ("KSFO", -8), "KXHIGHTLV": ("KLAS", -8), "KXHIGHTSEA": ("KSEA", -8),
    "KXLOWTMIN": ("KMSP", -6), "KXLOWTMIA": ("KMIA", -5), "KXLOWTLAX": ("KLAX", -8),
    # added 2026-11-08: established thinner markets (stations from each series' rules)
    "KXHIGHTNOLA": ("KMSY", -6),
    "KXLOWTNYC": ("KNYC", -5), "KXLOWTPHIL": ("KPHL", -5), "KXLOWTATL": ("KATL", -5), "KXLOWTDC": ("KDCA", -5),
    "KXLOWTBOS": ("KBOS", -5), "KXLOWTCHI": ("KMDW", -6), "KXLOWTDAL": ("KDFW", -6), "KXLOWTSATX": ("KSAT", -6),
    "KXLOWTHOU": ("KHOU", -6), "KXLOWTNOLA": ("KMSY", -6), "KXLOWTAUS": ("KAUS", -6), "KXLOWTOKC": ("KOKC", -6),
    "KXLOWTPHX": ("KPHX", -7), "KXLOWTDEN": ("KDEN", -7), "KXLOWTLV": ("KLAS", -8), "KXLOWTSEA": ("KSEA", -8),
    "KXLOWTSFO": ("KSFO", -8),
}
LOW_SERIES = {s for s in STATIONS if s.startswith("KXLOW")}
UA = {"User-Agent": "kalshi-weather-research-bot", "Accept": "application/geo+json"}
_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def f_from_c(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def market_date(event_ticker: str):
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})$", event_ticker or "")
    if not m or m.group(2) not in _MONTHS:
        return None
    return (2000 + int(m.group(1)), _MONTHS[m.group(2)], int(m.group(3)))


def climate_day(ymd, std_offset_h: int):
    start = datetime(*ymd, tzinfo=timezone.utc).timestamp() - std_offset_h * 3600
    return start, start + 86400


def settle_view(strike_type: str, floor: Optional[float], cap: Optional[float],
                max_so_far: Optional[float], margin: float = 1.0) -> dict:
    if max_so_far is None:
        return {}
    st = strike_type or ""
    if st in ("greater", "greater_or_equal") and floor is not None:
        if max_so_far >= floor + margin:
            return {"locked": "yes", "dead": "no"}
    elif st in ("less", "less_or_equal"):
        k = cap if cap is not None else floor
        if k is not None and max_so_far >= k + margin:
            return {"locked": "no", "dead": "yes"}
    elif st == "between" and cap is not None:
        if max_so_far >= cap + margin:
            return {"locked": "no", "dead": "yes"}
    return {}


def settle_view_low(strike_type: str, floor: Optional[float], cap: Optional[float],
                    min_so_far: Optional[float]) -> dict:
    """Daily LOW markets: the lowest reading so far is an upper bound on the day's low
    (hourly readings can only miss colder minutes), so once it reaches a strike one side is decided.
      greater K (YES iff low >= K+1): min <= K       -> YES dead
      less C    (YES iff low <= C-1): min <= C-1     -> YES locked
      between a-b:                    min <= a-1     -> YES dead"""
    if min_so_far is None:
        return {}
    st = strike_type or ""
    if st in ("greater", "greater_or_equal") and floor is not None and min_so_far <= floor:
        return {"locked": "no", "dead": "yes"}
    if st in ("less", "less_or_equal"):
        k = cap if cap is not None else floor
        if k is not None and min_so_far <= k - 1:
            return {"locked": "yes", "dead": "no"}
    if st == "between" and floor is not None and min_so_far <= floor - 1:
        return {"locked": "no", "dead": "yes"}
    return {}


class WeatherObs:
    name = "nws-obs"
    URL = "https://api.weather.gov/stations/{st}/observations"

    def __init__(self, session=None, min_interval_s: float = 300):
        self.s = session or requests.Session()
        self.min_interval = min_interval_s
        self.last_poll: dict = {}
        self.obs = defaultdict(dict)          # station -> {ts: tempF}
        self.errors: dict = {}

    def poll(self, now: float, series) -> int:
        n = 0
        for ser in series:
            if ser not in STATIONS:
                continue
            st = STATIONS[ser][0]
            if self.last_poll.get(st) == now:
                continue
            if now - self.last_poll.get(st, -1e18) < self.min_interval:
                continue
            self.last_poll[st] = now
            try:
                r = self.s.get(self.URL.format(st=st), headers=UA, params={"limit": 50}, timeout=10)
                r.raise_for_status()
                for f in r.json().get("features", []):
                    p = f.get("properties", {})
                    c = (p.get("temperature") or {}).get("value")
                    if c is None:
                        continue
                    ts = datetime.fromisoformat(p["timestamp"]).timestamp()
                    self.obs[st][ts] = f_from_c(float(c))
                    n += 1
                self.errors.pop(st, None)
            except Exception as e:
                self.errors[st] = str(e)[:160]
                self.last_poll[st] = now - self.min_interval + 60   # retry in a minute
        return n

    def max_f(self, station: str, a: float, b: float) -> Optional[float]:
        vals = [v for t, v in self.obs.get(station, {}).items() if a <= t < b]
        return max(vals) if vals else None

    def min_f(self, station: str, a: float, b: float) -> Optional[float]:
        vals = [v for t, v in self.obs.get(station, {}).items() if a <= t < b]
        return min(vals) if vals else None

    def last_obs_age(self, station: str, now: float) -> float:
        ts = self.obs.get(station)
        return now - max(ts) if ts else float("inf")

    def latest(self, station: str):
        ts = self.obs.get(station)
        if not ts:
            return None
        t = max(ts)
        return t, ts[t]
