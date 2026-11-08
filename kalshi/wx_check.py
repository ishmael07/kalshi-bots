"""Would a forecast veto have helped? Tests it on the stored weather fills.

For every favorite-side maker fill in the last 24h before close, take the forecast for
that climate day made ~1 day earlier (Open-Meteo previous-runs API: GFS + ECMWF, deterministic),
turn it into P(our side wins) with the same rounding + smear as the live signal, and
split the fills into "would veto" (p_side < price - gap) vs "keep".

The veto helps only if vetoed fills LOSE money; if they win as often as their price says,
the market already knew the forecast and the veto just deletes edge.
python3 -m kalshi wxcheck [--gap 0.20] [--smear 3.0]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import requests

from . import weather as wx
from .fees import fee_for
from .signals import COORDS, p_band, wx_yes_threshold
from .store import Store

PREV = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODELS = ("gfs_seamless", "ecmwf_ifs025")


def fetch_prev(station: str, start: str, end: str, cache: Path) -> dict:
    f = cache / f"prev_{station}_{start}_{end}.json"
    if f.exists():
        return json.loads(f.read_text())
    lat, lon = COORDS[station]
    out = {}
    for model in MODELS:
        r = requests.get(PREV, params=dict(latitude=lat, longitude=lon, hourly="temperature_2m_previous_day1",
                                           models=model, start_date=start, end_date=end,
                                           temperature_unit="fahrenheit", timezone="GMT", timeformat="unixtime"),
                         timeout=60)
        r.raise_for_status()
        h = r.json()["hourly"]
        out[model] = dict(zip(h["time"], h["temperature_2m_previous_day1"]))
    f.write_text(json.dumps(out))
    return out


def day_max(series_hours: dict, a: float, b: float):
    vals = [v for t, v in series_hours.items() if a <= float(t) < b and v is not None]
    return max(vals) if len(vals) >= 20 else None


def run(db="data/kalshi.db", gap=0.20, smear=3.0, bias=0.5, start="2025-05-01", end="2026-11-06",
        model_path="data/cell_model.json", margin=0.03, pre_day=False):
    """pre_day: only fills placed before the climate day starts (no observations yet), which
    isolates what the forecast adds; later fills are dominated by what the station already saw."""
    from .live import CellModel, cell_key
    model = CellModel.load(model_path) if model_path else None
    st = Store(db)
    cache = Path("data/wx_prev")
    cache.mkdir(parents=True, exist_ok=True)
    prev = {s: fetch_prev(wx.STATIONS[s][0], start, end, cache) for s in wx.STATIONS}
    for s in prev:                     # json keys come back as strings
        prev[s] = {m: {float(t): v for t, v in d.items()} for m, d in prev[s].items()}
    q = ("SELECT m.ticker, m.event, m.series, m.result, m.close_ts, m.strike_type, m.floor_strike, m.cap_strike, "
         "t.ts, t.yes_price, t.count, t.taker_yes, s.fee_type, s.fee_mult FROM trades t JOIN markets m ON m.id=t.mid "
         "JOIN series s ON s.ticker=m.series WHERE m.series IN (%s) AND m.result IN ('yes','no') "
         "AND t.ts >= m.close_ts - 86400" % ",".join("?" * len(wx.STATIONS)))
    pos = defaultdict(lambda: [0.0, 0.0, 0.0, None, None, None])   # (ticker, side) -> [ret*cnt, cnt, price*cnt, p, veto, event]
    pcache = {}
    for (tk, ev, ser, res, close_ts, stype, fl, cp, ts, yp, cnt, ty, fee_type, fee_mult) in st.db.execute(q, list(wx.STATIONS)):
        side = "no" if ty else "yes"
        price = yp if side == "yes" else 1 - yp
        if not (0.5 <= price < 1) or cnt <= 0:
            continue
        if model is not None:                 # only the trades the bot would make
            qm = model.q(cell_key("Climate and Weather", price, close_ts - ts), price)
            if qm is None or qm - price < margin or price > 0.97:
                continue
        if tk not in pcache:
            ymd = wx.market_date(ev)
            th = wx_yes_threshold(stype, fl, cp)
            p, a = None, None
            if ymd and th:
                a, b = wx.climate_day(ymd, wx.STATIONS[ser][1])
                maxes = [day_max(prev[ser][m], a, b) for m in MODELS]
                maxes = [x + bias for x in maxes if x is not None]
                if maxes:
                    p = sum(p_band([x], th[0], th[1], smear) for x in maxes) / len(maxes)
            pcache[tk] = (p, a)
        p_yes, day_start = pcache[tk]
        if pre_day and (day_start is None or ts >= day_start):
            continue
        if p_yes is None:
            continue
        p_side = p_yes if side == "yes" else 1 - p_yes
        fee = fee_for("maker", price, cnt, fee_type or "quadratic", fee_mult or 1.0) / cnt
        r = ((1.0 if res == side else 0.0) - price - fee) / price
        k = (tk, side)
        P = pos[k]
        P[0] += r * cnt
        P[1] += cnt
        P[2] += price * cnt
        P[3] = p_side
        P[5] = ev
        P[4] = (P[4] or False) or (p_side < price - gap)
    groups = {"veto": [], "keep": []}
    for (tk, side), (rc, c, pc, p, veto, ev) in pos.items():
        groups["veto" if veto else "keep"].append((rc / c, pc / c, p, ev))
    print(f"Weather forecast veto check — gap {gap:.2f}, smear {smear}°F, bias +{bias}°F "
          f"(day-ahead GFS + ECMWF, {start}..{end})")
    for g, xs in groups.items():
        if not xs:
            print(f"  {g}: none")
            continue
        n = len(xs)
        mean = sum(x[0] for x in xs) / n
        sd = math.sqrt(sum((x[0] - mean) ** 2 for x in xs) / max(1, n - 1))
        # cluster by event for the standard error (markets in one event are correlated)
        ev = defaultdict(list)
        for x in xs:
            ev[x[3]].append(x[0])
        em = [sum(v) / len(v) for v in ev.values()]
        se = (math.sqrt(sum((m - sum(em) / len(em)) ** 2 for m in em) / max(1, len(em) - 1)) /
              math.sqrt(len(em))) if len(em) > 1 else float("nan")
        print(f"  {g:4s}: {n:6,} positions in {len(ev):,} events · avg price {sum(x[1] for x in xs) / n:.3f} · "
              f"model p {sum(x[2] for x in xs) / n:.3f} · return/position {mean * 100:+.2f}% "
              f"(event-clustered se {se * 100:.2f}%) · sd {sd * 100:.0f}%")
    return groups


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kalshi wxcheck")
    ap.add_argument("--db", default="data/kalshi.db")
    ap.add_argument("--gap", type=float, default=0.20)
    ap.add_argument("--smear", type=float, default=3.0)
    ap.add_argument("--all-fills", action="store_true", help="every favorite fill, not just the bot's cells")
    ap.add_argument("--pre-day", action="store_true", help="only fills before the climate day starts")
    a = ap.parse_args(argv)
    run(a.db, a.gap, a.smear, model_path=None if a.all_fills else "data/cell_model.json", pre_day=a.pre_day)
