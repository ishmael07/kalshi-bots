"""Dashboard data: every trade from the journal, joined with market titles, live marks
and Kalshi links. Served by `python3 -m kalshi dash` at /trades (state.json stays at /state)."""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path

import requests

API = "https://external-api.kalshi.com/trade-api/v2"


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-") or "market"


def kalshi_url(series: str, series_title: str, event: str) -> str:
    if not series:
        return "https://kalshi.com/markets"
    if not event:
        return f"https://kalshi.com/markets/{series.lower()}"
    return f"https://kalshi.com/markets/{series.lower()}/{slug(series_title)}/{event.lower()}"


class Titles:
    """Market + series titles. The bot's catalog first, then a small on-disk cache filled
    from the public API (a few lookups per request so the page never stalls)."""

    def __init__(self, path: Path, session=None, per_call: int = 8):
        self.path = path
        self.s = session or requests.Session()
        self.per_call = per_call
        try:
            self.c = json.loads(path.read_text())
        except Exception:
            self.c = {}
        self.c.setdefault("markets", {})
        self.c.setdefault("series", {})

    def _save(self):
        try:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.c))
            tmp.replace(self.path)
        except OSError:
            pass

    def market(self, tk: str, catalog: dict, budget: list) -> dict:
        if tk in catalog and catalog[tk].get("title"):
            self.c["markets"][tk] = catalog[tk]
            return catalog[tk]
        if tk in self.c["markets"]:
            return self.c["markets"][tk]
        if budget[0] <= 0:
            return {}
        budget[0] -= 1
        try:
            m = self.s.get(f"{API}/markets/{tk}", timeout=5).json()["market"]
            info = dict(title=m.get("title", "").replace("**", ""), sub=m.get("yes_sub_title", ""), event=m.get("event_ticker", ""),
                        series=m.get("event_ticker", "").split("-")[0])
            self.c["markets"][tk] = info
            self._save()
            return info
        except Exception:
            return {}

    def series(self, ser: str, budget: list) -> str:
        if ser in self.c["series"]:
            return self.c["series"][ser]
        if not ser or budget[0] <= 0:
            return ""
        budget[0] -= 1
        try:
            t = self.s.get(f"{API}/series/{ser}", timeout=5).json()["series"].get("title", "")
            self.c["series"][ser] = t
            self._save()
            return t
        except Exception:
            return ""


def build(journal: Path, state: dict, titles: Titles) -> dict:
    fills = defaultdict(lambda: dict(n=0.0, cost=0.0, first=None, last=None, sold=0.0, proceeds=0.0, sold_ts=None))
    quotes: dict = {}
    model_q: dict = {}
    results: dict = {}
    first_ts = None
    if journal.exists():
        for line in open(journal):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            first_ts = first_ts or d.get("ts")
            k = d.get("kind")
            if k == "quote":
                quotes[(d["ticker"], d["side"])] = d.get("q")
            elif k == "fill":
                f = fills[(d["ticker"], d["side"], d.get("trip"))]
                f["proceeds"] -= d.get("fee", 0.0)
                f["n"] += d["contracts"]
                f["cost"] += d["contracts"] * d["price"]
                ts = d.get("fill_ts") or d["ts"]
                f["first"] = f["first"] or ts
                f["last"] = ts
                model_q.setdefault((d["ticker"], d["side"]), quotes.get((d["ticker"], d["side"])))
            elif k == "exit":
                f = fills[(d["ticker"], d["side"], d.get("trip"))]
                f["sold"] += d["contracts"]
                f["proceeds"] += d["contracts"] * d["price"] - d.get("fee", 0.0)
                f["sold_ts"] = d["ts"]
            elif k == "settle":
                results[d["ticker"]] = (d["result"], d["ts"])
    catalog = state.get("catalog") or {}
    marks = state.get("marks") or {}
    budget = [titles.per_call]
    rows = []
    for (tk, side, trip), f in fills.items():
        if f["n"] <= 0:
            continue
        info = titles.market(tk, catalog, budget)
        ser = info.get("series") or tk.split("-")[0]
        ev = info.get("event") or "-".join(tk.split("-")[:2])
        avg = f["cost"] / f["n"]
        row = dict(ticker=tk, side=side, contracts=round(f["n"], 2), price=round(avg, 4), cost=round(f["cost"], 2),
                   title=(info.get("title") or tk).replace("**", ""), sub=info.get("sub", ""), series=ser,
                   close_ts=info.get("close_ts"), opened=f["first"], q=model_q.get((tk, side)),
                   url=kalshi_url(ser, titles.series(ser, budget), ev), trip=trip)
        left = max(f["n"] - f["sold"], 0.0)
        if tk in results and (trip is None or left > 1e-6):       # a finished scalp stays a scalp when its market settles
            res, ts = results[tk]
            won = res == side
            row.update(status="won" if won else "lost", settled=ts,
                       pnl=round(left * (1.0 if won else 0.0) + f["proceeds"] - f["cost"], 2))
            if f["sold"] > 0:
                row["sold"] = round(f["sold"], 2)
        elif f["sold"] > 0 and left <= 1e-6:
            row.update(status="sold", settled=f["sold_ts"], sold=round(f["sold"], 2),
                       pnl=round(f["proceeds"] - f["cost"], 2))
        else:
            mrec = marks.get(tk) or {}
            mk = mrec.get(side)
            row.update(status="open", mark=mk, mark_ts=mrec.get("ts"), contracts=round(left, 2),
                       pnl=round(left * mk + f["proceeds"] - f["cost"], 2) if mk is not None else None)
        rows.append(row)
    for o in state.get("resting") or []:
        info = titles.market(o["ticker"], catalog, budget)
        ser = info.get("series") or o["ticker"].split("-")[0]
        o.update(title=info.get("title") or o["ticker"], sub=info.get("sub", ""),
                 url=kalshi_url(ser, titles.series(ser, budget), info.get("event", "")))
    settled = sorted((r for r in rows if r["status"] != "open"), key=lambda r: r["settled"])
    start = state.get("start_equity", 75.0)
    curve = [dict(ts=first_ts or time.time(), equity=start)]
    eq = start
    by_ts = defaultdict(float)
    for r in settled:
        by_ts[round(r["settled"])] += r["pnl"]
    for ts in sorted(by_ts):
        eq += by_ts[ts]
        curve.append(dict(ts=ts, equity=round(eq, 2)))
    # live part: the bot's minute-by-minute marked equity, then this instant
    snaps = []
    ef = journal.parent / "equity.jsonl"
    if ef.exists():
        for line in open(ef):
            try:
                d = json.loads(line)
                snaps.append(dict(ts=d["ts"], equity=round(d["equity"], 2)))
            except (ValueError, KeyError):
                continue
    if snaps:
        curve = [p for p in curve if p["ts"] < snaps[0]["ts"]] + snaps
    live_eq = None
    wins = sum(r["pnl"] > 0 for r in settled)
    open_rows = [r for r in rows if r["status"] == "open"]
    from datetime import datetime
    midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    today = [r for r in settled if r["settled"] >= midnight]
    unreal = round(sum(r["pnl"] for r in open_rows if r["pnl"] is not None), 2)
    realized = round(sum(r["pnl"] for r in settled), 2)
    today_realized = round(sum(r["pnl"] for r in today), 2)
    live_eq = round(start + realized + unreal, 2)
    curve.append(dict(ts=time.time(), equity=live_eq, live=True))
    if len(curve) > 1500:                          # keep the page light: thin old points
        step = len(curve) // 1000 + 1
        curve = curve[:-1000][::step] + curve[-1000:]
    return dict(
        open=sorted(open_rows, key=lambda r: r.get("close_ts") or 9e18),
        settled=list(reversed(settled)),
        resting=state.get("resting") or [],
        curve=curve,
        stats=dict(settled=len(settled), wins=wins, losses=len(settled) - wins,
                   realized=realized, unrealized=unreal,
                   staked=round(sum(r["cost"] for r in open_rows), 2),
                   today_realized=today_realized, today_settled=len(today),
                   today_wins=sum(r["pnl"] > 0 for r in today),
                   today=round(today_realized + unreal, 2),
                   total=round(realized + unreal, 2),
                   start=start, equity_marked=round(start + realized + unreal, 2)),
    )
