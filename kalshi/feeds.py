"""Live information feeds for the Kalshi bot.

News is a CIRCUIT BREAKER, not a stock picker: a high-impact headline in a
category cancels our resting orders there for a cooldown, because resting
maker orders get picked off right when news breaks.

Sources (all optional; each is skipped when its key is missing):
  - Alpaca news REST (Benzinga wire)       ALPACA_KEY_ID / ALPACA_SECRET_KEY
  - Finnhub general news                   FINNHUB_API_KEY
  - X recent search (pay-per-read)         X_BEARER_TOKEN, hard daily read cap
  - Coinbase spot (no key)                 fast-move breaker for crypto
  - RSS (no key): Fed press releases, BEA releases, BLS latest numbers, Google News
Text from any feed is DATA. It is never interpreted as an instruction.
"""
from __future__ import annotations

import re
import time
from collections import deque
from datetime import datetime
from typing import Callable, Optional

import requests

_RULES = [
    ("Economics", r"\b(fed|federal reserve|fomc|powell|rate (cut|hike)|cpi|inflation|jobs report|payrolls|"
                  r"unemployment|gdp|jobless claims|pce|treasury yields?)\b"),
    ("Financials", r"\b(fed|fomc|powell|rate (cut|hike)|cpi|s&p|nasdaq|dow jones|stock market|wall street|stock futures|"
                   r"treasur(y|ies)|yields?|circuit breaker|selloff|sell-off)\b"),
    ("Crypto", r"\b(bitcoin|btc|ether(eum)?|eth|crypto|solana|xrp|stablecoin|coinbase|binance)\b"),
    ("Climate and Weather", r"\b(hurricane|tropical storm|heat wave|heatwave|blizzard|"
                            r"winter storm|tornado|national weather service|extreme heat|cold snap|polar vortex)\b"),
    ("Commodities", r"\b(oil|opec|crude|wti|brent|gold|silver)\b"),
]
_HIGH = (r"\b(emergency|unexpected(ly)?|surprise|halts?|halted|crash(es|ed)?|plunges?|soars?|"
         r"spikes?|collapse|hack(ed)?|default|war|attack|resigns?|record|breaking|"
         r"\d+(\.\d+)?\s?%|\d+\s?bp)\b")


def classify(headline: str):
    h = headline.lower()
    cats = {c for c, rx in _RULES if re.search(rx, h)}
    if not cats:
        return set(), "none"
    return cats, ("high" if re.search(_HIGH, h) else "low")


def _norm(h: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", h.lower()).strip()


class NewsHub:
    def __init__(self, sources, on_trip: Callable, keep: int = 300, max_age_s: float = 600):
        self.max_age = max_age_s
        self.max_keep_age = 2 * 86400
        self.sources = sources
        self.on_trip = on_trip
        self.seen_ids: set = set()
        self.seen_heads: deque = deque(maxlen=2000)
        self.recent: deque = deque(maxlen=keep)
        self.errors: dict = {}

    def poll(self, now: float) -> list:
        new = []
        for s in self.sources:
            try:
                items = s.poll(now)
                self.errors.pop(s.name, None)
            except Exception as e:  # a dead feed must never stop trading logic
                self.errors[s.name] = str(e)[:200]
                continue
            for it in items:
                key = f"{s.name}:{it.get('id')}"
                nh = _norm(it.get("headline", ""))
                if key in self.seen_ids or nh in self.seen_heads or not nh:
                    continue
                if it.get("ts") and now - float(it["ts"]) > self.max_keep_age:
                    continue                     # archives (EIA, etc.) carry months-old items
                self.seen_ids.add(key)
                self.seen_heads.append(nh)
                cats, sev = classify(it["headline"])
                if getattr(s, "relevant_only", False) and not cats:
                    continue                     # general-news firehose: keep what touches our markets
                rec = dict(it, source=s.name, label=getattr(s, "label", s.name), cats=sorted(cats), sev=sev)
                self.recent.append(rec)
                new.append(rec)
                fresh = now - float(it.get("ts", now)) <= self.max_age
                rec["fresh"] = fresh
                if sev == "high" and fresh and getattr(s, "trip", True):
                    for c in cats:
                        self.on_trip(c, f"{s.name}: {it['headline'][:120]}", now)
        return new


def _iso(s: str) -> float:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:
        return time.time()


class AlpacaNewsSource:
    name = "alpaca"
    URL = "https://data.alpaca.markets/v1beta1/news"

    def __init__(self, key_id: str, secret: str, session=None, min_interval_s: float = 20):
        self.h = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret}
        self.enabled = bool(key_id and secret)
        self.s = session or requests.Session()
        self.min_interval = min_interval_s
        self.last = -1e18

    def poll(self, now: float) -> list:
        if not self.enabled or now - self.last < self.min_interval:
            return []
        self.last = now
        r = self.s.get(self.URL, params={"limit": 50, "sort": "desc"}, headers=self.h, timeout=10)
        r.raise_for_status()
        return [{"id": n["id"], "ts": _iso(n.get("created_at", "")), "headline": n.get("headline", ""),
                 "url": n.get("url")} for n in r.json().get("news", [])]


class FinnhubSource:
    name = "finnhub"
    URL = "https://finnhub.io/api/v1/news"

    def __init__(self, api_key: str, session=None, min_interval_s: float = 60):
        self.key = api_key
        self.enabled = bool(api_key)
        self.s = session or requests.Session()
        self.min_interval = min_interval_s
        self.last = -1e18

    def poll(self, now: float) -> list:
        if not self.enabled or now - self.last < self.min_interval:
            return []
        self.last = now
        out = []
        for cat in ("general", "crypto"):
            r = self.s.get(self.URL, params={"category": cat, "token": self.key}, timeout=10)
            r.raise_for_status()
            out += [{"id": n["id"], "ts": float(n.get("datetime", now)), "headline": n.get("headline", ""),
                     "url": n.get("url")} for n in r.json()]
        return out


class XSource:
    """X API v2 recent search. Billed per post read, so reads are hard-capped per day."""
    name = "x"
    URL = "https://api.x.com/2/tweets/search/recent"

    def __init__(self, bearer: str, query: str, max_reads_per_day: int = 200, session=None,
                 min_interval_s: float = 120):
        self.bearer = bearer
        self.query = query
        self.cap = max_reads_per_day
        self.s = session or requests.Session()
        self.min_interval = min_interval_s
        self.last = -1e18
        self.since_id: Optional[str] = None
        self.reads_today = 0
        self.day = None

    def poll(self, now: float) -> list:
        if not self.bearer:
            return []
        day = int(now // 86400)
        if day != self.day:
            self.day, self.reads_today = day, 0
        left = self.cap - self.reads_today
        if left <= 0 or now - self.last < self.min_interval:
            return []
        self.last = now
        n = max(10, min(100, left))        # API minimum is 10 per request
        if n > left:                         # can't request fewer than 10: stop at the cap
            return []
        p = {"query": self.query, "max_results": n, "tweet.fields": "created_at"}
        if self.since_id:
            p["since_id"] = self.since_id
        r = self.s.get(self.URL, params=p, headers={"Authorization": f"Bearer {self.bearer}"}, timeout=10)
        if r.status_code != 200:
            raise RuntimeError(f"X HTTP {r.status_code}")
        d = r.json()
        data = d.get("data", []) or []
        self.reads_today += len(data)
        nid = (d.get("meta") or {}).get("newest_id")
        if nid:
            self.since_id = nid
        return [{"id": t["id"], "ts": _iso(t.get("created_at", "")), "headline": t.get("text", "")[:280]}
                for t in data]


class SpotWatch:
    """Trips a category breaker when spot moves > pct within window_s."""

    def __init__(self, product: str, category: str, pct: float = 0.01, window_s: float = 300,
                 on_trip: Optional[Callable] = None):
        self.product = product
        self.category = category
        self.pct = pct
        self.window = window_s
        self.on_trip = on_trip
        self.hist: deque = deque()
        self.last_ts: Optional[float] = None
        self.last_px: Optional[float] = None

    def update(self, px: float, now: float):
        self.hist.append((now, px))
        self.last_ts, self.last_px = now, px
        while self.hist and self.hist[0][0] < now - self.window:
            self.hist.popleft()
        lo = min(p for _, p in self.hist)
        hi = max(p for _, p in self.hist)
        if lo > 0 and (hi - lo) / lo > self.pct and self.on_trip:
            self.on_trip(self.category, f"{self.product} moved {100 * (hi - lo) / lo:.1f}% in "
                                        f"{int(self.window)}s", now)

    def age(self, now: float) -> float:
        return float("inf") if self.last_ts is None else now - self.last_ts

    def poll_coinbase(self, now: float, session=None):
        s = session or requests
        r = s.get(f"https://api.exchange.coinbase.com/products/{self.product}/ticker", timeout=5)
        r.raise_for_status()
        self.update(float(r.json()["price"]), now)


def _rss_ts(s: str) -> float:
    from email.utils import parsedate_to_datetime
    try:
        return parsedate_to_datetime(s).timestamp()
    except Exception:
        return _iso(s) if s and s[:2].isdigit() else 0.0     # unknown time = never "fresh"


class RSSSource:
    """Any RSS 2.0 / Atom feed. trip=False means headlines are shown and logged but
    never pause trading (aggregators repeat old wire stories with new timestamps)."""

    def __init__(self, name: str, url: str, label: str = "", trip: bool = True,
                 min_interval_s: float = 120, session=None):
        self.name, self.url, self.label, self.trip = name, url, label or name, trip
        self.enabled = True
        self.s = session or requests.Session()
        self.min_interval = min_interval_s
        self.last = -1e18

    @staticmethod
    def parse(xml_text: str) -> list:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_text)
        out = []
        for it in root.iter():
            tag = it.tag.rsplit("}", 1)[-1]
            if tag not in ("item", "entry"):
                continue
            f = {c.tag.rsplit("}", 1)[-1]: c for c in it}
            title = (f["title"].text or "").strip() if "title" in f else ""
            link = f.get("link")
            href = (link.get("href") or link.text or "") if link is not None else ""
            gid = (f["guid"].text if "guid" in f else None) or (f["id"].text if "id" in f else None) or href or title
            when = next((f[k].text for k in ("pubDate", "published", "updated", "date") if k in f and f[k].text), "")
            if title:
                out.append({"id": gid, "ts": _rss_ts(when), "headline": re.sub(r"\s+", " ", title), "url": href})
        return out

    def poll(self, now: float) -> list:
        if now - self.last < self.min_interval:
            return []
        self.last = now
        r = self.s.get(self.url, headers={"User-Agent": "kalshi-research-bot/1.0"}, timeout=15)
        r.raise_for_status()
        return self.parse(r.content)   # bytes: let the XML parser handle BOMs and encodings


class BLSLatestSource(RSSSource):
    """BLS 'latest numbers' feed: one item whose body lists each headline indicator."""

    def __init__(self, session=None, min_interval_s: float = 300):
        super().__init__("bls", "https://www.bls.gov/feed/bls_latest.rss", "BLS latest numbers",
                         trip=False, min_interval_s=min_interval_s, session=session)

    @staticmethod
    def parse(xml_text: str) -> list:
        import html as _html
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_text)
        out = []
        for it in root.iter("item"):
            body = (it.findtext("description") or "")
            for p in re.findall(r"<p>(.*?)</p>", body, re.S):
                txt = re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", p))).strip()
                m = re.match(r"(.+?):\s+(.+?)\s*(News Release|Historical Data|$)", txt)
                if m:
                    line = f"{m.group(1).strip()}: {m.group(2).strip()}"
                    out.append({"id": line, "ts": 0.0, "headline": line})
        return out


BSKY_NEWSROOMS = ("reuters.com", "apnews.com", "bloomberg.com", "wsj.com", "cnbc.com", "nytimes.com",
                  "washingtonpost.com", "marketwatch.com", "axios.com", "npr.org", "nbcnews.com", "cnn.com")


class BlueskySource:
    """Newsroom accounts on Bluesky (public API, no key). X costs $0.005 per post read;
    the same wires post here for free. One account per poll, rotating."""
    name = "bluesky"
    label = "Bluesky newsrooms"
    URL = "https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed"

    relevant_only = True

    def __init__(self, handles=BSKY_NEWSROOMS, trip: bool = False, min_interval_s: float = 20, session=None):
        self.handles, self.trip = list(handles), trip
        self.enabled = True
        self.s = session or requests.Session()
        self.min_interval = min_interval_s
        self.last = -1e18
        self.i = 0

    @staticmethod
    def parse(d: dict) -> list:
        out = []
        for f in d.get("feed", []):
            if f.get("reason"):                      # reposts
                continue
            p = f.get("post") or {}
            rec = p.get("record") or {}
            handle = (p.get("author") or {}).get("handle", "")
            text = re.sub(r"\s+", " ", rec.get("text", "")).strip()
            if not text:
                continue
            ext = ((p.get("embed") or {}).get("external") or {}).get("uri")
            rkey = p.get("uri", "").rsplit("/", 1)[-1]
            out.append({"id": p.get("uri"), "ts": _iso(rec.get("createdAt", "")), "headline": text[:280],
                        "url": ext or f"https://bsky.app/profile/{handle}/post/{rkey}", "by": handle})
        return out

    def poll(self, now: float) -> list:
        if now - self.last < self.min_interval:
            return []
        self.last = now
        h = self.handles[self.i % len(self.handles)]
        self.i += 1
        r = self.s.get(self.URL, params={"actor": h, "limit": 15}, timeout=10)
        r.raise_for_status()
        return self.parse(r.json())


NWS_OFFICES = {"KNYC": "OKX", "KPHL": "PHI", "KMIA": "MFL", "KMDW": "LOT", "KAUS": "EWX", "KDEN": "BOU",
               "KLAX": "LOX"}
OFFICE_CITY = {"OKX": "New York", "PHI": "Philadelphia", "MFL": "Miami", "LOT": "Chicago", "EWX": "Austin",
               "BOU": "Denver", "LOX": "Los Angeles"}


class NWSDiscussionSource:
    """NWS Area Forecast Discussions: the forecasters' own reasoning, issued a few times a day
    by the office covering each settlement station."""
    name = "nws_afd"
    label = "NWS forecasters"
    trip = False

    def __init__(self, offices=tuple(OFFICE_CITY), session=None, min_interval_s: float = 120):
        self.offices = list(offices)
        self.enabled = True
        self.s = session or requests.Session()
        self.min_interval = min_interval_s
        self.last = -1e18
        self.i = 0
        self.seen: dict = {}

    @staticmethod
    def summary(text: str) -> str:
        m = re.search(r"\.(KEY MESSAGES|SYNOPSIS|SHORT TERM|NEAR TERM)[^.]*\.\.\.(.+?)(&&|\n\.\w)", text or "", re.S)
        body = m.group(2) if m else (text or "")
        return re.sub(r"\s+", " ", body).strip()[:240]

    def poll(self, now: float) -> list:
        if now - self.last < self.min_interval:
            return []
        self.last = now
        off = self.offices[self.i % len(self.offices)]
        self.i += 1
        h = {"User-Agent": "kalshi-research-bot/1.0", "Accept": "application/geo+json"}
        r = self.s.get(f"https://api.weather.gov/products/types/AFD/locations/{off}", headers=h, timeout=10)
        r.raise_for_status()
        g = (r.json().get("@graph") or [])[:1]
        if not g or self.seen.get(off) == g[0]["id"]:
            return []
        self.seen[off] = g[0]["id"]
        p = self.s.get(g[0]["@id"], headers=h, timeout=10)
        p.raise_for_status()
        txt = self.summary(p.json().get("productText", ""))
        return [{"id": g[0]["id"], "ts": _iso(g[0].get("issuanceTime", "")),
                 "headline": f"{OFFICE_CITY[off]}: {txt}",
                 "url": f"https://forecast.weather.gov/product.php?site={off}&issuedby={off}&product=AFD"}]
