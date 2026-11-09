"""python3 -m kalshi train | run | dash

train: learn the cell win-rate table from every stored favorite-side maker fill.
run:   the always-on bot in SHADOW mode (public data only, virtual money).
dash:  local dashboard on http://127.0.0.1:8789
"""
from __future__ import annotations

import os
import signal
import time
import tomllib
from pathlib import Path

ENV_FILES = [Path.home() / "dawn-raid" / ".env", Path(".env")]
DEFAULT_CFG = Path("kalshi.toml")


def load_env():
    for p in ENV_FILES:
        if p.exists():
            for line in p.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_cfg(path=DEFAULT_CFG) -> dict:
    return tomllib.loads(Path(path).read_text()) if Path(path).exists() else {}


def ntfy(topic: str):
    import requests

    def send(msg: str, priority: str = "default", title: str = "Kalshi bot"):
        if not topic:
            return
        try:
            requests.post(f"https://ntfy.sh/{topic}", data=msg.encode()[:3900], timeout=5,
                          headers={"Title": title, "Priority": priority})
        except Exception:
            pass                          # alerts must never stop the loop
    return send


def cmd_train(argv):
    import argparse
    from datetime import datetime, timezone
    from .live import CellModel
    from .run_strategy import load_fills
    from .store import Store
    ap = argparse.ArgumentParser(prog="kalshi train")
    ap.add_argument("--db", default="data/kalshi.db")
    ap.add_argument("--since", default="2025-05-01")
    ap.add_argument("--out", default="data/cell_model.json")
    ap.add_argument("--min-prior", type=float, default=30, help="min distinct events per cell")
    a = ap.parse_args(argv)
    ts = datetime.strptime(a.since, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
    fills = load_fills(Store(a.db), ts, cell_mode="cat_band_ttc")
    m = CellModel.from_fills(fills, a.min_prior)
    m.save(a.out)
    good = sorted(((k, v[0] / v[1], v[1]) for k, v in m.table.items() if v[1] >= a.min_prior),
                  key=lambda x: x[0])
    print(f"trained on {len(fills):,} fills -> {len(good)} cells with >= {a.min_prior:.0f} events")
    for k, q, n in good:
        lo = int(k.split("|")[1].split("-")[0]) / 100
        print(f"  {k:45s} win {q * 100:5.1f}%  vs price ~{lo + 0.025:.3f}  edge {(q - lo - 0.025) * 100:+5.1f}c  "
              f"({n:,.0f} events)")


def cmd_run(argv):
    import argparse
    from .api import KalshiPublic
    from .bot import Bot, BotConfig
    from .feeds import AlpacaNewsSource, FinnhubSource, NewsHub, SpotWatch, XSource
    from .live import Breakers, CellModel, ShadowExecutor
    ap = argparse.ArgumentParser(prog="kalshi run")
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--interval", type=float, default=20.0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args(argv)
    load_env()
    raw = load_cfg(a.config)
    bot_cfg = raw.get("bot", {})
    cfg = BotConfig(**{k: v for k, v in bot_cfg.items() if k in BotConfig.__dataclass_fields__})
    model_path = Path(raw.get("model", {}).get("path", "data/cell_model.json"))
    model = CellModel.load(model_path)
    model_mtime = model_path.stat().st_mtime
    breakers = Breakers(cooldown_s=raw.get("news", {}).get("cooldown_s", 600))
    trip = lambda cat, why, now: breakers.trip(cat, why, now)
    ncfg0 = raw.get("news", {})
    srcs = []
    if ncfg0.get("alpaca", True):
        srcs.append(AlpacaNewsSource(os.environ.get("ALPACA_KEY_ID", ""), os.environ.get("ALPACA_SECRET_KEY", "")))
    if ncfg0.get("finnhub", True):
        srcs.append(FinnhubSource(os.environ.get("FINNHUB_API_KEY", "")))
    xcfg = raw.get("x", {})
    if xcfg.get("enabled") and os.environ.get("X_BEARER_TOKEN"):
        srcs.append(XSource(os.environ["X_BEARER_TOKEN"], xcfg.get("query", "bitcoin OR fed OR cpi"),
                            max_reads_per_day=int(xcfg.get("max_reads_per_day", 200))))
    from urllib.parse import quote
    from .feeds import BLSLatestSource, RSSSource
    ncfg = raw.get("news", {})
    if ncfg.get("rss", True):
        gq = quote(ncfg.get("google_query", '("Federal Reserve" OR inflation OR CPI OR "jobs report" OR payrolls '
                                            'OR "jobless claims" OR GDP OR "gas prices" OR TSA) when:1h'))
        srcs += [RSSSource("fed", "https://www.federalreserve.gov/feeds/press_all.xml", "Federal Reserve press"),
                 RSSSource("bea", "https://apps.bea.gov/rss/rss.xml", "BEA releases", min_interval_s=300),
                 BLSLatestSource(),
                 RSSSource("gnews", f"https://news.google.com/rss/search?q={gq}&hl=en-US&gl=US&ceid=US:en",
                           "Google News (Reuters, AP, ...)", trip=False, min_interval_s=300),
                 RSSSource("marketwatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories",
                           "MarketWatch", trip=False, min_interval_s=180),
                 RSSSource("cnbc", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
                           "CNBC economy", trip=False, min_interval_s=180),
                 RSSSource("dol", "https://www.dol.gov/rss/releases.xml", "Labor Dept releases", min_interval_s=300),
                 RSSSource("eia", "https://www.eia.gov/rss/todayinenergy.xml", "EIA energy", trip=False,
                           min_interval_s=900),
                 RSSSource("nhc", "https://www.nhc.noaa.gov/index-at.xml", "Hurricane Center", min_interval_s=600)]
    if ncfg.get("bluesky", True):
        from .feeds import BlueskySource
        srcs.append(BlueskySource())
    if ncfg.get("nws_discussions", True):
        from .feeds import NWSDiscussionSource
        srcs.append(NWSDiscussionSource())
    news = NewsHub(srcs, on_trip=trip)
    spots = [SpotWatch(p, "Crypto", on_trip=trip) for p in raw.get("spot", {}).get("products", [])]
    send = ntfy(os.environ.get("NTFY_TOPIC", "") if raw.get("notify", {}).get("enabled", True) else "")
    from .weather import WeatherObs
    wobs = WeatherObs() if raw.get("weather", {}).get("enabled", True) else None
    from . import signals as sg
    scfg = raw.get("signals", {})
    trip_for = lambda key, why, now, cd=None: breakers.trip(key, why, now, cooldown_s=cd)
    sigs = []
    if scfg.get("calendar", True):
        sigs.append(sg.ReleaseCalendar(trip_for, before_s=scfg.get("pause_before_s", 0),
                                       after_s=scfg.get("pause_after_s", 1800)))
    if scfg.get("nowcasts", True):
        sigs.append(sg.Nowcasts())
    if scfg.get("weather_forecast", True):
        sigs.append(sg.WeatherForecast(wobs, cfg.series))
    if scfg.get("gas", True) and "KXAAAGASD" in cfg.series:
        sigs.append(sg.GasPrices())
    if scfg.get("tsa", True) and "KXTSAW" in cfg.series:
        sigs.append(sg.TSAThroughput())
    if scfg.get("polymarket", True) and "KXFEDDECISION" in cfg.series:
        sigs.append(sg.PolymarketFed())
    if scfg.get("equities", True):
        sigs.append(sg.EquityWatch(os.environ.get("ALPACA_KEY_ID", ""), os.environ.get("ALPACA_SECRET_KEY", ""),
                                   trip_for))
    bot = Bot(cfg, KalshiPublic(rps=float(raw.get("api", {}).get("rps", 8))), model, ShadowExecutor(cfg.fill_share), breakers, news, spots, send,
              weather=wobs, signals=sigs)
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    send(f"Kalshi bot started (shadow, ${cfg.start_equity:.0f} virtual, {len(cfg.series)} series)")
    print(f"shadow bot running; state -> {cfg.out_dir}/state.json", flush=True)
    while not stop["flag"]:
        t0 = time.time()
        try:                                   # hot-reload a retrained model
            mt = model_path.stat().st_mtime
            if mt != model_mtime:
                bot.model = CellModel.load(model_path)
                bot.quoter.model = bot.model
                bot.last_discover = -1e18
                model_mtime = mt
                bot.log("model_reload", t0, cells=len(bot.model.table))
        except Exception as e:
            bot.log("error", t0, error=f"model reload: {e}"[:300])
        bot.step(t0)
        if a.once:
            break
        time.sleep(max(1.0, a.interval - (time.time() - t0)))
    send("Kalshi bot stopped")


ACCOUNT_DIRS = {"steady": Path("data/bot"), "aggressive": Path("data/bot_aggr"), "scalper": Path("data/bot_scalp")}


def accounts_summary(accounts) -> dict:
    import json
    out = {}
    for name, d in accounts.items():
        try:
            st = json.loads((d / "state.json").read_text())
            out[name] = dict(equity=st.get("equity_marked", st.get("equity")), start=st.get("start_equity"),
                             label=st.get("label") or name.title(), ts=st.get("ts"))
        except (OSError, ValueError):
            continue
    return out


def cmd_export(argv):
    """Write the dashboard as static files (for GitHub Pages): <out>/index.html + <out>/data/*.json."""
    import json
    import shutil
    from .dash import Titles, build
    out = Path(argv[0] if argv else "docs")
    (out / "data").mkdir(parents=True, exist_ok=True)
    titles = Titles(ACCOUNT_DIRS["steady"] / "titles.json")
    for name, d in ACCOUNT_DIRS.items():
        try:
            st = json.loads((d / "state.json").read_text())
        except (OSError, ValueError):
            continue
        (out / "data" / f"state_{name}.json").write_text(json.dumps(st))
        (out / "data" / f"trades_{name}.json").write_text(json.dumps(build(d / "journal.jsonl", st, titles)))
    (out / "data" / "accounts.json").write_text(json.dumps(accounts_summary(ACCOUNT_DIRS)))
    shutil.copy(Path(__file__).parent / "dashboard.html", out / "index.html")
    (out / ".nojekyll").write_text("")


def cmd_scalp(argv):
    import argparse
    from .api import KalshiPublic
    from .bot import BotConfig
    from .scalp import ScalpBot, ScalpExecutor, ScalpParams
    ap = argparse.ArgumentParser(prog="kalshi scalp")
    ap.add_argument("--config", default="kalshi_scalp.toml")
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args(argv)
    raw = load_cfg(a.config)
    cfg = BotConfig(**{k: v for k, v in raw.get("bot", {}).items() if k in BotConfig.__dataclass_fields__})
    bot = ScalpBot(cfg, KalshiPublic(rps=float(raw.get("api", {}).get("rps", 4))), ScalpExecutor(cfg.fill_share),
                   ScalpParams(**raw.get("scalp", {})))
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    print(f"scalper running; state -> {cfg.out_dir}/state.json", flush=True)
    while not stop["flag"]:
        t0 = time.time()
        bot.step(t0)
        if a.once:
            break
        time.sleep(max(1.0, a.interval - (time.time() - t0)))


def cmd_dash(argv):
    import http.server
    import json
    import socketserver
    from .dash import Titles, build
    from urllib.parse import parse_qs, urlparse
    here = Path(__file__).parent
    ACCOUNTS = ACCOUNT_DIRS
    titles = Titles(ACCOUNTS["steady"] / "titles.json")

    def account(path: str) -> Path:
        a = (parse_qs(urlparse(path).query).get("acct") or ["steady"])[0]
        return ACCOUNTS.get(a, ACCOUNTS["steady"])

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            state = account(self.path) / "state.json"
            if self.path.startswith("/state"):
                body = state.read_bytes() if state.exists() else b"{}"
                ctype = "application/json"
            elif self.path.startswith("/trades"):
                try:
                    st = json.loads(state.read_text()) if state.exists() else {}
                except ValueError:
                    st = {}
                body = json.dumps(build(state.parent / "journal.jsonl", st, titles)).encode()
                ctype = "application/json"
            elif self.path.startswith("/accounts"):
                body = json.dumps(accounts_summary(ACCOUNTS)).encode()
                ctype = "application/json"
            else:
                body = (here / "dashboard.html").read_bytes()
                ctype = "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    port = int(argv[0]) if argv else 8789
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("127.0.0.1", port), H) as srv:
        print(f"dashboard: http://127.0.0.1:{port}", flush=True)
        srv.serve_forever()
