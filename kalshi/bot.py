"""The Kalshi bot loop (shadow mode by default).

step(now):
  1. feeds: poll news hub + spot watches -> trip breakers; stale market data trips '*'
  2. settle closed markets we hold -> realized P&L
  3. read the public tape for markets with resting orders -> shadow fills
  4. re-quote every discovered market (favorite-side maker bids), honoring
     breakers and RiskGuard
  5. write state.json (dashboard) and append journal.jsonl
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .live import Book, Breakers, CellModel, Quoter, ShadowExecutor, VirtualOrder, apply_view
from . import weather as wx
from .run_strategy import corr_group
from .store import iso_ts
from .fees import taker_fee
from .trading import RiskBlocked, RiskGuard


@dataclass
class BotConfig:
    series: list
    start_equity: float = 75.0
    margin: float = 0.03
    kelly_mult: float = 0.25
    max_frac: float = 0.10
    fill_share: float = 0.25
    min_ttc_s: float = 120.0
    max_ttc_s: float = 2 * 86400.0
    max_order_dollars: float = 7.5
    max_open_dollars: float = 60.0
    max_event_dollars: float = 7.5
    daily_loss_halt: float = 5.0
    sizing: str = "kelly"            # "fixed" = fixed fraction of equity per bet
    fixed_frac: float = 0.01
    group_cap_frac: float = 0.0      # >0: per-correlated-group cap as a fraction of equity
    open_cap_frac: float = 0.0       # >0: total in-play cap as a fraction of equity
    label: str = "Steady"            # account name shown on the dashboard
    veto_gap: float = 0.20           # a data signal disagrees when it puts our side this far below our price
    veto_enforce: list = field(default_factory=list)   # signals whose disagreement blocks the bid (else log only)
    discover_every_s: float = 300.0
    max_books_per_step: int = 60
    out_dir: str = "data/bot"


class Bot:
    def __init__(self, cfg: BotConfig, public, model: CellModel, executor: ShadowExecutor,
                 breakers: Breakers, news=None, spots=(), notify: Optional[Callable] = None,
                 weather=None, signals=()):
        self.weather = weather
        self.signals = list(signals)
        self.marks: dict = {}                # ticker -> best bids {"yes": p, "no": p, "ts": t}
        self.last_snap = -1e18
        self.vetoes: list = []
        self.vetoed: dict = {}
        self.locked_seen: set = set()
        self.cfg = cfg
        self.pub = public
        self.model = model
        self.quoter = Quoter(model, cfg.margin, cfg.kelly_mult, cfg.max_frac,
                             sizing=cfg.sizing, fixed_frac=cfg.fixed_frac)
        self.executor = executor
        self.breakers = breakers
        self.news = news
        self.spots = list(spots)
        self.notify = notify or (lambda *a, **k: None)
        self.guard = RiskGuard(cfg.max_order_dollars, cfg.max_open_dollars, cfg.max_event_dollars,
                               cfg.daily_loss_halt)
        self.out = Path(cfg.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.journal = open(self.out / "journal.jsonl", "a")
        self.book_path = self.out / "book.json"
        self.executor.load(self.book_path)
        self.markets: dict = {}              # ticker -> dict(event, category, close_ts, tick)
        self.categories: dict = {}
        self.last_discover = -1e18
        self.tape_cursor: dict = {}          # ticker -> last trade ts seen
        self.rotate = 0
        self.day = None
        self.day_start_realized = 0.0
        self.recent_fills: list = []
        self.errors = 0
        self.last_ok = None
        self.disk_errors = 0

    # ---------------------------------------------------------------- helpers
    def log(self, kind: str, now: float, **kw):
        try:
            self.journal.write(json.dumps({"kind": kind, "ts": now, **kw}) + "\n")
            self.journal.flush()
        except OSError:
            self.disk_errors += 1          # disk full: keep trading logic alive

    def equity(self) -> float:
        return self.cfg.start_equity + self.executor.realized

    def _category(self, series: str) -> str:
        if series not in self.categories:
            self.categories[series] = (self.pub.series(series) or {}).get("category", "Other")
        return self.categories[series]

    def discover(self, now: float):
        """Only keep markets worth an order-book call: inside a time window where the
        model has trusted cells, with a favorite currently priced 45-98c."""
        from .study import ttc_bucket
        found = {}
        windows = self.model.windows()
        for s in self.cfg.series:
            cat = self._category(s)
            for m in self.pub.paginate("/markets", "markets", {"series_ticker": s, "status": "open"}):
                close = iso_ts(m.get("close_time", ""))
                ttc = close - now
                if not (self.cfg.min_ttc_s <= ttc <= self.cfg.max_ttc_s):
                    continue
                ev = m.get("event_ticker", m["ticker"])
                wx_today = False
                if s in wx.STATIONS and self.weather is not None:
                    ymd = wx.market_date(ev)
                    wx_today = bool(ymd) and wx.climate_day(ymd, wx.STATIONS[s][1])[0] <= now
                if not wx_today:              # live observations can decide today's markets
                    if (cat, ttc_bucket(ttc)) not in windows:
                        continue
                    try:
                        fav = max(float(m.get("yes_bid_dollars") or 0), float(m.get("no_bid_dollars") or 0))
                    except ValueError:
                        fav = 0.0
                    if not (0.45 <= fav <= 0.98):
                        continue
                tick = 0.01 if m.get("price_level_structure", "linear_cent") == "linear_cent" else 0.001
                fl, cp = m.get("floor_strike"), m.get("cap_strike")
                found[m["ticker"]] = dict(event=ev, category=cat, close_ts=close, tick=tick, series=s,
                                          group=corr_group(cat, ev, close), strike_type=m.get("strike_type"),
                                          floor=float(fl) if fl is not None else None,
                                          cap=float(cp) if cp is not None else None,
                                          title=m.get("title", ""), sub=m.get("yes_sub_title", ""))
        # keep markets we hold even if they left the window
        for tk in list(self.executor.positions) + list(self.executor.orders):
            if tk in self.markets and tk not in found:
                found[tk] = self.markets[tk]
        self.markets = found
        self.last_discover = now
        for sig in self.signals:
            if hasattr(sig, "note_markets"):
                sig.note_markets(found)

    def book(self, tk: str) -> Book:
        d = self.pub.get(f"/markets/{tk}/orderbook") or {}
        ob = d.get("orderbook_fp") or {}
        conv = lambda lvls: sorted(((float(p), float(s)) for p, s in (lvls or [])), reverse=True)
        m = self.markets[tk]
        return Book(tk, m["event"], m["category"], m["close_ts"], conv(ob.get("yes_dollars")),
                    conv(ob.get("no_dollars")), m["tick"])

    # ---------------------------------------------------------------- the loop
    def step(self, now: float):
        day = int(now // 86400)
        if day != self.day:
            self.day, self.day_start_realized = day, self.executor.realized
        # 1. feeds
        if self.news is not None:
            for item in self.news.poll(now):
                self.log("news", now, source=item["source"], headline=item["headline"][:200],
                         cats=item["cats"], sev=item["sev"])
        for w in self.spots:
            try:
                w.poll_coinbase(now)
            except Exception as e:
                self.log("feed_error", now, feed=w.product, error=str(e)[:200])
            if w.age(now) > 120:
                self.breakers.trip(w.category, f"{w.product} spot stale", now, cooldown_s=120)
        if self.weather is not None:
            self.weather.poll(now, self.cfg.series)
        for sig in self.signals:
            try:
                sig.poll(now)
            except Exception as e:             # belt and braces: signals already swallow errors
                self.log("feed_error", now, feed=sig.name, error=str(e)[:200])
        try:
            if now - self.last_discover >= self.cfg.discover_every_s:
                self.discover(now)
            self._settle(now)
            self._tape(now)
            self._quote(now)
            self.last_ok, self.errors = now, 0
        except Exception as e:  # network trouble: pull all quotes, keep going
            self.errors += 1
            self.log("error", now, error=str(e)[:300])
            if self.errors >= 3:
                self.breakers.trip("*", f"market data errors x{self.errors}", now, cooldown_s=300)
                self.executor.cancel_all()
        try:
            self._snapshot(now)
            self._write_state(now)
        except OSError as e:                 # e.g. disk full: never let this kill the loop
            self.disk_errors += 1
            if self.disk_errors in (1, 20, 200):
                self.notify(f"Kalshi bot: cannot write state ({e.strerror}). Free disk space.",
                            priority="high")

    def _settle(self, now: float):
        for tk in list(self.executor.positions):
            m = self.markets.get(tk)
            if m and now < m["close_ts"]:
                continue
            res = (self.pub.get(f"/markets/{tk}") or {}).get("market", {})
            if res.get("result") in ("yes", "no"):
                pnl = self.executor.settle(tk, res["result"])
                self.log("settle", now, ticker=tk, result=res["result"], pnl=round(pnl, 4),
                         equity=round(self.equity(), 4))
                self.notify(f"Settled {tk}: {res['result'].upper()} {pnl:+.2f} (equity "
                            f"${self.equity():.2f})", priority="high" if abs(pnl) > 1 else "default")
                self.markets.pop(tk, None)

    def _tape(self, now: float):
        for tk in list(self.executor.orders):
            since = self.tape_cursor.get(tk, 0.0)
            trades = []
            for t in self.pub.paginate("/markets/trades", "trades", {"ticker": tk, "min_ts": int(since)}):
                ts = iso_ts(t["created_time"])
                if ts <= since:
                    continue
                trades.append(dict(ts=ts, yes_price=float(t["yes_price_dollars"]),
                                   count=float(t["count_fp"]), taker_side=t["taker_side"]))
            if trades:
                self.tape_cursor[tk] = max(t["ts"] for t in trades)
            for o, n, ts in self.executor.on_trades(tk, trades):
                rec = dict(ticker=tk, side=o.side, price=o.price, contracts=round(n, 2), fill_ts=ts)
                self.log("fill", now, **rec)
                self.recent_fills = ([rec] + self.recent_fills)[:50]
                self.notify(f"Filled {tk} {o.side.upper()} {n:.2f} @ {o.price:.2f}", priority="default")

    def _quote(self, now: float):
        tks = sorted(self.markets)
        if not tks:
            return
        n = min(self.cfg.max_books_per_step, len(tks))
        batch = [tks[(self.rotate + i) % len(tks)] for i in range(n)]
        self.rotate = (self.rotate + n) % len(tks)
        day_pnl = self.executor.realized - self.day_start_realized
        eq = self.equity()
        if self.cfg.group_cap_frac > 0:
            self.guard.max_event = min(self.cfg.max_event_dollars, self.cfg.group_cap_frac * eq)
        if self.cfg.open_cap_frac > 0:
            self.guard.max_open = min(self.cfg.max_open_dollars, self.cfg.open_cap_frac * eq)
        for tk in batch:
            m = self.markets[tk]
            bk = None
            if tk in self.executor.positions and now < m["close_ts"]:
                bk = self.book(tk)                 # live mark for open bets, even while paused
                self._mark(tk, bk, now)
                self._exit_dead(tk, m, bk, now)
            if now >= m["close_ts"] - self.cfg.min_ttc_s:
                self._cancel(tk, now, "closing")
                continue
            if self.breakers.blocked(m["category"], now) or self.breakers.blocked(m.get("series", ""), now):
                self._cancel(tk, now, "breaker")
                continue
            if bk is None:
                bk = self.book(tk)
                self._mark(tk, bk, now)
            wants = self.quoter.desired(bk, now, self.equity())
            view = self._wx_view(tk, now)
            if view:
                wants = apply_view(self.quoter, bk, wants, view, self.equity())
                if tk not in self.locked_seen:
                    self.locked_seen.add(tk)
                    self.log("weather_lock", now, ticker=tk, **view)
            for sig in self.signals:
                v = sig.view(m, now)
                if v.get("locked"):
                    wants = apply_view(self.quoter, bk, wants, v, self.equity())
                    if tk not in self.locked_seen:
                        self.locked_seen.add(tk)
                        self.log("data_lock", now, ticker=tk, source=sig.name, **v)
                elif "p_yes" in v:
                    wants = self._veto(tk, wants, v, sig.name, now)
            resting = self.executor.resting(tk)
            keep = []
            for w in wants:
                same = [o for o in resting if o.side == w.side and abs(o.price - w.price) < 1e-9]
                if same:
                    keep += same
                    continue
                cost = w.price * w.count
                try:
                    self.guard.check(m["group"], cost, self.executor.open_cost() + self._resting_cost(),
                                     self.executor.open_cost(m["group"]) + self._resting_cost(m["group"]),
                                     day_pnl)
                except RiskBlocked as e:
                    self.log("blocked", now, ticker=tk, reason=str(e))
                    continue
                o = VirtualOrder(tk, m["group"], w.side, w.price, w.count, now, w.improved)
                self.executor.place(o)
                keep.append(o)
                self.log("quote", now, ticker=tk, side=w.side, price=w.price, count=w.count,
                         q=round(w.q, 4), improved=w.improved)
            for o in resting:
                if o not in keep:
                    o.cancelled = True
                    self.log("cancel", now, ticker=tk, side=o.side, price=o.price, reason="requote")
            self.executor._gc()

    def equity_marked(self) -> float:
        """Cash + open bets valued at the best bid we could sell into (cost if no live mark)."""
        val = 0.0
        for tk, lst in self.executor.positions.items():
            mk = self.marks.get(tk) or {}
            for side, price, n, _ev in lst:
                val += n * (mk.get(side) if mk.get(side) is not None else price) - n * price
        return self.equity() + val

    def _snapshot(self, now: float, every: float = 60.0):
        if now - self.last_snap < every:
            return
        self.last_snap = now
        with open(self.out / "equity.jsonl", "a") as f:
            f.write(json.dumps({"ts": round(now), "equity": round(self.equity_marked(), 4),
                                "realized": round(self.equity(), 4)}) + "\n")

    def _exit_dead(self, tk: str, m: dict, bk, now: float):
        """The one sell the bot makes: when published data (station readings, TSA days already
        counted) proves our side has lost, holding is worth $0, so any bid above 1c is free money.
        Everything else is held to settlement: the edge was priced at entry, and stop-losses on
        binary bets sell good bets cheap after a scary move."""
        view = self._wx_view(tk, now)
        if not view.get("dead"):
            view = next((v for v in (sig.view(m, now) for sig in self.signals) if v.get("dead")), {})
        dead = view.get("dead")
        held = sum(n for s, _p, n, _e in self.executor.positions.get(tk, []) if s == dead) if dead else 0.0
        bids = (bk.yes_bids if dead == "yes" else bk.no_bids) if dead else []
        if held <= 1e-9 or not bids or bids[0][0] <= 0.01:
            return
        px, size = bids[0]
        n = round(min(held, size), 2)
        fee = taker_fee(px, n)
        if n <= 0 or n * px - fee <= 0:
            return
        pnl = self.executor.sell(tk, dead, px, n, fee)
        rec = dict(ticker=tk, side=dead, contracts=n, price=px, fee=round(fee, 4), pnl=round(pnl, 4),
                   why=view.get("why", "station readings decided the market"))
        self.log("exit", now, **rec)
        self.recent_fills = ([dict(rec, sold=True)] + self.recent_fills)[:50]
        self.notify(f"Sold {tk} {dead.upper()} {n:.2f} @ {px:.2f}: already lost per live data, "
                    f"saved ${n * px - fee:.2f}", priority="default")

    def _mark(self, tk: str, bk, now: float):
        self.marks[tk] = dict(yes=bk.yes_bids[0][0] if bk.yes_bids else None,
                              no=bk.no_bids[0][0] if bk.no_bids else None, ts=now)

    def _veto(self, tk: str, wants: list, v: dict, source: str, now: float) -> list:
        """Signals were not in the backtest, so they may only remove a bid, never add one, and
        only when listed in veto_enforce. Otherwise the disagreement is logged for evaluation."""
        keep = []
        enforce = source in self.cfg.veto_enforce
        for w in wants:
            p_side = v["p_yes"] if w.side == "yes" else 1.0 - v["p_yes"]
            if p_side >= w.price - self.cfg.veto_gap or not enforce:
                keep.append(w)
            if p_side >= w.price - self.cfg.veto_gap:
                continue
            key = (tk, w.side)
            if now - self.vetoed.get(key, -1e18) > 3600:      # log each veto at most hourly
                self.vetoed[key] = now
                rec = dict(ticker=tk, side=w.side, price=w.price, p=round(p_side, 3), source=source,
                           why=v.get("why", ""), ts=now, enforced=enforce)
                self.vetoes = ([rec] + [x for x in self.vetoes if (x["ticker"], x["side"]) != key])[:30]
                self.log("veto" if enforce else "disagree", now, **rec)
        return keep

    def _wx_view(self, tk: str, now: float) -> dict:
        m = self.markets.get(tk, {})
        ser = m.get("series")
        if self.weather is None or ser not in wx.STATIONS:
            return {}
        ymd = wx.market_date(m.get("event", ""))
        if not ymd:
            return {}
        st, off = wx.STATIONS[ser]
        a, b = wx.climate_day(ymd, off)
        if now < a:
            return {}
        if ser in wx.LOW_SERIES:
            # a reading at or below a strike can't be undone, so the lock holds the moment it happens
            return wx.settle_view_low(m.get("strike_type"), m.get("floor"), m.get("cap"),
                                      self.weather.min_f(st, a, b))
        return wx.settle_view(m.get("strike_type"), m.get("floor"), m.get("cap"),
                              self.weather.max_f(st, a, b), margin=1.0)

    def feed_status(self, now: float) -> list:
        out = [dict(name="Kalshi markets + order books", ok=self.errors == 0 and self.last_ok is not None,
                    detail=f"{len(self.markets)} markets", age=None if self.last_ok is None else now - self.last_ok)]
        if self.news is not None:
            for src in self.news.sources:
                items = [r for r in self.news.recent if r.get("source") == src.name]
                err = self.news.errors.get(src.name)
                enabled = getattr(src, "enabled", bool(getattr(src, "bearer", True)))
                out.append(dict(name={"alpaca": "Benzinga news (Alpaca)", "finnhub": "Finnhub news",
                                      "x": "X posts"}.get(src.name, getattr(src, "label", src.name)),
                                ok=enabled and not err, detail=err or ("off" if not enabled else f"{len(items)} recent"),
                                age=(now - max(float(r.get("ts", 0)) for r in items)) if items else None))
        if self.weather is not None:
            shown = set()
            for ser, (st, _off) in wx.STATIONS.items():
                if ser not in self.cfg.series or st in shown:
                    continue
                shown.add(st)
                lt = self.weather.latest(st)
                out.append(dict(name=f"Weather station {st}", ok=lt is not None and now - lt[0] < 7200,
                                detail=f"{lt[1]:.0f}°F" if lt else self.weather.errors.get(st, "waiting"),
                                age=(now - lt[0]) if lt else None))
        for sig in self.signals:
            try:
                out += sig.status(now)
            except Exception as e:
                out.append(dict(name=sig.name, ok=False, detail=str(e)[:80], age=None))
        for w in self.spots:
            out.append(dict(name=f"Spot {w.product}", ok=w.age(now) < 120,
                            detail=f"{w.last_px}" if w.last_px else "waiting", age=w.age(now)))
        return out

    def _resting_cost(self, event: Optional[str] = None) -> float:
        return sum(o.price * (o.count - o.filled) for lst in self.executor.orders.values() for o in lst
                   if not o.cancelled and (event is None or o.event == event))

    def _cancel(self, tk: str, now: float, reason: str):
        if self.executor.resting(tk):
            self.executor.cancel_all(tk)
            self.log("cancel", now, ticker=tk, reason=reason)

    def _write_state(self, now: float):
        st = dict(
            ts=now, written=time.time(), mode="shadow", label=self.cfg.label, start_equity=self.cfg.start_equity, equity=round(self.equity(), 4),
            equity_marked=round(self.equity_marked(), 4),
            realized=round(self.executor.realized, 4), open_cost=round(self.executor.open_cost(), 4),
            markets_watched=len(self.markets),
            resting=[dict(ticker=o.ticker, side=o.side, price=o.price, count=round(o.count - o.filled, 2))
                     for lst in self.executor.orders.values() for o in lst],
            positions=[dict(ticker=tk, side=s, price=p, contracts=round(n, 2))
                       for tk, lst in self.executor.positions.items() for (s, p, n, _e) in lst],
            fills=self.recent_fills[:20],
            breakers=self.breakers.active(now),
            halted=self.guard.halted,
            news=list(self.news.recent)[-15:] if self.news is not None else [],
            data_feeds=self.feed_status(now),
            locks=sorted(self.locked_seen)[-20:],
            vetoes=self.vetoes[:15],
            marks={tk: v for tk, v in self.marks.items() if tk in self.executor.positions or tk in self.executor.orders},
            catalog={tk: dict(title=m.get("title", ""), sub=m.get("sub", ""), series=m.get("series", ""),
                              event=m.get("event", ""), close_ts=m.get("close_ts"), category=m.get("category", ""))
                     for tk, m in self.markets.items()},
            calendar=[dict(ts=t, kind=k) for sig in self.signals if hasattr(sig, "upcoming")
                      for (t, k) in sig.upcoming(now, 6)],
            feeds=dict(news_errors=getattr(self.news, "errors", {}),
                       spots={w.product: dict(px=w.last_px, age=w.age(now)) for w in self.spots}),
        )
        self.executor.save(self.book_path)
        tmp = self.out / "state.json.tmp"
        tmp.write_text(json.dumps(st, default=str))
        tmp.replace(self.out / "state.json")
