"""Scalper: a two-sided, in-and-out paper account across every active Kalshi market.

Each cycle it takes the busiest open markets with a spread worth capturing and rests a YES bid and a NO bid at the best
prices (buying NO at the NO bid is selling YES at the ask). The first fill opens a trade; the other side stays up as the
exit. YES + NO in one market always pays $1, so the pair is a locked profit of the spread, less fees. A trade that can't
get out passively within `max_hold_s` (or before the market is about to resolve) is sold into the bid, paying the taker fee.

Replay of stored tapes at a 60s reaction (`kalshi scalpcheck`) lost money in every group but econ/TSA, and it had to guess
quotes from recent prints. This runs on the real book at about 10s, as a forward paper test. Virtual money only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .bot import Bot, BotConfig
from .store import iso_ts
from .fees import maker_fee, taker_fee
from .live import Breakers, CellModel, ShadowExecutor, VirtualOrder

_EPS = 1e-9


@dataclass
class ScalpParams:
    top_n: int = 12                  # markets quoted at once
    frac: float = 0.10               # equity per trade
    open_cap_frac: float = 1.0       # money tied up, open trades plus resting entries
    min_spread: float = 0.02
    min_volume: float = 500.0        # contracts traded in the last 24h
    min_px: float = 0.08
    max_px: float = 0.92
    max_hold_s: float = 900.0        # then sell into the bid
    stop_before_s: float = 600.0     # flat this long before the market is expected to resolve


class ScalpExecutor(ShadowExecutor):
    def __init__(self, fill_share: float = 0.25):
        super().__init__(fill_share)
        self.trips: dict = {}        # ticker -> {"id": first fill ts, "since": first fill ts}

    def on_trades(self, ticker: str, trades: list) -> list:
        """Price-time priority: a print at our price first eats the queue that was ahead of us when we joined;
        only what's left reaches us. A print through our price means the whole level is gone."""
        fills = []
        for t in sorted(trades, key=lambda x: x["ts"]):
            left = t["count"]
            for o in self.resting(ticker):
                if t["ts"] <= o.placed_ts or left <= _EPS:
                    continue
                if t["taker_side"] != ("no" if o.side == "yes" else "yes"):
                    continue
                px = t["yes_price"] if o.side == "yes" else 1.0 - t["yes_price"]
                if px > o.price + _EPS:
                    continue
                if px < o.price - _EPS:
                    o.ahead = 0.0
                used = min(getattr(o, "ahead", 0.0), left)
                o.ahead = getattr(o, "ahead", 0.0) - used
                left -= used
                n = min(left, o.count - o.filled)
                if n <= _EPS:
                    continue
                left -= n
                o.filled += n
                self.positions.setdefault(ticker, []).append((o.side, o.price, n, o.event))
                fills.append((o, n, t["ts"]))
        self._gc()
        return fills

    def held(self, ticker: str) -> tuple:
        """(side, contracts) of the net position, (None, 0.0) if flat."""
        y = sum(n for s, _p, n, _e in self.positions.get(ticker, []) if s == "yes")
        no = sum(n for s, _p, n, _e in self.positions.get(ticker, []) if s == "no")
        if y > no + _EPS:
            return "yes", y - no
        if no > y + _EPS:
            return "no", no - y
        return None, 0.0

    def net(self, ticker: str) -> float:
        """YES + NO pays $1 whatever happens: close matched lots, oldest first. Returns the locked gain."""
        lots = [list(x) for x in self.positions.get(ticker, [])]
        gain = 0.0
        while True:
            y = next((x for x in lots if x[0] == "yes" and x[2] > _EPS), None)
            no = next((x for x in lots if x[0] == "no" and x[2] > _EPS), None)
            if y is None or no is None:
                break
            q = min(y[2], no[2])
            gain += q * (1.0 - y[1] - no[1])
            y[2] -= q
            no[2] -= q
        keep = [tuple(x) for x in lots if x[2] > _EPS]
        if keep:
            self.positions[ticker] = keep
        else:
            self.positions.pop(ticker, None)
        self.realized += gain
        return gain

    def settle(self, ticker: str, result: str) -> float:
        self.trips.pop(ticker, None)
        return super().settle(ticker, result)

    def save(self, path):
        path = Path(path)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"positions": self.positions, "realized": self.realized, "trips": self.trips}))
        tmp.replace(path)

    def load(self, path):
        super().load(path)
        if Path(path).exists():
            self.trips = json.load(open(path)).get("trips", {})


class ScalpBot(Bot):
    def __init__(self, cfg: BotConfig, public, executor: ScalpExecutor, params: ScalpParams = None, notify=None):
        super().__init__(cfg, public, CellModel(), executor, Breakers(), notify=notify)
        self.p = params or ScalpParams()

    def _safe_category(self, series: str) -> str:
        try:
            return self._category(series)
        except Exception:
            self.categories[series] = "Other"
            return "Other"

    def discover(self, now: float):
        p, cand = self.p, []
        for m in self.pub.paginate("/markets", "markets", {"status": "open", "mve_filter": "exclude"}):
            try:
                vol, bid, ask = (float(m.get(k) or 0) for k in ("volume_24h_fp", "yes_bid_dollars", "yes_ask_dollars"))
                close = iso_ts(m["close_time"])
                end = min(close, iso_ts(m["expected_expiration_time"])) if m.get("expected_expiration_time") else close
            except (ValueError, KeyError):
                continue
            if vol < p.min_volume or ask - bid < p.min_spread - _EPS or bid < p.min_px or ask > p.max_px:
                continue
            if end - now < p.stop_before_s + 600:
                continue
            cand.append((vol, m, close, end))
        cand.sort(key=lambda c: -c[0])
        found, events = {}, set()
        for vol, m, close, end in cand:              # one market per event: strikes of one event are one bet
            ev = m.get("event_ticker", m["ticker"])
            if ev in events:
                continue
            events.add(ev)
            s = ev.split("-")[0]
            found[m["ticker"]] = dict(event=ev, category=self._safe_category(s), close_ts=close, series=s, group=ev,
                                      stop_ts=end - p.stop_before_s, volume=vol,
                                      tick=0.01 if m.get("price_level_structure", "linear_cent") == "linear_cent" else 0.001,
                                      title=m.get("title", ""), sub=m.get("yes_sub_title", ""))
            if len(found) >= p.top_n:
                break
        for tk in list(self.executor.positions) + list(self.executor.orders):
            if tk in self.markets and tk not in found:
                found[tk] = self.markets[tk]
        self.markets = found
        self.last_discover = now

    # ---------------------------------------------------------------- fills
    def _tape(self, now: float):
        ex = self.executor
        for tk in list(ex.orders):
            since = self.tape_cursor.get(tk, now)
            trades = []
            for t in self.pub.paginate("/markets/trades", "trades", {"ticker": tk, "min_ts": int(since)}):
                ts = iso_ts(t["created_time"])
                if ts > since:
                    trades.append(dict(ts=ts, yes_price=float(t["yes_price_dollars"]), count=float(t["count_fp"]),
                                       taker_side=t["taker_side"]))
            if trades:
                self.tape_cursor[tk] = max(t["ts"] for t in trades)
            side, n = ex.held(tk)
            net = n if side == "yes" else -n
            for o, k, ts in ex.on_trades(tk, trades):
                net = self._record(tk, o, k, ts, net, now)
            ex.net(tk)

    def _record(self, tk: str, o, k: float, ts: float, net: float, now: float) -> float:
        """A fill against the position closes the trade; anything beyond that opens a new one."""
        ex, sgn = self.executor, 1.0 if o.side == "yes" else -1.0
        closing = min(abs(net), k) if net * sgn < 0 else 0.0
        opening = k - closing
        if closing > _EPS:
            fee = maker_fee(o.price, closing)
            ex.realized -= fee
            self.log("exit", now, ticker=tk, side="yes" if net > 0 else "no", price=round(1.0 - o.price, 4),
                     contracts=round(closing, 6), fee=fee, reason="scalp", trip=ex.trips.get(tk, {}).get("id"))
            net += sgn * closing
            if abs(net) < _EPS:
                ex.trips.pop(tk, None)
        if opening > _EPS:
            ex.trips.setdefault(tk, dict(id=int(ts), since=ts))
            fee = maker_fee(o.price, opening)
            ex.realized -= fee
            rec = dict(ticker=tk, side=o.side, price=o.price, contracts=round(opening, 6), fill_ts=ts)
            self.log("fill", now, fee=fee, trip=ex.trips[tk]["id"], **rec)
            self.recent_fills = ([rec] + self.recent_fills)[:50]
            net += sgn * opening
        return net

    # ---------------------------------------------------------------- quotes
    def _reserved(self, exclude: str) -> float:
        """Open cost plus the larger resting entry per market: only one side of a two-sided quote can tie up money."""
        ex, total = self.executor, self.executor.open_cost()
        for tk in ex.orders:
            if tk != exclude and ex.held(tk)[0] is None:
                total += max((o.price * (o.count - o.filled) for o in ex.resting(tk)), default=0.0)
        return total

    def _flatten(self, tk: str, bk, side: str, n: float, now: float, reason: str):
        bids = bk.yes_bids if side == "yes" else bk.no_bids
        if not bids or bids[0][0] <= 0.01:
            return
        ex = self.executor
        px, size = bids[0]
        q = min(n, size)
        fee = taker_fee(px, q)
        trip = ex.trips.get(tk, {}).get("id")
        pnl = ex.sell(tk, side, px, q, fee)
        self.log("exit", now, ticker=tk, side=side, price=px, contracts=round(q, 6), fee=fee, pnl=round(pnl, 4),
                 reason=reason, trip=trip)
        if ex.held(tk)[0] is None:
            ex.trips.pop(tk, None)

    def _wants(self, bk, side, n: float, tk: str) -> list:
        """[(side, price, count, queue ahead)]. Holding: the exit, a tick ahead of the line if that doesn't cross.
        Flat: both sides, a tick ahead of the line only if the spread still pays for the fees after giving up two ticks."""
        p = self.p
        if not bk.yes_bids or not bk.no_bids:
            return []
        (yb, ys), (nb, ns) = bk.yes_bids[0], bk.no_bids[0]
        spread, tick = (1.0 - nb) - yb, bk.tick
        if side:
            s, px, size = ("no", nb, ns) if side == "yes" else ("yes", yb, ys)
            return [(s, round(px + tick, 4), n, 0.0) if spread >= 2 * tick - _EPS else (s, px, n, size)]
        if spread < p.min_spread - _EPS or yb < p.min_px or 1.0 - nb > p.max_px:
            return []
        if spread >= 4 * tick - _EPS:
            ypx, npx, yq, nq = round(yb + tick, 4), round(nb + tick, 4), 0.0, 0.0
        else:
            ypx, npx, yq, nq = yb, nb, ys, ns
        eq = self.equity()
        top = max(ypx, npx)
        count = int(p.frac * eq / top)               # same size both sides, so a full fill pairs off
        if count < 1 or self._reserved(tk) + count * top > p.open_cap_frac * eq + _EPS:
            return []
        return [("yes", ypx, float(count), yq), ("no", npx, float(count), nq)]

    def _quote(self, now: float):
        ex = self.executor
        for tk in sorted(self.markets):
            m = self.markets[tk]
            if now >= m["close_ts"]:
                self._cancel(tk, now, "closed")
                continue
            bk = self.book(tk)
            self._mark(tk, bk, now)
            side, n = ex.held(tk)
            if side:
                trip = ex.trips.setdefault(tk, dict(id=int(now), since=now))
                out_of_time = now >= m.get("stop_ts", m["close_ts"])
                if out_of_time or now - trip["since"] >= self.p.max_hold_s:
                    self._cancel(tk, now, "time stop")
                    self._flatten(tk, bk, side, n, now, "resolving" if out_of_time else "time stop")
                    continue
            elif now >= m.get("stop_ts", m["close_ts"]):
                self._cancel(tk, now, "resolving")
                continue
            wants = self._wants(bk, side, n, tk)
            resting, keep = ex.resting(tk), []
            if not resting:
                self.tape_cursor[tk] = max(self.tape_cursor.get(tk, 0.0), now)
            for s, px, count, ahead in wants:
                same = [o for o in resting if o.side == s and abs(o.price - px) < _EPS
                        and abs((o.count - o.filled) - count) < 1e-6]
                if same:
                    keep += same
                    continue
                o = VirtualOrder(tk, m["group"], s, px, count, now, False)
                o.ahead = ahead
                ex.place(o)
                keep.append(o)
            for o in resting:
                if o not in keep:
                    o.cancelled = True
            ex._gc()
