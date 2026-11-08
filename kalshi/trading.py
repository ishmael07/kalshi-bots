"""Authenticated Kalshi trading client (V2 order API) + hard risk rails.

Signing: base64(sign(timestamp_ms + METHOD + "/trade-api/v2" + path_without_query))
  Ed25519 keys sign the bytes directly; RSA keys use PSS(SHA-256, MGF1-SHA-256, salt=32).
V2 orders are expressed on the YES leg: side "bid" = buy YES, "ask" = sell YES.
Buying NO at price p == selling YES at 1 - p (positions net).

Safety: the client talks to the DEMO exchange unless live=True AND a gates file
says all gates passed and the user approved AND env KALSHI_LIVE_OK=1.
"""
from __future__ import annotations

import base64
import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa

from .api import BASE, DEMO_BASE

API_PREFIX = "/trade-api/v2"
WS_PATH = "/trade-api/ws/v2"
WS_DEMO = "wss://external-api-ws.demo.kalshi.co/trade-api/ws/v2"
WS_PROD = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"


class LiveNotAllowed(RuntimeError):
    pass


class RiskBlocked(RuntimeError):
    pass


def order_body(ticker: str, side: str, count: float, price: float, *, post_only: bool = True,
               client_order_id: Optional[str] = None, expiration_time: Optional[int] = None,
               time_in_force: str = "good_till_canceled") -> dict:
    """side: which contract we BUY ('yes' or 'no'); price: what we pay for it (dollars)."""
    if not (0.0 < price < 1.0):
        raise ValueError(f"price {price} outside (0,1)")
    if count < 0.01:
        raise ValueError(f"count {count} below 0.01 minimum")
    if side == "yes":
        v2_side, yes_px = "bid", price
    elif side == "no":
        v2_side, yes_px = "ask", 1.0 - price
    else:
        raise ValueError(side)
    b = {"ticker": ticker, "side": v2_side, "count": f"{count:.2f}", "price": f"{yes_px:.4f}",
         "time_in_force": time_in_force, "self_trade_prevention_type": "maker",
         "post_only": post_only, "client_order_id": client_order_id or str(uuid.uuid4())}
    if expiration_time:
        b["expiration_time"] = int(expiration_time)
    return b


class KalshiTrader:
    def __init__(self, key_id: str, private_key_path, *, live: bool = False,
                 gates_file="data/gates_passed.json", session=None,
                 clock_ms=lambda: int(time.time() * 1000)):
        if live:
            gf = Path(gates_file)
            ok = False
            if gf.exists():
                g = json.loads(gf.read_text())
                ok = bool(g.get("all_pass")) and bool(g.get("approved_by_user"))
            if not ok:
                raise LiveNotAllowed("gates not passed / not approved by user")
            if os.environ.get("KALSHI_LIVE_OK") != "1":
                raise LiveNotAllowed("set KALSHI_LIVE_OK=1 to trade real money")
        self.base = BASE if live else DEMO_BASE
        self.ws_url = WS_PROD if live else WS_DEMO
        self.live = live
        self.key_id = key_id
        pem = Path(private_key_path).read_bytes()
        self.key = serialization.load_pem_private_key(pem, password=None)
        if not isinstance(self.key, (rsa.RSAPrivateKey, ed25519.Ed25519PrivateKey)):
            raise ValueError("unsupported key type")
        self.s = session or requests.Session()
        self.clock_ms = clock_ms

    # ---- auth ----
    def _sign(self, msg: bytes) -> str:
        if isinstance(self.key, ed25519.Ed25519PrivateKey):
            sig = self.key.sign(msg)
        else:
            sig = self.key.sign(msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
                                hashes.SHA256())
        return base64.b64encode(sig).decode()

    def headers(self, method: str, path: str) -> dict:
        ts = str(self.clock_ms())
        clean = path.split("?", 1)[0]
        return {"KALSHI-ACCESS-KEY": self.key_id, "KALSHI-ACCESS-TIMESTAMP": ts,
                "KALSHI-ACCESS-SIGNATURE": self._sign((ts + method.upper() + API_PREFIX + clean).encode()),
                "Content-Type": "application/json"}

    def ws_headers(self) -> dict:
        ts = str(self.clock_ms())
        return {"KALSHI-ACCESS-KEY": self.key_id, "KALSHI-ACCESS-TIMESTAMP": ts,
                "KALSHI-ACCESS-SIGNATURE": self._sign((ts + "GET" + WS_PATH).encode())}

    def _req(self, method: str, path: str, params=None, body=None) -> dict:
        r = self.s.request(method, self.base + path, params=params, timeout=15,
                           headers=self.headers(method, path),
                           data=json.dumps(body) if body is not None else None)
        if r.status_code >= 300:
            raise RuntimeError(f"{method} {path} -> {r.status_code} {r.text[:300]}")
        return r.json() if r.text else {}

    # ---- account ----
    def balance(self) -> float:
        return float(self._req("GET", "/portfolio/balance").get("balance", 0)) / 100.0

    def positions(self) -> list:
        return self._req("GET", "/portfolio/positions", {"limit": 1000}).get("market_positions", [])

    def fills(self, min_ts: Optional[int] = None) -> list:
        p = {"limit": 1000}
        if min_ts:
            p["min_ts"] = int(min_ts)
        return self._req("GET", "/portfolio/fills", p).get("fills", [])

    # ---- orders ----
    def place(self, ticker: str, side: str, count: float, price: float, **kw) -> dict:
        return self._req("POST", "/portfolio/events/orders", body=order_body(ticker, side, count, price, **kw))

    def cancel(self, order_id: str, market_ticker: str) -> dict:
        return self._req("DELETE", f"/portfolio/events/orders/{order_id}",
                         params={"market_ticker": market_ticker})


class RiskGuard:
    """Code-enforced rails. Every order passes through check() before placement."""

    def __init__(self, max_order_dollars: float, max_open_dollars: float,
                 max_event_dollars: float, daily_loss_halt: float):
        self.max_order = max_order_dollars
        self.max_open = max_open_dollars
        self.max_event = max_event_dollars
        self.daily_loss_halt = daily_loss_halt
        self.halted: Optional[str] = None

    def halt(self, reason: str):
        self.halted = reason

    def resume(self):
        self.halted = None

    def check(self, event: str, cost: float, open_dollars: float, event_dollars: float,
              day_pnl: float):
        if self.halted:
            raise RiskBlocked(f"halted: {self.halted}")
        if day_pnl <= -self.daily_loss_halt:
            self.halted = f"daily loss {day_pnl:.2f}"
            raise RiskBlocked(self.halted)
        if cost > self.max_order:
            raise RiskBlocked(f"order ${cost:.2f} > max ${self.max_order:.2f}")
        if open_dollars + cost > self.max_open:
            raise RiskBlocked(f"open ${open_dollars + cost:.2f} > max ${self.max_open:.2f}")
        if event_dollars + cost > self.max_event:
            raise RiskBlocked(f"event {event} ${event_dollars + cost:.2f} > max ${self.max_event:.2f}")
