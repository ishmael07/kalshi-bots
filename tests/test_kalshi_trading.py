import base64
import json

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa

from kalshi.trading import KalshiTrader, RiskGuard, RiskBlocked, order_body, LiveNotAllowed
from kalshi.api import DEMO_BASE, BASE


def _rsa_pem(tmp_path):
    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    p = tmp_path / "rsa.pem"
    p.write_bytes(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                  serialization.NoEncryption()))
    return k, p


def _ed_pem(tmp_path):
    k = ed25519.Ed25519PrivateKey.generate()
    p = tmp_path / "ed.pem"
    p.write_bytes(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                  serialization.NoEncryption()))
    return k, p


def test_rsa_signature_covers_ts_method_full_path_without_query(tmp_path):
    k, p = _rsa_pem(tmp_path)
    t = KalshiTrader("KEY", p, clock_ms=lambda: 1700000000123)
    h = t.headers("GET", "/portfolio/fills?ticker=X&limit=5")
    assert h["KALSHI-ACCESS-KEY"] == "KEY"
    assert h["KALSHI-ACCESS-TIMESTAMP"] == "1700000000123"
    msg = b"1700000000123GET/trade-api/v2/portfolio/fills"
    k.public_key().verify(base64.b64decode(h["KALSHI-ACCESS-SIGNATURE"]), msg,
                          padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
                          hashes.SHA256())


def test_ed25519_signature(tmp_path):
    k, p = _ed_pem(tmp_path)
    t = KalshiTrader("KEY", p, clock_ms=lambda: 5)
    h = t.headers("POST", "/portfolio/events/orders")
    k.public_key().verify(base64.b64decode(h["KALSHI-ACCESS-SIGNATURE"]),
                          b"5POST/trade-api/v2/portfolio/events/orders")


def test_ws_headers_sign_ws_path(tmp_path):
    k, p = _ed_pem(tmp_path)
    t = KalshiTrader("KEY", p, clock_ms=lambda: 7)
    h = t.ws_headers()
    k.public_key().verify(base64.b64decode(h["KALSHI-ACCESS-SIGNATURE"]), b"7GET/trade-api/ws/v2")


def test_order_body_buy_yes_post_only():
    b = order_body("T", "yes", 10, 0.56, client_order_id="c1")
    assert b == {"ticker": "T", "side": "bid", "count": "10.00", "price": "0.5600",
                 "time_in_force": "good_till_canceled", "self_trade_prevention_type": "maker",
                 "post_only": True, "client_order_id": "c1"}


def test_order_body_buy_no_is_ask_on_yes_at_complement():
    b = order_body("T", "no", 2.5, 0.80, client_order_id="c2")
    assert b["side"] == "ask" and b["price"] == "0.2000" and b["count"] == "2.50"


def test_order_body_rejects_bad_price_and_count():
    with pytest.raises(ValueError):
        order_body("T", "yes", 1, 1.0)
    with pytest.raises(ValueError):
        order_body("T", "yes", 0.001, 0.5)


def test_defaults_to_demo_and_live_needs_explicit_unlock(tmp_path, monkeypatch):
    _, p = _ed_pem(tmp_path)
    assert KalshiTrader("K", p).base == DEMO_BASE
    monkeypatch.delenv("KALSHI_LIVE_OK", raising=False)
    with pytest.raises(LiveNotAllowed):
        KalshiTrader("K", p, live=True, gates_file=tmp_path / "nope.json")
    gf = tmp_path / "gates.json"
    gf.write_text(json.dumps({"all_pass": True, "approved_by_user": True}))
    with pytest.raises(LiveNotAllowed):      # still needs the env switch
        KalshiTrader("K", p, live=True, gates_file=gf)
    monkeypatch.setenv("KALSHI_LIVE_OK", "1")
    assert KalshiTrader("K", p, live=True, gates_file=gf).base == BASE


def test_risk_guard_caps():
    g = RiskGuard(max_order_dollars=5.0, max_open_dollars=20.0, max_event_dollars=8.0,
                  daily_loss_halt=2.0)
    g.check("E1", cost=4.0, open_dollars=0.0, event_dollars=0.0, day_pnl=0.0)
    with pytest.raises(RiskBlocked):
        g.check("E1", cost=6.0, open_dollars=0.0, event_dollars=0.0, day_pnl=0.0)
    with pytest.raises(RiskBlocked):
        g.check("E1", cost=4.0, open_dollars=18.0, event_dollars=0.0, day_pnl=0.0)
    with pytest.raises(RiskBlocked):
        g.check("E1", cost=4.0, open_dollars=0.0, event_dollars=5.0, day_pnl=0.0)
    with pytest.raises(RiskBlocked):
        g.check("E1", cost=1.0, open_dollars=0.0, event_dollars=0.0, day_pnl=-2.5)
    g.halt("kill switch")
    with pytest.raises(RiskBlocked):
        g.check("E2", cost=1.0, open_dollars=0.0, event_dollars=0.0, day_pnl=0.0)
