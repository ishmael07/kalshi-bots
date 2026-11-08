"""SQLite store for Kalshi series, settled markets and trades."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator


def iso_ts(s: str) -> float:
    if not s:
        return 0.0
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


SCHEMA = """
CREATE TABLE IF NOT EXISTS series(
  ticker TEXT PRIMARY KEY, category TEXT, frequency TEXT,
  fee_type TEXT, fee_mult REAL, title TEXT);
CREATE TABLE IF NOT EXISTS markets(
  id INTEGER PRIMARY KEY, ticker TEXT UNIQUE, event TEXT, series TEXT, open_ts REAL, close_ts REAL,
  result TEXT, strike_type TEXT, floor_strike REAL, cap_strike REAL,
  volume REAL, expiration_value REAL, title TEXT, subtitle TEXT);
CREATE INDEX IF NOT EXISTS markets_series ON markets(series, close_ts);
CREATE TABLE IF NOT EXISTS trades(
  mid INTEGER, ts REAL, yes_price REAL, count REAL, taker_yes INTEGER);
CREATE INDEX IF NOT EXISTS trades_mid ON trades(mid, ts);
CREATE TABLE IF NOT EXISTS fetched(ticker TEXT PRIMARY KEY, n INTEGER);
"""


def _retry_locked(fn):
    """Several collectors share one DB; ride out long lock waits instead of dying."""
    import functools
    import time as _t

    @functools.wraps(fn)
    def wrap(self, *a, **kw):
        for attempt in range(8):
            try:
                return fn(self, *a, **kw)
            except sqlite3.OperationalError as e:
                if "locked" not in str(e) or attempt == 7:
                    raise
                try:
                    self.db.rollback()
                except Exception:
                    pass
                _t.sleep(5 * (attempt + 1))
    return wrap


class Store:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), timeout=600)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    @_retry_locked
    def upsert_series(self, rows: Iterable[dict]):
        self.db.executemany(
            "INSERT OR REPLACE INTO series VALUES(?,?,?,?,?,?)",
            [(r["ticker"], r.get("category"), r.get("frequency"), r.get("fee_type"),
              _f(r.get("fee_multiplier"), 1.0), r.get("title")) for r in rows])
        self.db.commit()

    @_retry_locked
    def upsert_markets(self, series: str, rows: Iterable[dict]):
        self.db.executemany(
            "INSERT INTO markets(ticker,event,series,open_ts,close_ts,result,strike_type,"
            "floor_strike,cap_strike,volume,expiration_value,title,subtitle) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(ticker) DO UPDATE SET "
            "result=excluded.result, volume=excluded.volume, "
            "expiration_value=excluded.expiration_value",
            [(m["ticker"], m.get("event_ticker"), series, iso_ts(m.get("open_time", "")),
              iso_ts(m.get("close_time", "")), m.get("result") or "", m.get("strike_type"),
              _f(m.get("floor_strike")), _f(m.get("cap_strike")),
              _f(m.get("volume_fp"), 0.0), _f(m.get("expiration_value")),
              m.get("title"), m.get("yes_sub_title")) for m in rows])
        self.db.commit()

    def _mid(self, ticker: str) -> int:
        r = self.db.execute("SELECT id FROM markets WHERE ticker=?", (ticker,)).fetchone()
        if r is None:
            raise KeyError(f"market {ticker} not stored")
        return r[0]

    @_retry_locked
    def clear_trades(self, ticker: str):
        self.db.execute("DELETE FROM trades WHERE mid=?", (self._mid(ticker),))

    @_retry_locked
    def insert_trades(self, rows: Iterable[dict]) -> int:
        """Rows from one or more markets (already deduped by the API client)."""
        cache: dict = {}
        vals = []
        for t in rows:
            tk = t["ticker"]
            if tk not in cache:
                cache[tk] = self._mid(tk)
            vals.append((cache[tk], iso_ts(t["created_time"]), _f(t.get("yes_price_dollars")),
                         _f(t.get("count_fp"), 0.0), 1 if t.get("taker_side") == "yes" else 0))
        self.db.executemany("INSERT INTO trades VALUES(?,?,?,?,?)", vals)
        self.db.commit()
        return len(vals)

    @_retry_locked
    def mark_fetched(self, ticker: str, n: int = 0):
        self.db.execute("INSERT OR REPLACE INTO fetched VALUES(?,?)", (ticker, n))
        self.db.commit()

    def is_fetched(self, ticker: str) -> bool:
        return self.db.execute("SELECT 1 FROM fetched WHERE ticker=?", (ticker,)).fetchone() is not None

    def markets_for(self, series: str) -> list:
        return self.db.execute(
            "SELECT * FROM markets WHERE series=? ORDER BY close_ts", (series,)).fetchall()

    def trade_rows(self, where: str = "", params: tuple = ()) -> Iterator[dict]:
        q = ("SELECT m.ticker, t.ts, t.yes_price, t.count, "
             "CASE t.taker_yes WHEN 1 THEN 'yes' ELSE 'no' END AS taker_side, "
             "m.event, m.series, m.result, m.open_ts, m.close_ts, s.category, s.frequency, "
             "s.fee_type, s.fee_mult FROM trades t JOIN markets m ON m.id=t.mid "
             "JOIN series s ON s.ticker=m.series JOIN fetched f ON f.ticker=m.ticker ")
        if where:
            q += "WHERE " + where
        for r in self.db.execute(q, params):
            yield dict(r)
