from datetime import datetime, timezone

import pytest

from kalshi.weather import (STATIONS, climate_day, market_date, settle_view, WeatherObs, f_from_c)


def test_station_map_covers_all_cities():
    assert STATIONS["KXHIGHNY"][0] == "KNYC" and STATIONS["KXHIGHCHI"][0] == "KMDW"
    assert set(STATIONS) >= {"KXHIGHNY", "KXHIGHCHI", "KXHIGHMIA", "KXHIGHAUS", "KXHIGHLAX",
                             "KXHIGHDEN", "KXHIGHPHIL"}


def test_market_date_from_event_ticker():
    assert market_date("KXHIGHNY-26NOV07") == (2026, 11, 7)
    assert market_date("KXHIGHCHI-26JAN31") == (2026, 1, 31)
    assert market_date("garbage") is None


def test_climate_day_is_local_standard_midnight_to_midnight():
    a, b = climate_day((2026, 11, 7), -5)
    assert datetime.fromtimestamp(a, timezone.utc) == datetime(2026, 11, 7, 5, tzinfo=timezone.utc)
    assert b - a == 86400


def test_greater_strike_locks_yes_once_observed_max_clears_it():
    # "high > 56" and Midway already reached 58F -> YES certain, NO dead
    v = settle_view("greater", 56.0, None, max_so_far=58.0, margin=1.0)
    assert v == {"locked": "yes", "dead": "no"}
    assert settle_view("greater", 56.0, None, max_so_far=56.5, margin=1.0) == {}


def test_less_strike_dies_once_max_exceeds_cap():
    v = settle_view("less", None, 50.0, max_so_far=52.0, margin=1.0)
    assert v == {"locked": "no", "dead": "yes"}


def test_between_bucket_dies_once_max_passes_top():
    assert settle_view("between", 60.0, 61.0, max_so_far=63.0, margin=1.0) == {"locked": "no", "dead": "yes"}
    assert settle_view("between", 60.0, 61.0, max_so_far=61.5, margin=1.0) == {}


def test_weather_obs_running_max_within_climate_day():
    class S:
        def get(self, url, headers=None, params=None, timeout=None):
            class R:
                status_code = 200

                def json(self_inner):
                    return {"features": [
                        {"properties": {"timestamp": "2026-11-07T20:53:00+00:00", "temperature": {"value": 14.4}}},
                        {"properties": {"timestamp": "2026-11-07T18:53:00+00:00", "temperature": {"value": 15.0}}},
                        {"properties": {"timestamp": "2026-11-07T03:53:00+00:00", "temperature": {"value": 20.0}}},
                        {"properties": {"timestamp": "2026-11-07T19:53:00+00:00", "temperature": {"value": None}}},
                    ]}
                def raise_for_status(self_inner):
                    pass
            return R()
    w = WeatherObs(session=S(), min_interval_s=0)
    w.poll(now=datetime(2026, 11, 7, 21, tzinfo=timezone.utc).timestamp(), series=["KXHIGHNY"])
    a, b = climate_day((2026, 11, 7), -5)
    # 03:53Z is Nov 6 local (before 05:00Z) -> excluded; max of the day is 15.0C = 59F
    assert w.max_f("KNYC", a, b) == pytest.approx(f_from_c(15.0))
    assert w.last_obs_age("KNYC", now=datetime(2026, 11, 7, 21, tzinfo=timezone.utc).timestamp()) == pytest.approx(7 * 60)


def test_apply_view_vetoes_dead_side_and_bids_locked_side():
    from kalshi.live import Quoter, Book, Want, CellModel, apply_view
    q = Quoter(CellModel(), sizing="fixed", fixed_frac=0.01)
    b = Book("T", "KXHIGHCHI-26NOV07", "Climate and Weather", 1e12,
             yes_bids=[(0.80, 10.0)], no_bids=[(0.12, 5.0)], tick=0.01)
    wants = [Want("T", "E", "no", 0.13, 5.0, 0.2, True)]
    out = apply_view(q, b, wants, {"locked": "yes", "dead": "no"}, equity=100.0)
    assert [w.side for w in out] == ["yes"]
    assert out[0].price == pytest.approx(0.81) and out[0].q == pytest.approx(0.99)
    assert out[0].count * out[0].price == pytest.approx(1.0, rel=0.02)


def test_low_markets_lock_from_the_running_minimum():
    from kalshi.weather import settle_view_low, LOW_SERIES
    assert "KXLOWTMIN" in LOW_SERIES and "KXHIGHNY" not in LOW_SERIES
    # "low 41 or above" (greater 40): already saw 40 -> YES dead
    assert settle_view_low("greater", 40.0, None, 40.0) == {"locked": "no", "dead": "yes"}
    assert settle_view_low("greater", 40.0, None, 40.6) == {}
    # "low 35 or below" (less 36): saw 35 -> YES locked
    assert settle_view_low("less", None, 36.0, 35.0) == {"locked": "yes", "dead": "no"}
    assert settle_view_low("less", None, 36.0, 35.5) == {}
    # band 38-39: saw 37 -> YES dead
    assert settle_view_low("between", 38.0, 39.0, 37.0) == {"locked": "no", "dead": "yes"}
