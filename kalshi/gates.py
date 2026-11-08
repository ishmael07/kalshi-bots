"""The statistical gates from 'One-Person Hedge Fund' (Table III).

A strategy is called profitable only if it passes all of them:
  n >= 30 trades; positive out-of-sample after costs; deflated Sharpe >= 0.95;
  >= 70% walk-forward windows positive; latest period positive; neighbouring
  parameters within 20%; survives a 2x-cost / 1-step-delay noise test.
"""
from __future__ import annotations

import math
from statistics import NormalDist

_N = NormalDist()
EULER = 0.5772156649


def _moments(x):
    n = len(x)
    m = sum(x) / n
    var = sum((v - m) ** 2 for v in x) / (n - 1) if n > 1 else 0.0
    sd = math.sqrt(var)
    if sd == 0:
        return m, 0.0, 0.0, 3.0
    skew = sum(((v - m) / sd) ** 3 for v in x) / n
    kurt = sum(((v - m) / sd) ** 4 for v in x) / n
    return m, sd, skew, kurt


def deflated_sharpe(returns, n_trials: int) -> float:
    """Bailey & Lopez de Prado DSR on per-period returns (not annualised)."""
    T = len(returns)
    if T < 3:
        return 0.0
    m, sd, g3, g4 = _moments(returns)
    if sd == 0:
        return 1.0 if m > 0 else 0.0
    sr = m / sd
    n = max(int(n_trials), 1)
    if n == 1:
        sr0 = 0.0
    else:
        sr0 = math.sqrt(1.0 / (T - 1)) * ((1 - EULER) * _N.inv_cdf(1 - 1.0 / n)
                                          + EULER * _N.inv_cdf(1 - 1.0 / (n * math.e)))
    denom = 1 - g3 * sr + (g4 - 1) / 4.0 * sr * sr
    if denom <= 0:
        denom = 1e-9
    return _N.cdf((sr - sr0) * math.sqrt(T - 1) / math.sqrt(denom))


def gate_report(weekly_returns, n_trades, n_trials, neighbor_returns, base_return,
                stressed_return, latest_weeks: int = 4) -> dict:
    wk = list(weekly_returns)
    pos = sum(1 for r in wk if r > 0) / len(wk) if wk else 0.0
    total = sum(wk)
    latest = sum(wk[-latest_weeks:]) if wk else 0.0
    dsr = deflated_sharpe(wk, n_trials)
    if base_return > 0 and neighbor_returns:
        plateau = min(neighbor_returns) >= 0.8 * base_return
        plat_val = min(neighbor_returns) / base_return
    else:
        plateau, plat_val = False, 0.0
    g = {
        "trade_count": {"value": n_trades, "pass": n_trades >= 30},
        "oos_positive": {"value": total, "pass": total > 0},
        "deflated_sharpe": {"value": dsr, "pass": dsr >= 0.95},
        "walk_forward": {"value": pos, "pass": pos >= 0.70},
        "latest_period": {"value": latest, "pass": latest > 0},
        "parameter_plateau": {"value": plat_val, "pass": plateau},
        "noise_test": {"value": stressed_return, "pass": stressed_return > 0},
    }
    g["all_pass"] = all(v["pass"] for v in g.values())
    return g
