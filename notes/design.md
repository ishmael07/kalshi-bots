# Kalshi favorites machine — design (2026-11-06)

## Goal
An always-on bot that trades many short-dated Kalshi markets a day and compounds a
small account, but only after the edge is proven on real history. No money moves
until the gates below pass and the user says so.

## Why Kalshi (evidence)
- Bürgi, Deng & Whelan, GWU WP 2026-001 (Kalshi trades 2021 → Apr 2025):
  overall return per contract about −20%; Makers −9.64% vs Takers −31.46%; contracts
  under 10¢ lose over 60%; Makers buying contracts at 50¢ or more earn +2.6% (SD 33%).
  This is the favorite–longshot bias.
- Kalshi's public API exposes every trade back to 2022 with `taker_side`, so we can
  re-measure the effect **out of sample** (May 2025 → now, after the paper's data ends).

## Phases
1. **Edge study (this branch, no account, no money).** Collect settled markets plus all
   their trades for a stratified sample of fast series (15-min, hourly and daily, across
   crypto, financials, weather and economics; no sports). Compute maker and taker
   returns after the current fee schedule, by price band, category and series,
   with event-clustered bootstrap CIs and week-by-week stability.
2. **Model-priced strategy replay.** For BTC/ETH 15-min and hourly markets, price each
   strike from spot plus realized vol (lognormal, matched to CF RTI's 60-second settlement
   average). For weather markets, price from ensemble forecasts. Simulate maker-only
   bids on the favorite side when model probability exceeds price plus a margin.
   Fills come from real prints that cross our price, with a queue haircut. Run a
   bankroll simulation from $75 at quarter Kelly.
3. **Live paper bot on Kalshi's demo environment**, with live feeds: the Kalshi WS order book,
   Coinbase/Kraken spot, Open-Meteo/NWS ensembles, the econ calendar, Alpaca (Benzinga)
   and Finnhub news, and X (paid API, behind a spend cap; the user must opt in). News is a
   **circuit breaker**: it cancels resting orders on breaking events. It does not pick trades.
4. **Live, tiny**, only after phase 2 and 3 gates pass and the user approves.

## Gates (from the "one-person hedge fund" paper)
n ≥ 30 trades per strategy; positive out-of-sample after costs; deflated Sharpe ≥ 0.95
for the number of variants tried; ≥ 70% of walk-forward weeks positive; latest
period positive; neighbouring parameters within 20%; survives a 2× cost / 1-step delay
noise test. Paper vs. replay consistency z > −2 over ≥ 20 paper trades.

## Risk rules (live)
Quarter Kelly with a per-market cap; correlated markets (same underlying + same
expiry) count as one bet; daily loss of 2% halts; 3 losses in a row pauses; a stale feed
cancels all resting orders. No sports contracts (state litigation).

## Caveats
Historical fees are recomputed with today's schedule (series `fee_type` /
`fee_multiplier`). The schedule changed during the sample, so this is an approximation.
Maker fills observed in history were someone else's queue position, so we
haircut them.

## Data feeds (2026-11-07)
All free. Kalshi books + tape; Benzinga (Alpaca) + Finnhub news; Fed press, BEA, BLS latest-numbers
and Google News RSS; NWS observations at the 7 settlement stations; GEFS + ECMWF ensembles (Open-Meteo,
82 members), NWS hourly forecast and alerts; Cleveland Fed CPI nowcast (m/m, y/y) and GDPNow (FRED);
BLS release calendar + FOMC dates + Kalshi econ close times; AAA national gas average; TSA checkpoint
counts; SPY/QQQ (Alpaca). X is wired but off (paid per read).

Rule: a feed may only remove or pause trades, or bid a side that published numbers already decide
(weather observations, TSA days already counted). Probability opinions are logged as "disagree" and
only block bids for signals listed in `bot.veto_enforce` (empty by default), because the stored data says:
- Weather forecast veto (`kalshi wxcheck --pre-day`): before the climate day, fills the day-ahead
  GFS/ECMWF forecast disagreed with earned the same as the rest (+3..+6%/position for gap 0.1-0.3,
  smear 2-4F). The market already prices forecasts. (Counting all in-day fills makes the veto look
  great, -13% vs -1%, but that is fills on sides observations had already decided, which the
  observation veto handles.)
- Release pause: econ fills in the last 15 min before their own release earned +12%/position, the
  best window, so a series is never paused ahead of its own release; only Fed markets pause for
  30 min after a CPI/jobs/GDP print (no stored fills fall there, so it is cheap insurance).
- Gas: on Nov 7 the market priced tomorrow's AAA number inside a 1.5c band while an AAA-trend model
  had +-2c; traders have intraday pump data. Log only.
