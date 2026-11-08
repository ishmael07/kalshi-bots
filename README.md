# Kalshi bots (paper money)

Two shadow bots trading Kalshi's public order books with virtual money ($75 each, Steady and Aggressive).
Fills are simulated from the public trade tape; no Kalshi account or key is used.

- Runs 24/7 on GitHub Actions (`.github/workflows/bots.yml`), state committed to `data/`.
- Dashboard: GitHub Pages from `docs/` (refreshed every ~6 min).
- Strategy and evidence: `notes/design.md`.

Local: `python3 -m kalshi run`, `python3 -m kalshi dash` (http://127.0.0.1:8789), `python3 -m pytest -q`.
