"""python3 -m kalshi <command> [options]

  collect   download settled markets + trade tapes (public API)
  study     maker/taker edge study
  crypto    crypto fair-value model study
  strategy  walk-forward favorite-maker test, gates, $75 simulation
  train     learn the cell win-rate table for the live bot
  run       always-on bot (shadow mode: live public data, virtual money)
  dash      local dashboard (http://127.0.0.1:8789)
  export    static dashboard for GitHub Pages (docs/)
  wxcheck   would a weather-forecast veto have helped? (stored fills vs day-ahead forecasts)
  citycheck does the model make money on cities it never saw? (out-of-sample)
"""
import sys

COMMANDS = {
    "collect": ("kalshi.collect", "main"),
    "study": ("kalshi.study", "main"),
    "crypto": ("kalshi.crypto_eval", "main"),
    "strategy": ("kalshi.run_strategy", "main"),
    "train": ("kalshi.run_bot", "cmd_train"),
    "run": ("kalshi.run_bot", "cmd_run"),
    "dash": ("kalshi.run_bot", "cmd_dash"),
    "export": ("kalshi.run_bot", "cmd_export"),
    "wxcheck": ("kalshi.wx_check", "main"),
    "citycheck": ("kalshi.city_check", "main"),
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        return 2
    import importlib
    mod, fn = COMMANDS[sys.argv[1]]
    return getattr(importlib.import_module(mod), fn)(sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main() or 0)
