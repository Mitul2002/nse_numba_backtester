# test_strategy_diversity.py
# Runs a diverse set of 20 strategies (drawn from alpha_strategy_parser's own example
# library) across the full symbol universe, to validate strategy_library_engine.py's
# generality across indicator families, timeframes, and parameter combinations --
# not just the one or two strategies used elsewhere in this repo's README.
import sys
import os
import time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from strategy_library_engine import run_strategy_for_symbol

DATA_FOLDER = os.environ.get(
    "BT_DATA_FOLDER",
    os.path.join(os.path.dirname(__file__), '..', 'data_sample')
)

STRATEGIES = [
    ("Basic: EMA trend + RSI",            "ema(close, 50) > ema(close, 200) AND rsi(close, 14) > 40"),
    ("Basic: SMA + MACD",                 "sma(close, 20) > sma(close, 50) AND macd(close, 12, 26, 9) > 0"),
    ("Crossover: RSI + ADX",              "rsi(close, 14) crossover 50 AND adx(high, low, close, 14) > 20"),
    ("Volume-based: MFI + Williams%R",    "mfi(high, low, close, volume, 14) > 70 AND willr(high, low, close, 14) < -50"),
    ("Stochastic",                        "stoch(high, low, close, 14) > 80 AND rsi(close, 14) > 60"),
    ("StochRSI crossover",                "stochrsi(close, 14) crossover 50 AND ema(close, 20) > ema(close, 50)"),
    ("NATR + MACD crossover",             "natr(high, low, close, 14) > 2 AND macd(close, 12, 26, 9) crossover 0"),
    ("Bollinger Bands (dot-property)",    "bbands(close, 20, 2).upper < close AND rsi(close, 14) > 70"),
    ("CMO + Ultimate Oscillator",         "cmo(close, 14) > 40 AND ultosc(high, low, close, 7, 14, 28) > 50"),
    ("DI crossover + ADX",                "plus_di(high, low, close, 14) crossover minus_di(high, low, close, 14) AND adx(high, low, close, 14) > 25"),
    ("Multi-timeframe: RSI daily+weekly", "tf(rsi(close, 14) > 70, 'daily') AND tf(rsi(close, 14) < 30, 'weekly')"),
    ("Multi-timeframe: EMA+MACD d/w",     "tf(ema(close, 50) > ema(close, 200), 'daily') AND tf(macd(close, 12, 26, 9) > 0, 'weekly')"),
    ("Aggregation: count()",              "count(rsi(close, 14) > 70, 10) >= 5 AND ema(close, 50) > ema(close, 200)"),
    ("Aggregation: countstreak()",        "countstreak(ema(close, 21) > ema(close, 55), 8) >= 1 AND rsi(close, 14) > 50"),
    ("Lag: n_days_ago",                   "ema(close, 21) > n_days_ago(ema(close, 21), 5) AND rsi(close, 14) > 50"),
    ("Lag: n_weeks_ago",                  "n_weeks_ago(ema(close, 34), 4) < ema(close, 34) AND mfi(high, low, close, volume, 14) > 60"),
    ("Nested: stddev of stddev",          "stddev(close, 20) > sma(stddev(close, 20), 10) AND ema(close, 50) > ema(close, 200)"),
    ("Nested: abs(macd) vs sma(abs)",     "abs(macd(close, 12, 26, 9)) > sma(abs(macd(close, 12, 26, 9)), 10) AND stoch(high, low, close, 14) > 80"),
    ("Golden cross (EMA crossover)",      "ema(close, 50) crossover ema(close, 200) AND rsi(close, 14) > 40"),
    ("PPO crossover",                     "ppo(close, 12, 26) crossover 0 AND stoch(high, low, close, 14) > 70"),
]

csv_files = sorted(f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv'))
print(f"{len(STRATEGIES)} strategies x {len(csv_files)} symbols\n")

summary = []
for name, strategy in STRATEGIES:
    t0 = time.perf_counter()
    n_success, n_error, n_skipped = 0, 0, 0
    total_trades = 0
    returns = []
    errors_seen = set()
    for f in csv_files:
        symbol = os.path.splitext(f)[0]
        r = run_strategy_for_symbol(os.path.join(DATA_FOLDER, f), symbol, strategy)
        status = r['status']
        if status == 'Success':
            n_success += 1
            tt = r.get('Total Trades', 0)
            if isinstance(tt, (int, float)) and not np.isnan(tt):
                total_trades += tt
            ret = r.get('Total Return [%]', np.nan)
            if isinstance(ret, (int, float)) and not np.isnan(ret):
                returns.append(ret)
        elif status.startswith('Skipped'):
            n_skipped += 1
        else:
            n_error += 1
            errors_seen.add(status[:120])
    dt = time.perf_counter() - t0
    avg_ret = np.mean(returns) if returns else float('nan')
    print(f"[{name}]")
    print(f"  strategy: {strategy}")
    print(f"  success={n_success} error={n_error} skipped={n_skipped}  total_trades={total_trades}  avg_return={avg_ret:.2f}%  time={dt:.1f}s")
    if errors_seen:
        print(f"  ERRORS SEEN: {list(errors_seen)[:5]}")
    print()
    summary.append((name, n_success, n_error, n_skipped, total_trades, avg_ret, dt))

print("=" * 100)
print(f"{'Strategy':<38}{'Success':>8}{'Error':>7}{'Skip':>6}{'Trades':>10}{'AvgRet%':>10}{'Time(s)':>9}")
for name, s, e, sk, tt, ar, dt in summary:
    print(f"{name:<38}{s:>8}{e:>7}{sk:>6}{tt:>10}{ar:>10.2f}{dt:>9.1f}")
