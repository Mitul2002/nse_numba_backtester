# NSE Numba Backtester

A vectorized-signal / Numba-JIT-executed backtesting engine, benchmarked against a
[vectorbt](https://github.com/polakowo/vectorbt) + TA-Lib baseline across 2,050 NSE
equities.

Strategy (identical in both engines, for apples-to-apples comparison): enter when
RSI(14) crosses above 40 while EMA(10) > EMA(50); exit on an ATR(14)-based dynamic
stop-loss (2x ATR) or take-profit (3:1 reward:risk), simulated bar-by-bar.

## Why this exists

`vectorbt`'s `Portfolio.from_signals` is fast, but it's a general-purpose black box —
it doesn't cleanly expose exactly how a bar-by-bar SL/TP with position sizing and fees
resolves internally. `src/numba_engine.py` re-implements the same execution logic as an
explicit, auditable, `@njit`-compiled loop (`custom_backtest_numba_loop_atr`), so the
exact fill/fee/sizing mechanics are visible and controllable, not hidden inside a
library.

## Verified results (full universe, 2,050 NSE symbols, `evidence/`)

| | Baseline (`vectorbt` + TA-Lib) | Custom (Numba JIT) |
|---|---|---|
| Total wall time, full universe | **172.04s** | **17.54s** |
| Success rate | 2045 / 2050 (99.8%)* | 2050 / 2050 (100%) |
| Avg time / symbol | 83.9ms | 8.5ms |

**~9.8x speedup**, single process, no parallelism — the Numba JIT compilation of the
per-bar SL/TP/position loop alone accounts for the difference (the indicator layer,
TA-Lib, is identical in both engines). Raw per-symbol timing and full metric output for
both runs are committed under `evidence/*_FULL_2050_symbols.csv` — nothing here is
recomputed or estimated, it's the literal output of the two scripts, run on the actual
2,050-symbol NSE OHLCV dataset.

\* 5 symbols skipped by the baseline for having <70 rows of history (a stricter
minimum-length check than the custom engine uses); the custom engine's lower threshold
(+5 vs +20 bars) let it process all 2,050.

An earlier validation pass on a 1,014-symbol subset (43.57s -> 7.06s) is kept in
`evidence/*_FULL_1014_symbols.csv` for reference; the 2,050-symbol run above is the
current, primary result and the one that matches the full NSE universe this was
benchmarked against.

Run it yourself: `python src/baseline_vbt_talib.py` and `python src/numba_engine.py`,
pointed at `BT_DATA_FOLDER` (see Setup below).

## Numerical parity: two real bugs found, then fixed

`src/validate_parity.py` does an element-wise `np.isclose` comparison (rtol=1e-4,
atol=1e-6) across 25 performance metrics for every symbol. The first full-universe run
showed only 79.2% exact trade-count agreement and Sharpe correlation of 0.89 between the
two engines — different enough to investigate rather than wave off as float noise.
Root-caused by diffing individual trades (entry/exit dates and prices) symbol by symbol
against the raw OHLCV bars:

1. **Look-ahead bug in the baseline.** `vbt.Portfolio.from_signals()` was called without
   a `price=` argument, which defaults to `np.inf` — vectorbt resolves that to the
   **close of the same bar that generated the entry signal**. So the baseline was
   trading on today's RSI cross at today's close, information not actually available in
   real time. The custom engine does this correctly (signal known at bar X's close,
   filled at bar X+1's open); the baseline needed `entries.shift(1)` and an explicit
   `price=open_prices` to match.
2. **Wrong reference price for the stop-loss/take-profit distance.** vectorbt's
   `stop_entry_price` defaults to its internal valuation price, not the trade's actual
   fill price — so the SL/TP percentage was being converted to an absolute price level
   using the wrong base price, silently letting trades run far past where they should
   have exited (in one traced example, a real stop-out on day 3 turned into a fantasy
   +29% hold for three months). Setting `stop_entry_price='fillprice'` fixed it.
3. **The custom engine silently dropped still-open positions.** After the first two
   fixes, ~180 symbols still mismatched on trade count — 146 of them by exactly +1, all
   in vectorbt's favor. Tracing them showed the pattern: a position still open when the
   data ends (the dataset's last date). vectorbt's `Total Trades` includes open
   positions, marked-to-market at the last close; the custom engine's
   `custom_backtest_numba_loop_atr` only ever appends a trade record on an SL/TP exit,
   and `calculate_custom_metrics` hard-coded `Total Open Trades = 0` — so an open
   position vanished from every trade-count metric entirely. Fixed by having the Numba
   loop also return the open position's shares/entry price if one exists at the end,
   and counting it in `Total Trades` / `Total Open Trades` / `Open Trade PnL`
   (mark-to-market against the last close), matching vectorbt's convention.

4. **Same-bar re-entry.** After fix 3, ~180 symbols still mismatched, this time with no
   consistent direction. Diffing entry/exit timestamps directly (not just counts) showed
   the custom engine allows opening a new position on the exact same bar it just closed
   one (its exit check and entry check both run within a single loop iteration);
   vectorbt's `from_signals`, in this configuration, does not register a same-bar
   re-entry — that signal is dropped, not delayed. Matched the custom engine to
   vectorbt's actual behavior by skipping the entry check on any bar where an exit just
   happened (`just_exited` flag in the loop).

**After all four fixes**, on the same 2,050-symbol universe:

| | Before any fix | After 1-2 | After 3 | After 4 |
|---|---|---|---|---|
| Exact trade-count match | 1,624 / 2,050 (79.2%) | 1,865 / 2,050 (91.0%) | 2,007 / 2,050 (97.9%) | **2,039 / 2,050 (99.5%)** |
| Mean abs. trade-count diff | 0.23 | 0.089 | 0.019 | **0.0029** |
| Sharpe Ratio correlation | 0.89 | 0.997 | 0.997 | **0.998** |
| Total Return correlation | — | 0.995 | 0.995 | **0.995** |
| Win rate, mean abs. diff | 4.15 pts | 0.13 pts | 0.13 pts | **0.06 pts** |

Restricting to the 2,045 symbols where *both* engines actually produced a result (5
symbols are skipped by the baseline's stricter minimum-history check and were never a
simulation disagreement to begin with), the exact trade-count match is **2,039 / 2,045
(99.7%)**.

**The last 6 symbols were traced individually and left as-is deliberately.** Their
pattern is different in kind from fixes 1-4: on a bar where price grazes the SL/TP level
without clearly blowing through it, the two engines' independently-computed stop price
(via slightly different float rounding through the entry fill/slippage/ATR chain) sit on
opposite sides of that bar's low/high by a fraction of a percent — one engine calls it a
hit, the other doesn't, and a trade that should exit in month 1 instead runs until a
later, coincidentally similar-priced touch. Forcing that shut would mean making the
Numba loop replicate vectorbt's internal floating-point operations bit-for-bit, at which
point it stops being an independent second implementation to validate against — the
entire point of building it. 99.7% agreement from two independently-written engines is
the intended outcome; 100% would mean one of them is a re-skin of the other.

The strict element-wise match rate at rtol=1e-4 (0.01%) is still ~48% — that's expected
and separate from the above: it flags essentially any residual floating-point-level
difference in a metric compounded over a 10-20 year daily series (Total Return, Sharpe,
Expectancy), not a disagreement about what trades happened. Full detail in
`evidence/mismatches_detailed_per_source_FULL_2050_symbols.csv`.

## Setup

```bash
pip install -r requirements.txt
# TA-Lib's C library must be installed separately before `pip install TA-Lib` works;
# see https://github.com/TA-Lib/ta-lib-python#installation
```

By default both scripts read from `data_sample/` (6 symbols, included, for a quick
smoke test). To reproduce the full run, point `BT_DATA_FOLDER` at a folder of
per-symbol daily OHLCV CSVs. Both a `date` and a `datetime` timestamp column name are
supported, alongside `open,high,low,close,volume`:

```bash
export BT_DATA_FOLDER=/path/to/your/nse_ohlcv_csvs
python src/baseline_vbt_talib.py
python src/numba_engine.py
python src/validate_parity.py
```

## Layout

```
src/
  baseline_vbt_talib.py       # reference implementation via vectorbt + TA-Lib
  numba_engine.py             # custom engine: @njit execution loop, this is the core work
  validate_parity.py          # element-wise comparison + mismatch report generator
  strategy_library_engine.py  # generalizes the entry signal to any strategy string (see below)
  baseline_library_vbt.py     # vectorbt counterpart to strategy_library_engine.py
  test_strategy_diversity.py  # runs 20 diverse strategies across the full universe (robustness)
  test_strategy_parity_50.py  # cross-validates 50 strategies x 250 symbols vs vectorbt (correctness)
data_sample/               # 6 symbols for a fast local smoke test
evidence/                  # committed output from the full 2,050-symbol run (not regenerated by CI)
logs/                      # gitignored; parity_50x250.log etc.
```

## Generalizing beyond one hardcoded strategy

`numba_engine.py`'s execution loop (`custom_backtest_numba_loop_atr`) takes
`entry_signals` as a plain boolean array — it has no idea whether that array came from
a hardcoded RSI-cross rule or anything else. `strategy_library_engine.py` swaps out
*only* the signal-generation step (which runs in ordinary Python, before the JIT loop
is ever called) for a call into a sibling project, `alpha_strategy_parser`: its
grammar-based parser and `FunctionRegistry` (40+ indicators — RSI, SMA/EMA, MACD,
Bollinger Bands, Stochastic, ADX/DI, ATR, OBV, MFI, CCI, Williams %R, SAR, `crossover`
as a first-class operator) turn an arbitrary strategy string into the same shape of
boolean array the hardcoded version used to build by hand. Nothing about the loop
itself, or the fee/slippage/SL-TP mechanics, changed.

```bash
python src/strategy_library_engine.py \
  --strategy "ema(close, 50) > ema(close, 200) AND rsi(close, 14) > 40" \
  --data-folder data_sample --limit 6
```

Two levels of validation were done, matching the same "don't just claim it, check it"
standard as the rest of this repo:

**1. Robustness across 20 strategies, full 2,050-symbol universe** (`src/test_strategy_diversity.py`,
strategies drawn from `alpha_strategy_parser`'s own example library, not hand-picked to
look good) — **20/20 succeeded, 0 errors**, spanning basic trend+momentum, all 3
`crossover` forms, 5 oscillators, directional indicators, Bollinger dot-property access,
multi-timeframe (`tf()`), both aggregation functions, both lag functions on *computed*
indicators (not just raw price), and nested expressions. Full output:
`evidence/strategy_library_20_strategies_FULL_2050_symbols.txt`.

**2. Cross-validated against vectorbt on 50 strategies x 250 symbols** (`src/test_strategy_parity_50.py`) —
this is the important one: it doesn't just check that the Numba engine runs without
crashing, it generates the entry signal once per (strategy, symbol) pair and runs
*both* engines on the exact same signal array, the same way `validate_parity.py` did
for the single original strategy. Results:

| | Value |
|---|---|
| Strategies | 50 (spanning every category above, none hand-picked) |
| Symbols per strategy | 248/250 (2 consistent short-history skips) |
| Errors | **0** across all 12,500 attempted pairs |
| Exact trade-count match | mean **95.3%**, range 82.7%-100.0%, 10/50 strategies at exactly 100% |
| Mean abs. trade-count diff | 0.07 trades |
| Sharpe Ratio correlation | mean **0.991**, min 0.874, max 1.000 |
| Runtime | 142.8s (8 workers) |

This testing process itself found and fixed **3 more real bugs** in `alpha_strategy_parser`,
none related to this repo's execution loop: `bbands(...).upper - bbands(...).lower`
style arithmetic crashed because the resolved operands came back as pandas Series and a
type-check only accepted `np.ndarray`; `countstreak()` on a raw boolean comparison
crashed with `KeyError` because its internal loop indexed a sliced Series positionally
(`data[i]`) when a sliced Series keeps its *original* labels, not a fresh 0..n range;
and — the deepest one — `bbands`'s scalar `period`/`nbdev` arguments were being
incorrectly broadcast into full-length arrays by the same parameter-normalization code
that fixed the first bug, which then made `talib.BBANDS` fail because it requires a
real scalar for `timeperiod`. Full per-pair detail:
`evidence/strategy_parity_50_strategies_250_symbols.csv` and `.log`.

**What this does and doesn't prove:** the entry side is genuinely general and now
cross-validated, not just "doesn't crash" — any strategy string the parser can express,
the Numba loop can execute, with results that agree with an independent reference
implementation at the same 95%+ level established for the single original strategy. The
exit side (SL/TP mechanics) is not general: it's still the fixed 2x-ATR-stop /
3:1-take-profit shape from `numba_engine.py`. Trailing stops, time-based exits, or
multi-leg exits would need actual loop changes — Numba's `nopython` mode can't
dynamically interpret arbitrary exit logic the way the (pure-Python, pre-loop) signal
generation step can.

## Parallelism across symbols

`numba_engine.py` defaults to single-process (`BT_WORKERS=1`), identical to the
verified-results run above. Set `BT_WORKERS=N` to fan out across N worker processes:

```bash
BT_WORKERS=8 python src/numba_engine.py
```

Tested three backends empirically on the full 2,050-symbol universe (24 logical cores
available) before picking one, rather than assuming stdlib is good enough or that the
fastest number wins by default:

| Backend | Best worker count found | Best wall time |
|---|---|---|
| stdlib `concurrent.futures.ProcessPoolExecutor` | 4 | 8.4-9.5s |
| **joblib (`loky` backend)** — shipped | **6-8** | **6.3-7.5s** |
| dask (`processes` scheduler) | 8 | 6.2s |

dask was the fastest by a small, consistent margin (~10%), but it's a full task-graph
scheduling engine meant for complex/distributed pipelines — pulling it in as a
dependency isn't justified by a 10% edge on what's just "run the same function on 2,050
independent files." joblib's `Parallel`/`delayed` is the standard, purpose-built tool
for exactly this embarrassingly-parallel pattern, already a near-ubiquitous transitive
dependency in the Python data stack, and came in a close second. Shipped with joblib.

All three plateau (or reverse) well before 24 workers — each symbol is only ~8ms of
real work, so past roughly 6-8 processes, spawn/IPC overhead outweighs the parallelism
gained. Confirmed this wasn't measurement noise by repeating the extremes: joblib at
24 workers ranged from 6.9s to 13.8s across repeated runs, while 8 workers stayed
consistently in the 6.3-7.5s band.

The parallel output was verified to be numerically identical to the sequential run
(`Total Trades` summed and compared row-for-row) before trusting the speedup — parallel
execution changes nothing about which trades happen, only how the per-symbol loop is
scheduled. Combined with the 9.8x Numba-vs-vectorbt speedup, this puts the full-universe
run at roughly 172s (vectorbt) -> ~7s (Numba, joblib, 8 workers), about **~25x**.

## Scope notes

- One strategy (RSI/EMA/ATR) is used throughout, chosen to give both engines an
  identical, non-trivial execution path (dynamic SL/TP, fees, slippage) to compare on —
  not a claim that this specific strategy is alpha-generating.
- `data_sample/` is a handful of large-cap NSE names for a quick start; the full
  2,050-symbol universe used for the verified numbers above is not included in this
  repo (~430MB of CSVs) — the `evidence/` folder is the record of that run.
