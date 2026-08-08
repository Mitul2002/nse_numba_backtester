# How to talk about this project

A companion to README.md — this one is for you, not for the repo visitor. Read this
before the interview, not during it.

## The 30-second version

I built a custom backtesting engine (`numba_engine.py`) with a bar-by-bar execution
loop compiled with Numba JIT — real stop-loss/take-profit/position-sizing logic, not a
library call. To prove it was correct, not just fast, I benchmarked it against a
vectorbt + TA-Lib reference implementation across 2,050 NSE stocks, wrote a tolerance-
based comparison tool, and when the two engines only agreed on 79% of trades initially,
I traced individual trades against the raw OHLCV bars until I found four concrete bugs
— three in my own reference script — fixed them, and got to 99.7% agreement on the
comparable universe. Final numbers: 9.8x speedup (172s -> 17.5s across all 2,050
symbols), both close to 100% success rate. Then generalized it past that one hardcoded
strategy by wiring its entry-signal step to `alpha_strategy_parser`'s grammar-based
parser (RSI/MACD/Bollinger/Stochastic/ADX/... — 40+ indicators) — the Numba loop
already took entry signals as a plain array, so this was glue code, not a redesign.

## Why two engines at all

`vectorbt`'s `Portfolio.from_signals` is fast but it's a black box — you hand it
entries/exits and stop percentages and it does... something internally. I wanted a
version where every fill price, every fee, every stop trigger is a line of code I wrote
and can point to. So I built `numba_engine.py` as an independent, from-scratch
implementation of the same strategy, then used vectorbt as the reference to check my
engine against — not the other way around.

## The story arc (this is the part worth telling well)

**Act 1 — build it, benchmark it.** RSI/EMA/ATR strategy, same in both engines. First
run: 43.6s (vectorbt) vs 7.1s (custom) on a 1,014-symbol subset. Looked great.

**Act 2 — validate it, don't just trust it.** Wrote `validate_parity.py`: element-wise
comparison of 25 metrics per symbol, tight tolerance (rtol=1e-4). Result: only 79.2%
exact trade-count agreement, Sharpe correlation 0.89. Good enough to demo, not good
enough to trust.

**Act 3 — the debugging.** Instead of loosening the tolerance to make the number look
better, traced actual trades (entry/exit dates, fill prices) against the raw bars for
mismatched symbols. Found:
- Bug 1: the *reference* script had a look-ahead bug (filled orders at the same bar's
  close that generated the signal, not the next bar's open).
- Bug 2: the reference script's stop-loss/take-profit was anchored to the wrong
  reference price (vectorbt's internal valuation price, not the actual fill price) —
  this one is the good story, because in one traced case a real 3-day stop-out was
  turning into a fake 3-month +29% hold.
- Bug 3: *my own engine* was silently dropping trades that were still open when the
  data ended (hardcoded `Total Open Trades = 0`), while vectorbt correctly counted them.
- Bug 4: my engine allowed re-entering a position on the exact same bar it exited one;
  vectorbt doesn't. Aligned mine to match.

**Act 4 — know when to stop.** Landed at 99.7% (2,039/2,045 comparable symbols). The
last 6 disagreements are float-rounding sensitivity on a coin-flip stop-touch — closing
that gap would mean making one engine numerically imitate the other's internals, which
defeats having two independent implementations to cross-check. Said so explicitly in
the README instead of hiding it.

**Act 5 — parallelize, but test first.** Rather than assume "more workers = faster,"
benchmarked `BT_WORKERS` from 1 to 24 on the full universe before picking a default —
and didn't stop at the first working option either. Tested three backends: stdlib
`ProcessPoolExecutor` (best: 4 workers, ~8.4-9.5s), joblib/loky (best: 6-8 workers,
~6.3-7.5s), and dask's processes scheduler (best: 8 workers, ~6.2s, fastest by ~10%).
Shipped joblib, not dask, despite dask winning on raw speed — dask is a task-graph
engine built for complex/distributed pipelines, and a 10% edge doesn't justify that
dependency weight for "run the same function on 2,050 independent files." joblib is the
standard, purpose-built tool for exactly this pattern. Also caught that 24 workers
looked artifactually fast in one joblib run (6.9s) until repeating it showed it was
actually the least stable config (6.9-13.8s across runs) — 8 workers was the reliable
choice. Verified the parallel run was numerically identical to sequential (summed
`Total Trades`, compared row-for-row) before trusting any of these numbers. Combined
with the JIT speedup: ~172s (vectorbt) -> ~7s (Numba, joblib, 8 workers), ~25x.

## Numbers to have cold

| Metric | Value |
|---|---|
| Universe | 2,050 NSE symbols |
| Custom engine time (full universe) | 17.5s |
| Reference (vectorbt) time (full universe) | 172s |
| Speedup | 9.8x |
| Success rate (custom) | 2,050 / 2,050 |
| Exact trade-count match (final) | 99.7% (2,039/2,045 comparable) |
| Exact trade-count match (before any fix) | 79.2% |
| Sharpe Ratio correlation (final) | 0.998 |
| Bugs found and fixed | 7 total (4 on the original strategy; 3 more found via 50-strategy testing) |
| Parallel backends tested | 3 (ProcessPoolExecutor, joblib, dask) |
| Best parallel config (shipped) | joblib, 8 workers, 21s -> ~7s |
| Combined speedup (JIT + parallel) vs vectorbt | ~25x |
| Generalization: strategies tested for robustness | 20, full 2,050-symbol universe, 0 errors |
| Generalization: strategies cross-validated vs vectorbt | 50 x 250 symbols, 0 errors, 95.3% mean exact match, 0.991 mean Sharpe corr |

## If they push on specifics

- **"Why is the reference slower than a naive first run?"** Because I fixed a bug in
  it (`stop_entry_price='fillprice'`) that happened to make it do more internal work —
  correctness cost some speed, and I'd rather report the honest slower-but-correct
  number than the faster-but-buggy one.
- **"Why not 100%?"** Explained above — floating-point boundary sensitivity on
  near-miss stop touches, not a logic disagreement. I can name the exact mechanism if
  asked (entry fill price differs by a hair -> SL/TP price differs by a hair -> a bar
  that just grazes the level lands on different sides of that hair in each engine).
- **"What would you do with more time?"** Both parallelism and multi-strategy
  generalization are already done (see below) — if pushed further: parallelize
  `strategy_library_engine.py` the same way `numba_engine.py` is (it's still
  single-process), and generalize the exit side (trailing stops, time-based exits),
  which is the one piece that's still hardcoded.
- **"Does it only support one strategy?"** Not anymore, and it's cross-validated, not
  just "doesn't crash." `strategy_library_engine.py` swaps the hardcoded entry-signal
  construction for a call into `alpha_strategy_parser`'s grammar-based parser and
  40-plus-function registry. Two rounds of testing: (1) 20 structurally different
  strategies across the full 2,050-symbol universe, 0 errors — proves robustness; (2)
  the real one — **50 strategies x 250 symbols, each one run through both the Numba
  engine AND vectorbt on the identical generated signal array**, the same cross-check
  `validate_parity.py` did for the original single strategy. Result: 0 errors across
  12,500 pairs, mean 95.3% exact trade-count match (range 82.7-100%, 10/50 strategies
  at exactly 100%), mean Sharpe correlation 0.991 (min 0.874). That's the number to
  lead with if asked "does it actually work," not the 20-strategy robustness pass.
- **"Did testing more strategies find anything?"** Yes — 3 more real bugs in
  `alpha_strategy_parser`, all found because 50 diverse strategies exercise code paths
  20 didn't: `bbands(...).upper - bbands(...).lower` arithmetic crashed (resolved
  operands came back as pandas Series, a type-check only accepted `np.ndarray`);
  `countstreak()` on a raw boolean comparison crashed with `KeyError` (its internal
  loop indexes a sliced Series positionally, but a slice keeps its *original* row
  labels, not a fresh 0..n range); and the fix for the first bug accidentally exposed
  a second, deeper one — `bbands`'s scalar `period`/`nbdev` args were being broadcast
  into full-length arrays by the same normalization code, which then made
  `talib.BBANDS` fail because it needs a real scalar for `timeperiod`. All three fixed
  and re-verified. This is the strongest single point for "why test more than a
  couple of examples" if asked.
- **"How do you know the multi-timeframe results are actually right, not just
  running without crashing?"** Two answers now. First, the 20-strategy pass: the two
  `tf()` strategies there produced very different trade counts (113 vs. 102,361)
  depending on whether the daily+weekly combination was restrictive or lenient — that
  differentiation is itself evidence of real, sensitive bucketing. Second, and
  stronger: 6 of the 50 strategies in the parity run were multi-timeframe, and they
  cross-validated against vectorbt at the same 83-100% level as everything else — not
  just internally consistent, but agreeing with an independent engine.
- **"Is the exit logic general too?"** No, and say so directly if asked — that's the
  honest boundary. SL/TP is still the fixed 2x-ATR / 3:1 shape. Numba's `nopython`
  mode can't interpret arbitrary exit rules at runtime the way the pure-Python signal
  step can; genuinely general exits would need new loop variants (or a small bytecode
  interpreter), not attempted here.
- **"Why 8 workers and not more?"** Tested it, didn't assume it. 24 workers was *less
  stable* than 8, not faster (6.9-13.8s across repeated runs vs a tight 6.3-7.5s band)
  because each symbol is only ~8ms of work — past a handful of processes, spawn/IPC
  overhead exceeds the parallelism gained. This is the answer to give if they're
  testing whether you actually benchmark or just guess.
- **"Why joblib and not dask, if dask was faster?"** It was, by about 10% (6.2s vs
  6.9s) — but dask is a task-graph scheduling engine meant for complex, often
  distributed pipelines. Pulling in that dependency weight isn't justified by a 10%
  edge on a workload that's just "run the same pure function on 2,050 independent
  files." joblib's `Parallel`/`delayed` is the right-sized tool for that exact pattern.
  Good answer if they're probing engineering judgment, not just benchmark-chasing.
- **"Is the strategy itself good?"** Deliberately not the point — it's the same
  non-trivial strategy (dynamic ATR stops, fees, slippage) run through both engines
  specifically so there's something real to disagree about. Not a claim of alpha.

## What NOT to say

- Don't round "2,050 symbols" up or down inconsistently — it's the real count, and it's
  close enough to any "2,000+" claim elsewhere that you don't need to fudge it.
- Don't claim the parity checker (`validate_parity.py`) as more sophisticated than it
  is: it's `np.isclose` across 25 columns plus a same-symbol merge. The value is that
  it's real and its output is committed as evidence, not that the method is exotic.
