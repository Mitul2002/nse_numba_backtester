# test_strategy_parity_50.py
# The rigorous version of test_strategy_diversity.py: for 50 diverse strategies (drawn
# from alpha_strategy_parser's own example library) across 250 real NSE symbols, runs
# BOTH engines -- strategy_library_engine.py (Numba) and baseline_library_vbt.py
# (vectorbt) -- on the exact same generated entry-signal array, and reports per-strategy
# trade-count agreement and Sharpe/Return correlation. This is the same correctness
# check validate_parity.py did for the one original hardcoded strategy, repeated across
# 50 strategies to back up the generalization claim with more than "it didn't crash."
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import talib
import vectorbt as vbt
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(__file__))
from strategy_library_engine import generate_entry_signals  # noqa: E402
from numba_engine import (  # noqa: E402
    custom_backtest_numba_loop_atr, calculate_custom_metrics,
    ATR_PERIOD, ATR_MULTIPLIER_FOR_SL, RR_RATIO_FOR_TP, INIT_CASH, FEES_PCT, SLIPPAGE_PCT,
)
from baseline_library_vbt import format_stat_value  # noqa: E402
from numba import float64  # noqa: E402

STRATEGIES = [
    "tf(rsi(close, 14) > 70, 'daily') AND tf(rsi(close, 14) < 30, 'weekly')",
    "tf(ema(close, 50) > ema(close, 200), 'daily') AND tf(macd(close, 12, 26, 9) > 0, 'weekly')",
    "tf(sma(close, 20) > sma(close, 50), 'daily') AND tf(adx(high, low, close, 14) > 25, 'monthly')",
    "tf(cci(high, low, close, 14) > 60, 'daily') AND tf(ema(close, 21) > ema(close, 55), 'weekly') AND tf(macd(close, 12, 26, 9) > 0, 'monthly')",
    "tf(stoch(high, low, close, 14) > 80, 'daily') AND tf(mfi(high, low, close, volume, 14) > 70, 'weekly') AND tf(atr(high, low, close, 14) > 5, 'monthly')",
    "tf(count(rsi(close, 14) > 70, 10) >= 5, 'daily') AND tf(max(ema(close, 21), 20) > ema(close, 55), 'weekly')",
    "ema(close, 50) > ema(close, 200) AND rsi(close, 14) > 40",
    "mfi(high, low, close, volume, 14) > 70 AND willr(high, low, close, 14) < -50",
    "stoch(high, low, close, 14) > 80 AND rsi(close, 14) > 60",
    "natr(high, low, close, 14) > 2 AND macd(close, 12, 26, 9) crossover 0",
    "cmo(close, 14) > 40 AND ultosc(high, low, close, 7, 14, 28) > 50",
    "sma(close, 10) > sma(close, 30) AND rsi(close, 14) crossover 40 AND obv(close, volume) > sma(obv(close, volume), 50)",
    "stoch(high, low, close, 14) crossover 20 AND plus_di(high, low, close, 14) > minus_di(high, low, close, 14) AND rsi(close, 14) > 50",
    "stochrsi(close, 14) > 80 AND ultosc(high, low, close, 7, 14, 28) > 70 AND obv(close, volume) > sma(obv(close, volume), 30)",
    "ema(close, 20) > ema(close, 100) AND stochrsi(close, 14) crossover 40 AND mfi(high, low, close, volume, 14) > 60 AND adx(high, low, close, 14) > 25",
    "sma(close, 30) > sma(close, 100) AND mfi(high, low, close, volume, 14) < 40 AND adx(high, low, close, 14) > 25",
    "plus_di(high, low, close, 14) > minus_di(high, low, close, 14) AND macd(close, 12, 26, 9) > 0 AND sma(close, 20) > sma(close, 50)",
    "sma(close, 20) < sma(close, 100) AND macd(close, 12, 26, 9) < 0 AND stoch(high, low, close, 14) < 30",
    "sma(close, 10) > sma(close, 30) AND atr(high, low, close, 14) < 10 AND macd(close, 12, 26, 9) > 0",
    "ema(close, 20) > ema(close, 50) AND stochrsi(close, 14) > 60 AND ultosc(high, low, close, 7, 14, 28) > 65",
    "macd(close, 12, 26, 9) crossover 0 AND adx(high, low, close, 14) > 30 AND willr(high, low, close, 14) < -40",
    "sma(close, 30) > sma(close, 90) AND stoch(high, low, close, 14) > 80 AND macd(close, 12, 26, 9) > 0 AND rsi(close, 14) > 60",
    "ema(close, 50) > ema(close, 200) AND ultosc(high, low, close, 7, 14, 28) > 65 AND atr(high, low, close, 14) > 10 AND macd(close, 12, 26, 9) > 0",
    "macd(close, 12, 26, 9) crossover 0 AND bbands(close, 20, 2).upper - bbands(close, 20, 2).lower < sma(atr(high, low, close, 14), 20)",
    "ema(close, 50) > ema(close, 200) AND n_days_ago(ema(close, 50), 5) < ema(close, 50) AND rsi(close, 14) > 55",
    "stddev(close, 20) > sma(stddev(close, 20), 10) AND macd(close, 12, 26, 9) > 0 AND rsi(close, 14) > 60",
    "plus_di(high, low, close, 14) crossover minus_di(high, low, close, 14) AND rsi(close, 14) > 55 AND countstreak(close > sma(close, 20), 8) >= 3",
    "rsi(close, 14) < 30 AND bbands(close, 20, 2).lower > close AND count(close < n_days_ago(close, 1), 7) >= 4",
    "bbands(close, 20, 2).upper < close AND rsi(close, 14) > 65 AND count(rsi(close, 14) > 60, 14) >= 5",
    "obv(close, volume) > sma(obv(close, volume), 100) AND ema(close, 20) > ema(close, 50) AND rsi(close, 14) > 55",
    "sma(close, 20) > sma(close, 200) AND bbands(close, 20, 2).upper - bbands(close, 20, 2).lower < n_days_ago(atr(high, low, close, 14), 10)",
    "stoch(high, low, close, 14) crossover 80 AND willr(high, low, close, 14) > -20 AND rsi(close, 14) > 60",
    "ultosc(high, low, close, 7, 14, 28) > 65 AND adx(high, low, close, 14) > 25 AND ema(close, 50) > n_weeks_ago(ema(close, 50), 2)",
    "macd(close, 12, 26, 9) > 0 AND rsi(close, 14) crossover 50 AND bbands(close, 20, 2).middle < close",
    "bbands(close, 20, 2).upper < close AND rsi(close, 14) > 70 AND countstreak(rsi(close, 14) > 60, 7) >= 4",
    "ema(close, 21) crossover ema(close, 200) AND ultosc(high, low, close, 7, 14, 28) > 60 AND willr(high, low, close, 14) > -40",
    "bbands(close, 20, 2).middle crossover sma(close, 50) AND adx(high, low, close, 14) > 23 AND mfi(high, low, close, volume, 14) > 60",
    "plus_di(high, low, close, 14) > minus_di(high, low, close, 14) AND rsi(close, 14) > 50 AND n_days_ago(adx(high, low, close, 14), 5) < adx(high, low, close, 14)",
    "sma(close, 20) > sma(close, 50) AND obv(close, volume) crossover sma(obv(close, volume), 30) AND adx(high, low, close, 14) > 18",
    "stddev(close, 14) < n_days_ago(stddev(close, 14), 7) AND atr(high, low, close, 14) < 9 AND sma(close, 20) > sma(close, 50)",
    "bbands(close, 20, 2).upper < close AND count(close > bbands(close, 20, 2).upper, 10) >= 2 AND mfi(high, low, close, volume, 14) > 65",
    "stochrsi(close, 14) > 80 AND rsi(close, 14) > 60 AND countstreak(rsi(close, 14) > 55, 6) >= 4",
    "macd(close, 12, 26, 9) crossover 0 AND atr(high, low, close, 14) > 15 AND bb_upper(close, 20, 2) > close",
    "count(bb_upper(close, 20, 2) < close, 12) >= 4 AND volume > sma(volume, 20)",
    "countstreak(close > sma(close, 20), 7) >= 1 AND obv(close, volume) > sma(obv(close, volume), 20)",
    "round(ema(close, 21)) > sma(close, 34) AND atr(high, low, close, 14) > 5",
    "max(ema(close, 21), 20) > ema(close, 55) AND adx(high, low, close, 14) > 25",
    "sma(close, 50) > n_weeks_ago(sma(close, 50), 2) AND macd(close, 12, 26, 9) > 0",
    "n_weeks_ago(ema(close, 34), 4) < ema(close, 34) AND mfi(high, low, close, volume, 14) > 60",
    "n_days_ago(rsi(close, 14), 7) < rsi(close, 14) AND macd(close, 12, 26, 9) > 0",
]


def run_pair(file_path, symbol_name, strategy_str):
    """Load once, generate the entry signal once, run BOTH engines on that exact same
    signal array, return both results so the caller can compare them directly."""
    out = {'symbol': symbol_name, 'strategy': strategy_str,
           'numba_status': 'Failed', 'vbt_status': 'Failed',
           'numba_trades': np.nan, 'vbt_trades': np.nan,
           'numba_return': np.nan, 'vbt_return': np.nan,
           'numba_sharpe': np.nan, 'vbt_sharpe': np.nan}
    try:
        header_cols = pd.read_csv(file_path, nrows=0).columns
        date_col = 'date' if 'date' in header_cols else 'datetime'
        raw_df = pd.read_csv(file_path, index_col=date_col, parse_dates=True,
                              usecols=[date_col, 'open', 'high', 'low', 'close', 'volume'])
        raw_df.sort_index(inplace=True)
        if len(raw_df) < 90:
            out['numba_status'] = out['vbt_status'] = f"Skipped: rows {len(raw_df)} < 90"
            return out

        entry_signals = generate_entry_signals(strategy_str, raw_df)

        cap_df = raw_df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
        close_np = np.ascontiguousarray(cap_df['Close'].to_numpy(dtype=np.double))
        high_np = np.ascontiguousarray(cap_df['High'].to_numpy(dtype=np.double))
        low_np = np.ascontiguousarray(cap_df['Low'].to_numpy(dtype=np.double))
        open_np = np.ascontiguousarray(cap_df['Open'].to_numpy(dtype=np.double))
        atr_np = talib.ATR(high_np, low_np, close_np, timeperiod=ATR_PERIOD)

        # --- Numba engine ---
        try:
            equity, trades, fees, open_sh, open_ep, _ = custom_backtest_numba_loop_atr(
                open_np, high_np, low_np, close_np, atr_np, entry_signals.astype(np.bool_),
                float64(INIT_CASH), float64(ATR_MULTIPLIER_FOR_SL), float64(RR_RATIO_FOR_TP),
                float64(FEES_PCT), float64(SLIPPAGE_PCT)
            )
            metrics = calculate_custom_metrics(cap_df, equity, trades, INIT_CASH, fees, open_sh, open_ep)
            out['numba_trades'] = metrics['Total Trades']
            out['numba_return'] = metrics['Total Return [%]']
            out['numba_sharpe'] = metrics['Sharpe Ratio']
            out['numba_status'] = 'Success'
        except Exception as e:
            out['numba_status'] = f"Error: {type(e).__name__}: {e}"

        # --- vectorbt, same entry_signals array ---
        try:
            entries = pd.Series(entry_signals, index=raw_df.index)
            atr_pd = pd.Series(atr_np, index=raw_df.index)
            atr_on_signal_bar = atr_pd.shift(1)
            entries_shifted = entries.shift(1).fillna(False).astype(bool)
            safe_open = raw_df['open'].replace(0, np.nan).where(raw_df['open'] > 0, np.nan)
            sl_pct = (atr_on_signal_bar * ATR_MULTIPLIER_FOR_SL) / safe_open
            tp_pct = sl_pct * RR_RATIO_FOR_TP
            pf = vbt.Portfolio.from_signals(
                close=raw_df['close'], open=raw_df['open'], high=raw_df['high'], low=raw_df['low'],
                entries=entries_shifted, price=raw_df['open'],
                sl_stop=sl_pct, tp_stop=tp_pct, stop_entry_price='fillprice',
                init_cash=INIT_CASH, fees=FEES_PCT, slippage=SLIPPAGE_PCT, freq='D'
            )
            stats = pf.stats()
            out['vbt_trades'] = format_stat_value(stats.get('Total Trades', np.nan))
            out['vbt_return'] = format_stat_value(stats.get('Total Return [%]', np.nan))
            out['vbt_sharpe'] = format_stat_value(stats.get('Sharpe Ratio', np.nan))
            out['vbt_status'] = 'Success'
        except Exception as e:
            out['vbt_status'] = f"Error: {type(e).__name__}: {e}"

    except Exception as e:
        out['numba_status'] = out['vbt_status'] = f"Error: {type(e).__name__}: {e}"
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-folder', default=os.environ.get('BT_DATA_FOLDER', 'data_sample/'))
    ap.add_argument('--num-symbols', type=int, default=250)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    csv_files = sorted(f for f in os.listdir(args.data_folder) if f.endswith('.csv'))[:args.num_symbols]
    file_paths = [os.path.join(args.data_folder, f) for f in csv_files]
    symbol_names = [os.path.splitext(f)[0] for f in csv_files]

    print(f"{len(STRATEGIES)} strategies x {len(csv_files)} symbols = {len(STRATEGIES)*len(csv_files)} pairs, {args.workers} workers\n")

    t0 = time.perf_counter()
    all_rows = []
    for si, strategy in enumerate(STRATEGIES, 1):
        print(f"--- [{si}/{len(STRATEGIES)}] {strategy}", flush=True)
        rows = Parallel(n_jobs=args.workers, backend='loky', verbose=10)(
            delayed(run_pair)(fp, sn, strategy) for fp, sn in zip(file_paths, symbol_names)
        )
        all_rows.extend(rows)
        both_ok = [r for r in rows if r['numba_status'] == 'Success' and r['vbt_status'] == 'Success']
        n_ok = len(both_ok)
        n_err = sum(1 for r in rows if r['numba_status'].startswith('Error') or r['vbt_status'].startswith('Error'))
        exact = sum(1 for r in both_ok if r['numba_trades'] == r['vbt_trades'])
        mean_diff = np.mean([abs(r['numba_trades'] - r['vbt_trades']) for r in both_ok]) if both_ok else float('nan')
        nsh = np.array([r['numba_sharpe'] for r in both_ok], dtype=float)
        vsh = np.array([r['vbt_sharpe'] for r in both_ok], dtype=float)
        mask = ~np.isnan(nsh) & ~np.isnan(vsh)
        corr = np.corrcoef(nsh[mask], vsh[mask])[0, 1] if mask.sum() > 2 else float('nan')
        pct = 100 * exact / n_ok if n_ok else float('nan')
        print(f"[{si:>2}/{len(STRATEGIES)}] both_ok={n_ok:<4} errors={n_err:<3} exact_trade_match={exact:>3}/{n_ok:<3}({pct:5.1f}%) mean_abs_diff={mean_diff:5.2f} sharpe_corr={corr:5.3f}  {strategy[:70]}", flush=True)

    dt = time.perf_counter() - t0
    print(f"\nTotal time: {dt:.1f}s for {len(STRATEGIES)} strategies x {len(csv_files)} symbols", flush=True)

    out_path = os.path.join('evidence', 'strategy_parity_50_strategies_250_symbols.csv')
    os.makedirs('evidence', exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"Full per-symbol-per-strategy detail written to {out_path}")
