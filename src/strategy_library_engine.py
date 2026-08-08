# strategy_library_engine.py
# Generalizes numba_engine.py from one hardcoded strategy to ANY strategy string, by
# delegating entry-signal generation to alpha_strategy_parser's grammar-based parser +
# FunctionRegistry (40+ indicators: RSI, SMA/EMA, MACD, Bollinger Bands, Stochastic,
# ADX/DI, ATR, OBV, MFI, CCI, Williams %R, SAR, crossover, ...).
#
# The Numba execution loop itself (custom_backtest_numba_loop_atr) is untouched -- it
# already takes entry_signals as a plain boolean array and has no idea whether that
# array came from a hardcoded RSI-cross rule or an arbitrary parsed strategy string.
# Only the signal-generation step, which runs in ordinary Python *before* the JIT loop,
# is swapped out. Position sizing, fees, slippage, and the ATR-based dynamic SL/TP
# mechanism are unchanged from numba_engine.py.
import os
import sys
import time
import numpy as np
import pandas as pd
import talib
from numba import float64

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'alpha_strategy_parser', 'src'))
from simple_parser import SimpleStrategyParser   # noqa: E402
from strategy_executor import StrategyExecutor   # noqa: E402

from numba_engine import (  # noqa: E402
    custom_backtest_numba_loop_atr, calculate_custom_metrics,
    METRIC_NAMES, TIMING_COLUMNS, OTHER_COLUMNS,
    INIT_CASH, FEES_PCT, SLIPPAGE_PCT, ATR_MULTIPLIER_FOR_SL, RR_RATIO_FOR_TP, ATR_PERIOD,
)

CSV_COLUMNS = OTHER_COLUMNS + TIMING_COLUMNS + METRIC_NAMES

_parser = SimpleStrategyParser()
_executor = StrategyExecutor()


def generate_entry_signals(strategy_str: str, lowercase_price_df: pd.DataFrame) -> np.ndarray:
    """Parse an arbitrary strategy string and evaluate it against OHLCV data, returning
    a boolean per-bar entry-signal array -- the same shape/semantics as the hardcoded
    RSI-cross array numba_engine.py builds by hand."""
    parsed = _parser.parse(strategy_str)
    if parsed is None:
        raise ValueError(f"Could not parse strategy: {strategy_str}")
    result = _executor.execute(parsed, lowercase_price_df)
    signals = np.asarray(result)
    if signals.dtype != bool:
        signals = np.nan_to_num(signals.astype(np.float64), nan=0.0) != 0.0
    return signals


def run_strategy_for_symbol(file_path: str, symbol_name: str, strategy_str: str) -> dict:
    symbol_results = {col: np.nan for col in CSV_COLUMNS}
    symbol_results['symbol'] = symbol_name
    symbol_results['status'] = 'Failed'
    t0 = time.perf_counter()

    try:
        header_cols = pd.read_csv(file_path, nrows=0).columns
        date_col = 'date' if 'date' in header_cols else 'datetime'
        raw_df = pd.read_csv(
            file_path, index_col=date_col, parse_dates=True,
            usecols=[date_col, 'open', 'high', 'low', 'close', 'volume']
        )
        raw_df.sort_index(inplace=True)

        min_len = 70
        if len(raw_df) < min_len:
            symbol_results['status'] = f"Skipped: Data rows {len(raw_df)} < {min_len}"
            symbol_results['total_symbol_time_s'] = time.perf_counter() - t0
            return symbol_results

        # lowercase-column view for the parser (its FunctionRegistry expects
        # data['close'], data['open'], ... exactly as the raw CSV/parquet provides)
        entry_signals = generate_entry_signals(strategy_str, raw_df)

        capitalized_df = raw_df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        })

        close_np = np.ascontiguousarray(capitalized_df['Close'].to_numpy(dtype=np.double))
        high_np = np.ascontiguousarray(capitalized_df['High'].to_numpy(dtype=np.double))
        low_np = np.ascontiguousarray(capitalized_df['Low'].to_numpy(dtype=np.double))
        open_np = np.ascontiguousarray(capitalized_df['Open'].to_numpy(dtype=np.double))
        atr_np = talib.ATR(high_np, low_np, close_np, timeperiod=ATR_PERIOD)

        equity_curve, trades_raw, total_fees_loop, open_shares, open_entry_price, _ = custom_backtest_numba_loop_atr(
            open_np, high_np, low_np, close_np, atr_np,
            entry_signals.astype(np.bool_),
            float64(INIT_CASH),
            float64(ATR_MULTIPLIER_FOR_SL), float64(RR_RATIO_FOR_TP),
            float64(FEES_PCT), float64(SLIPPAGE_PCT)
        )

        metrics = calculate_custom_metrics(
            capitalized_df, equity_curve, trades_raw, INIT_CASH, total_fees_loop,
            open_shares, open_entry_price
        )
        for k in METRIC_NAMES:
            symbol_results[k] = metrics.get(k, np.nan)
        symbol_results['status'] = 'Success'

    except Exception as e:
        symbol_results['status'] = f"Error: {type(e).__name__}: {e}"

    symbol_results['total_symbol_time_s'] = time.perf_counter() - t0
    return symbol_results


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--strategy', required=True)
    ap.add_argument('--data-folder', default=os.environ.get('BT_DATA_FOLDER', 'data_sample/'))
    ap.add_argument('--limit', type=int, default=10)
    args = ap.parse_args()

    csv_files = sorted(f for f in os.listdir(args.data_folder) if f.endswith('.csv'))[:args.limit]
    print(f"Strategy: {args.strategy}")
    print(f"Running on {len(csv_files)} symbols from {args.data_folder}\n")
    for f in csv_files:
        symbol = os.path.splitext(f)[0]
        r = run_strategy_for_symbol(os.path.join(args.data_folder, f), symbol, args.strategy)
        trades = r.get('Total Trades', 0)
        ret = r.get('Total Return [%]', float('nan'))
        print(f"  {symbol:<15} status={r['status']:<10} trades={trades}  return={ret}")
