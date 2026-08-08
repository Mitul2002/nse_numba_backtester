# baseline_library_vbt.py
# The vectorbt-side counterpart to strategy_library_engine.py: same generalization
# (arbitrary strategy string via alpha_strategy_parser), but executed through vectorbt
# instead of the custom Numba loop, using the exact fixed conventions established in
# baseline_vbt_talib.py (price=open, stop_entry_price='fillprice', entries shifted one
# bar) -- so this is a like-for-like reference to validate strategy_library_engine.py
# against, the same way baseline_vbt_talib.py validated numba_engine.py.
#
# Signal generation itself (generate_entry_signals) is imported, not reimplemented --
# both engines parse and evaluate the SAME strategy string through the SAME
# alpha_strategy_parser call, so any divergence between the two reports is coming from
# the execution/simulation layer, not from two different interpretations of the
# strategy string.
import os
import time
import numpy as np
import pandas as pd
import talib
import vectorbt as vbt

from strategy_library_engine import generate_entry_signals  # noqa: E402
from numba_engine import ATR_PERIOD, ATR_MULTIPLIER_FOR_SL, RR_RATIO_FOR_TP, INIT_CASH, FEES_PCT, SLIPPAGE_PCT  # noqa: E402

METRIC_NAMES = [
    "Start", "End", "Period", "Start Value", "End Value", "Total Return [%]",
    "Benchmark Return [%]", "Max Gross Exposure [%]", "Total Fees Paid",
    "Max Drawdown [%]", "Max Drawdown Duration", "Total Trades", "Total Closed Trades",
    "Total Open Trades", "Open Trade PnL", "Win Rate [%]", "Best Trade [%]",
    "Worst Trade [%]", "Avg Winning Trade [%]", "Avg Losing Trade [%]",
    "Avg Winning Trade Duration", "Avg Losing Trade Duration", "Profit Factor",
    "Expectancy", "Sharpe Ratio", "Calmar Ratio", "Omega Ratio", "Sortino Ratio"
]


def format_stat_value(value):
    if isinstance(value, pd.Timestamp):
        return value.strftime('%Y-%m-%d %H:%M:%S')
    if isinstance(value, pd.Timedelta):
        days = value.days
        seconds = value.seconds
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{days} days {hours:02d}:{minutes:02d}:{secs:02d}"
    if isinstance(value, (float, np.floating)):
        return round(value, 6)
    return value


def run_strategy_for_symbol_vbt(file_path: str, symbol_name: str, strategy_str: str) -> dict:
    result = {col: np.nan for col in ['symbol', 'status'] + METRIC_NAMES}
    result['symbol'] = symbol_name
    result['status'] = 'Failed'
    t0 = time.perf_counter()

    try:
        header_cols = pd.read_csv(file_path, nrows=0).columns
        date_col = 'date' if 'date' in header_cols else 'datetime'
        raw_df = pd.read_csv(
            file_path, index_col=date_col, parse_dates=True,
            usecols=[date_col, 'open', 'high', 'low', 'close', 'volume']
        )
        raw_df.sort_index(inplace=True)

        if len(raw_df) < 90:
            result['status'] = f"Skipped: Data rows {len(raw_df)} < 90"
            result['total_symbol_time_s'] = time.perf_counter() - t0
            return result

        entry_signals = generate_entry_signals(strategy_str, raw_df)
        entries = pd.Series(entry_signals, index=raw_df.index)

        open_p, high_p, low_p, close_p = raw_df['open'], raw_df['high'], raw_df['low'], raw_df['close']
        atr = pd.Series(
            talib.ATR(high_p.to_numpy(dtype=np.double), low_p.to_numpy(dtype=np.double),
                      close_p.to_numpy(dtype=np.double), timeperiod=ATR_PERIOD),
            index=raw_df.index
        )
        atr_on_signal_bar = atr.shift(1)

        # Same fixed conventions as baseline_vbt_talib.py: entries shifted one bar to
        # the action bar (matching entry_signals[i-1] in the Numba loop), filled at
        # that bar's open, SL/TP anchored to the actual fill price.
        entries_shifted = entries.shift(1).fillna(False).astype(bool)
        safe_open = open_p.replace(0, np.nan).where(open_p > 0, np.nan)
        sl_pct = (atr_on_signal_bar * ATR_MULTIPLIER_FOR_SL) / safe_open
        tp_pct = sl_pct * RR_RATIO_FOR_TP

        pf = vbt.Portfolio.from_signals(
            close=close_p, open=open_p, high=high_p, low=low_p,
            entries=entries_shifted, price=open_p,
            sl_stop=sl_pct, tp_stop=tp_pct, stop_entry_price='fillprice',
            init_cash=INIT_CASH, fees=FEES_PCT, slippage=SLIPPAGE_PCT, freq='D'
        )
        stats = pf.stats()
        for k in METRIC_NAMES:
            result[k] = format_stat_value(stats.get(k, np.nan))
        result['status'] = 'Success'

    except Exception as e:
        result['status'] = f"Error: {type(e).__name__}: {e}"

    result['total_symbol_time_s'] = time.perf_counter() - t0
    return result


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--strategy', required=True)
    ap.add_argument('--data-folder', default=os.environ.get('BT_DATA_FOLDER', 'data_sample/'))
    ap.add_argument('--limit', type=int, default=10)
    args = ap.parse_args()

    csv_files = sorted(f for f in os.listdir(args.data_folder) if f.endswith('.csv'))[:args.limit]
    print(f"Strategy: {args.strategy}")
    print(f"Running (vectorbt) on {len(csv_files)} symbols from {args.data_folder}\n")
    for f in csv_files:
        symbol = os.path.splitext(f)[0]
        r = run_strategy_for_symbol_vbt(os.path.join(args.data_folder, f), symbol, args.strategy)
        print(f"  {symbol:<15} status={r['status']:<10} trades={r.get('Total Trades')}  return={r.get('Total Return [%]')}")
