# baseline_vbt_talib.py
# Reference baseline: identical RSI/EMA/ATR strategy executed through vectorbt's
# Portfolio.from_signals (TA-Lib indicators). Used as the ground truth to validate
# numba_engine.py's custom execution loop against, and as the speed comparison point.
import vectorbt as vbt
import pandas as pd
import numpy as np
import time
import os
from tqdm import tqdm

# --- Strategy Parameters (Global) ---
RSI_PERIOD = 14
EMA_SHORT_PERIOD = 10
EMA_LONG_PERIOD = 50
ATR_PERIOD = 14
INIT_CASH = 100000.0
FEES = 0.001
SLIPPAGE = 0.001
# SL/TP parameters
ATR_MULTIPLIER_FOR_SL = 2.0
RR_RATIO_FOR_TP = 3.0

DATA_FOLDER = os.environ.get("BT_DATA_FOLDER", "data_sample/")
OUTPUT_CSV_FILENAME = "results/baseline_vbt_talib_report.csv"

METRIC_NAMES = [
    "Start", "End", "Period", "Start Value", "End Value", "Total Return [%]",
    "Benchmark Return [%]", "Max Gross Exposure [%]", "Total Fees Paid",
    "Max Drawdown [%]", "Max Drawdown Duration", "Total Trades", "Total Closed Trades",
    "Total Open Trades", "Open Trade PnL", "Win Rate [%]", "Best Trade [%]",
    "Worst Trade [%]", "Avg Winning Trade [%]", "Avg Losing Trade [%]",
    "Avg Winning Trade Duration", "Avg Losing Trade Duration", "Profit Factor",
    "Expectancy", "Sharpe Ratio", "Calmar Ratio", "Omega Ratio", "Sortino Ratio"
]
TIMING_COLUMNS = ['load_time_s', 'indicator_calc_time_s', 'signal_gen_time_s', 'portfolio_sim_time_s', 'total_symbol_time_s']
OTHER_COLUMNS = ['symbol', 'status']
CSV_COLUMNS = OTHER_COLUMNS + TIMING_COLUMNS + METRIC_NAMES


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


def run_strategy_for_symbol_vbt_talib(file_path, symbol_name):
    symbol_results = {col: np.nan for col in CSV_COLUMNS}
    symbol_results['symbol'] = symbol_name
    symbol_results['status'] = 'Failed'
    overall_symbol_start_time = time.perf_counter()

    try:
        load_start_time = time.perf_counter()
        price_df = pd.read_csv(
            file_path, index_col='date', parse_dates=True,
            usecols=['date', 'open', 'high', 'low', 'close', 'volume']
        )
        price_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        price_df.sort_index(inplace=True)

        min_required_length = max(RSI_PERIOD, EMA_LONG_PERIOD, ATR_PERIOD) + 20
        if len(price_df) < min_required_length:
            symbol_results['status'] = f"Skipped: Data rows {len(price_df)} < {min_required_length}"
            symbol_results['total_symbol_time_s'] = time.perf_counter() - overall_symbol_start_time
            return symbol_results

        load_end_time = time.perf_counter()
        symbol_results['load_time_s'] = load_end_time - load_start_time

        indicator_start_time = time.perf_counter()
        close_prices = price_df['Close']
        high_prices = price_df['High']
        low_prices = price_df['Low']
        open_prices = price_df['Open']

        rsi = vbt.talib('RSI').run(close_prices, timeperiod=RSI_PERIOD).real
        ema_short = vbt.talib('EMA').run(close_prices, timeperiod=EMA_SHORT_PERIOD).real
        ema_long = vbt.talib('EMA').run(close_prices, timeperiod=EMA_LONG_PERIOD).real
        atr = vbt.talib('ATR').run(high_prices, low_prices, close_prices, timeperiod=ATR_PERIOD).real
        indicator_end_time = time.perf_counter()
        symbol_results['indicator_calc_time_s'] = indicator_end_time - indicator_start_time

        signal_start_time = time.perf_counter()
        rsi_crossed_above_40 = (rsi > 40) & (rsi.shift(1) <= 40)
        trend_filter = ema_short > ema_long
        base_entry_signals = rsi_crossed_above_40 & trend_filter

        atr_on_signal_bar = atr.shift(1)
        valid_atr_mask = atr_on_signal_bar.notna() & (atr_on_signal_bar > 1e-9)

        final_entry_signals = base_entry_signals & valid_atr_mask
        final_entry_signals = final_entry_signals.fillna(False)
        signal_end_time = time.perf_counter()
        symbol_results['signal_gen_time_s'] = signal_end_time - signal_start_time

        portfolio_start_time = time.perf_counter()

        # SL/TP offsets in points -> converted to percentages of entry price for vectorbt's sl_stop/tp_stop
        sl_offset_points = atr_on_signal_bar * ATR_MULTIPLIER_FOR_SL
        tp_offset_points = sl_offset_points * RR_RATIO_FOR_TP

        safe_open_prices = open_prices.replace(0, np.nan).where(open_prices > 0, np.nan)

        sl_stop_pct_series = sl_offset_points / safe_open_prices
        tp_stop_pct_series = tp_offset_points / safe_open_prices

        pf = vbt.Portfolio.from_signals(
            close=close_prices, open=open_prices, high=high_prices, low=low_prices,
            entries=final_entry_signals,
            sl_stop=sl_stop_pct_series,
            tp_stop=tp_stop_pct_series,
            init_cash=INIT_CASH, fees=FEES, slippage=SLIPPAGE, freq='D'
        )
        stats = pf.stats()
        portfolio_end_time = time.perf_counter()
        symbol_results['portfolio_sim_time_s'] = portfolio_end_time - portfolio_start_time

        for metric_key in METRIC_NAMES:
            symbol_results[metric_key] = format_stat_value(stats.get(metric_key, np.nan))

        symbol_results['status'] = 'Success'

    except Exception as e:
        symbol_results['status'] = f"Error: {type(e).__name__}"

    symbol_results['total_symbol_time_s'] = time.perf_counter() - overall_symbol_start_time
    return symbol_results


if __name__ == '__main__':
    main_start_time = time.perf_counter()

    if not os.path.isdir(DATA_FOLDER):
        print(f"FATAL: Data folder '{DATA_FOLDER}' not found.")
        exit()

    all_symbol_results = []
    try:
        csv_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv') and os.path.isfile(os.path.join(DATA_FOLDER, f))]
    except FileNotFoundError:
        print(f"FATAL: Data folder '{DATA_FOLDER}' not found when listing files.")
        exit()

    if not csv_files:
        print(f"No CSV files found in '{DATA_FOLDER}'.")
        exit()

    print(f"Found {len(csv_files)} CSV files. Processing with vectorbt (ATR SL/TP)...\n")
    os.makedirs(os.path.dirname(OUTPUT_CSV_FILENAME), exist_ok=True)

    for csv_file in tqdm(csv_files, desc="Processing (vectorbt, ATR SL/TP)"):
        symbol_name = os.path.splitext(csv_file)[0]
        file_path = os.path.join(DATA_FOLDER, csv_file)
        result = run_strategy_for_symbol_vbt_talib(file_path, symbol_name)
        all_symbol_results.append(result)
        if result['status'] != 'Success':
            tqdm.write(f"Symbol {symbol_name}: {result['status']} (Time: {result.get('total_symbol_time_s', 0):.2f}s)")

    main_end_time = time.perf_counter()
    print(f"\n\nFinished processing {len(csv_files)} symbols in {main_end_time - main_start_time:.2f} seconds.")

    results_df = pd.DataFrame(all_symbol_results, columns=CSV_COLUMNS)
    results_df.to_csv(OUTPUT_CSV_FILENAME, index=False)
    print(f"\nSummary report saved to {OUTPUT_CSV_FILENAME}")

    successful_runs = results_df[results_df['status'] == 'Success'].copy()
    if not successful_runs.empty:
        for col in TIMING_COLUMNS:
            successful_runs[col] = pd.to_numeric(successful_runs[col], errors='coerce')

        print("\nAverage processing times (successful runs):")
        for col in TIMING_COLUMNS:
            print(f"  Avg {col.replace('_s', '').replace('_', ' ').capitalize():<27}: {successful_runs[col].mean():.4f} s")
    else:
        print("\nNo symbols processed successfully for average times.")

    status_counts = results_df['status'].value_counts()
    print("\nProcessing Status Summary:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
