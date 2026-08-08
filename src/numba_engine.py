# numba_engine.py
# Custom vectorized-signal + Numba-JIT execution backtest engine.
# Entry: RSI(14) crosses above 40 while EMA(10) > EMA(50) (trend filter).
# Exit: ATR-based dynamic stop-loss (2x ATR) / take-profit (3:1 reward:risk), simulated
# bar-by-bar inside a single @njit-compiled loop (no Python-level per-bar overhead).
import pandas as pd
import numpy as np
import time
import os
import talib
from joblib import Parallel, delayed
from tqdm import tqdm
from numba import njit, float64, int64
from numba.typed import List as NumbaList

# --- Strategy Parameters (Global) ---
RSI_PERIOD = 14
EMA_SHORT_PERIOD = 10
EMA_LONG_PERIOD = 50
ATR_PERIOD = 14  # For ATR calculation
INIT_CASH = 100000.0
FEES_PCT = 0.001
SLIPPAGE_PCT = 0.001
# SL/TP parameters
ATR_MULTIPLIER_FOR_SL = 2.0
RR_RATIO_FOR_TP = 3.0

DATA_FOLDER = os.environ.get("BT_DATA_FOLDER", "data_sample/")
OUTPUT_CSV_FILENAME = "results/numba_engine_report.csv"
# 1 = current, default, sequential behavior. >1 spawns that many worker processes via
# joblib (loky backend). Empirically (see WALKTHROUGH.md), the sweet spot on this
# machine was ~6-8 workers -- both fewer and many more were slower, because each symbol
# is only ~8ms of actual work and past a handful of processes, spawn/IPC overhead
# outweighs the gain. Also benchmarked stdlib ProcessPoolExecutor (slower, ~9.5s at its
# own best of 4 workers) and dask's processes scheduler (~10% faster than joblib at
# 6.2s, but not worth the much heavier dependency for this embarrassingly-parallel
# workload). Benchmark on your own hardware before assuming a specific number transfers.
BT_WORKERS = int(os.environ.get("BT_WORKERS", "1"))

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


@njit
def custom_backtest_numba_loop_atr(
    open_prices: np.ndarray, high_prices: np.ndarray, low_prices: np.ndarray, close_prices: np.ndarray,
    atr_values: np.ndarray,  # ATR series (original, not shifted yet)
    entry_signals: np.ndarray, initial_cash: float64,
    atr_multiplier_sl: float64, rr_ratio_tp: float64,
    fee_pct: float64, slippage_pct: float64
):
    n_days = len(open_prices)
    cash = initial_cash
    equity = np.full(n_days, initial_cash, dtype=float64)
    current_shares = float64(0.0)
    entry_price_avg_cost_per_share = float64(0.0)
    entry_price_for_sl_tp_calc = float64(0.0)

    stop_loss_price = float64(0.0)
    take_profit_price = float64(0.0)
    in_position = False
    total_fees_paid_accumulator = float64(0.0)

    trades = NumbaList()
    current_trade_entry_idx = int64(0)

    for i in range(1, n_days):
        equity[i-1] = cash + current_shares * close_prices[i-1] if in_position else cash

        current_open = open_prices[i]
        current_high = high_prices[i]
        current_low = low_prices[i]
        exit_reason_code = 0
        just_exited = False

        if in_position:
            exit_price_triggered = float64(0.0)
            # Check SL/TP only if they are validly set (not inf)
            if stop_loss_price > -np.inf and current_low <= stop_loss_price:  # SL is valid and hit
                exit_price_triggered = stop_loss_price
                exit_reason_code = 1
            elif take_profit_price < np.inf and current_high >= take_profit_price:  # TP is valid and hit
                exit_price_triggered = take_profit_price
                exit_reason_code = 2

            if exit_reason_code > 0:
                actual_exit_price_before_fees = exit_price_triggered * (1.0 - slippage_pct)
                proceeds_before_fees = current_shares * actual_exit_price_before_fees
                exit_fees = proceeds_before_fees * fee_pct
                total_fees_paid_accumulator += exit_fees
                cash += (proceeds_before_fees - exit_fees)

                pnl_trade = (proceeds_before_fees - exit_fees) - (current_shares * entry_price_avg_cost_per_share)
                return_pct_trade = (actual_exit_price_before_fees / entry_price_for_sl_tp_calc - 1.0) if entry_price_for_sl_tp_calc > 1e-9 else 0.0

                trades.append((
                    current_trade_entry_idx, int64(i), current_shares,
                    entry_price_for_sl_tp_calc,
                    actual_exit_price_before_fees * (1.0 - fee_pct),
                    pnl_trade, return_pct_trade
                ))
                in_position = False
                current_shares = float64(0.0)
                just_exited = True

        # Entry: signal from previous bar (i-1), entry at open of current bar (i).
        # `not just_exited` matches vectorbt's from_signals behavior in this
        # configuration, which does not register a new entry on the same bar as a
        # stop exit -- without this, the two engines disagree by exactly one trade
        # on any symbol where a fresh signal happens to land on an exit bar.
        if not in_position and not just_exited and i > 0 and entry_signals[i-1] and cash > 1.0:
            entry_price_for_sl_tp_calc = current_open * (1.0 + slippage_pct)

            if entry_price_for_sl_tp_calc > 1e-9:
                cost_per_share_for_sizing = entry_price_for_sl_tp_calc * (1.0 + fee_pct)
                if cost_per_share_for_sizing > 1e-9:
                    shares_to_buy = np.floor(cash / cost_per_share_for_sizing)

                    if shares_to_buy > 0.0:
                        entry_cost_with_slippage_no_fees = shares_to_buy * entry_price_for_sl_tp_calc
                        entry_fees = entry_cost_with_slippage_no_fees * fee_pct
                        total_entry_cost = entry_cost_with_slippage_no_fees + entry_fees

                        if cash >= total_entry_cost:
                            cash -= total_entry_cost
                            current_shares = shares_to_buy
                            entry_price_avg_cost_per_share = total_entry_cost / shares_to_buy if shares_to_buy > 0 else 0.0

                            # Calculate dynamic SL/TP based on ATR of signal bar (i-1)
                            atr_of_signal_bar = atr_values[i-1]

                            if not np.isnan(atr_of_signal_bar) and atr_of_signal_bar > 1e-9:
                                risk_per_share = atr_of_signal_bar * atr_multiplier_sl
                                stop_loss_price = entry_price_for_sl_tp_calc - risk_per_share
                                take_profit_price = entry_price_for_sl_tp_calc + (risk_per_share * rr_ratio_tp)
                            else:  # Invalid ATR, effectively disable SL/TP by setting them far
                                stop_loss_price = -np.inf
                                take_profit_price = np.inf

                            in_position = True
                            current_trade_entry_idx = int64(i)

        equity[i] = cash + current_shares * close_prices[i] if in_position else cash

    # A position still open when the data ends never gets appended to `trades` (that
    # only happens on an SL/TP exit), so without this it silently vanishes from every
    # trade-count metric -- vectorbt's Total Trades includes still-open positions,
    # marked-to-market at the last close.
    open_shares = current_shares if in_position else float64(0.0)
    open_entry_price = entry_price_avg_cost_per_share if in_position else float64(0.0)
    open_entry_idx = current_trade_entry_idx if in_position else int64(-1)

    return equity, trades, total_fees_paid_accumulator, open_shares, open_entry_price, open_entry_idx


def format_timedelta_custom(td_obj_or_days_val):
    if pd.isna(td_obj_or_days_val):
        return ""
    td_obj = None
    if isinstance(td_obj_or_days_val, (int, float, np.integer, np.floating)):
        if np.isnan(td_obj_or_days_val):
            return ""
        try:
            td_obj = pd.Timedelta(days=float(td_obj_or_days_val))
        except OverflowError:
            return f"{td_obj_or_days_val:.0f} days (approx)"
    elif isinstance(td_obj_or_days_val, pd.Timedelta):
        td_obj = td_obj_or_days_val
    else:
        return str(td_obj_or_days_val)
    if td_obj is None:
        return ""
    days = td_obj.days
    seconds = td_obj.seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{days} days {hours:02d}:{minutes:02d}:{secs:02d}"


def calculate_custom_metrics(price_df_indexed, equity_curve, trades_raw, initial_cash, total_fees_paid_from_loop,
                              open_shares=0.0, open_entry_price=0.0):
    metrics = {col: np.nan for col in METRIC_NAMES}
    dates = price_df_indexed.index
    if not dates.empty:
        metrics['Start'] = dates[0].strftime('%Y-%m-%d %H:%M:%S')
        metrics['End'] = dates[-1].strftime('%Y-%m-%d %H:%M:%S')
        metrics['Period'] = format_timedelta_custom(dates[-1] - dates[0])
    metrics['Start Value'] = float(initial_cash)
    metrics['End Value'] = float(equity_curve[-1]) if len(equity_curve) > 0 else float(initial_cash)
    if initial_cash > 1e-9 and len(equity_curve) > 0:
        metrics['Total Return [%]'] = (equity_curve[-1] / initial_cash - 1.0) * 100.0
    else:
        metrics['Total Return [%]'] = 0.0
    if not price_df_indexed['Close'].empty and abs(price_df_indexed['Close'].iloc[0]) > 1e-9:
        metrics['Benchmark Return [%]'] = (price_df_indexed['Close'].iloc[-1] / price_df_indexed['Close'].iloc[0] - 1.0) * 100.0
    else:
        metrics['Benchmark Return [%]'] = 0.0
    metrics['Max Gross Exposure [%]'] = 100.0
    metrics['Total Fees Paid'] = float(total_fees_paid_from_loop)

    if len(equity_curve) > 1:
        equity_series = pd.Series(equity_curve, index=dates[:len(equity_curve)])
        cumulative_max = equity_series.cummax()
        drawdown_pct_series = (cumulative_max - equity_series) / cumulative_max.replace(0, np.nan)
        drawdown_pct_series.replace([np.inf, -np.inf], np.nan, inplace=True)
        max_dd_val = drawdown_pct_series.max()
        metrics['Max Drawdown [%]'] = max_dd_val * 100.0 if pd.notna(max_dd_val) else 0.0
        if pd.notna(max_dd_val) and max_dd_val > 0:
            try:
                if not drawdown_pct_series.dropna().empty:
                    dd_trough_date = drawdown_pct_series.idxmax()
                    series_before_trough = cumulative_max.loc[:dd_trough_date]
                    peak_value_for_mdd = series_before_trough.loc[dd_trough_date]
                    dd_peak_date_candidates = series_before_trough[series_before_trough == peak_value_for_mdd]
                    if not dd_peak_date_candidates.empty:
                        dd_peak_date = dd_peak_date_candidates.index[0]
                        equity_after_peak = equity_series.loc[dd_peak_date:]
                        recovery_date_candidates = equity_after_peak[equity_after_peak > peak_value_for_mdd]
                        if not recovery_date_candidates.empty:
                            recovery_date = recovery_date_candidates.index[0]
                            metrics['Max Drawdown Duration'] = format_timedelta_custom(recovery_date - dd_peak_date)
                        else:
                            metrics['Max Drawdown Duration'] = format_timedelta_custom(dates[-1] - dd_peak_date)
            except Exception:
                metrics['Max Drawdown Duration'] = format_timedelta_custom(pd.Timedelta(seconds=0))
    num_trades = len(trades_raw) if trades_raw else 0
    has_open_position = open_shares > 1e-9
    metrics['Total Trades'] = num_trades + (1 if has_open_position else 0)
    metrics['Total Closed Trades'] = num_trades
    metrics['Total Open Trades'] = 1 if has_open_position else 0
    if has_open_position and not price_df_indexed['Close'].empty:
        last_close = float(price_df_indexed['Close'].iloc[-1])
        metrics['Open Trade PnL'] = open_shares * (last_close - open_entry_price)
    else:
        metrics['Open Trade PnL'] = 0.0
    if num_trades > 0:
        pnls = np.array([t[5] for t in trades_raw], dtype=np.float64)
        returns_pct_trade = np.array([t[6] for t in trades_raw], dtype=np.float64)
        metrics['Win Rate [%]'] = (np.sum(pnls > 0) / num_trades) * 100.0 if num_trades > 0 else 0.0
        metrics['Best Trade [%]'] = np.max(returns_pct_trade) * 100.0 if len(returns_pct_trade) > 0 else 0.0
        metrics['Worst Trade [%]'] = np.min(returns_pct_trade) * 100.0 if len(returns_pct_trade) > 0 else 0.0
        winning_returns = returns_pct_trade[pnls > 0]
        losing_returns = returns_pct_trade[pnls <= 0]
        metrics['Avg Winning Trade [%]'] = np.mean(winning_returns) * 100.0 if len(winning_returns) > 0 else 0.0
        metrics['Avg Losing Trade [%]'] = np.mean(losing_returns) * 100.0 if len(losing_returns) > 0 else 0.0
        trade_durations_days = []
        if not dates.empty:
            for t in trades_raw:  # t[0] is entry_idx (iloc), t[1] is exit_idx (iloc)
                entry_idx, exit_idx = t[0], t[1]
                if 0 <= entry_idx < len(dates) and 0 <= exit_idx < len(dates):
                    trade_durations_days.append((dates[exit_idx] - dates[entry_idx]).total_seconds() / (24 * 3600.0))
        winning_durations_days = [trade_durations_days[i] for i, pnl in enumerate(pnls) if pnl > 0 and i < len(trade_durations_days)]
        losing_durations_days = [trade_durations_days[i] for i, pnl in enumerate(pnls) if pnl <= 0 and i < len(trade_durations_days)]
        metrics['Avg Winning Trade Duration'] = format_timedelta_custom(np.mean(winning_durations_days) if len(winning_durations_days) > 0 else 0.0)
        metrics['Avg Losing Trade Duration'] = format_timedelta_custom(np.mean(losing_durations_days) if len(losing_durations_days) > 0 else 0.0)
        gross_profit = np.sum(pnls[pnls > 0])
        gross_loss = abs(np.sum(pnls[pnls <= 0]))
        metrics['Profit Factor'] = gross_profit / gross_loss if gross_loss > 1e-9 else np.inf
        metrics['Expectancy'] = np.mean(pnls) if len(pnls) > 0 else 0.0
    if len(equity_curve) > 1:
        daily_returns = pd.Series(equity_curve, index=dates[:len(equity_curve)]).pct_change().dropna()
        if len(daily_returns) > 1:
            mean_daily_return = np.mean(daily_returns)
            std_dev_daily_returns = np.std(daily_returns)
            if std_dev_daily_returns > 1e-9:
                metrics['Sharpe Ratio'] = (mean_daily_return / std_dev_daily_returns) * np.sqrt(252.0)
            annual_return_geom = 0.0
            if initial_cash > 1e-9 and len(equity_curve) > 0:
                num_periods = len(equity_curve)
                annual_return_geom = (equity_curve[-1] / initial_cash) ** (252.0 / num_periods) - 1.0 if num_periods > 0 else 0.0
            max_dd_pct_val = metrics.get('Max Drawdown [%]', np.nan)
            if pd.notna(max_dd_pct_val) and max_dd_pct_val > 1e-9:
                metrics['Calmar Ratio'] = annual_return_geom / (max_dd_pct_val / 100.0)
            downside_returns = daily_returns[daily_returns < 0.0]
            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns)
                if downside_std > 1e-9:
                    metrics['Sortino Ratio'] = (mean_daily_return / downside_std) * np.sqrt(252.0)
            sum_positive_returns = np.sum(daily_returns[daily_returns > 0.0])
            sum_abs_negative_returns = np.sum(np.abs(daily_returns[daily_returns < 0.0]))
            if sum_abs_negative_returns > 1e-9:
                metrics['Omega Ratio'] = sum_positive_returns / sum_abs_negative_returns
    for key in METRIC_NAMES:
        if isinstance(metrics.get(key), (float, np.floating)) and pd.notna(metrics.get(key)):
            metrics[key] = round(metrics[key], 6)
    return metrics


def run_strategy_for_symbol_custom(file_path, symbol_name):
    symbol_results = {col: np.nan for col in CSV_COLUMNS}
    symbol_results['symbol'] = symbol_name
    symbol_results['status'] = 'Failed'
    overall_symbol_start_time = time.perf_counter()

    try:
        load_start_time = time.perf_counter()
        header_cols = pd.read_csv(file_path, nrows=0).columns
        date_col = 'date' if 'date' in header_cols else 'datetime'
        price_df = pd.read_csv(
            file_path, index_col=date_col, parse_dates=True,
            usecols=[date_col, 'open', 'high', 'low', 'close', 'volume']
        )
        price_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        price_df.sort_index(inplace=True)

        min_required_length = max(RSI_PERIOD, EMA_LONG_PERIOD, ATR_PERIOD) + 5
        if len(price_df) < min_required_length:
            symbol_results['status'] = f"Skipped: Data rows {len(price_df)} < {min_required_length}"
            symbol_results['total_symbol_time_s'] = time.perf_counter() - overall_symbol_start_time
            return symbol_results

        load_end_time = time.perf_counter()
        symbol_results['load_time_s'] = load_end_time - load_start_time

        indicator_start_time = time.perf_counter()
        close_np = np.ascontiguousarray(price_df['Close'].to_numpy(dtype=np.double))
        high_np = np.ascontiguousarray(price_df['High'].to_numpy(dtype=np.double))
        low_np = np.ascontiguousarray(price_df['Low'].to_numpy(dtype=np.double))
        open_np = np.ascontiguousarray(price_df['Open'].to_numpy(dtype=np.double))

        rsi_np_talib = talib.RSI(close_np, timeperiod=RSI_PERIOD)
        ema_short_np_talib = talib.EMA(close_np, timeperiod=EMA_SHORT_PERIOD)
        ema_long_np_talib = talib.EMA(close_np, timeperiod=EMA_LONG_PERIOD)
        atr_np_talib = talib.ATR(high_np, low_np, close_np, timeperiod=ATR_PERIOD)
        indicator_end_time = time.perf_counter()
        symbol_results['indicator_calc_time_s'] = indicator_end_time - indicator_start_time

        signal_start_time = time.perf_counter()
        rsi_pd = pd.Series(rsi_np_talib, index=price_df.index)
        ema_short_pd = pd.Series(ema_short_np_talib, index=price_df.index)
        ema_long_pd = pd.Series(ema_long_np_talib, index=price_df.index)
        atr_pd_shifted_for_signal = pd.Series(atr_np_talib, index=price_df.index).shift(1)

        if rsi_pd.isnull().all() or ema_short_pd.isnull().all() or ema_long_pd.isnull().all() or atr_pd_shifted_for_signal.isnull().all():
            symbol_results['status'] = "Skipped: Base indicators or shifted ATR resulted in all NaNs"
            symbol_results['signal_gen_time_s'] = time.perf_counter() - signal_start_time
            symbol_results['total_symbol_time_s'] = time.perf_counter() - overall_symbol_start_time
            return symbol_results

        rsi_crossed = (rsi_pd > 40.0) & (rsi_pd.shift(1) <= 40.0)
        trend_ok = ema_short_pd > ema_long_pd
        base_entry_signals = rsi_crossed & trend_ok

        valid_atr_for_entry_mask = atr_pd_shifted_for_signal.notna() & (atr_pd_shifted_for_signal > 1e-9)

        final_entry_signals_np = (base_entry_signals & valid_atr_for_entry_mask).fillna(False).to_numpy()
        signal_end_time = time.perf_counter()
        symbol_results['signal_gen_time_s'] = signal_end_time - signal_start_time

        portfolio_start_time = time.perf_counter()
        equity_curve, trades_raw, total_fees_loop, open_shares, open_entry_price, _open_entry_idx = custom_backtest_numba_loop_atr(
            open_np, high_np, low_np, close_np,
            atr_np_talib,
            final_entry_signals_np,
            float64(INIT_CASH),
            float64(ATR_MULTIPLIER_FOR_SL), float64(RR_RATIO_FOR_TP),
            float64(FEES_PCT), float64(SLIPPAGE_PCT)
        )
        portfolio_end_time = time.perf_counter()
        symbol_results['portfolio_sim_time_s'] = portfolio_end_time - portfolio_start_time

        calculated_metrics = calculate_custom_metrics(price_df, equity_curve, trades_raw, INIT_CASH, total_fees_loop,
                                                        open_shares, open_entry_price)
        for key_metric_name in METRIC_NAMES:
            symbol_results[key_metric_name] = calculated_metrics.get(key_metric_name, np.nan)

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
    os.makedirs(os.path.dirname(OUTPUT_CSV_FILENAME), exist_ok=True)

    if BT_WORKERS > 1:
        print(f"Found {len(csv_files)} CSV files. Processing with Numba engine across {BT_WORKERS} worker processes (joblib/loky)...\n")
        file_paths = [os.path.join(DATA_FOLDER, f) for f in csv_files]
        symbol_names = [os.path.splitext(f)[0] for f in csv_files]
        all_symbol_results = Parallel(n_jobs=BT_WORKERS, backend='loky', verbose=5)(
            delayed(run_strategy_for_symbol_custom)(fp, sn)
            for fp, sn in zip(file_paths, symbol_names)
        )
        for result in all_symbol_results:
            if result['status'] != 'Success':
                print(f"Symbol {result['symbol']}: {result['status']} (Time: {result.get('total_symbol_time_s', 0):.2f}s)")
    else:
        print(f"Found {len(csv_files)} CSV files. Processing with Numba engine (single process, ATR SL/TP)...\n")
        for csv_file in tqdm(csv_files, desc="Processing (Numba, ATR SL/TP)"):
            symbol_name = os.path.splitext(csv_file)[0]
            file_path = os.path.join(DATA_FOLDER, csv_file)
            result = run_strategy_for_symbol_custom(file_path, symbol_name)
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
