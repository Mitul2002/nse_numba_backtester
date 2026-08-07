# validate_parity.py
# Element-wise numerical parity check between baseline_vbt_talib.py and numba_engine.py
# outputs. Tolerance is intentionally tight (rtol=1e-4, atol=1e-6) so it flags essentially
# any floating-point-level divergence between the two independently implemented engines,
# not just economically meaningful differences.
import pandas as pd
import numpy as np
import os

FILE1_PATH = "results/baseline_vbt_talib_report.csv"
FILE2_PATH = "results/numba_engine_report.csv"
OUTPUT_MISMATCH_CSV = "results/mismatches_detailed_per_source.csv"

COLS_TO_COMPARE = [
    "Start Value", "End Value", "Total Return [%]", "Benchmark Return [%]",
    "Max Gross Exposure [%]", "Total Fees Paid", "Max Drawdown [%]",
    "Max Drawdown Duration", "Total Trades", "Total Closed Trades",
    "Total Open Trades", "Open Trade PnL", "Win Rate [%]", "Best Trade [%]",
    "Worst Trade [%]", "Avg Winning Trade [%]", "Avg Losing Trade [%]",
    "Avg Winning Trade Duration", "Avg Losing Trade Duration",
    "Profit Factor", "Expectancy", "Sharpe Ratio", "Calmar Ratio",
    "Omega Ratio", "Sortino Ratio"
]

RTOL = 1e-4
ATOL = 1e-6

print("Loading data...")
df1_original = pd.read_csv(FILE1_PATH)
df2_original = pd.read_csv(FILE2_PATH)
print(f"Loaded {len(df1_original)} rows from {FILE1_PATH}, {len(df2_original)} rows from {FILE2_PATH}.")

df1 = df1_original.copy()
df2 = df2_original.copy()

# vectorbt reports NaN for "Avg Winning Trade [%]" / duration when a symbol had zero
# winning trades; the custom engine reports 0.0 / "0 days" for the same case. Align
# semantics before comparing so a "no wins" symbol doesn't spuriously mismatch.
if "Avg Winning Trade [%]" in df1.columns:
    df1["Avg Winning Trade [%]"] = pd.to_numeric(df1["Avg Winning Trade [%]"], errors='coerce')
    df1.loc[df1["Avg Winning Trade [%]"].isna(), "Avg Winning Trade [%]"] = 0.0

if "Avg Winning Trade Duration" in df1.columns:
    df1["Avg Winning Trade Duration"] = df1["Avg Winning Trade Duration"].astype(object)
    nan_mask = df1["Avg Winning Trade Duration"].isna() | \
        (df1["Avg Winning Trade Duration"].astype(str).str.lower() == 'nan') | \
        (df1["Avg Winning Trade Duration"].astype(str).str.strip() == '')
    df1.loc[nan_mask, "Avg Winning Trade Duration"] = "0 days 00:00:00"

all_needed_cols = ["symbol"] + COLS_TO_COMPARE
cols_to_compare_final = COLS_TO_COMPARE.copy()
for current_df, name in [(df1, "baseline (modified)"), (df2, "numba engine")]:
    missing = [c for c in all_needed_cols if c not in current_df.columns]
    if missing:
        print(f"WARNING: Missing columns in {name}: {missing}")
        for m_col in missing:
            if m_col in cols_to_compare_final:
                cols_to_compare_final.remove(m_col)

df1_sub = df1[["symbol"] + cols_to_compare_final].copy()
df2_sub = df2[["symbol"] + cols_to_compare_final].copy()

merged_df = pd.merge(df1_sub, df2_sub, on="symbol", suffixes=('_vbt', '_custom'), how='inner')
if merged_df.empty:
    print("Merge resulted in an empty DataFrame -- no common symbols between the two reports.")
    exit()

print("\nPerforming element-wise comparison...")
all_comparisons = []
for col in cols_to_compare_final:
    s1 = merged_df[col + '_vbt']
    s2 = merged_df[col + '_custom']

    s1_numeric = pd.to_numeric(s1, errors='coerce')
    s2_numeric = pd.to_numeric(s2, errors='coerce')

    col_match_series = pd.Series(False, index=s1.index)

    both_nan_mask = s1_numeric.isna() & s2_numeric.isna()
    col_match_series[both_nan_mask] = True

    compare_numeric_mask = ~both_nan_mask & s1_numeric.notna() & s2_numeric.notna()
    if compare_numeric_mask.any():
        col_match_series[compare_numeric_mask] = np.isclose(
            s1_numeric[compare_numeric_mask], s2_numeric[compare_numeric_mask],
            rtol=RTOL, atol=ATOL, equal_nan=False
        )

    potential_str_compare_mask = ~col_match_series & s1.notna() & s2.notna()
    if potential_str_compare_mask.any():
        s1_eval = s1[potential_str_compare_mask].astype(str).str.strip()
        s2_eval = s2[potential_str_compare_mask].astype(str).str.strip()
        col_match_series[potential_str_compare_mask] = (s1_eval == s2_eval)

    all_comparisons.append(col_match_series.rename(col))

comparison_df = pd.concat(all_comparisons, axis=1)
total_elements = comparison_df.size
matched_elements = int(comparison_df.sum().sum())
mismatched_elements = total_elements - matched_elements
match_pct = (matched_elements / total_elements) * 100.0 if total_elements > 0 else 0.0

print(f"\n--- Overall Comparison Summary (tolerance: rtol={RTOL}, atol={ATOL}) ---")
print(f"Symbols compared: {len(merged_df)}")
print(f"Metric columns compared: {len(cols_to_compare_final)}")
print(f"Total elements compared: {total_elements}")
print(f"Matched elements: {matched_elements} ({match_pct:.2f}%)")
print(f"Mismatched elements: {mismatched_elements} ({100.0 - match_pct:.2f}%)")

symbols_fully_matched = int((comparison_df.all(axis=1)).sum())
print(f"Symbols with ALL {len(cols_to_compare_final)} metrics within tolerance: {symbols_fully_matched} / {len(merged_df)}")

# Trade-count agreement specifically -- the metric that most directly proves the two
# execution engines make the same entry/exit decisions, independent of downstream
# ratio-calculation rounding.
if "Total Trades_vbt" in merged_df.columns or "Total Trades" in cols_to_compare_final:
    tt1 = pd.to_numeric(merged_df["Total Trades_vbt"], errors='coerce')
    tt2 = pd.to_numeric(merged_df["Total Trades_custom"], errors='coerce')
    exact_trade_match = int((tt1 == tt2).sum())
    print(f"Symbols with EXACTLY matching trade count: {exact_trade_match} / {len(merged_df)} "
          f"(mean abs diff: {(tt1 - tt2).abs().mean():.3f} trades)")

mismatch_output_list = []
for idx in merged_df.index:
    symbol = merged_df.loc[idx, "symbol"]
    row = comparison_df.loc[idx]
    if not row.all():
        mismatched_metrics = [c for c, ok in row.items() if not ok]
        row1 = {'Symbol': symbol, 'Source': 'File 1 (vectorbt)'}
        row2 = {'Symbol': symbol, 'Source': 'File 2 (custom)'}
        for m in mismatched_metrics:
            row1[m] = merged_df.loc[idx, m + '_vbt']
            row2[m] = merged_df.loc[idx, m + '_custom']
        mismatch_output_list.append(row1)
        mismatch_output_list.append(row2)

if mismatch_output_list:
    os.makedirs(os.path.dirname(OUTPUT_MISMATCH_CSV), exist_ok=True)
    mismatches_df = pd.DataFrame(mismatch_output_list)
    ordered_cols = ['Symbol', 'Source'] + sorted([c for c in mismatches_df.columns if c not in ['Symbol', 'Source']])
    mismatches_df[ordered_cols].to_csv(OUTPUT_MISMATCH_CSV, index=False)
    print(f"\nPer-symbol mismatch detail written to '{OUTPUT_MISMATCH_CSV}'")
else:
    print("\nNo mismatches found within tolerance.")
