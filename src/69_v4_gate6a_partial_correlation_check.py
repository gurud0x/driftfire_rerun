import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# V4 GATE 6a - CONTINUATION of the UNDER REVIEW correlation/ordering check
# (src/68). Two narrowly-scoped diagnostics, NOT a new finding - resolving
# the same open question:
#
#   1. FAMA-MACBETH-STYLE (per-date) CORRELATION: compression_decile vs
#      IV_ATM_var, computed WITHIN each date's own cross-section (matching
#      MIN_XSEC=30, the same date-inclusion convention as the estimator
#      that produced the leave-one-out finding), then summarized across
#      dates - mean, std, % of days with |r| > 0.10. This is the
#      correlation structure the actual Fama-MacBeth estimator sees; the
#      earlier POOLED correlation (src/68, Pearson -0.0177) may be masking
#      it by averaging across regimes/dates.
#
#   2. PARTIAL CORRELATION: compression_decile and IV_ATM_var, each
#      residualized (single pooled OLS, not per-date) against the OTHER 9
#      benchmark features, then correlated. Tests whether the leave-one-out
#      effect (src/67) is a suppression effect - two variables with low
#      raw correlation that still swing each other's coefficients once
#      other correlated regressors are conditioned on.
#
# Raw numbers only. No interpretation. Appended to the SAME UNDER REVIEW
# gate_log.md entry (src/68's), not a new dated entry - this project's
# append-only convention means new content can only be added at the file's
# end, so this is written as an explicit CONTINUATION block, not a fresh
# "## " header, to keep it visually and semantically part of the same
# entry. Still holding on Gate 6b, hedging, and P&L work.
# ---------------------------------------------------------------------------

DEV_START = pd.Timestamp('2015-01-01')
DEV_END = pd.Timestamp('2021-12-31')
CAL_WINDOW_DAYS = 30
TRADING_DAYS_PER_YEAR = 252.0
MIN_XSEC = 30

project_root = Path(__file__).parent.parent
crsp_path = project_root / 'data' / 'processed' / 'crsp_combined.parquet'
univ_path = project_root / 'data' / 'processed' / 'universe_membership.parquet'
features_path = project_root / 'data' / 'processed' / 'v4_gate6a_features_full.parquet'
out_json = project_root / 'results' / '69_v4_gate6a_partial_correlation_check.json'
log_path = project_root / 'results' / 'gate_log.md'

print('=' * 96)
print('V4 GATE 6a - CONTINUATION: per-date correlation + partial correlation')
print('(resolving the same UNDER REVIEW open question as src/68 - not a new finding)')
print('=' * 96)

BENCHMARK_COLS = ['RV1', 'Persist20', 'RV20', 'RV60', 'DSV5', 'JumpVar', 'VoV20',
                  'IV_ATM_var', 'IVTermStruct_var', 'IVSkew_var']
OTHER_9 = [c for c in BENCHMARK_COLS if c != 'IV_ATM_var']


# ==========================================================================
# 1. REBUILD PANEL (identical construction to src/66/67/68)
# ==========================================================================
print('\n' + '-' * 92)
print('1. REBUILD PANEL')
print('-' * 92)

cal_all = pd.DatetimeIndex(sorted(
    pd.read_parquet(crsp_path, columns=['DlyCalDt'])['DlyCalDt'].unique()))
dev_cal = cal_all[(cal_all >= DEV_START) & (cal_all <= DEV_END)]
dev_pos = cal_all.get_indexer(dev_cal)
n_t_arr = np.zeros(len(dev_cal), dtype=int)
for i, p in enumerate(dev_pos):
    hi = dev_cal[i] + pd.Timedelta(days=CAL_WINDOW_DAYS)
    future = cal_all[p + 1:]
    n_t_arr[i] = int((future <= hi).sum())
cal_pos_of = pd.Series(np.arange(len(cal_all)), index=cal_all)

univ = pd.read_parquet(univ_path)
univ_in = univ[univ['in_universe']][['PERMNO', 'year_month', 'decile']].copy()
ever = univ_in['PERMNO'].unique()

feat = pd.read_parquet(features_path)
feat = feat.rename(columns={'date_d': 'DlyCalDt'})
print(f"Feature panel loaded: {len(feat):,} rows (RV2_primary not needed for this check - "
      f"neither diagnostic uses the dependent variable)")


# ==========================================================================
# 2. FAMA-MACBETH-STYLE (per-date) CORRELATION
# ==========================================================================
print('\n' + '=' * 92)
print('2. PER-DATE CROSS-SECTIONAL CORRELATION (compression_decile vs IV_ATM_var)')
print('=' * 92)

pair = feat[['DlyCalDt', 'compression_decile', 'IV_ATM_var']].dropna()
daily_r = []
daily_n = []
n_dates_checked = 0
n_dates_used = 0
for dt, gg in pair.groupby('DlyCalDt', sort=True):
    n_dates_checked += 1
    if len(gg) < MIN_XSEC:
        continue
    if gg['compression_decile'].nunique() < 2 or gg['IV_ATM_var'].nunique() < 2:
        continue
    r, _ = pearsonr(gg['compression_decile'], gg['IV_ATM_var'])
    daily_r.append(r)
    daily_n.append(len(gg))
    n_dates_used += 1

daily_r = np.array(daily_r)
mean_r = float(np.mean(daily_r))
std_r = float(np.std(daily_r, ddof=1))
pct_above_010 = float((np.abs(daily_r) > 0.10).mean() * 100)

print(f"dates checked: {n_dates_checked:,}   dates used (n>=30, both non-degenerate): {n_dates_used:,}")
print(f"mean cross-sectional n on used dates: {np.mean(daily_n):.1f}")
print(f"mean daily r:   {mean_r:+.6f}")
print(f"std daily r:    {std_r:.6f}")
print(f"% of days with |r| > 0.10: {pct_above_010:.2f}%")
print(f"daily r distribution: min={daily_r.min():+.4f}  p5={np.percentile(daily_r,5):+.4f}  "
      f"median={np.median(daily_r):+.4f}  p95={np.percentile(daily_r,95):+.4f}  max={daily_r.max():+.4f}")


# ==========================================================================
# 3. PARTIAL CORRELATION (pooled residualization against the other 9)
# ==========================================================================
print('\n' + '=' * 92)
print(f"3. PARTIAL CORRELATION - compression_decile & IV_ATM_var, residualized against "
      f"the other {len(OTHER_9)} benchmark features (single pooled OLS)")
print('=' * 92)
print(f"Control set (the 'other 9'): {OTHER_9}")

cols_needed = ['compression_decile', 'IV_ATM_var'] + OTHER_9
pc = feat[cols_needed].dropna()
n_pc = len(pc)
print(f"n (pooled, all {len(cols_needed)} columns non-null): {n_pc:,} of {len(feat):,} "
      f"({n_pc / len(feat) * 100:.2f}%)")

Z = np.column_stack([np.ones(n_pc)] + [pc[c].to_numpy(float) for c in OTHER_9])

y_comp = pc['compression_decile'].to_numpy(float)
beta_comp, _, _, _ = np.linalg.lstsq(Z, y_comp, rcond=None)
resid_comp = y_comp - Z @ beta_comp

y_iv = pc['IV_ATM_var'].to_numpy(float)
beta_iv, _, _, _ = np.linalg.lstsq(Z, y_iv, rcond=None)
resid_iv = y_iv - Z @ beta_iv

partial_r, partial_p = pearsonr(resid_comp, resid_iv)

r2_comp_on_others = 1.0 - np.sum(resid_comp ** 2) / np.sum((y_comp - y_comp.mean()) ** 2)
r2_iv_on_others = 1.0 - np.sum(resid_iv ** 2) / np.sum((y_iv - y_iv.mean()) ** 2)

print(f"\nR2 of compression_decile ~ other 9 features (pooled): {r2_comp_on_others:.6f}")
print(f"R2 of IV_ATM_var ~ other 9 features (pooled):          {r2_iv_on_others:.6f}")
print(f"\nPartial correlation (compression_decile, IV_ATM_var | other 9): "
      f"{partial_r:+.6f}   (p-value {partial_p:.3e})")
print(f"For reference - raw pooled Pearson r (src/68, unconditional): -0.017655")


# ==========================================================================
# 4. WRITE JSON + APPEND TO THE SAME gate_log.md ENTRY (continuation, not new)
# ==========================================================================
out = {
    'phase': 'V4 Gate 6a - continuation of correlation/ordering check (UNDER REVIEW, src/68)',
    'generated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'script': 'src/69_v4_gate6a_partial_correlation_check.py',
    'status': 'UNDER REVIEW - continuation, not a new finding, still holding on Gate 6b/hedging/P&L',
    'per_date_correlation': {
        'n_dates_checked': n_dates_checked, 'n_dates_used': n_dates_used,
        'mean_daily_r': mean_r, 'std_daily_r': std_r, 'pct_days_abs_r_gt_010': pct_above_010,
        'daily_r_min': float(daily_r.min()), 'daily_r_p5': float(np.percentile(daily_r, 5)),
        'daily_r_median': float(np.median(daily_r)), 'daily_r_p95': float(np.percentile(daily_r, 95)),
        'daily_r_max': float(daily_r.max()),
    },
    'partial_correlation': {
        'control_set_other_9': OTHER_9, 'n_pooled': int(n_pc),
        'r2_compression_on_other9': float(r2_comp_on_others),
        'r2_iv_atm_on_other9': float(r2_iv_on_others),
        'partial_r': float(partial_r), 'partial_p': float(partial_p),
        'reference_raw_pooled_pearson_r_src68': -0.017655,
    },
}
with open(out_json, 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f"\n[OK] wrote {out_json}")

# CONTINUATION block - deliberately no '## ' header and no '---' rule, so it
# reads as part of the immediately preceding UNDER REVIEW entry rather than
# a new dated finding, per instruction.
log_block = f"""

**[CONTINUED {out['generated']} - src/69_v4_gate6a_partial_correlation_check.py - same UNDER REVIEW status as the entry above, resolving the same open question, not a new finding. Still holding on Gate 6b, hedging, and P&L work.]**

```
FAMA-MACBETH-STYLE (per-date) CORRELATION, compression_decile vs IV_ATM_var:
  dates checked: {n_dates_checked:,}   dates used (n>=30): {n_dates_used:,}
  mean daily r:   {mean_r:+.6f}
  std daily r:    {std_r:.6f}
  % of days with |r| > 0.10: {pct_above_010:.2f}%
  daily r range:  min={daily_r.min():+.4f}  median={np.median(daily_r):+.4f}  max={daily_r.max():+.4f}

PARTIAL CORRELATION (compression_decile, IV_ATM_var | other 9 benchmark features), n={n_pc:,}:
  R2(compression_decile ~ other 9):  {r2_comp_on_others:.6f}
  R2(IV_ATM_var ~ other 9):          {r2_iv_on_others:.6f}
  partial r:  {partial_r:+.6f}  (p={partial_p:.3e})
  [reference: raw pooled Pearson r, unconditional (src/68) = -0.017655]
```
"""
with open(log_path, 'a') as f:
    f.write(log_block)
print(f"[OK] appended continuation to {log_path} (same UNDER REVIEW entry, no new header)")

print('\n' + '=' * 96)
print('CONTINUATION COMPLETE. Raw numbers only, no interpretation.')
print('Still UNDER REVIEW. Still holding on Gate 6b, hedging, and P&L work.')
print('=' * 96)
