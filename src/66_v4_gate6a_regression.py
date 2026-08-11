import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# V4 GATE 6a - signal-level, DEV-only diagnostic: does compression carry
# incremental predictive power over the 10-feature benchmark
# (results/61_v4_gate6_benchmark_model_spec.md)?
#
# METHODOLOGY CORRECTION, disclosed: an earlier draft of the addendum said
# Gate 6a would reuse prereg_V4.md Section 4.2's expanding-window walk-
# forward scheme. On review, that scheme exists to build out-of-sample
# Score_{i,t} for live trading decisions (feeding Gate 6b's portfolio) -
# a different job from Gate 6a's actual purpose, which is the same kind of
# test V3 itself ran: a SINGLE joint Fama-MacBeth regression over the full
# DEV sample, testing whether a coefficient survives controlling for other
# regressors. Built here exactly as V3's own src/49 built its regressions
# (same estimator, same NW convention, same MIN_XSEC) - not the walk-
# forward scheme - per "keep this comparable in spirit to V1-V3's
# methodology" and "do not introduce a more complex model without separate
# justification."
#
# DEV ONLY. No holdout code path - matches prereg_V4.md Section 3's
# unconditional prohibition, unchanged by this diagnostic.
#
# gate_log.md receives numbers only, logged as Gate 6a, cross-referenced to
# but distinct from Section 11 item 6 (Gate 6b).
# ---------------------------------------------------------------------------

DEV_START = pd.Timestamp('2015-01-01')
DEV_END = pd.Timestamp('2021-12-31')
CAL_WINDOW_DAYS = 30
TRADING_DAYS_PER_YEAR = 252.0
MIN_XSEC = 30
NW_MAXLAGS = 21  # matches V3's own primary-horizon convention (prereg_V3 Sec.5)

project_root = Path(__file__).parent.parent
crsp_path = project_root / 'data' / 'processed' / 'crsp_combined.parquet'
univ_path = project_root / 'data' / 'processed' / 'universe_membership.parquet'
features_path = project_root / 'data' / 'processed' / 'v4_gate6a_features_full.parquet'
prereg_path = project_root / 'results' / 'prereg_V4.md'
out_json = project_root / 'results' / '66_v4_gate6a_regression.json'
log_path = project_root / 'results' / 'gate_log.md'

print('=' * 96)
print('V4 GATE 6a - SIGNAL-LEVEL BENCHMARK TEST (DEV only, no holdout code path)')
print('=' * 96)

prereg_text = prereg_path.read_text(encoding='utf-8', errors='replace')
if 'Status: LOCKED' not in prereg_text:
    raise SystemExit('STOP: prereg_V4.md is not marked LOCKED.')
print('[OK] Locked V4 pre-registration found.')


def nw_ols_const(x, maxlags):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan, len(x)
    m = sm.OLS(x, np.ones(len(x))).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
    return float(m.params[0]), float(m.tvalues[0]), len(x)


def fama_macbeth(df, ycol, xcols, maxlags, min_xsec=MIN_XSEC):
    """Daily cross-sectional OLS; NW t-test on each daily coefficient series.
    Also collects each day's cross-sectional R^2 for the incremental-R^2
    report. Identical estimator design to src/49 (V3's own build)."""
    use = df.dropna(subset=[ycol] + xcols)
    k = len(xcols) + 1
    coefs, dates, ns, r2s = [], [], [], []
    n_dropped = 0
    for dt, gg in use.groupby('DlyCalDt', sort=True):
        if len(gg) < min_xsec:
            n_dropped += 1
            continue
        X = np.column_stack([np.ones(len(gg))] + [gg[c].to_numpy(float) for c in xcols])
        y = gg[ycol].to_numpy(float)
        if not np.isfinite(X).all() or np.linalg.matrix_rank(X) < k:
            n_dropped += 1
            continue
        beta, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        coefs.append(beta)
        dates.append(dt)
        ns.append(len(gg))
        r2s.append(r2)
    if not coefs:
        return None
    C = np.asarray(coefs)
    out = {'n_dates': len(dates), 'n_dates_dropped': n_dropped,
          'mean_xsec_n': float(np.mean(ns)) if ns else float('nan'),
          'mean_daily_r2': float(np.mean(r2s)) if r2s else float('nan'),
          'coefs': {}}
    for j, name in enumerate(['const'] + xcols):
        s = C[:, j]
        m, t, n = nw_ols_const(s, maxlags)
        out['coefs'][name] = {'mean_coef': m, 'nw_t': t, 'n_dates_used': n}
    return out


# ==========================================================================
# 1. TRADING CALENDAR + n_t (identical construction to src/49, prereg_V3 Sec.3.3-F)
# ==========================================================================
print('\n' + '-' * 92)
print('1. TRADING CALENDAR AND n_t (30-calendar-day window), reused from src/49')
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
n_t_by_date = pd.Series(n_t_arr, index=dev_cal)
print(f"n_t: min={n_t_arr.min()}, max={n_t_arr.max()}, mean={n_t_arr.mean():.2f}")
assert n_t_arr.min() >= 17 and n_t_arr.max() <= 22, 'n_t moved outside the confirmed 17-22 range'
cal_pos_of = pd.Series(np.arange(len(cal_all)), index=cal_all)

univ = pd.read_parquet(univ_path)
univ_in = univ[univ['in_universe']][['PERMNO', 'year_month', 'decile']].copy()
ever = univ_in['PERMNO'].unique()

need = ['PERMNO', 'DlyCalDt', 'DlyRet']
d = pd.read_parquet(crsp_path, columns=need)
d = d[d['PERMNO'].isin(ever)]
d = d[(d['DlyCalDt'] >= '2014-01-01') & (d['DlyCalDt'] <= '2022-06-30')]
d = d.drop_duplicates(['PERMNO', 'DlyCalDt'], keep='first')
d = d.sort_values(['PERMNO', 'DlyCalDt']).reset_index(drop=True)
d['pos'] = np.arange(len(d))
pn = d['PERMNO'].to_numpy()
a_ret = d['DlyRet'].to_numpy(float)
d['cal_pos'] = cal_pos_of.reindex(d['DlyCalDt']).to_numpy()
a_calpos = d['cal_pos'].to_numpy(float)
N = len(d)
print(f"Daily rows (extended range): {len(d):,}  PERMNOs: {d['PERMNO'].nunique():,}")


# ==========================================================================
# 2. PRIMARY RV2 - identical construction to src/49 / prereg_V3 Sec.3.3-F
# Resolution (calendar-anchored, gap-checked, no fallback window).
# ==========================================================================
print('\n' + '-' * 92)
print('2. PRIMARY RV2 (30 calendar days), gap-checked, reused from src/49')
print('-' * 92)

d['n_t'] = d['DlyCalDt'].map(n_t_by_date)
rv2_primary = np.full(N, np.nan)
drop_gap = drop_boundary = drop_nan_return = 0
n_computed = 0
for k in sorted(pd.Series(n_t_arr).unique()):
    anchor_mask = (d['n_t'].to_numpy() == k)
    if not anchor_mask.any():
        continue
    p = np.where(anchor_mask)[0]
    end = p + k
    inb = end < N
    p_ok, end_ok = p[inb], end[inb]
    same_permno = pn[end_ok] == pn[p_ok]
    no_gap = (a_calpos[end_ok] - a_calpos[p_ok]) == k
    complete = same_permno & no_gap
    drop_boundary += int((~same_permno).sum())
    drop_gap += int((same_permno & ~no_gap).sum())
    p_c = p_ok[complete]
    if len(p_c) == 0:
        continue
    off = np.arange(1, k + 1)
    idx = p_c[:, None] + off[None, :]
    window_ret = a_ret[idx]
    all_valid = np.all(np.isfinite(window_ret), axis=1)
    drop_nan_return += int((~all_valid).sum())
    p_v = p_c[all_valid]
    if len(p_v) == 0:
        continue
    var = np.var(window_ret[all_valid], axis=1, ddof=1)
    rv2_primary[p_v] = var * TRADING_DAYS_PER_YEAR
    n_computed += len(p_v)
d['RV2_primary'] = rv2_primary
print(f"RV2_primary computed: {n_computed:,}  "
      f"(dropped: boundary {drop_boundary:,}, gap {drop_gap:,}, nan-return {drop_nan_return:,})")


# ==========================================================================
# 3. MERGE WITH 10-FEATURE PANEL, BUILD BENCHMARK AND AUGMENTED MODELS
# ==========================================================================
print('\n' + '-' * 92)
print('3. MERGE WITH FEATURE PANEL, BUILD MODEL SPECIFICATIONS')
print('-' * 92)

feat = pd.read_parquet(features_path)
feat = feat.rename(columns={'date_d': 'DlyCalDt'})
panel = feat.merge(d[['PERMNO', 'DlyCalDt', 'RV2_primary']], on=['PERMNO', 'DlyCalDt'], how='left')
print(f"Panel after merge: {len(panel):,} rows")
print(f"RV2_primary non-null: {panel['RV2_primary'].notna().sum():,} "
      f"({panel['RV2_primary'].notna().mean()*100:.2f}%)")

# DISCLOSED STRUCTURAL EXCLUSION: MktRV5 (feature 10) is a market-level
# variable - the SAME value for every stock on a given date. In a
# Fama-MacBeth CROSS-SECTIONAL regression (regressing across stocks i at a
# FIXED date t), any pure date-level regressor is a constant WITHIN that
# cross-section and is therefore perfectly collinear with the intercept -
# not a coding bug, a structural property of the estimator. This was
# discovered when fama_macbeth() returned None for every single DEV date
# (all 1,568 dates failed the full-rank design-matrix check identically -
# the signature of an exact, universal collinearity, not incidental
# rank-deficiency on a handful of days). Fama-MacBeth cannot identify a
# cross-sectional coefficient on a variable that has zero cross-sectional
# variation; a second-pass time-series regression would be needed to test
# it, which is a different, more complex specification this task did not
# ask for. MktRV5 remains in the built feature panel
# (v4_gate6a_features_full.parquet) for any future specification that can
# use it; it is EXCLUDED from Gate 6a's regressor list for this
# structural reason, not silently dropped.
BENCHMARK_COLS = ['RV1', 'Persist20', 'RV20', 'RV60', 'DSV5', 'JumpVar', 'VoV20',
                  'IV_ATM_var', 'IVTermStruct_var', 'IVSkew_var']
AUGMENTED_COLS = BENCHMARK_COLS + ['compression_decile']

print(f"\nBenchmark model regressors ({len(BENCHMARK_COLS)}): {BENCHMARK_COLS}")
print(f"Augmented model regressors ({len(AUGMENTED_COLS)}): benchmark + compression_decile")
print("\n*** DISCLOSED: MktRV5 (feature 10) excluded from this regression's regressor list.")
print("*** It is a market-level (date-only) variable, perfectly collinear with the daily")
print("*** intercept in a cross-sectional Fama-MacBeth design - a structural estimator")
print("*** limitation confirmed by ALL 1,568 DEV dates failing the rank check identically,")
print("*** not incidental rank-deficiency. Retained in the feature panel, excluded here.")

for c in BENCHMARK_COLS + ['compression_decile', 'RV2_primary']:
    s = panel[c]
    print(f"  {c:<18} non-null: {s.notna().sum():,} ({s.notna().mean()*100:.2f}%)")


# ==========================================================================
# 4. FIT BOTH MODELS - single joint Fama-MacBeth regression, full DEV sample
# ==========================================================================
print('\n' + '=' * 92)
print('4. FIT BENCHMARK-ONLY AND AUGMENTED MODELS (DEV, single joint FM regression)')
print('=' * 92)

fm_benchmark = fama_macbeth(panel, 'RV2_primary', BENCHMARK_COLS, NW_MAXLAGS)
fm_augmented = fama_macbeth(panel, 'RV2_primary', AUGMENTED_COLS, NW_MAXLAGS)

print(f"\nBenchmark-only: n_dates={fm_benchmark['n_dates']:,} "
      f"(dropped {fm_benchmark['n_dates_dropped']:,})  "
      f"mean_xsec_n={fm_benchmark['mean_xsec_n']:.1f}  "
      f"mean_daily_R2={fm_benchmark['mean_daily_r2']:.6f}")
print(f"Augmented:      n_dates={fm_augmented['n_dates']:,} "
      f"(dropped {fm_augmented['n_dates_dropped']:,})  "
      f"mean_xsec_n={fm_augmented['mean_xsec_n']:.1f}  "
      f"mean_daily_R2={fm_augmented['mean_daily_r2']:.6f}")

incremental_r2 = fm_augmented['mean_daily_r2'] - fm_benchmark['mean_daily_r2']

comp_coef = fm_augmented['coefs']['compression_decile']
print(f"\n--- RAW NUMBERS (no verdict language) ---")
print(f"compression_decile coefficient (augmented model): {comp_coef['mean_coef']:+.8e}")
print(f"compression_decile NW t-stat (maxlags={NW_MAXLAGS}):         {comp_coef['nw_t']:+.4f}")
print(f"n_dates_used for compression_decile coefficient:  {comp_coef['n_dates_used']:,}")
print(f"mean daily cross-sectional R^2, benchmark-only:    {fm_benchmark['mean_daily_r2']:.6f}")
print(f"mean daily cross-sectional R^2, augmented:         {fm_augmented['mean_daily_r2']:.6f}")
print(f"incremental R^2 (augmented - benchmark-only):      {incremental_r2:+.6f}")

print(f"\nFull benchmark-model coefficients:")
for name, cv in fm_benchmark['coefs'].items():
    print(f"  {name:<20} coef={cv['mean_coef']:+.6e}  NW t={cv['nw_t']:+.4f}  n={cv['n_dates_used']:,}")
print(f"\nFull augmented-model coefficients:")
for name, cv in fm_augmented['coefs'].items():
    print(f"  {name:<20} coef={cv['mean_coef']:+.6e}  NW t={cv['nw_t']:+.4f}  n={cv['n_dates_used']:,}")


# ==========================================================================
# 5. WRITE OUTPUT + LOG TO gate_log.md
# ==========================================================================
def strip_for_json(fm):
    return fm  # already scalar-only, no daily_series retained in this script

out = {
    'phase': 'V4 Gate 6a - signal-level benchmark diagnostic',
    'generated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'script': 'src/66_v4_gate6a_regression.py',
    'pre_registration': 'results/61_v4_gate6_benchmark_model_spec.md (all 7 flags resolved 2026-08-04)',
    'window': 'DEV 2015-01-01 to 2021-12-31 - no holdout code path',
    'estimator': ('single joint Fama-MacBeth cross-sectional OLS, full DEV sample - NOT the '
                 'expanding-window walk-forward scheme (methodology correction, see script '
                 'header); NW maxlags=21, MIN_XSEC=30, matching V3 src/49 exactly'),
    'cross_reference': 'distinct from and does not substitute for prereg_V4.md Sec.11 item 6 (Gate 6b, portfolio-level)',
    'disclosed_exclusion': ('MktRV5 (feature 10) excluded from the regressor list - a '
                            'market-level (date-only) variable is perfectly collinear with '
                            'the daily intercept in a cross-sectional Fama-MacBeth design '
                            '(confirmed: all 1,568 DEV dates failed the rank check '
                            'identically before this fix). Retained in '
                            'v4_gate6a_features_full.parquet, excluded here only.'),
    'benchmark_regressors': BENCHMARK_COLS,
    'augmented_regressors': AUGMENTED_COLS,
    'benchmark_model': strip_for_json(fm_benchmark),
    'augmented_model': strip_for_json(fm_augmented),
    'incremental_R2': incremental_r2,
    'compression_coefficient_augmented': comp_coef,
}
with open(out_json, 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f"\n[OK] wrote {out_json}")

log_block = f"""

---

## V4 Gate 6a - signal-level benchmark diagnostic (DEV only)

Run {out['generated']} - script src/66_v4_gate6a_regression.py - pre-registration results/61_v4_gate6_benchmark_model_spec.md (all 7 flags resolved 2026-08-04). DEV 2015-01-01 to 2021-12-31, no holdout code path. Single joint Fama-MacBeth regression (NOT the Section 4.2 expanding-window scheme - methodology correction disclosed in the script), NW maxlags=21, MIN_XSEC=30, matching V3 src/49's estimator exactly. DISCLOSED EXCLUSION: MktRV5 (feature 10) dropped from the regressor list - a market-level (date-only) variable is perfectly collinear with the daily intercept in a cross-sectional Fama-MacBeth design (confirmed: all 1,568 DEV dates failed the rank check identically before this fix); retained in v4_gate6a_features_full.parquet, excluded from this regression only. CROSS-REFERENCED TO, DISTINCT FROM, AND DOES NOT SUBSTITUTE FOR prereg_V4.md Section 11 item 6 (Gate 6b, portfolio-level, full-cost). Machine-readable copy: results/66_v4_gate6a_regression.json.

```
Benchmark-only model ({len(BENCHMARK_COLS)} regressors, no compression):
  n_dates {fm_benchmark['n_dates']:,} (dropped {fm_benchmark['n_dates_dropped']:,})   mean_xsec_n {fm_benchmark['mean_xsec_n']:.1f}
  mean daily cross-sectional R2: {fm_benchmark['mean_daily_r2']:.6f}

Augmented model ({len(AUGMENTED_COLS)} regressors, benchmark + compression_decile):
  n_dates {fm_augmented['n_dates']:,} (dropped {fm_augmented['n_dates_dropped']:,})   mean_xsec_n {fm_augmented['mean_xsec_n']:.1f}
  mean daily cross-sectional R2: {fm_augmented['mean_daily_r2']:.6f}
  compression_decile coef: {comp_coef['mean_coef']:+.6e}   NW t: {comp_coef['nw_t']:+.4f}   n_dates_used: {comp_coef['n_dates_used']:,}

Incremental R2 (augmented - benchmark-only): {incremental_r2:+.6f}
```
"""
with open(log_path, 'a') as f:
    f.write(log_block)
print(f"[OK] appended Gate 6a result to {log_path}")

print('\n' + '=' * 96)
print('GATE 6a REGRESSION COMPLETE. Raw numbers reported above and logged - NO VERDICT')
print('LANGUAGE APPLIED. Stopping here for review before Hou-Loh decomposition or any')
print('hedging/P&L work, per instruction.')
print('=' * 96)
