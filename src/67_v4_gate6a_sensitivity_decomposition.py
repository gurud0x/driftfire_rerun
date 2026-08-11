import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# V4 GATE 6a - TWO DIAGNOSTIC FOLLOW-UPS, both reusing src/66's already-
# produced panel and estimator (no new data contact):
#
#   A. LEAVE-ONE-OUT SANITY CHECK: does compression's NW t-stat move
#      materially when any single benchmark feature is dropped, or does it
#      stay stable? (compression's t=-9.3978 in the full model is larger in
#      magnitude than V1/V2's own original dev results, -6.883/-7.016,
#      despite 9 added controls - checked here rather than trusted blind.)
#
#   B. HOU-LOH-STYLE DECOMPOSITION (Branger, Hulsbusch & Middelhoff 2017
#      Table 16 / Hou & Loh 2016): decompose compression's INCREMENTAL R^2
#      (+0.003489, not the 0.282 baseline) into the portion attributable to
#      each of the 10 benchmark features vs a residual.
#
#      DISCLOSED METHODOLOGY, not the literal original: Hou-Loh's original
#      method uses portfolio double-sorts, an infrastructure this project
#      does not have and was not asked to build ("this is a
#      reporting/interpretation layer on data already produced in Step 3 -
#      it does not require new data contact"). The regression-based
#      analogue used here: for each feature Zj, refit BOTH the benchmark-
#      only and augmented models with Zj REMOVED from the benchmark set,
#      and compare compression's incremental R^2 in that (9-feature)
#      specification to its incremental R^2 in the FULL (10-feature)
#      specification. If removing Zj INCREASES compression's incremental
#      R^2, Zj was absorbing/overlapping with part of compression's power
#      when present - attributed to Zj. This is a marginal/leave-one-out
#      decomposition, not the original portfolio-sort method, and is
#      disclosed as such rather than presented as identical to it.
#
# Raw numbers only - no interpretation, no verdict language.
# ---------------------------------------------------------------------------

DEV_START = pd.Timestamp('2015-01-01')
DEV_END = pd.Timestamp('2021-12-31')
CAL_WINDOW_DAYS = 30
TRADING_DAYS_PER_YEAR = 252.0
MIN_XSEC = 30
NW_MAXLAGS = 21

project_root = Path(__file__).parent.parent
crsp_path = project_root / 'data' / 'processed' / 'crsp_combined.parquet'
univ_path = project_root / 'data' / 'processed' / 'universe_membership.parquet'
features_path = project_root / 'data' / 'processed' / 'v4_gate6a_features_full.parquet'
out_json = project_root / 'results' / '67_v4_gate6a_sensitivity_decomposition.json'
log_path = project_root / 'results' / 'gate_log.md'

print('=' * 96)
print('V4 GATE 6a - SENSITIVITY CHECK + HOU-LOH-STYLE DECOMPOSITION')
print('=' * 96)


def nw_ols_const(x, maxlags):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan, len(x)
    m = sm.OLS(x, np.ones(len(x))).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
    return float(m.params[0]), float(m.tvalues[0]), len(x)


def fama_macbeth(df, ycol, xcols, maxlags, min_xsec=MIN_XSEC):
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
          'mean_daily_r2': float(np.mean(r2s)) if r2s else float('nan'), 'coefs': {}}
    for j, name in enumerate(['const'] + xcols):
        s = C[:, j]
        m, t, n = nw_ols_const(s, maxlags)
        out['coefs'][name] = {'mean_coef': m, 'nw_t': t, 'n_dates_used': n}
    return out


# ==========================================================================
# 1. REBUILD THE PANEL (identical to src/66, self-contained reproducibility)
# ==========================================================================
print('\n' + '-' * 92)
print('1. REBUILD PANEL (RV2_primary + 10-feature benchmark, identical to src/66)')
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

d['n_t'] = d['DlyCalDt'].map(n_t_by_date)
rv2_primary = np.full(N, np.nan)
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
    p_c = p_ok[complete]
    if len(p_c) == 0:
        continue
    off = np.arange(1, k + 1)
    idx = p_c[:, None] + off[None, :]
    window_ret = a_ret[idx]
    all_valid = np.all(np.isfinite(window_ret), axis=1)
    p_v = p_c[all_valid]
    if len(p_v) == 0:
        continue
    var = np.var(window_ret[all_valid], axis=1, ddof=1)
    rv2_primary[p_v] = var * TRADING_DAYS_PER_YEAR
d['RV2_primary'] = rv2_primary

feat = pd.read_parquet(features_path)
feat = feat.rename(columns={'date_d': 'DlyCalDt'})
panel = feat.merge(d[['PERMNO', 'DlyCalDt', 'RV2_primary']], on=['PERMNO', 'DlyCalDt'], how='left')
print(f"Panel rebuilt: {len(panel):,} rows, RV2_primary non-null "
      f"{panel['RV2_primary'].notna().sum():,}")

BENCHMARK_COLS = ['RV1', 'Persist20', 'RV20', 'RV60', 'DSV5', 'JumpVar', 'VoV20',
                  'IV_ATM_var', 'IVTermStruct_var', 'IVSkew_var']
COMP_COL = 'compression_decile'


# ==========================================================================
# 2. BASELINE (FULL) MODELS - consistency check against src/66's own numbers
# ==========================================================================
print('\n' + '-' * 92)
print('2. BASELINE MODELS (must match src/66 exactly - consistency check)')
print('-' * 92)

fm_bench_full = fama_macbeth(panel, 'RV2_primary', BENCHMARK_COLS, NW_MAXLAGS)
fm_aug_full = fama_macbeth(panel, 'RV2_primary', BENCHMARK_COLS + [COMP_COL], NW_MAXLAGS)
r2_bench_full = fm_bench_full['mean_daily_r2']
r2_aug_full = fm_aug_full['mean_daily_r2']
inc_r2_full = r2_aug_full - r2_bench_full
comp_t_full = fm_aug_full['coefs'][COMP_COL]['nw_t']
comp_coef_full = fm_aug_full['coefs'][COMP_COL]['mean_coef']

print(f"benchmark-only R2: {r2_bench_full:.6f}  [src/66 reference: 0.282186]")
print(f"augmented R2:      {r2_aug_full:.6f}  [src/66 reference: 0.285674]")
print(f"incremental R2:    {inc_r2_full:+.6f}  [src/66 reference: +0.003489]")
print(f"compression coef:  {comp_coef_full:+.6e}  NW t: {comp_t_full:+.4f}  "
      f"[src/66 reference: -6.312308e-03, t=-9.3978]")
consistency_ok = (abs(r2_bench_full - 0.282186) < 1e-4 and abs(r2_aug_full - 0.285674) < 1e-4
                  and abs(comp_t_full - (-9.3978)) < 0.01)
print(f"Consistency with src/66: {'OK' if consistency_ok else '*** MISMATCH ***'}")
if not consistency_ok:
    raise SystemExit('STOP: rebuilt panel does not reproduce src/66 baseline - investigate before continuing.')


# ==========================================================================
# 3. LEAVE-ONE-FEATURE-OUT: sanity check (A) and decomposition inputs (B)
# ==========================================================================
print('\n' + '=' * 92)
print('3. LEAVE-ONE-FEATURE-OUT (10 features, each removed one at a time)')
print('=' * 92)

loo_results = {}
for j, feat_name in enumerate(BENCHMARK_COLS, 1):
    remaining = [c for c in BENCHMARK_COLS if c != feat_name]
    fm_b = fama_macbeth(panel, 'RV2_primary', remaining, NW_MAXLAGS)
    fm_a = fama_macbeth(panel, 'RV2_primary', remaining + [COMP_COL], NW_MAXLAGS)
    r2_b = fm_b['mean_daily_r2']
    r2_a = fm_a['mean_daily_r2']
    inc_r2_minus_j = r2_a - r2_b
    comp_t_minus_j = fm_a['coefs'][COMP_COL]['nw_t']
    comp_coef_minus_j = fm_a['coefs'][COMP_COL]['mean_coef']
    own_coef_full_model = fm_aug_full['coefs'][feat_name]['mean_coef']
    own_t_full_model = fm_aug_full['coefs'][feat_name]['nw_t']

    attribution = inc_r2_minus_j - inc_r2_full
    pct_attributable = attribution / inc_r2_full * 100.0

    loo_results[feat_name] = {
        'own_coef_in_full_augmented_model': own_coef_full_model,
        'own_nw_t_in_full_augmented_model': own_t_full_model,
        'compression_t_with_feature_removed': comp_t_minus_j,
        'compression_t_delta_vs_full': comp_t_minus_j - comp_t_full,
        'compression_coef_with_feature_removed': comp_coef_minus_j,
        'benchmark_only_R2_minus_feature': r2_b,
        'augmented_R2_minus_feature': r2_a,
        'incremental_R2_minus_feature': inc_r2_minus_j,
        'attribution_to_feature': attribution,
        'pct_of_compression_explanatory_power': pct_attributable,
    }
    print(f"  [{j:>2}/10] removed={feat_name:<18}  "
          f"compression NW t (feature removed)={comp_t_minus_j:+.4f}  "
          f"(delta vs full: {comp_t_minus_j - comp_t_full:+.4f})  "
          f"incR2_minus_j={inc_r2_minus_j:+.6f}  attribution={attribution:+.6f}  "
          f"pct={pct_attributable:+.2f}%")

residual_pct = 100.0 - sum(v['pct_of_compression_explanatory_power'] for v in loo_results.values())
print(f"\nResidual % (not attributable to any single benchmark feature): {residual_pct:+.2f}%")


# ==========================================================================
# 4. REPORT - TASK A: sensitivity table
# ==========================================================================
print('\n' + '=' * 92)
print('TASK A - LEAVE-ONE-OUT SANITY CHECK ON COMPRESSION NW t-STAT')
print(f"(full-model baseline: compression NW t = {comp_t_full:+.4f})")
print('=' * 92)
print(f"{'feature removed':<18} {'t-stat (removed)':>18} {'delta vs full':>15}")
for feat_name, r in loo_results.items():
    print(f"{feat_name:<18} {r['compression_t_with_feature_removed']:>18.4f} "
          f"{r['compression_t_delta_vs_full']:>15.4f}")
max_abs_delta = max(abs(r['compression_t_delta_vs_full']) for r in loo_results.values())
max_delta_feature = max(loo_results, key=lambda k: abs(loo_results[k]['compression_t_delta_vs_full']))
print(f"\nLargest single-feature-removal delta: {max_delta_feature} "
      f"({loo_results[max_delta_feature]['compression_t_delta_vs_full']:+.4f})")


# ==========================================================================
# 5. REPORT - TASK B: decomposition table
# ==========================================================================
print('\n' + '=' * 92)
print('TASK B - HOU-LOH-STYLE DECOMPOSITION OF COMPRESSION INCREMENTAL R^2')
print(f"(total incremental R^2 to decompose: {inc_r2_full:+.6f})")
print('=' * 92)
print(f"{'feature':<18} {'own coefficient':>18} {'% of comp. power':>18}")
for feat_name, r in loo_results.items():
    print(f"{feat_name:<18} {r['own_coef_in_full_augmented_model']:>18.6e} "
          f"{r['pct_of_compression_explanatory_power']:>17.2f}%")
print(f"{'RESIDUAL':<18} {'':>18} {residual_pct:>17.2f}%")


# ==========================================================================
# 6. WRITE JSON + LOG
# ==========================================================================
out = {
    'phase': 'V4 Gate 6a - sensitivity check and Hou-Loh-style decomposition',
    'generated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'script': 'src/67_v4_gate6a_sensitivity_decomposition.py',
    'cross_reference': 'diagnostic follow-up to Gate 6a (results/66_v4_gate6a_regression.json), NOT a new gate',
    'decomposition_methodology_disclosure': (
        'Regression-based analogue of Hou & Loh (2016) / Branger, Hulsbusch & '
        'Middelhoff (2017) Table 16 - NOT the original portfolio-double-sort method '
        '(no such infrastructure exists in this project and building one was not '
        'requested). For each benchmark feature Zj, both the benchmark-only and '
        'augmented models are refit with Zj removed; the change in compression\'s '
        'incremental R^2 when Zj is absent is attributed to Zj.'),
    'baseline': {'benchmark_only_R2': r2_bench_full, 'augmented_R2': r2_aug_full,
                'incremental_R2': inc_r2_full, 'compression_coef': comp_coef_full,
                'compression_nw_t': comp_t_full},
    'leave_one_out': loo_results,
    'residual_pct_of_compression_power': residual_pct,
}
with open(out_json, 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f"\n[OK] wrote {out_json}")

sens_lines = "\n".join(
    f"  {feat_name:<18} t(removed)={r['compression_t_with_feature_removed']:+.4f}  "
    f"delta={r['compression_t_delta_vs_full']:+.4f}"
    for feat_name, r in loo_results.items())
decomp_lines = "\n".join(
    f"  {feat_name:<18} coef={r['own_coef_in_full_augmented_model']:+.6e}  "
    f"pct_of_compression_power={r['pct_of_compression_explanatory_power']:+.2f}%"
    for feat_name, r in loo_results.items())

log_block = f"""

---

## V4 Gate 6a - sensitivity check and Hou-Loh-style decomposition (diagnostic, NOT a new gate)

Run {out['generated']} - script src/67_v4_gate6a_sensitivity_decomposition.py - diagnostic follow-up to Gate 6a (results/66_v4_gate6a_regression.json), distinct from that regression result and from Section 11 item 6 (Gate 6b). DEV only, no holdout code path. Decomposition methodology is a regression-based analogue of Hou & Loh (2016)/Branger-Hulsbusch-Middelhoff (2017) Table 16 (leave-one-feature-out on compression's incremental R^2), NOT the original portfolio-sort method - disclosed in the script. Machine-readable copy: results/67_v4_gate6a_sensitivity_decomposition.json.

```
Baseline (full model, consistency-checked against Gate 6a): benchmark R2={r2_bench_full:.6f}  augmented R2={r2_aug_full:.6f}  incremental R2={inc_r2_full:+.6f}
compression coef {comp_coef_full:+.6e}   NW t {comp_t_full:+.4f}

TASK A - leave-one-out sensitivity of compression NW t-stat:
{sens_lines}
  largest single-feature delta: {max_delta_feature} ({loo_results[max_delta_feature]['compression_t_delta_vs_full']:+.4f})

TASK B - decomposition of compression's incremental R2 ({inc_r2_full:+.6f} total):
{decomp_lines}
  RESIDUAL pct_of_compression_power: {residual_pct:+.2f}%
```
"""
with open(log_path, 'a') as f:
    f.write(log_block)
print(f"[OK] appended to {log_path}")

print('\n' + '=' * 96)
print('SENSITIVITY CHECK + DECOMPOSITION COMPLETE. Raw numbers only, no interpretation.')
print('=' * 96)
