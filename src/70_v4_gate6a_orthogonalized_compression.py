import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# V4 GATE 6a - CLOSE-OUT: orthogonalized compression measure follow-up.
#
#   1. compression_orthogonal := residual of compression_decile ~ const +
#      the other 9 benchmark features (pooled OLS) - REUSES the exact
#      partial-correlation regression already built in src/69, not a new
#      construction. Orthogonalized against the other 9 only, NOT against
#      IV_ATM_var itself - IV_ATM_var remains a separate regressor in the
#      full model, since the point is to see compression's incremental
#      content once cleaned of overlap with the other 9, while still
#      competing with IV_ATM_var normally (the suppression story's own
#      subject) in the augmented regression.
#   2. Re-run Gate 6a's Fama-MacBeth regression with compression_orthogonal
#      in place of raw compression_decile, same 10-feature benchmark, DEV
#      only, no holdout.
#   3. Re-run the leave-one-out Hou-Loh-style decomposition (src/67's exact
#      method) using compression_orthogonal, fixed throughout (not
#      re-orthogonalized per iteration), mirroring how raw
#      compression_decile was held fixed in src/67.
#
# Raw numbers only, no verdict language beyond what results/61's own
# consolidation summary needs. STOPS after reporting - no Gate 6b,
# hedging, or P&L work follows.
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
out_json = project_root / 'results' / '70_v4_gate6a_orthogonalized_compression.json'
log_path = project_root / 'results' / 'gate_log.md'

print('=' * 96)
print('V4 GATE 6a CLOSE-OUT: orthogonalized compression measure')
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


BENCHMARK_COLS = ['RV1', 'Persist20', 'RV20', 'RV60', 'DSV5', 'JumpVar', 'VoV20',
                  'IV_ATM_var', 'IVTermStruct_var', 'IVSkew_var']
OTHER_9 = [c for c in BENCHMARK_COLS if c != 'IV_ATM_var']


# ==========================================================================
# 1. REBUILD PANEL (identical construction to src/66-69)
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
print(f"Panel rebuilt: {len(panel):,} rows")


# ==========================================================================
# 2. ORTHOGONALIZED COMPRESSION MEASURE (reuses src/69's exact regression)
# ==========================================================================
print('\n' + '-' * 92)
print('2. compression_orthogonal := resid(compression_decile ~ const + other 9), pooled OLS')
print('-' * 92)

cols_needed = ['compression_decile'] + OTHER_9
pc = panel[cols_needed].dropna()
n_pc = len(pc)
Z = np.column_stack([np.ones(n_pc)] + [pc[c].to_numpy(float) for c in OTHER_9])
y_comp = pc['compression_decile'].to_numpy(float)
beta_comp, _, _, _ = np.linalg.lstsq(Z, y_comp, rcond=None)
resid_comp = y_comp - Z @ beta_comp
r2_comp_on_others = 1.0 - np.sum(resid_comp ** 2) / np.sum((y_comp - y_comp.mean()) ** 2)
print(f"n (pooled, compression_decile + other 9 non-null): {n_pc:,}")
print(f"R2 of compression_decile ~ other 9 (consistency check vs src/69's 0.077720): "
      f"{r2_comp_on_others:.6f}")

panel = panel.copy()
panel['compression_orthogonal'] = np.nan
panel.loc[pc.index, 'compression_orthogonal'] = resid_comp
print(f"compression_orthogonal: mean={panel['compression_orthogonal'].mean():+.6e}  "
      f"std={panel['compression_orthogonal'].std():.6e}  "
      f"non-null={panel['compression_orthogonal'].notna().sum():,}")


# ==========================================================================
# 3. GATE 6a REGRESSION - orthogonalized compression, same 10-feature benchmark
# ==========================================================================
print('\n' + '=' * 92)
print('3. GATE 6a REGRESSION - compression_orthogonal in place of compression_decile')
print('=' * 92)

fm_bench = fama_macbeth(panel, 'RV2_primary', BENCHMARK_COLS, NW_MAXLAGS)
fm_aug_orth = fama_macbeth(panel, 'RV2_primary', BENCHMARK_COLS + ['compression_orthogonal'], NW_MAXLAGS)

r2_bench = fm_bench['mean_daily_r2']
r2_aug_orth = fm_aug_orth['mean_daily_r2']
inc_r2_orth = r2_aug_orth - r2_bench
comp_orth = fm_aug_orth['coefs']['compression_orthogonal']

print(f"benchmark-only R2 (unchanged): {r2_bench:.6f}  [reference 0.282186]")
print(f"augmented R2 (with compression_orthogonal): {r2_aug_orth:.6f}")
print(f"compression_orthogonal coef: {comp_orth['mean_coef']:+.6e}   "
      f"NW t: {comp_orth['nw_t']:+.4f}   n_dates: {comp_orth['n_dates_used']:,}")
print(f"incremental R2 (orthogonalized): {inc_r2_orth:+.6f}")

print(f"\n--- SIDE BY SIDE ---")
print(f"{'measure':<24} {'compression NW t':>18} {'incremental R2':>18}")
print(f"{'raw compression_decile':<24} {-9.3978:>18.4f} {0.003489:>18.6f}")
print(f"{'compression_orthogonal':<24} {comp_orth['nw_t']:>18.4f} {inc_r2_orth:>18.6f}")


# ==========================================================================
# 4. LEAVE-ONE-OUT DECOMPOSITION with compression_orthogonal (fixed, not
#    re-orthogonalized per iteration - mirrors src/67's treatment of raw
#    compression_decile as a fixed regressor throughout)
# ==========================================================================
print('\n' + '=' * 92)
print('4. HOU-LOH-STYLE DECOMPOSITION - compression_orthogonal (same method as src/67)')
print('=' * 92)

loo_results = {}
for j, feat_name in enumerate(BENCHMARK_COLS, 1):
    remaining = [c for c in BENCHMARK_COLS if c != feat_name]
    fm_b = fama_macbeth(panel, 'RV2_primary', remaining, NW_MAXLAGS)
    fm_a = fama_macbeth(panel, 'RV2_primary', remaining + ['compression_orthogonal'], NW_MAXLAGS)
    r2_b = fm_b['mean_daily_r2']
    r2_a = fm_a['mean_daily_r2']
    inc_r2_minus_j = r2_a - r2_b
    comp_t_minus_j = fm_a['coefs']['compression_orthogonal']['nw_t']
    own_coef_full_model = fm_aug_orth['coefs'][feat_name]['mean_coef']

    attribution = inc_r2_minus_j - inc_r2_orth
    pct_attributable = attribution / inc_r2_orth * 100.0 if inc_r2_orth != 0 else np.nan

    loo_results[feat_name] = {
        'own_coef_in_full_augmented_model': own_coef_full_model,
        'compression_orth_t_with_feature_removed': comp_t_minus_j,
        'compression_orth_t_delta_vs_full': comp_t_minus_j - comp_orth['nw_t'],
        'incremental_R2_minus_feature': inc_r2_minus_j,
        'attribution_to_feature': attribution,
        'pct_of_compression_orth_explanatory_power': pct_attributable,
    }
    print(f"  [{j:>2}/10] removed={feat_name:<18}  "
          f"orth-compression NW t={comp_t_minus_j:+.4f}  "
          f"(delta vs full: {comp_t_minus_j - comp_orth['nw_t']:+.4f})  "
          f"incR2_minus_j={inc_r2_minus_j:+.6f}  pct={pct_attributable:+.2f}%")

residual_pct = 100.0 - sum(v['pct_of_compression_orth_explanatory_power'] for v in loo_results.values())
print(f"\nResidual %: {residual_pct:+.2f}%")

print(f"\n{'feature':<18} {'own coefficient':>18} {'% of orth-comp. power':>22}")
for feat_name, r in loo_results.items():
    print(f"{feat_name:<18} {r['own_coef_in_full_augmented_model']:>18.6e} "
          f"{r['pct_of_compression_orth_explanatory_power']:>21.2f}%")
print(f"{'RESIDUAL':<18} {'':>18} {residual_pct:>21.2f}%")


# ==========================================================================
# 5. WRITE JSON + LOG
# ==========================================================================
out = {
    'phase': 'V4 Gate 6a close-out - orthogonalized compression measure',
    'generated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'script': 'src/70_v4_gate6a_orthogonalized_compression.py',
    'cross_reference': ('close-out follow-up to the RESOLVED "V4 Gate 6a - correlation and '
                        'regression-order check" gate_log.md entry (src/68/69) and the '
                        'sensitivity/decomposition entry (src/67)'),
    'orthogonalization': {
        'method': 'resid(compression_decile ~ const + other 9 benchmark features), pooled OLS',
        'other_9': OTHER_9, 'n_pooled': int(n_pc),
        'r2_compression_on_other9': float(r2_comp_on_others),
    },
    'side_by_side': {
        'raw_compression_decile': {'nw_t': -9.3978, 'incremental_R2': 0.003489},
        'compression_orthogonal': {'nw_t': comp_orth['nw_t'], 'incremental_R2': inc_r2_orth,
                                   'coef': comp_orth['mean_coef'], 'n_dates': comp_orth['n_dates_used']},
    },
    'decomposition_orthogonalized': loo_results,
    'residual_pct': residual_pct,
}
with open(out_json, 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f"\n[OK] wrote {out_json}")

decomp_lines = "\n".join(
    f"  {feat_name:<18} coef={r['own_coef_in_full_augmented_model']:+.6e}  "
    f"pct_of_orth_compression_power={r['pct_of_compression_orth_explanatory_power']:+.2f}%"
    for feat_name, r in loo_results.items())

log_block = f"""

---

## V4 Gate 6a CLOSE-OUT - orthogonalized compression measure

Run {out['generated']} - script src/70_v4_gate6a_orthogonalized_compression.py - close-out follow-up to the RESOLVED "V4 Gate 6a - correlation and regression-order check" entry above (src/68/69) and the sensitivity/decomposition entry (src/67). compression_orthogonal = resid(compression_decile ~ const + other 9 benchmark features, pooled OLS) - reuses src/69's exact partial-correlation regression. DEV only, no holdout code path. Machine-readable copy: results/70_v4_gate6a_orthogonalized_compression.json.

```
SIDE BY SIDE:
  raw compression_decile:    NW t = -9.3978   incremental R2 = +0.003489
  compression_orthogonal:    NW t = {comp_orth['nw_t']:+.4f}   incremental R2 = {inc_r2_orth:+.6f}
  compression_orthogonal coef: {comp_orth['mean_coef']:+.6e}   n_dates: {comp_orth['n_dates_used']:,}

DECOMPOSITION (compression_orthogonal, same leave-one-out method as src/67):
{decomp_lines}
  RESIDUAL: {residual_pct:+.2f}%
```
"""
with open(log_path, 'a') as f:
    f.write(log_block)
print(f"[OK] appended to {log_path}")

print('\n' + '=' * 96)
print('GATE 6a CLOSE-OUT COMPLETE. Stopping - no Gate 6b, hedging, or P&L work follows.')
print('=' * 96)
