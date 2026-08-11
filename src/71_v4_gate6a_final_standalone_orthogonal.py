import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# V4 GATE 6a - FINAL CLOSING TEST.
#
#   1. compression_orthogonal_full := resid(compression_decile ~ const +
#      ALL 10 benchmark features, including IV_ATM_var), pooled OLS - the
#      version of the orthogonalization that src/70 could not run, because
#      re-inserting a variable into a model that contains everything it was
#      orthogonalized against is mechanically zero-incremental by
#      Frisch-Waugh-Lovell.
#   2. Test compression_orthogonal_full as a STANDALONE univariate
#      Fama-MacBeth forecaster of RV2_primary - NOT re-inserted into the
#      10-feature model. Report its own R^2 and NW t-stat directly. This
#      is the number that answers: does compression carry any forecasting
#      content once everything in the benchmark, including IV level, is
#      stripped out.
#
# Raw number, reported plainly, no reframing regardless of size or
# significance. STOPS after reporting - no Gate 6b, hedging, or P&L work.
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
out_json = project_root / 'results' / '71_v4_gate6a_final_standalone_orthogonal.json'
log_path = project_root / 'results' / 'gate_log.md'

print('=' * 96)
print('V4 GATE 6a - FINAL TEST: compression_orthogonal_full, standalone univariate forecast')
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


# ==========================================================================
# 1. REBUILD PANEL
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
# 2. compression_orthogonal_full - residualized against ALL 10 benchmark
#    features (including IV_ATM_var)
# ==========================================================================
print('\n' + '-' * 92)
print('2. compression_orthogonal_full := resid(compression_decile ~ const + ALL 10 benchmark)')
print('-' * 92)

cols_needed = ['compression_decile'] + BENCHMARK_COLS
pc = panel[cols_needed].dropna()
n_pc = len(pc)
Z = np.column_stack([np.ones(n_pc)] + [pc[c].to_numpy(float) for c in BENCHMARK_COLS])
y_comp = pc['compression_decile'].to_numpy(float)
beta_comp, _, _, _ = np.linalg.lstsq(Z, y_comp, rcond=None)
resid_comp_full = y_comp - Z @ beta_comp
r2_comp_on_all10 = 1.0 - np.sum(resid_comp_full ** 2) / np.sum((y_comp - y_comp.mean()) ** 2)

print(f"n (pooled, compression_decile + all 10 benchmark non-null): {n_pc:,}")
print(f"R2 of compression_decile ~ all 10 benchmark features: {r2_comp_on_all10:.6f}")
print(f"(for reference: R2 of compression_decile ~ other 9 only, src/69/70: 0.077720)")

panel = panel.copy()
panel['compression_orthogonal_full'] = np.nan
panel.loc[pc.index, 'compression_orthogonal_full'] = resid_comp_full
print(f"compression_orthogonal_full: mean={panel['compression_orthogonal_full'].mean():+.6e}  "
      f"std={panel['compression_orthogonal_full'].std():.6e}  "
      f"non-null={panel['compression_orthogonal_full'].notna().sum():,}")


# ==========================================================================
# 3. STANDALONE UNIVARIATE FAMA-MACBETH FORECAST TEST
# ==========================================================================
print('\n' + '=' * 92)
print('3. STANDALONE UNIVARIATE FORECAST: RV2_primary ~ const + compression_orthogonal_full')
print('   (NOT re-inserted into the 10-feature model - would be mechanically zero by FWL)')
print('=' * 92)

fm_standalone = fama_macbeth(panel, 'RV2_primary', ['compression_orthogonal_full'], NW_MAXLAGS)
r2_standalone = fm_standalone['mean_daily_r2']
coef_standalone = fm_standalone['coefs']['compression_orthogonal_full']

print(f"n_dates: {fm_standalone['n_dates']:,} (dropped {fm_standalone['n_dates_dropped']:,})")
print(f"mean cross-sectional n: {fm_standalone['mean_xsec_n']:.1f}")
print(f"R^2 (own daily cross-sectional, averaged): {r2_standalone:.6f}")
print(f"coefficient: {coef_standalone['mean_coef']:+.6e}")
print(f"NW t-stat (maxlags={NW_MAXLAGS}): {coef_standalone['nw_t']:+.4f}")

# Reference: raw compression_decile alone, unconditional (already logged, src/68 order-b step1)
print(f"\nFor reference (already logged, src/68 nested order (b), step 1):")
print(f"  raw compression_decile alone: R2=0.005387, NW t=(from src/66 single-regressor context)")


# ==========================================================================
# 4. WRITE JSON + LOG
# ==========================================================================
out = {
    'phase': 'V4 Gate 6a - FINAL closing test - compression_orthogonal_full standalone forecast',
    'generated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'script': 'src/71_v4_gate6a_final_standalone_orthogonal.py',
    'cross_reference': ('closes the question flagged as not-yet-run in results/61 consolidation '
                        '(b) and results/70 - answers whether compression carries forecasting '
                        'content independent of ALL 10 benchmark features including IV_ATM_var'),
    'orthogonalization_full': {
        'method': 'resid(compression_decile ~ const + ALL 10 benchmark features), pooled OLS',
        'benchmark_cols': BENCHMARK_COLS, 'n_pooled': int(n_pc),
        'r2_compression_on_all10': float(r2_comp_on_all10),
    },
    'standalone_univariate_forecast': {
        'n_dates': fm_standalone['n_dates'], 'n_dates_dropped': fm_standalone['n_dates_dropped'],
        'mean_xsec_n': fm_standalone['mean_xsec_n'],
        'R2': r2_standalone, 'coefficient': coef_standalone['mean_coef'],
        'nw_t': coef_standalone['nw_t'],
    },
}
with open(out_json, 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f"\n[OK] wrote {out_json}")

log_block = f"""

---

## V4 Gate 6a - FINAL CLOSING TEST - compression_orthogonal_full, standalone univariate forecast

Run {out['generated']} - script src/71_v4_gate6a_final_standalone_orthogonal.py - closes the question left open by results/70 (which could only orthogonalize compression against 9 of the 10 benchmark features, since re-inserting a variable orthogonalized against everything already in its own model is mechanically zero-incremental by Frisch-Waugh-Lovell). compression_orthogonal_full = resid(compression_decile ~ const + ALL 10 benchmark features including IV_ATM_var), pooled OLS. Tested as a STANDALONE univariate Fama-MacBeth forecaster of RV2_primary, NOT reinserted into the 10-feature model. DEV only, no holdout code path. Machine-readable copy: results/71_v4_gate6a_final_standalone_orthogonal.json.

```
compression_decile ~ all 10 benchmark features (pooled OLS): R2 = {r2_comp_on_all10:.6f}

STANDALONE FORECAST: RV2_primary ~ const + compression_orthogonal_full
  n_dates: {fm_standalone['n_dates']:,} (dropped {fm_standalone['n_dates_dropped']:,})   mean_xsec_n: {fm_standalone['mean_xsec_n']:.1f}
  R^2:          {r2_standalone:.6f}
  coefficient:  {coef_standalone['mean_coef']:+.6e}
  NW t-stat:    {coef_standalone['nw_t']:+.4f}
```
"""
with open(log_path, 'a') as f:
    f.write(log_block)
print(f"[OK] appended to {log_path}")

print('\n' + '=' * 96)
print('FINAL TEST COMPLETE. Reported plainly, no reframing. Stopping - no Gate 6b, hedging,')
print('or P&L work follows.')
print('=' * 96)
