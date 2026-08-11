import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr

# ---------------------------------------------------------------------------
# V4 GATE 6a - DIRECT CHECK on what the leave-one-out sensitivity and
# Hou-Loh-style decomposition (src/67) were pointing at but did not measure
# directly: the raw relationship between compression_decile and
# IV_ATM_var, and whether the order in which the two enter a regression
# changes the picture of "who is doing the work."
#
# THREE THINGS ONLY, NO FURTHER STEPS:
#   1. Pooled Pearson and Spearman correlation, compression_decile vs
#      IV_ATM_var, across the full DEV panel.
#   2. Nested regression (a): IV_ATM_var alone, then + compression -
#      compression's incremental R^2 and t-stat in THIS order.
#   3. Nested regression (b): compression alone, then + IV_ATM_var -
#      IV_ATM_var's incremental R^2 and t-stat in THIS order.
#
# Raw numbers only. No interpretation. Logged as its own dated gate_log.md
# entry, explicitly linked to src/67's decomposition entry, flagged
# "under review" - not resolved either direction. STOPS after reporting -
# no hedging, P&L, or Gate 6b work follows this script.
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
out_json = project_root / 'results' / '68_v4_gate6a_correlation_ordering_check.json'
log_path = project_root / 'results' / 'gate_log.md'

print('=' * 96)
print('V4 GATE 6a - CORRELATION + REGRESSION-ORDER CHECK (compression_decile vs IV_ATM_var)')
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
# 1. REBUILD PANEL (identical to src/66/src/67)
# ==========================================================================
print('\n' + '-' * 92)
print('1. REBUILD PANEL (identical construction to src/66/src/67)')
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
# 2. POOLED CORRELATION - compression_decile vs IV_ATM_var
# ==========================================================================
print('\n' + '=' * 92)
print('2. POOLED CORRELATION: compression_decile vs IV_ATM_var (full DEV panel)')
print('=' * 92)

pair = panel[['compression_decile', 'IV_ATM_var']].dropna()
n_pair = len(pair)
pearson_r, pearson_p = pearsonr(pair['compression_decile'], pair['IV_ATM_var'])
spearman_rho, spearman_p = spearmanr(pair['compression_decile'], pair['IV_ATM_var'])

print(f"n (pooled, both non-null): {n_pair:,} of {len(panel):,} "
      f"({n_pair / len(panel) * 100:.2f}%)")
print(f"Pearson r:  {pearson_r:+.6f}   (p-value {pearson_p:.3e})")
print(f"Spearman rho: {spearman_rho:+.6f}   (p-value {spearman_p:.3e})")


# ==========================================================================
# 3. NESTED REGRESSIONS - order (a): IV_ATM_var first, then + compression
# ==========================================================================
print('\n' + '=' * 92)
print('3. NESTED REGRESSION (a): IV_ATM_var alone, then + compression_decile')
print('=' * 92)

fm_a1 = fama_macbeth(panel, 'RV2_primary', ['IV_ATM_var'], NW_MAXLAGS)
fm_a2 = fama_macbeth(panel, 'RV2_primary', ['IV_ATM_var', 'compression_decile'], NW_MAXLAGS)
r2_a1 = fm_a1['mean_daily_r2']
r2_a2 = fm_a2['mean_daily_r2']
inc_r2_a = r2_a2 - r2_a1
comp_in_a = fm_a2['coefs']['compression_decile']

print(f"Step 1: RV2 ~ const + IV_ATM_var                          R2={r2_a1:.6f}  "
      f"n_dates={fm_a1['n_dates']:,}")
print(f"Step 2: RV2 ~ const + IV_ATM_var + compression_decile     R2={r2_a2:.6f}  "
      f"n_dates={fm_a2['n_dates']:,}")
print(f"compression_decile coef (step 2): {comp_in_a['mean_coef']:+.6e}   "
      f"NW t: {comp_in_a['nw_t']:+.4f}")
print(f"compression's incremental R2 in THIS order: {inc_r2_a:+.6f}")


# ==========================================================================
# 4. NESTED REGRESSIONS - order (b): compression first, then + IV_ATM_var
# ==========================================================================
print('\n' + '=' * 92)
print('4. NESTED REGRESSION (b): compression_decile alone, then + IV_ATM_var')
print('=' * 92)

fm_b1 = fama_macbeth(panel, 'RV2_primary', ['compression_decile'], NW_MAXLAGS)
fm_b2 = fama_macbeth(panel, 'RV2_primary', ['compression_decile', 'IV_ATM_var'], NW_MAXLAGS)
r2_b1 = fm_b1['mean_daily_r2']
r2_b2 = fm_b2['mean_daily_r2']
inc_r2_b = r2_b2 - r2_b1
iv_in_b = fm_b2['coefs']['IV_ATM_var']

print(f"Step 1: RV2 ~ const + compression_decile                  R2={r2_b1:.6f}  "
      f"n_dates={fm_b1['n_dates']:,}")
print(f"Step 2: RV2 ~ const + compression_decile + IV_ATM_var     R2={r2_b2:.6f}  "
      f"n_dates={fm_b2['n_dates']:,}")
print(f"IV_ATM_var coef (step 2): {iv_in_b['mean_coef']:+.6e}   NW t: {iv_in_b['nw_t']:+.4f}")
print(f"IV_ATM_var's incremental R2 in THIS order: {inc_r2_b:+.6f}")


# ==========================================================================
# 5. COMPARISON TABLE - three orderings side by side
# ==========================================================================
print('\n' + '=' * 92)
print('5. COMPARISON: compression\'s incremental contribution under three orderings')
print('=' * 92)
print(f"{'ordering':<45} {'compression incR2':>18} {'compression NW t':>18}")
print(f"{'all-10-features, then +compression (src/66)':<45} {'+0.003489':>18} {'-9.3978':>18}")
print(f"{'IV_ATM_var alone, then +compression':<45} {inc_r2_a:>+18.6f} {comp_in_a['nw_t']:>18.4f}")
print(f"{'compression alone, then +IV_ATM_var (IV incR2)':<45} {inc_r2_b:>+18.6f} {iv_in_b['nw_t']:>18.4f}")


# ==========================================================================
# 6. WRITE JSON + LOG (flagged, not resolved)
# ==========================================================================
out = {
    'phase': 'V4 Gate 6a - correlation and regression-order check (UNDER REVIEW)',
    'generated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'script': 'src/68_v4_gate6a_correlation_ordering_check.py',
    'cross_reference': ('direct follow-up to src/67 (sensitivity check + Hou-Loh-style '
                        'decomposition), which flagged IV_ATM_var as the single feature '
                        'materially moving compressions NW t-stat and absorbing >100% of '
                        'its incremental R2 in the leave-one-out attribution'),
    'status': 'UNDER REVIEW - implicates Gate 6a incremental R2 interpretation, not resolved either direction',
    'pooled_correlation': {
        'n_pairs': int(n_pair), 'pearson_r': float(pearson_r), 'pearson_p': float(pearson_p),
        'spearman_rho': float(spearman_rho), 'spearman_p': float(spearman_p),
    },
    'nested_order_a_iv_first': {
        'step1_R2_iv_only': r2_a1, 'step2_R2_iv_plus_compression': r2_a2,
        'compression_incremental_R2': inc_r2_a,
        'compression_coef': comp_in_a['mean_coef'], 'compression_nw_t': comp_in_a['nw_t'],
    },
    'nested_order_b_compression_first': {
        'step1_R2_compression_only': r2_b1, 'step2_R2_compression_plus_iv': r2_b2,
        'iv_incremental_R2': inc_r2_b,
        'iv_coef': iv_in_b['mean_coef'], 'iv_nw_t': iv_in_b['nw_t'],
    },
    'all_10_features_reference_from_src66': {'incremental_R2': 0.003489, 'compression_nw_t': -9.3978},
}
with open(out_json, 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f"\n[OK] wrote {out_json}")

log_block = f"""

---

## V4 Gate 6a - correlation and regression-order check - **UNDER REVIEW** (linked to the decomposition entry above)

Run {out['generated']} - script src/68_v4_gate6a_correlation_ordering_check.py - direct follow-up to the immediately preceding "V4 Gate 6a - sensitivity check and Hou-Loh-style decomposition" entry, which flagged IV_ATM_var as the one feature materially moving compression's NW t-stat (delta -4.03 vs the next-largest -0.85) and attributing >100% of compression's incremental R2 in the leave-one-out decomposition. This entry measures the raw compression_decile / IV_ATM_var relationship and regression-order dependence directly. DEV only, no holdout code path. **STATUS: UNDER REVIEW - implicates Gate 6a's incremental R2 interpretation. NOT resolved in either direction. No hedging, P&L, or Gate 6b work follows this entry.** Machine-readable copy: results/68_v4_gate6a_correlation_ordering_check.json.

```
POOLED CORRELATION (compression_decile vs IV_ATM_var, full DEV panel, n={n_pair:,}):
  Pearson r:    {pearson_r:+.6f}  (p={pearson_p:.3e})
  Spearman rho: {spearman_rho:+.6f}  (p={spearman_p:.3e})

NESTED ORDER (a): IV_ATM_var alone -> + compression_decile
  R2 (IV_ATM_var alone):            {r2_a1:.6f}
  R2 (+ compression_decile):        {r2_a2:.6f}
  compression incremental R2:       {inc_r2_a:+.6f}
  compression coef / NW t:          {comp_in_a['mean_coef']:+.6e}  /  {comp_in_a['nw_t']:+.4f}

NESTED ORDER (b): compression_decile alone -> + IV_ATM_var
  R2 (compression alone):           {r2_b1:.6f}
  R2 (+ IV_ATM_var):                {r2_b2:.6f}
  IV_ATM_var incremental R2:        {inc_r2_b:+.6f}
  IV_ATM_var coef / NW t:           {iv_in_b['mean_coef']:+.6e}  /  {iv_in_b['nw_t']:+.4f}

REFERENCE - all-10-features-then-compression (src/66, Gate 6a primary result):
  compression incremental R2: +0.003489   compression NW t: -9.3978
```
"""
with open(log_path, 'a') as f:
    f.write(log_block)
print(f"[OK] appended to {log_path} (flagged UNDER REVIEW)")

print('\n' + '=' * 96)
print('CORRELATION + ORDERING CHECK COMPLETE. Raw numbers only, no interpretation.')
print('Flagged UNDER REVIEW in gate_log.md - not resolved either direction.')
print('Stopping here - no hedging, P&L, or Gate 6b work follows.')
print('=' * 96)
