import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# V4 Gate 6a - Step 2, PART 1: realized-measure features (1,2,3,4,5,6,10),
# all CRSP-only, no OptionMetrics pull needed. Per results/61_v4_gate6_
# benchmark_model_spec.md, all seven flags resolved 2026-08-04.
#
# Features built here:
#   1. RV1     - 1-day realized variance (annualized)
#   2. Persist - 20-session rolling lag-1 autocorrelation of RV1
#   3. RV20/RV60 - 20/60-day realized variance (annualized)
#   4. DSV5    - 5-day downside (semi-)variance (annualized)
#   5. JumpVar - RV20 - BPV20 (daily-frequency proxy, floored at 0)
#   6. VoV20   - 20-day rolling stdev of |DlyRet|
#   10. MktRV5 - 5-day realized variance of vwretd (market-level, DEV-blind
#       return-scoped)
#
# All shift(1)-aligned. Panel = the same base universe V3/V4 already use
# (decile 6-8, DEV, V1 compression-defined), matching V3/V4's own
# Fama-MacBeth estimation population - not restricted to tradeable
# candidates, since Gate 6a is a forecasting-level test.
# ---------------------------------------------------------------------------

DEV_START = pd.Timestamp('2015-01-01')
DEV_END = pd.Timestamp('2021-12-31')
PRIOR_VOL_WIN = 20          # reused, E1/V3 precedent
LONG_WIN_A, LONG_WIN_B = 20, 60
DOWNSIDE_WIN = 5
JUMP_WIN = 20
VOV_WIN = 20
MKT_WIN = 5
TRADING_DAYS_PER_YEAR = 252.0

project_root = Path(__file__).parent.parent
crsp_path = project_root / 'data' / 'processed' / 'crsp_combined.parquet'
univ_path = project_root / 'data' / 'processed' / 'universe_membership.parquet'
v1_path = project_root / 'data' / 'processed' / 'compression_signal_v1.parquet'
out_path = project_root / 'data' / 'processed' / 'v4_gate6a_features_realized.parquet'

print('=' * 96)
print('V4 GATE 6A - REALIZED-MEASURE FEATURES (1,2,3,4,5,6,10) - CRSP only')
print('=' * 96)

t0 = time.time()

univ = pd.read_parquet(univ_path)
univ_in = univ[univ['in_universe']][['PERMNO', 'year_month', 'decile']].copy()
ever = set(univ_in['PERMNO'].unique())
print(f"Ever-in-universe PERMNOs: {len(ever):,}")

base_v1 = pd.read_parquet(v1_path, columns=['PERMNO', 'DlyCalDt', 'compression_ratio',
                                            'compression_decile'])
base_v1 = base_v1[(base_v1['DlyCalDt'] >= DEV_START) & (base_v1['DlyCalDt'] <= DEV_END)]
base_v1 = base_v1[base_v1['compression_ratio'].notna()]
base_v1['year_month'] = base_v1['DlyCalDt'].dt.to_period('M').astype(str)
base_v1 = base_v1.merge(univ_in, on=['PERMNO', 'year_month'], how='inner')
base_v1 = base_v1[['PERMNO', 'DlyCalDt', 'decile', 'compression_decile']].drop_duplicates(
    ['PERMNO', 'DlyCalDt'])
print(f"shared base universe: {len(base_v1):,} stock-days [reference: 1,733,857]")
assert len(base_v1) == 1_733_857

# --------------------------------------------------------------------------
# Daily panel - PERMNO, DlyCalDt, DlyRet, extended slightly before DEV_START
# so 60-day rolling windows are populated at the very start of DEV (not
# padded, not substituted - just enough real history to fill min_periods).
# --------------------------------------------------------------------------
need = ['PERMNO', 'DlyCalDt', 'DlyRet']
d = pd.read_parquet(crsp_path, columns=need)
d = d[d['PERMNO'].isin(ever)]
d = d[(d['DlyCalDt'] >= '2014-06-01') & (d['DlyCalDt'] <= DEV_END)]
d = d.drop_duplicates(['PERMNO', 'DlyCalDt'], keep='first')
d = d.sort_values(['PERMNO', 'DlyCalDt']).reset_index(drop=True)
print(f"Daily rows (extended pre-DEV for rolling windows): {len(d):,}  "
      f"PERMNOs: {d['PERMNO'].nunique():,}")

g = d.groupby('PERMNO', sort=False)
r = d['DlyRet']

print('\nComputing features 1, 3, 4, 5, 6...')

# Feature 1: 1-day realized variance, annualized, shift(1)
d['RV1'] = g['DlyRet'].transform(lambda s: (s.shift(1) ** 2) * TRADING_DAYS_PER_YEAR)

# Feature 3: 20-day and 60-day realized variance, annualized, shift(1)
d['RV20'] = g['DlyRet'].transform(
    lambda s: s.rolling(LONG_WIN_A, min_periods=15).std(ddof=1).shift(1)) ** 2 * TRADING_DAYS_PER_YEAR
d['RV60'] = g['DlyRet'].transform(
    lambda s: s.rolling(LONG_WIN_B, min_periods=45).std(ddof=1).shift(1)) ** 2 * TRADING_DAYS_PER_YEAR

# Feature 4: 5-day downside semi-variance, annualized (252/5), shift(1)
neg_sq = (d['DlyRet'].where(d['DlyRet'] < 0, 0.0)) ** 2
d['_neg_sq'] = neg_sq
d['DSV5'] = d.groupby('PERMNO')['_neg_sq'].transform(
    lambda s: s.rolling(DOWNSIDE_WIN, min_periods=DOWNSIDE_WIN).sum().shift(1)
) * (TRADING_DAYS_PER_YEAR / DOWNSIDE_WIN)
d = d.drop(columns=['_neg_sq'])

# Feature 5: daily-frequency bipower variation proxy over 20 sessions,
# then JumpVar = max(RV20 - BPV20, 0). BPV computed on shift(1)-aligned
# returns (i.e. using only information through t-1), consistent with RV20.
abs_r = d['DlyRet'].abs()
d['_abs_r'] = abs_r
d['_abs_r_lag1_within'] = d.groupby('PERMNO')['_abs_r'].shift(1)
prod = d['_abs_r'] * d['_abs_r_lag1_within']
d['_bpv_prod'] = prod
bpv_sum = d.groupby('PERMNO')['_bpv_prod'].transform(
    lambda s: s.rolling(JUMP_WIN, min_periods=JUMP_WIN - 1).sum().shift(1))
d['BPV20'] = (np.pi / 2.0) * bpv_sum * (TRADING_DAYS_PER_YEAR / JUMP_WIN)
d['JumpVar'] = (d['RV20'] - d['BPV20']).clip(lower=0.0)
d = d.drop(columns=['_abs_r', '_abs_r_lag1_within', '_bpv_prod'])

# Feature 6: 20-day rolling stdev of |DlyRet| (vol-of-vol), shift(1)
d['VoV20'] = g['DlyRet'].transform(
    lambda s: s.abs().rolling(VOV_WIN, min_periods=VOV_WIN).std(ddof=1).shift(1))

print('Computing feature 2 (20-session rolling autocorrelation of RV1) - vectorized...')

# Feature 2: rolling lag-1 autocorrelation of the RV1 series over the prior
# 20 sessions. Computed via the standard rolling-moment formula (all
# vectorized pandas rolling().sum() calls) rather than
# rolling().apply(lambda ...) with a Python callback per window - the
# latter is a well-known slow pandas anti-pattern at this scale (~3,700
# PERMNOs x ~1,700 rows). RV1 is already shift(1)-aligned, so this
# operates on already-lagged values throughout - no further shift needed.
#
# corr(x_t, x_{t-1}) over a window of W pairs:
#   num = W*sum(x*xlag) - sum(x)*sum(xlag)
#   den = sqrt[(W*sum(x^2)-sum(x)^2) * (W*sum(xlag^2)-sum(xlag)^2)]
t_pc = time.time()
x = d['RV1']
xlag = d.groupby('PERMNO')['RV1'].shift(1)
xxlag = x * xlag
x2 = x * x
xlag2 = xlag * xlag
valid = x.notna() & xlag.notna()

W = PRIOR_VOL_WIN
gW = d.groupby('PERMNO')
sum_x = gW['RV1'].transform(lambda s: s.rolling(W, min_periods=W).sum())
sum_xlag = xlag.groupby(d['PERMNO']).transform(lambda s: s.rolling(W, min_periods=W).sum())
sum_xxlag = xxlag.groupby(d['PERMNO']).transform(lambda s: s.rolling(W, min_periods=W).sum())
sum_x2 = x2.groupby(d['PERMNO']).transform(lambda s: s.rolling(W, min_periods=W).sum())
sum_xlag2 = xlag2.groupby(d['PERMNO']).transform(lambda s: s.rolling(W, min_periods=W).sum())
n_valid = valid.astype(float).groupby(d['PERMNO']).transform(lambda s: s.rolling(W, min_periods=W).sum())

num = n_valid * sum_xxlag - sum_x * sum_xlag
den = np.sqrt((n_valid * sum_x2 - sum_x ** 2) * (n_valid * sum_xlag2 - sum_xlag ** 2))
with np.errstate(all='ignore'):
    autocorr = np.where(den > 0, num / den, np.nan)
d['Persist20'] = autocorr
print(f"Feature 2 complete (vectorized): {(time.time()-t_pc)/60:.2f} min")

# --------------------------------------------------------------------------
# Feature 10: market realized variance, from vwretd (E1 5.1 precedent -
# vwretd over sprtrn because the universe is deciles 6-8, mid-caps).
# --------------------------------------------------------------------------
print('\nComputing feature 10 (market RV, vwretd, E1 5.1 precedent)...')
mkt = pd.read_parquet(crsp_path, columns=['DlyCalDt', 'vwretd'])
mkt = mkt.drop_duplicates('DlyCalDt').sort_values('DlyCalDt').reset_index(drop=True)
assert mkt['vwretd'].isna().sum() == 0, 'vwretd has nulls over the loaded range - unexpected per E1 5.1'
mkt['MktRV5'] = (mkt['vwretd'].rolling(MKT_WIN, min_periods=MKT_WIN).std(ddof=1).shift(1)) ** 2 * TRADING_DAYS_PER_YEAR
mkt = mkt[['DlyCalDt', 'MktRV5']]
print(f"Market panel: {len(mkt):,} dates, MktRV5 non-null: {mkt['MktRV5'].notna().sum():,}")

d = d.merge(mkt, on='DlyCalDt', how='left')

# --------------------------------------------------------------------------
# Restrict to DEV base universe, attach compression, write
# --------------------------------------------------------------------------
d = d[(d['DlyCalDt'] >= DEV_START) & (d['DlyCalDt'] <= DEV_END)]
panel = base_v1.merge(d, on=['PERMNO', 'DlyCalDt'], how='left')
print(f"\nFinal panel: {len(panel):,} rows (base universe, DEV)")

feat_cols = ['RV1', 'Persist20', 'RV20', 'RV60', 'DSV5', 'JumpVar', 'VoV20', 'MktRV5']
print('\nRequired console output - per-feature summary stats over the DEV panel:')
for c in feat_cols:
    s = panel[c]
    n_miss = s.isna().sum()
    print(f"  {c:<10} mean={s.mean():+.6e}  std={s.std():.6e}  "
          f"min={s.min():+.6e}  max={s.max():+.6e}  "
          f"missing={n_miss:,} ({n_miss / len(panel) * 100:.2f}%)")

out_cols = ['PERMNO', 'DlyCalDt', 'decile', 'compression_decile'] + feat_cols
panel[out_cols].to_parquet(out_path, index=False)
print(f"\n[OK] wrote {out_path}  ({len(panel):,} rows)")
print(f"Total wall time: {(time.time()-t0)/60:.2f} min")
print('=' * 96)
