import pandas as pd
import numpy as np
from scipy.stats import norm
from pathlib import Path

# ---------------------------------------------------------------------------
# K1 Section 6 sanity check: COUNT-ONLY. How many dev-window V1 decile-1
# stock-days clear the LOCKED T=25% entry rule? This is a sample-size check,
# not a performance test - no return, P&L, win rate, or forward outcome is
# computed or printed anywhere in this script. T is not swept; it is already
# locked in the (uncommitted) K1 pre-registration draft's Section 6.
#
# Pricing spec matches that draft exactly:
#   market_value = matched call impl_premium + matched put impl_premium
#   theoretical_value = Black-Scholes call+put using TRAIL20 as sigma,
#     S = DlyClose on day t, K = average of the two matched legs'
#     impl_strike, T = 30/365, r = RF*252 on day t, q = 0 (no dividend
#     adjustment - stated explicitly below, per the draft's disclosure).
#   qualifies = theoretical_value >= market_value * 1.25
#
# Conventions match src/28 (lowercase columns, cusip '99999999' excluded,
# 8-char uppercased CUSIP bridge, trading-day merge_asof tolerance +/- 3
# days) and src/21 (TRAIL20 = rolling(20, min_periods=15).std()*sqrt(252),
# no shift - the forecast made at day t's close using data through t).
# vol_surface.csv is already restricted upstream (src/27) to days in
# {10, 30} and near-ATM delta [0.35, 0.65]; this script filters to days==30
# only and does not need to re-apply the delta band.
# ---------------------------------------------------------------------------

DEV_START = '2015-01-01'
DEV_END = '2021-12-31'
TENOR = 30
DATE_TOL_TDAYS = 3
T_THRESHOLD = 0.25          # LOCKED in the K1 draft's Section 6 - not swept
CHUNKSIZE = 5_000_000

project_root = Path(__file__).parent.parent
surf_path = project_root / 'data' / 'raw' / 'optionmetrics' / 'vol_surface.csv'
om_path = project_root / 'data' / 'raw' / 'optionmetrics' / 'om_security_names.csv'
v1_path = project_root / 'data' / 'processed' / 'compression_signal_v1.parquet'
univ_path = project_root / 'data' / 'processed' / 'universe_membership.parquet'
names_path = project_root / 'data' / 'raw' / 'crsp' / 'crsp_names.parquet'
crsp_path = project_root / 'data' / 'processed' / 'crsp_combined.parquet'
fac_path = project_root / 'data' / 'processed' / 'factors_daily.parquet'

print("=" * 78)
print("K1 SECTION 6 SANITY CHECK - COUNTS ONLY, T=25% (locked, not swept)")
print("No returns, P&L, win rate, or forward outcome computed or printed.")
print("=" * 78)
print("\nAssumption stated per spec: q = 0, no dividend adjustment in the")
print("Black-Scholes theoretical value.")

# --------------------------------------------------------------------------
# Surface: read in chunks (already filtered upstream to days in {10,30} and
# near-ATM delta by src/27), keep days==30 only, reduce to the best
# (closest to |delta|=0.50) row per secid-date-side, carrying strike+premium.
# --------------------------------------------------------------------------
peek = pd.read_csv(surf_path, nrows=5)
usecols = [c for c in peek.columns if c.lower() in
           ('secid', 'date', 'days', 'delta', 'impl_volatility',
            'impl_strike', 'impl_premium', 'cp_flag')]
DT = {'secid': 'int32', 'date': 'str', 'days': 'int16', 'delta': 'float32',
      'impl_volatility': 'float64', 'impl_strike': 'float64',
      'impl_premium': 'float64', 'cp_flag': 'category'}

print(f"\nScanning {surf_path.name} in chunks of {CHUNKSIZE:,}, keeping "
      f"days=={TENOR} and reducing to the best near-ATM point per "
      f"secid-date-side...")
parts = []
rows = 0
reader = pd.read_csv(surf_path, usecols=usecols, dtype=DT,
                     chunksize=CHUNKSIZE)
for i, ch in enumerate(reader, 1):
    ch.columns = [c.lower() for c in ch.columns]
    rows += len(ch)
    sel = ch[ch['days'] == TENOR].copy()
    if len(sel):
        sel['dpen'] = (sel['delta'].abs() / 100.0 - 0.50).abs()
        sel = (sel.sort_values(['secid', 'date', 'cp_flag', 'dpen'])
               .drop_duplicates(['secid', 'date', 'cp_flag'], keep='first'))
        parts.append(sel[['secid', 'date', 'cp_flag', 'impl_strike',
                          'impl_premium', 'dpen']])
    if i % 5 == 0 or i == 1:
        print(f"  chunk {i:>3}: {rows:>14,} rows scanned")

side = pd.concat(parts, ignore_index=True)
del parts
side = (side.sort_values(['secid', 'date', 'cp_flag', 'dpen'])
        .drop_duplicates(['secid', 'date', 'cp_flag'], keep='first'))
print(f"\nRows scanned: {rows:,}; best-per-side rows retained: {len(side):,}")

calls = (side[side['cp_flag'] == 'C']
         [['secid', 'date', 'impl_strike', 'impl_premium']]
         .rename(columns={'impl_strike': 'k_call', 'impl_premium': 'prem_call'}))
puts = (side[side['cp_flag'] == 'P']
        [['secid', 'date', 'impl_strike', 'impl_premium']]
        .rename(columns={'impl_strike': 'k_put', 'impl_premium': 'prem_put'}))
straddle = calls.merge(puts, on=['secid', 'date'], how='inner')
straddle['market_value'] = straddle['prem_call'] + straddle['prem_put']
straddle['strike'] = (straddle['k_call'] + straddle['k_put']) / 2.0
straddle['date'] = pd.to_datetime(straddle['date'], cache=True)
print(f"Matched secid-dates with both legs present: {len(straddle):,} "
      f"(consistent with src/28's 100% both-legs-present finding)")

# --------------------------------------------------------------------------
# Bridge: secid -> cusip -> PERMNO (src/23/src/28 conventions)
# --------------------------------------------------------------------------
om = pd.read_csv(om_path)
om.columns = [c.lower() for c in om.columns]
om = om[om['cusip'] != '99999999'].dropna(subset=['cusip', 'secid'])
om['c8'] = om['cusip'].astype(str).str.upper().str[:8]

univ = pd.read_parquet(univ_path)
upermnos = set(univ.loc[univ['in_universe'], 'PERMNO'].unique())
names = pd.read_parquet(names_path)
crsp_map = (names[names['PERMNO'].isin(upermnos)][['PERMNO', 'CUSIP']]
            .dropna(subset=['CUSIP']).drop_duplicates())
crsp_map['c8'] = crsp_map['CUSIP'].astype(str).str.upper().str[:8]
bridge = (crsp_map.merge(om[['secid', 'c8']].drop_duplicates(), on='c8',
                         how='inner')[['PERMNO', 'secid']].drop_duplicates())
name_map = (names.sort_values('SecInfoStartDt')
            .drop_duplicates('PERMNO', keep='last')
            .set_index('PERMNO')['IssuerNm'])
print(f"\nBridge: {len(bridge):,} PERMNO-secid pairs, "
      f"{bridge['PERMNO'].nunique():,} PERMNOs")

# --------------------------------------------------------------------------
# V1 decile-1 stock-days, DEV WINDOW ONLY, with CRSP size decile
# --------------------------------------------------------------------------
v1 = pd.read_parquet(v1_path,
                     columns=['PERMNO', 'DlyCalDt', 'compression_decile'])
d1 = v1[(v1['compression_decile'] == 1) &
        (v1['DlyCalDt'] >= DEV_START) & (v1['DlyCalDt'] <= DEV_END)
        ][['PERMNO', 'DlyCalDt']].copy()
d1['year_month'] = d1['DlyCalDt'].dt.to_period('M').astype(str)
umem = univ[univ['in_universe']][['PERMNO', 'year_month', 'decile']]
d1 = d1.merge(umem, on=['PERMNO', 'year_month'], how='left')
print(f"\nV1 decile-1 stock-days, dev window: {len(d1):,} "
      f"({d1['PERMNO'].nunique():,} PERMNOs)")

# --------------------------------------------------------------------------
# TRAIL20 and spot close, "as of day t" (src/21 construction, no shift)
# --------------------------------------------------------------------------
px = pd.read_parquet(crsp_path,
                     columns=['PERMNO', 'DlyCalDt', 'DlyRet', 'DlyClose'])
px = px[px['PERMNO'].isin(d1['PERMNO'].unique())]
px = px[(px['DlyCalDt'] >= '2014-01-01') & (px['DlyCalDt'] <= DEV_END)]
# established multi-distribution dedupe (audited in src/03_signal.py)
px = (px.drop_duplicates(['PERMNO', 'DlyCalDt'], keep='first')
        .sort_values(['PERMNO', 'DlyCalDt']).reset_index(drop=True))
px['trail20'] = px.groupby('PERMNO', sort=False)['DlyRet'].transform(
    lambda s: s.rolling(20, min_periods=15).std()) * np.sqrt(252)
print(f"\nTRAIL20 built on {len(px):,} daily rows "
      f"({px['PERMNO'].nunique():,} PERMNOs, 2014 warmup + dev)")

fac = pd.read_parquet(fac_path, columns=['date', 'RF'])
fac['r_annual'] = fac['RF'] * 252.0

d1 = d1.merge(px[['PERMNO', 'DlyCalDt', 'trail20', 'DlyClose']],
             on=['PERMNO', 'DlyCalDt'], how='left')
d1 = d1.merge(fac[['date', 'r_annual']], left_on='DlyCalDt',
             right_on='date', how='left').drop(columns='date')

# --------------------------------------------------------------------------
# Match each decile-1 stock-day to the nearest straddle (secid-date) within
# +/- 3 trading days, same tolerance/logic as src/28.
# --------------------------------------------------------------------------
cal = pd.Index(pd.Series(v1['DlyCalDt'].unique()).sort_values())
straddle = straddle[straddle['date'] >= cal[0] - pd.Timedelta(days=7)].copy()
straddle['tidx'] = cal.get_indexer(straddle['date'], method='nearest')
d1['tidx'] = cal.get_indexer(d1['DlyCalDt'], method='nearest')

cand = d1.merge(bridge, on='PERMNO', how='left')
left = (cand[cand['secid'].notna()]
        .astype({'secid': 'int64'}).sort_values('tidx'))
right = (straddle.astype({'secid': 'int64'}).sort_values('tidx')
         .rename(columns={'date': 'surf_date'})
         [['secid', 'tidx', 'surf_date', 'market_value', 'strike']])
joined = pd.merge_asof(left, right, on='tidx', by='secid',
                       direction='nearest', tolerance=DATE_TOL_TDAYS)
joined['hit'] = joined['market_value'].notna()
per_day = (joined.sort_values('hit', ascending=False)
           .drop_duplicates(['PERMNO', 'DlyCalDt'], keep='first'))
res = d1.merge(
    per_day[['PERMNO', 'DlyCalDt', 'hit', 'market_value', 'strike']],
    on=['PERMNO', 'DlyCalDt'], how='left')
res['hit'] = res['hit'].fillna(False)

matched = res[res['hit']].copy()
print(f"\nDev decile-1 stock-days matched to a straddle (both legs, "
      f"within +/-{DATE_TOL_TDAYS} trading days): {len(matched):,} of "
      f"{len(res):,}")

# --------------------------------------------------------------------------
# Black-Scholes theoretical value, using S/r/sigma at day t and the
# matched strike. q = 0.
# --------------------------------------------------------------------------
S = matched['DlyClose'].to_numpy()
K = matched['strike'].to_numpy()
r = matched['r_annual'].to_numpy()
sigma = matched['trail20'].to_numpy()
Tyr = TENOR / 365.0

valid = (S > 0) & (K > 0) & (sigma > 0) & np.isfinite(S) & np.isfinite(K) & \
        np.isfinite(sigma) & np.isfinite(r)
n_degenerate = int((~valid).sum())
print(f"Rows with degenerate BS inputs (S/K/sigma<=0 or non-finite), "
      f"excluded: {n_degenerate:,} of {len(matched):,}")

theo = np.full(len(matched), np.nan)
with np.errstate(all='ignore'):
    d1_ = (np.log(S[valid] / K[valid]) +
          (r[valid] + 0.5 * sigma[valid] ** 2) * Tyr) / (sigma[valid] * np.sqrt(Tyr))
    d2_ = d1_ - sigma[valid] * np.sqrt(Tyr)
    call = S[valid] * norm.cdf(d1_) - K[valid] * np.exp(-r[valid] * Tyr) * norm.cdf(d2_)
    put = K[valid] * np.exp(-r[valid] * Tyr) * norm.cdf(-d2_) - S[valid] * norm.cdf(-d1_)
    theo[valid] = call + put

matched['theoretical_value'] = theo
matched['qualifies'] = (matched['theoretical_value'] >=
                        matched['market_value'] * (1.0 + T_THRESHOLD))
matched.loc[~valid, 'qualifies'] = False

qual = matched[matched['qualifies']].copy()

# --------------------------------------------------------------------------
# REPORT: counts and identity only
# --------------------------------------------------------------------------
print("\n" + "=" * 78)
print(f"SECTION 6 RULE (T=25%): counts of qualifying dev-window decile-1 "
      f"stock-days")
print("=" * 78)

print(f"\nTotal matched (priceable) candidate stock-days: {len(matched):,}")
print(f"TOTAL QUALIFYING STOCK-DAYS: {len(qual):,} "
      f"({len(qual)/len(matched)*100:.2f}% of matched candidates)")
print(f"UNIQUE PERMNOs among qualifying stock-days: "
      f"{qual['PERMNO'].nunique():,}")
print(f"  (overlapping positions on the same stock are allowed by design; "
      f"this separates trade volume from actual breadth)")

print(f"\nBreakdown by CRSP decile:")
print(f"  {'decile':>7} {'qualifying':>11} {'matched':>10} {'qual pct':>9}")
for dec in [6, 7, 8]:
    m = matched[matched['decile'] == dec]
    q = qual[qual['decile'] == dec]
    if len(m):
        print(f"  {dec:>7} {len(q):>11,} {len(m):>10,} "
              f"{len(q)/len(m)*100:>8.2f}%")

print(f"\nBreakdown by year:")
qual['year'] = qual['DlyCalDt'].dt.year
matched['year'] = matched['DlyCalDt'].dt.year
print(f"  {'year':>6} {'qualifying':>11} {'matched':>10} {'qual pct':>9}")
for y in sorted(matched['year'].unique()):
    m = matched[matched['year'] == y]
    q = qual[qual['year'] == y]
    print(f"  {y:>6} {len(q):>11,} {len(m):>10,} "
          f"{len(q)/len(m)*100 if len(m) else 0:>8.2f}%")

print(f"\nTop 10 most-frequently-qualifying PERMNOs (concentration check):")
top10 = qual.groupby('PERMNO').size().sort_values(ascending=False).head(10)
print(f"  {'PERMNO':>7} {'qual days':>10}  company")
for p, n in top10.items():
    nm = name_map.get(p, '(name unavailable)')
    print(f"  {p:>7} {n:>10,}  {str(nm)[:40]}")
top10_share = top10.sum() / len(qual) * 100 if len(qual) else 0
print(f"\n  Top 10 PERMNOs account for {top10.sum():,} of {len(qual):,} "
      f"qualifying stock-days ({top10_share:.1f}%)")

print("\n" + "=" * 78)
print("SANITY CHECK COMPLETE - counts and stock identity only. No return,")
print("P&L, win rate, or forward outcome was computed anywhere above.")
print("=" * 78)
