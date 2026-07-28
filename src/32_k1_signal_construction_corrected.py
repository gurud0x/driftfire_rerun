import pandas as pd
import numpy as np
from scipy.stats import norm
from pathlib import Path

# ---------------------------------------------------------------------------
# Phase K1 CORRECTED construction (theta-decay exit fix).
#
# WHAT THIS FIXES: src/30 valued exits at the price of a FRESH 30-day
# straddle at the exit date. Section 5 of the committed pre-registration
# (commit 3875314) specifies exiting the HELD option - which by exit has
# only ~16-20 calendar days remaining and a strike fixed at entry. That
# omission of theta decay inflated the dev result to an implausible
# +185%/yr. This script implements Section 5 as written; no pre-registered
# number changes.
#
# EXIT VALUATION (the one change from src/30):
#   remaining = 30 - actual calendar days elapsed entry->exit, per trade.
#   sigma_exit: linear-in-days interpolation between the exit date's 10d
#     and 30d near-ATM IVs when BOTH exist (preferred); else the disclosed
#     sqrt-time proxy sigma = IV30_exit * sqrt(remaining/30). Note the
#     proxy typically UNDERSTATES short-tenor IV after vol spikes
#     (term structures invert), i.e. it is conservative for a long-
#     straddle exit.
#   exit value = Black-Scholes call+put at the ORIGINAL strike, exit-date
#     spot and RF, T = remaining/365 - so intrinsic value from spot moves
#     is captured, unlike any fresh-ATM proxy.
#
# Everything else - entry logic, T=25% rule, no-spread cost note,
# overlapping uncapped positions - is identical to src/30.
# ---------------------------------------------------------------------------

DEV_START = '2015-01-01'
DEV_END = '2021-12-31'
TENOR = 30
HOLD_TDAYS = 10
DATE_TOL_TDAYS = 3
T_THRESHOLD = 0.25
CHUNKSIZE = 5_000_000

project_root = Path(__file__).parent.parent
surf_path = project_root / 'data' / 'raw' / 'optionmetrics' / 'vol_surface.csv'
om_path = project_root / 'data' / 'raw' / 'optionmetrics' / 'om_security_names.csv'
v1_path = project_root / 'data' / 'processed' / 'compression_signal_v1.parquet'
univ_path = project_root / 'data' / 'processed' / 'universe_membership.parquet'
names_path = project_root / 'data' / 'raw' / 'crsp' / 'crsp_names.parquet'
crsp_path = project_root / 'data' / 'processed' / 'crsp_combined.parquet'
fac_path = project_root / 'data' / 'processed' / 'factors_daily.parquet'
out_path = (project_root / 'data' / 'processed' /
            'k1_portfolio_returns_daily_corrected.parquet')

print("=" * 78)
print("K1 CORRECTED CONSTRUCTION - theta-decay exit fix (Section 5 as "
      "written)")
print("=" * 78)

# --------------------------------------------------------------------------
# Surface: BOTH tenors this time (30d for entry pricing + IVs; 10d IV for
# exit interpolation). Best near-ATM point per secid-date-side-tenor.
# --------------------------------------------------------------------------
peek = pd.read_csv(surf_path, nrows=5)
usecols = [c for c in peek.columns if c.lower() in
           ('secid', 'date', 'days', 'delta', 'impl_volatility',
            'impl_strike', 'impl_premium', 'cp_flag')]
DT = {'secid': 'int32', 'date': 'str', 'days': 'int16', 'delta': 'float32',
      'impl_volatility': 'float64', 'impl_strike': 'float64',
      'impl_premium': 'float64', 'cp_flag': 'category'}

print(f"\nScanning surface in chunks of {CHUNKSIZE:,} (both tenors)...")
parts = []
rows = 0
for i, ch in enumerate(pd.read_csv(surf_path, usecols=usecols, dtype=DT,
                                   chunksize=CHUNKSIZE), 1):
    ch.columns = [c.lower() for c in ch.columns]
    rows += len(ch)
    ch = ch[ch['impl_volatility'].notna() & (ch['impl_volatility'] > 0)].copy()
    ch['dpen'] = (ch['delta'].abs() / 100.0 - 0.50).abs()
    ch = (ch.sort_values(['secid', 'date', 'days', 'cp_flag', 'dpen'])
          .drop_duplicates(['secid', 'date', 'days', 'cp_flag'],
                           keep='first'))
    parts.append(ch[['secid', 'date', 'days', 'cp_flag', 'impl_strike',
                     'impl_premium', 'impl_volatility', 'dpen']])
    if i % 5 == 0 or i == 1:
        print(f"  chunk {i:>3}: {rows:>14,} rows scanned")

side = pd.concat(parts, ignore_index=True)
del parts
side = (side.sort_values(['secid', 'date', 'days', 'cp_flag', 'dpen'])
        .drop_duplicates(['secid', 'date', 'days', 'cp_flag'], keep='first'))

# 30d: full straddle (premiums + strike + IV); 10d: IV only (for interp)
s30 = side[side['days'] == 30]
c30 = (s30[s30['cp_flag'] == 'C']
       [['secid', 'date', 'impl_strike', 'impl_premium', 'impl_volatility']]
       .rename(columns={'impl_strike': 'k_call', 'impl_premium': 'prem_call',
                        'impl_volatility': 'iv_call30'}))
p30 = (s30[s30['cp_flag'] == 'P']
       [['secid', 'date', 'impl_strike', 'impl_premium', 'impl_volatility']]
       .rename(columns={'impl_strike': 'k_put', 'impl_premium': 'prem_put',
                        'impl_volatility': 'iv_put30'}))
straddle = c30.merge(p30, on=['secid', 'date'], how='inner')
straddle['market_value'] = straddle['prem_call'] + straddle['prem_put']
straddle['strike'] = (straddle['k_call'] + straddle['k_put']) / 2.0
straddle['iv30'] = (straddle['iv_call30'] + straddle['iv_put30']) / 2.0

s10 = side[side['days'] == 10]
iv10 = (s10.groupby(['secid', 'date'], observed=True)['impl_volatility']
        .mean().rename('iv10').reset_index())
straddle = straddle.merge(iv10, on=['secid', 'date'], how='left')
straddle['date'] = pd.to_datetime(straddle['date'], cache=True)
print(f"\nStraddle-priceable secid-dates: {len(straddle):,} "
      f"(with a 10d IV also present: {straddle['iv10'].notna().sum():,}, "
      f"{straddle['iv10'].notna().mean()*100:.1f}%)")

# --------------------------------------------------------------------------
# Bridge, decile-1 dev days, TRAIL20/spot/RF - identical to src/30
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

v1 = pd.read_parquet(v1_path,
                     columns=['PERMNO', 'DlyCalDt', 'compression_decile'])
d1 = v1[(v1['compression_decile'] == 1) &
        (v1['DlyCalDt'] >= DEV_START) & (v1['DlyCalDt'] <= DEV_END)
        ][['PERMNO', 'DlyCalDt']].copy()

px = pd.read_parquet(crsp_path,
                     columns=['PERMNO', 'DlyCalDt', 'DlyRet', 'DlyClose'])
px = px[px['PERMNO'].isin(d1['PERMNO'].unique())]
px = px[(px['DlyCalDt'] >= '2014-01-01') & (px['DlyCalDt'] <= DEV_END)]
px = (px.drop_duplicates(['PERMNO', 'DlyCalDt'], keep='first')
        .sort_values(['PERMNO', 'DlyCalDt']).reset_index(drop=True))
px['trail20'] = px.groupby('PERMNO', sort=False)['DlyRet'].transform(
    lambda s: s.rolling(20, min_periods=15).std()) * np.sqrt(252)

fac = pd.read_parquet(fac_path, columns=['date', 'RF'])
fac['r_annual'] = fac['RF'] * 252.0

d1 = d1.merge(px[['PERMNO', 'DlyCalDt', 'trail20', 'DlyClose']],
              on=['PERMNO', 'DlyCalDt'], how='left')
d1 = d1.merge(fac[['date', 'r_annual']], left_on='DlyCalDt',
              right_on='date', how='left').drop(columns='date')

cal = pd.Index(pd.Series(v1['DlyCalDt'].unique()).sort_values())
straddle = straddle[straddle['date'] >= cal[0] - pd.Timedelta(days=7)].copy()
straddle['tidx'] = cal.get_indexer(straddle['date'], method='nearest')
d1['tidx'] = cal.get_indexer(d1['DlyCalDt'], method='nearest')

cand = d1.merge(bridge, on='PERMNO', how='left')
left = cand[cand['secid'].notna()].astype({'secid': 'int64'}).sort_values('tidx')
right = (straddle.astype({'secid': 'int64'}).sort_values('tidx')
         .rename(columns={'date': 'surf_date'})
         [['secid', 'tidx', 'surf_date', 'market_value', 'strike']])
joined = pd.merge_asof(left, right, on='tidx', by='secid',
                       direction='nearest', tolerance=DATE_TOL_TDAYS)
joined['hit'] = joined['market_value'].notna()
per_day = (joined.sort_values('hit', ascending=False)
           .drop_duplicates(['PERMNO', 'DlyCalDt'], keep='first'))
matched = per_day[per_day['hit']].copy()
print(f"Dev decile-1 stock-days matched: {len(matched):,}")

# --------------------------------------------------------------------------
# Entry rule - identical to src/30
# --------------------------------------------------------------------------
S = matched['DlyClose'].to_numpy()
K = matched['strike'].to_numpy()
r = matched['r_annual'].to_numpy()
sigma = matched['trail20'].to_numpy()
Tyr = TENOR / 365.0
valid = ((S > 0) & (K > 0) & (sigma > 0) & np.isfinite(S) & np.isfinite(K) &
         np.isfinite(sigma) & np.isfinite(r))
theo = np.full(len(matched), np.nan)
with np.errstate(all='ignore'):
    dd1 = ((np.log(S[valid] / K[valid]) +
            (r[valid] + 0.5 * sigma[valid] ** 2) * Tyr) /
           (sigma[valid] * np.sqrt(Tyr)))
    dd2 = dd1 - sigma[valid] * np.sqrt(Tyr)
    c = S[valid] * norm.cdf(dd1) - K[valid] * np.exp(-r[valid] * Tyr) * norm.cdf(dd2)
    p = K[valid] * np.exp(-r[valid] * Tyr) * norm.cdf(-dd2) - S[valid] * norm.cdf(-dd1)
    theo[valid] = c + p
matched['theoretical_value'] = theo
matched['qualifies'] = (matched['theoretical_value'] >=
                        matched['market_value'] * (1.0 + T_THRESHOLD))
matched.loc[~valid, 'qualifies'] = False
trades = matched[matched['qualifies']].copy()
print(f"Qualifying stock-days under locked T={T_THRESHOLD:.0%}: "
      f"{len(trades):,}")

# --------------------------------------------------------------------------
# Trades: entry identical to src/30 (fresh 30d straddle at entry date).
# Strike and entry price fixed at entry.
# --------------------------------------------------------------------------
straddle_u = (straddle.sort_values(['secid', 'tidx', 'date'])
              .drop_duplicates(['secid', 'tidx'], keep='last'))
sidx = straddle_u.set_index(['secid', 'tidx'])

trades['entry_tidx'] = trades['tidx'] + 1
trades['exit_tidx'] = trades['entry_tidx'] + HOLD_TDAYS
trades = trades[trades['exit_tidx'] < len(cal)].copy()

ent_mi = pd.MultiIndex.from_frame(
    trades[['secid', 'entry_tidx']].astype({'secid': 'int64'}))
trades['entry_price'] = sidx['market_value'].reindex(ent_mi).to_numpy()
trades['entry_strike'] = sidx['strike'].reindex(ent_mi).to_numpy()
trades['entry_date'] = cal[trades['entry_tidx']]
trades['exit_date'] = cal[trades['exit_tidx']]

# exit-side lookups: same-secid surface IVs at the exit trading day
ext_mi = pd.MultiIndex.from_frame(
    trades[['secid', 'exit_tidx']].astype({'secid': 'int64'}))
trades['iv30_exit'] = sidx['iv30'].reindex(ext_mi).to_numpy()
trades['iv10_exit'] = sidx['iv10'].reindex(ext_mi).to_numpy()

# exit-date spot and rate
px_idx = px.set_index(['PERMNO', 'DlyCalDt'])['DlyClose']
exit_mi = pd.MultiIndex.from_arrays(
    [trades['PERMNO'].to_numpy(), trades['exit_date'].to_numpy()])
trades['spot_exit'] = px_idx.reindex(exit_mi).to_numpy()
fac_idx = fac.set_index('date')['r_annual']
trades['r_exit'] = fac_idx.reindex(trades['exit_date']).to_numpy()

# remaining tenor: 30 calendar days bought at entry, minus days elapsed
elapsed = (trades['exit_date'] - trades['entry_date']).dt.days
trades['rem_days'] = (TENOR - elapsed).clip(lower=1)

n0 = len(trades)
trades = trades[trades['entry_price'].notna() & (trades['entry_price'] > 0) &
                trades['entry_strike'].notna() &
                trades['spot_exit'].notna() & (trades['spot_exit'] > 0) &
                trades['iv30_exit'].notna()].copy()
print(f"\nTrades dropped for missing entry/exit inputs "
      f"(incl. exits past {DEV_END} or no exit-date surface): "
      f"{n0 - len(trades):,} of {n0:,}")

# sigma at exit: prefer 10d/30d interpolation, else sqrt-time proxy
has10 = trades['iv10_exit'].notna()
w = (trades['rem_days'].clip(10, 30) - 10) / 20.0
sig_interp = trades['iv10_exit'] + (trades['iv30_exit'] -
                                    trades['iv10_exit']) * w
sig_proxy = trades['iv30_exit'] * np.sqrt(trades['rem_days'] / TENOR)
trades['sigma_exit'] = np.where(has10, sig_interp, sig_proxy)
print(f"Exit sigma source: 10d/30d interpolation {int(has10.sum()):,} "
      f"({has10.mean()*100:.1f}%), sqrt-time proxy "
      f"{int((~has10).sum()):,} ({(~has10).mean()*100:.1f}%)")
print(f"Remaining tenor at exit: min={trades['rem_days'].min()}, "
      f"max={trades['rem_days'].max()}, "
      f"mean={trades['rem_days'].mean():.1f} calendar days")

# exit value: BS at ORIGINAL strike, exit spot/rate, remaining tenor
Se = trades['spot_exit'].to_numpy()
Ke = trades['entry_strike'].to_numpy()
re_ = trades['r_exit'].fillna(0.0).to_numpy()
se = trades['sigma_exit'].to_numpy()
Te = trades['rem_days'].to_numpy() / 365.0
with np.errstate(all='ignore'):
    d1e = (np.log(Se / Ke) + (re_ + 0.5 * se ** 2) * Te) / (se * np.sqrt(Te))
    d2e = d1e - se * np.sqrt(Te)
    ce = Se * norm.cdf(d1e) - Ke * np.exp(-re_ * Te) * norm.cdf(d2e)
    pe = Ke * np.exp(-re_ * Te) * norm.cdf(-d2e) - Se * norm.cdf(-d1e)
trades['exit_value'] = ce + pe
trades = trades[np.isfinite(trades['exit_value'])].copy()

trades['net_ret'] = trades['exit_value'] / trades['entry_price'] - 1.0

# old (flawed) exit for the before/after comparison: fresh 30d straddle
trades['exit_value_flawed'] = sidx['market_value'].reindex(
    pd.MultiIndex.from_frame(
        trades[['secid', 'exit_tidx']].astype({'secid': 'int64'}))).to_numpy()
flawed_ret = trades['exit_value_flawed'] / trades['entry_price'] - 1.0

print("\n" + "=" * 78)
print("BEFORE / AFTER (mean per-trade net return)")
print("=" * 78)
print(f"  OLD (flawed, fresh-30d exit):     {flawed_ret.mean()*100:+7.2f}% "
      f"per trade")
print(f"  CORRECTED (held option at exit):  "
      f"{trades['net_ret'].mean()*100:+7.2f}% per trade")
print(f"  difference = the omitted theta decay + strike drift, "
      f"{(trades['net_ret'].mean() - flawed_ret.mean())*100:+.2f} pp")

print(f"\nFinal trade count: {len(trades):,}; unique PERMNOs: "
      f"{trades['PERMNO'].nunique():,}")
trades['year'] = pd.to_datetime(trades['entry_date']).year if False else \
    pd.to_datetime(pd.Series(trades['entry_date'])).dt.year.to_numpy()
by_year = pd.Series(trades['year']).value_counts().sort_index()
print("Trades by year:")
for y, n in by_year.items():
    print(f"  {y}: {n:,}")

# --------------------------------------------------------------------------
# Daily portfolio series - identical construction to src/30
# --------------------------------------------------------------------------
rows_out = []
for tr in trades.itertuples(index=False):
    n = HOLD_TDAYS
    per_day_ret = (1.0 + tr.net_ret) ** (1.0 / n) - 1.0
    for k in range(1, n + 1):
        rows_out.append((tr.entry_tidx + k, per_day_ret))
daily = pd.DataFrame(rows_out, columns=['tidx', 'ret'])
port = daily.groupby('tidx').agg(port_ret_net=('ret', 'mean'),
                                 n_open_positions=('ret', 'size'))
entries = pd.Series(trades['entry_tidx']).value_counts().rename('n_new_entries')
port = port.join(entries, how='left')
port['n_new_entries'] = port['n_new_entries'].fillna(0).astype(int)
port = port.reset_index()
port['date'] = cal[port['tidx'].clip(0, len(cal) - 1)]
port = port[(port['date'] >= DEV_START) & (port['date'] <= DEV_END)]
port = port[['date', 'port_ret_net', 'n_open_positions', 'n_new_entries']]
port.to_parquet(out_path, index=False)
print(f"\nDaily series: {len(port):,} days, "
      f"{port['date'].min().date()} to {port['date'].max().date()}")
print(f"[OK] Saved {out_path}")

print("\n" + "=" * 78)
print("K1 CORRECTED CONSTRUCTION COMPLETE - no gate decision made here.")
print("=" * 78)
