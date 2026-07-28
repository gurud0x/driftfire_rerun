import pandas as pd
import numpy as np
from scipy.stats import norm
from pathlib import Path

# ---------------------------------------------------------------------------
# Phase K1, Part B: trade construction under the LOCKED Section 6 entry rule,
# per docs/PhaseK1_PreRegistration_StraddleOnCompression.md (commit 3875314).
#
# Matched-dataset construction is REUSED verbatim from
# src/29_k1_threshold_calibration.py (chunked surface scan -> best near-ATM
# point per secid-date-side -> call/put straddle merge -> CUSIP bridge ->
# TRAIL20 + spot + RF -> merge_asof within +/- 3 trading days). Nothing about
# the join is recomputed differently here.
#
# COST NOTE (pre-registration Section 7, disclosed before commit): the pulled
# OptionMetrics files contain NO bid/ask. impl_premium (OM's fitted price per
# leg) is used as the transacted price for both entry and exit, with no
# explicit spread cost applied. Any positive result here is therefore an
# upper bound, not a conservative estimate.
#
# Dev window only in this script; the gate check (src/31) enforces the
# holdout boundary independently.
# ---------------------------------------------------------------------------

DEV_START = '2015-01-01'
DEV_END = '2021-12-31'
TENOR = 30                 # calendar days, Section 5
HOLD_TDAYS = 10            # trading days, Section 5
DATE_TOL_TDAYS = 3
T_THRESHOLD = 0.25         # LOCKED, Section 6
CHUNKSIZE = 5_000_000

project_root = Path(__file__).parent.parent
surf_path = project_root / 'data' / 'raw' / 'optionmetrics' / 'vol_surface.csv'
om_path = project_root / 'data' / 'raw' / 'optionmetrics' / 'om_security_names.csv'
v1_path = project_root / 'data' / 'processed' / 'compression_signal_v1.parquet'
univ_path = project_root / 'data' / 'processed' / 'universe_membership.parquet'
names_path = project_root / 'data' / 'raw' / 'crsp' / 'crsp_names.parquet'
crsp_path = project_root / 'data' / 'processed' / 'crsp_combined.parquet'
fac_path = project_root / 'data' / 'processed' / 'factors_daily.parquet'
out_path = project_root / 'data' / 'processed' / 'k1_portfolio_returns_daily.parquet'

print("=" * 78)
print("K1 SIGNAL CONSTRUCTION - locked T=25% entry rule, dev window")
print("=" * 78)
print("\nCost model: impl_premium used as transacted price on both legs,")
print("entry and exit. No bid/ask exists in the pulled data; no spread")
print("cost applied (pre-registration Section 7, disclosed before commit).")

# --------------------------------------------------------------------------
# Surface -> per-leg premiums at the 30d near-ATM point (src/29 logic)
# --------------------------------------------------------------------------
peek = pd.read_csv(surf_path, nrows=5)
usecols = [c for c in peek.columns if c.lower() in
           ('secid', 'date', 'days', 'delta', 'impl_strike',
            'impl_premium', 'cp_flag')]
DT = {'secid': 'int32', 'date': 'str', 'days': 'int16', 'delta': 'float32',
      'impl_strike': 'float64', 'impl_premium': 'float64',
      'cp_flag': 'category'}

print(f"\nScanning surface in chunks of {CHUNKSIZE:,} (days=={TENOR})...")
parts = []
rows = 0
for i, ch in enumerate(pd.read_csv(surf_path, usecols=usecols, dtype=DT,
                                   chunksize=CHUNKSIZE), 1):
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
print(f"\nStraddle-priceable secid-dates (both legs): {len(straddle):,}")

# --------------------------------------------------------------------------
# Bridge + decile-1 dev stock-days + TRAIL20/spot/RF (src/29 logic)
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
d1['year_month'] = d1['DlyCalDt'].dt.to_period('M').astype(str)
umem = univ[univ['in_universe']][['PERMNO', 'year_month', 'decile']]
d1 = d1.merge(umem, on=['PERMNO', 'year_month'], how='left')

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
         [['secid', 'tidx', 'surf_date', 'market_value', 'strike',
           'prem_call', 'prem_put']])
joined = pd.merge_asof(left, right, on='tidx', by='secid',
                       direction='nearest', tolerance=DATE_TOL_TDAYS)
joined['hit'] = joined['market_value'].notna()
per_day = (joined.sort_values('hit', ascending=False)
           .drop_duplicates(['PERMNO', 'DlyCalDt'], keep='first'))
matched = per_day[per_day['hit']].copy()
print(f"Dev decile-1 stock-days matched to a straddle: {len(matched):,}")

# --------------------------------------------------------------------------
# Black-Scholes theoretical value + locked entry rule (src/29 implementation)
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
# Trade construction: entry next trading day, exit at min(entry+10 tdays,
# expiration). Entry/exit prices are the same-secid straddle premium on the
# respective dates, from the surface (impl_premium, per Section 7 note).
# --------------------------------------------------------------------------
print(f"\nBuilding trades: entry next trading day, exit at "
      f"min(entry+{HOLD_TDAYS} tdays, expiration)...")

# two calendar quote dates can map to the same trading-day tidx (weekends/
# holidays); keep the later quote for that trading day, making the index
# unique before lookups
straddle_u = (straddle.sort_values(['secid', 'tidx', 'date'])
              .drop_duplicates(['secid', 'tidx'], keep='last'))
straddle_idx = straddle_u.set_index(['secid', 'tidx'])['market_value']

# signal at tidx -> entry at tidx+1 (next trading day, Section 5)
trades['entry_tidx'] = trades['tidx'] + 1
# expiration = entry + 30 calendar days; the option written at the signal
# date's 30d tenor expires 30 calendar days after that quote date
trades['expiry_date'] = trades['surf_date'] + pd.Timedelta(days=TENOR)
# expiration in trading-day index terms (nearest trading day at/below expiry)
exp_pos = cal.searchsorted(trades['expiry_date'].to_numpy(), side='right') - 1
trades['expiry_tidx'] = np.clip(exp_pos, 0, len(cal) - 1)
trades['exit_tidx'] = np.minimum(trades['entry_tidx'] + HOLD_TDAYS,
                                 trades['expiry_tidx'])

ent = trades[['secid', 'entry_tidx']].rename(columns={'entry_tidx': 'tidx'})
ext = trades[['secid', 'exit_tidx']].rename(columns={'exit_tidx': 'tidx'})
trades['entry_price'] = straddle_idx.reindex(
    pd.MultiIndex.from_frame(ent)).to_numpy()
trades['exit_price'] = straddle_idx.reindex(
    pd.MultiIndex.from_frame(ext)).to_numpy()

n_before = len(trades)
trades = trades[trades['entry_price'].notna() & trades['exit_price'].notna() &
                (trades['entry_price'] > 0) &
                (trades['exit_tidx'] > trades['entry_tidx'])].copy()
print(f"  trades dropped for missing entry/exit quote or non-positive hold: "
      f"{n_before - len(trades):,} of {n_before:,}")

# net return per trade: no explicit spread (see cost note); long straddle
trades['net_ret'] = trades['exit_price'] / trades['entry_price'] - 1.0
trades['hold_tdays'] = trades['exit_tidx'] - trades['entry_tidx']
trades['entry_date'] = cal[trades['entry_tidx'].clip(0, len(cal) - 1)]
trades['exit_date'] = cal[trades['exit_tidx'].clip(0, len(cal) - 1)]

print(f"\nFinal trade count: {len(trades):,}")
print(f"Unique PERMNOs: {trades['PERMNO'].nunique():,}")

print(f"\nTrades by year (compare to src/29's qualifying counts):")
trades['year'] = trades['entry_date'].year if hasattr(trades['entry_date'], 'year') \
    else pd.to_datetime(trades['entry_date']).dt.year
trades['year'] = pd.to_datetime(trades['entry_date']).dt.year
by_year = trades.groupby('year').size()
for y, n in by_year.items():
    print(f"  {y}: {n:,}")
y2020 = int(by_year.get(2020, 0))
print(f"  2020 share: {y2020:,} of {len(trades):,} "
      f"({y2020/len(trades)*100:.1f}%)  [src/29 count-check found 44.1%]")

print(f"\nRealized holding period (trading days), distribution:")
hp = trades['hold_tdays'].value_counts().sort_index()
for h, n in hp.items():
    print(f"  {h:>3} tdays: {n:>7,} ({n/len(trades)*100:5.1f}%)")
print(f"  mean realized hold: {trades['hold_tdays'].mean():.2f} tdays "
      f"(max possible {HOLD_TDAYS}; shorter = exited at expiration)")
n_early = int((trades['hold_tdays'] < HOLD_TDAYS).sum())
print(f"  exited early at expiration: {n_early:,} "
      f"({n_early/len(trades)*100:.1f}%)")

# --------------------------------------------------------------------------
# Daily portfolio return: equal-weighted across positions open that day.
# A position open from entry_tidx to exit_tidx contributes its per-day
# return on each held day.
# --------------------------------------------------------------------------
print(f"\nBuilding daily portfolio return series...")
rows_out = []
for tr in trades.itertuples(index=False):
    n = tr.hold_tdays
    if n <= 0:
        continue
    # per-day geometric decomposition of the trade's total net return
    per_day_ret = (1.0 + tr.net_ret) ** (1.0 / n) - 1.0
    for k in range(1, n + 1):
        rows_out.append((tr.entry_tidx + k, per_day_ret, tr.entry_tidx + 1))
daily = pd.DataFrame(rows_out, columns=['tidx', 'ret', 'first_tidx'])
port = daily.groupby('tidx').agg(port_ret_net=('ret', 'mean'),
                                 n_open_positions=('ret', 'size'))
entries = trades.groupby('entry_tidx').size().rename('n_new_entries')
port = port.join(entries, how='left')
port['n_new_entries'] = port['n_new_entries'].fillna(0).astype(int)
port = port.reset_index()
port['date'] = cal[port['tidx'].clip(0, len(cal) - 1)]
port = port[(port['date'] >= DEV_START) & (port['date'] <= DEV_END)]
port = port[['date', 'port_ret_net', 'n_open_positions', 'n_new_entries']]
port.to_parquet(out_path, index=False)

print(f"\nDaily series: {len(port):,} trading days with >=1 open position")
print(f"  date range: {port['date'].min().date()} to {port['date'].max().date()}")
print(f"  open positions per day: min={port['n_open_positions'].min()}, "
      f"max={port['n_open_positions'].max()}, "
      f"mean={port['n_open_positions'].mean():.1f}")
print(f"[OK] Saved {out_path}")

print("\n" + "=" * 78)
print("K1 SIGNAL CONSTRUCTION COMPLETE - no gate decision made here.")
print("=" * 78)
