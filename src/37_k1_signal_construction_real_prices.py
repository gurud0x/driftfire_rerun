import pandas as pd
import numpy as np
from scipy.stats import norm
from pathlib import Path

# ---------------------------------------------------------------------------
# K1 rebuild, Step 2: reconstruct K1 trades using REAL quoted contract
# prices from opprcd, eliminating the surface-reconstruction problem.
#
# WHY: src/30 priced exits as a FRESH 30-day straddle (no theta -> +185%/yr).
# src/32 corrected with a sqrt-time IV haircut on 94% of exits, which
# double-counts decay (BS already scales with T) -> -490%/yr. Both were
# model reconstructions of a price that was never observed. opprcd gives
# the actual quoted bid/ask of the actual held contract on both dates, so
# no reconstruction is needed at all.
#
# ENTRY/EXIT CONTRACT SELECTION (disclosed):
#  - The entry rule (T=25%, TRAIL20 vs surface premium) is UNCHANGED and
#    re-derived exactly as in src/32 - the locked Section 6 rule decides
#    WHICH stock-days trade. opprcd only changes how the trade is PRICED.
#  - Contract choice at entry: among contracts quoted for that secid on the
#    entry date having BOTH a call and a put at the same strike/exdate,
#    pick the exdate nearest to entry+30 calendar days (Section 5's tenor)
#    subject to exdate > exit date, then the strike nearest the entry-date
#    close (ATM, Section 5). Requiring exdate > exit date keeps Section 5's
#    10-trading-day hold feasible; trades whose only contracts expire
#    sooner are EXCLUDED and counted, never re-dated.
#  - Exit: the SAME contracts, verified by optionid, quoted on the exit
#    date. No modelling, no interpolation, no theta assumption.
#
# TWO PRICING SCENARIOS, both reported:
#  - MID   = (best_bid + best_offer)/2 on both legs, both dates. Matches
#            Section 5's literal "mid price".
#  - BID   = buy at best_offer on entry, sell at best_bid on exit. The
#            worst realistic fill. No spread ASSUMPTION is needed now -
#            these are the real quotes, so this finally implements what
#            Section 7 intended before we learned no bid/ask had been pulled.
# ---------------------------------------------------------------------------

DEV_START = '2015-01-01'
DEV_END = '2021-12-31'
TENOR = 30
HOLD_TDAYS = 10
DATE_TOL_TDAYS = 3
T_THRESHOLD = 0.25
SURF_CHUNK = 5_000_000
OPP_CHUNK = 5_000_000

project_root = Path(__file__).parent.parent
repo_om = project_root / 'data' / 'raw' / 'optionmetrics'
staging_om = (Path.home() / 'Downloads' / 'quantdata' / 'driftfire' / 'raw' /
              'optionmetrics')
surf_path = repo_om / 'vol_surface.csv'
om_path = repo_om / 'om_security_names.csv'
v1_path = project_root / 'data' / 'processed' / 'compression_signal_v1.parquet'
univ_path = project_root / 'data' / 'processed' / 'universe_membership.parquet'
names_path = project_root / 'data' / 'raw' / 'crsp' / 'crsp_names.parquet'
crsp_path = project_root / 'data' / 'processed' / 'crsp_combined.parquet'
fac_path = project_root / 'data' / 'processed' / 'factors_daily.parquet'
out_path = (project_root / 'data' / 'processed' /
            'k1_portfolio_returns_real_prices.parquet')

opp_path = None
for c in [repo_om / 'opprcd.csv', staging_om / 'opprcd.csv']:
    if c.exists():
        opp_path = c
        break
if opp_path is None:
    print("STOP: opprcd.csv not found.")
    raise SystemExit(1)

print("=" * 78)
print("K1 REBUILD ON REAL CONTRACT PRICES (opprcd)")
print("=" * 78)
print(f"\nopprcd source: {opp_path} ({opp_path.stat().st_size/1e9:.1f} GB)")

# --------------------------------------------------------------------------
# PART 1: re-derive the locked trade list exactly as src/32 does
# --------------------------------------------------------------------------
print("\n" + "-" * 78)
print("Re-deriving the locked T=25% trade list (unchanged Section 6 rule)")
print("-" * 78)

peek = pd.read_csv(surf_path, nrows=5)
usecols = [c for c in peek.columns if c.lower() in
           ('secid', 'date', 'days', 'delta', 'impl_strike',
            'impl_premium', 'cp_flag')]
SDT = {'secid': 'int32', 'date': 'str', 'days': 'int16', 'delta': 'float32',
       'impl_strike': 'float64', 'impl_premium': 'float64',
       'cp_flag': 'category'}
parts = []
for i, ch in enumerate(pd.read_csv(surf_path, usecols=usecols, dtype=SDT,
                                   chunksize=SURF_CHUNK), 1):
    ch.columns = [c.lower() for c in ch.columns]
    sel = ch[ch['days'] == TENOR].copy()
    if len(sel):
        sel['dpen'] = (sel['delta'].abs() / 100.0 - 0.50).abs()
        sel = (sel.sort_values(['secid', 'date', 'cp_flag', 'dpen'])
               .drop_duplicates(['secid', 'date', 'cp_flag'], keep='first'))
        parts.append(sel[['secid', 'date', 'cp_flag', 'impl_strike',
                          'impl_premium', 'dpen']])
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
         [['secid', 'tidx', 'surf_date', 'market_value', 'strike']])
joined = pd.merge_asof(left, right, on='tidx', by='secid',
                       direction='nearest', tolerance=DATE_TOL_TDAYS)
joined['hit'] = joined['market_value'].notna()
per_day = (joined.sort_values('hit', ascending=False)
           .drop_duplicates(['PERMNO', 'DlyCalDt'], keep='first'))
matched = per_day[per_day['hit']].copy()

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
    c_ = S[valid] * norm.cdf(dd1) - K[valid] * np.exp(-r[valid] * Tyr) * norm.cdf(dd2)
    p_ = K[valid] * np.exp(-r[valid] * Tyr) * norm.cdf(-dd2) - S[valid] * norm.cdf(-dd1)
    theo[valid] = c_ + p_
matched['theoretical_value'] = theo
matched['qualifies'] = (matched['theoretical_value'] >=
                        matched['market_value'] * (1.0 + T_THRESHOLD))
matched.loc[~valid, 'qualifies'] = False
trades = matched[matched['qualifies']].copy()
trades['entry_tidx'] = trades['tidx'] + 1
trades['exit_tidx'] = trades['entry_tidx'] + HOLD_TDAYS
trades = trades[trades['exit_tidx'] < len(cal)].copy()
trades['entry_date'] = cal[trades['entry_tidx']]
trades['exit_date'] = cal[trades['exit_tidx']]
trades = trades.reset_index(drop=True)
trades['trade_id'] = np.arange(len(trades))
trades['secid'] = trades['secid'].astype('int64')

# ATM reference: the entry-date close (contract is chosen at entry)
spot_idx = px.set_index(['PERMNO', 'DlyCalDt'])['DlyClose']
trades['spot_entry'] = spot_idx.reindex(
    pd.MultiIndex.from_arrays([trades['PERMNO'].to_numpy(),
                               trades['entry_date'].to_numpy()])).to_numpy()
trades['target_exp'] = trades['entry_date'] + pd.Timedelta(days=TENOR)
n0 = len(trades)
trades = trades[trades['spot_entry'].notna() & (trades['spot_entry'] > 0)].copy()
print(f"\nLocked trade list re-derived: {len(trades):,} trades "
      f"({n0 - len(trades):,} dropped for missing entry-date close)")
print(f"  (src/30 built 5,351 and src/32 5,277 from this same rule; small")
print(f"   differences come only from each script's own input requirements)")

# --------------------------------------------------------------------------
# PART 2: single chunked pass over opprcd, keeping only needed (secid, date)
# --------------------------------------------------------------------------
secid_set = set(trades['secid'].unique().tolist())
entry_str = trades['entry_date'].dt.strftime('%Y-%m-%d')
exit_str = trades['exit_date'].dt.strftime('%Y-%m-%d')
date_set = set(entry_str) | set(exit_str)
print(f"\nScanning opprcd for {len(secid_set):,} secids x "
      f"{len(date_set):,} relevant dates...")

ONEED = ['secid', 'date', 'exdate', 'cp_flag', 'strike_price', 'best_bid',
         'best_offer', 'volume', 'open_interest', 'optionid']
ODT = {'secid': 'int32', 'date': 'str', 'exdate': 'str',
       'cp_flag': 'category', 'strike_price': 'float64',
       'best_bid': 'float64', 'best_offer': 'float64', 'volume': 'float64',
       'open_interest': 'float64', 'optionid': 'int64'}
keep = []
scanned = 0
# Whole-file liquidity accumulators, taken BEFORE any filtering so they
# describe the full opprcd population and cross-check src/36's validation
# (expected: 49.78% both-zero, 23.07% zero-bid). Added for the persisted
# summary artifact; they do not touch any K1 computation.
wf_zero_liq = 0
wf_no_bid = 0
for i, ch in enumerate(pd.read_csv(opp_path, usecols=ONEED, dtype=ODT,
                                   chunksize=OPP_CHUNK), 1):
    scanned += len(ch)
    wf_zero_liq += int(((ch['volume'].fillna(0) == 0) &
                        (ch['open_interest'].fillna(0) == 0)).sum())
    wf_no_bid += int((ch['best_bid'].fillna(-1) == 0).sum())
    ch = ch[ch['secid'].isin(secid_set)]
    if len(ch):
        ch = ch[ch['date'].isin(date_set)]
        if len(ch):
            keep.append(ch)
    if i % 20 == 0 or i == 1:
        kept = sum(len(k) for k in keep)
        print(f"  chunk {i:>4}: {scanned:>15,} scanned, {kept:>12,} kept")

opp = pd.concat(keep, ignore_index=True)
del keep
opp['strike'] = opp['strike_price'] / 1000.0     # OM x1000 convention
opp['date_d'] = pd.to_datetime(opp['date'], cache=True)
opp['exdate_d'] = pd.to_datetime(opp['exdate'], cache=True)
print(f"\nScanned {scanned:,} opprcd rows; retained {len(opp):,} "
      f"on relevant secid-dates")

# --------------------------------------------------------------------------
# PART 3: entry contract selection (both legs, same strike + exdate)
# --------------------------------------------------------------------------
print("\n" + "-" * 78)
print("Matching entry contracts (nearest exdate to entry+30d that outlives")
print("the exit date; then strike nearest the entry-date close)")
print("-" * 78)

ent_universe = trades[['trade_id', 'secid', 'entry_date', 'exit_date',
                       'target_exp', 'spot_entry']].merge(
    opp, left_on=['secid', 'entry_date'], right_on=['secid', 'date_d'],
    how='inner')
print(f"Trade x candidate-contract rows on entry dates: {len(ent_universe):,}")

# contract must outlive the exit date so Section 5's 10-day hold is feasible
ent_universe = ent_universe[ent_universe['exdate_d'] > ent_universe['exit_date']]

ec = (ent_universe[ent_universe['cp_flag'] == 'C']
      [['trade_id', 'exdate_d', 'strike', 'best_bid', 'best_offer',
        'volume', 'open_interest', 'optionid', 'target_exp', 'spot_entry']]
      .rename(columns={'best_bid': 'bid_c_e', 'best_offer': 'off_c_e',
                       'volume': 'vol_c_e', 'open_interest': 'oi_c_e',
                       'optionid': 'oid_c'}))
ep = (ent_universe[ent_universe['cp_flag'] == 'P']
      [['trade_id', 'exdate_d', 'strike', 'best_bid', 'best_offer',
        'volume', 'open_interest', 'optionid']]
      .rename(columns={'best_bid': 'bid_p_e', 'best_offer': 'off_p_e',
                       'volume': 'vol_p_e', 'open_interest': 'oi_p_e',
                       'optionid': 'oid_p'}))
pair = ec.merge(ep, on=['trade_id', 'exdate_d', 'strike'], how='inner')
print(f"Trade x (exdate, strike) pairs with BOTH legs quoted: {len(pair):,}")

pair['exp_pen'] = (pair['exdate_d'] - pair['target_exp']).dt.days.abs()
pair['atm_pen'] = (pair['strike'] - pair['spot_entry']).abs()
pair = pair.sort_values(['trade_id', 'exp_pen', 'atm_pen'])
entry_pick = pair.drop_duplicates('trade_id', keep='first')
print(f"Trades with a usable entry contract: {len(entry_pick):,} of "
      f"{len(trades):,}")

# --------------------------------------------------------------------------
# PART 4: exit quotes for the SAME optionids (identity-verified)
# --------------------------------------------------------------------------
exit_q = opp[['optionid', 'date_d', 'best_bid', 'best_offer', 'volume',
              'open_interest']].drop_duplicates(['optionid', 'date_d'])
exit_q = exit_q.set_index(['optionid', 'date_d'])

t = trades[['trade_id', 'PERMNO', 'secid', 'decile', 'entry_date',
            'exit_date', 'entry_tidx']].merge(entry_pick, on='trade_id',
                                              how='inner')
mi_c = pd.MultiIndex.from_arrays([t['oid_c'].to_numpy(),
                                  t['exit_date'].to_numpy()])
mi_p = pd.MultiIndex.from_arrays([t['oid_p'].to_numpy(),
                                  t['exit_date'].to_numpy()])
for tag, mi in (('c', mi_c), ('p', mi_p)):
    sub = exit_q.reindex(mi)
    t[f'bid_{tag}_x'] = sub['best_bid'].to_numpy()
    t[f'off_{tag}_x'] = sub['best_offer'].to_numpy()
    t[f'vol_{tag}_x'] = sub['volume'].to_numpy()
    t[f'oi_{tag}_x'] = sub['open_interest'].to_numpy()

has_exit = t['bid_c_x'].notna() & t['bid_p_x'].notna()
print(f"\nOf those, trades ALSO quoted on the exit date (same optionid): "
      f"{int(has_exit.sum()):,}")
t = t[has_exit].copy()

# --------------------------------------------------------------------------
# PART 5: pricing - MID and BID scenarios
# --------------------------------------------------------------------------
t['entry_mid'] = (t['bid_c_e'] + t['off_c_e']) / 2.0 + \
                 (t['bid_p_e'] + t['off_p_e']) / 2.0
t['exit_mid'] = (t['bid_c_x'] + t['off_c_x']) / 2.0 + \
                (t['bid_p_x'] + t['off_p_x']) / 2.0
t['entry_ask'] = t['off_c_e'] + t['off_p_e']      # pay the offer
t['exit_bid'] = t['bid_c_x'] + t['bid_p_x']       # sell into the bid

n_before = len(t)
t = t[(t['entry_mid'] > 0) & (t['entry_ask'] > 0)].copy()
print(f"Trades dropped for non-positive entry price (no market): "
      f"{n_before - len(t):,}")

t['ret_mid'] = t['exit_mid'] / t['entry_mid'] - 1.0
t['ret_bid'] = t['exit_bid'] / t['entry_ask'] - 1.0
t['liq_ok'] = (((t['vol_c_e'].fillna(0) > 0) | (t['oi_c_e'].fillna(0) > 0)) &
               ((t['vol_p_e'].fillna(0) > 0) | (t['oi_p_e'].fillna(0) > 0)) &
               ((t['vol_c_x'].fillna(0) > 0) | (t['oi_c_x'].fillna(0) > 0)) &
               ((t['vol_p_x'].fillna(0) > 0) | (t['oi_p_x'].fillna(0) > 0)))

match_rate = len(t) / len(trades) * 100
print("\n" + "=" * 78)
print("MATCH RATE")
print("=" * 78)
print(f"  original locked trades:              {len(trades):>8,}")
print(f"  matched to real quotes (entry+exit): {len(t):>8,} "
      f"({match_rate:.1f}%)")
print(f"  excluded (no contract / no exit quote / no market): "
      f"{len(trades)-len(t):,} - counted, never filled or estimated")
print(f"  of matched, passing the liquidity filter "
      f"(vol>0 or OI>0 on all 4 quotes): {int(t['liq_ok'].sum()):,} "
      f"({t['liq_ok'].mean()*100:.1f}%)")

print("\n" + "=" * 78)
print("BEFORE / AFTER: mean per-trade return")
print("=" * 78)
print(f"  src/30 flawed reconstruction (fresh-30d exit): "
      f"{+9.96:>+8.2f}%  [no theta at all]")
print(f"  src/32 corrected reconstruction (sqrt-time IV): "
      f"{-11.74:>+8.2f}%  [double-counts theta]")
print(f"  REAL QUOTES, mid-to-mid:                       "
      f"{t['ret_mid'].mean()*100:>+8.2f}%  [observed]")
print(f"  REAL QUOTES, buy-ask/sell-bid:                 "
      f"{t['ret_bid'].mean()*100:>+8.2f}%  [observed, worst fill]")
print(f"  REAL QUOTES, mid-to-mid, liquidity-filtered:   "
      f"{t.loc[t['liq_ok'], 'ret_mid'].mean()*100:>+8.2f}%")
sp = ((t['entry_ask'] - t['entry_mid']) / t['entry_mid'] * 100)
print(f"\n  observed entry half-spread as % of mid: median "
      f"{sp.median():.1f}%, mean {sp.mean():.1f}%")
print(f"  (this is the cost Section 7 always intended to charge but could")
print(f"   not, because no bid/ask existed in the surface pull)")

print(f"\nMatched trades by year:")
t['year'] = pd.to_datetime(pd.Series(t['entry_date'])).dt.year.to_numpy()
for y, n in pd.Series(t['year']).value_counts().sort_index().items():
    print(f"  {y}: {n:,}")

# --------------------------------------------------------------------------
# PART 6: daily portfolio series (same construction as src/30 and src/32)
# --------------------------------------------------------------------------
def build_daily(df, ret_col):
    rows_out = []
    for tr in df.itertuples(index=False):
        rr = getattr(tr, ret_col)
        if not np.isfinite(rr) or rr <= -1.0:
            rr = max(rr, -0.999999) if np.isfinite(rr) else np.nan
            if not np.isfinite(rr):
                continue
        per_day = (1.0 + rr) ** (1.0 / HOLD_TDAYS) - 1.0
        for k in range(1, HOLD_TDAYS + 1):
            rows_out.append((tr.entry_tidx + k, per_day))
    d = pd.DataFrame(rows_out, columns=['tidx', 'ret'])
    g = d.groupby('tidx').agg(**{ret_col: ('ret', 'mean'),
                                 'n_' + ret_col: ('ret', 'size')})
    return g


g_mid = build_daily(t, 'ret_mid')
g_bid = build_daily(t, 'ret_bid')
g_liq = build_daily(t[t['liq_ok']], 'ret_mid').rename(
    columns={'ret_mid': 'ret_mid_liq', 'n_ret_mid': 'n_ret_mid_liq'})
port = g_mid.join(g_bid, how='outer').join(g_liq, how='outer').reset_index()
entries = pd.Series(t['entry_tidx']).value_counts().rename('n_new_entries')
port = port.join(entries, on='tidx')
port['n_new_entries'] = port['n_new_entries'].fillna(0).astype(int)
port['date'] = cal[port['tidx'].clip(0, len(cal) - 1)]
port = port[(port['date'] >= DEV_START) & (port['date'] <= DEV_END)]
port = port.rename(columns={'ret_mid': 'port_ret_mid',
                            'ret_bid': 'port_ret_bid',
                            'ret_mid_liq': 'port_ret_mid_liq',
                            'n_ret_mid': 'n_open_positions'})
port = port[['date', 'port_ret_mid', 'port_ret_bid', 'port_ret_mid_liq',
             'n_open_positions', 'n_new_entries']]
port.to_parquet(out_path, index=False)
print(f"\nDaily series: {len(port):,} days, "
      f"{port['date'].min().date()} to {port['date'].max().date()}")
print(f"  open positions per day: mean {port['n_open_positions'].mean():.1f}, "
      f"max {port['n_open_positions'].max()}")
print(f"[OK] Saved {out_path}")

# --------------------------------------------------------------------------
# PART 7: PERSIST the per-trade record and summary.
#
# Added after the fact so the paper's central K1 numbers are mechanically
# verifiable like every other reported figure. NOTHING above this point was
# changed: no computation, no filter, no threshold. This section only writes
# out values that the run already produced, so a re-run reproduces the logged
# gate result byte for byte.
#
# The per-trade parquet is the durable fix - with it, any future summary can
# be regenerated WITHOUT re-scanning 80 GB of opprcd.
# --------------------------------------------------------------------------
import json
from datetime import datetime

trade_cols = ['trade_id', 'PERMNO', 'secid', 'decile', 'entry_date',
              'exit_date', 'exdate_d', 'strike', 'oid_c', 'oid_p',
              'bid_c_e', 'off_c_e', 'bid_p_e', 'off_p_e',
              'bid_c_x', 'off_c_x', 'bid_p_x', 'off_p_x',
              'entry_mid', 'exit_mid', 'entry_ask', 'exit_bid',
              'ret_mid', 'ret_bid', 'liq_ok', 'year']
trades_out = project_root / 'data' / 'processed' / 'k1_trades_real_prices.parquet'
t[[c for c in trade_cols if c in t.columns]].to_parquet(trades_out,
                                                        index=False)
print(f"\n[OK] Saved per-trade record: {trades_out}  ({len(t):,} trades)")
print("     Future summaries regenerate from this file - no 80 GB re-scan.")

liq = t[t['liq_ok']]
summary = {
    'phase': 'K1 REAL PRICES (opprcd)',
    'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'script': 'src/37_k1_signal_construction_real_prices.py',
    'pre_registration_commit': '3875314',
    'window': f'DEV {DEV_START} to {DEV_END}',
    'note': ('Per-trade statistics supplementing the daily-series statistics '
             'in the K1 REAL PRICES gate_log entry. Prices are real quoted '
             'bid/ask of the held contracts, identity-verified by optionid.'),
    'trades': {
        'locked_trade_list': int(len(trades)),
        'matched_entry_and_exit': int(len(t)),
        'match_rate_pct': round(len(t) / len(trades) * 100, 2),
        'excluded': int(len(trades) - len(t)),
        'unique_permnos': int(t['PERMNO'].nunique()),
        'passing_liquidity_filter': int(t['liq_ok'].sum()),
        'passing_liquidity_filter_pct': round(t['liq_ok'].mean() * 100, 2),
    },
    'mean_per_trade_return_pct': {
        'mid_to_mid': round(float(t['ret_mid'].mean()) * 100, 4),
        'bid_based_worst_fill': round(float(t['ret_bid'].mean()) * 100, 4),
        'mid_liquidity_filtered': round(float(liq['ret_mid'].mean()) * 100, 4),
    },
    'entry_half_spread_pct_of_mid': {
        'median': round(float(sp.median()), 4),
        'mean': round(float(sp.mean()), 4),
    },
    'opprcd_whole_file_liquidity_pct': {
        'zero_volume_and_zero_open_interest': round(
            wf_zero_liq / scanned * 100, 2),
        'zero_bid': round(wf_no_bid / scanned * 100, 2),
        'rows_scanned': int(scanned),
        'cross_check': ('src/36 validation reported 49.78% and 23.07% over '
                        'the same file'),
    },
}
json_out = project_root / 'data' / 'processed' / 'k1_trade_summary.json'
json_out.write_text(json.dumps(summary, indent=2))
print(f"[OK] Saved summary artifact: {json_out}")

# Append-only block in gate_log.md. Never edits the existing K1 REAL PRICES
# entry; guarded so repeated runs cannot duplicate it.
MARKER = '## K1 REAL PRICES (opprcd) — per-trade statistics'
log_path = project_root / 'results' / 'gate_log.md'
log_text = log_path.read_text(encoding='utf-8', errors='replace')
if MARKER in log_text:
    print(f"[SKIP] gate_log.md already carries the per-trade block; "
          f"not appending a duplicate.")
else:
    m = summary['mean_per_trade_return_pct']
    s = summary['entry_half_spread_pct_of_mid']
    tr = summary['trades']
    wf = summary['opprcd_whole_file_liquidity_pct']
    block = [
        f"\n---\n\n{MARKER}",
        "",
        "Supplements the daily-series statistics logged in the K1 REAL "
        "PRICES entry above; that entry is unchanged. Same run, same "
        "trades, same pre-registration (commit 3875314). Machine-readable "
        "copy: data/processed/k1_trade_summary.json; per-trade detail: "
        "data/processed/k1_trades_real_prices.parquet.",
        "",
        "```",
        f"trades (locked list):                  {tr['locked_trade_list']:,}",
        f"matched to real quotes (entry+exit):   {tr['matched_entry_and_exit']:,}"
        f"  ({tr['match_rate_pct']:.2f}%)",
        f"excluded (no contract/quote/market):   {tr['excluded']:,}",
        f"passing liquidity filter:              "
        f"{tr['passing_liquidity_filter']:,} "
        f"({tr['passing_liquidity_filter_pct']:.2f}%)",
        "",
        f"mean per-trade return, mid-to-mid:            "
        f"{m['mid_to_mid']:+.2f}%",
        f"mean per-trade return, bid-based worst fill:  "
        f"{m['bid_based_worst_fill']:+.2f}%",
        f"mean per-trade return, mid, liquidity-filtered: "
        f"{m['mid_liquidity_filtered']:+.2f}%",
        "",
        f"entry-side half-spread, median:  {s['median']:.2f}% of mid",
        f"entry-side half-spread, mean:    {s['mean']:.2f}% of mid",
        f"  (a round trip pays this twice - on entry and again on exit)",
        "",
        f"opprcd whole-file liquidity ({wf['rows_scanned']:,} contract-days):",
        f"  zero volume AND zero open interest: "
        f"{wf['zero_volume_and_zero_open_interest']:.2f}%",
        f"  zero bid:                           {wf['zero_bid']:.2f}%",
        "```",
    ]
    with open(log_path, 'a', encoding='utf-8') as fh:
        fh.write("\n".join(block) + "\n")
    print(f"[OK] Appended per-trade block to {log_path} "
          f"(existing entries untouched)")

print("\n" + "=" * 78)
print("K1 REAL-PRICE CONSTRUCTION COMPLETE - no gate decision made here.")
print("=" * 78)
