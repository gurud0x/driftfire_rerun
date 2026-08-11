import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

# ---------------------------------------------------------------------------
# V4 FINAL DESIGN RECONCILIATION - RETURN-BLIND.
#
# Covers audit items 1-5 of the final reconciliation:
#   1. entry DTE band comparison ([25,38] vs [40,60] vs [45,60]) under a
#      FIXED 30-CALENDAR-DAY EXIT (not hold-to-expiry)
#   2. capacity across five NAV levels
#   3. spread ceiling 10% vs 15%
#   4. dynamic vs fixed liquidity cap
#   5. quote survival (full hold) and pre-entry quote continuity
#
# NOTHING IN THIS SCRIPT COMPUTES A RETURN, P&L, FORWARD REALIZED VARIANCE,
# A STRATEGY SCORE USING FUTURE OUTCOMES, OR A TEST STATISTIC. DlyRet is
# never loaded. The only forward-looking quantities measured are QUOTE
# EXISTENCE and CALENDAR/DTE ARITHMETIC across the holding path - i.e.
# "does a tradeable quote exist on this date", never "what was it worth".
# Entry premium (a cost at entry) and capital utilization are computed;
# no exit price is ever differenced against an entry price.
#
# gate_log.md is NOT touched.
# ---------------------------------------------------------------------------

DEV_START = pd.Timestamp('2015-01-01')
DEV_END = pd.Timestamp('2021-12-31')
HOLD_CAL_DAYS = 30                      # FIXED 30-calendar-day exit

# Three candidate entry bands. target = tie-break anchor (Section 7.1 rule 2).
BANDS = [
    ('B25_38', 25, 38, 30),   # currently locked band; target 30 is the locked value
    ('B40_60', 40, 60, 50),   # midpoint anchor
    ('B45_60', 45, 60, 52),   # midpoint anchor
]
SCAN_DTE_LO, SCAN_DTE_HI = 25, 60       # PASS A entry-candidate scan range

DELTA_LO, DELTA_HI = 0.40, 0.60
RISK_FREE_RATE = 0.01

IV_SANITY_LO, IV_SANITY_HI = 0.05, 2.00
MIN_MID_PRICE = 0.50
MIN_BID_PRICE = 0.20
SPREAD_LOOSE = 0.15
SPREAD_TIGHT = 0.10
MIN_OI = 100
MIN_SAME_DAY_VOL = 10
MIN_5D_VOL = 50

# item 5 locked missing-quote rule parameters
PRE_ENTRY_SESSIONS = 20
PRE_ENTRY_MIN_VALID_FRAC = 0.90
PRE_ENTRY_MAX_CONSEC_MISSING = 1
MID_HOLD_MAX_CARRY_DAYS = 1             # carry last valid delta at most 1 day
MID_HOLD_EXIT_AFTER_CONSEC = 2          # exit after 2 consecutive missing days

# item 2 capacity
NAV_LEVELS = [100_000, 250_000, 500_000, 1_000_000, 2_000_000]
MAX_POSITIONS = 20
MIN_BREADTH = 5
VEGA_CAP_FRAC = 0.005                   # aggregate $ vega per vol point / NAV
GAMMA_CAP_FRAC = 0.05
SECTOR_CAP_N = 4
PREMIUM_SKIP_MULT = 2.0

# item 4 dynamic capacity rule
DYN_VOL_FRAC = 0.05                     # 5% of trailing 20d AVERAGE contract volume
DYN_OI_FRAC = 0.01                      # 1% of open interest
# existing fixed capacity cap, for comparison
FIX_OI_FRAC = 0.10
FIX_VOL5SUM_FRAC = 0.20

OPPRCD_CHUNKSIZE = 5_000_000
_MAX_CHUNKS_TEST = int(os.environ.get("V55_MAX_CHUNKS_TEST", "0"))

project_root = Path(__file__).parent.parent
crsp_path = project_root / 'data' / 'processed' / 'crsp_combined.parquet'
univ_path = project_root / 'data' / 'processed' / 'universe_membership.parquet'
v1_path = project_root / 'data' / 'processed' / 'compression_signal_v1.parquet'
om_names_path = project_root / 'data' / 'raw' / 'optionmetrics' / 'om_security_names.csv'
crsp_names_path = project_root / 'data' / 'raw' / 'crsp' / 'crsp_names.parquet'
link_path = project_root / 'data' / 'raw' / 'compustat' / 'ccm_link_gics.csv'
rdq_path = project_root / 'data' / 'raw' / 'compustat' / 'rdq_pull_fundq_2014_2026.parquet'
out_json = project_root / 'results' / '55_v4_design_reconciliation.json'

repo_om = project_root / 'data' / 'raw' / 'optionmetrics'
staging_om = (Path.home() / 'Downloads' / 'quantdata' / 'driftfire' / 'raw' / 'optionmetrics')
opp_path = None
for c in [repo_om / 'opprcd.csv', staging_om / 'opprcd.csv']:
    if c.exists():
        opp_path = c
        break
if opp_path is None:
    raise SystemExit('STOP: opprcd.csv not found.')

print('=' * 96)
print('V4 FINAL DESIGN RECONCILIATION - RETURN-BLIND (no returns/P&L/RV/scores/t-stats)')
print(f'FIXED {HOLD_CAL_DAYS}-CALENDAR-DAY EXIT (not hold-to-expiry)')
print('=' * 96)


def as_cusip8(s):
    return (s.astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            .str.upper().str[:8].str.zfill(8))


def dist(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {'n': 0}
    q = np.percentile(a, [5, 25, 50, 75, 95])
    return {'n': int(len(a)), 'min': float(a.min()), 'p5': float(q[0]),
            'q1': float(q[1]), 'median': float(q[2]), 'q3': float(q[3]),
            'p95': float(q[4]), 'max': float(a.max()), 'mean': float(a.mean())}


# ==========================================================================
# 1. CALENDAR + FIXED-30-DAY EXIT MAP
# ==========================================================================
print('\n' + '-' * 96)
print('1. CALENDAR AND FIXED 30-CALENDAR-DAY EXIT MAP')
print('-' * 96)

cal_all = pd.DatetimeIndex(sorted(
    pd.read_parquet(crsp_path, columns=['DlyCalDt'])['DlyCalDt'].unique()))
cal_np = cal_all.to_numpy()
dev_cal = cal_all[(cal_all >= DEV_START) & (cal_all <= DEV_END)]
print(f'Trading calendar: {len(cal_all):,} sessions.  DEV sessions: {len(dev_cal):,}')

# exit session = LAST trading session on or before entry + 30 calendar days
# (V3's own n_t convention: sessions in (t, t+30cal]).
dev_pos = cal_all.get_indexer(dev_cal)
exit_pos = np.searchsorted(cal_np, (dev_cal + pd.Timedelta(days=HOLD_CAL_DAYS)).to_numpy(),
                           side='right') - 1
n_hold = exit_pos - dev_pos
ok_exit = (exit_pos < len(cal_all)) & (n_hold > 0)
exit_map = pd.DataFrame({
    'date_d': dev_cal, 'entry_pos': dev_pos, 'exit_pos': exit_pos,
    'n_hold_sessions': n_hold, 'exit_ok': ok_exit,
})
exit_map['exit_date'] = cal_np[np.clip(exit_pos, 0, len(cal_all) - 1)]
print(f'n_hold_sessions over DEV: min={n_hold.min()}, max={n_hold.max()}, '
      f'mean={n_hold.mean():.2f}, mode={pd.Series(n_hold).mode().iloc[0]}')
cal_pos_of = pd.Series(np.arange(len(cal_all)), index=cal_all)


# ==========================================================================
# 2. UNIVERSE / BRIDGE / PRICES / LINK
# ==========================================================================
print('\n' + '-' * 96)
print('2. UNIVERSE, BRIDGE, PRICES, CCM LINK')
print('-' * 96)

univ = pd.read_parquet(univ_path)
univ_in = univ[univ['in_universe']][['PERMNO', 'year_month', 'decile']].copy()
ever = set(univ_in['PERMNO'].unique())

base_v1 = pd.read_parquet(v1_path, columns=['PERMNO', 'DlyCalDt', 'compression_ratio'])
base_v1 = base_v1[(base_v1['DlyCalDt'] >= DEV_START) & (base_v1['DlyCalDt'] <= DEV_END)]
base_v1 = base_v1[base_v1['compression_ratio'].notna()][['PERMNO', 'DlyCalDt']]
base_v1['year_month'] = base_v1['DlyCalDt'].dt.to_period('M').astype(str)
base_v1 = base_v1.merge(univ_in, on=['PERMNO', 'year_month'], how='inner')
base_v1 = base_v1[['PERMNO', 'DlyCalDt', 'decile']].drop_duplicates(['PERMNO', 'DlyCalDt'])
base_v1 = base_v1.rename(columns={'DlyCalDt': 'date_d'})
assert len(base_v1) == 1_733_857, 'base universe row count moved'
print(f'base universe: {len(base_v1):,} stock-days')

dvol = pd.read_parquet(crsp_path, columns=['PERMNO', 'DlyCalDt', 'DlyPrcVol'])
dvol = dvol.drop_duplicates(['PERMNO', 'DlyCalDt']).rename(columns={'DlyCalDt': 'date_d'})
bd = base_v1.merge(dvol, on=['PERMNO', 'date_d'], how='left')
bd['c8_ok'] = bd.groupby('date_d')['DlyPrcVol'].rank(pct=True) >= 0.50
base_key = bd[['PERMNO', 'date_d', 'decile', 'c8_ok']].copy()
base_key['decile'] = base_key['decile'].astype('int8')
print(f"C8 (dvol >= same-day median): {bd['c8_ok'].mean() * 100:.2f}%")

om = pd.read_csv(om_names_path)
om.columns = [c.lower() for c in om.columns]
om = om.dropna(subset=['secid', 'cusip']).copy()
om['c8'] = as_cusip8(om['cusip'])
om = om[om['c8'].str.len() == 8][['secid', 'c8']].drop_duplicates()
crsp_names = pd.read_parquet(crsp_names_path, columns=['PERMNO', 'CUSIP'])
crsp_names = crsp_names[crsp_names['PERMNO'].isin(ever)].dropna(subset=['CUSIP']).copy()
crsp_names['c8'] = as_cusip8(crsp_names['CUSIP'])
crsp_names = crsp_names[crsp_names['c8'].str.len() == 8][['PERMNO', 'c8']].drop_duplicates()
bridge = crsp_names.merge(om, on='c8', how='inner')[['PERMNO', 'secid']].drop_duplicates()
bridge['secid'] = bridge['secid'].astype('int64')
secid_whitelist = set(bridge['secid'].tolist())
print(f'bridge pairs: {len(bridge):,}  unique secid: {len(secid_whitelist):,}')

px = pd.read_parquet(crsp_path, columns=['PERMNO', 'DlyCalDt', 'DlyClose'])
px = px[px['PERMNO'].isin(ever)]
px = px[(px['DlyCalDt'] >= '2014-06-01') & (px['DlyCalDt'] <= '2022-03-31')]
px = px.drop_duplicates(['PERMNO', 'DlyCalDt']).rename(
    columns={'DlyCalDt': 'date_d', 'DlyClose': 'S'})
px['S'] = px['S'].astype('float32')
print(f'price rows: {len(px):,}')

link = pd.read_csv(link_path)
link = link[link['LINKTYPE'].isin(['LC', 'LU']) & link['LINKPRIM'].isin(['P', 'C'])]
link = link[link['LPERMNO'].notna()].copy()
link['linkdt'] = pd.to_datetime(link['LINKDT'])
link['linkend'] = pd.to_datetime(link['LINKENDDT'].replace('E', pd.NaT), errors='coerce')
link['linkend'] = link['linkend'].fillna(pd.Timestamp('2262-01-01'))
link['lpermno'] = link['LPERMNO'].astype(int)
link['gvkey'] = link['gvkey'].astype(int)
lk = link[['lpermno', 'gvkey', 'linkdt', 'linkend', 'LINKPRIM', 'GSECTOR']].copy()

rdq = pd.read_parquet(rdq_path)
rdq.columns = [c.lower() for c in rdq.columns]
for col, keep in [('indfmt', 'INDL'), ('datafmt', 'STD'), ('popsrc', 'D'), ('consol', 'C')]:
    if col in rdq.columns:
        rdq = rdq[rdq[col] == keep]
rdq['rdq'] = pd.to_datetime(rdq['rdq'], errors='coerce')
rdq = rdq.dropna(subset=['rdq'])
rdq['gvkey'] = pd.to_numeric(rdq['gvkey'], errors='coerce')
rdq = rdq.dropna(subset=['gvkey'])
rdq['gvkey'] = rdq['gvkey'].astype(int)
rdq = rdq[['gvkey', 'rdq']].drop_duplicates().sort_values(['gvkey', 'rdq'])
rdq_by_gvkey = {g: s['rdq'].to_numpy() for g, s in rdq.groupby('gvkey')}
print(f'(gvkey, rdq) pairs: {len(rdq):,}')


# ==========================================================================
# 3. PASS A - entry-candidate scan, screens applied PER CHUNK
# ==========================================================================
print('\n' + '-' * 96)
print(f'3. PASS A - entry candidates, DTE [{SCAN_DTE_LO},{SCAN_DTE_HI}], screens per chunk')
print('-' * 96)

NEED = ['secid', 'date', 'exdate', 'cp_flag', 'strike_price', 'best_bid',
        'best_offer', 'volume', 'open_interest', 'impl_volatility',
        'optionid', 'index_flag']
DT = {'secid': 'int32', 'date': 'str', 'exdate': 'str', 'cp_flag': 'category',
      'strike_price': 'float64', 'best_bid': 'float64', 'best_offer': 'float64',
      'volume': 'float64', 'open_interest': 'float64',
      'impl_volatility': 'float64', 'optionid': 'int64', 'index_flag': 'int8'}
DEV_START_S, DEV_END_S = DEV_START.strftime('%Y-%m-%d'), DEV_END.strftime('%Y-%m-%d')

screen_counts = {k: 0 for k in
                 ['dte_universe_eligible', 'c1_valid_quote', 'c2_no_arb', 'c3_iv_sane',
                  'c4_min_price', 'c5_spread15', 'c5_spread10', 'c6_oi', 'c7_sameday_vol',
                  'delta_band', 'all_screens_15', 'all_screens_10']}
parts = []
rows_scanned = 0
for i, ch in enumerate(pd.read_csv(opp_path, usecols=NEED, dtype=DT,
                                   chunksize=OPPRCD_CHUNKSIZE), 1):
    if _MAX_CHUNKS_TEST and i > _MAX_CHUNKS_TEST:
        print(f'  [TEST MODE] stop after {_MAX_CHUNKS_TEST} chunks')
        break
    rows_scanned += len(ch)
    ch = ch[(ch['index_flag'] == 0)]
    ch = ch[ch['secid'].astype('int64').isin(secid_whitelist)]
    if len(ch) == 0:
        continue
    ch = ch[(ch['date'] >= DEV_START_S) & (ch['date'] <= DEV_END_S)]
    if len(ch) == 0:
        continue
    ch = ch.copy()
    ch['date_d'] = pd.to_datetime(ch['date'])
    ch['exdate_d'] = pd.to_datetime(ch['exdate'])
    ch['dte'] = (ch['exdate_d'] - ch['date_d']).dt.days.astype('int16')
    ch = ch[(ch['dte'] >= SCAN_DTE_LO) & (ch['dte'] <= SCAN_DTE_HI)]
    if len(ch) == 0:
        continue
    ch['secid'] = ch['secid'].astype('int64')
    ch = ch.merge(bridge, on='secid', how='inner')
    if len(ch) == 0:
        continue
    ch = ch.merge(base_key, on=['PERMNO', 'date_d'], how='inner')
    if len(ch) == 0:
        continue
    ch = ch.merge(px, on=['PERMNO', 'date_d'], how='inner')
    ch = ch[np.isfinite(ch['S']) & (ch['S'] > 0)]
    if len(ch) == 0:
        continue

    screen_counts['dte_universe_eligible'] += len(ch)
    S = ch['S'].to_numpy(dtype=float)
    K = (ch['strike_price'].to_numpy(dtype=float) / 1000.0)
    T = ch['dte'].to_numpy(dtype=float) / 365.0
    bid = ch['best_bid'].to_numpy(dtype=float)
    off = ch['best_offer'].to_numpy(dtype=float)
    mid = (bid + off) / 2.0
    sig = ch['impl_volatility'].to_numpy(dtype=float)
    is_call = (ch['cp_flag'] == 'C').to_numpy()

    c1 = np.isfinite(bid) & np.isfinite(off) & (bid > 0) & (off > bid)
    disc = np.exp(-RISK_FREE_RATE * T)
    lb = np.where(is_call, np.maximum(S - K * disc, 0.0), np.maximum(K * disc - S, 0.0))
    ub = np.where(is_call, S, K * disc)
    c2 = c1 & (mid >= lb - 1e-9) & (mid <= ub + 1e-9)
    c3 = np.isfinite(sig) & (sig >= IV_SANITY_LO) & (sig <= IV_SANITY_HI)
    c4 = (mid >= MIN_MID_PRICE) & (bid >= MIN_BID_PRICE)
    with np.errstate(all='ignore'):
        rel = np.where(mid > 0, (off - bid) / mid, np.nan)
    c5_15 = rel <= SPREAD_LOOSE
    c5_10 = rel <= SPREAD_TIGHT
    oi = ch['open_interest'].to_numpy(dtype=float)
    vol = ch['volume'].to_numpy(dtype=float)
    c6 = oi >= MIN_OI
    c7d = vol >= MIN_SAME_DAY_VOL
    with np.errstate(all='ignore'):
        sq = sig * np.sqrt(T)
        d1 = (np.log(S / K) + (RISK_FREE_RATE + 0.5 * sig ** 2) * T) / sq
        dc = norm.cdf(d1)
        vega_pt = 100.0 * S * norm.pdf(d1) * np.sqrt(T) * 0.01
        gam = norm.pdf(d1) / (S * sig * np.sqrt(T))
    dv = np.isfinite(sig) & (sig > 0) & np.isfinite(d1)
    delta = np.where(is_call, dc, dc - 1.0)
    ad = np.abs(np.where(dv, delta, np.nan))
    dband = dv & (ad >= DELTA_LO) & (ad <= DELTA_HI)

    for k, arr in [('c1_valid_quote', c1), ('c2_no_arb', c2), ('c3_iv_sane', c3),
                   ('c4_min_price', c4), ('c5_spread15', c5_15), ('c5_spread10', c5_10),
                   ('c6_oi', c6), ('c7_sameday_vol', c7d), ('delta_band', dband)]:
        screen_counts[k] += int(np.nansum(arr))

    keep15 = (c1 & c2 & c3 & c4 & c5_15 & c6 & c7d & dband &
              ch['c8_ok'].to_numpy())
    screen_counts['all_screens_15'] += int(keep15.sum())
    screen_counts['all_screens_10'] += int((keep15 & c5_10).sum())
    if not keep15.any():
        continue
    sub = ch.loc[keep15, ['PERMNO', 'date_d', 'exdate_d', 'dte', 'cp_flag',
                          'optionid', 'best_bid', 'best_offer', 'open_interest',
                          'volume', 'decile']].copy()
    sub['S'] = S[keep15].astype('float32')
    sub['K'] = K[keep15].astype('float32')
    sub['mid'] = mid[keep15].astype('float32')
    sub['rel_spread'] = rel[keep15].astype('float32')
    sub['abs_delta'] = ad[keep15].astype('float32')
    sub['vega_pt'] = vega_pt[keep15].astype('float32')
    sub['gamma'] = gam[keep15].astype('float32')
    sub['spread10_ok'] = c5_10[keep15]
    parts.append(sub)
    if i == 1 or i % 20 == 0:
        print(f'  chunk {i:>4}: scanned {rows_scanned:>13,}  kept {sum(len(p) for p in parts):,}')

cand = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
del parts
print(f'\nPASS A rows scanned: {rows_scanned:,}')
print(f'DTE+universe eligible: {screen_counts["dte_universe_eligible"]:,}')
for k in ['c1_valid_quote', 'c2_no_arb', 'c3_iv_sane', 'c4_min_price', 'c5_spread15',
          'c5_spread10', 'c6_oi', 'c7_sameday_vol', 'delta_band']:
    d = screen_counts['dte_universe_eligible']
    print(f'  {k:<18} {screen_counts[k]:>12,}  ({screen_counts[k] / max(d, 1) * 100:6.2f}%)')
print(f'  ALL screens @15% spread: {screen_counts["all_screens_15"]:,}')
print(f'  ALL screens @10% spread: {screen_counts["all_screens_10"]:,}')
print(f'Entry candidates retained: {len(cand):,}')


# ==========================================================================
# 4. EXIT MAP + EARNINGS EXCLUSION over (entry, EXIT] -- not (entry, exdate]
# ==========================================================================
print('\n' + '-' * 96)
print('4. EXIT MAP + EARNINGS EXCLUSION over (entry, EXIT]  (fixed-30d exit window)')
print('-' * 96)

cand = cand.merge(exit_map[['date_d', 'entry_pos', 'exit_pos', 'exit_date',
                            'n_hold_sessions', 'exit_ok']], on='date_d', how='left')
cand = cand[cand['exit_ok'].fillna(False)].copy()
print(f'candidates with a usable fixed-30d exit session: {len(cand):,}')

sd = cand[['PERMNO', 'date_d']].drop_duplicates()
m = sd.merge(lk, left_on='PERMNO', right_on='lpermno', how='left')
m = m[((m['date_d'] >= m['linkdt']) & (m['date_d'] <= m['linkend'])) | m['lpermno'].isna()]
m['_pr'] = np.where(m['LINKPRIM'] == 'P', 0, 1)
m = m.sort_values(['PERMNO', 'date_d', '_pr', 'gvkey']).drop_duplicates(
    ['PERMNO', 'date_d'], keep='first')
sd = sd.merge(m[['PERMNO', 'date_d', 'gvkey', 'GSECTOR']], on=['PERMNO', 'date_d'], how='left')
cand = cand.merge(sd, on=['PERMNO', 'date_d'], how='left')

gv = cand['gvkey'].to_numpy()
d0 = cand['date_d'].to_numpy()
dx = cand['exit_date'].to_numpy()
earn_in = np.zeros(len(cand), dtype=bool)
hist = np.zeros(len(cand), dtype=bool)
for g in np.unique(gv[~np.isnan(gv)]):
    arr = rdq_by_gvkey.get(int(g))
    sel = np.where(gv == g)[0]
    if arr is None or len(arr) == 0:
        continue
    hist[sel] = True
    earn_in[sel] = (np.searchsorted(arr, dx[sel], side='right') >
                    np.searchsorted(arr, d0[sel], side='right'))
cand['earn_ok'] = hist & (~earn_in) & cand['gvkey'].notna()
print(f'  no gvkey: {int(cand["gvkey"].isna().sum()):,}   '
      f'RDQ inside (entry,exit]: {int(earn_in.sum()):,}')
cand = cand[cand['earn_ok']].copy()
print(f'candidates after earnings exclusion: {len(cand):,}')

cand['exp_before_exit'] = cand['exdate_d'] < cand['exit_date']
cand['rem_dte_at_exit'] = (cand['exdate_d'] - cand['exit_date']).dt.days


# ==========================================================================
# 5. PER-BAND SELECTION (Section 7.1 tie-break, delta-first)
# ==========================================================================
print('\n' + '-' * 96)
print('5. PER-BAND SELECTION')
print('-' * 96)

selected = {}
for tag, lo, hi, target in BANDS:
    for sp_tag, sp_mask in [('sp15', np.ones(len(cand), dtype=bool)),
                            ('sp10', cand['spread10_ok'].to_numpy())]:
        b = cand[(cand['dte'] >= lo) & (cand['dte'] <= hi) & sp_mask].copy()
        if len(b) == 0:
            selected[(tag, sp_tag)] = b
            continue
        b['k1'] = (b['abs_delta'] - 0.50).abs().round(2)
        b['k2'] = (b['dte'] - target).abs()
        b['k3'] = b['rel_spread'].fillna(np.inf)
        b['k4'] = -b['open_interest'].fillna(0.0)
        b = b.sort_values(['PERMNO', 'date_d', 'cp_flag', 'k1', 'k2', 'k3', 'k4', 'optionid'])
        b = b.drop_duplicates(['PERMNO', 'date_d', 'cp_flag'], keep='first')
        selected[(tag, sp_tag)] = b
        if sp_tag == 'sp15':
            print(f'  {tag} [{lo},{hi}] target {target}: {len(b):,} selected '
                  f'(calls {int((b["cp_flag"] == "C").sum()):,}, '
                  f'puts {int((b["cp_flag"] == "P").sum()):,})  '
                  f'exp_before_exit {b["exp_before_exit"].mean() * 100:.2f}%')

all_optionids = set()
for b in selected.values():
    if len(b):
        all_optionids.update(b['optionid'].tolist())
print(f'unique optionids needing path data: {len(all_optionids):,}')


# ==========================================================================
# 6. PASS B - path scan for selected optionids (quote EXISTENCE only)
# ==========================================================================
print('\n' + '-' * 96)
print('6. PASS B - path scan (quote existence + volume history) for selected optionids')
print('-' * 96)

PB_START = (DEV_START - pd.Timedelta(days=60)).strftime('%Y-%m-%d')
PB_NEED = ['secid', 'date', 'best_bid', 'best_offer', 'volume', 'optionid', 'index_flag']
PB_DT = {'secid': 'int32', 'date': 'str', 'best_bid': 'float64',
         'best_offer': 'float64', 'volume': 'float64', 'optionid': 'int64',
         'index_flag': 'int8'}
pparts = []
pscanned = 0
for i, ch in enumerate(pd.read_csv(opp_path, usecols=PB_NEED, dtype=PB_DT,
                                   chunksize=OPPRCD_CHUNKSIZE), 1):
    if _MAX_CHUNKS_TEST and i > _MAX_CHUNKS_TEST:
        print(f'  [TEST MODE] stop after {_MAX_CHUNKS_TEST} chunks')
        break
    pscanned += len(ch)
    ch = ch[ch['index_flag'] == 0]
    ch = ch[(ch['date'] >= PB_START) & (ch['date'] <= DEV_END_S)]
    if len(ch) == 0:
        continue
    ch = ch[ch['optionid'].isin(all_optionids)]
    if len(ch) == 0:
        continue
    ch = ch.copy()
    ch['date_d'] = pd.to_datetime(ch['date'])
    ch['valid_q'] = (np.isfinite(ch['best_bid']) & np.isfinite(ch['best_offer']) &
                     (ch['best_bid'] > 0) & (ch['best_offer'] > ch['best_bid']))
    ch['has_bid'] = np.isfinite(ch['best_bid']) & (ch['best_bid'] > 0)
    pparts.append(ch[['optionid', 'date_d', 'valid_q', 'has_bid', 'volume']])
    if i == 1 or i % 20 == 0:
        print(f'  chunk {i:>4}: scanned {pscanned:>13,}  kept {sum(len(p) for p in pparts):,}')

path = pd.concat(pparts, ignore_index=True) if pparts else pd.DataFrame(
    columns=['optionid', 'date_d', 'valid_q', 'has_bid', 'volume'])
del pparts
print(f'\nPASS B rows scanned: {pscanned:,}   path rows kept: {len(path):,}')

path['cal_pos'] = cal_pos_of.reindex(path['date_d']).to_numpy()
path = path.dropna(subset=['cal_pos'])
path['cal_pos'] = path['cal_pos'].astype(int)

valid_by_oid = {oid: set(g.loc[g['valid_q'], 'cal_pos'].tolist())
                for oid, g in path.groupby('optionid')}
bid_by_oid = {oid: set(g.loc[g['has_bid'], 'cal_pos'].tolist())
              for oid, g in path.groupby('optionid')}
vol_by_oid = {oid: dict(zip(g['cal_pos'], g['volume'].fillna(0.0)))
              for oid, g in path.groupby('optionid')}
print(f'optionids with path data: {len(valid_by_oid):,}')


def pre_entry_stats(oid, entry_pos):
    vs = valid_by_oid.get(oid, set())
    flags = [(entry_pos - k) in vs for k in range(1, PRE_ENTRY_SESSIONS + 1)]
    n_valid = sum(flags)
    worst = cur = 0
    for f in flags:
        cur = 0 if f else cur + 1
        worst = max(worst, cur)
    return n_valid / PRE_ENTRY_SESSIONS, worst


def hold_stats(oid, entry_pos, exit_pos_):
    vs = valid_by_oid.get(oid, set())
    bs = bid_by_oid.get(oid, set())
    sess = list(range(entry_pos + 1, exit_pos_ + 1))
    if not sess:
        return 1.0, 0, True, len(sess)
    flags = [p in vs for p in sess]
    n_valid = sum(flags)
    worst = cur = 0
    for f in flags:
        cur = 0 if f else cur + 1
        worst = max(worst, cur)
    bid_at_exit = (exit_pos_ in bs)
    return n_valid / len(sess), worst, bid_at_exit, len(sess)


def trailing_vol(oid, entry_pos, n_sessions):
    vd = vol_by_oid.get(oid, {})
    vals = [vd.get(entry_pos - k, np.nan) for k in range(0, n_sessions)]
    vals = [v for v in vals if np.isfinite(v)]
    return (np.nansum(vals) if vals else 0.0), (np.mean(vals) if vals else 0.0)


# ==========================================================================
# 7. ITEM 1 + ITEM 5 - per-band DTE / path / quote-continuity report
# ==========================================================================
print('\n' + '-' * 96)
print('7. ITEM 1 (DTE bands) + ITEM 5 (quote survival) -- fixed 30-cal-day exit')
print('-' * 96)

band_report = {}
for tag, lo, hi, target in BANDS:
    b = selected[(tag, 'sp15')]
    if len(b) == 0:
        continue
    rows = []
    for r in b.itertuples(index=False):
        pf, pw = pre_entry_stats(r.optionid, r.entry_pos)
        hf, hw, bx, ns = hold_stats(r.optionid, r.entry_pos, r.exit_pos)
        v5s, _ = trailing_vol(r.optionid, r.entry_pos, 5)
        _, v20a = trailing_vol(r.optionid, r.entry_pos, 20)
        rows.append((r.PERMNO, r.date_d, r.cp_flag, r.optionid, r.dte,
                     r.exp_before_exit, r.rem_dte_at_exit, pf, pw, hf, hw, bx, ns,
                     v5s, v20a, r.open_interest, r.vega_pt, r.gamma,
                     r.best_offer, r.best_bid, r.S, r.GSECTOR, r.decile))
    bb = pd.DataFrame(rows, columns=[
        'PERMNO', 'date_d', 'cp_flag', 'optionid', 'dte', 'exp_before_exit',
        'rem_dte_at_exit', 'pre_valid_frac', 'pre_worst_gap', 'hold_valid_frac',
        'hold_worst_gap', 'bid_at_exit', 'n_hold_sessions', 'vol5_sum', 'vol20_avg',
        'open_interest', 'vega_pt', 'gamma', 'best_offer', 'best_bid', 'S',
        'GSECTOR', 'decile'])
    selected[(tag, 'sp15_enriched')] = bb

    rep = {'band': [lo, hi], 'tie_break_target_dte': target, 'n_selected': len(bb)}
    for side, lbl in [('C', 'calls'), ('P', 'puts')]:
        s = bb[bb['cp_flag'] == side]
        if len(s) == 0:
            rep[lbl] = {'n': 0}
            continue
        rep[lbl] = {
            'n': len(s),
            'unique_underlyings': int(s['PERMNO'].nunique()),
            'pct_expiring_before_exit': float(s['exp_before_exit'].mean() * 100),
            'remaining_dte_at_exit': dist(s['rem_dte_at_exit']),
            'pct_hold_path_fully_quoted': float((s['hold_valid_frac'] >= 0.999).mean() * 100),
            'hold_valid_quote_frac': dist(s['hold_valid_frac']),
            'hold_worst_consecutive_gap': dist(s['hold_worst_gap']),
            'pct_bid_present_at_exit_session': float(s['bid_at_exit'].mean() * 100),
            'pre_entry_valid_frac': dist(s['pre_valid_frac']),
            'pct_pre_entry_ge_90pct': float((s['pre_valid_frac'] >= PRE_ENTRY_MIN_VALID_FRAC).mean() * 100),
            'pct_pre_entry_maxgap_le_1': float((s['pre_worst_gap'] <= PRE_ENTRY_MAX_CONSEC_MISSING).mean() * 100),
            'pct_passing_full_pre_entry_rule': float(
                ((s['pre_valid_frac'] >= PRE_ENTRY_MIN_VALID_FRAC) &
                 (s['pre_worst_gap'] <= PRE_ENTRY_MAX_CONSEC_MISSING)).mean() * 100),
            'by_year': {int(k): int(v) for k, v in
                        s['date_d'].dt.year.value_counts().sort_index().items()},
        }
    band_report[tag] = rep
    print(f'\n  {tag} [{lo},{hi}]  n={len(bb):,}')
    for lbl in ('calls', 'puts'):
        r = rep[lbl]
        if r.get('n', 0) == 0:
            continue
        print(f'    {lbl}: n={r["n"]:,}  expiring-before-exit={r["pct_expiring_before_exit"]:.2f}%  '
              f'rem_DTE@exit median={r["remaining_dte_at_exit"].get("median")}  '
              f'hold fully quoted={r["pct_hold_path_fully_quoted"]:.2f}%  '
              f'bid@exit={r["pct_bid_present_at_exit_session"]:.2f}%  '
              f'pre-entry rule pass={r["pct_passing_full_pre_entry_rule"]:.2f}%')


# ==========================================================================
# 8. ITEM 4 - dynamic vs fixed liquidity cap
# ==========================================================================
print('\n' + '-' * 96)
print('8. ITEM 4 - dynamic vs fixed liquidity cap')
print('-' * 96)

liq_report = {}
for tag, lo, hi, target in BANDS:
    bb = selected.get((tag, 'sp15_enriched'))
    if bb is None or len(bb) == 0:
        continue
    fixed_ok = (bb['open_interest'] >= MIN_OI) & (bb['vol5_sum'] >= MIN_5D_VOL)
    fix_cap = np.minimum(np.floor(FIX_OI_FRAC * bb['open_interest'].fillna(0)),
                         np.floor(FIX_VOL5SUM_FRAC * bb['vol5_sum'].fillna(0)))
    dyn_cap = np.minimum(np.floor(DYN_VOL_FRAC * bb['vol20_avg'].fillna(0)),
                         np.floor(DYN_OI_FRAC * bb['open_interest'].fillna(0)))
    liq_report[tag] = {
        'n': len(bb),
        'fixed_rule_pass_pct': float(fixed_ok.mean() * 100),
        'fixed_cap_ge1_pct': float((fix_cap >= 1).mean() * 100),
        'dynamic_cap_ge1_pct': float((dyn_cap >= 1).mean() * 100),
        'both_ge1_pct': float(((fix_cap >= 1) & (dyn_cap >= 1)).mean() * 100),
        'dynamic_only_ge1_pct': float(((dyn_cap >= 1) & (fix_cap < 1)).mean() * 100),
        'fixed_only_ge1_pct': float(((fix_cap >= 1) & (dyn_cap < 1)).mean() * 100),
        'fixed_cap_contracts': dist(fix_cap),
        'dynamic_cap_contracts': dist(dyn_cap),
        'median_ratio_dyn_over_fixed': float(np.nanmedian(
            np.where(fix_cap > 0, dyn_cap / np.maximum(fix_cap, 1e-9), np.nan))),
    }
    r = liq_report[tag]
    print(f'  {tag}: fixed-thresh pass {r["fixed_rule_pass_pct"]:.2f}%  '
          f'fixed cap>=1 {r["fixed_cap_ge1_pct"]:.2f}%  dyn cap>=1 {r["dynamic_cap_ge1_pct"]:.2f}%  '
          f'median fixed cap {r["fixed_cap_contracts"].get("median")}  '
          f'median dyn cap {r["dynamic_cap_contracts"].get("median")}')


# ==========================================================================
# 9. ITEMS 2 + 3 - capacity across NAV levels x spread ceilings
# ==========================================================================
print('\n' + '-' * 96)
print('9. ITEMS 2+3 - capacity across NAV levels and spread ceilings')
print('-' * 96)


def simulate(bb, nav, use_dynamic_cap=False):
    tgt_vega = (VEGA_CAP_FRAC * nav) / MAX_POSITIONS
    vega_cap = VEGA_CAP_FRAC * nav
    gamma_cap = GAMMA_CAP_FRAC * nav
    slot_cap = nav / MAX_POSITIONS
    cc = {k: 0 for k in ['premium', 'rounding_up_forced', 'volume', 'oi',
                         'sector', 'vega', 'gamma']}
    days, considered = [], 0
    per_pos_contracts = []
    for date_d, g in bb.groupby('date_d', sort=True):
        g = g.sort_values('PERMNO')
        considered += len(g)
        filled, sec, av, ag = 0, {}, 0.0, 0.0
        prem = []
        for r in g.itertuples(index=False):
            if filled >= MAX_POSITIONS:
                break
            vp = r.vega_pt
            if not (np.isfinite(vp) and vp > 0):
                continue
            ideal = tgt_vega / vp
            n = max(1, int(round(ideal)))
            if ideal < 0.5:
                cc['rounding_up_forced'] += 1
            one_prem = 100.0 * r.best_offer
            if one_prem > PREMIUM_SKIP_MULT * slot_cap:
                cc['premium'] += 1
                continue
            # np.nan_to_num, not `x or 0`: NaN is TRUTHY in Python, so
            # `nan or 0` returns nan and int(nan) raises. Defensive.
            oi_v = float(np.nan_to_num(r.open_interest, nan=0.0))
            v20_v = float(np.nan_to_num(r.vol20_avg, nan=0.0))
            v5_v = float(np.nan_to_num(r.vol5_sum, nan=0.0))
            if use_dynamic_cap:
                ocap = np.floor(DYN_OI_FRAC * oi_v)
                vcap = np.floor(DYN_VOL_FRAC * v20_v)
            else:
                ocap = np.floor(FIX_OI_FRAC * oi_v)
                vcap = np.floor(FIX_VOL5SUM_FRAC * v5_v)
            ob, vb = ocap < n, vcap < n
            if ob:
                cc['oi'] += 1
            if vb:
                cc['volume'] += 1
            if ob or vb:
                n = min(n, int(ocap), int(vcap))
            n = max(0, int(n))
            if n < 1:
                continue
            aff = int(np.floor((PREMIUM_SKIP_MULT * slot_cap) / one_prem))
            if n > aff:
                n = max(1, aff)
                cc['premium'] += 1
            cs = sec.get(r.GSECTOR, 0)
            if cs + 1 > SECTOR_CAP_N:
                cc['sector'] += 1
                continue
            if av + n * vp > vega_cap:
                cc['vega'] += 1
                continue
            addg = (r.gamma * n * 100.0 * 0.01 * r.S) if np.isfinite(r.gamma) else 0.0
            if ag + addg > gamma_cap:
                cc['gamma'] += 1
                continue
            filled += 1
            sec[r.GSECTOR] = cs + 1
            av += n * vp
            ag += addg
            prem.append(n * one_prem)
            per_pos_contracts.append(n)
        tot = sum(prem)
        eff_hhi = ((sum(prem) ** 2 / sum(p * p for p in prem)) if prem else 0.0)
        days.append({'n_elig': len(g), 'filled': filled, 'util': tot / nav,
                     'eff_hhi': eff_hhi})
    d = pd.DataFrame(days)
    if len(d) == 0:
        return None
    inv = d[d['filled'] >= MIN_BREADTH]
    return {
        'n_days_with_candidates': int(len(d)),
        'median_filled': float(d['filled'].median()),
        'median_eff_hhi': float(d['eff_hhi'].median()),
        'n_invested_days': int(len(inv)),
        'pct_invested_days': float(len(inv) / len(d) * 100),
        'median_invested_day_utilization_pct': float(inv['util'].median() * 100) if len(inv) else 0.0,
        'median_utilization_all_days_pct': float(d['util'].median() * 100),
        'max_utilization_pct': float(d['util'].max() * 100),
        'median_contracts_per_position': float(np.median(per_pos_contracts)) if per_pos_contracts else 0.0,
        'constraints_pct_of_considered': {k: v / max(considered, 1) * 100 for k, v in cc.items()},
        'total_candidates_considered': considered,
    }


cap_report = {}
for tag, lo, hi, target in BANDS:
    bb15 = selected.get((tag, 'sp15_enriched'))
    if bb15 is None or len(bb15) == 0:
        continue
    oid10 = set(selected[(tag, 'sp10')]['optionid'].tolist()) if len(selected[(tag, 'sp10')]) else set()
    key10 = set(zip(selected[(tag, 'sp10')]['PERMNO'], selected[(tag, 'sp10')]['date_d'],
                    selected[(tag, 'sp10')]['cp_flag'])) if len(selected[(tag, 'sp10')]) else set()
    bb10 = bb15[[ (p, d, c) in key10 for p, d, c in
                  zip(bb15['PERMNO'], bb15['date_d'], bb15['cp_flag']) ]]
    for sp_tag, bbx in [('sp15', bb15), ('sp10', bb10)]:
        for side, slbl in [('C', 'calls'), ('P', 'puts')]:
            s = bbx[bbx['cp_flag'] == side]
            if len(s) == 0:
                continue
            for nav in NAV_LEVELS:
                r = simulate(s, nav)
                if r is None:
                    continue
                cap_report[f'{tag}|{sp_tag}|{slbl}|{int(nav)}'] = r
            # item 4 interacts with item 2: the dynamic cap is run across ALL
            # NAV levels, not one, so the memo can state whether ANY NAV
            # reaches the utilization floor under the stricter rule.
            if sp_tag == 'sp15':
                for nav in NAV_LEVELS:
                    rdyn = simulate(s, nav, use_dynamic_cap=True)
                    if rdyn:
                        cap_report[f'{tag}|{sp_tag}|{slbl}|{int(nav)}|DYNCAP'] = rdyn

for k in sorted(cap_report):
    r = cap_report[k]
    print(f'  {k:<45} filled_med={r["median_filled"]:5.1f}  eff_hhi={r["median_eff_hhi"]:5.2f}  '
          f'inv_days={r["n_invested_days"]:>5}  '
          f'inv_util_med={r["median_invested_day_utilization_pct"]:6.2f}%  '
          f'contracts_med={r["median_contracts_per_position"]:5.0f}  '
          f'round_up_forced={r["constraints_pct_of_considered"]["rounding_up_forced"]:5.2f}%  '
          f'oi_bind={r["constraints_pct_of_considered"]["oi"]:5.1f}%  '
          f'vol_bind={r["constraints_pct_of_considered"]["volume"]:5.1f}%')


# ==========================================================================
# 10. WRITE
# ==========================================================================
out = {
    'meta': {
        'generated_by': 'src/55_v4_design_reconciliation.py',
        'scope': ('RETURN-BLIND. No return, P&L, forward realized variance, '
                  'outcome-based score, or test statistic computed. Forward-looking '
                  'measurements are QUOTE EXISTENCE and calendar/DTE arithmetic only.'),
        'exit_convention': (f'FIXED {HOLD_CAL_DAYS}-calendar-day exit: exit session is the '
                            'last trading session on or before entry + 30 calendar days '
                            "(V3's n_t convention). NOT hold-to-expiry."),
        'bands_tested': [{'tag': t, 'lo': lo, 'hi': hi, 'tie_break_target': tg}
                         for t, lo, hi, tg in BANDS],
        'nav_levels': NAV_LEVELS,
        'spread_ceilings': {'loose': SPREAD_LOOSE, 'tight': SPREAD_TIGHT},
        'pre_entry_rule': {'sessions': PRE_ENTRY_SESSIONS,
                           'min_valid_frac': PRE_ENTRY_MIN_VALID_FRAC,
                           'max_consecutive_missing': PRE_ENTRY_MAX_CONSEC_MISSING},
        'dynamic_cap_rule': {'vol_frac_of_trailing20_avg': DYN_VOL_FRAC,
                             'oi_frac': DYN_OI_FRAC},
        'fixed_cap_rule': {'oi_frac': FIX_OI_FRAC, 'vol5sum_frac': FIX_VOL5SUM_FRAC},
        'passA_rows_scanned': int(rows_scanned),
        'passB_rows_scanned': int(pscanned),
        'screen_counts': screen_counts,
    },
    'item1_dte_bands_and_item5_quote_survival': band_report,
    'item4_liquidity_cap': liq_report,
    'item2_3_capacity': cap_report,
}
out_json.parent.mkdir(parents=True, exist_ok=True)
with open(out_json, 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f'\n[OK] wrote {out_json}')
print('\n' + '=' * 96)
print('RECONCILIATION SCAN COMPLETE. No returns/P&L/RV/scores/t-stats computed.')
print('gate_log.md not touched. No V4 trading script written or run.')
print('=' * 96)
