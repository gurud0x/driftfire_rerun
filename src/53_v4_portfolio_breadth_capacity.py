import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

# ---------------------------------------------------------------------------
# V4 AUDIT ITEM 3 - PORTFOLIO BREADTH AND CAPACITY. RETURN-BLIND.
#
# Measures whether V4's own locked instrument definition (prereg_V4.md
# Section 7.1: delta [0.40,0.60], DTE [25,38]) and liquid-universe screen
# (Section 6.3, C1-C9) can actually fill a 20-position, equal-vega-sized,
# $2,000,000-NAV book on real opprcd quotes - CAPACITY ONLY. No portfolio
# return, P&L, or Score-based ranking is computed anywhere in this script.
# Score requires the expanding-window forecast (Section 4.2), a separate,
# not-yet-audited component; this script never builds or uses one. Where a
# day's eligible-candidate count exceeds 20, slots are filled in a
# NEUTRAL, deterministic, Score-independent order (ascending PERMNO),
# disclosed explicitly - see prereg_V4.md Section 8.1.
#
# Run independently for calls and puts (two separate hypothetical $2M/20-
# slot books) - the real build's cross-book 1-per-underlying rule needs
# the Score ranking this audit does not use, so each book's OWN capacity
# is isolated rather than an invented cross-book resolution.
#
# gate_log.md is NOT touched.
# ---------------------------------------------------------------------------

DEV_START = pd.Timestamp('2015-01-01')
DEV_END = pd.Timestamp('2021-12-31')
DTE_LO, DTE_HI = 25, 38                 # prereg_V4.md Section 7.1
DELTA_LO, DELTA_HI = 0.40, 0.60         # prereg_V4.md Section 7.1
DELTA_TARGET = 0.50
# Extraction band widened past DTE_HI only, to cover a trailing-5-trading-
# day volume lookback (~7-9 calendar days) for candidates near DTE=38
# without truncation - DTE=25 candidates need no buffer since their
# lookback stays inside [25,38]. See src/50's identical reasoning.
SCAN_DTE_LO, SCAN_DTE_HI = 25, 45
RECENT_VOL_TDAYS = 5

RISK_FREE_RATE = 0.01
Q_DIV = 0.0

# Section 6.3 liquidity thresholds (C1-C9)
IV_SANITY_LO, IV_SANITY_HI = 0.05, 2.00      # C3
MIN_MID_PRICE = 0.50                          # C4
MIN_BID_PRICE = 0.20                          # C4
MAX_REL_SPREAD = 0.15                         # C5 (full spread / mid)
MIN_OI = 100                                  # C6
MIN_SAME_DAY_VOL = 10                         # C7
MIN_5D_VOL = 50                               # C7

EARNINGS_WINDOW_NA = 'excluded'  # any RDQ inside (entry, exdate] -> excluded

# Section 8 portfolio construction (REVISED: 20 positions, equal vega)
NAV = 2_000_000.0
MAX_POSITIONS = 20
MIN_INVESTED_DAY = 5
AGG_VEGA_CAP_PER_PT = 0.005 * NAV        # $10,000/pt
TARGET_VEGA_PER_POSITION = AGG_VEGA_CAP_PER_PT / MAX_POSITIONS   # $500/pt
SECTOR_CAP_FRAC = 4 / MAX_POSITIONS      # rescaled from 10/50
AGG_GAMMA_CAP = 0.05 * NAV               # $100,000: max delta change on a 1% move
NOMINAL_SLOT_CAPITAL = NAV / MAX_POSITIONS   # $100,000, reference only
PREMIUM_SKIP_MULT = 2.0                  # skip if 1-contract premium > 2x nominal slot
WHOLE_CONTRACT_TOL = 0.20                # flag if forced count off ideal by >20%

OPPRCD_CHUNKSIZE = 5_000_000

project_root = Path(__file__).parent.parent
crsp_path = project_root / 'data' / 'processed' / 'crsp_combined.parquet'
univ_path = project_root / 'data' / 'processed' / 'universe_membership.parquet'
v1_path = project_root / 'data' / 'processed' / 'compression_signal_v1.parquet'
om_names_path = project_root / 'data' / 'raw' / 'optionmetrics' / 'om_security_names.csv'
crsp_names_path = project_root / 'data' / 'raw' / 'crsp' / 'crsp_names.parquet'
link_path = project_root / 'data' / 'raw' / 'compustat' / 'ccm_link_gics.csv'
rdq_path = project_root / 'data' / 'raw' / 'compustat' / 'rdq_pull_fundq_2014_2026.parquet'
out_json = project_root / 'results' / '53_v4_portfolio_breadth_capacity.json'

repo_om = project_root / 'data' / 'raw' / 'optionmetrics'
staging_om = (Path.home() / 'Downloads' / 'quantdata' / 'driftfire' / 'raw' /
              'optionmetrics')
opp_path = None
for c in [repo_om / 'opprcd.csv', staging_om / 'opprcd.csv']:
    if c.exists():
        opp_path = c
        break
if opp_path is None:
    print('STOP: opprcd.csv not found.')
    raise SystemExit(1)

print('=' * 92)
print('V4 AUDIT ITEM 3 - PORTFOLIO BREADTH AND CAPACITY (return-blind, capacity only)')
print(f"NAV=${NAV:,.0f}  max_positions={MAX_POSITIONS}  "
      f"target_vega/pt/position=${TARGET_VEGA_PER_POSITION:,.0f}  "
      f"sector_cap={SECTOR_CAP_FRAC:.0%}  gamma_cap=${AGG_GAMMA_CAP:,.0f}")
print('=' * 92)
print(f"\nopprcd source: {opp_path} ({opp_path.stat().st_size / 1e9:.2f} GB)")


def as_cusip8(s):
    return (s.astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            .str.upper().str[:8].str.zfill(8))


# ==========================================================================
# 1. UNIVERSE - shared base population (same construction as src/49/src/50)
# ==========================================================================
print('\n' + '-' * 92)
print('1. UNIVERSE')
print('-' * 92)

univ = pd.read_parquet(univ_path)
univ_in = univ[univ['in_universe']][['PERMNO', 'year_month', 'decile']].copy()
ever = set(univ_in['PERMNO'].unique())
print(f"Ever-in-universe PERMNOs: {len(ever):,}")

base_v1 = pd.read_parquet(v1_path, columns=['PERMNO', 'DlyCalDt', 'compression_ratio'])
base_v1 = base_v1[(base_v1['DlyCalDt'] >= DEV_START) & (base_v1['DlyCalDt'] <= DEV_END)]
base_v1 = base_v1[base_v1['compression_ratio'].notna()][['PERMNO', 'DlyCalDt']]
base_v1['year_month'] = base_v1['DlyCalDt'].dt.to_period('M').astype(str)
base_v1 = base_v1.merge(univ_in, on=['PERMNO', 'year_month'], how='inner')
base_v1 = base_v1[['PERMNO', 'DlyCalDt', 'decile']].drop_duplicates(['PERMNO', 'DlyCalDt'])
base_v1 = base_v1.rename(columns={'DlyCalDt': 'date_d'})
print(f"shared base universe: {len(base_v1):,} stock-days [reference: 1,733,857]")
assert len(base_v1) == 1_733_857

# --- C8: underlying dollar volume >= same-day cross-sectional median ---
dvol = pd.read_parquet(crsp_path, columns=['PERMNO', 'DlyCalDt', 'DlyPrcVol'])
dvol = dvol.drop_duplicates(['PERMNO', 'DlyCalDt'], keep='first')
dvol = dvol.rename(columns={'DlyCalDt': 'date_d'})
base_dv = base_v1.merge(dvol, on=['PERMNO', 'date_d'], how='left')
base_dv['dvol_pct'] = base_dv.groupby('date_d')['DlyPrcVol'].rank(pct=True)
base_dv['c8_ok'] = base_dv['dvol_pct'] >= 0.50
print(f"C8 (dvol >= same-day median) rate on base universe: "
      f"{base_dv['c8_ok'].mean() * 100:.2f}%  [prereg_V3 6(b) reference: 44.89%]")
base_key = base_dv[['PERMNO', 'date_d', 'decile', 'c8_ok']].drop_duplicates(['PERMNO', 'date_d'])


# ==========================================================================
# 2. secid <-> PERMNO BRIDGE
# ==========================================================================
print('\n' + '-' * 92)
print('2. secid <-> PERMNO BRIDGE')
print('-' * 92)

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
secid_whitelist = set(bridge['secid'].unique().tolist())
print(f"bridge pairs: {len(bridge):,}  unique secid: {len(secid_whitelist):,}")


# ==========================================================================
# 3. UNDERLYING CLOSE
# ==========================================================================
print('\n' + '-' * 92)
print('3. UNDERLYING CLOSE (DlyClose only)')
print('-' * 92)

px = pd.read_parquet(crsp_path, columns=['PERMNO', 'DlyCalDt', 'DlyClose'])
px = px[px['PERMNO'].isin(ever)]
px = px[(px['DlyCalDt'] >= '2014-06-01') & (px['DlyCalDt'] <= '2022-03-31')]
px = px.drop_duplicates(['PERMNO', 'DlyCalDt'], keep='first')
print(f"Price rows loaded: {len(px):,}  PERMNOs: {px['PERMNO'].nunique():,}")


# ==========================================================================
# 4. CCM LINK - gvkey + gsector attachment (date-windowed)
# ==========================================================================
print('\n' + '-' * 92)
print('4. CCM LINK - gvkey + gsector (date-windowed)')
print('-' * 92)

link = pd.read_csv(link_path)
link = link[link['LINKTYPE'].isin(['LC', 'LU'])]
link = link[link['LINKPRIM'].isin(['P', 'C'])]
link = link[link['LPERMNO'].notna()]
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
print(f"usable (gvkey, rdq) pairs: {len(rdq):,}  gvkeys: {rdq['gvkey'].nunique():,}")


def attach_gvkey_sector(df, date_col):
    n_before = len(df)
    m = df.merge(lk, left_on='PERMNO', right_on='lpermno', how='left')
    in_window = (m[date_col] >= m['linkdt']) & (m[date_col] <= m['linkend'])
    m = m[in_window | m['lpermno'].isna()]
    m['_pr'] = np.where(m['LINKPRIM'] == 'P', 0, 1)
    m = m.sort_values(['PERMNO', date_col, '_pr', 'gvkey'])
    m = m.drop_duplicates(['PERMNO', date_col], keep='first')
    m = m.drop(columns=['lpermno', 'linkdt', 'linkend', 'LINKPRIM', '_pr'])
    print(f"  {n_before:,} -> {len(m):,} after date-windowed link join; "
          f"gvkey matched {int(m['gvkey'].notna().sum()):,} "
          f"({m['gvkey'].notna().mean() * 100:.2f}%)")
    return m


# ==========================================================================
# 5. opprcd SCAN - DTE in [25,45], secid whitelist, DEV dates
# ==========================================================================
print('\n' + '-' * 92)
print(f"5. opprcd SCAN - DTE in [{SCAN_DTE_LO},{SCAN_DTE_HI}]")
print('-' * 92)

NEED = ['secid', 'date', 'exdate', 'cp_flag', 'strike_price', 'best_bid',
        'best_offer', 'volume', 'open_interest', 'impl_volatility',
        'optionid', 'index_flag']
DT = {'secid': 'int32', 'date': 'str', 'exdate': 'str', 'cp_flag': 'category',
      'strike_price': 'float64', 'best_bid': 'float64', 'best_offer': 'float64',
      'volume': 'float64', 'open_interest': 'float64',
      'impl_volatility': 'float64', 'optionid': 'int64', 'index_flag': 'int8'}

DEV_START_S = DEV_START.strftime('%Y-%m-%d')
DEV_END_S = DEV_END.strftime('%Y-%m-%d')

parts = []
rows_scanned = 0
for i, ch in enumerate(pd.read_csv(opp_path, usecols=NEED, dtype=DT,
                                   chunksize=OPPRCD_CHUNKSIZE), 1):
    rows_scanned += len(ch)
    ch = ch[ch['index_flag'] == 0]
    ch = ch[ch['secid'].astype('int64').isin(secid_whitelist)]
    if len(ch) == 0:
        if i == 1 or i % 20 == 0:
            print(f"  chunk {i:>4}: scanned {rows_scanned:>14,} rows  (kept so far: 0)")
        continue
    ch = ch[(ch['date'] >= DEV_START_S) & (ch['date'] <= DEV_END_S)]
    if len(ch) == 0:
        continue
    ch = ch.copy()
    ch['date_d'] = pd.to_datetime(ch['date'])
    ch['exdate_d'] = pd.to_datetime(ch['exdate'])
    ch['dte'] = (ch['exdate_d'] - ch['date_d']).dt.days
    ch = ch[(ch['dte'] >= SCAN_DTE_LO) & (ch['dte'] <= SCAN_DTE_HI)]
    if len(ch) == 0:
        continue
    keep_cols = ['secid', 'date_d', 'exdate_d', 'dte', 'cp_flag', 'strike_price',
                'best_bid', 'best_offer', 'volume', 'open_interest',
                'impl_volatility', 'optionid']
    parts.append(ch[keep_cols])
    if i == 1 or i % 20 == 0:
        kept = sum(len(p) for p in parts)
        print(f"  chunk {i:>4}: scanned {rows_scanned:>14,} rows  (kept so far: {kept:,})")

opp = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=keep_cols)
del parts
print(f"\nTotal opprcd rows scanned: {rows_scanned:,}")
print(f"Candidate contract-day rows retained: {len(opp):,}")
opp['secid'] = opp['secid'].astype('int64')


# ==========================================================================
# 6. JOIN TO PERMNO, RESTRICT TO BASE UNIVERSE, VOLUME ROLLING, NARROW BAND
# ==========================================================================
print('\n' + '-' * 92)
print('6. JOIN, RESTRICT TO BASE UNIVERSE, RECENT VOLUME, NARROW TO ELIGIBLE DTE BAND')
print('-' * 92)

opp = opp.merge(bridge, on='secid', how='inner')
print(f"After PERMNO join: {len(opp):,} rows")

opp = opp.merge(base_key, on=['PERMNO', 'date_d'], how='inner')
print(f"After restricting to base universe + attaching decile/C8: {len(opp):,} rows")

opp = opp.sort_values(['optionid', 'date_d'])
opp['recent_vol_5d'] = (opp.groupby('optionid')['volume']
                        .transform(lambda s: s.rolling(RECENT_VOL_TDAYS, min_periods=1).sum()))
print("[OK] recent_vol_5d computed per optionid, on the full wide band")

n_wide = len(opp)
opp = opp[(opp['dte'] >= DTE_LO) & (opp['dte'] <= DTE_HI)].copy()
print(f"Narrowed to eligible DTE band [{DTE_LO},{DTE_HI}]: {n_wide:,} -> {len(opp):,} rows")

px_s = px.rename(columns={'DlyCalDt': 'date_d', 'DlyClose': 'S'})
opp = opp.merge(px_s, on=['PERMNO', 'date_d'], how='left')
n_before_s = len(opp)
opp = opp[np.isfinite(opp['S']) & (opp['S'] > 0)].copy()
print(f"After requiring a valid underlying close: {len(opp):,} "
      f"(dropped {n_before_s - len(opp):,})")

opp['K'] = (opp['strike_price'] / 1000.0).astype('float32')
opp['T'] = (opp['dte'] / 365.0).astype('float32')
opp['mid'] = ((opp['best_bid'] + opp['best_offer']) / 2.0).astype('float32')
opp['S'] = opp['S'].astype('float32')


# ==========================================================================
# 7. FULL C1-C9 LIQUIDITY FUNNEL + delta band + Greeks (from opprcd's own IV)
# ==========================================================================
print('\n' + '-' * 92)
print('7. LIQUIDITY FUNNEL (C1-C9) + DELTA BAND + GREEKS')
print('-' * 92)

opp['is_call'] = (opp['cp_flag'] == 'C')
opp['c1_valid_quote'] = (np.isfinite(opp['best_bid']) & np.isfinite(opp['best_offer']) &
                         (opp['best_bid'] > 0) & (opp['best_offer'] > opp['best_bid']))

disc = np.exp(-RISK_FREE_RATE * opp['T'].to_numpy(dtype=float))
S = opp['S'].to_numpy(dtype=float)
K = opp['K'].to_numpy(dtype=float)
mid = opp['mid'].to_numpy(dtype=float)
is_call = opp['is_call'].to_numpy()
lb = np.where(is_call, np.maximum(S - K * disc, 0.0), np.maximum(K * disc - S, 0.0))
ub = np.where(is_call, S, K * disc)
opp['c2_no_arb_ok'] = opp['c1_valid_quote'].to_numpy() & (mid >= lb - 1e-9) & (mid <= ub + 1e-9)

sig = opp['impl_volatility'].to_numpy(dtype=float)
opp['c3_iv_sane'] = np.isfinite(sig) & (sig >= IV_SANITY_LO) & (sig <= IV_SANITY_HI)

opp['c4_min_price'] = (opp['mid'] >= MIN_MID_PRICE) & (opp['best_bid'] >= MIN_BID_PRICE)

opp['rel_spread'] = np.where(opp['mid'] > 0,
                             (opp['best_offer'] - opp['best_bid']) / opp['mid'], np.nan)
opp['c5_spread_ok'] = opp['rel_spread'] <= MAX_REL_SPREAD

opp['c6_oi_ok'] = opp['open_interest'] >= MIN_OI
opp['c7_vol_ok'] = (opp['volume'] >= MIN_SAME_DAY_VOL) & (opp['recent_vol_5d'] >= MIN_5D_VOL)

T_ = opp['T'].to_numpy(dtype=float)
with np.errstate(all='ignore'):
    sq = sig * np.sqrt(T_)
    d1 = (np.log(S / K) + (RISK_FREE_RATE + 0.5 * sig ** 2) * T_) / sq
    delta_call = norm.cdf(d1)
    delta_put = delta_call - 1.0
    vega_per_pt = 100.0 * S * norm.pdf(d1) * np.sqrt(T_) * 0.01   # $ per contract per 1 vol pt
    gamma = norm.pdf(d1) / (S * sig * np.sqrt(T_))
delta_valid = np.isfinite(sig) & (sig > 0) & np.isfinite(d1)
delta = np.where(is_call, delta_call, delta_put)
opp['delta'] = np.where(delta_valid, delta, np.nan)
opp['abs_delta'] = np.abs(opp['delta'])
opp['vega_per_pt'] = np.where(delta_valid, vega_per_pt, np.nan)
opp['gamma'] = np.where(delta_valid, gamma, np.nan)
opp['delta_band_ok'] = delta_valid & (opp['abs_delta'] >= DELTA_LO) & (opp['abs_delta'] <= DELTA_HI)

opp['c9_data_ok'] = np.isfinite(opp['S']) & (opp['S'] > 0)   # entry-day completeness only (capacity audit)

funnel_cols = ['c1_valid_quote', 'c2_no_arb_ok', 'c3_iv_sane', 'c4_min_price',
              'c5_spread_ok', 'c6_oi_ok', 'c7_vol_ok', 'c8_ok', 'c9_data_ok',
              'delta_band_ok']
opp['all_screens_ok'] = opp[funnel_cols].all(axis=1)

for c in funnel_cols:
    print(f"  {c:<16} pass rate (of {len(opp):,} DTE/universe-eligible rows): "
          f"{opp[c].mean() * 100:6.2f}%")
print(f"  ALL SCREENS (incl. delta band)                 pass rate: "
      f"{opp['all_screens_ok'].mean() * 100:6.2f}%   "
      f"n={int(opp['all_screens_ok'].sum()):,}")


# ==========================================================================
# 8. EARNINGS EXCLUSION - exact per-candidate (entry_date, exdate] RDQ check
# ==========================================================================
print('\n' + '-' * 92)
print('8. EARNINGS EXCLUSION - per-candidate (entry_date, exdate] RDQ range')
print('-' * 92)

cand = opp[opp['all_screens_ok']].copy()
print(f"Candidates entering earnings check: {len(cand):,}")

cand_sd = attach_gvkey_sector(cand[['PERMNO', 'date_d']].drop_duplicates(), 'date_d')
cand = cand.merge(cand_sd[['PERMNO', 'date_d', 'gvkey', 'GSECTOR']],
                  on=['PERMNO', 'date_d'], how='left')

gv = cand['gvkey'].to_numpy()
d0 = cand['date_d'].to_numpy()
exd = cand['exdate_d'].to_numpy()
has_earn_flag = np.zeros(len(cand), dtype=bool)   # True = an RDQ falls in window -> excluded
has_history = np.zeros(len(cand), dtype=bool)
for g in np.unique(gv[~np.isnan(gv)]):
    arr = rdq_by_gvkey.get(int(g))
    sel = np.where(gv == g)[0]
    if arr is None or len(arr) == 0:
        continue
    has_history[sel] = True
    lo = np.searchsorted(arr, d0[sel], side='right')
    hi = np.searchsorted(arr, exd[sel], side='right')
    has_earn_flag[sel] = (hi > lo)

cand['no_gvkey'] = cand['gvkey'].isna()
cand['no_rdq_history'] = (~has_history) & cand['gvkey'].notna()
cand['earnings_in_window'] = has_earn_flag
cand['earnings_eligible'] = (~cand['no_gvkey']) & (~cand['no_rdq_history']) & (~cand['earnings_in_window'])

print(f"  no gvkey (dropped): {int(cand['no_gvkey'].sum()):,}")
print(f"  gvkey, no RDQ history (dropped): {int(cand['no_rdq_history'].sum()):,}")
print(f"  RDQ falls inside (entry,exdate] (dropped): {int(cand['earnings_in_window'].sum()):,}")
print(f"  earnings-eligible: {int(cand['earnings_eligible'].sum()):,} "
      f"({cand['earnings_eligible'].mean() * 100:.2f}%)")

cand = cand[cand['earnings_eligible']].copy()
print(f"Candidates after earnings exclusion: {len(cand):,}")


# ==========================================================================
# 9. DETERMINISTIC TIE-BREAK (prereg_V4.md Section 7.1, delta-FIRST)
#    "tied within 0.01" on |delta-0.50| implemented as rounding that key to
#    2 decimals before sorting - values in the same rounded bucket are
#    tied on rule 1 and fall through to rule 2 (DTE distance), matching
#    the locked tolerance-based rule via a standard, vectorizable
#    tie-grouping mechanism. Disclosed here, not silently substituted.
# ==========================================================================
print('\n' + '-' * 92)
print('9. DETERMINISTIC TIE-BREAK (delta-first, Section 7.1)')
print('-' * 92)

cand['k1_delta_dist_rounded'] = (cand['abs_delta'] - DELTA_TARGET).abs().round(2)
cand['k2_dte_dist'] = (cand['dte'] - 30).abs()
cand['k3_rel_spread'] = cand['rel_spread'].fillna(np.inf)
cand['k4_neg_oi'] = -cand['open_interest'].fillna(0.0)
cand['k5_optionid'] = cand['optionid']

cand = cand.sort_values(
    ['PERMNO', 'date_d', 'cp_flag', 'k1_delta_dist_rounded', 'k2_dte_dist',
     'k3_rel_spread', 'k4_neg_oi', 'k5_optionid'])
selected = cand.drop_duplicates(['PERMNO', 'date_d', 'cp_flag'], keep='first').copy()
print(f"Selected (one contract per stock-date x side): {len(selected):,}")
print(f"  calls: {int((selected['cp_flag'] == 'C').sum()):,}   "
      f"puts: {int((selected['cp_flag'] == 'P').sum()):,}")
print(f"  unique underlyings: {selected['PERMNO'].nunique():,}")


# ==========================================================================
# 10. DAILY PORTFOLIO CAPACITY SIMULATION - per book (calls, puts)
#     independently. NEUTRAL order (ascending PERMNO), no Score. Equal-vega
#     sizing, whole contracts, capacity cap, premium/sector/vega/gamma
#     constraints, in the fixed check order disclosed below. NO RETURN,
#     NO P&L computed - fill/skip/contract-count bookkeeping only.
# ==========================================================================
print('\n' + '-' * 92)
print('10. DAILY PORTFOLIO CAPACITY SIMULATION (per book, neutral order, capacity only)')
print('-' * 92)


def simulate_book(df_side, side_label):
    daily_records = []
    constraint_counts = {
        'premium_constrained': 0, 'whole_contract_rounding_constrained': 0,
        'volume_constrained': 0, 'oi_constrained': 0,
        'sector_constrained': 0, 'vega_constrained': 0, 'gamma_constrained': 0,
    }
    n_eligible_total = 0
    n_filled_total = 0
    position_contract_counts = []

    for date_d, g in df_side.groupby('date_d', sort=True):
        g = g.sort_values('PERMNO')  # neutral, deterministic, Score-independent order
        n_eligible = len(g)
        n_eligible_total += n_eligible

        filled = 0
        sector_counts = {}
        agg_vega = 0.0
        agg_gamma_dollar = 0.0
        day_contracts = []
        day_premiums = []

        for row in g.itertuples(index=False):
            if filled >= MAX_POSITIONS:
                break
            vega_pt = row.vega_per_pt
            gamma_ = row.gamma
            ask = row.best_offer
            S_ = row.S
            oi = row.open_interest
            recvol = row.recent_vol_5d
            sector = row.GSECTOR

            if not (np.isfinite(vega_pt) and vega_pt > 0):
                continue  # cannot size - not counted as a named constraint (upstream data issue)

            ideal_contracts = TARGET_VEGA_PER_POSITION / vega_pt
            contracts_vega_sized = max(1, round(ideal_contracts))
            whole_contract_flag = (abs(contracts_vega_sized / max(ideal_contracts, 1e-9) - 1.0)
                                   > WHOLE_CONTRACT_TOL)

            # early skip: even 1 contract already exceeds the premium ceiling
            # (the low-vega/high-price case no amount of contract-count
            # reduction can fix)
            one_contract_premium = 100.0 * ask
            if one_contract_premium > PREMIUM_SKIP_MULT * NOMINAL_SLOT_CAPITAL:
                constraint_counts['premium_constrained'] += 1
                continue

            # capacity cap: OI and recent-volume ceilings, reported as two
            # INDEPENDENT, non-exclusive binding flags (both may bind on
            # the same candidate) - counted whenever the cap is BELOW the
            # vega-implied target, not only when it reduces the position
            # all the way to zero. The earlier version only counted the
            # zero-out case, silently missing every candidate whose size
            # was merely shrunk (the common case at these contract counts).
            oi_cap = np.floor(0.10 * oi) if np.isfinite(oi) else 0
            vol_cap = np.floor(0.20 * recvol) if np.isfinite(recvol) else 0
            oi_binds = oi_cap < contracts_vega_sized
            vol_binds = vol_cap < contracts_vega_sized
            if oi_binds:
                constraint_counts['oi_constrained'] += 1
            if vol_binds:
                constraint_counts['volume_constrained'] += 1
            capped_contracts = contracts_vega_sized
            if oi_binds or vol_binds:
                capped_contracts = min(capped_contracts, int(oi_cap), int(vol_cap))
            capped_contracts = max(0, int(capped_contracts))
            if capped_contracts < 1:
                continue   # already counted above (oi_binds/vol_binds)
            if whole_contract_flag and capped_contracts == contracts_vega_sized:
                constraint_counts['whole_contract_rounding_constrained'] += 1

            # PREMIUM CEILING ON THE ACTUAL SIZED POSITION - the case the
            # 1-contract check above cannot catch: a low-vega contract needs
            # MANY contracts to hit the vega target, and their combined
            # premium can still blow the budget even though a single
            # contract was affordable. Reduce to the affordable count rather
            # than abandon the trade outright (a real trader would size
            # down, not walk away, unless even 1 contract is unaffordable -
            # that case is already handled above).
            max_affordable = int(np.floor(
                (PREMIUM_SKIP_MULT * NOMINAL_SLOT_CAPITAL) / one_contract_premium))
            if capped_contracts > max_affordable:
                capped_contracts = max(1, max_affordable)
                constraint_counts['premium_constrained'] += 1

            # sector cap
            cur_sector_n = sector_counts.get(sector, 0)
            if cur_sector_n + 1 > SECTOR_CAP_FRAC * MAX_POSITIONS:
                constraint_counts['sector_constrained'] += 1
                continue

            # aggregate vega cap
            add_vega = capped_contracts * vega_pt
            if agg_vega + add_vega > AGG_VEGA_CAP_PER_PT:
                constraint_counts['vega_constrained'] += 1
                continue

            # aggregate gamma cap: dollar delta change on a simultaneous 1% move
            add_gamma_dollar = (gamma_ * capped_contracts * 100.0 * (0.01 * S_)) if np.isfinite(gamma_) else 0.0
            if agg_gamma_dollar + add_gamma_dollar > AGG_GAMMA_CAP:
                constraint_counts['gamma_constrained'] += 1
                continue

            # POSITION ENTERED
            filled += 1
            sector_counts[sector] = cur_sector_n + 1
            agg_vega += add_vega
            agg_gamma_dollar += add_gamma_dollar
            day_contracts.append(capped_contracts)
            day_premiums.append(capped_contracts * 100.0 * ask)
            position_contract_counts.append(capped_contracts)

        n_filled_total += filled
        daily_records.append({
            'date': str(date_d.date()), 'n_eligible': n_eligible, 'n_filled': filled,
            'agg_vega_pt': agg_vega, 'agg_gamma_dollar': agg_gamma_dollar,
            'capital_deployed': sum(day_premiums),
        })

    daily_df = pd.DataFrame(daily_records)
    n_days = len(daily_df)
    n_thin_days = int((daily_df['n_eligible'] < MIN_INVESTED_DAY).sum()) if n_days else 0
    n_invested_days = int((daily_df['n_filled'] >= MIN_INVESTED_DAY).sum()) if n_days else 0

    print(f"\n  --- {side_label} ---")
    print(f"    trading days with >=1 candidate: {n_days:,}")
    print(f"    pct days with <{MIN_INVESTED_DAY} eligible candidates: "
          f"{n_thin_days / max(n_days, 1) * 100:.2f}%")
    print(f"    median eligible/day: {daily_df['n_eligible'].median() if n_days else 0:.1f}   "
          f"median filled/day: {daily_df['n_filled'].median() if n_days else 0:.1f}   "
          f"max filled/day: {daily_df['n_filled'].max() if n_days else 0}")
    print(f"    pct days classified 'invested' (>= {MIN_INVESTED_DAY} filled): "
          f"{n_invested_days / max(n_days, 1) * 100:.2f}%")
    if position_contract_counts:
        pc = np.array(position_contract_counts)
        print(f"    contract count per position: median {np.median(pc):.1f}  "
              f"max {pc.max()}  mean {pc.mean():.2f}")
    total_candidates_considered = n_eligible_total
    print(f"    total position-candidates considered (sum over days): {total_candidates_considered:,}")
    for k, v in constraint_counts.items():
        print(f"    {k}: {v:,} ({v / max(total_candidates_considered, 1) * 100:.2f}% of candidates considered)")

    utilization = (daily_df['capital_deployed'] / NAV) if n_days else pd.Series(dtype=float)
    print(f"    projected capital utilization: median {utilization.median() * 100 if n_days else 0:.2f}%  "
          f"mean {utilization.mean() * 100 if n_days else 0:.2f}%  "
          f"max {utilization.max() * 100 if n_days else 0:.2f}%")

    return {
        'n_trading_days_with_candidate': n_days,
        'pct_days_below_min_invested_eligible': n_thin_days / max(n_days, 1) * 100,
        'pct_days_invested': n_invested_days / max(n_days, 1) * 100,
        'median_eligible_per_day': float(daily_df['n_eligible'].median()) if n_days else None,
        'median_filled_per_day': float(daily_df['n_filled'].median()) if n_days else None,
        'max_filled_per_day': int(daily_df['n_filled'].max()) if n_days else None,
        'contracts_per_position': {
            'median': float(np.median(position_contract_counts)) if position_contract_counts else None,
            'max': int(np.max(position_contract_counts)) if position_contract_counts else None,
            'mean': float(np.mean(position_contract_counts)) if position_contract_counts else None,
        },
        'total_candidates_considered': total_candidates_considered,
        'constraint_counts': constraint_counts,
        'constraint_pct_of_candidates': {k: v / max(total_candidates_considered, 1) * 100
                                         for k, v in constraint_counts.items()},
        'capital_utilization_pct': {
            'median': float(utilization.median() * 100) if n_days else None,
            'mean': float(utilization.mean() * 100) if n_days else None,
            'max': float(utilization.max() * 100) if n_days else None,
        },
    }


results = {}
for side_label, flag in [('CALLS', True), ('PUTS', False)]:
    df_side = selected[selected['is_call'] == flag]
    results[side_label] = simulate_book(df_side, side_label)


# ==========================================================================
# 11. WRITE OUTPUT
# ==========================================================================
out = {
    'meta': {
        'generated_by': 'src/53_v4_portfolio_breadth_capacity.py',
        'scope': ('RETURN-BLIND capacity audit (V4 audit item 3). No P&L, '
                  'return, or Score-based ranking computed. Score requires '
                  'the not-yet-audited expanding-window forecast (Section '
                  '4.2); candidate slot-filling uses a neutral, '
                  'deterministic, Score-independent order (ascending '
                  'PERMNO) instead.'),
        'window': f'{DEV_START.date()} to {DEV_END.date()} (DEV only)',
        'instrument_definition': {'dte_band': [DTE_LO, DTE_HI],
                                  'delta_band': [DELTA_LO, DELTA_HI]},
        'portfolio_params': {
            'NAV': NAV, 'max_positions': MAX_POSITIONS,
            'min_invested_day': MIN_INVESTED_DAY,
            'sizing': 'equal_vega',
            'target_vega_per_position_per_pt': TARGET_VEGA_PER_POSITION,
            'aggregate_vega_cap_per_pt': AGG_VEGA_CAP_PER_PT,
            'sector_cap_fraction': SECTOR_CAP_FRAC,
            'aggregate_gamma_cap_dollar': AGG_GAMMA_CAP,
            'premium_skip_multiple_of_nominal_slot': PREMIUM_SKIP_MULT,
            'whole_contract_rounding_tolerance': WHOLE_CONTRACT_TOL,
        },
        'constraint_check_order': ['premium', 'whole_contract_rounding (flagged, not skipped alone)',
                                   'capacity_cap (OI then volume)', 'sector', 'aggregate_vega',
                                   'aggregate_gamma'],
        'opprcd_rows_scanned': int(rows_scanned),
    },
    'results': results,
}
out_json.parent.mkdir(parents=True, exist_ok=True)
with open(out_json, 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f"\n[OK] wrote {out_json}")

print('\n' + '=' * 92)
print('AUDIT COMPLETE. Capacity only - no return, P&L, or Score computed.')
print('gate_log.md was not touched.')
print('=' * 92)
