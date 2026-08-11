import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

# ---------------------------------------------------------------------------
# V4 PENDING MEASUREMENTS - RETURN-BLIND. Closes the two open items that
# block LOCK on results/prereg_V4.md:
#
#   O1 (section 8.2)  COMBINED CASE: 10% spread ceiling AND dynamic
#                     liquidity cap together, reporting utilization,
#                     effective breadth, and invested days. Previously each
#                     was measured only separately.
#
#   O2 (section 7.8)  LISTING-RECENCY SEPARATION: of the ~38%/32% pre-entry
#                     exclusion, how much is contracts NOT YET LISTED at the
#                     20-session lookback point vs contracts LISTED BUT
#                     GENUINELY UNQUOTED. Keyed on each contract's FIRST
#                     OBSERVED QUOTE DATE.
#
# NOTHING HERE COMPUTES A RETURN, P&L, FORWARD REALIZED VARIANCE, AN
# OUTCOME-BASED SCORE, OR A TEST STATISTIC. DlyRet is never loaded. The only
# forward-looking quantities are QUOTE EXISTENCE and calendar/DTE
# arithmetic. Entry premium (a cost at entry) and capital utilization are
# computed; no exit price is ever differenced against an entry price.
#
# gate_log.md is NOT touched. No V4 trading script is written or run.
#
# For O2 the contract's first-quote date must be found across the WHOLE
# file, not just inside the entry band - a contract's earliest quote can sit
# at any DTE. PASS B therefore scans without a DTE filter, keyed on the
# selected optionid set.
# ---------------------------------------------------------------------------

DEV_START = pd.Timestamp('2015-01-01')
DEV_END = pd.Timestamp('2021-12-31')
HOLD_CAL_DAYS = 30

DTE_LO, DTE_HI, DTE_TARGET = 40, 60, 50      # LOCKED band (prereg 7.1)
SCAN_DTE_LO, SCAN_DTE_HI = 40, 60
DELTA_LO, DELTA_HI = 0.40, 0.60
RISK_FREE_RATE = 0.01

IV_SANITY_LO, IV_SANITY_HI = 0.05, 2.00
MIN_MID_PRICE = 0.50
MIN_BID_PRICE = 0.20
SPREAD_CEILING = 0.10                         # LOCKED 10% (prereg 6.3 C5)
MIN_OI = 100
MIN_SAME_DAY_VOL = 10
MIN_5D_VOL = 50

PRE_ENTRY_SESSIONS = 20
PRE_ENTRY_MIN_VALID_FRAC = 0.90
PRE_ENTRY_MAX_CONSEC_MISSING = 1

NAV_LEVELS = [100_000, 250_000, 500_000, 1_000_000, 2_000_000]
NAV_PRIMARY = 100_000
MAX_POSITIONS = 20
MIN_BREADTH = 5
VEGA_CAP_FRAC = 0.005
GAMMA_CAP_FRAC = 0.05
SECTOR_CAP_N = 4
PREMIUM_SKIP_MULT = 2.0

FIX_OI_FRAC, FIX_VOL5SUM_FRAC = 0.10, 0.20    # incumbent primary cap
DYN_OI_FRAC, DYN_VOL20AVG_FRAC = 0.01, 0.05   # pending alternative

OPPRCD_CHUNKSIZE = 5_000_000
_MAX_CHUNKS_TEST = 0  # full run; set >0 only for pipeline validation

project_root = Path(__file__).parent.parent
crsp_path = project_root / 'data' / 'processed' / 'crsp_combined.parquet'
univ_path = project_root / 'data' / 'processed' / 'universe_membership.parquet'
v1_path = project_root / 'data' / 'processed' / 'compression_signal_v1.parquet'
om_names_path = project_root / 'data' / 'raw' / 'optionmetrics' / 'om_security_names.csv'
crsp_names_path = project_root / 'data' / 'raw' / 'crsp' / 'crsp_names.parquet'
link_path = project_root / 'data' / 'raw' / 'compustat' / 'ccm_link_gics.csv'
rdq_path = project_root / 'data' / 'raw' / 'compustat' / 'rdq_pull_fundq_2014_2026.parquet'
out_json = project_root / 'results' / '56_v4_pending_measurements.json'

repo_om = project_root / 'data' / 'raw' / 'optionmetrics'
staging_om = (Path.home() / 'Downloads' / 'quantdata' / 'driftfire' / 'raw' / 'optionmetrics')
opp_path = next((c for c in [repo_om / 'opprcd.csv', staging_om / 'opprcd.csv'] if c.exists()), None)
if opp_path is None:
    raise SystemExit('STOP: opprcd.csv not found.')

print('=' * 96)
print('V4 PENDING MEASUREMENTS - RETURN-BLIND')
print('O1: combined 10% ceiling + dynamic cap    O2: listing-recency separation')
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
    return {'n': int(len(a)), 'min': float(a.min()), 'p5': float(q[0]), 'q1': float(q[1]),
            'median': float(q[2]), 'q3': float(q[3]), 'p95': float(q[4]),
            'max': float(a.max()), 'mean': float(a.mean())}


# ---- calendar + fixed 30d exit map ----
cal_all = pd.DatetimeIndex(sorted(
    pd.read_parquet(crsp_path, columns=['DlyCalDt'])['DlyCalDt'].unique()))
cal_np = cal_all.to_numpy()
dev_cal = cal_all[(cal_all >= DEV_START) & (cal_all <= DEV_END)]
dev_pos = cal_all.get_indexer(dev_cal)
exit_pos = np.searchsorted(cal_np, (dev_cal + pd.Timedelta(days=HOLD_CAL_DAYS)).to_numpy(),
                           side='right') - 1
n_hold = exit_pos - dev_pos
exit_map = pd.DataFrame({'date_d': dev_cal, 'entry_pos': dev_pos, 'exit_pos': exit_pos,
                         'exit_ok': (exit_pos < len(cal_all)) & (n_hold > 0)})
exit_map['exit_date'] = cal_np[np.clip(exit_pos, 0, len(cal_all) - 1)]
cal_pos_of = pd.Series(np.arange(len(cal_all)), index=cal_all)
print(f'\nDEV sessions {len(dev_cal):,};  n_hold {n_hold.min()}-{n_hold.max()} '
      f'(mean {n_hold.mean():.2f})')

# ---- universe / bridge / prices / link ----
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
assert len(base_v1) == 1_733_857
dvol = pd.read_parquet(crsp_path, columns=['PERMNO', 'DlyCalDt', 'DlyPrcVol'])
dvol = dvol.drop_duplicates(['PERMNO', 'DlyCalDt']).rename(columns={'DlyCalDt': 'date_d'})
bd = base_v1.merge(dvol, on=['PERMNO', 'date_d'], how='left')
bd['c8_ok'] = bd.groupby('date_d')['DlyPrcVol'].rank(pct=True) >= 0.50
base_key = bd[['PERMNO', 'date_d', 'decile', 'c8_ok']].copy()

om = pd.read_csv(om_names_path)
om.columns = [c.lower() for c in om.columns]
om = om.dropna(subset=['secid', 'cusip']).copy()
om['c8'] = as_cusip8(om['cusip'])
om = om[om['c8'].str.len() == 8][['secid', 'c8']].drop_duplicates()
cn = pd.read_parquet(crsp_names_path, columns=['PERMNO', 'CUSIP'])
cn = cn[cn['PERMNO'].isin(ever)].dropna(subset=['CUSIP']).copy()
cn['c8'] = as_cusip8(cn['CUSIP'])
cn = cn[cn['c8'].str.len() == 8][['PERMNO', 'c8']].drop_duplicates()
bridge = cn.merge(om, on='c8', how='inner')[['PERMNO', 'secid']].drop_duplicates()
bridge['secid'] = bridge['secid'].astype('int64')
secid_whitelist = set(bridge['secid'].tolist())

px = pd.read_parquet(crsp_path, columns=['PERMNO', 'DlyCalDt', 'DlyClose'])
px = px[px['PERMNO'].isin(ever)]
px = px[(px['DlyCalDt'] >= '2014-06-01') & (px['DlyCalDt'] <= '2022-03-31')]
px = px.drop_duplicates(['PERMNO', 'DlyCalDt']).rename(
    columns={'DlyCalDt': 'date_d', 'DlyClose': 'S'})
px['S'] = px['S'].astype('float32')

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


# ==========================================================================
# PASS A - entry candidates at the LOCKED band and LOCKED 10% ceiling
# ==========================================================================
print('\n' + '-' * 96)
print(f'PASS A - entry candidates, DTE [{DTE_LO},{DTE_HI}], spread ceiling '
      f'{SPREAD_CEILING:.0%}')
print('-' * 96)

NEED = ['secid', 'date', 'exdate', 'cp_flag', 'strike_price', 'best_bid', 'best_offer',
        'volume', 'open_interest', 'impl_volatility', 'optionid', 'index_flag']
DT = {'secid': 'int32', 'date': 'str', 'exdate': 'str', 'cp_flag': 'category',
      'strike_price': 'float64', 'best_bid': 'float64', 'best_offer': 'float64',
      'volume': 'float64', 'open_interest': 'float64', 'impl_volatility': 'float64',
      'optionid': 'int64', 'index_flag': 'int8'}
DEV_START_S, DEV_END_S = DEV_START.strftime('%Y-%m-%d'), DEV_END.strftime('%Y-%m-%d')

parts, rows_scanned = [], 0
for i, ch in enumerate(pd.read_csv(opp_path, usecols=NEED, dtype=DT,
                                   chunksize=OPPRCD_CHUNKSIZE), 1):
    if _MAX_CHUNKS_TEST and i > _MAX_CHUNKS_TEST:
        print(f'  [TEST MODE] stop after {_MAX_CHUNKS_TEST}')
        break
    rows_scanned += len(ch)
    ch = ch[ch['index_flag'] == 0]
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

    S = ch['S'].to_numpy(dtype=float)
    K = ch['strike_price'].to_numpy(dtype=float) / 1000.0
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
    c5 = rel <= SPREAD_CEILING
    oi = ch['open_interest'].to_numpy(dtype=float)
    c6 = oi >= MIN_OI
    c7d = ch['volume'].to_numpy(dtype=float) >= MIN_SAME_DAY_VOL
    with np.errstate(all='ignore'):
        sq = sig * np.sqrt(T)
        d1 = (np.log(S / K) + (RISK_FREE_RATE + 0.5 * sig ** 2) * T) / sq
        dc = norm.cdf(d1)
        vega_pt = 100.0 * S * norm.pdf(d1) * np.sqrt(T) * 0.01
        gam = norm.pdf(d1) / (S * sig * np.sqrt(T))
    dv = np.isfinite(sig) & (sig > 0) & np.isfinite(d1)
    ad = np.abs(np.where(dv, np.where(is_call, dc, dc - 1.0), np.nan))
    dband = dv & (ad >= DELTA_LO) & (ad <= DELTA_HI)

    keep = c1 & c2 & c3 & c4 & c5 & c6 & c7d & dband & ch['c8_ok'].to_numpy()
    if not keep.any():
        continue
    sub = ch.loc[keep, ['PERMNO', 'date_d', 'exdate_d', 'dte', 'cp_flag', 'optionid',
                        'best_bid', 'best_offer', 'open_interest', 'decile']].copy()
    sub['S'] = S[keep].astype('float32')
    sub['rel_spread'] = rel[keep].astype('float32')
    sub['abs_delta'] = ad[keep].astype('float32')
    sub['vega_pt'] = vega_pt[keep].astype('float32')
    sub['gamma'] = gam[keep].astype('float32')
    parts.append(sub)
    if i == 1 or i % 25 == 0:
        print(f'  chunk {i:>4}: scanned {rows_scanned:>13,}  kept {sum(len(p) for p in parts):,}')

cand = pd.concat(parts, ignore_index=True)
del parts
print(f'\nPASS A scanned {rows_scanned:,}; candidates at 10% ceiling: {len(cand):,}')

# ---- exit map + earnings over (entry, exit] ----
cand = cand.merge(exit_map, on='date_d', how='left')
cand = cand[cand['exit_ok'].fillna(False)].copy()
sd = cand[['PERMNO', 'date_d']].drop_duplicates()
m = sd.merge(lk, left_on='PERMNO', right_on='lpermno', how='left')
m = m[((m['date_d'] >= m['linkdt']) & (m['date_d'] <= m['linkend'])) | m['lpermno'].isna()]
m['_pr'] = np.where(m['LINKPRIM'] == 'P', 0, 1)
m = m.sort_values(['PERMNO', 'date_d', '_pr', 'gvkey']).drop_duplicates(
    ['PERMNO', 'date_d'], keep='first')
cand = cand.merge(m[['PERMNO', 'date_d', 'gvkey', 'GSECTOR']], on=['PERMNO', 'date_d'], how='left')

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
cand = cand[hist & (~earn_in) & cand['gvkey'].notna()].copy()
print(f'after earnings exclusion: {len(cand):,}')

# ---- tie-break selection (prereg 7.1, anchor 50) ----
cand['k1'] = (cand['abs_delta'] - 0.50).abs().round(2)
cand['k2'] = (cand['dte'] - DTE_TARGET).abs()
cand['k3'] = cand['rel_spread'].fillna(np.inf)
cand['k4'] = -cand['open_interest'].fillna(0.0)
cand = cand.sort_values(['PERMNO', 'date_d', 'cp_flag', 'k1', 'k2', 'k3', 'k4', 'optionid'])
sel = cand.drop_duplicates(['PERMNO', 'date_d', 'cp_flag'], keep='first').copy()
print(f'selected (1 per PERMNO x date x side): {len(sel):,}  '
      f'calls {int((sel["cp_flag"] == "C").sum()):,}  puts {int((sel["cp_flag"] == "P").sum()):,}')
sel_oids = set(sel['optionid'].tolist())


# ==========================================================================
# PASS B - full contract history (NO DTE filter): first-quote date for O2,
# plus trailing volumes and pre-entry quote existence.
# ==========================================================================
print('\n' + '-' * 96)
print('PASS B - full contract history for selected optionids (no DTE filter)')
print('-' * 96)

PB_NEED = ['date', 'best_bid', 'best_offer', 'volume', 'optionid', 'index_flag']
PB_DT = {'date': 'str', 'best_bid': 'float64', 'best_offer': 'float64',
         'volume': 'float64', 'optionid': 'int64', 'index_flag': 'int8'}
pparts, pscanned = [], 0
for i, ch in enumerate(pd.read_csv(opp_path, usecols=PB_NEED, dtype=PB_DT,
                                   chunksize=OPPRCD_CHUNKSIZE), 1):
    if _MAX_CHUNKS_TEST and i > _MAX_CHUNKS_TEST:
        print(f'  [TEST MODE] stop after {_MAX_CHUNKS_TEST}')
        break
    pscanned += len(ch)
    ch = ch[ch['index_flag'] == 0]
    ch = ch[ch['optionid'].isin(sel_oids)]
    if len(ch) == 0:
        continue
    ch = ch.copy()
    ch['date_d'] = pd.to_datetime(ch['date'])
    ch['valid_q'] = (np.isfinite(ch['best_bid']) & np.isfinite(ch['best_offer']) &
                     (ch['best_bid'] > 0) & (ch['best_offer'] > ch['best_bid']))
    pparts.append(ch[['optionid', 'date_d', 'valid_q', 'volume']])
    if i == 1 or i % 25 == 0:
        print(f'  chunk {i:>4}: scanned {pscanned:>13,}  kept {sum(len(p) for p in pparts):,}')

path = pd.concat(pparts, ignore_index=True)
del pparts
path['cal_pos'] = cal_pos_of.reindex(path['date_d']).to_numpy()
path = path.dropna(subset=['cal_pos'])
path['cal_pos'] = path['cal_pos'].astype(int)
print(f'\nPASS B scanned {pscanned:,}; path rows {len(path):,}; '
      f'optionids {path["optionid"].nunique():,}')

# FIRST OBSERVED QUOTE POSITION per contract - the O2 key. "Listed" is
# defined as: any row exists for that optionid on that date (quoted or not),
# so first_row_pos is the earliest date the contract appears in opprcd at
# all. first_valid_pos is the earliest date it had a VALID two-sided quote.
first_row = path.groupby('optionid')['cal_pos'].min()
first_valid = path[path['valid_q']].groupby('optionid')['cal_pos'].min()
valid_by_oid = {o: set(g.loc[g['valid_q'], 'cal_pos'].tolist())
                for o, g in path.groupby('optionid')}
rows_by_oid = {o: set(g['cal_pos'].tolist()) for o, g in path.groupby('optionid')}
vol_by_oid = {o: dict(zip(g['cal_pos'], g['volume'].fillna(0.0)))
              for o, g in path.groupby('optionid')}
first_row_d = first_row.to_dict()
first_valid_d = first_valid.to_dict()


# ==========================================================================
# O2 - LISTING-RECENCY SEPARATION
# ==========================================================================
print('\n' + '=' * 96)
print('O2 - LISTING-RECENCY SEPARATION of the pre-entry exclusion')
print('=' * 96)

o2_rows = []
for r in sel.itertuples(index=False):
    oid, ep = r.optionid, r.entry_pos
    vs = valid_by_oid.get(oid, set())
    rs = rows_by_oid.get(oid, set())
    fr = first_row_d.get(oid, np.nan)
    lookback = [ep - k for k in range(1, PRE_ENTRY_SESSIONS + 1)]
    n_valid = sum(1 for p in lookback if p in vs)
    # sessions BEFORE the contract's first appearance anywhere in opprcd
    n_prelist = sum(1 for p in lookback if np.isfinite(fr) and p < fr)
    # sessions where the contract existed (had a row) but had no valid quote
    n_listed_unquoted = sum(1 for p in lookback if (p in rs) and (p not in vs))
    # sessions with neither a row nor pre-listing explanation (gap in file)
    n_norow_postlist = sum(1 for p in lookback
                           if (p not in rs) and np.isfinite(fr) and p >= fr)
    worst = cur = 0
    for p in lookback:
        cur = 0 if p in vs else cur + 1
        worst = max(worst, cur)
    frac = n_valid / PRE_ENTRY_SESSIONS
    passes = (frac >= PRE_ENTRY_MIN_VALID_FRAC) and (worst <= PRE_ENTRY_MAX_CONSEC_MISSING)
    # counterfactual: apply the rule ONLY over sessions at/after first listing
    eff = [p for p in lookback if np.isfinite(fr) and p >= fr]
    if eff:
        nv2 = sum(1 for p in eff if p in vs)
        w2 = c2_ = 0
        for p in eff:
            c2_ = 0 if p in vs else c2_ + 1
            w2 = max(w2, c2_)
        frac2 = nv2 / len(eff)
        passes_adj = (frac2 >= PRE_ENTRY_MIN_VALID_FRAC) and (w2 <= PRE_ENTRY_MAX_CONSEC_MISSING)
    else:
        frac2, passes_adj = np.nan, False
    o2_rows.append((r.cp_flag, frac, worst, passes, n_prelist, n_listed_unquoted,
                    n_norow_postlist, len(eff), frac2, passes_adj))

o2 = pd.DataFrame(o2_rows, columns=[
    'cp_flag', 'pre_valid_frac', 'worst_gap', 'passes_asis', 'n_prelist',
    'n_listed_unquoted', 'n_norow_postlist', 'n_eff_sessions', 'pre_valid_frac_adj',
    'passes_adj'])

o2_report = {}
for side, lbl in [('C', 'calls'), ('P', 'puts')]:
    s = o2[o2['cp_flag'] == side]
    if len(s) == 0:
        continue
    fail = s[~s['passes_asis']]
    # attribute each FAILING candidate to its dominant cause
    dom_prelist = int((fail['n_prelist'] > fail['n_listed_unquoted']).sum())
    dom_unquoted = int((fail['n_listed_unquoted'] >= fail['n_prelist']).sum())
    rescued = int((~s['passes_asis'] & s['passes_adj']).sum())
    o2_report[lbl] = {
        'n_selected': len(s),
        'pct_pass_rule_asis': float(s['passes_asis'].mean() * 100),
        'pct_fail_rule_asis': float((~s['passes_asis']).mean() * 100),
        'n_fail': int(len(fail)),
        'among_failures_pct_dominated_by_NOT_YET_LISTED': float(dom_prelist / max(len(fail), 1) * 100),
        'among_failures_pct_dominated_by_LISTED_BUT_UNQUOTED': float(dom_unquoted / max(len(fail), 1) * 100),
        'mean_lookback_sessions_before_first_listing': float(s['n_prelist'].mean()),
        'mean_lookback_sessions_listed_but_unquoted': float(s['n_listed_unquoted'].mean()),
        'mean_lookback_sessions_norow_after_listing': float(s['n_norow_postlist'].mean()),
        'pct_pass_rule_ADJUSTED_from_listing_date': float(s['passes_adj'].mean() * 100),
        'n_rescued_by_listing_adjustment': rescued,
        'pct_of_all_selected_rescued': float(rescued / max(len(s), 1) * 100),
        'pre_valid_frac_asis': dist(s['pre_valid_frac']),
        'pre_valid_frac_adjusted': dist(s['pre_valid_frac_adj']),
        'n_prelist_sessions': dist(s['n_prelist']),
    }
    r = o2_report[lbl]
    print(f'\n  {lbl}: n={r["n_selected"]:,}')
    print(f'    rule as-is:   pass {r["pct_pass_rule_asis"]:.2f}%  fail {r["pct_fail_rule_asis"]:.2f}% '
          f'({r["n_fail"]:,} candidates)')
    print(f'    of those failures: NOT-YET-LISTED dominant {r["among_failures_pct_dominated_by_NOT_YET_LISTED"]:.2f}%  '
          f'| LISTED-BUT-UNQUOTED dominant {r["among_failures_pct_dominated_by_LISTED_BUT_UNQUOTED"]:.2f}%')
    print(f'    mean lookback sessions before first listing: {r["mean_lookback_sessions_before_first_listing"]:.2f} of {PRE_ENTRY_SESSIONS}')
    print(f'    mean lookback sessions listed-but-unquoted:  {r["mean_lookback_sessions_listed_but_unquoted"]:.2f}')
    print(f'    rule ADJUSTED to run only from listing date: pass {r["pct_pass_rule_ADJUSTED_from_listing_date"]:.2f}% '
          f'(rescues {r["n_rescued_by_listing_adjustment"]:,} = {r["pct_of_all_selected_rescued"]:.2f}% of all selected)')


# ==========================================================================
# O1 - COMBINED 10% CEILING + DYNAMIC CAP
# ==========================================================================
print('\n' + '=' * 96)
print('O1 - COMBINED CASE: 10% ceiling AND dynamic cap')
print('=' * 96)


def trailing(oid, ep, n):
    vd = vol_by_oid.get(oid, {})
    vals = [vd.get(ep - k, np.nan) for k in range(0, n)]
    vals = [v for v in vals if np.isfinite(v)]
    return (float(np.sum(vals)) if vals else 0.0), (float(np.mean(vals)) if vals else 0.0)


tv = [trailing(r.optionid, r.entry_pos, 20) for r in sel.itertuples(index=False)]
sel['vol5_sum'] = [trailing(r.optionid, r.entry_pos, 5)[0] for r in sel.itertuples(index=False)]
sel['vol20_avg'] = [t[1] for t in tv]


def simulate(bb, nav, dynamic):
    tgt = (VEGA_CAP_FRAC * nav) / MAX_POSITIONS
    vcap_agg, gcap, slot = VEGA_CAP_FRAC * nav, GAMMA_CAP_FRAC * nav, nav / MAX_POSITIONS
    cc = {k: 0 for k in ['premium', 'rounding_up_forced', 'volume', 'oi', 'sector', 'vega', 'gamma']}
    days, considered, percon = [], 0, []
    for date_d, g in bb.groupby('date_d', sort=True):
        g = g.sort_values('PERMNO')
        considered += len(g)
        filled, sec, av, ag, prem = 0, {}, 0.0, 0.0, []
        for r in g.itertuples(index=False):
            if filled >= MAX_POSITIONS:
                break
            vp = r.vega_pt
            if not (np.isfinite(vp) and vp > 0):
                continue
            ideal = tgt / vp
            n = max(1, int(round(ideal)))
            if ideal < 0.5:
                cc['rounding_up_forced'] += 1
            one = 100.0 * r.best_offer
            if one > PREMIUM_SKIP_MULT * slot:
                cc['premium'] += 1
                continue
            oi_v = float(np.nan_to_num(r.open_interest, nan=0.0))
            if dynamic:
                ocap = np.floor(DYN_OI_FRAC * oi_v)
                vcap = np.floor(DYN_VOL20AVG_FRAC * float(np.nan_to_num(r.vol20_avg, nan=0.0)))
            else:
                ocap = np.floor(FIX_OI_FRAC * oi_v)
                vcap = np.floor(FIX_VOL5SUM_FRAC * float(np.nan_to_num(r.vol5_sum, nan=0.0)))
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
            aff = int(np.floor((PREMIUM_SKIP_MULT * slot) / one))
            if n > aff:
                n = max(1, aff)
                cc['premium'] += 1
            cs = sec.get(r.GSECTOR, 0)
            if cs + 1 > SECTOR_CAP_N:
                cc['sector'] += 1
                continue
            if av + n * vp > vcap_agg:
                cc['vega'] += 1
                continue
            addg = (r.gamma * n * 100.0 * 0.01 * r.S) if np.isfinite(r.gamma) else 0.0
            if ag + addg > gcap:
                cc['gamma'] += 1
                continue
            filled += 1
            sec[r.GSECTOR] = cs + 1
            av += n * vp
            ag += addg
            prem.append(n * one)
            percon.append(n)
        tot = sum(prem)
        days.append({'filled': filled, 'util': tot / nav,
                     'eff': (tot ** 2 / sum(p * p for p in prem)) if prem else 0.0})
    d = pd.DataFrame(days)
    inv = d[d['filled'] >= MIN_BREADTH]
    return {
        'median_filled': float(d['filled'].median()),
        'median_eff_hhi': float(d['eff'].median()),
        'n_invested_days': int(len(inv)),
        'median_invested_day_utilization_pct': float(inv['util'].median() * 100) if len(inv) else 0.0,
        'median_contracts_per_position': float(np.median(percon)) if percon else 0.0,
        'constraints_pct': {k: v / max(considered, 1) * 100 for k, v in cc.items()},
    }


o1_report = {}
for side, lbl in [('C', 'calls'), ('P', 'puts')]:
    s = sel[sel['cp_flag'] == side]
    if len(s) == 0:
        continue
    for nav in NAV_LEVELS:
        for dyn, dlbl in [(False, 'FIXEDCAP'), (True, 'DYNCAP')]:
            o1_report[f'{lbl}|{int(nav)}|{dlbl}'] = simulate(s, nav, dyn)

for k in sorted(o1_report):
    r = o1_report[k]
    print(f'  {k:<28} filled_med={r["median_filled"]:5.1f}  eff_hhi={r["median_eff_hhi"]:5.2f}  '
          f'inv_days={r["n_invested_days"]:>5}  util_med={r["median_invested_day_utilization_pct"]:6.2f}%  '
          f'contracts_med={r["median_contracts_per_position"]:5.0f}')

out = {
    'meta': {
        'generated_by': 'src/56_v4_pending_measurements.py',
        'scope': ('RETURN-BLIND. No return, P&L, forward realized variance, '
                  'outcome-based score, or test statistic computed. Forward-looking '
                  'measurement is QUOTE EXISTENCE and calendar/DTE arithmetic only.'),
        'locked_design': {'dte_band': [DTE_LO, DTE_HI], 'dte_anchor': DTE_TARGET,
                          'spread_ceiling': SPREAD_CEILING, 'nav_primary': NAV_PRIMARY,
                          'max_positions': MAX_POSITIONS, 'exit': 'fixed 30 calendar days'},
        'passA_rows_scanned': int(rows_scanned), 'passB_rows_scanned': int(pscanned),
        'n_selected': int(len(sel)),
    },
    'O1_combined_ceiling_and_cap': o1_report,
    'O2_listing_recency_separation': o2_report,
}
out_json.parent.mkdir(parents=True, exist_ok=True)
with open(out_json, 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f'\n[OK] wrote {out_json}')
print('\n' + '=' * 96)
print('PENDING MEASUREMENTS COMPLETE. No returns/P&L/RV/scores/t-stats computed.')
print('gate_log.md not touched. No V4 trading script written or run.')
print('=' * 96)
