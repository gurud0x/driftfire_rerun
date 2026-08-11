import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import norm

# ---------------------------------------------------------------------------
# prereg_V4.md CONSTRUCTION STEP: C1-C9 eligibility funnel on top of the
# Section 13 raw extract. STILL CONSTRUCTION/FEASIBILITY WORK - no Greeks-
# based hedging simulation, no forward return computation, no P&L. Stops
# after producing the funnel + final candidate table for review.
#
# Input: data/processed/opprcd_v4_contracts_dte8_90.parquet (255,378,123
# rows, DTE [8,90], the widened band that supports full-hold path tracking
# through a fixed 30-calendar-day exit from a [40,60]-DTE entry).
#
# Engineering note: the input is processed in row-group batches (304
# groups, ~1.05M rows each), never loaded whole into one DataFrame -
# repeated OOMs earlier this session at DTE bands far narrower than [8,90]
# (which retains ~29% of the full file) make that the only safe approach
# on this machine's 15.8GB RAM. Cheap, single-row filters (C1,C2,C3,C4,C5,
# C6,C8, delta band) are applied FIRST to shrink the candidate set before
# the expensive rolling-history lookups (C7's trailing volume, Section
# 7.8's listing-history rule) - this changes compute ORDER for efficiency
# only; every step's attrition is reported in the LOCKED logical order
# (C1..C9, then Section 7.7, then Section 7.8), not the compute order.
# ---------------------------------------------------------------------------

DEV_START = pd.Timestamp('2015-01-01')
DEV_END = pd.Timestamp('2021-12-31')
DTE_ENTRY_LO, DTE_ENTRY_HI = 40, 60   # Section 7.1 locked entry band
DTE_TARGET = 50
DELTA_LO, DELTA_HI = 0.40, 0.60
RISK_FREE_RATE = 0.01   # Section 7.2/7.5
Q_DIV = 0.0

IV_SANITY_LO, IV_SANITY_HI = 0.05, 2.00        # C3
MIN_MID_PRICE, MIN_BID_PRICE = 0.50, 0.20      # C4
SPREAD_CEILING = 0.10                          # C5, LOCKED at final reconciliation
MIN_OI = 100                                   # C6
MIN_SAME_DAY_VOL, MIN_5D_VOL = 10, 50          # C7 (eligibility thresholds, NOT the sizing cap)
PRE_ENTRY_LOOKBACK = 20                        # Section 7.8
PRE_ENTRY_MIN_HISTORY = 10                     # Section 7.8, locked history floor
PRE_ENTRY_MIN_VALID_FRAC = 0.90
PRE_ENTRY_MAX_CONSEC_MISSING = 1

project_root = Path(__file__).parent.parent
crsp_path = project_root / 'data' / 'processed' / 'crsp_combined.parquet'
univ_path = project_root / 'data' / 'processed' / 'universe_membership.parquet'
v1_path = project_root / 'data' / 'processed' / 'compression_signal_v1.parquet'
om_names_path = project_root / 'data' / 'raw' / 'optionmetrics' / 'om_security_names.csv'
crsp_names_path = project_root / 'data' / 'raw' / 'crsp' / 'crsp_names.parquet'
link_path = project_root / 'data' / 'raw' / 'compustat' / 'ccm_link_gics.csv'
rdq_path = project_root / 'data' / 'raw' / 'compustat' / 'rdq_pull_fundq_2014_2026.parquet'
raw_path = project_root / 'data' / 'processed' / 'opprcd_v4_contracts_dte8_90.parquet'
out_json = project_root / 'results' / '60_v4_c1c9_funnel.json'
out_parquet = project_root / 'data' / 'processed' / 'v4_entry_candidates_selected.parquet'

_MAX_RG_TEST = int(os.environ.get('V60_MAX_RG_TEST', '0'))

print('=' * 96)
print('V4 C1-C9 ELIGIBILITY FUNNEL - construction/feasibility only, no P&L, no hedging sim')
print('=' * 96)


def as_cusip8(s):
    return (s.astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            .str.upper().str[:8].str.zfill(8))


# ==========================================================================
# 0. SHARED SETUP - bridge, universe, CRSP, CCM link, trading calendar
# ==========================================================================
print('\n' + '-' * 96)
print('0. SHARED SETUP')
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
base_v1 = base_v1.rename(columns={'DlyCalDt': 'date'})
assert len(base_v1) == 1_733_857
print(f"shared base universe: {len(base_v1):,} stock-days [reference: 1,733,857]")

dvol = pd.read_parquet(crsp_path, columns=['PERMNO', 'DlyCalDt', 'DlyPrcVol'])
dvol = dvol.drop_duplicates(['PERMNO', 'DlyCalDt']).rename(columns={'DlyCalDt': 'date'})
bd = base_v1.merge(dvol, on=['PERMNO', 'date'], how='left')
bd['c8_ok'] = bd.groupby('date')['DlyPrcVol'].rank(pct=True) >= 0.50
base_key = bd[['PERMNO', 'date', 'decile', 'c8_ok']].copy()
print(f"C8 (dvol >= same-day median) base rate: {base_key['c8_ok'].mean() * 100:.2f}%")

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
print(f"bridge: {len(bridge):,} pairs, {bridge['secid'].nunique():,} unique secid")

# CRSP close: a compact (PERMNO,date)->close dict for the O(1) join used in
# Step 1 - DEV window only (fixed-exit design guarantees any valid trade's
# entire hold path stays within DEV, per Section 5.3's own "not entered if
# exit falls after DEV_END" rule, so no data past DEV_END is ever needed).
px = pd.read_parquet(crsp_path, columns=['PERMNO', 'DlyCalDt', 'DlyClose'])
px = px[px['PERMNO'].isin(ever)]
px = px[(px['DlyCalDt'] >= DEV_START) & (px['DlyCalDt'] <= DEV_END)]
px = px.drop_duplicates(['PERMNO', 'DlyCalDt'], keep='first')
crsp_close = px.set_index(['PERMNO', 'DlyCalDt'])['DlyClose']
crsp_dates_by_permno = px.groupby('PERMNO')['DlyCalDt'].apply(lambda s: frozenset(s.to_numpy()))
print(f"CRSP close index: {len(crsp_close):,} (PERMNO,date) pairs")

cal_all = pd.DatetimeIndex(sorted(px['DlyCalDt'].unique()))
cal_pos_of = pd.Series(np.arange(len(cal_all)), index=cal_all)
print(f"trading calendar (DEV): {len(cal_all):,} sessions")

link = pd.read_csv(link_path)
link = link[link['LINKTYPE'].isin(['LC', 'LU']) & link['LINKPRIM'].isin(['P', 'C'])]
link = link[link['LPERMNO'].notna()].copy()
link['linkdt'] = pd.to_datetime(link['LINKDT'])
link['linkend'] = pd.to_datetime(link['LINKENDDT'].replace('E', pd.NaT), errors='coerce')
link['linkend'] = link['linkend'].fillna(pd.Timestamp('2262-01-01'))
link['lpermno'] = link['LPERMNO'].astype(int)
link['gvkey'] = link['gvkey'].astype(int)
lk = link[['lpermno', 'gvkey', 'linkdt', 'linkend', 'LINKPRIM']].copy()

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
print(f"usable (gvkey, rdq) pairs: {len(rdq):,}")

secid_to_permno = bridge.set_index('secid')['PERMNO'].to_dict()


# ==========================================================================
# STEP 1 + partial STEP 2 - stream the raw extract in row-group batches:
# join PERMNO+S for ALL rows (join-rate measured on the full [8,90] extract,
# per the request), retain only DTE[40,60] rows with a valid join, and
# apply the cheap per-row funnel checks (C1,C2,C3,C4,C5,C6,C8,delta band)
# immediately to shrink the candidate set before the expensive lookups.
# ==========================================================================
print('\n' + '-' * 96)
print('STEP 1: UNDERLYING JOIN (measured on the FULL [8,90] extract, all rows)')
print('-' * 96)

pf = pq.ParquetFile(raw_path)
n_rg = pf.num_row_groups if not _MAX_RG_TEST else min(_MAX_RG_TEST, pf.num_row_groups)
print(f"row groups to process: {n_rg} of {pf.num_row_groups}")

t0 = time.time()
join_total = 0
join_matched = 0
nonmatch_by_year = {}
nonmatch_permno_sample = set()

cand_parts = []  # DTE[40,60] rows surviving cheap filters, with S/Greeks attached

for rg_i in range(n_rg):
    batch = pf.read_row_group(rg_i).to_pandas()
    join_total += len(batch)

    permno = batch['secid'].map(secid_to_permno)
    has_permno = permno.notna()
    batch = batch[has_permno].copy()
    batch['PERMNO'] = permno[has_permno].astype('int64')

    keys = list(zip(batch['PERMNO'].to_numpy(), batch['date'].to_numpy()))
    S = crsp_close.reindex(keys).to_numpy()
    matched = np.isfinite(S)
    join_matched += int(matched.sum())

    if (~matched).any():
        miss = batch.loc[~matched]
        for yr, cnt in miss['date'].dt.year.value_counts().items():
            nonmatch_by_year[yr] = nonmatch_by_year.get(yr, 0) + int(cnt)
        if len(nonmatch_permno_sample) < 200:
            nonmatch_permno_sample.update(miss['PERMNO'].unique().tolist()[:50])

    batch['S'] = S.astype('float32')
    batch = batch[matched]

    # restrict to base universe (decile 6-8, in_universe) + attach C8
    batch = batch.merge(base_key, on=['PERMNO', 'date'], how='inner')

    cand = batch[(batch['dte'] >= DTE_ENTRY_LO) & (batch['dte'] <= DTE_ENTRY_HI)].copy()
    if len(cand) == 0:
        if rg_i == 0 or (rg_i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  rg {rg_i + 1:>4}/{n_rg}: join_total {join_total:>13,}  "
                  f"cand_kept 0  ({el / 60:.1f} min)")
        continue

    K = cand['strike_price'].to_numpy(dtype=float) / 1000.0  # OM x1000 convention
    T = cand['dte'].to_numpy(dtype=float) / 365.0
    Sv = cand['S'].to_numpy(dtype=float)
    bid = cand['best_bid'].to_numpy(dtype=float)
    off = cand['best_offer'].to_numpy(dtype=float)
    mid = (bid + off) / 2.0
    sig = cand['impl_volatility'].to_numpy(dtype=float)
    is_call = (cand['cp_flag'] == 'C').to_numpy()

    c1 = np.isfinite(bid) & np.isfinite(off) & (bid > 0) & (off > bid)
    disc = np.exp(-RISK_FREE_RATE * T)
    lb = np.where(is_call, np.maximum(Sv - K * disc, 0.0), np.maximum(K * disc - Sv, 0.0))
    ub = np.where(is_call, Sv, K * disc)
    c2 = c1 & (mid >= lb - 1e-9) & (mid <= ub + 1e-9)
    c3 = np.isfinite(sig) & (sig >= IV_SANITY_LO) & (sig <= IV_SANITY_HI)
    c4 = (mid >= MIN_MID_PRICE) & (bid >= MIN_BID_PRICE)
    with np.errstate(all='ignore'):
        rel = np.where(mid > 0, (off - bid) / mid, np.nan)
    c5 = rel <= SPREAD_CEILING
    oi = cand['open_interest'].to_numpy(dtype=float)
    c6 = oi >= MIN_OI

    with np.errstate(all='ignore'):
        sq = sig * np.sqrt(T)
        d1 = (np.log(Sv / K) + (RISK_FREE_RATE + 0.5 * sig ** 2) * T) / sq
        d2 = d1 - sq
        dcall = norm.cdf(d1)
        gamma = norm.pdf(d1) / (Sv * sig * np.sqrt(T))
        vega = Sv * norm.pdf(d1) * np.sqrt(T)
        theta_call = (-(Sv * norm.pdf(d1) * sig) / (2 * np.sqrt(T))
                     - RISK_FREE_RATE * K * disc * norm.cdf(d2))
        theta_put = (-(Sv * norm.pdf(d1) * sig) / (2 * np.sqrt(T))
                    + RISK_FREE_RATE * K * disc * norm.cdf(-d2))
    dvalid = np.isfinite(sig) & (sig > 0) & np.isfinite(d1)
    delta = np.where(is_call, dcall, dcall - 1.0)
    ad = np.abs(np.where(dvalid, delta, np.nan))
    dband = dvalid & (ad >= DELTA_LO) & (ad <= DELTA_HI)
    theta = np.where(is_call, theta_call, theta_put)

    cand['c1'] = c1
    cand['c2'] = c2
    cand['c3'] = c3
    cand['c4'] = c4
    cand['c5'] = c5
    cand['c6'] = c6
    cand['delta_band'] = dband
    cand['rel_spread'] = rel.astype('float32')
    cand['delta'] = np.where(dvalid, delta, np.nan).astype('float32')
    cand['abs_delta'] = ad.astype('float32')
    cand['gamma'] = np.where(dvalid, gamma, np.nan).astype('float32')
    cand['vega'] = np.where(dvalid, vega, np.nan).astype('float32')
    cand['theta'] = np.where(dvalid, theta, np.nan).astype('float32')

    cheap_ok = c1 & c2 & c3 & c4 & c5 & c6 & dband & cand['c8_ok'].to_numpy()
    cand['cheap_ok'] = cheap_ok
    cand_parts.append(cand)

    if rg_i == 0 or (rg_i + 1) % 25 == 0:
        el = time.time() - t0
        kept = sum(len(p) for p in cand_parts)
        print(f"  rg {rg_i + 1:>4}/{n_rg}: join_total {join_total:>13,}  "
              f"cand_kept {kept:>12,}  ({el / 60:.1f} min)")

el = time.time() - t0
print(f"\nStep 1 join complete: {join_total:,} rows, matched {join_matched:,} "
      f"({join_matched / join_total * 100:.4f}%), {el / 60:.2f} min")
if nonmatch_by_year:
    print("  non-match by year:", dict(sorted(nonmatch_by_year.items())))
    print(f"  sample of non-matched PERMNOs (up to 200): "
          f"{sorted(nonmatch_permno_sample)[:20]} ...")

cand_all = pd.concat(cand_parts, ignore_index=True) if cand_parts else pd.DataFrame()
del cand_parts
print(f"\nSTEP 2 (entry DTE [{DTE_ENTRY_LO},{DTE_ENTRY_HI}] window) candidate count "
      f"(after underlying join, before C1-C9): {len(cand_all):,}")
print(f"  calls: {int((cand_all['cp_flag'] == 'C').sum()):,}   "
      f"puts: {int((cand_all['cp_flag'] == 'P').sum()):,}")


# ==========================================================================
# STEP 4a: C1-C6, C8, delta-band attrition report (already computed above,
# cheap/single-row conditions applied in one pass for efficiency; reported
# here in the LOCKED C1..C9 order). Cumulative AND, calls/puts separately.
# ==========================================================================
print('\n' + '=' * 96)
print('STEP 4: C1-C9 SEQUENTIAL FUNNEL (locked order; C7/C9/Sec.7.7/Sec.7.8 finish below)')
print('=' * 96)


def flag_char(is_call):
    return 'C' if is_call else 'P'


def report_step(df, mask_col, label, cum_mask):
    new_cum = cum_mask & df[mask_col].to_numpy()
    for side, flag in [('calls', True), ('puts', False)]:
        s = (df['cp_flag'] == flag_char(flag)).to_numpy()
        n_before = int((cum_mask & s).sum())
        n_after = int((new_cum & s).sum())
        print(f"  {label:<38} [{side:<5}] {n_before:>10,} -> {n_after:>10,}  "
              f"(-{n_before - n_after:,}, {(n_after / max(n_before, 1)) * 100:5.1f}% survive)")
    return new_cum


n0 = len(cand_all)
cum = np.ones(n0, dtype=bool)
print(f"Starting pool (entry DTE [{DTE_ENTRY_LO},{DTE_ENTRY_HI}], underlying joined): calls "
      f"{int((cand_all['cp_flag'] == 'C').sum()):,}  puts {int((cand_all['cp_flag'] == 'P').sum()):,}")

cum = report_step(cand_all, 'c1', 'C1 valid two-sided quote', cum)
cum = report_step(cand_all, 'c2', 'C2 no static-arbitrage', cum)
cum = report_step(cand_all, 'c3', 'C3 IV sanity [0.05,2.00]', cum)
cum = report_step(cand_all, 'c4', 'C4 minimum option price', cum)
cum = report_step(cand_all, 'c5', 'C5 max spread <=10% (REVISED)', cum)
cum = report_step(cand_all, 'c6', 'C6 minimum open interest >=100', cum)
cum = report_step(cand_all, 'c8_ok', 'C8 underlying dollar volume', cum)
cum = report_step(cand_all, 'delta_band', 'delta band [0.40,0.60] (Sec.7.1)', cum)

cand_all['cum_thru_c8_delta'] = cum
survivors = cand_all[cum].copy()
print(f"\nSurvivors through C1-C6+C8+delta-band: {len(survivors):,} "
      f"(calls {int((survivors['cp_flag'] == 'C').sum()):,}, "
      f"puts {int((survivors['cp_flag'] == 'P').sum()):,})")
print("NOTE: computed in this order for efficiency (cheap single-row checks first, to shrink")
print("the working set before the expensive rolling-history lookups below) - not a reordering")
print("of the LOCKED funnel semantics, which is an unordered AND of all conditions.")


# ==========================================================================
# STEP 4b: PASS 2 - targeted re-read of the wide [8,90] extract, restricted
# to optionids surviving the cheap filters, for C7 (trailing volume) and
# Section 7.8 (listing-history / pre-entry continuity). Established
# reasoning (see chat record): the required lookback for any [40,60]-DTE
# entry candidate (up to 20 trading sessions =~ 28 calendar days) stays
# entirely inside the [8,90] DTE range already captured by Section 13's
# pull, at both the 40-DTE and 60-DTE edges - so this second pass over the
# SAME cached extract (not a new opprcd.csv scan) is sufficient.
# ==========================================================================
print('\n' + '-' * 96)
print('PASS 2 - targeted history lookup for C7 and Section 7.8 (survivor optionids only)')
print('-' * 96)

survivor_oids = set(survivors['optionid'].unique().tolist())
print(f"survivor optionids requiring history: {len(survivor_oids):,}")

t1 = time.time()
hist_parts = []
if len(survivor_oids) == 0:
    print("  no survivors entering Pass 2 - skipping history scan")
    hist = pd.DataFrame(columns=['optionid', 'date', 'valid_q', 'volume'])
else:
    for rg_i in range(n_rg):
        batch = pf.read_row_group(rg_i).to_pandas()
        batch = batch[batch['optionid'].isin(survivor_oids)]
        if len(batch) == 0:
            continue
        batch['valid_q'] = (np.isfinite(batch['best_bid']) & np.isfinite(batch['best_offer']) &
                            (batch['best_bid'] > 0) & (batch['best_offer'] > batch['best_bid']))
        hist_parts.append(batch[['optionid', 'date', 'valid_q', 'volume']])
        if rg_i == 0 or (rg_i + 1) % 50 == 0:
            el = time.time() - t1
            kept = sum(len(p) for p in hist_parts)
            print(f"  rg {rg_i + 1:>4}/{n_rg}: history rows kept {kept:>12,}  ({el / 60:.1f} min)")
    hist = pd.concat(hist_parts, ignore_index=True) if hist_parts else pd.DataFrame(
        columns=['optionid', 'date', 'valid_q', 'volume'])
    del hist_parts
hist['cal_pos'] = cal_pos_of.reindex(hist['date']).to_numpy()
hist = hist.dropna(subset=['cal_pos'])
hist['cal_pos'] = hist['cal_pos'].astype(int)
hist = hist.sort_values(['optionid', 'cal_pos'])
print(f"\nPASS 2 complete: {len(hist):,} history rows for {hist['optionid'].nunique():,} "
      f"optionids, {(time.time() - t1) / 60:.2f} min")

# per-optionid sorted arrays for vectorized lookups
hist_by_oid = {}
for oid, g in hist.groupby('optionid', sort=False):
    hist_by_oid[oid] = (g['cal_pos'].to_numpy(), g['valid_q'].to_numpy(),
                        g['volume'].fillna(0.0).to_numpy())

survivors['entry_pos'] = cal_pos_of.reindex(survivors['date']).to_numpy()
survivors = survivors.dropna(subset=['entry_pos']).copy()
survivors['entry_pos'] = survivors['entry_pos'].astype(int)

c7_ok = np.zeros(len(survivors), dtype=bool)
recent_vol_5d = np.full(len(survivors), np.nan)
n_since_listing = np.full(len(survivors), -1, dtype=int)
pre_entry_pass = np.zeros(len(survivors), dtype=bool)

oid_arr = survivors['optionid'].to_numpy()
ep_arr = survivors['entry_pos'].to_numpy()
same_day_vol = survivors['volume'].to_numpy(dtype=float)

for i in range(len(survivors)):
    oid = oid_arr[i]
    ep = ep_arr[i]
    arr = hist_by_oid.get(oid)
    if arr is None:
        continue
    positions, valid, vol = arr
    # positions strictly before entry (mid-hold/entry-day quote already
    # captured separately by C1 above; this is the PRE-entry lookback)
    idx = np.searchsorted(positions, ep, side='left')
    n_since_listing[i] = idx  # count of rows strictly before entry_pos

    # C7: same-day volume (already have) + trailing 5-session sum
    # (5 most recent rows at/before entry, inclusive, matching the
    # project's established "trailing N sessions incl. current day" convention)
    lo5 = max(0, idx - 4)
    if idx < len(positions) and positions[idx] == ep:
        v5 = vol[lo5:idx + 1].sum()
    else:
        v5 = vol[lo5:idx].sum() + same_day_vol[i]
    recent_vol_5d[i] = v5
    c7_ok[i] = (same_day_vol[i] >= MIN_SAME_DAY_VOL) and (v5 >= MIN_5D_VOL)

    # Section 7.8 pre-entry rule
    if idx < PRE_ENTRY_MIN_HISTORY:
        pre_entry_pass[i] = False
        continue
    W = min(PRE_ENTRY_LOOKBACK, idx)
    lo = idx - W
    wpos, wvalid = positions[lo:idx], valid[lo:idx]
    n_valid = int(wvalid.sum())
    worst = cur = 0
    for k in range(len(wvalid)):
        cur = 0 if wvalid[k] else cur + 1
        worst = max(worst, cur)
    frac = n_valid / max(W, 1)
    pre_entry_pass[i] = (frac >= PRE_ENTRY_MIN_VALID_FRAC) and (worst <= PRE_ENTRY_MAX_CONSEC_MISSING)

survivors['c7'] = c7_ok
survivors['recent_vol_5d'] = recent_vol_5d
survivors['n_since_listing'] = n_since_listing
survivors['pre_entry_ok'] = pre_entry_pass

cum2 = np.ones(len(survivors), dtype=bool)
cum2 = report_step(survivors, 'c7', 'C7 recent option volume (fixed thresholds)', cum2)


# ==========================================================================
# STEP 4c: C9 - underlying data completeness (CRSP close on entry AND every
# hold session; exit session must fall inside DEV, Section 5.3). Row
# EXISTENCE only, matching the project's return-blind convention for this
# check - no price VALUE beyond the entry close already joined in Step 1.
# ==========================================================================
n_hold_arr = np.zeros(len(survivors), dtype=int)
exit_ok_arr = np.zeros(len(survivors), dtype=bool)
c9_ok = np.zeros(len(survivors), dtype=bool)

ep2 = survivors['entry_pos'].to_numpy()
permno2 = survivors['PERMNO'].to_numpy()
cal_np = cal_all.to_numpy()
DEV_END_POS = int(cal_pos_of.get(cal_all[cal_all <= DEV_END][-1]))

for i in range(len(survivors)):
    ep = ep2[i]
    exit_target = cal_np[ep] + pd.Timedelta(days=30)
    exit_pos = np.searchsorted(cal_np, exit_target, side='right') - 1
    n_hold = exit_pos - ep
    n_hold_arr[i] = n_hold
    if exit_pos > DEV_END_POS or n_hold <= 0:
        exit_ok_arr[i] = False
        continue
    exit_ok_arr[i] = True
    have = crsp_dates_by_permno.get(permno2[i])
    if have is None:
        c9_ok[i] = False
        continue
    hold_dates = cal_np[ep + 1: exit_pos + 1]
    c9_ok[i] = exit_ok_arr[i] and all(d in have for d in hold_dates)

survivors['exit_in_dev'] = exit_ok_arr
survivors['n_hold'] = n_hold_arr
survivors['c9'] = c9_ok & exit_ok_arr
cum2 = report_step(survivors, 'c9', 'C9 underlying data completeness + exit-in-DEV', cum2)
survivors['cum_thru_c9'] = cum2


# ==========================================================================
# STEP 4d: Section 7.7 earnings-window exclusion
#
# FLAGGED DISCREPANCY, not silently resolved: Section 7.7's literal text
# locks the exclusion window as "entry date through expiration date,
# inclusive." That text predates the final-reconciliation exit-rule
# revision (Section 5.3): the position is no longer held to expiration,
# only through the FIXED 30-CALENDAR-DAY EXIT. At the locked [40,60] entry
# band, expiration sits 10-30+ days PAST the actual exit, so "through
# expiration" would exclude candidates for earnings risk they are never
# actually exposed to (the position is sold before expiration). This
# script uses (entry, exit] - the position's ACTUAL risk window under the
# locked exit design, matching the same construction already used in
# src/56's earlier feasibility measurements - not (entry, expiration].
# THIS IS A DISCOVERED TEXTUAL STALENESS IN prereg_V4.md SECTION 7.7,
# parallel to the hold-to-expiry language already caught and fixed
# elsewhere in the final reconciliation, but not previously caught in
# Section 7.7 specifically. Flagged for confirmation, not silently patched
# into the locked document without sign-off.
# ==========================================================================
print('\n' + '-' * 96)
print('STEP 4d: Section 7.7 earnings-window exclusion, evaluated over (entry, exit]')
print('  *** FLAGGED: Sec.7.7 literal text says "through expiration" - stale, see comment ***')
print('-' * 96)

sd = survivors[['PERMNO', 'date']].drop_duplicates()
m = sd.merge(lk, left_on='PERMNO', right_on='lpermno', how='left')
m = m[((m['date'] >= m['linkdt']) & (m['date'] <= m['linkend'])) | m['lpermno'].isna()]
m['_pr'] = np.where(m['LINKPRIM'] == 'P', 0, 1)
m = m.sort_values(['PERMNO', 'date', '_pr', 'gvkey']).drop_duplicates(['PERMNO', 'date'], keep='first')
survivors = survivors.merge(m[['PERMNO', 'date', 'gvkey']], on=['PERMNO', 'date'], how='left')

exit_dates = cal_np[np.clip(ep2 + n_hold_arr, 0, len(cal_np) - 1)]
gv = survivors['gvkey'].to_numpy()
d0 = survivors['date'].to_numpy()
earn_in = np.zeros(len(survivors), dtype=bool)
has_hist = np.zeros(len(survivors), dtype=bool)
for g in np.unique(gv[~np.isnan(gv)]):
    arr = rdq_by_gvkey.get(int(g))
    sel = np.where(gv == g)[0]
    if arr is None or len(arr) == 0:
        continue
    has_hist[sel] = True
    earn_in[sel] = (np.searchsorted(arr, exit_dates[sel], side='right') >
                    np.searchsorted(arr, d0[sel], side='right'))
survivors['earn_ok'] = has_hist & (~earn_in) & survivors['gvkey'].notna()
cum3 = report_step(survivors, 'earn_ok', 'Sec.7.7 earnings-window exclusion (entry,exit]', cum2)


# ==========================================================================
# STEP 4e: Section 7.8 pre-entry rule (already computed in Pass 2, LOCKED
# listing-adjusted form with 10-session floor) - final funnel step.
# ==========================================================================
cum3 = report_step(survivors, 'pre_entry_ok', 'Sec.7.8 pre-entry rule (listing-adjusted, floor=10)', cum3)
survivors['cum_final'] = cum3

print(f"\n{'=' * 96}\nFULL FUNNEL SURVIVORS: {int(cum3.sum()):,} "
      f"(calls {int((cum3 & (survivors['cp_flag'] == 'C').to_numpy()).sum()):,}, "
      f"puts {int((cum3 & (survivors['cp_flag'] == 'P').to_numpy()).sum()):,})\n{'=' * 96}")


# ==========================================================================
# STEP 4f: MID-HOLD OPTION-QUOTE PATH COMPLETENESS - DIAGNOSTIC ONLY, NOT
# A DROP. FLAGGED INTERPRETATION, not silently applied as attrition.
#
# Section 7.8's LOCKED mid-hold table is explicit: "A trade is NEVER
# dropped solely because quotes went missing mid-hold" - 1 missing day
# carries the last valid delta forward, 2+ consecutive triggers an early
# exit at the first subsequent valid bid, and a bid that never returns
# settles at intrinsic. None of those are eligibility drops; they are
# EXECUTION-TIME fallback rules for a hedging simulation, which this
# construction step explicitly does not run (owner instruction: "no
# Greeks-based hedging simulation"). Treating this as a hard funnel filter
# would silently contradict the locked "never dropped" rule. This script
# therefore MEASURES mid-hold path completeness as a diagnostic annotation
# on each surviving candidate - not as an attrition step - and flags this
# interpretation explicitly for confirmation rather than guessing which of
# (a) hard drop or (b) diagnostic-only the requested "complete daily
# IV/delta path requirement" step meant.
# ==========================================================================
print('\n' + '-' * 96)
print('STEP 4f (DIAGNOSTIC ONLY, NOT APPLIED AS A DROP): mid-hold option-quote path completeness')
print('  *** FLAGGED INTERPRETATION - see script comment above this block ***')
print('-' * 96)

final = survivors[cum3].copy()
full_path = np.zeros(len(final), dtype=bool)
worst_gap = np.zeros(len(final), dtype=int)
ep3 = final['entry_pos'].to_numpy()
nh3 = final['n_hold'].to_numpy()
oid3 = final['optionid'].to_numpy()
for i in range(len(final)):
    arr = hist_by_oid.get(oid3[i])
    if arr is None:
        continue
    positions, valid, vol = arr
    ep, nh = ep3[i], nh3[i]
    hold_positions = set(range(ep + 1, ep + nh + 1))
    have_positions = positions[(positions > ep) & (positions <= ep + nh)]
    have_valid = valid[(positions > ep) & (positions <= ep + nh)]
    missing = len(hold_positions) - len(have_positions)
    n_invalid = int((~have_valid).sum()) if len(have_valid) else 0
    full_path[i] = (missing == 0) and (n_invalid == 0)
    # worst gap: longest run of (missing row OR invalid quote) across the hold
    present_valid_by_pos = {p: v for p, v in zip(have_positions, have_valid)}
    cur = w = 0
    for p in range(ep + 1, ep + nh + 1):
        ok = present_valid_by_pos.get(p, False)
        cur = 0 if ok else cur + 1
        w = max(w, cur)
    worst_gap[i] = w

final['mid_hold_full_path'] = full_path
final['mid_hold_worst_gap'] = worst_gap
for side, flag in [('calls', 'C'), ('puts', 'P')]:
    s = final[final['cp_flag'] == flag]
    print(f"  [{side}] n={len(s):,}  full-path rate: {s['mid_hold_full_path'].mean() * 100:.2f}%  "
          f"worst-gap median {int(s['mid_hold_worst_gap'].median()) if len(s) else 0}, "
          f"p95 {int(np.percentile(s['mid_hold_worst_gap'], 95)) if len(s) else 0}")
print("(Reported for information only - no candidate above was dropped for this.)")


# ==========================================================================
# STEP 5: TIE-BREAK SELECTION - prereg_V4.md Section 7.1, exactly as
# written, applied verbatim (confirmed against the locked document, not
# improvised):
#   1. smallest |delta - 0.50|
#   2. if tied within 0.01 (rounded to 2 decimals), smallest |DTE - 50|
#   3. if still tied, smallest relative spread (ask-bid)/mid
#   4. if still tied, largest open_interest
#   5. if still tied, smallest optionid
# One contract per (PERMNO, entry date, side).
# ==========================================================================
print('\n' + '-' * 96)
print('STEP 5: TIE-BREAK SELECTION (Sec.7.1, applied exactly as locked)')
print('-' * 96)

final['k1'] = (final['abs_delta'] - 0.50).abs().round(2)
final['k2'] = (final['dte'] - DTE_TARGET).abs()
final['k3'] = final['rel_spread'].fillna(np.inf)
final['k4'] = -final['open_interest'].fillna(0.0)
final = final.sort_values(['PERMNO', 'date', 'cp_flag', 'k1', 'k2', 'k3', 'k4', 'optionid'])
selected = final.drop_duplicates(['PERMNO', 'date', 'cp_flag'], keep='first').copy()
n_ties_broken = len(final) - len(selected)
print(f"raw funnel survivors: {len(final):,}   after tie-break: {len(selected):,}   "
      f"(multi-contract days resolved: {n_ties_broken:,})")


# ==========================================================================
# STEP 6: FINAL OUTPUT - one row per (underlying, entry_date, side), by
# year, calls/puts separately - format matching the earlier feasibility
# audits (src/50, src/53, src/55, src/56) for direct comparability.
# ==========================================================================
print('\n' + '=' * 96)
print('STEP 6: FINAL SELECTED ENTRY CANDIDATES')
print('=' * 96)

by_year = {}
for side, flag in [('calls', 'C'), ('puts', 'P')]:
    s = selected[selected['cp_flag'] == flag]
    print(f"\n  --- {side.upper()} : {len(s):,} selected entry candidates ---")
    print(f"    unique underlyings: {s['PERMNO'].nunique():,}   "
          f"unique contracts (optionid): {s['optionid'].nunique():,}")
    yr = s['date'].dt.year.value_counts().sort_index()
    print(f"    by year: {yr.to_dict()}")
    print(f"    by decile: {s['decile'].value_counts().sort_index().to_dict()}")
    by_year[side] = {int(k): int(v) for k, v in yr.items()}

out_cols = ['PERMNO', 'secid', 'optionid', 'date', 'exdate', 'dte', 'cp_flag',
           'strike_price', 'best_bid', 'best_offer', 'S', 'impl_volatility',
           'delta', 'gamma', 'vega', 'theta', 'open_interest', 'volume',
           'recent_vol_5d', 'n_since_listing', 'n_hold', 'mid_hold_full_path',
           'mid_hold_worst_gap', 'decile']
selected[out_cols].to_parquet(out_parquet, index=False)
print(f"\n[OK] wrote {out_parquet}  ({len(selected):,} rows)")

summary = {
    'meta': {
        'generated_by': 'src/60_v4_c1c9_funnel.py',
        'scope': 'construction/feasibility only - no hedging simulation, no forward return, no P&L',
        'input': str(raw_path.name),
        'flagged_interpretations': [
            "Sec.7.7 earnings window evaluated over (entry, exit] not (entry, expiration] "
            "- Sec.7.7's literal text predates the fixed-30-day-exit revision; flagged, "
            "not silently patched into the locked document.",
            "Mid-hold option-quote path completeness computed as a DIAGNOSTIC only, not "
            "an attrition step - Sec.7.8 explicitly forbids dropping trades for mid-hold "
            "quote gaps (carry-forward/early-exit fallbacks instead); flagged rather than "
            "silently either contradicting that rule or omitting the requested measurement.",
        ],
    },
    'step1_underlying_join': {
        'rows_scanned': int(join_total), 'rows_matched': int(join_matched),
        'match_rate_pct': join_matched / join_total * 100,
        'nonmatch_by_year': {int(k): int(v) for k, v in nonmatch_by_year.items()},
    },
    'step2_entry_dte_window_count': int(len(cand_all)),
    'final_funnel_survivors': int(cum3.sum()),
    'selected_after_tiebreak': {'total': int(len(selected)), 'by_year': by_year},
}
import json
with open(out_json, 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(f"[OK] wrote {out_json}")

print('\n' + '=' * 96)
print('CONSTRUCTION STEP COMPLETE. No Greeks-based hedging simulation, no forward return,')
print('no P&L computed. Stopping for review before any trading logic is written.')
print('gate_log.md not touched.')
print('=' * 96)
