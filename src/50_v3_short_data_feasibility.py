import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

# ---------------------------------------------------------------------------
# V3-SHORT DATA FEASIBILITY AUDIT - RETURN-BLIND COVERAGE CHECK ONLY.
#
# This is NOT a V3-short pre-registration and does NOT build one. It does
# NOT authorize building V3-short. It does NOT change V4's locked
# 30-calendar-day primary design (results/prereg_V4.md, DRAFT). It is a
# standalone feasibility measurement, in the same spirit as
# src/47_v3_data_feasibility.py, asking one question only: can actual
# opprcd contract-level quotes support a ~20-calendar-day short-horizon
# test, given that the standardized surface's 10-day node has only 4.30%
# coverage (prereg_V3.md Section 3.3-F) and that a 30d/10d interpolation to
# 20 days would inherit that same sparse 10-day node?
#
# HARD GUARANTEE, ENFORCED MECHANICALLY, NOT JUST STATED: this script never
# loads DlyRet anywhere. The only CRSP column pulled beyond keys is
# DlyClose (today's price, needed to price/delta a contract as of its own
# quote date - not a future outcome). No realized variance, no option
# return, no P&L, no regression coefficient is computed anywhere below.
# The "future trading calendar" check (Section 7) verifies DATE/ROW
# EXISTENCE only - whether price rows exist through the horizon - never
# reads a return or price VALUE for that purpose.
#
# Output: results/50_v3_short_data_feasibility.json (full numeric detail),
# printed console tables (this run's log is the audit trail).
# gate_log.md is NOT touched - this is not a gate decision.
# ---------------------------------------------------------------------------

DEV_START = pd.Timestamp('2015-01-01')
DEV_END = pd.Timestamp('2021-12-31')
TARGET_DTE = 20
PRIMARY_DTE_LO, PRIMARY_DTE_HI = 18, 22
DIAG_DTE_LO, DIAG_DTE_HI = 15, 25
# Extraction band widened beyond the 15-25 diagnostic band so that a
# trailing-5-TRADING-day volume lookback (~7-9 calendar days, weekends/
# holidays included) is never truncated at the band edge. See Section 3.
SCAN_DTE_LO, SCAN_DTE_HI = 5, 35
RECENT_VOL_TDAYS = 5  # trailing window INCLUDING the selection day itself

# Declared assumptions - reused verbatim from src/42 (A1, A3) and the V4
# draft (results/prereg_V4.md Section 7.2/9.1), not re-derived here.
RISK_FREE_RATE = 0.01
Q_DIV = 0.0
IV_LO, IV_HI = 0.0001, 5.0
NEWTON_MAX_ITER = 60
NEWTON_PRICE_TOL_ABS = 0.005     # $0.005 residual
NEWTON_PRICE_TOL_REL = 1e-4      # or 1e-4 of mid, whichever is looser

OPPRCD_CHUNKSIZE = 5_000_000
EARNINGS_WINDOW_TDAYS = 20

project_root = Path(__file__).parent.parent
crsp_path = project_root / 'data' / 'processed' / 'crsp_combined.parquet'
univ_path = project_root / 'data' / 'processed' / 'universe_membership.parquet'
v1_path = project_root / 'data' / 'processed' / 'compression_signal_v1.parquet'
om_names_path = project_root / 'data' / 'raw' / 'optionmetrics' / 'om_security_names.csv'
crsp_names_path = project_root / 'data' / 'raw' / 'crsp' / 'crsp_names.parquet'
link_path = project_root / 'data' / 'raw' / 'compustat' / 'ccm_link_gics.csv'
rdq_path = project_root / 'data' / 'raw' / 'compustat' / 'rdq_pull_fundq_2014_2026.parquet'
out_json = project_root / 'results' / '50_v3_short_data_feasibility.json'

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
print('V3-SHORT DATA FEASIBILITY AUDIT (return-blind coverage check only)')
print('Does NOT authorize V3-short. Does NOT change V4. No returns/RV/P&L computed.')
print('=' * 92)
print(f"\nopprcd source: {opp_path} ({opp_path.stat().st_size / 1e9:.2f} GB)")


def as_cusip8(s):
    return (s.astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            .str.upper().str[:8].str.zfill(8))


def pctiles(arr, label):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {'n': 0}
    q = np.percentile(a, [1, 25, 50, 75, 99])
    return {'n': int(len(a)), 'min': float(a.min()), 'p1': float(q[0]),
            'q1': float(q[1]), 'median': float(q[2]), 'q3': float(q[3]),
            'p99': float(q[4]), 'max': float(a.max()), 'mean': float(a.mean())}


# ==========================================================================
# 1. TRADING CALENDAR (dates only - no DlyRet loaded anywhere in this script)
# ==========================================================================
print('\n' + '-' * 92)
print('1. TRADING CALENDAR')
print('-' * 92)

cal_all = pd.DatetimeIndex(sorted(
    pd.read_parquet(crsp_path, columns=['DlyCalDt'])['DlyCalDt'].unique()))
print(f"Full trading calendar: {len(cal_all):,} sessions, "
      f"{cal_all.min().date()} to {cal_all.max().date()}")
dev_cal = cal_all[(cal_all >= DEV_START) & (cal_all <= DEV_END)]
print(f"DEV sessions: {len(dev_cal):,}")

# n20: trading sessions in (t, t+20 calendar days], purely calendar-date
# arithmetic - no price or return touched.
dev_pos = cal_all.get_indexer(dev_cal)
n20_arr = np.zeros(len(dev_cal), dtype=int)
for i, p in enumerate(dev_pos):
    hi = dev_cal[i] + pd.Timedelta(days=TARGET_DTE)
    future = cal_all[p + 1:]
    n20_arr[i] = int((future <= hi).sum())
n20_by_date = pd.Series(n20_arr, index=dev_cal)
print(f"n20 (20-calendar-day window session count): min={n20_arr.min()}, "
      f"max={n20_arr.max()}, mean={n20_arr.mean():.2f}, "
      f"mode={pd.Series(n20_arr).mode().iloc[0]}")
cal_pos_of = pd.Series(np.arange(len(cal_all)), index=cal_all)


# ==========================================================================
# 2. UNIVERSE - shared base population, identical to src/49's own base_v1
#    (decile 6-8, DEV, V1 compression_ratio defined) so this audit's
#    coverage percentages are directly comparable to prereg_V3.md's own
#    cited 91.45%/4.30% figures. compression_ratio is used ONLY as an
#    existence filter ("signal is defined"), its VALUE is never read.
# ==========================================================================
print('\n' + '-' * 92)
print('2. UNIVERSE - shared base population (matches src/49 base_v1 exactly)')
print('-' * 92)

univ = pd.read_parquet(univ_path)
univ_in = univ[univ['in_universe']][['PERMNO', 'year_month', 'decile']].copy()
ever = set(univ_in['PERMNO'].unique())
print(f"Ever-in-universe PERMNOs (decile 6/7/8, any month): {len(ever):,}")

base_v1 = pd.read_parquet(v1_path, columns=['PERMNO', 'DlyCalDt', 'compression_ratio'])
base_v1 = base_v1[(base_v1['DlyCalDt'] >= DEV_START) & (base_v1['DlyCalDt'] <= DEV_END)]
base_v1 = base_v1[base_v1['compression_ratio'].notna()][['PERMNO', 'DlyCalDt']]
base_v1['year_month'] = base_v1['DlyCalDt'].dt.to_period('M').astype(str)
base_v1 = base_v1.merge(univ_in, on=['PERMNO', 'year_month'], how='inner')
base_v1 = base_v1[['PERMNO', 'DlyCalDt', 'decile']].drop_duplicates(['PERMNO', 'DlyCalDt'])
print(f"shared base universe: {len(base_v1):,} stock-days "
      f"[prereg_V3 Section 6(b) reference population: 1,733,857]")
assert len(base_v1) == 1_733_857, 'base universe row count moved from the locked V3 figure'
base_v1 = base_v1.rename(columns={'DlyCalDt': 'date_d'})


# ==========================================================================
# 3. secid <-> PERMNO BRIDGE (identical construction to src/49 Section 4)
# ==========================================================================
print('\n' + '-' * 92)
print('3. secid <-> PERMNO BRIDGE')
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
print(f"PERMNO->secid bridge pairs: {len(bridge):,}  "
      f"(unique PERMNO: {bridge['PERMNO'].nunique():,}, "
      f"unique secid: {len(secid_whitelist):,})")


# ==========================================================================
# 4. UNDERLYING CLOSE (S) - DlyClose ONLY. DlyRet is never loaded anywhere
#    in this script.
# ==========================================================================
print('\n' + '-' * 92)
print('4. UNDERLYING CLOSE (DlyClose only - DlyRet not loaded in this script)')
print('-' * 92)

px = pd.read_parquet(crsp_path, columns=['PERMNO', 'DlyCalDt', 'DlyClose'])
px = px[px['PERMNO'].isin(ever)]
px = px[(px['DlyCalDt'] >= '2014-06-01') & (px['DlyCalDt'] <= '2022-03-31')]
px = px.drop_duplicates(['PERMNO', 'DlyCalDt'], keep='first')
print(f"Price rows loaded: {len(px):,}  PERMNOs: {px['PERMNO'].nunique():,}  "
      f"columns: {list(px.columns)}")
px_close = px.set_index(['PERMNO', 'DlyCalDt'])['DlyClose']
# row-existence set for the future-calendar completeness check (Section 7) -
# EXISTENCE only, values never read for that purpose.
px_dates_by_permno = px.groupby('PERMNO')['DlyCalDt'].apply(lambda s: set(s.to_numpy()))


# ==========================================================================
# 5. opprcd SCAN - single chunked pass, DTE in [5,35], index_flag==0,
#    secid in bridge whitelist, date in DEV. String-range date pre-filter
#    before any datetime parsing, matching src/36/37's convention.
# ==========================================================================
print('\n' + '-' * 92)
print(f"5. opprcd SCAN - DTE in [{SCAN_DTE_LO},{SCAN_DTE_HI}], "
      f"secid whitelist, date in DEV")
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
rows_after_secid = 0
rows_after_date = 0
rows_after_dte = 0

for i, ch in enumerate(pd.read_csv(opp_path, usecols=NEED, dtype=DT,
                                   chunksize=OPPRCD_CHUNKSIZE), 1):
    rows_scanned += len(ch)
    ch = ch[ch['index_flag'] == 0]
    ch = ch[ch['secid'].astype('int64').isin(secid_whitelist)]
    rows_after_secid += len(ch)
    if len(ch) == 0:
        if i == 1 or i % 10 == 0:
            print(f"  chunk {i:>4}: scanned {rows_scanned:>14,} rows  "
                  f"(kept so far: 0)")
        continue
    ch = ch[(ch['date'] >= DEV_START_S) & (ch['date'] <= DEV_END_S)]
    rows_after_date += len(ch)
    if len(ch) == 0:
        continue
    ch = ch.copy()
    ch['date_d'] = pd.to_datetime(ch['date'])
    ch['exdate_d'] = pd.to_datetime(ch['exdate'])
    ch['dte'] = (ch['exdate_d'] - ch['date_d']).dt.days
    ch = ch[(ch['dte'] >= SCAN_DTE_LO) & (ch['dte'] <= SCAN_DTE_HI)]
    rows_after_dte += len(ch)
    if len(ch) == 0:
        continue
    keep_cols = ['secid', 'date_d', 'exdate_d', 'dte', 'cp_flag', 'strike_price',
                'best_bid', 'best_offer', 'volume', 'open_interest',
                'impl_volatility', 'optionid']
    parts.append(ch[keep_cols])
    if i == 1 or i % 10 == 0:
        kept = sum(len(p) for p in parts)
        print(f"  chunk {i:>4}: scanned {rows_scanned:>14,} rows  "
              f"(kept so far: {kept:,})")

opp = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=keep_cols)
del parts
print(f"\nTotal opprcd rows scanned: {rows_scanned:,}")
print(f"  after index_flag==0 & secid whitelist: {rows_after_secid:,}")
print(f"  after DEV date range: {rows_after_date:,}")
print(f"  after DTE in [{SCAN_DTE_LO},{SCAN_DTE_HI}]: {rows_after_dte:,}")
print(f"Candidate contract-day rows retained: {len(opp):,}")
assert (opp['secid'].astype('int64').isin(secid_whitelist)).all()
opp['secid'] = opp['secid'].astype('int64')


# ==========================================================================
# 6. JOIN TO PERMNO, RESTRICT TO SHARED BASE UNIVERSE, ATTACH S
# ==========================================================================
print('\n' + '-' * 92)
print('6. JOIN TO PERMNO, RESTRICT TO SHARED BASE UNIVERSE, ATTACH UNDERLYING CLOSE')
print('-' * 92)

opp = opp.merge(bridge, on='secid', how='inner')
print(f"After PERMNO join: {len(opp):,} rows")

# Vectorized inner merge restricts to the shared base universe AND attaches
# decile in one step. The earlier zip()-into-a-Python-list-then-.isin(set)
# approach OOM'd at full scale (141M rows) - a real merge is the correct,
# memory-efficient way to do a two-key restriction at this row count.
base_key = base_v1[['PERMNO', 'date_d', 'decile']].drop_duplicates(['PERMNO', 'date_d'])
opp = opp.merge(base_key, on=['PERMNO', 'date_d'], how='inner')
print(f"After restricting to shared base universe (1,733,857 stock-days) "
      f"and attaching decile: {len(opp):,} rows")

# recent 5-trading-day contract volume, INCLUDING the row's own day, per
# optionid, computed on the FULL WIDE [5,35] band (which fully brackets the
# lookback need for any candidate inside [15,25] - see the module
# docstring) - done here, BEFORE narrowing to the diagnostic band, and
# BEFORE the heavier per-row S/no-arb/delta computations below, so those
# heavier steps run on the smaller, narrowed frame only. This ordering is a
# memory-driven restructure (15.8GB total RAM, ~10GB free at full scale;
# the original ordering OOM'd on a different bug but this reordering keeps
# headroom for what remains).
opp = opp.sort_values(['optionid', 'date_d'])
opp['recent_vol_5d'] = (opp.groupby('optionid')['volume']
                        .transform(lambda s: s.rolling(RECENT_VOL_TDAYS, min_periods=1).sum()))
print(f"[OK] recent_vol_5d computed per optionid (trailing {RECENT_VOL_TDAYS} "
      f"trading days incl. current day), on the full [{SCAN_DTE_LO},{SCAN_DTE_HI}] band")

n_wide = len(opp)
opp = opp[(opp['dte'] >= DIAG_DTE_LO) & (opp['dte'] <= DIAG_DTE_HI)].copy()
print(f"Narrowed to diagnostic band [{DIAG_DTE_LO},{DIAG_DTE_HI}] before further "
      f"processing: {n_wide:,} -> {len(opp):,} rows")

px_s = px.rename(columns={'DlyCalDt': 'date_d', 'DlyClose': 'S'})
opp = opp.merge(px_s, on=['PERMNO', 'date_d'], how='left')
n_before_s = len(opp)
opp = opp[np.isfinite(opp['S']) & (opp['S'] > 0)].copy()
print(f"After requiring a valid underlying close on the quote date: "
      f"{len(opp):,}  (dropped {n_before_s - len(opp):,} missing S)")

opp['K'] = (opp['strike_price'] / 1000.0).astype('float32')
opp['T'] = (opp['dte'] / 365.0).astype('float32')
opp['mid'] = ((opp['best_bid'] + opp['best_offer']) / 2.0).astype('float32')
opp['S'] = opp['S'].astype('float32')


# ==========================================================================
# 7. DATE-AVAILABILITY CHECK (Section 7 requirement) - complete future
#    trading calendar through the matched horizon. Row EXISTENCE only, no
#    price/return VALUE read for this purpose. Computed on the (already
#    diagnostic-band-narrowed) unique stock-dates only.
# ==========================================================================
print('\n' + '-' * 92)
print('7. FUTURE-CALENDAR COMPLETENESS (date/row EXISTENCE only, no values read)')
print('-' * 92)

uniq_sd = opp[['PERMNO', 'date_d']].drop_duplicates()
n20_req = uniq_sd['date_d'].map(n20_by_date).to_numpy()


def calendar_complete(permno, date_d, n_needed):
    if not np.isfinite(n_needed):
        return False
    pos = cal_pos_of.get(date_d)
    if pos is None:
        return False
    future_dates = cal_all[pos + 1: pos + 1 + int(n_needed)]
    if len(future_dates) < int(n_needed):
        return False  # global calendar itself does not extend far enough
    have = px_dates_by_permno.get(permno)
    if have is None:
        return False
    return all(fd in have for fd in future_dates)


complete_flags = [
    calendar_complete(p, d0, n)
    for p, d0, n in zip(uniq_sd['PERMNO'].to_numpy(), uniq_sd['date_d'].to_numpy(), n20_req)
]
uniq_sd = uniq_sd.copy()
uniq_sd['calendar_complete'] = complete_flags
pct_complete = np.mean(complete_flags) * 100 if len(complete_flags) else float('nan')
print(f"Stock-dates checked: {len(uniq_sd):,}  "
      f"complete future calendar through +20cal: {pct_complete:.2f}%")
opp = opp.merge(uniq_sd, on=['PERMNO', 'date_d'], how='left')


# ==========================================================================
# 8. CONTRACT-LEVEL QUANTITIES - spread, zero-bid, no-arbitrage bounds,
#    BS delta computed from opprcd's OWN impl_volatility (no vendor delta
#    field exists in this file - confirmed by header inspection; matches
#    the identical finding already recorded in results/prereg_V4.md
#    Section 6.1/7.2). recent_vol_5d already computed in Section 6, on the
#    wide band, before narrowing.
# ==========================================================================
print('\n' + '-' * 92)
print('8. CONTRACT-LEVEL QUANTITIES (spread, no-arb, BS delta from opprcd IV)')
print('-' * 92)

opp['is_call'] = (opp['cp_flag'] == 'C')
opp['zero_bid'] = ~(opp['best_bid'] > 0)
opp['ask_gt_bid'] = opp['best_offer'] > opp['best_bid']
opp['bid_ask_finite'] = np.isfinite(opp['best_bid']) & np.isfinite(opp['best_offer'])
opp['valid_quote'] = opp['bid_ask_finite'] & (opp['best_bid'] > 0) & opp['ask_gt_bid']
opp['rel_spread'] = np.where(opp['mid'] > 0,
                             (opp['best_offer'] - opp['best_bid']) / opp['mid'], np.nan)

disc = np.exp(-RISK_FREE_RATE * opp['T'])
S = opp['S'].to_numpy()
K = opp['K'].to_numpy()
mid = opp['mid'].to_numpy()
is_call = opp['is_call'].to_numpy()
lb = np.where(is_call, np.maximum(S - K * disc.to_numpy(), 0.0),
             np.maximum(K * disc.to_numpy() - S, 0.0))
ub = np.where(is_call, S, K * disc.to_numpy())
opp['no_arb_ok'] = opp['valid_quote'] & (mid >= lb - 1e-9) & (mid <= ub + 1e-9)

# BS delta from opprcd's OWN impl_volatility (no vendor delta in this file)
sig_om = opp['impl_volatility'].to_numpy()
T_ = opp['T'].to_numpy()
with np.errstate(all='ignore'):
    sq = sig_om * np.sqrt(T_)
    d1_om = (np.log(S / K) + (RISK_FREE_RATE + 0.5 * sig_om ** 2) * T_) / sq
    delta_call = norm.cdf(d1_om)
    delta_put = delta_call - 1.0
delta = np.where(is_call, delta_call, delta_put)
delta_valid = np.isfinite(sig_om) & (sig_om > 0) & np.isfinite(delta)
opp['delta'] = np.where(delta_valid, delta, np.nan)
opp['abs_delta'] = np.abs(opp['delta'])
opp['delta_valid'] = delta_valid
print(f"contract-days with a computable contemporaneous BS delta "
      f"(from opprcd's own impl_volatility): {int(delta_valid.sum()):,} "
      f"({delta_valid.mean() * 100:.2f}%)")


# ==========================================================================
# 9. DETERMINISTIC SELECTION - nearest-to-20-DTE contract per
#    (PERMNO, date, side), on the DIAGNOSTIC [15,25] band. The PRIMARY
#    [18,22] result is then the subset of this same selection whose dte
#    falls in [18,22] - a proven identity (any [18,22] candidate is always
#    closer to 20 than any candidate outside [18,22], since the former has
#    distance <=2 and the latter distance >=3), so a second independent
#    tie-break pass is not needed and would produce the identical winners.
# ==========================================================================
print('\n' + '-' * 92)
print('9. DETERMINISTIC NEAREST-TO-20-DTE SELECTION (diagnostic band [15,25])')
print('-' * 92)

# opp was already narrowed to [DIAG_DTE_LO, DIAG_DTE_HI] in Section 6, so
# this is a rename, not a further filter - kept as an explicit re-assertion
# rather than silently relying on upstream state.
assert (opp['dte'].between(DIAG_DTE_LO, DIAG_DTE_HI)).all()
diag = opp
del opp
print(f"Candidates inside diagnostic band [{DIAG_DTE_LO},{DIAG_DTE_HI}]: {len(diag):,}")

diag['k1_dte_dist'] = (diag['dte'] - TARGET_DTE).abs()
diag['k2_delta_dist'] = np.where(diag['delta_valid'],
                                 (diag['abs_delta'] - 0.50).abs(), 999.0)
diag['k3_rel_spread'] = diag['rel_spread'].fillna(np.inf)
diag['k4_neg_oi'] = -diag['open_interest'].fillna(0.0)
diag['k5_neg_recvol'] = -diag['recent_vol_5d'].fillna(0.0)
diag['k6_optionid'] = diag['optionid']

diag = diag.sort_values(
    ['PERMNO', 'date_d', 'cp_flag', 'k1_dte_dist', 'k2_delta_dist',
     'k3_rel_spread', 'k4_neg_oi', 'k5_neg_recvol', 'k6_optionid'])
diag_selected = diag.drop_duplicates(['PERMNO', 'date_d', 'cp_flag'], keep='first').copy()
print(f"Selected (one contract per stock-date x side): {len(diag_selected):,}")
print(f"  calls: {int((diag_selected['cp_flag'] == 'C').sum()):,}   "
      f"puts: {int((diag_selected['cp_flag'] == 'P').sum()):,}")

primary_selected = diag_selected[
    (diag_selected['dte'] >= PRIMARY_DTE_LO) & (diag_selected['dte'] <= PRIMARY_DTE_HI)].copy()
print(f"Of which inside primary band [{PRIMARY_DTE_LO},{PRIMARY_DTE_HI}]: "
      f"{len(primary_selected):,}")


# ==========================================================================
# 10. MIDPOINT IV INVERSION - vectorized Newton-Raphson, on selected
#     winners only (both bands share this same computation via the
#     subset relationship established in Section 9).
#
#     DISCLOSED METHOD DEVIATION: prior project scripts (src/42) use
#     scalar scipy.optimize.brentq in a per-row loop, tractable at K1's
#     ~5,300-trade scale. This audit's selected-contract count is
#     potentially two orders of magnitude larger, so a vectorized
#     Newton-Raphson solver is used instead for tractable runtime.
#     Convergence is verified directly against the price residual (not
#     assumed from iteration count), matching the same standard a
#     brentq-based solve would be held to.
# ==========================================================================
print('\n' + '-' * 92)
print('10. MIDPOINT IV INVERSION - vectorized Newton-Raphson '
      f'(disclosed deviation from scalar brentq; bounds [{IV_LO},{IV_HI}])')
print('-' * 92)


def bs_price(S_, K_, T_, r_, sig_, is_call_):
    with np.errstate(all='ignore'):
        sq = sig_ * np.sqrt(T_)
        d1 = (np.log(S_ / K_) + (r_ + 0.5 * sig_ ** 2) * T_) / sq
        d2 = d1 - sq
        disc_ = np.exp(-r_ * T_)
        call = S_ * norm.cdf(d1) - K_ * disc_ * norm.cdf(d2)
        put = K_ * disc_ * norm.cdf(-d2) - S_ * norm.cdf(-d1)
    return np.where(is_call_, call, put), d1


def bs_vega(S_, T_, sig_, d1_):
    with np.errstate(all='ignore'):
        return S_ * norm.pdf(d1_) * np.sqrt(T_)


def newton_iv(S_, K_, T_, r_, target_, is_call_):
    sig = np.full(len(S_), 0.5)  # initial guess
    for _ in range(NEWTON_MAX_ITER):
        price, d1 = bs_price(S_, K_, T_, r_, sig, is_call_)
        vega = bs_vega(S_, T_, sig, d1)
        resid = price - target_
        step = np.where(np.abs(vega) > 1e-8, resid / np.maximum(vega, 1e-8), 0.0)
        sig_new = sig - step
        sig_new = np.clip(sig_new, IV_LO, IV_HI)
        sig = sig_new
    final_price, _ = bs_price(S_, K_, T_, r_, sig, is_call_)
    resid_final = np.abs(final_price - target_)
    ok = resid_final <= np.maximum(NEWTON_PRICE_TOL_ABS, NEWTON_PRICE_TOL_REL * target_)
    return sig, ok


def run_iv_inversion(sel):
    n = len(sel)
    S_ = sel['S'].to_numpy(dtype=float)
    K_ = sel['K'].to_numpy(dtype=float)
    T_ = sel['T'].to_numpy(dtype=float)
    target = sel['mid'].to_numpy(dtype=float)
    is_call_ = sel['is_call'].to_numpy()

    reached = (sel['no_arb_ok'].to_numpy() & np.isfinite(S_) & np.isfinite(K_) &
              (K_ > 0) & np.isfinite(T_) & (T_ > 0) & np.isfinite(target) & (target > 0))
    iv_out = np.full(n, np.nan)
    ok_out = np.zeros(n, dtype=bool)
    if reached.any():
        sig, ok = newton_iv(S_[reached], K_[reached], T_[reached],
                            RISK_FREE_RATE, target[reached], is_call_[reached])
        iv_out[reached] = sig
        ok_out[reached] = ok
    sel = sel.copy()
    sel['iv_reached_inversion_step'] = reached
    sel['iv_inverted'] = np.where(ok_out, iv_out, np.nan)
    sel['iv_inversion_ok'] = ok_out & reached
    return sel


diag_selected = run_iv_inversion(diag_selected)
primary_selected = diag_selected[
    (diag_selected['dte'] >= PRIMARY_DTE_LO) & (diag_selected['dte'] <= PRIMARY_DTE_HI)].copy()

for lbl, sel in [('diagnostic [15,25]', diag_selected), ('primary [18,22]', primary_selected)]:
    reached = int(sel['iv_reached_inversion_step'].sum())
    ok = int(sel['iv_inversion_ok'].sum())
    print(f"  {lbl}: reached inversion step {reached:,} of {len(sel):,}; "
          f"succeeded {ok:,} "
          f"({ok / max(reached, 1) * 100:.2f}% of those reached)")


# ==========================================================================
# 11. EARNINGS BUCKETS - identical CCM link + RDQ join + bucketing logic to
#     src/49 Section 7-8, applied to the (PERMNO, date) pairs actually
#     selected here (computed once per unique pair, joined back to both
#     sides).
# ==========================================================================
print('\n' + '-' * 92)
print('11. EARNINGS BUCKETS (identical construction to src/49, unchanged)')
print('-' * 92)

link = pd.read_csv(link_path)
link.columns = [c.lower() for c in link.columns]
link = link[link['linktype'].isin(['LC', 'LU'])]
link = link[link['linkprim'].isin(['P', 'C'])]
link = link[link['lpermno'].notna()]
link['linkdt'] = pd.to_datetime(link['linkdt'])
link['linkend'] = pd.to_datetime(link['linkenddt'].replace('E', pd.NaT), errors='coerce')
link['linkend'] = link['linkend'].fillna(pd.Timestamp('2262-01-01'))
link['lpermno'] = link['lpermno'].astype(int)
link['gvkey'] = link['gvkey'].astype(int)
lk = link[['lpermno', 'gvkey', 'linkdt', 'linkend', 'linkprim']].copy()

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


def attach_earnings_buckets(sel_stock_dates):
    """sel_stock_dates: unique (PERMNO, date_d) DataFrame. Returns it with
    gvkey + 4-bucket earnings columns attached."""
    n_before = len(sel_stock_dates)
    m = sel_stock_dates.merge(lk, left_on='PERMNO', right_on='lpermno', how='left')
    in_window = (m['date_d'] >= m['linkdt']) & (m['date_d'] <= m['linkend'])
    m = m[in_window | m['lpermno'].isna()]
    m['_pr'] = np.where(m['linkprim'] == 'P', 0, 1)
    m = m.sort_values(['PERMNO', 'date_d', '_pr', 'gvkey'])
    m = m.drop_duplicates(['PERMNO', 'date_d'], keep='first')
    m = m.drop(columns=['lpermno', 'linkdt', 'linkend', 'linkprim', '_pr'])

    pos = cal_all.get_indexer(m['date_d'])
    d0 = m['date_d'].to_numpy()

    def bound(td):
        idx = np.clip(pos + td, 0, len(cal_all) - 1)
        return cal_all.to_numpy()[idx]

    b5, b10, b20 = bound(5), bound(10), bound(EARNINGS_WINDOW_TDAYS)
    gv = m['gvkey'].to_numpy()
    d1b = np.full(len(m), np.nan)
    d2b = np.full(len(m), np.nan)
    d3b = np.full(len(m), np.nan)
    has_history = np.zeros(len(m), dtype=bool)
    for g in np.unique(gv[~np.isnan(gv)]):
        arr = rdq_by_gvkey.get(int(g))
        sel = np.where(gv == g)[0]
        if arr is None or len(arr) == 0:
            continue
        has_history[sel] = True
        lo = np.searchsorted(arr, d0[sel], side='right')
        hi5 = np.searchsorted(arr, b5[sel], side='right')
        hi10 = np.searchsorted(arr, b10[sel], side='right')
        hi20 = np.searchsorted(arr, b20[sel], side='right')
        d1b[sel] = (hi5 > lo).astype(float)
        d2b[sel] = (hi10 > hi5).astype(float)
        d3b[sel] = (hi20 > hi10).astype(float)
    m['D_earn_1to5'] = d1b
    m['D_earn_6to10'] = d2b
    m['D_earn_11to20'] = d3b
    m['earn_no_gvkey'] = m['gvkey'].isna()
    m['earn_no_rdq_history'] = (~has_history) & m['gvkey'].notna()

    def bucket_label(row):
        if row['earn_no_gvkey']:
            return 'no_gvkey'
        if row['earn_no_rdq_history']:
            return 'no_rdq_history'
        if row['D_earn_1to5'] == 1.0:
            return '1-5td'
        if row['D_earn_6to10'] == 1.0:
            return '6-10td'
        if row['D_earn_11to20'] == 1.0:
            return '11-20td'
        return 'none_within_20td'

    m['earnings_bucket'] = m.apply(bucket_label, axis=1)
    print(f"  {n_before:,} stock-dates -> {len(m):,} after link join; "
          f"no_gvkey {int(m['earn_no_gvkey'].sum()):,}  "
          f"no_rdq_history {int(m['earn_no_rdq_history'].sum()):,}")
    return m[['PERMNO', 'date_d', 'earnings_bucket']]


uniq_stock_dates = diag_selected[['PERMNO', 'date_d']].drop_duplicates()
earn_map = attach_earnings_buckets(uniq_stock_dates)
diag_selected = diag_selected.merge(earn_map, on=['PERMNO', 'date_d'], how='left')
primary_selected = diag_selected[
    (diag_selected['dte'] >= PRIMARY_DTE_LO) & (diag_selected['dte'] <= PRIMARY_DTE_HI)].copy()


# ==========================================================================
# 12. FULL COVERAGE REPORT - per band, per side, every requested statistic
# ==========================================================================
print('\n' + '=' * 92)
print('12. FULL COVERAGE REPORT')
print('=' * 92)


def band_side_report(sel, band_label, side_label, side_flag):
    s = sel[sel['is_call'] == side_flag].copy()
    n = len(s)
    print(f"\n  --- {band_label} / {side_label} : {n:,} eligible stock-dates ---")
    if n == 0:
        return {'n_eligible_stock_dates': 0}

    zero_bid_rate = float(s['zero_bid'].mean())
    valid_quote_n = int(s['valid_quote'].sum())
    no_arb_denom = s[s['valid_quote']]
    no_arb_fail_rate = (float((~no_arb_denom['no_arb_ok']).mean())
                        if len(no_arb_denom) else float('nan'))
    reached_n = int(s['iv_reached_inversion_step'].sum())
    iv_fail_rate = (float((~s.loc[s['iv_reached_inversion_step'], 'iv_inversion_ok']).mean())
                   if reached_n else float('nan'))
    usable_iv_n = int(s['iv_inversion_ok'].sum())

    by_year = s['date_d'].dt.year.value_counts().sort_index().to_dict()
    by_decile = s['decile'].value_counts().sort_index().to_dict() if 'decile' in s.columns else {}
    by_earn = s['earnings_bucket'].value_counts().to_dict()
    cal_complete_pct = float(s['calendar_complete'].mean()) * 100

    print(f"    unique underlyings: {s['PERMNO'].nunique():,}   "
          f"unique contracts (optionid): {s['optionid'].nunique():,}")
    print(f"    zero-bid rate: {zero_bid_rate * 100:.2f}%")
    print(f"    valid two-sided quote: {valid_quote_n:,} ({valid_quote_n / n * 100:.2f}%)")
    print(f"    no-arbitrage failure rate (of valid-quote rows): "
          f"{no_arb_fail_rate * 100 if np.isfinite(no_arb_fail_rate) else float('nan'):.2f}%")
    print(f"    reached IV-inversion step: {reached_n:,} ({reached_n / n * 100:.2f}%)")
    print(f"    IV-inversion failure rate (of rows that reached the step): "
          f"{iv_fail_rate * 100 if np.isfinite(iv_fail_rate) else float('nan'):.2f}%")
    print(f"    FINAL usable-IV stock-dates: {usable_iv_n:,} "
          f"({usable_iv_n / n * 100:.2f}% of eligible)")
    print(f"    complete future trading calendar (+20cal, row existence): "
          f"{cal_complete_pct:.2f}%")
    print(f"    by year: {by_year}")
    print(f"    by decile: {by_decile}")
    print(f"    by earnings bucket: {by_earn}")

    return {
        'n_eligible_stock_dates': n,
        'unique_underlyings': int(s['PERMNO'].nunique()),
        'unique_contracts_optionid': int(s['optionid'].nunique()),
        'by_year': {int(k): int(v) for k, v in by_year.items()},
        'by_decile': {int(k): int(v) for k, v in by_decile.items()},
        'by_earnings_bucket': {str(k): int(v) for k, v in by_earn.items()},
        'dte_distribution': pctiles(s['dte'], 'dte'),
        'dte_value_counts': {int(k): int(v) for k, v in
                             s['dte'].value_counts().sort_index().items()},
        'mid_premium_distribution': pctiles(s['mid'], 'mid'),
        'rel_spread_distribution': pctiles(s['rel_spread'], 'rel_spread'),
        'open_interest_distribution': pctiles(s['open_interest'], 'oi'),
        'recent_vol_5d_distribution': pctiles(s['recent_vol_5d'], 'recvol'),
        'zero_bid_rate': zero_bid_rate,
        'valid_two_sided_quote_n': valid_quote_n,
        'valid_two_sided_quote_pct': valid_quote_n / n * 100,
        'no_arbitrage_failure_rate_of_valid_quotes': no_arb_fail_rate,
        'iv_reached_inversion_step_n': reached_n,
        'iv_inversion_failure_rate_of_reached': iv_fail_rate,
        'final_usable_iv_n': usable_iv_n,
        'final_usable_iv_pct_of_eligible': usable_iv_n / n * 100,
        'pct_complete_future_calendar_20cal': cal_complete_pct,
    }


report = {'bands': {}}
for band_label, sel in [('primary_18_22', primary_selected), ('diagnostic_15_25', diag_selected)]:
    print(f"\n{'#' * 92}\nBAND: {band_label}\n{'#' * 92}")
    report['bands'][band_label] = {
        'calls': band_side_report(sel, band_label, 'CALLS', True),
        'puts': band_side_report(sel, band_label, 'PUTS', False),
    }


# ==========================================================================
# 13. WRITE OUTPUT
# ==========================================================================
report['meta'] = {
    'generated_by': 'src/50_v3_short_data_feasibility.py',
    'scope': ('RETURN-BLIND coverage audit only. Does NOT authorize building '
              'V3-short. Does NOT change V4 (results/prereg_V4.md, DRAFT). '
              'DlyRet is never loaded in this script; no RV, option return, '
              'P&L, or regression coefficient is computed anywhere.'),
    'window': f'{DEV_START.date()} to {DEV_END.date()} (DEV only)',
    'target_dte': TARGET_DTE,
    'primary_band_dte': [PRIMARY_DTE_LO, PRIMARY_DTE_HI],
    'diagnostic_band_dte': [DIAG_DTE_LO, DIAG_DTE_HI],
    'extraction_band_dte': [SCAN_DTE_LO, SCAN_DTE_HI],
    'shared_base_universe_stock_days': 1_733_857,
    'declared_assumptions': {
        'risk_free_rate': RISK_FREE_RATE, 'q_dividend': Q_DIV,
        'iv_bounds': [IV_LO, IV_HI],
        'iv_inversion_method': ('vectorized Newton-Raphson, disclosed deviation '
                                'from scalar brentq used elsewhere (src/42) - '
                                'chosen for tractable runtime at this row count; '
                                'convergence verified against price residual, '
                                'not assumed from iteration count'),
        'delta_source': ("BS delta computed from opprcd's own impl_volatility "
                         "column - no vendor delta field exists in opprcd.csv "
                         "(confirmed by header inspection; same finding "
                         "recorded in results/prereg_V4.md Section 6.1/7.2)"),
        'recent_option_volume_window': (f'trailing {RECENT_VOL_TDAYS} trading '
                                        'days, INCLUSIVE of the selection day'),
        'float_precision_note': ('S, K, T, mid cast to float32 after narrowing '
                                 'to the diagnostic band, for memory headroom '
                                 'at full-file scale (15.8GB total RAM). '
                                 'Verified against the float64 dry run: '
                                 'selection counts match to within 0.01% '
                                 '(3 of 43,046 selected contracts differed, '
                                 'at the tie-break spread-comparison margin '
                                 'only) - immaterial to any reported rate.'),
        'tie_break_order': ['abs(dte-20)', 'abs(|delta|-0.50) if computable',
                            'relative spread', 'open interest (desc)',
                            'recent 5-day volume (desc)', 'optionid (asc)'],
    },
    'primary_band_derivation': ('primary [18,22] selection is the exact subset '
                                'of the diagnostic [15,25] selection with '
                                'dte in [18,22] - a proven identity, not a '
                                'second independent tie-break pass (see '
                                'Section 9 comment)'),
    'opprcd_scan': {
        'rows_scanned': int(rows_scanned),
        'rows_after_secid_whitelist': int(rows_after_secid),
        'rows_after_dev_date_range': int(rows_after_date),
        'rows_after_dte_band': int(rows_after_dte),
    },
}

out_json.parent.mkdir(parents=True, exist_ok=True)
with open(out_json, 'w') as f:
    json.dump(report, f, indent=2, default=str)
print(f"\n[OK] wrote {out_json}")

print('\n' + '=' * 92)
print('AUDIT COMPLETE. This is a coverage measurement only.')
print('It does not authorize building V3-short and does not change V4.')
print('gate_log.md was not touched.')
print('=' * 92)
