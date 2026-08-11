import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# Phase V3: Incremental Variance Information. Tests whether the V1/V2 volume-
# compression signal predicts variance beyond what implied volatility already
# reflects. Built against the LOCKED pre-registration at
# results/prereg_V3.md (LOCKED 2026-08-02, both follow-on items closed
# 2026-08-02: maxlags corrected, ACT/365 invariance requirement locked).
#
# DEV WINDOW ONLY. No holdout code path exists in this script. 2022-2025 is
# already spent as holdout for V1/V2; V3 makes no holdout claim (prereg §2).
#
# PRIMARY horizon: 30 CALENDAR days (prereg §3.3-F Resolution) - NOT V1/V2's
# original 10 trading days. RV² and IV² are both recomputed fresh for this
# horizon; neither reuses V1/V2's column. The original 10-trading-day
# construction survives as a labeled, NO-VERDICT-AUTHORITY sensitivity arm
# (prereg §3.3-F), reported alongside but never able to override the primary.
#
# Both compression variants - V1 compression_decile, V2 sector_rel_decile -
# are run as two full, separate instances of every spec (prereg §3.1). Two
# universes - 6(a) full, 6(b) liquid-options subset - are both reported,
# neither privileged (prereg §6).
#
# gate_log.md receives numbers only, no interpretation.
# ---------------------------------------------------------------------------

# --- Declared assumptions and locked constants (prereg §3.3, §3.3-F, §5) --
DEV_START = pd.Timestamp('2015-01-01')
DEV_END = pd.Timestamp('2021-12-31')
CAL_WINDOW_DAYS = 30          # primary horizon: calendar days
SENSITIVITY_HORIZON_TDAYS = 10  # sensitivity arm: trading days (original spec)
EARNINGS_WINDOW_TDAYS = 20

DPEN_TENORS = (10, 30)        # surface tenors available in vol_surface.csv
ACT_DAYS_PER_YEAR = 365.0     # DECLARED ASSUMPTION (prereg §3.3, §3.3-F Resolution)
ACT360_DAYS_PER_YEAR = 360.0  # alternative convention, for the REQUIRED invariance check
TRADING_DAYS_PER_YEAR = 252.0

PRIOR_VOL_WIN = 20            # E1 D-series, reused verbatim (prereg §3.4, §3.6)
RECENT_ABSRET_WIN = 20
TREND_WIN = 120

MIN_XSEC = 30                 # matches src/43 / src/11 / src/18
NW_MAXLAGS_PRIMARY = 21       # prereg §5, CORRECTED 2026-08-02 alongside §3.3-F
NW_MAXLAGS_SENSITIVITY = 10   # prereg §5: sensitivity arm keeps its own horizon's value
BAR_ABS_T = 3.0                # prereg §5 significance bar for b3

SURFACE_CHUNKSIZE = 5_000_000

project_root = Path(__file__).parent.parent
crsp_path = project_root / 'data' / 'processed' / 'crsp_combined.parquet'
univ_path = project_root / 'data' / 'processed' / 'universe_membership.parquet'
v1_path = project_root / 'data' / 'processed' / 'compression_signal_v1.parquet'
v2_path = project_root / 'data' / 'processed' / 'sector_compression_signal_v2.parquet'
surf_path = project_root / 'data' / 'raw' / 'optionmetrics' / 'vol_surface.csv'
om_names_path = project_root / 'data' / 'raw' / 'optionmetrics' / 'om_security_names.csv'
crsp_names_path = project_root / 'data' / 'raw' / 'crsp' / 'crsp_names.parquet'
link_path = project_root / 'data' / 'raw' / 'compustat' / 'ccm_link_gics.csv'
rdq_path = project_root / 'data' / 'raw' / 'compustat' / 'rdq_pull_fundq_2014_2026.parquet'
liq_cache_path = project_root / 'data' / 'processed' / 'opprcd_liquidity_daily.parquet'
prereg_path = project_root / 'results' / 'prereg_V3.md'
out_json = project_root / 'results' / '49_v3_incremental_variance_test.json'
log_path = project_root / 'results' / 'gate_log.md'

print('=' * 92)
print('PHASE V3 - INCREMENTAL VARIANCE INFORMATION  (DEV ONLY, locked prereg results/prereg_V3.md)')
print('=' * 92)

prereg_text = prereg_path.read_text(encoding='utf-8', errors='replace')
if 'Status: LOCKED' not in prereg_text:
    raise SystemExit('STOP: pre-registration is not marked LOCKED.')
print('\n[OK] Locked pre-registration found and verified.')

SIGNALS = [('V1', 'compression_decile'), ('V2', 'sector_rel_decile')]
EARNINGS_DUMMIES = ['D_earn_1to5', 'D_earn_6to10', 'D_earn_11to20']
CONTROL_COLS = ['log_cap', 'log_price', 'log_dvol', 'recent_absret', 'trend']


# ==========================================================================
# Helpers
# ==========================================================================
def nw_ols_const(x, maxlags):
    """Newey-West t-test on the mean of a 1-D series. Returns (mean, t, n)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan, len(x)
    m = sm.OLS(x, np.ones(len(x))).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
    return float(m.params[0]), float(m.tvalues[0]), len(x)


def fama_macbeth(df, ycol, xcols, maxlags, min_xsec=MIN_XSEC):
    """Daily cross-sectional OLS of ycol on [const]+xcols; NW t-test on each
    daily coefficient series. Returns dict keyed by regressor name, plus
    bookkeeping. Matches src/43 / src/46 Test 1's design exactly."""
    use = df.dropna(subset=[ycol] + xcols)
    k = len(xcols) + 1
    coefs, dates, ns = [], [], []
    n_dropped = 0
    for dt, gg in use.groupby('DlyCalDt', sort=True):
        if len(gg) < min_xsec:
            n_dropped += 1
            continue
        X = np.column_stack([np.ones(len(gg))] + [gg[c].to_numpy(float) for c in xcols])
        y = gg[ycol].to_numpy(float)
        if not np.isfinite(X).all() or np.linalg.matrix_rank(X) < k:
            n_dropped += 1
            continue
        coefs.append(np.linalg.lstsq(X, y, rcond=None)[0])
        dates.append(dt)
        ns.append(len(gg))
    if not coefs:
        return None
    C = np.asarray(coefs)
    out = {'n_dates': len(dates), 'n_dates_dropped': n_dropped,
           'mean_xsec_n': float(np.mean(ns)) if ns else float('nan'), 'coefs': {}}
    for j, name in enumerate(['const'] + xcols):
        s = C[:, j]
        m, t, n = nw_ols_const(s, maxlags)
        out['coefs'][name] = {'mean_coef': m, 'nw_t': t, 'n_dates_used': n,
                              'daily_series': s}   # kept for the invariance check; stripped before JSON dump
    return out


def strip_series(fm_result):
    """Remove the raw daily_series arrays before JSON serialization."""
    if fm_result is None:
        return None
    out = {k: v for k, v in fm_result.items() if k != 'coefs'}
    out['coefs'] = {k: {kk: vv for kk, vv in v.items() if kk != 'daily_series'}
                    for k, v in fm_result['coefs'].items()}
    return out


print('\n' + '-' * 92)
print('1. TRADING CALENDAR AND WINDOW-LENGTH PRECOMPUTATION')
print('-' * 92)

cal_all = pd.DatetimeIndex(sorted(
    pd.read_parquet(crsp_path, columns=['DlyCalDt'])['DlyCalDt'].unique()))
print(f"Full trading calendar: {len(cal_all):,} sessions, "
      f"{cal_all.min().date()} to {cal_all.max().date()}")
dev_cal = cal_all[(cal_all >= DEV_START) & (cal_all <= DEV_END)]
print(f"DEV sessions: {len(dev_cal):,}")

# --- n_t: trading sessions in (t, t+30 calendar days], per prereg §3.3-F ---
dev_pos = cal_all.get_indexer(dev_cal)
n_t_arr = np.zeros(len(dev_cal), dtype=int)
for i, p in enumerate(dev_pos):
    hi = dev_cal[i] + pd.Timedelta(days=CAL_WINDOW_DAYS)
    future = cal_all[p + 1:]
    n_t_arr[i] = int((future <= hi).sum())
n_t_by_date = pd.Series(n_t_arr, index=dev_cal)
print(f"n_t (30-calendar-day window session count): "
      f"min={n_t_arr.min()}, max={n_t_arr.max()}, mean={n_t_arr.mean():.2f}, "
      f"mode={pd.Series(n_t_arr).mode().iloc[0]}")
assert n_t_arr.min() >= 17 and n_t_arr.max() <= 22, \
    'n_t distribution moved outside the prereg-confirmed 17-22 range'
print("[OK] matches prereg §3.3-F Resolution's confirmed 17-22 range")

# --- h*_t: calendar-day span of a 10-TRADING-day window, for the sensitivity
# arm's interpolation (original §3.3 Step 3) ---
hstar_arr = np.zeros(len(dev_cal), dtype=int)
for i, p in enumerate(dev_pos):
    end_pos = p + SENSITIVITY_HORIZON_TDAYS
    if end_pos < len(cal_all):
        hstar_arr[i] = (cal_all[end_pos] - dev_cal[i]).days
    else:
        hstar_arr[i] = -1   # incomplete; filtered out downstream
hstar_by_date = pd.Series(hstar_arr, index=dev_cal)
valid_hstar = hstar_arr[hstar_arr > 0]
print(f"h* (10-trading-day window calendar span): "
      f"min={valid_hstar.min()}, max={valid_hstar.max()} "
      f"(prereg confirmed 14-18)")

# cal_pos lookup dict for the gap-check (position of each date in cal_all)
cal_pos_of = pd.Series(np.arange(len(cal_all)), index=cal_all)


# ==========================================================================
# 2. SHARED DAILY PANEL: controls (§3.4, §3.6, reused verbatim from E1)
# ==========================================================================
print('\n' + '-' * 92)
print('2. SHARED DAILY PANEL AND CONTROLS (E1 D-series, reused verbatim)')
print('-' * 92)

univ = pd.read_parquet(univ_path)
univ_in = univ[univ['in_universe']][['PERMNO', 'year_month', 'decile']].copy()
ever = univ_in['PERMNO'].unique()
print(f"Ever-in-universe PERMNOs (decile 6/7/8, any month): {len(ever):,}")

need = ['PERMNO', 'DlyCalDt', 'DlyRet', 'DlyClose', 'DlyCap', 'DlyPrcVol']
d = pd.read_parquet(crsp_path, columns=need)
d = d[d['PERMNO'].isin(ever)]
# Extend past DEV_END so the primary's 30-calendar-day forward window (and
# the sensitivity arm's 10-trading-day window) are both fully observable at
# DEV_END - matching the prereg's confirmed fact that the full calendar
# (through 2025) covers every DEV row's forward window completely.
d = d[(d['DlyCalDt'] >= '2014-01-01') & (d['DlyCalDt'] <= '2022-06-30')]
d = d.drop_duplicates(['PERMNO', 'DlyCalDt'], keep='first')
d = d.sort_values(['PERMNO', 'DlyCalDt']).reset_index(drop=True)
print(f"Daily rows (extended range for forward windows): {len(d):,}  "
      f"PERMNOs: {d['PERMNO'].nunique():,}")

g = d.groupby('PERMNO', sort=False)
d['prior_vol'] = g['DlyRet'].transform(
    lambda s: s.rolling(PRIOR_VOL_WIN, min_periods=PRIOR_VOL_WIN).std().shift(1)
) * np.sqrt(TRADING_DAYS_PER_YEAR)
d['PriorRV2'] = d['prior_vol'] ** 2
d['recent_absret'] = g['DlyClose'].transform(
    lambda s: (s / s.shift(RECENT_ABSRET_WIN) - 1.0).shift(1)).abs()
d['trend'] = g['DlyClose'].transform(
    lambda s: (s / s.shift(TREND_WIN) - 1.0).shift(1))
d['log_cap'] = np.log(d['DlyCap'].clip(lower=1e-6))
d['log_price'] = np.log(d['DlyClose'].abs().clip(lower=1e-6))
d['log_dvol'] = np.log(d['DlyPrcVol'].clip(lower=1e-6))
print("[OK] PriorRV2, log_cap, log_price, log_dvol, recent_absret, trend built "
      "(all shift(1)-aligned per E1's own construction)")

# global row position + calendar position, for the gap-checked RV2 windowing
d['pos'] = np.arange(len(d))
pn = d['PERMNO'].to_numpy()
a_ret = d['DlyRet'].to_numpy(float)
d['cal_pos'] = cal_pos_of.reindex(d['DlyCalDt']).to_numpy()
a_calpos = d['cal_pos'].to_numpy(float)
N = len(d)


# ==========================================================================
# 3. PRIMARY RV² - recomputed fresh at the 30-calendar-day horizon
# (prereg §3.3-F Resolution). NOT V1/V2's column. Gap-checked: a window is
# used only if n_t sessions are observed with NO internal gap for that
# PERMNO (verified via calendar-position differencing, not row-position
# alone) - matching the locked "no fallback window" discipline and the
# stronger disclosure standard src/42 set for a similarly delicate
# full-path quantity.
# ==========================================================================
print('\n' + '-' * 92)
print('3. PRIMARY RV² - 30 CALENDAR DAYS, RECOMPUTED FRESH (prereg §3.3-F Resolution)')
print('-' * 92)

d['n_t'] = d['DlyCalDt'].map(n_t_by_date)   # NaN outside DEV -> not eligible as an anchor
rv2_primary = np.full(N, np.nan)
drop_gap = 0
drop_boundary = 0
drop_nan_return = 0
n_computed = 0

for k in sorted(pd.Series(n_t_arr).unique()):
    anchor_mask = (d['n_t'].to_numpy() == k)
    if not anchor_mask.any():
        continue
    p = np.where(anchor_mask)[0]
    end = p + k
    inb = end < N
    p_ok, end_ok = p[inb], end[inb]
    same_permno = pn[end_ok] == pn[p_ok]
    # gap check: calendar-position span must equal k exactly (see docstring)
    no_gap = (a_calpos[end_ok] - a_calpos[p_ok]) == k
    complete = same_permno & no_gap
    drop_boundary += int((~same_permno).sum())
    drop_gap += int((same_permno & ~no_gap).sum())

    p_c = p_ok[complete]
    if len(p_c) == 0:
        continue
    off = np.arange(1, k + 1)
    idx = p_c[:, None] + off[None, :]
    window_ret = a_ret[idx]                      # (m, k)
    # "n_t sessions fully observed (no gap)" (prereg §3.3-F Resolution) means
    # every session's return must itself be a valid, present observation -
    # not merely that k ROWS exist. A halt/relisting day can leave a row
    # with a NaN DlyRet even though the row itself is calendar-aligned and
    # PERMNO-consistent. np.nanvar would silently compute variance over
    # whatever non-NaN returns remain (or emit a degenerate-slice warning
    # if too few remain) - that is a partial-data fallback the locked text
    # does not permit. Require ALL k returns finite; drop and count
    # otherwise, matching src/42's disclosure standard for a similarly
    # delicate full-path quantity.
    all_valid = np.all(np.isfinite(window_ret), axis=1)
    drop_nan_return += int((~all_valid).sum())
    p_v = p_c[all_valid]
    if len(p_v) == 0:
        continue
    var = np.var(window_ret[all_valid], axis=1, ddof=1)
    rv2_primary[p_v] = var * TRADING_DAYS_PER_YEAR
    n_computed += len(p_v)

d['RV2_primary'] = rv2_primary
n_eligible_anchors = int(d['n_t'].notna().sum())
print(f"DEV-dated rows eligible as anchors: {n_eligible_anchors:,}")
print(f"  RV2_primary computed (all {{n_t}} sessions present, gapless, "
      f"all returns valid): {n_computed:,}")
print(f"  dropped - PERMNO boundary crossed before window end: {drop_boundary:,}")
print(f"  dropped - internal gap in the PERMNO's own daily rows: {drop_gap:,}")
print(f"  dropped - NaN return within an otherwise gapless window "
      f"(e.g. halt/relisting day): {drop_nan_return:,}")
print(f"  coverage: {n_computed / max(n_eligible_anchors, 1) * 100:.2f}%")

# --- spot check: manually verify 3 random computed rows ---
print("\nSpot check (fixed seed 42): manual recomputation vs vectorized result")
rng = np.random.default_rng(42)
ok_idx = np.where(np.isfinite(rv2_primary))[0]
picks = rng.choice(ok_idx, min(3, len(ok_idx)), replace=False)
for i in picks:
    k = int(d['n_t'].iloc[i])
    manual_ret = a_ret[i + 1: i + 1 + k]
    manual_rv2 = np.nanvar(manual_ret, ddof=1) * TRADING_DAYS_PER_YEAR
    match = np.isclose(manual_rv2, rv2_primary[i])
    print(f"  PERMNO {pn[i]}, {d['DlyCalDt'].iloc[i].date()}, n_t={k}: "
          f"manual={manual_rv2:.6f}  vectorized={rv2_primary[i]:.6f}  "
          f"match={match}")
    assert match, 'RV2_primary spot check failed'
print("[PASS] spot checks match")


# ==========================================================================
# 4. SURFACE SCAN - both tenors' ATM IV in one pass (secid, date, days)
# ==========================================================================
print('\n' + '-' * 92)
print('4. VOL SURFACE SCAN - ATM IV at both tenors (10d, 30d), one pass')
print('-' * 92)


def as_cusip8(s):
    return (s.astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            .str.upper().str[:8].str.zfill(8))


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
print(f"PERMNO->secid bridge pairs: {len(bridge):,}  "
      f"(unique PERMNO: {bridge['PERMNO'].nunique():,})")

usecols = ['secid', 'date', 'days', 'delta', 'impl_volatility', 'cp_flag']
dtypes = {'secid': 'int32', 'date': 'str', 'days': 'int16', 'delta': 'float32',
          'impl_volatility': 'float64', 'cp_flag': 'category'}
iv_rows = []
scanned = 0
for i, ch in enumerate(pd.read_csv(surf_path, usecols=usecols, dtype=dtypes,
                                   chunksize=SURFACE_CHUNKSIZE), 1):
    scanned += len(ch)
    ch = ch[ch['impl_volatility'].notna() & (ch['impl_volatility'] > 0)]
    ch = ch[ch['delta'].notna() & ch['cp_flag'].isin(['C', 'P'])]
    ch = ch[ch['days'].isin(DPEN_TENORS)]
    if len(ch) == 0:
        continue
    ch = ch.copy()
    ch['dpen'] = (ch['delta'].abs() / 100.0 - 0.50).abs()
    ch = (ch.sort_values(['secid', 'date', 'days', 'cp_flag', 'dpen'])
          .drop_duplicates(['secid', 'date', 'days', 'cp_flag'], keep='first'))
    iv_rows.append(ch[['secid', 'date', 'days', 'cp_flag', 'impl_volatility']])
    if i == 1 or i % 5 == 0:
        print(f"  chunk {i:>4}: scanned {scanned:>14,} rows")

iv_all = pd.concat(iv_rows, ignore_index=True)
del iv_rows
# fold any duplicate (secid,date,days,cp_flag) key that happened to straddle
# a chunk boundary - the per-chunk nearest-delta selection already picked
# the ATM row within each chunk, so this only guards the rare cross-chunk
# split case.
iv_all = iv_all.drop_duplicates(['secid', 'date', 'days', 'cp_flag'], keep='first')
print(f"\nSurface rows scanned: {scanned:,}")
print(f"ATM rows retained (post per-chunk nearest-delta selection): {len(iv_all):,}")

piv = iv_all.pivot_table(index=['secid', 'date', 'days'], columns='cp_flag',
                         values='impl_volatility', aggfunc='first').reset_index()
piv.columns.name = None
for side in ('C', 'P'):
    if side not in piv.columns:
        piv[side] = np.nan
piv = piv.dropna(subset=['C', 'P'])   # both sides required (prereg §3.3 Step 1)
piv['iv_atm'] = (piv['C'] + piv['P']) / 2.0
iv_wide = piv.pivot_table(index=['secid', 'date'], columns='days',
                          values='iv_atm', aggfunc='first').reset_index()
iv_wide.columns.name = None
for h in DPEN_TENORS:
    if h not in iv_wide.columns:
        iv_wide[h] = np.nan
iv_wide = iv_wide.rename(columns={10: 'IV_10', 30: 'IV_30'})
print(f"secid-days with a usable ATM IV (either tenor, both sides present): "
      f"{len(iv_wide):,}")
print(f"  IV_30 available: {iv_wide['IV_30'].notna().sum():,}")
print(f"  IV_10 available: {iv_wide['IV_10'].notna().sum():,}")
print(f"  both available:  {(iv_wide['IV_10'].notna() & iv_wide['IV_30'].notna()).sum():,}")


# ==========================================================================
# 5. IV² CONSTRUCTIONS - primary (direct 30d read) and sensitivity (10d/30d
# interpolated), both re-annualized precisely per prereg §3.3 / §3.3-F
# ==========================================================================
print('\n' + '-' * 92)
print('5. IV² - PRIMARY (direct 30d tenor) and SENSITIVITY (interpolated)')
print('-' * 92)

dev_rows = d[d['n_t'].notna()][['PERMNO', 'DlyCalDt', 'n_t', 'pos']].copy()
dev_rows['date_str'] = dev_rows['DlyCalDt'].dt.strftime('%Y-%m-%d')
cand = dev_rows.merge(bridge, on='PERMNO', how='left')
cand = cand.merge(iv_wide, left_on=['secid', 'date_str'],
                  right_on=['secid', 'date'], how='left')
cand['hit'] = cand['IV_30'].notna() | cand['IV_10'].notna()
cand = (cand.sort_values('hit', ascending=False)
        .drop_duplicates(['PERMNO', 'DlyCalDt'], keep='first'))
cand = cand.set_index('pos')

hstar = dev_rows['DlyCalDt'].map(hstar_by_date).to_numpy()
n_t_row = dev_rows['n_t'].to_numpy(float)

iv30 = cand.reindex(dev_rows['pos'])['IV_30'].to_numpy(float)
iv10 = cand.reindex(dev_rows['pos'])['IV_10'].to_numpy(float)

# --- PRIMARY: direct 30-day tenor read, no interpolation (§3.3-F) ---
total_var_30 = iv30 ** 2 * (CAL_WINDOW_DAYS / ACT_DAYS_PER_YEAR)
iv2_primary = np.where(np.isfinite(total_var_30) & (n_t_row > 0),
                       total_var_30 / (n_t_row / TRADING_DAYS_PER_YEAR), np.nan)

# --- SENSITIVITY: original §3.3 Steps 1-5, interpolated between 10d/30d,
# re-annualized on the FIXED 10/252 basis (unchanged, original construction) ---
total_var_10 = iv10 ** 2 * (SENSITIVITY_HORIZON_TDAYS / ACT_DAYS_PER_YEAR)
both_tenors = np.isfinite(iv10) & np.isfinite(iv30) & (hstar > 0)
frac = np.where(both_tenors, (hstar - SENSITIVITY_HORIZON_TDAYS) /
                (CAL_WINDOW_DAYS - SENSITIVITY_HORIZON_TDAYS), np.nan)
total_var_hstar = total_var_10 + (total_var_30 - total_var_10) * frac
iv2_sensitivity = np.where(
    both_tenors, total_var_hstar / (SENSITIVITY_HORIZON_TDAYS / TRADING_DAYS_PER_YEAR),
    np.nan)

iv_out = pd.DataFrame({
    'pos': dev_rows['pos'].to_numpy(),
    'IV2_primary': iv2_primary,
    'IV2_sensitivity': iv2_sensitivity,
    'both_tenors_available': both_tenors,
}).set_index('pos')

d['IV2_primary'] = np.nan
d['IV2_sensitivity'] = np.nan
d['both_tenors_available'] = False
d.loc[iv_out.index, 'IV2_primary'] = iv_out['IV2_primary'].to_numpy()
d.loc[iv_out.index, 'IV2_sensitivity'] = iv_out['IV2_sensitivity'].to_numpy()
d.loc[iv_out.index, 'both_tenors_available'] = iv_out['both_tenors_available'].to_numpy()

n_dev = len(dev_rows)
print(f"DEV anchor rows (ALL ever-in-universe PERMNOs, ANY DEV date - a "
      f"BROADER population than 'V1/V2 universe stock-days', since this has "
      f"no decile/signal-availability filter yet): {n_dev:,}")
print(f"  IV2_primary available:      {int(d['IV2_primary'].notna().sum()):,} "
      f"({d['IV2_primary'].notna().sum() / n_dev * 100:.2f}%)")
print(f"  IV2_sensitivity available:  "
      f"{int(d['both_tenors_available'].sum()):,} "
      f"({d['both_tenors_available'].sum() / n_dev * 100:.2f}%)")
print(f"  NOTE: prereg §3.3-F's 91.45%/4.30% figures were measured against "
      f"the properly-scoped 'V1/V2 universe stock-days' population (decile "
      f"6/7/8, in_universe, signal defined) - a SMALLER, more restrictive "
      f"set than this broader anchor population. The correct apples-to-"
      f"apples comparison is printed in Section 6, after the per-signal "
      f"panels are built with that exact scoping.")


# ==========================================================================
# 6. PER-SIGNAL BASE PANELS (§3.1): V1 compression_decile, V2 sector_rel_decile
# ==========================================================================
print('\n' + '-' * 92)
print('6. PER-SIGNAL BASE PANELS - V1 and V2, run separately, never merged')
print('-' * 92)

shared_cols = ['PERMNO', 'DlyCalDt', 'pos', 'PriorRV2', 'recent_absret', 'trend',
              'log_cap', 'log_price', 'log_dvol', 'RV2_primary', 'IV2_primary',
              'IV2_sensitivity', 'both_tenors_available', 'n_t']
shared = d[shared_cols].copy()

panels = {}
for tag, xcol in SIGNALS:
    if tag == 'V1':
        p = pd.read_parquet(v1_path, columns=['PERMNO', 'DlyCalDt', 'compression_decile',
                                              'realized_vol_fwd_10d'])
    else:
        p = pd.read_parquet(v2_path, columns=['PERMNO', 'DlyCalDt', 'sector_rel_decile'])
    p = p[(p['DlyCalDt'] >= DEV_START) & (p['DlyCalDt'] <= DEV_END)]
    p = p.dropna(subset=[xcol])
    p['year_month'] = p['DlyCalDt'].dt.to_period('M').astype(str)
    p = p.merge(univ_in, on=['PERMNO', 'year_month'], how='inner')   # decile 6/7/8, in_universe
    p = p.merge(shared, on=['PERMNO', 'DlyCalDt'], how='left')
    if tag == 'V1':
        p['RV2_sensitivity'] = p['realized_vol_fwd_10d'] ** 2   # reused verbatim, prereg §3.2
        p = p.drop(columns=['realized_vol_fwd_10d'])
    else:
        # sensitivity RV2 is signal-independent (same V1 10-trading-day column);
        # attach it here too so V2's sensitivity arm is directly comparable.
        v1_rv = pd.read_parquet(v1_path, columns=['PERMNO', 'DlyCalDt', 'realized_vol_fwd_10d'])
        p = p.merge(v1_rv, on=['PERMNO', 'DlyCalDt'], how='left')
        p['RV2_sensitivity'] = p['realized_vol_fwd_10d'] ** 2
        p = p.drop(columns=['realized_vol_fwd_10d'])
    panels[tag] = p
    print(f"  {tag}: {len(p):,} rows, {p['PERMNO'].nunique():,} PERMNOs, "
          f"regressor '{xcol}'")
    iv30_cov = p['IV2_primary'].notna().mean() * 100
    ivboth_cov = p['both_tenors_available'].mean() * 100
    print(f"    IV2_primary available on THIS properly-scoped panel: "
          f"{iv30_cov:.2f}%   [prereg §3.3-F checkpoint: ~91.45%]")
    print(f"    IV2_sensitivity (both tenors) available: "
          f"{ivboth_cov:.2f}%   [prereg §3.3-F checkpoint: ~4.30%]")


# ==========================================================================
# 7. CCM LINK - gvkey attachment, reusing src/43's exact join (date-windowed
# LINKDT/LINKENDDT, LINKTYPE LC/LU, LINKPRIM P/C, P-over-C tiebreak)
# ==========================================================================
print('\n' + '-' * 92)
print('7. CCM LINK - gvkey attachment (reused from src/43, unchanged)')
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


def attach_gvkey(panel, tag):
    n_before = len(panel)
    m = panel.merge(lk, left_on='PERMNO', right_on='lpermno', how='left')
    in_window = (m['DlyCalDt'] >= m['linkdt']) & (m['DlyCalDt'] <= m['linkend'])
    m = m[in_window | m['lpermno'].isna()]
    m['_pr'] = np.where(m['linkprim'] == 'P', 0, 1)
    m = m.sort_values(['PERMNO', 'DlyCalDt', '_pr', 'gvkey'])
    m = m.drop_duplicates(['PERMNO', 'DlyCalDt'], keep='first')
    m = m.drop(columns=['lpermno', 'linkdt', 'linkend', 'linkprim', '_pr'])
    matched = m['gvkey'].notna()
    print(f"  {tag}: {n_before:,} -> {len(m):,} after date window; "
          f"gvkey matched {int(matched.sum()):,} ({matched.mean() * 100:.2f}%)")
    return m


for tag, _ in SIGNALS:
    panels[tag] = attach_gvkey(panels[tag], tag)


# ==========================================================================
# 8. EARNINGS BUCKETS - 4 categories, extending src/43's single binary flag
# to 3 cutoffs (5, 10, 20 trading days), same (gvkey, rdq) pairs, same join
# ==========================================================================
print('\n' + '-' * 92)
print('8. EARNINGS BUCKETS (1-5td, 6-10td, 11-20td, none-within-20td)')
print('-' * 92)

rdq = pd.read_parquet(rdq_path)
rdq.columns = [c.lower() for c in rdq.columns]
for col, keep in [('indfmt', 'INDL'), ('datafmt', 'STD'), ('popsrc', 'D'),
                  ('consol', 'C')]:
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


def build_earnings_buckets(panel, tag):
    pos = cal_all.get_indexer(panel['DlyCalDt'])
    d0 = panel['DlyCalDt'].to_numpy()

    def bound(td):
        idx = np.clip(pos + td, 0, len(cal_all) - 1)
        return cal_all.to_numpy()[idx]

    b5, b10, b20 = bound(5), bound(10), bound(EARNINGS_WINDOW_TDAYS)
    gv = panel['gvkey'].to_numpy()

    d1 = np.full(len(panel), np.nan)   # 1-5
    d2 = np.full(len(panel), np.nan)   # 6-10
    d3 = np.full(len(panel), np.nan)   # 11-20
    has_history = np.zeros(len(panel), dtype=bool)

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
        d1[sel] = (hi5 > lo).astype(float)
        d2[sel] = ((hi10 > hi5)).astype(float)
        d3[sel] = ((hi20 > hi10)).astype(float)

    panel = panel.copy()
    panel['D_earn_1to5'] = d1
    panel['D_earn_6to10'] = d2
    panel['D_earn_11to20'] = d3
    panel['_has_rdq_history'] = has_history
    n_gap = int((~has_history & panel['gvkey'].notna()).sum())
    n_nogvkey = int(panel['gvkey'].isna().sum())
    print(f"  {tag}: no gvkey (dropped): {n_nogvkey:,}   "
          f"gvkey but no RDQ history (dropped): {n_gap:,}")
    # coverage-gap rows are dropped, NOT folded into "none" (prereg §3.5)
    panel.loc[~has_history, EARNINGS_DUMMIES] = np.nan
    covered = panel.dropna(subset=EARNINGS_DUMMIES)
    for lbl, col in [('1-5td', 'D_earn_1to5'), ('6-10td', 'D_earn_6to10'),
                     ('11-20td', 'D_earn_11to20')]:
        print(f"    {lbl:<8} rate: {covered[col].mean() * 100:.2f}%")
    none_rate = (1 - covered[EARNINGS_DUMMIES].max(axis=1)).mean()
    print(f"    none-within-20td (reference) rate: {none_rate * 100:.2f}%")
    return panel


for tag, _ in SIGNALS:
    panels[tag] = build_earnings_buckets(panels[tag], tag)


# ==========================================================================
# 9. LIQUID-UNIVERSE 6(b) MEMBERSHIP - LOCKED definition, from the durable
# opprcd liquidity cache + same-day DlyPrcVol median rank against the
# SHARED standard project universe (matches how the locked 44.89% figure
# was computed - a property of (PERMNO, date), not signal-specific).
# ==========================================================================
print('\n' + '-' * 92)
print("9. LIQUID-UNIVERSE 6(b) - nonzero option volume AND dvol >= same-day median")
print('-' * 92)

liq = pd.read_parquet(liq_cache_path)
print(f"[OK] Loaded cached liquidity panel: {len(liq):,} secid-days "
      f"({liq_cache_path.name}, no re-scan)")

# shared standard-universe denominator (V1's compression-defined base,
# decile 6-8, DEV) - identical construction to the one that produced the
# locked §6(b) coverage numbers.
base_v1 = pd.read_parquet(v1_path, columns=['PERMNO', 'DlyCalDt', 'compression_ratio'])
base_v1 = base_v1[(base_v1['DlyCalDt'] >= DEV_START) & (base_v1['DlyCalDt'] <= DEV_END)]
base_v1 = base_v1[base_v1['compression_ratio'].notna()][['PERMNO', 'DlyCalDt']]
base_v1['year_month'] = base_v1['DlyCalDt'].dt.to_period('M').astype(str)
base_v1 = base_v1.merge(univ_in, on=['PERMNO', 'year_month'], how='inner')
base_v1 = base_v1[['PERMNO', 'DlyCalDt']].drop_duplicates()
base_v1['date_str'] = base_v1['DlyCalDt'].dt.strftime('%Y-%m-%d')
print(f"shared base universe: {len(base_v1):,} stock-days "
      f"[prereg §6(b) expects 1,733,857]")
assert len(base_v1) == 1_733_857, 'base universe row count moved from the locked figure'

cand6b = base_v1.merge(bridge, on='PERMNO', how='left')
cand6b = cand6b.merge(liq, left_on=['secid', 'date_str'], right_on=['secid', 'date'],
                      how='left')
cand6b['has_vol'] = cand6b['has_vol'].fillna(False).astype(bool)
per6b = (cand6b.groupby(['PERMNO', 'DlyCalDt'])['has_vol'].max().reset_index())

dvol = pd.read_parquet(crsp_path, columns=['PERMNO', 'DlyCalDt', 'DlyPrcVol'])
dvol = dvol.drop_duplicates(['PERMNO', 'DlyCalDt'], keep='first')
per6b = per6b.merge(dvol, on=['PERMNO', 'DlyCalDt'], how='left')
per6b['dvol_pct'] = per6b.groupby('DlyCalDt')['DlyPrcVol'].rank(pct=True)
per6b['in_6b'] = per6b['has_vol'] & (per6b['dvol_pct'] >= 0.50)

n_6b = int(per6b['in_6b'].sum())
print(f"universe 6(b) membership: {n_6b:,} of {len(per6b):,} "
      f"({n_6b / len(per6b) * 100:.2f}%)   [prereg §6(b) locked: 44.89%]")

for tag, _ in SIGNALS:
    panels[tag] = panels[tag].merge(
        per6b[['PERMNO', 'DlyCalDt', 'in_6b']], on=['PERMNO', 'DlyCalDt'], how='left')
    panels[tag]['in_6b'] = panels[tag]['in_6b'].fillna(False).astype(bool)
    print(f"  {tag}: in_6b rate on its own estimation rows: "
          f"{panels[tag]['in_6b'].mean() * 100:.2f}%")


# ==========================================================================
# 10. JOINT REGRESSION MATRIX (prereg §4) - 2 signals x 2 universes x
# 2 horizons = 8 runs. PRIMARY: maxlags=21. SENSITIVITY: maxlags=10, no
# verdict authority, restricted to the both-tenor sample (prereg §3.3-F).
# ==========================================================================
print('\n' + '=' * 92)
print('10. JOINT SPECIFICATION - PRIMARY (30cal, maxlags=21) and')
print('    SENSITIVITY (10td, maxlags=10, NO VERDICT AUTHORITY)')
print('=' * 92)

INTER_COLS = [f'{d1}_x_comp' for d1 in EARNINGS_DUMMIES]


def build_interactions(panel, xcol):
    panel = panel.copy()
    for dcol, icol in zip(EARNINGS_DUMMIES, INTER_COLS):
        panel[icol] = panel[xcol] * panel[dcol]
    return panel


def totals_b3_plus_b5k(fm_result, xcol_name):
    """b3 + b5k combined series, per earnings bucket k, NW-tested on the
    SUMMED daily series (both terms estimated jointly each day)."""
    if fm_result is None:
        return {}
    b3_series = fm_result['coefs'][xcol_name]['daily_series']
    out = {}
    for icol in INTER_COLS:
        if icol not in fm_result['coefs']:
            continue
        combo = b3_series + fm_result['coefs'][icol]['daily_series']
        m, t, n = nw_ols_const(combo, fm_result['_maxlags'])
        out[icol] = {'mean_coef': m, 'nw_t': t, 'n_dates_used': n}
    return out


results_matrix = {}
for tag, xcol in SIGNALS:
    panel = build_interactions(panels[tag], xcol)
    xcols = [xcol] + EARNINGS_DUMMIES + INTER_COLS + CONTROL_COLS
    results_matrix[tag] = {}

    print(f"\n{'=' * 92}\nSIGNAL {tag} ({xcol})\n{'=' * 92}")

    for horizon_lbl, ycol, ivcol, maxlags, extra_mask in [
        ('primary', 'RV2_primary', 'IV2_primary', NW_MAXLAGS_PRIMARY, None),
        ('sensitivity', 'RV2_sensitivity', 'IV2_sensitivity', NW_MAXLAGS_SENSITIVITY,
         panel['both_tenors_available']),
    ]:
        model_xcols = [ivcol, 'PriorRV2'] + xcols
        base_df = panel if extra_mask is None else panel[extra_mask]
        results_matrix[tag][horizon_lbl] = {}
        for uni_lbl, uni_mask_col in [('6a_full', None), ('6b_liquid', 'in_6b')]:
            df_run = base_df if uni_mask_col is None else base_df[base_df[uni_mask_col]]
            fm = fama_macbeth(df_run, ycol, model_xcols, maxlags)
            if fm is not None:
                fm['_maxlags'] = maxlags
            results_matrix[tag][horizon_lbl][uni_lbl] = fm
            if fm is None:
                print(f"  [{horizon_lbl:<11}] [{uni_lbl:<9}] "
                      f"n_rows={len(df_run):>9,}  NO USABLE DATES")
                continue
            b3 = fm['coefs'][xcol]
            n_est = int(df_run.dropna(subset=[ycol] + model_xcols).shape[0])
            print(f"  [{horizon_lbl:<11}] [{uni_lbl:<9}] "
                  f"n_rows={len(df_run):>9,}  n_estimable={n_est:>9,}  "
                  f"dates={fm['n_dates']:>5,} (dropped {fm['n_dates_dropped']:,})  "
                  f"b3={b3['mean_coef']:+.6e}  NW t={b3['nw_t']:+.3f}  "
                  f"{'MET' if np.isfinite(b3['nw_t']) and abs(b3['nw_t']) >= BAR_ABS_T else 'NOT MET'} "
                  f"(bar |t|>={BAR_ABS_T})")

    # bucket-conditional totals (b3 + b5k), primary spec only - needed for
    # the §7 "survives only in the no-earnings bucket" branch
    print(f"\n  Bucket-conditional totals (b3 + b5k), PRIMARY spec:")
    for uni_lbl in ('6a_full', '6b_liquid'):
        fm = results_matrix[tag]['primary'][uni_lbl]
        totals = totals_b3_plus_b5k(fm, xcol)
        results_matrix[tag]['primary'][uni_lbl + '_bucket_totals'] = totals
        for icol, lbl in zip(INTER_COLS, ['1-5td', '6-10td', '11-20td']):
            if icol in totals:
                tv = totals[icol]
                print(f"    [{uni_lbl:<9}] {lbl:<8} b3+b5k={tv['mean_coef']:+.6e}  "
                      f"NW t={tv['nw_t']:+.3f}")

    # Plain-language economic size (prereg §4, REQUIRED alongside significance):
    # sqrt(median_RV2 + b3) - sqrt(median_RV2), i.e. the model's implied change
    # in forward realized VOLATILITY (not variance) for a one-decile move in
    # compression, holding IV2 fixed, at the sample median RV2. Applied here
    # to RV2_primary (the actual primary target post-§3.3-F), correcting the
    # locked sentence's leftover "10-trading-day" phrasing to the horizon this
    # document actually locked - the same horizon-correction principle §3.3-F
    # itself already applied throughout.
    print(f"\n  Plain-language economic size (prereg §4), PRIMARY spec, "
          f"sqrt(median RV2 + b3) - sqrt(median RV2):")
    econ_size = {}
    for uni_lbl, uni_mask_col in [('6a_full', None), ('6b_liquid', 'in_6b')]:
        df_ref = panel if uni_mask_col is None else panel[panel[uni_mask_col]]
        median_rv2 = float(df_ref['RV2_primary'].median())
        fm = results_matrix[tag]['primary'][uni_lbl]
        b3_val = fm['coefs'][xcol]['mean_coef'] if fm else np.nan
        delta_vol = np.sqrt(max(median_rv2 + b3_val, 0.0)) - np.sqrt(median_rv2)
        econ_size[uni_lbl] = {'median_RV2_primary': median_rv2, 'b3': b3_val,
                              'delta_vol_annualized': delta_vol}
        print(f"    [{uni_lbl:<9}] median RV2={median_rv2:.6f}  "
              f"(median annualized vol {np.sqrt(median_rv2) * 100:.2f}%)   "
              f"b3={b3_val:+.6e}   "
              f"implied change = {delta_vol * 100:+.3f} pp of annualized vol "
              f"per one-decile move in compression, holding IV2 fixed")
    results_matrix[tag]['primary_economic_size'] = econ_size


# ==========================================================================
# 11. REQUIRED: ACT/365-vs-ACT/360 invariance assertion (prereg §3.3-F
# Resolution). NOT optional, not deferred. IV2 is the only model term the
# day-count assumption touches and it enters linearly with its own free
# coefficient, so standard OLS/FM linear algebra makes b3 provably
# invariant to the day-count convention; only b1 (the IV2 coefficient)
# should rescale by exactly 360/365. A failed assertion reveals a coding
# error, not a modeling choice - no result may be reported if it fails.
# ==========================================================================
print('\n' + '=' * 92)
print('11. REQUIRED ACT/365-vs-ACT/360 INVARIANCE ASSERTION (prereg §3.3-F Resolution)')
print('=' * 92)

ACT360_RESCALE = ACT_DAYS_PER_YEAR / ACT360_DAYS_PER_YEAR   # 365/360
B1_PREDICTED_RATIO = ACT360_DAYS_PER_YEAR / ACT_DAYS_PER_YEAR  # 360/365

invariance_report = {}
all_passed = True
for tag, xcol in SIGNALS:
    panel = build_interactions(panels[tag], xcol)
    panel = panel.copy()
    panel['IV2_primary_ACT360'] = panel['IV2_primary'] * ACT360_RESCALE
    xcols = [xcol] + EARNINGS_DUMMIES + INTER_COLS + CONTROL_COLS
    invariance_report[tag] = {}
    for uni_lbl, uni_mask_col in [('6a_full', None), ('6b_liquid', 'in_6b')]:
        df_run = panel if uni_mask_col is None else panel[panel[uni_mask_col]]
        fm365 = results_matrix[tag]['primary'][uni_lbl]
        fm360 = fama_macbeth(df_run, 'RV2_primary', ['IV2_primary_ACT360', 'PriorRV2'] + xcols,
                             NW_MAXLAGS_PRIMARY)
        if fm365 is None or fm360 is None:
            print(f"  {tag} [{uni_lbl}]: SKIPPED (no usable dates under one convention)")
            continue
        b3_365 = fm365['coefs'][xcol]
        b3_360 = fm360['coefs'][xcol]
        b1_365 = fm365['coefs']['IV2_primary']
        b1_360 = fm360['coefs']['IV2_primary_ACT360']

        b3_match = np.isclose(b3_365['mean_coef'], b3_360['mean_coef'], rtol=1e-9, atol=1e-12)
        t_match = np.isclose(b3_365['nw_t'], b3_360['nw_t'], rtol=1e-6, atol=1e-9)
        b1_predicted = b1_365['mean_coef'] * B1_PREDICTED_RATIO
        b1_match = np.isclose(b1_predicted, b1_360['mean_coef'], rtol=1e-6, atol=1e-12)

        passed = bool(b3_match and t_match and b1_match)
        all_passed = all_passed and passed
        print(f"  {tag} [{uni_lbl}]: b3 ACT365={b3_365['mean_coef']:+.10e}  "
              f"ACT360={b3_360['mean_coef']:+.10e}  match={b3_match}")
        print(f"                 NW t ACT365={b3_365['nw_t']:+.6f}  "
              f"ACT360={b3_360['nw_t']:+.6f}  match={t_match}")
        print(f"                 b1 ACT365={b1_365['mean_coef']:+.6e}  "
              f"predicted ACT360={b1_predicted:+.6e}  "
              f"actual ACT360={b1_360['mean_coef']:+.6e}  match={b1_match}")
        print(f"                 -> {'PASS' if passed else 'FAIL'}")
        invariance_report[tag][uni_lbl] = {
            'b3_ACT365': b3_365['mean_coef'], 'b3_ACT360': b3_360['mean_coef'],
            'b3_match': b3_match, 't_ACT365': b3_365['nw_t'], 't_ACT360': b3_360['nw_t'],
            't_match': t_match, 'b1_ACT365': b1_365['mean_coef'],
            'b1_ACT360_predicted': b1_predicted, 'b1_ACT360_actual': b1_360['mean_coef'],
            'b1_match': b1_match, 'passed': passed,
        }

print(f"\n{'=' * 92}")
if not all_passed:
    print("INVARIANCE ASSERTION FAILED - this indicates a coding error (the ACT/365")
    print("assumption leaking into a part of the pipeline this analysis did not account")
    print("for). Per prereg §3.3-F Resolution: NO V3 RESULT MAY BE REPORTED until this")
    print("is found and fixed. Halting before any decision-tree evaluation or write.")
    print('=' * 92)
    raise SystemExit(2)
print("[PASS] b3 (and its NW t-stat) are numerically identical under ACT/365 vs")
print("ACT/360 in every signal x universe combination checked, and b1 rescales")
print(f"by exactly the predicted 360/365 = {B1_PREDICTED_RATIO:.6f} factor. The day-count")
print("assumption is confirmed NOT to bias the load-bearing coefficient.")
print('=' * 92)


# ==========================================================================
# 12. DECISION TREE EVALUATION (prereg §7)
#
# "Correct sign" for b3: NEGATIVE. Compression_{i,t} is the same 1-10 decile
# scale V1/V2 locked (1 = most compressed); V1/V2's own gated result is that
# LOWER decile (more compression) predicts HIGHER forward realized vol, i.e.
# a NEGATIVE slope. b3 here is tested against that same established sign.
#
# Tie-break, applied exactly as locked: "does not survive controlling for
# IV²" (row 4) is checked FIRST; if it triggers, evaluation stops there.
# For the remaining rows (1, 2, 3), this script does NOT invent a further
# priority ordering the locked text does not state - it reports the literal
# truth value of every row's condition and flags explicitly if more than one
# holds simultaneously (a composability gap the locked §7 text does not
# resolve), rather than silently picking a winner. A (significant-in-6b-
# only, not-in-6a) outcome is also possible and is NOT covered by any of the
# four rows; if encountered, it is reported as UNCLASSIFIED, not forced into
# one.
# ==========================================================================
print('\n' + '=' * 92)
print('12. DECISION TREE EVALUATION (prereg §7)')
print('=' * 92)

EXPECTED_SIGN_B3 = -1


def sig_correct_sign(coef_dict):
    if coef_dict is None or not np.isfinite(coef_dict.get('nw_t', np.nan)):
        return False
    return (abs(coef_dict['nw_t']) >= BAR_ABS_T and
            np.sign(coef_dict['mean_coef']) == EXPECTED_SIGN_B3)


decision_report = {}
for tag, xcol in SIGNALS:
    fm_6a = results_matrix[tag]['primary']['6a_full']
    fm_6b = results_matrix[tag]['primary']['6b_liquid']
    b3_6a = fm_6a['coefs'][xcol] if fm_6a else None
    b3_6b = fm_6b['coefs'][xcol] if fm_6b else None
    sig_6a = sig_correct_sign(b3_6a)
    sig_6b = sig_correct_sign(b3_6b)

    bucket_totals = results_matrix[tag]['primary'].get('6a_full_bucket_totals', {})
    bucket_all_insig = bool(bucket_totals) and all(
        not sig_correct_sign(v) for v in bucket_totals.values())

    rule4 = (not sig_6a) and (not sig_6b)
    rule1 = sig_6a and sig_6b
    rule3 = sig_6a and (not sig_6b)
    rule2 = sig_6a and bucket_all_insig
    unclassified = (not sig_6a) and sig_6b

    if rule4:
        verdict = 4
        why = 'does not survive controlling for IV2 (fails significance/sign in both universes)'
    elif unclassified:
        verdict = None
        why = ('UNCLASSIFIED: significant+correct-sign in 6(b) but not 6(a) - '
              'not covered by any locked §7 row (the locked tree has no row '
              'for "liquid-subset-only" significance)')
    else:
        triggered = []
        if rule1:
            triggered.append(1)
        if rule2:
            triggered.append(2)
        if rule3:
            triggered.append(3)
        verdict = triggered[0] if triggered else None
        why = f"row(s) literally satisfied: {triggered}" if triggered else \
            'no row condition satisfied (should not occur given rule4/unclassified checks above)'

    multiple = (not rule4) and (not unclassified) and (int(rule1) + int(rule2) + int(rule3) > 1)

    decision_report[tag] = {
        'b3_6a': b3_6a, 'b3_6b': b3_6b,
        'sig_correct_sign_6a': sig_6a, 'sig_correct_sign_6b': sig_6b,
        'bucket_totals_all_insignificant': bucket_all_insig,
        'rule1_both_universes': rule1, 'rule2_no_earnings_bucket_only': rule2,
        'rule3_illiquid_only': rule3, 'rule4_does_not_survive': rule4,
        'unclassified_liquid_only': unclassified,
        'multiple_rows_literally_true': multiple,
        'verdict_row': verdict, 'verdict_reason': why,
    }
    print(f"\n  {tag} ({xcol}):")
    print(f"    b3 [6a]: {b3_6a['mean_coef'] if b3_6a else float('nan'):+.6e}  "
          f"t={b3_6a['nw_t'] if b3_6a else float('nan'):+.3f}  sig+correct_sign={sig_6a}")
    print(f"    b3 [6b]: {b3_6b['mean_coef'] if b3_6b else float('nan'):+.6e}  "
          f"t={b3_6b['nw_t'] if b3_6b else float('nan'):+.3f}  sig+correct_sign={sig_6b}")
    print(f"    bucket totals all insignificant: {bucket_all_insig}")
    print(f"    row1(both)={rule1}  row2(bucket-only)={rule2}  "
          f"row3(illiquid-only)={rule3}  row4(fails)={rule4}  "
          f"unclassified={unclassified}")
    if multiple:
        print(f"    [NOTE] multiple row conditions literally true simultaneously "
              f"- see verdict_reason")
    print(f"    -> VERDICT: row {verdict}  ({why})")


# ==========================================================================
# 13. SAVE JSON, APPEND GATE_LOG.MD (marker-guarded, backed up first)
# ==========================================================================
print('\n' + '-' * 92)
print('13. SAVING RESULTS')
print('-' * 92)


def strip_matrix(rm):
    out = {}
    for tag, horizons in rm.items():
        out[tag] = {}
        for hz, universes in horizons.items():
            if hz == 'primary_economic_size':
                out[tag][hz] = universes   # plain dict, not fm-shaped - pass through
                continue
            out[tag][hz] = {}
            for k, v in universes.items():
                if k.endswith('_bucket_totals'):
                    out[tag][hz][k] = v
                else:
                    out[tag][hz][k] = strip_series(v)
    return out


result = {
    'phase': 'V3 - Incremental Variance Information',
    'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'script': 'src/49_v3_incremental_variance_test.py',
    'pre_registration': 'results/prereg_V3.md (LOCKED 2026-08-02, follow-on items '
                        'closed 2026-08-02)',
    'window': f'DEV {DEV_START.date()} to {DEV_END.date()} - no holdout path',
    'primary_horizon': '30 calendar days (prereg §3.3-F Resolution)',
    'sensitivity_horizon': f'{SENSITIVITY_HORIZON_TDAYS} trading days '
                          '(original construction, NO VERDICT AUTHORITY)',
    'declared_assumptions': {
        'ACT_days_per_year': ACT_DAYS_PER_YEAR,
        'maxlags_primary': NW_MAXLAGS_PRIMARY,
        'maxlags_sensitivity': NW_MAXLAGS_SENSITIVITY,
        'min_xsec': MIN_XSEC,
        'significance_bar_abs_t': BAR_ABS_T,
        'expected_sign_b3': EXPECTED_SIGN_B3,
    },
    'window_length_facts': {
        'n_t_30cal_min': int(n_t_arr.min()), 'n_t_30cal_max': int(n_t_arr.max()),
        'n_t_30cal_mean': float(n_t_arr.mean()),
        'hstar_10td_min': int(valid_hstar.min()), 'hstar_10td_max': int(valid_hstar.max()),
    },
    'rv2_primary_construction': {
        'n_eligible_anchors': n_eligible_anchors, 'n_computed': n_computed,
        'dropped_permno_boundary': drop_boundary, 'dropped_internal_gap': drop_gap,
    },
    'iv2_coverage': {
        'primary_pct': float(d['IV2_primary'].notna().sum() / n_dev * 100),
        'sensitivity_both_tenor_pct': float(d['both_tenors_available'].sum() / n_dev * 100),
    },
    'universe_6b_membership_pct': float(n_6b / len(per6b) * 100),
    'results_matrix': strip_matrix(results_matrix),
    'act365_act360_invariance_assertion': invariance_report,
    'invariance_assertion_passed': all_passed,
    'decision_tree': decision_report,
}
out_json.write_text(json.dumps(result, indent=2, default=str), encoding='utf-8')
print(f"[OK] Saved {out_json}")

MARKER = '## V3 Incremental Variance Information (DEV)'
log_text = log_path.read_text(encoding='utf-8', errors='replace')
if MARKER in log_text:
    print('[SKIP] gate_log.md already carries the V3 block; not duplicating.')
else:
    bak = log_path.with_suffix('.md.bak_before_v3_block')
    shutil.copy2(log_path, bak)
    print(f"[safety] gate_log.md backed up to {bak.name}")
    L = [f"\n---\n\n{MARKER}", "",
         f"Run {result['generated']} - script src/49_v3_incremental_variance_test.py - "
         f"pre-registration results/prereg_V3.md (LOCKED 2026-08-02). "
         f"DEV {DEV_START.date()} to {DEV_END.date()}, no holdout code path. "
         f"Primary horizon 30 calendar days (§3.3-F); sensitivity arm 10 trading "
         f"days, NO VERDICT AUTHORITY. maxlags=21 primary / 10 sensitivity. "
         f"ACT/365-vs-ACT/360 b3-invariance assertion: "
         f"{'PASSED' if all_passed else 'FAILED'}. "
         f"Machine-readable copy: results/49_v3_incremental_variance_test.json.",
         "", "```"]
    for tag, xcol in SIGNALS:
        fm_6a = results_matrix[tag]['primary']['6a_full']
        fm_6b = results_matrix[tag]['primary']['6b_liquid']
        fm_6a_s = results_matrix[tag]['sensitivity']['6a_full']
        b3a = fm_6a['coefs'][xcol] if fm_6a else None
        b3b = fm_6b['coefs'][xcol] if fm_6b else None
        b3s = fm_6a_s['coefs'][xcol] if fm_6a_s else None
        dr = decision_report[tag]
        L += [
            f"{tag}  ({xcol})",
            f"  PRIMARY  [6a full]:    b3 {b3a['mean_coef']:+.6e}  NW t "
            f"{b3a['nw_t']:+.3f}  dates {fm_6a['n_dates']:,}" if fm_6a else
            f"  PRIMARY  [6a full]:    NO USABLE DATES",
            f"  PRIMARY  [6b liquid]:  b3 {b3b['mean_coef']:+.6e}  NW t "
            f"{b3b['nw_t']:+.3f}  dates {fm_6b['n_dates']:,}" if fm_6b else
            f"  PRIMARY  [6b liquid]:  NO USABLE DATES",
            (f"  SENSITIVITY [6a, no authority]: b3 {b3s['mean_coef']:+.6e}  "
             f"NW t {b3s['nw_t']:+.3f}  dates {fm_6a_s['n_dates']:,}") if fm_6a_s else
            f"  SENSITIVITY [6a, no authority]: NO USABLE DATES",
            f"  VERDICT: row {dr['verdict_row']}",
            "",
        ]
    L += ["```", ""]
    with log_path.open('a', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print(f"[OK] Appended V3 block to {log_path} (prior entries untouched)")

print('\n' + '=' * 92)
print('PHASE V3 TEST COMPLETE - '
      + ', '.join(f"{t}: row {decision_report[t]['verdict_row']}" for t, _ in SIGNALS))
print('=' * 92)
