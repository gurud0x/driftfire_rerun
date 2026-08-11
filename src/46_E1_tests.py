import pandas as pd
import numpy as np
import json
import shutil
from datetime import datetime
from pathlib import Path
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# Phase E1: Tests 1-3 against the LOCKED pre-registration at
# results/prereg_E1.md (locked 2026-07-30).
#
# DEV WINDOW ONLY. There is no holdout code path in this script. The 2022-2025
# window has already been spent as holdout for R1/R2/V1/V2/K1; E1 makes no
# holdout claim (prereg section 3.1).
#
# ===========================================================================
# IMPLEMENTATION DECISIONS REQUIRED BY, BUT NOT SPECIFIED IN, THE LOCKED
# PREREG. All are fixed here BEFORE any outcome is computed, are echoed into
# results/46_E1_tests.json, and none of them relaxes a locked bar.
# ===========================================================================
#
# D1. ENTRY AND RETURN CONVENTION. Entry is the open of session t+1 where t is
#     the episode-start (or trigger) session, per prereg section 5. The h-session
#     forward return is close(t+h) / open(t+1) - 1, i.e. the window spans
#     sessions t+1..t+h inclusive. A window is used only if all h sessions
#     exist for that PERMNO inside DEV; otherwise the episode is dropped for
#     that horizon and counted (prereg section 4.3).
#
# D2. BARRIER TIES. A barrier race (+kATR vs -kATR) is resolved by first
#     touching session using daily high/low. If BOTH barriers are touched
#     within the same session, the ordering is unobservable at daily
#     frequency: that episode is EXCLUDED from that particular race and
#     counted. Ties are not assigned to either side, in either direction.
#
# D3. FAMA-MACBETH STANDARDIZATION. compression_ratio, continuous_residency_
#     depth (AMENDED, see BLOCKER note below) and ribbon_width are z-scored
#     cross-sectionally WITHIN each episode-start
#     date before the interaction is formed, so the interaction is a product
#     of demeaned variables and its coefficient is not an artifact of the
#     raw scales. Controls enter unstandardized.
#
# D4. DATE-CONSTANT CONTROLS. The market-vol regime control of prereg
#     section 5.1 is constant within an episode-start date and is therefore
#     absorbed by the per-date intercept of the Fama-MacBeth cross-sectional
#     regression by construction. It is not entered as a separate regressor
#     (it would be perfectly collinear with the intercept). This is a
#     property of the FM estimator, not a weakening of the control set.
#
# D5. MINIMUM CROSS-SECTION. An episode-start date enters the Fama-MacBeth
#     average only if it has at least MIN_XSEC usable observations and a
#     full-column-rank design matrix. Dates failing either are dropped and
#     counted.
#
# D6. TEST 1b MATCHED-CONTROL ESTIMATOR. Controls are in-universe PERMNO-days
#     that are NOT episode starts, matched on (date, size decile, gsector,
#     prior-realized-vol quintile). For episode i in cell c, the matched rate
#     p_c is the control mean in that cell; the per-episode contrast is
#     diff_i = y_i - p_c(i). The reported margin is mean(diff_i) and the
#     z-statistic uses a standard error clustered by episode-start date, per
#     the prereg section 7 clustering convention. Episodes whose cell contains
#     no control are dropped and counted.
#
# D7. TEST 3 BREAKOUT REFERENCE. The prereg specifies arm D's trigger as
#     "close above the high of the active episode" (running max from the
#     episode-start session through s-1). Arms A/B/C have no episode, so they
#     have no such reference; they use "close above the trailing 10-session
#     max high", the 10 sessions matching the primary residency window. This
#     asymmetry is inherent to comparing an episode-conditioned arm against
#     unconditioned ones. To quantify how much of the D-vs-A gap it drives, a
#     diagnostic arm D_ALT is also reported: identical to D but using the
#     trailing-10 reference. D_ALT has NO verdict authority; condition 3c is
#     judged on D as specified.
#
# D8. TRIGGER SEARCH WINDOW. A breakout trigger is searched over the
#     TRIGGER_WINDOW sessions following the episode start. Episodes with no
#     qualifying trigger in that window contribute no trade and are counted.
#
# D9. ARM B COMPRESSION CRITERION. "Compression" for arms B and D means
#     compression_ratio in the most-compressed cross-sectional decile on that
#     date, matching V1's convention (decile 1 = most compressed).
#
# D10. NO WINSORIZATION. Returns and excursions are not winsorized or
#      trimmed. Outliers are retained.
#
# D11. FORWARD REALIZED VOL uses DlyRet over the forward window, annualized
#      by sqrt(252).
#
# ===========================================================================
# BLOCKER FOUND ON FIRST RUN (2026-07-30) - condition 1a is NOT ESTIMABLE
# under the locked primary specification. Recorded here before any test
# statistic was computed.
#
#   residency_10_hits == 6.0 on ALL 67,224 primary episode rows (and on all
#   thr0 / thr05 grid episode sets), so residency_10 == 0.6 identically and
#   its cross-sectional standard deviation is 1.1e-16.
#
#   This is structural, not a data defect. Prereg section 5 defines an
#   episode as the FIRST session of a run where residency_10_hits >= 6. A
#   rolling 10-session sum of a 0/1 flag changes by at most 1 per session,
#   so the first session that crosses the >= 6 threshold necessarily has
#   EXACTLY 6 hits. Every episode-start row is therefore pinned to 0.6.
#
#   Consequence: the compression x residency interaction, on whose t-stat
#   prereg condition 1a is entirely defined, has zero variation and is
#   exactly collinear with the per-date intercept. 1a cannot be estimated
#   at any sample size. Prereg section 5 and section 7 Test 1a are in
#   direct conflict.
#
#   NOTE ON DISCIPLINE: no 1a coefficient, sign, or magnitude has been
#   observed, because none exists to observe. Any amendment resolving this
#   is therefore NOT a post-hoc reaction to an unfavourable result.
#
#   Unaffected and ready to run once 1a is resolved: 1b (matched-control
#   contrast, no regression), Test 2 (state classification uses
#   ema8/ema21/ema8_slope), Test 3 (arms), and the section 6 grid over
#   W in {5, 15, 20, 25}, which DO vary on episode rows (std 0.09 to 0.22).
#
#   The script halts before any test runs and does not write gate_log.md.
# ===========================================================================
#
# No result in this file may be used to alter the pre-registration. gate_log.md
# receives numbers only, no interpretation.
# ===========================================================================

# --- Declared assumptions (not results) -----------------------------------
DEV_START = pd.Timestamp('2015-01-01')
DEV_END = pd.Timestamp('2021-12-31')

RES_WIN_PRIMARY = 10
THRESH_PRIMARY = 'thr025'
VOL_MULT_PRIMARY = 2.0
VOL_MULT_GRID = [1.5, 2.0, 3.0]
THRESH_GRID = ['thr0', 'thr025', 'thr05']
RES_WIN_GRID = [5, 10, 15, 20, 25]

HORIZONS = [5, 10, 20]
PRIMARY_HORIZON = 10
MAX_HORIZON = 20
ATR_BARRIERS = [1.0, 2.0]
MOVE_THRESHOLDS_PCT = [0.03, 0.05]

VOL_AVG_WIN = 20
BREAKOUT_LOOKBACK = 10
RANGE_UPPER_FRAC = 0.75
TRIGGER_WINDOW = 20

REGIME_WIN = 10
REGIME_BAND = 0.01

PRIOR_VOL_WIN = 20
RECENT_ABSRET_WIN = 20
TREND_WIN = 120
N_VOL_QUINTILES = 5

MIN_XSEC = 50
NW_MAXLAGS_FM = 20
NW_MAXLAGS_TS = 10
COST_ROUNDTRIP_BPS = 30.0

# Locked bars (prereg section 8)
BAR_1A_T = 3.0
BAR_1B_PP = 3.0
BAR_1B_Z = 3.0
BAR_2_PP = 3.0
BAR_2_Z = 3.0
BAR_2BFLAT_PP = 1.5
BAR_2BFLAT_Z = 2.0
BAR_3A_T = 3.0
BAR_3C_BPS = 50.0
BAR_3C_T = 2.0
NEARPASS_T = 2.0
NEARPASS_MARGIN_FRAC = 0.5

project_root = Path(__file__).parent.parent
panel_path = project_root / 'data' / 'processed' / 'e1_signal_panel.parquet'
crsp_path = project_root / 'data' / 'processed' / 'crsp_combined.parquet'
univ_path = project_root / 'data' / 'processed' / 'universe_membership.parquet'
prereg_path = project_root / 'results' / 'prereg_E1.md'
out_json = project_root / 'results' / '46_E1_tests.json'
log_path = project_root / 'results' / 'gate_log.md'

print('=' * 88)
print('PHASE E1 - TESTS 1-3  (DEV ONLY, locked prereg results/prereg_E1.md)')
print('=' * 88)

if not prereg_path.exists():
    raise SystemExit('STOP: locked pre-registration not found.')
prereg_text = prereg_path.read_text(encoding='utf-8', errors='replace')
if 'Status: LOCKED' not in prereg_text:
    raise SystemExit('STOP: pre-registration is not marked LOCKED.')
print('\n[OK] Locked pre-registration found and verified.')


# --------------------------------------------------------------------------
# Statistical helpers
# --------------------------------------------------------------------------
def nw_tstat(x, maxlags):
    """Newey-West t-stat on the mean of a 1-D series."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan, len(x)
    res = sm.OLS(x, np.ones(len(x))).fit(
        cov_type='HAC', cov_kwds={'maxlags': maxlags})
    return float(res.params[0]), float(res.tvalues[0]), len(x)


def cluster_ols(y, X, clusters):
    """OLS with cluster-robust (CRVE) standard errors. Returns beta, se."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    g = np.asarray(clusters)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X, g = y[ok], X[ok], g[ok]
    n, k = X.shape
    if n <= k:
        return np.full(k, np.nan), np.full(k, np.nan)
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    uniq, gidx = np.unique(g, return_inverse=True)
    G = len(uniq)
    meat = np.zeros((k, k))
    for j in range(G):
        m = gidx == j
        Xg_u = X[m].T @ u[m]
        meat += np.outer(Xg_u, Xg_u)
    corr = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    V = XtX_inv @ meat @ XtX_inv * corr
    return beta, np.sqrt(np.maximum(np.diag(V), 0.0))


def diff_in_props(y, is_treat, clusters):
    """Treated-minus-control mean difference with date-clustered SE."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(is_treat, dtype=float)
    ok = np.isfinite(y) & np.isfinite(t)
    y, t, c = y[ok], t[ok], np.asarray(clusters)[ok]
    if t.sum() < 2 or (1 - t).sum() < 2:
        return np.nan, np.nan, int(t.sum()), int((1 - t).sum())
    X = np.column_stack([np.ones(len(y)), t])
    beta, se = cluster_ols(y, X, c)
    z = beta[1] / se[1] if se[1] > 0 else np.nan
    return float(beta[1]), float(z), int(t.sum()), int((1 - t).sum())


def mean_with_cluster_se(v, clusters):
    v = np.asarray(v, dtype=float)
    ok = np.isfinite(v)
    v, c = v[ok], np.asarray(clusters)[ok]
    if len(v) < 3:
        return np.nan, np.nan, len(v)
    beta, se = cluster_ols(v, np.ones((len(v), 1)), c)
    z = beta[0] / se[0] if se[0] > 0 else np.nan
    return float(beta[0]), float(z), len(v)


# --------------------------------------------------------------------------
# 1. Load daily CRSP (DEV only) and rebuild ribbon features
# --------------------------------------------------------------------------
print('\n' + '-' * 88)
print('1. DAILY PANEL AND FEATURE REBUILD')
print('-' * 88)

univ = pd.read_parquet(univ_path)
univ_in = univ[univ['in_universe']][['PERMNO', 'year_month', 'decile']].copy()
ever = univ_in['PERMNO'].unique()

need = ['PERMNO', 'DlyCalDt', 'DlyOpen', 'DlyHigh', 'DlyLow', 'DlyClose',
        'DlyRet', 'DlyVol', 'DlyPrcVol', 'DlyCap', 'vwretd']
d = pd.read_parquet(crsp_path, columns=need)
d = d[d['PERMNO'].isin(ever)]
d = d[(d['DlyCalDt'] >= DEV_START) & (d['DlyCalDt'] <= DEV_END)]
d = d.drop_duplicates(['PERMNO', 'DlyCalDt'], keep='first')
d = d.sort_values(['PERMNO', 'DlyCalDt']).reset_index(drop=True)
print(f"Daily rows (DEV, ever-in-universe): {len(d):,}  "
      f"PERMNOs: {d['PERMNO'].nunique():,}")
print(f"Date range: {d['DlyCalDt'].min().date()} to {d['DlyCalDt'].max().date()}")

g = d.groupby('PERMNO', sort=False)
d['ema8'] = g['DlyClose'].transform(
    lambda s: s.ewm(span=8, adjust=False, min_periods=8).mean().shift(1))
d['ema21'] = g['DlyClose'].transform(
    lambda s: s.ewm(span=21, adjust=False, min_periods=21).mean().shift(1))
d['ribbon_low'] = np.minimum(d['ema8'], d['ema21'])
d['ribbon_high'] = np.maximum(d['ema8'], d['ema21'])

# ATR14 (shift(1) aligned) - same definition as src/45, needed both for the
# barrier races below and for continuous_residency_depth just below.
prev_c = g['DlyClose'].shift(1)
tr = pd.concat([(d['DlyHigh'] - d['DlyLow']).abs(),
                (d['DlyHigh'] - prev_c).abs(),
                (d['DlyLow'] - prev_c).abs()], axis=1).max(axis=1)
d['_tr'] = tr
d['atr14'] = g['_tr'].transform(
    lambda s: s.rolling(14, min_periods=14).mean().shift(1))
del d['_tr']  # in-place delete (not .drop(), which would rebind d to a new
              # object and break the lazy g -> d column lookup used below)

# Body-to-ribbon geometry and normalized_body_distance - same definition as
# src/45 (open_lag1/close_lag1 body interval vs ribbon, divided by atr14).
# Reproduced here (not read from the panel) because the panel carries this
# value only on episode-start rows; the amended 1a term (continuous_
# residency_depth, prereg section 5.1a) needs the full daily series to take
# a trailing 10-session mean.
open_lag1 = g['DlyOpen'].shift(1)
close_lag1 = g['DlyClose'].shift(1)
body_low = np.minimum(open_lag1, close_lag1)
body_high = np.maximum(open_lag1, close_lag1)
body_overlaps = (body_low <= d['ribbon_high']) & (body_high >= d['ribbon_low'])
gap_below = d['ribbon_low'] - body_high
gap_above = body_low - d['ribbon_high']
d['body_to_ribbon_distance'] = np.where(
    body_overlaps, 0.0,
    np.where(body_high < d['ribbon_low'], gap_below, gap_above))
d['normalized_body_distance'] = np.where(
    d['atr14'] > 0, d['body_to_ribbon_distance'] / d['atr14'], np.nan)

# continuous_residency_depth (prereg section 5.1a, AMENDED 2026-07-30):
# mean of normalized_body_distance over the 10 sessions ending at, and
# including, the current session. normalized_body_distance is already
# shift(1)-aligned, so this rolling mean introduces no further lookahead.
# Lower value = tighter/deeper residency.
d['continuous_residency_depth'] = g['normalized_body_distance'].transform(
    lambda s: s.rolling(10, min_periods=10).mean())

d['prior_vol'] = g['DlyRet'].transform(
    lambda s: s.rolling(PRIOR_VOL_WIN, min_periods=PRIOR_VOL_WIN).std().shift(1)
) * np.sqrt(252.0)
d['recent_absret'] = g['DlyClose'].transform(
    lambda s: (s / s.shift(RECENT_ABSRET_WIN) - 1.0).shift(1)).abs()
d['trend'] = g['DlyClose'].transform(
    lambda s: (s / s.shift(TREND_WIN) - 1.0).shift(1))
d['avg_vol20'] = g['DlyVol'].transform(
    lambda s: s.rolling(VOL_AVG_WIN, min_periods=VOL_AVG_WIN).mean().shift(1))
d['trail_high10'] = g['DlyHigh'].transform(
    lambda s: s.rolling(BREAKOUT_LOOKBACK, min_periods=BREAKOUT_LOOKBACK)
    .max().shift(1))

rng = (d['DlyHigh'] - d['DlyLow'])
d['range_pos'] = np.where(rng > 0, (d['DlyClose'] - d['DlyLow']) / rng, np.nan)
d['vol_mult'] = np.where(d['avg_vol20'] > 0, d['DlyVol'] / d['avg_vol20'], np.nan)

print('[OK] Rebuilt ema8/ema21/ribbon and daily controls (all shift(1) aligned).')

# Market-direction regime (date level), prereg section 5.1
mkt = d[['DlyCalDt', 'vwretd']].drop_duplicates('DlyCalDt').sort_values('DlyCalDt')
mkt['mkt_10d'] = ((1.0 + mkt['vwretd']).rolling(REGIME_WIN, min_periods=REGIME_WIN)
                  .apply(np.prod, raw=True) - 1.0).shift(1)
mkt['regime'] = np.where(mkt['mkt_10d'] >= REGIME_BAND, 'mkt_up',
                  np.where(mkt['mkt_10d'] <= -REGIME_BAND, 'mkt_down', 'mkt_flat'))
mkt.loc[mkt['mkt_10d'].isna(), 'regime'] = 'undefined'
print('Market-direction regime (vwretd, trailing %d sessions, +/-%.1f%%):'
      % (REGIME_WIN, REGIME_BAND * 100))
for r, n in mkt['regime'].value_counts().items():
    print(f"    {r:<10} {n:>5,} sessions  ({n / len(mkt) * 100:5.1f}%)")

# Market-vol regime (date level control, absorbed by FM intercept per D4)
mvr = d.groupby('DlyCalDt')['prior_vol'].median().rename('mkt_vol_regime')

# --------------------------------------------------------------------------
# 2. Calendar assertions (prereg section 4)
# --------------------------------------------------------------------------
print('\n' + '-' * 88)
print('2. CALENDAR ASSERTIONS (prereg section 4)')
print('-' * 88)

cal_all = pd.DatetimeIndex(sorted(
    pd.read_parquet(crsp_path, columns=['DlyCalDt'])['DlyCalDt'].unique()))
dev_cal = cal_all[(cal_all >= DEV_START) & (cal_all <= DEV_END)]
print(f"DEV sessions: {len(dev_cal):,}  ({dev_cal.min().date()} to "
      f"{dev_cal.max().date()})")
assert len(dev_cal) == 1763, 'DEV session count changed'

warm = {22: '2015-02-03', 31: '2015-02-17', 46: '2015-03-10'}
for s, expect in warm.items():
    got = str(cal_all[s - 1].date())
    print(f"  session {s:>2} -> {got}   (prereg says {expect})  "
          f"{'[OK]' if got == expect else '[MISMATCH]'}")
    assert got == expect, f'warmup session {s} moved'

trunc = {5: '2021-12-23', 10: '2021-12-16', 20: '2021-12-02'}
for h, expect in trunc.items():
    got = str(dev_cal[-(h + 1)].date())
    print(f"  last eligible start, h={h:>2} -> {got}   (prereg says {expect})  "
          f"{'[OK]' if got == expect else '[MISMATCH]'}")
    assert got == expect, f'truncation date for h={h} moved'
print('[PASS] All prereg section 4 dates re-derived and matched.')

# --------------------------------------------------------------------------
# 3. Episode panel, provenance assertion, common-start filter
# --------------------------------------------------------------------------
print('\n' + '-' * 88)
print('3. EPISODE PANEL (prereg sections 4.2, 5, 10.2)')
print('-' * 88)

ep = pd.read_parquet(panel_path)
print(f"Panel rows: {len(ep):,}  PERMNOs: {ep['PERMNO'].nunique():,}  "
      f"{ep['DlyCalDt'].min().date()} to {ep['DlyCalDt'].max().date()}")

chk = ep[['PERMNO', 'DlyCalDt', 'ema8', 'ema21', 'normalized_body_distance']].merge(
    d[['PERMNO', 'DlyCalDt', 'ema8', 'ema21', 'normalized_body_distance']],
    on=['PERMNO', 'DlyCalDt'], how='left', suffixes=('_panel', '_recomp'))
for c in ['ema8', 'ema21', 'normalized_body_distance']:
    md = (chk[f'{c}_panel'] - chk[f'{c}_recomp']).abs().max()
    print(f"  provenance check {c}: max|panel - recomputed| = {md:.3e}  "
          f"{'[PASS]' if md < 1e-9 else '[FAIL]'}")
    assert md < 1e-9, f'{c} does not reproduce from crsp_combined'

n_before = int(ep['episode_start'].sum())
ep['eligible_common_start'] = ep['residency_25'].notna()
n_excl = int((ep['episode_start'] & ~ep['eligible_common_start']).sum())
print(f"\nPrimary episodes (episode_start, {THRESH_PRIMARY}): {n_before:,}")
print(f"  excluded by common-start rule (residency_25 null): {n_excl:,} "
      f"({n_excl / n_before * 100:.2f}%)")
print(f"  surviving: {n_before - n_excl:,}")

ep = ep.merge(mvr, on='DlyCalDt', how='left')
ep = ep.merge(mkt[['DlyCalDt', 'regime', 'mkt_10d']], on='DlyCalDt', how='left')
ep['year_month'] = ep['DlyCalDt'].dt.to_period('M').astype(str)
ep = ep.merge(univ_in, on=['PERMNO', 'year_month'], how='left')

# --------------------------------------------------------------------------
# 4. Forward-path machinery
# --------------------------------------------------------------------------
print('\n' + '-' * 88)
print('4. FORWARD OUTCOME CONSTRUCTION')
print('-' * 88)

d = d.reset_index(drop=True)
d['pos'] = np.arange(len(d))
pn = d['PERMNO'].to_numpy()
a_open = d['DlyOpen'].to_numpy(float)
a_high = d['DlyHigh'].to_numpy(float)
a_low = d['DlyLow'].to_numpy(float)
a_close = d['DlyClose'].to_numpy(float)
a_ret = d['DlyRet'].to_numpy(float)
a_rlo = d['ribbon_low'].to_numpy(float)
a_rhi = d['ribbon_high'].to_numpy(float)
a_volm = d['vol_mult'].to_numpy(float)
a_rpos = d['range_pos'].to_numpy(float)
a_th10 = d['trail_high10'].to_numpy(float)
N = len(d)

pos_lookup = d.set_index(['PERMNO', 'DlyCalDt'])['pos']


def build_paths(positions, horizon):
    """Return dict of (m x horizon) arrays for sessions t+1..t+horizon,
    plus validity mask and entry price. Positions crossing a PERMNO
    boundary or the end of DEV are marked invalid."""
    p = np.asarray(positions, dtype=np.int64)
    off = np.arange(1, horizon + 1, dtype=np.int64)
    idx = p[:, None] + off[None, :]
    inb = idx < N
    idxc = np.clip(idx, 0, N - 1)
    same = pn[idxc] == pn[p][:, None]
    valid = inb & same
    def pick(arr):
        out = np.where(valid, arr[idxc], np.nan)
        return out
    entry_pos = np.clip(p + 1, 0, N - 1)
    entry_ok = valid[:, 0]
    entry = np.where(entry_ok, a_open[entry_pos], np.nan)
    return {
        'valid': valid, 'entry': entry, 'entry_pos': entry_pos,
        'high': pick(a_high), 'low': pick(a_low), 'close': pick(a_close),
        'ret': pick(a_ret), 'rlo': pick(a_rlo), 'rhi': pick(a_rhi),
        'idx': idxc,
    }


def horizon_outcomes(paths, atr, horizon):
    v = paths['valid'][:, :horizon]
    complete = v.all(axis=1) & np.isfinite(paths['entry'])
    entry = paths['entry']
    hi = paths['high'][:, :horizon]
    lo = paths['low'][:, :horizon]
    cl = paths['close'][:, :horizon]
    rt = paths['ret'][:, :horizon]
    with np.errstate(invalid='ignore', divide='ignore'):
        fwd_ret = cl[:, horizon - 1] / entry - 1.0
        mfe = np.nanmax(hi, axis=1) / entry - 1.0
        mae = np.nanmin(lo, axis=1) / entry - 1.0
        fvol = np.nanstd(rt, axis=1, ddof=1) * np.sqrt(252.0)
        max_up_atr = (np.nanmax(hi, axis=1) - entry) / atr
        max_dn_atr = (entry - np.nanmin(lo, axis=1)) / atr
    fwd_ret = np.where(complete, fwd_ret, np.nan)
    mfe = np.where(complete, mfe, np.nan)
    mae = np.where(complete, mae, np.nan)
    fvol = np.where(complete, fvol, np.nan)
    max_abs = np.fmax(np.abs(mfe), np.abs(mae))
    max_atr = np.fmax(max_up_atr, max_dn_atr)
    max_atr = np.where(complete, max_atr, np.nan)
    return {
        'complete': complete, 'fwd_ret': fwd_ret, 'mfe': mfe, 'mae': mae,
        'fwd_vol': fvol, 'max_abs_move': max_abs, 'max_atr_move': max_atr,
    }


def barrier_race(paths, atr, k, horizon):
    """+kATR vs -kATR first-touch. Returns up_first (1/0/nan) and tie count."""
    v = paths['valid'][:, :horizon]
    complete = v.all(axis=1) & np.isfinite(paths['entry'])
    entry = paths['entry'][:, None]
    up_lvl = entry + k * atr[:, None]
    dn_lvl = entry - k * atr[:, None]
    up_hit = (paths['high'][:, :horizon] >= up_lvl) & v
    dn_hit = (paths['low'][:, :horizon] <= dn_lvl) & v
    big = horizon + 10
    first_up = np.where(up_hit.any(axis=1), up_hit.argmax(axis=1), big)
    first_dn = np.where(dn_hit.any(axis=1), dn_hit.argmax(axis=1), big)
    tie = (first_up == first_dn) & (first_up < big)
    out = np.full(len(entry), np.nan)
    ok = complete & ~tie
    out[ok & (first_up < first_dn)] = 1.0
    out[ok & (first_dn < first_up)] = 0.0
    neither = ok & (first_up == big) & (first_dn == big)
    out[neither] = np.nan
    return out, int(tie.sum()), int(neither.sum())


ep_primary = ep[ep['episode_start'] & ep['eligible_common_start']].copy()
ep_primary = ep_primary.merge(
    d[['PERMNO', 'DlyCalDt', 'pos', 'prior_vol', 'recent_absret', 'trend',
       'DlyPrcVol', 'continuous_residency_depth']],
    on=['PERMNO', 'DlyCalDt'], how='left')
ep_primary = ep_primary[ep_primary['pos'].notna()].copy()
ep_primary['pos'] = ep_primary['pos'].astype(np.int64)
print(f"Primary episodes joined to daily panel: {len(ep_primary):,}")
n_no_depth = int(ep_primary['continuous_residency_depth'].isna().sum())
print(f"  continuous_residency_depth (prereg 5.1a) null: {n_no_depth:,} "
      f"({n_no_depth / len(ep_primary) * 100:.2f}%) - dropped for Test 1 only")
_depth_sd = float(ep_primary['continuous_residency_depth'].std())
print(f"  continuous_residency_depth std across episodes: {_depth_sd:.6g}  "
      f"(non-degenerate check)")

paths_p = build_paths(ep_primary['pos'].to_numpy(), MAX_HORIZON)
atr_p = ep_primary['atr14'].to_numpy(float)

drops = {'common_start_rule': n_excl,
         'test1_missing_continuous_residency_depth': n_no_depth}
for h in HORIZONS:
    oc = horizon_outcomes(paths_p, atr_p, h)
    for key, val in oc.items():
        if key == 'complete':
            continue
        ep_primary[f'{key}_{h}'] = val
    ep_primary[f'complete_{h}'] = oc['complete']
    drops[f'incomplete_window_h{h}'] = int((~oc['complete']).sum())
    print(f"  h={h:>2}: complete windows {int(oc['complete'].sum()):,} / "
          f"{len(ep_primary):,}  (dropped {int((~oc['complete']).sum()):,})")

tie_counts = {}
for k in ATR_BARRIERS:
    up, nt, nn = barrier_race(paths_p, atr_p, k, PRIMARY_HORIZON)
    ep_primary[f'up_first_{int(k)}atr'] = up
    tie_counts[f'{int(k)}atr_same_session_ties'] = nt
    tie_counts[f'{int(k)}atr_neither_touched'] = nn
    print(f"  +/-{k:.0f}ATR race (h={PRIMARY_HORIZON}): resolved "
          f"{int(np.isfinite(up).sum()):,}, same-session ties {nt:,}, "
          f"neither touched {nn:,}")

# time to leave ribbon
v20 = paths_p['valid']
outside = ((paths_p['close'] > paths_p['rhi']) |
           (paths_p['close'] < paths_p['rlo'])) & v20
ttl = np.where(outside.any(axis=1), outside.argmax(axis=1) + 1.0, np.nan)
ep_primary['time_to_leave_ribbon'] = ttl
print(f"  time-to-leave-ribbon resolved for {int(np.isfinite(ttl).sum()):,} "
      f"episodes (median {np.nanmedian(ttl):.1f} sessions)")

for pct in MOVE_THRESHOLDS_PCT:
    ep_primary[f'move_{int(pct*100)}pct'] = (
        ep_primary[f'max_abs_move_{PRIMARY_HORIZON}'] >= pct).astype(float)
    ep_primary.loc[ep_primary[f'max_abs_move_{PRIMARY_HORIZON}'].isna(),
                   f'move_{int(pct*100)}pct'] = np.nan
ep_primary['move_2atr'] = (
    ep_primary[f'max_atr_move_{PRIMARY_HORIZON}'] >= 2.0).astype(float)
ep_primary.loc[ep_primary[f'max_atr_move_{PRIMARY_HORIZON}'].isna(),
               'move_2atr'] = np.nan

# --------------------------------------------------------------------------
# 5. TEST 1 - MAGNITUDE
# --------------------------------------------------------------------------
print('\n' + '=' * 88)
print('TEST 1 - MAGNITUDE')
print('=' * 88)

# AMENDED 2026-07-30 (prereg section 5.1a / section 13 item 11): the
# residency term is continuous_residency_depth, not residency_10.
# residency_10 is constant (0.6) on every primary episode by construction
# (see the module docstring BLOCKER note) and is excluded from this model
# entirely - including it alongside continuous_residency_depth would add a
# constant column to the design matrix and reproduce the same collinearity
# this amendment exists to fix.
SIG_COLS = ['compression_ratio', 'continuous_residency_depth', 'ribbon_width']
CTRL_COLS = ['prior_vol', 'log_cap', 'log_price', 'log_dvol',
             'recent_absret', 'trend']

ep_primary['log_cap'] = np.log(ep_primary['DlyCap'].clip(lower=1e-6))
ep_primary['log_price'] = np.log(ep_primary['DlyClose'].abs().clip(lower=1e-6))
ep_primary['log_dvol'] = np.log(ep_primary['DlyPrcVol'].clip(lower=1e-6))


def fama_macbeth(df, ycol, sig_cols, res_col='continuous_residency_depth'):
    """Per-date cross-sectional OLS; returns coefficient time series."""
    use = df.dropna(subset=[ycol, 'gsector'] + sig_cols + CTRL_COLS).copy()
    rows = []
    n_dropped_dates = 0
    for dt, gg in use.groupby('DlyCalDt', sort=True):
        if len(gg) < MIN_XSEC:
            n_dropped_dates += 1
            continue
        z = {}
        for c in sig_cols:
            v = gg[c].to_numpy(float)
            sd = v.std(ddof=0)
            z[c] = (v - v.mean()) / sd if sd > 0 else np.zeros(len(v))
        inter = z[sig_cols[0]] * z[res_col]
        secs = pd.get_dummies(gg['gsector'].astype(int), drop_first=True,
                              dtype=float).to_numpy()
        X = np.column_stack(
            [np.ones(len(gg))] + [z[c] for c in sig_cols] + [inter] +
            [gg[c].to_numpy(float) for c in CTRL_COLS] +
            ([secs] if secs.shape[1] else []))
        y = gg[ycol].to_numpy(float)
        if not np.isfinite(X).all() or np.linalg.matrix_rank(X) < X.shape[1]:
            n_dropped_dates += 1
            continue
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        rows.append({'DlyCalDt': dt, 'n': len(gg),
                     'b_comp': beta[1], 'b_res': beta[2],
                     'b_width': beta[3], 'b_inter': beta[4]})
    return pd.DataFrame(rows), n_dropped_dates


ycol = f'max_abs_move_{PRIMARY_HORIZON}'

# --- ESTIMABILITY PRE-FLIGHT (amended term; see BLOCKER note in the module
# docstring for the original residency_10 failure this replaces) -----------
# continuous_residency_depth (prereg section 5.1a) is a continuous mean of
# normalized_body_distance, not gated by the >=6-hits threshold that pinned
# residency_10 to a constant. Checked anyway, defensively, before any
# coefficient is computed - if it were ever degenerate the halt-before-write
# discipline from the original 1a failure applies identically here.
_depth_sd_check = float(ep_primary['continuous_residency_depth'].std())
if not (_depth_sd_check > 1e-10):
    print('\n' + '!' * 88)
    print('STOP - continuous_residency_depth IS DEGENERATE; 1a NOT ESTIMABLE')
    print('!' * 88)
    print(f"  std across {len(ep_primary):,} primary episodes: "
          f"{_depth_sd_check:.3e}")
    print("  No test was run. gate_log.md was NOT written.")
    print('!' * 88)
    raise SystemExit(2)
print(f"[OK] continuous_residency_depth is non-degenerate "
      f"(std={_depth_sd_check:.4f}); proceeding with amended 1a.")

fm, n_dd = fama_macbeth(ep_primary, ycol, SIG_COLS)
if len(fm) == 0:
    print(f"\nSTOP: no usable Fama-MacBeth dates "
          f"(all {n_dd:,} dates below MIN_XSEC={MIN_XSEC} or rank-deficient). "
          f"gate_log.md was NOT written.")
    raise SystemExit(2)
print(f"Fama-MacBeth on {ycol}: {len(fm):,} usable dates "
      f"(dropped {n_dd:,} dates below MIN_XSEC={MIN_XSEC} or rank-deficient)")
drops['fm_dates_dropped'] = n_dd

t1_res = {}
for lbl, col in [('compression', 'b_comp'), ('continuous_residency_depth', 'b_res'),
                 ('ribbon_width', 'b_width'), ('interaction', 'b_inter')]:
    m, t, n = nw_tstat(fm[col], NW_MAXLAGS_FM)
    t1_res[lbl] = {'mean_coef': m, 'nw_t': t, 'n_dates': n}
    print(f"  {lbl:<28} mean coef {m:+.6f}   NW t {t:+.3f}   ({n} dates)")

# AMENDED sign: continuous_residency_depth is coded opposite residency_10
# (lower = more resident), so the hypothesized interaction sign is NEGATIVE
# (prereg section 7 Test 1, section 13 item 11) - stated before this line
# was ever executed with real data.
t1a_val = t1_res['interaction']['nw_t']
t1a_pass = bool(np.isfinite(t1a_val) and t1a_val <= -BAR_1A_T)
print(f"\n  1a (AMENDED): interaction "
      f"(compression_ratio x continuous_residency_depth) |t| >= {BAR_1A_T} "
      f"and NEGATIVE  ->  t = {t1a_val:+.3f}   "
      f"{'MET' if t1a_pass else 'NOT MET'}")

# --- 1b matched controls ---------------------------------------------------
print('\n  1b: matched-control contrast on P(|move| > 2 ATR within 10 days)')
ctrl = d[['PERMNO', 'DlyCalDt', 'pos', 'prior_vol', 'DlyCap']].copy()
ctrl['year_month'] = ctrl['DlyCalDt'].dt.to_period('M').astype(str)
ctrl = ctrl.merge(univ_in, on=['PERMNO', 'year_month'], how='inner')

ep_keys = set(zip(ep['PERMNO'], ep['DlyCalDt']))
ctrl['is_ep'] = [k in ep_keys for k in zip(ctrl['PERMNO'], ctrl['DlyCalDt'])]
ctrl = ctrl[~ctrl['is_ep']].copy()
print(f"    control pool (in-universe, non-episode PERMNO-days): {len(ctrl):,}")

sec_map = ep[['PERMNO', 'DlyCalDt', 'gsector']].drop_duplicates()
sec_by_permno = (sec_map.dropna(subset=['gsector'])
                 .sort_values('DlyCalDt')
                 .drop_duplicates('PERMNO', keep='last')
                 .set_index('PERMNO')['gsector'])
ctrl['gsector'] = ctrl['PERMNO'].map(sec_by_permno)
ctrl = ctrl.dropna(subset=['gsector', 'prior_vol', 'decile'])

ctrl_paths = build_paths(ctrl['pos'].to_numpy(np.int64), PRIMARY_HORIZON)

# atr14 was already built in section 1 (needed there for
# continuous_residency_depth); reused here for the control ATR, not rebuilt.
a_atr = d['atr14'].to_numpy(float)
ctrl_atr = a_atr[ctrl['pos'].to_numpy(np.int64)]

oc_c = horizon_outcomes(ctrl_paths, ctrl_atr, PRIMARY_HORIZON)
ctrl['y'] = np.where(oc_c['complete'], (oc_c['max_atr_move'] >= 2.0).astype(float),
                     np.nan)
ctrl = ctrl.dropna(subset=['y'])

ctrl['vol_q'] = ctrl.groupby('DlyCalDt')['prior_vol'].transform(
    lambda s: pd.qcut(s, N_VOL_QUINTILES, labels=False, duplicates='drop'))
cells = (ctrl.dropna(subset=['vol_q'])
         .groupby(['DlyCalDt', 'decile', 'gsector', 'vol_q'])['y']
         .agg(['mean', 'size']).reset_index()
         .rename(columns={'mean': 'p_cell', 'size': 'n_cell'}))

epb = ep_primary.dropna(subset=['move_2atr', 'prior_vol', 'decile',
                                'gsector']).copy()
vq_edges = ctrl.dropna(subset=['vol_q'])[['DlyCalDt', 'prior_vol', 'vol_q']]
vq_bounds = (vq_edges.groupby(['DlyCalDt', 'vol_q'])['prior_vol']
             .max().reset_index().rename(columns={'prior_vol': 'ub'}))
epb = epb.sort_values(['DlyCalDt', 'prior_vol'])
assign = []
for dt, gg in epb.groupby('DlyCalDt', sort=True):
    b = vq_bounds[vq_bounds['DlyCalDt'] == dt].sort_values('ub')
    if len(b) == 0:
        assign.append(pd.Series(np.nan, index=gg.index))
        continue
    q = np.searchsorted(b['ub'].to_numpy(), gg['prior_vol'].to_numpy(),
                        side='left')
    q = np.clip(q, 0, len(b) - 1)
    assign.append(pd.Series(b['vol_q'].to_numpy()[q], index=gg.index))
epb['vol_q'] = pd.concat(assign).reindex(epb.index)

epb = epb.merge(cells, on=['DlyCalDt', 'decile', 'gsector', 'vol_q'], how='left')
n_nocell = int(epb['p_cell'].isna().sum())
drops['episodes_without_matched_control_cell'] = n_nocell
epb_ok = epb.dropna(subset=['p_cell']).copy()
epb_ok['diff'] = epb_ok['move_2atr'] - epb_ok['p_cell']
print(f"    episodes matched to a control cell: {len(epb_ok):,} "
      f"(dropped, no cell: {n_nocell:,})")

t1b_margin, t1b_z, t1b_n = mean_with_cluster_se(
    epb_ok['diff'].to_numpy(), epb_ok['DlyCalDt'].to_numpy())
ep_rate = float(epb_ok['move_2atr'].mean())
ct_rate = float(epb_ok['p_cell'].mean())
t1b_margin_pp = t1b_margin * 100.0
t1b_pass = bool(np.isfinite(t1b_margin_pp) and t1b_margin_pp >= BAR_1B_PP
                and abs(t1b_z) >= BAR_1B_Z)
print(f"    episode rate {ep_rate*100:.2f}%   matched control rate "
      f"{ct_rate*100:.2f}%")
print(f"    margin {t1b_margin_pp:+.2f} pp   z {t1b_z:+.3f}   (n={t1b_n:,})")
print(f"  1b: margin >= {BAR_1B_PP} pp and |z| >= {BAR_1B_Z}  ->  "
      f"{'MET' if t1b_pass else 'NOT MET'}")

test1_pass = bool(t1a_pass and t1b_pass)
print(f"\nTEST 1 VERDICT: {'PASS' if test1_pass else 'FAIL'}")

# --------------------------------------------------------------------------
# 6. TEST 2 - DIRECTION
# --------------------------------------------------------------------------
print('\n' + '=' * 88)
print('TEST 2 - DIRECTION')
print('=' * 88)

ep_primary['state'] = np.where(
    (ep_primary['ema8'] > ep_primary['ema21']) & (ep_primary['ema8_slope'] > 0),
    'bullish',
    np.where((ep_primary['ema8'] < ep_primary['ema21']) &
             (ep_primary['ema8_slope'] < 0), 'bearish', 'neutral'))
print('Episode state counts:')
for s, n in ep_primary['state'].value_counts().items():
    print(f"    {s:<8} {n:>7,}  ({n / len(ep_primary) * 100:5.1f}%)")

t2 = ep_primary.dropna(subset=['up_first_2atr']).copy()
t2['dn_first_2atr'] = 1.0 - t2['up_first_2atr']
print(f"\nEpisodes with a resolved +/-2ATR race: {len(t2):,}")

print('\nBarrier probabilities by state (h=%d):' % PRIMARY_HORIZON)
print(f"  {'state':<9} {'n':>8} {'P(+2ATR first)':>15} {'P(-2ATR first)':>15}")
state_rates = {}
for s in ['bullish', 'bearish', 'neutral']:
    m = t2['state'] == s
    if m.sum() == 0:
        continue
    up = float(t2.loc[m, 'up_first_2atr'].mean())
    state_rates[s] = {'n': int(m.sum()), 'p_up_first': up,
                      'p_dn_first': 1.0 - up}
    print(f"  {s:<9} {int(m.sum()):>8,} {up*100:>14.2f}% {(1-up)*100:>14.2f}%")


def contrast(df, treat_state, ycol):
    sub = df[df['state'].isin([treat_state, 'neutral'])]
    if len(sub) == 0:
        return np.nan, np.nan, 0, 0
    return diff_in_props(sub[ycol].to_numpy(),
                         (sub['state'] == treat_state).to_numpy(),
                         sub['DlyCalDt'].to_numpy())


m2a, z2a, n2a_t, n2a_c = contrast(t2, 'bullish', 'up_first_2atr')
m2a_pp = m2a * 100.0
t2a_pass = bool(np.isfinite(m2a_pp) and m2a_pp >= BAR_2_PP and abs(z2a) >= BAR_2_Z)
print(f"\n  2a: P(+2ATR first|bullish) - neutral = {m2a_pp:+.2f} pp   "
      f"z {z2a:+.3f}   (n_bull {n2a_t:,}, n_neut {n2a_c:,})")
print(f"      bar >= {BAR_2_PP} pp and |z| >= {BAR_2_Z}  ->  "
      f"{'MET' if t2a_pass else 'NOT MET'}")

m2b, z2b, n2b_t, n2b_c = contrast(t2, 'bearish', 'dn_first_2atr')
m2b_pp = m2b * 100.0
t2b_pooled_pass = bool(np.isfinite(m2b_pp) and m2b_pp >= BAR_2_PP
                       and abs(z2b) >= BAR_2_Z)
print(f"\n  2b (pooled): P(-2ATR first|bearish) - neutral = {m2b_pp:+.2f} pp   "
      f"z {z2b:+.3f}   (n_bear {n2b_t:,}, n_neut {n2b_c:,})")
print(f"      bar >= {BAR_2_PP} pp and |z| >= {BAR_2_Z}  ->  "
      f"{'MET' if t2b_pooled_pass else 'NOT MET'}")

print('\n  Regime split (vwretd trailing %d sessions, +/-%.1f%% bands):'
      % (REGIME_WIN, REGIME_BAND * 100))
regime_detail = {}
for rg in ['mkt_up', 'mkt_flat', 'mkt_down']:
    sub = t2[t2['regime'] == rg]
    if len(sub) == 0:
        continue
    ma, za, _, _ = contrast(sub, 'bullish', 'up_first_2atr')
    mb, zb, nbt, nbc = contrast(sub, 'bearish', 'dn_first_2atr')
    regime_detail[rg] = {
        'n_episodes': int(len(sub)),
        'bullish_margin_pp': ma * 100.0, 'bullish_z': za,
        'bearish_margin_pp': mb * 100.0, 'bearish_z': zb,
        'n_bearish': nbt, 'n_neutral': nbc,
    }
    print(f"    {rg:<9} n={len(sub):>7,}   2a-style {ma*100:+6.2f} pp "
          f"(z {za:+6.3f})   2b-style {mb*100:+6.2f} pp (z {zb:+6.3f})")

flat = regime_detail.get('mkt_flat', {})
m2bf_pp = flat.get('bearish_margin_pp', np.nan)
z2bf = flat.get('bearish_z', np.nan)
t2bflat_pass = bool(np.isfinite(m2bf_pp) and m2bf_pp >= BAR_2BFLAT_PP
                    and abs(z2bf) >= BAR_2BFLAT_Z)
print(f"\n  2b-flat (LOAD-BEARING): {m2bf_pp:+.2f} pp   z {z2bf:+.3f}")
print(f"      bar >= {BAR_2BFLAT_PP} pp and |z| >= {BAR_2BFLAT_Z}  ->  "
      f"{'MET' if t2bflat_pass else 'NOT MET'}")

t2b_pass = bool(t2b_pooled_pass and t2bflat_pass)
if t2b_pooled_pass and not t2bflat_pass:
    print("      NOTE: 2b fails on the mkt-flat condition despite the pooled "
          "number meeting its bar.")
test2_pass = bool(t2a_pass and t2b_pass)
print(f"\nTEST 2 VERDICT: {'PASS' if test2_pass else 'FAIL'}  "
      f"(2a {'MET' if t2a_pass else 'NOT MET'}, 2b pooled "
      f"{'MET' if t2b_pooled_pass else 'NOT MET'}, 2b-flat "
      f"{'MET' if t2bflat_pass else 'NOT MET'})")

# --------------------------------------------------------------------------
# 7. TEST 3 - BREAKOUT TRIGGER
# --------------------------------------------------------------------------
print('\n' + '=' * 88)
print('TEST 3 - BREAKOUT TRIGGER')
print('=' * 88)

comp_dec = (ep[['PERMNO', 'DlyCalDt', 'compression_ratio']].dropna()
            .assign(cd=lambda x: x.groupby('DlyCalDt')['compression_ratio']
                    .transform(lambda s: pd.qcut(s, 10, labels=False,
                                                 duplicates='drop'))))
comp_low = set(zip(comp_dec.loc[comp_dec['cd'] == 0, 'PERMNO'],
                   comp_dec.loc[comp_dec['cd'] == 0, 'DlyCalDt']))
print(f"Most-compressed decile PERMNO-days (arm B/D filter): {len(comp_low):,}")


def daily_breakout_mask(vol_mult_thresh):
    return ((a_close > a_th10) & (a_volm >= vol_mult_thresh) &
            (a_rpos >= RANGE_UPPER_FRAC))


def arm_trades_daily(mask_positions, vol_mult_thresh, horizon):
    """Trades entered at open(s+1) for trigger sessions s (positions)."""
    p = np.asarray(mask_positions, dtype=np.int64)
    if len(p) == 0:
        return pd.DataFrame(columns=['entry_date', 'ret'])
    pa = build_paths(p, horizon)
    oc = horizon_outcomes(pa, np.ones(len(p)), horizon)
    ok = oc['complete']
    ent_dates = d['DlyCalDt'].to_numpy()[np.clip(p + 1, 0, N - 1)]
    return pd.DataFrame({'entry_date': ent_dates[ok], 'ret': oc['fwd_ret'][ok],
                         'mfe': oc['mfe'][ok], 'mae': oc['mae'][ok]})


def episode_trigger_positions(sub, vol_mult_thresh, use_episode_ref):
    """First qualifying trigger session for each episode. Returns positions."""
    if len(sub) == 0:
        return np.array([], dtype=np.int64), 0
    p = sub['pos'].to_numpy(np.int64)
    pa = build_paths(p, TRIGGER_WINDOW)
    v = pa['valid']
    idx = pa['idx']
    if use_episode_ref:
        hi_ext = np.concatenate([a_high[p][:, None], pa['high']], axis=1)
        ref = np.maximum.accumulate(np.nan_to_num(hi_ext, nan=-np.inf),
                                    axis=1)[:, :-1]
    else:
        ref = np.where(v, a_th10[idx], np.nan)
    cl = pa['close']
    vm = np.where(v, a_volm[idx], np.nan)
    rp = np.where(v, a_rpos[idx], np.nan)
    trig = (cl > ref) & (vm >= vol_mult_thresh) & (rp >= RANGE_UPPER_FRAC) & v
    has = trig.any(axis=1)
    first = np.where(has, trig.argmax(axis=1), 0)
    trig_pos = idx[np.arange(len(p)), first]
    return trig_pos[has], int((~has).sum())


bull = ep_primary[ep_primary['state'] == 'bullish'].copy()
bull['is_comp'] = [k in comp_low for k in zip(bull['PERMNO'], bull['DlyCalDt'])]
print(f"Bullish primary episodes: {len(bull):,}  "
      f"of which most-compressed decile: {int(bull['is_comp'].sum()):,}")


def build_arms(vol_mult_thresh, horizon=PRIMARY_HORIZON):
    res = {}
    bo = daily_breakout_mask(vol_mult_thresh)
    d_res = d[['PERMNO', 'DlyCalDt']].copy()
    posA = np.flatnonzero(bo & np.isfinite(a_th10))
    res['A'] = arm_trades_daily(posA, vol_mult_thresh, horizon)

    keyB = np.array([k in comp_low for k in
                     zip(d_res['PERMNO'].to_numpy()[posA],
                         d_res['DlyCalDt'].to_numpy()[posA])])
    res['B'] = arm_trades_daily(posA[keyB], vol_mult_thresh, horizon)

    resid_keys = set(zip(ep.loc[ep[f'primary_residency_condition'], 'PERMNO'],
                         ep.loc[ep[f'primary_residency_condition'], 'DlyCalDt']))
    keyC = np.array([k in resid_keys for k in
                     zip(d_res['PERMNO'].to_numpy()[posA],
                         d_res['DlyCalDt'].to_numpy()[posA])])
    res['C'] = arm_trades_daily(posA[keyC], vol_mult_thresh, horizon)

    subD = bull[bull['is_comp']]
    posD, nomiss = episode_trigger_positions(subD, vol_mult_thresh, True)
    res['D'] = arm_trades_daily(posD, vol_mult_thresh, horizon)
    posDa, _ = episode_trigger_positions(subD, vol_mult_thresh, False)
    res['D_ALT'] = arm_trades_daily(posDa, vol_mult_thresh, horizon)
    res['_no_trigger'] = nomiss
    return res


arms = build_arms(VOL_MULT_PRIMARY)
drops['bullish_compressed_episodes_without_trigger'] = arms['_no_trigger']

print(f"\nArm trade counts (volume multiplier {VOL_MULT_PRIMARY}x, "
      f"h={PRIMARY_HORIZON}):")
arm_stats = {}
for a in ['A', 'B', 'C', 'D', 'D_ALT']:
    tr_df = arms[a]
    if len(tr_df) == 0:
        print(f"  arm {a:<5} no trades")
        continue
    daily = tr_df.groupby('entry_date')['ret'].mean()
    mu, t, nd = nw_tstat(daily.to_numpy(), NW_MAXLAGS_TS)
    arm_stats[a] = {
        'n_trades': int(len(tr_df)), 'n_days': nd,
        'mean_daily_ret': mu, 'nw_t': t,
        'mean_trade_ret': float(tr_df['ret'].mean()),
        'mean_mfe': float(tr_df['mfe'].mean()),
        'mean_mae': float(tr_df['mae'].mean()),
    }
    print(f"  arm {a:<5} trades {len(tr_df):>7,}  days {nd:>5,}  "
          f"mean daily ret {mu*100:+7.4f}%  NW t {t:+7.3f}  "
          f"mean trade ret {tr_df['ret'].mean()*100:+7.3f}%")

dD = arms['D'].groupby('entry_date')['ret'].mean() if len(arms['D']) else pd.Series(dtype=float)
dA = arms['A'].groupby('entry_date')['ret'].mean() if len(arms['A']) else pd.Series(dtype=float)

t3a_t = arm_stats.get('D', {}).get('nw_t', np.nan)
t3a_mu = arm_stats.get('D', {}).get('mean_daily_ret', np.nan)
t3a_pass = bool(np.isfinite(t3a_t) and t3a_mu > 0 and t3a_t >= BAR_3A_T)
print(f"\n  3a: arm D positive with NW t >= {BAR_3A_T}  ->  "
      f"mean {t3a_mu*100:+.4f}%/day, t {t3a_t:+.3f}   "
      f"{'MET' if t3a_pass else 'NOT MET'}")

d_trade_mean = arm_stats.get('D', {}).get('mean_trade_ret', np.nan)
net_d = d_trade_mean - COST_ROUNDTRIP_BPS / 1e4
t3b_pass = bool(np.isfinite(net_d) and net_d > 0)
print(f"  3b: arm D mean trade return net of {COST_ROUNDTRIP_BPS:.0f} bps "
      f"(PLACEHOLDER) > 0  ->  {d_trade_mean*100:+.3f}% - "
      f"{COST_ROUNDTRIP_BPS/100:.2f}% = {net_d*100:+.3f}%   "
      f"{'MET' if t3b_pass else 'NOT MET'}")

joint = pd.concat([dD.rename('D'), dA.rename('A')], axis=1).dropna()
if len(joint) >= 3:
    dif = (joint['D'] - joint['A']).to_numpy()
    m3c, t3c, n3c = nw_tstat(dif, NW_MAXLAGS_TS)
else:
    m3c, t3c, n3c = np.nan, np.nan, 0
m3c_bps = m3c * 1e4
t3c_pass = bool(np.isfinite(m3c_bps) and m3c_bps >= BAR_3C_BPS
                and abs(t3c) >= BAR_3C_T)
print(f"  3c: arm D - arm A >= {BAR_3C_BPS:.0f} bps with |t| >= {BAR_3C_T}  ->  "
      f"{m3c_bps:+.2f} bps, t {t3c:+.3f} ({n3c} common days)   "
      f"{'MET' if t3c_pass else 'NOT MET'}")

if len(arms['D_ALT']):
    dDa = arms['D_ALT'].groupby('entry_date')['ret'].mean()
    j2 = pd.concat([dDa.rename('Da'), dA.rename('A')], axis=1).dropna()
    if len(j2) >= 3:
        ma, ta, na = nw_tstat((j2['Da'] - j2['A']).to_numpy(), NW_MAXLAGS_TS)
        print(f"      [diagnostic, no authority] D_ALT - A = {ma*1e4:+.2f} bps, "
              f"t {ta:+.3f}")

test3_pass = bool(t3a_pass and t3b_pass and t3c_pass)
print(f"\nTEST 3 VERDICT: {'PASS' if test3_pass else 'FAIL'}")

# --------------------------------------------------------------------------
# 8. Sensitivity grid (no verdict authority, prereg section 6)
# --------------------------------------------------------------------------
print('\n' + '=' * 88)
print('SENSITIVITY GRID - REPORTED ONLY, NO VERDICT AUTHORITY')
print('=' * 88)

grid = {'test1_by_residency_window': {}, 'test2_by_threshold': {},
        'test3_by_volume_multiplier': {}}

print('\nTest 1 interaction across residency-SCORE windows W (thr025 episodes,')
print('using the discrete residency_{W} fraction, NOT continuous_residency_')
print('depth - a different, diagnostic-only quantity from the amended 1a):')
for w in RES_WIN_GRID:
    col = f'residency_{w}'
    if col not in ep_primary.columns:
        continue
    _sd = float(ep_primary[col].std())
    if not (_sd > 1e-10):
        print(f"  W={w:>2}  SKIPPED - residency_{w} is constant on the "
              f"episode set (same structural cause as the original 1a "
              f"failure; see module docstring)")
        continue
    sig = ['compression_ratio', col, 'ribbon_width']
    fmw, _ = fama_macbeth(ep_primary, ycol, sig, res_col=col)
    if len(fmw) == 0:
        print(f"  W={w:>2}  SKIPPED - no usable Fama-MacBeth dates")
        continue
    m, t, n = nw_tstat(fmw['b_inter'], NW_MAXLAGS_FM)
    grid['test1_by_residency_window'][f'W{w}'] = {
        'interaction_mean_coef': m, 'interaction_nw_t': t, 'n_dates': n}
    star = '  <-- W=10 matches the primary EPISODE window (not the amended 1a term)' if w == RES_WIN_PRIMARY else ''
    print(f"  W={w:>2}  interaction coef {m:+.6f}  NW t {t:+7.3f}  "
          f"({n} dates){star}")

print('\nTest 2 contrasts across episode thresholds T:')
for tag in THRESH_GRID:
    col = f'episode_start_{tag}'
    sub = ep[ep[col] & ep['eligible_common_start']].copy()
    sub = sub.merge(d[['PERMNO', 'DlyCalDt', 'pos']], on=['PERMNO', 'DlyCalDt'],
                    how='left').dropna(subset=['pos'])
    if len(sub) < 100:
        continue
    sub['pos'] = sub['pos'].astype(np.int64)
    pth = build_paths(sub['pos'].to_numpy(), PRIMARY_HORIZON)
    up, _, _ = barrier_race(pth, sub['atr14'].to_numpy(float), 2.0,
                            PRIMARY_HORIZON)
    sub['up_first_2atr'] = up
    sub['dn_first_2atr'] = 1.0 - up
    sub['state'] = np.where(
        (sub['ema8'] > sub['ema21']) & (sub['ema8_slope'] > 0), 'bullish',
        np.where((sub['ema8'] < sub['ema21']) & (sub['ema8_slope'] < 0),
                 'bearish', 'neutral'))
    sub = sub.dropna(subset=['up_first_2atr'])
    ma, za, _, _ = contrast(sub, 'bullish', 'up_first_2atr')
    mb, zb, _, _ = contrast(sub, 'bearish', 'dn_first_2atr')
    grid['test2_by_threshold'][tag] = {
        'n_episodes': int(len(sub)),
        'bullish_margin_pp': ma * 100.0, 'bullish_z': za,
        'bearish_margin_pp': mb * 100.0, 'bearish_z': zb}
    star = '  <-- PRIMARY' if tag == THRESH_PRIMARY else ''
    print(f"  {tag:<7} n={len(sub):>7,}  2a {ma*100:+6.2f} pp (z {za:+6.3f})  "
          f"2b {mb*100:+6.2f} pp (z {zb:+6.3f}){star}")

print('\nTest 3 arms across volume multipliers V:')
for vm in VOL_MULT_GRID:
    ar = build_arms(vm)
    if len(ar['D']) == 0 or len(ar['A']) == 0:
        continue
    dd = ar['D'].groupby('entry_date')['ret'].mean()
    da = ar['A'].groupby('entry_date')['ret'].mean()
    mu, t, _ = nw_tstat(dd.to_numpy(), NW_MAXLAGS_TS)
    jj = pd.concat([dd.rename('D'), da.rename('A')], axis=1).dropna()
    mc, tc, _ = nw_tstat((jj['D'] - jj['A']).to_numpy(), NW_MAXLAGS_TS)
    grid['test3_by_volume_multiplier'][f'{vm}x'] = {
        'n_trades_D': int(len(ar['D'])), 'n_trades_A': int(len(ar['A'])),
        'D_mean_daily_ret': mu, 'D_nw_t': t,
        'D_minus_A_bps': mc * 1e4, 'D_minus_A_t': tc}
    star = '  <-- PRIMARY' if vm == VOL_MULT_PRIMARY else ''
    print(f"  {vm}x  D trades {len(ar['D']):>6,}  D t {t:+7.3f}  "
          f"D-A {mc*1e4:+8.2f} bps (t {tc:+6.3f}){star}")

# --------------------------------------------------------------------------
# 9. Verdict and miss diagnostics (prereg section 9)
# --------------------------------------------------------------------------
print('\n' + '=' * 88)
print('VERDICT (prereg section 9)')
print('=' * 88)


def near(val, bar, tval, tbar=NEARPASS_T):
    return (np.isfinite(val) and val >= bar * NEARPASS_MARGIN_FRAC
            and np.isfinite(tval) and abs(tval) >= tbar)


conds = {
    '1a': {'pass': t1a_pass, 'value': t1a_val,
           'bar': f'NW t <= -{BAR_1A_T} (AMENDED, negative expected)'},
    '1b': {'pass': t1b_pass, 'value': t1b_margin_pp,
           'bar': f'>= {BAR_1B_PP} pp, |z| >= {BAR_1B_Z}'},
    '2a': {'pass': t2a_pass, 'value': m2a_pp,
           'bar': f'>= {BAR_2_PP} pp, |z| >= {BAR_2_Z}'},
    '2b': {'pass': t2b_pooled_pass, 'value': m2b_pp,
           'bar': f'>= {BAR_2_PP} pp, |z| >= {BAR_2_Z}'},
    '2b-flat': {'pass': t2bflat_pass, 'value': m2bf_pp,
                'bar': f'>= {BAR_2BFLAT_PP} pp, |z| >= {BAR_2BFLAT_Z}'},
    '3a': {'pass': t3a_pass, 'value': t3a_t, 'bar': f'NW t >= {BAR_3A_T}'},
    '3b': {'pass': t3b_pass, 'value': net_d * 100 if np.isfinite(net_d) else np.nan,
           'bar': '> 0 after 30 bps'},
    '3c': {'pass': t3c_pass, 'value': m3c_bps,
           'bar': f'>= {BAR_3C_BPS} bps, |t| >= {BAR_3C_T}'},
}
print(f"  {'cond':<9} {'value':>12}  {'bar':<28} result")
for k, v in conds.items():
    val = v['value']
    vs = f"{val:+.3f}" if np.isfinite(val) else 'nan'
    print(f"  {k:<9} {vs:>12}  {v['bar']:<28} "
          f"{'MET' if v['pass'] else 'NOT MET'}")

n_tests_pass = sum([test1_pass, test2_pass, test3_pass])
near_flags = {
    '1': near(t1a_val, BAR_1A_T, t1a_val, NEARPASS_T) and
         near(t1b_margin_pp, BAR_1B_PP, t1b_z),
    '2': (near(m2a_pp, BAR_2_PP, z2a) and near(m2b_pp, BAR_2_PP, z2b) and
          near(m2bf_pp, BAR_2BFLAT_PP, z2bf, BAR_2BFLAT_Z)),
    '3': (np.isfinite(t3a_t) and abs(t3a_t) >= NEARPASS_T and t3a_mu > 0 and
          t3b_pass and near(m3c_bps, BAR_3C_BPS, t3c)),
}
failed = [k for k, v in {'1': test1_pass, '2': test2_pass,
                         '3': test3_pass}.items() if not v]
if n_tests_pass == 3:
    verdict = 'PASS'
elif n_tests_pass == 2 and len(failed) == 1 and near_flags[failed[0]]:
    verdict = 'NEAR-PASS'
else:
    verdict = 'FAIL'
print(f"\nTests passing: {n_tests_pass}/3   ->   E1 VERDICT: {verdict}")

# Miss diagnostics (prereg section 9)
print('\n' + '-' * 88)
print('MISS DIAGNOSTICS (required by prereg section 9 whenever a test misses)')
print('-' * 88)
miss_diag = {}
if not test1_pass or not test2_pass:
    print('\nBy gsector (2ATR-move rate, and bearish 2b-style margin):')
    rows = []
    for sec, gg in t2.groupby('gsector'):
        if len(gg) < 200:
            continue
        mb, zb, _, _ = contrast(gg, 'bearish', 'dn_first_2atr')
        ma, za, _, _ = contrast(gg, 'bullish', 'up_first_2atr')
        rows.append((int(sec), len(gg), ma * 100, za, mb * 100, zb))
    rows.sort(key=lambda r: -abs(r[4]) if np.isfinite(r[4]) else 0)
    print(f"  {'sector':>7} {'n':>7} {'2a pp':>8} {'z':>7} {'2b pp':>8} {'z':>7}")
    for r in rows:
        print(f"  {r[0]:>7} {r[1]:>7,} {r[2]:>+8.2f} {r[3]:>+7.2f} "
              f"{r[4]:>+8.2f} {r[5]:>+7.2f}")
    miss_diag['by_gsector'] = [
        {'gsector': r[0], 'n': r[1], 'bullish_pp': r[2], 'bullish_z': r[3],
         'bearish_pp': r[4], 'bearish_z': r[5]} for r in rows]

    print('\nBy size decile:')
    rows = []
    for dec, gg in t2.groupby('decile'):
        if len(gg) < 200:
            continue
        mb, zb, _, _ = contrast(gg, 'bearish', 'dn_first_2atr')
        ma, za, _, _ = contrast(gg, 'bullish', 'up_first_2atr')
        rows.append((int(dec), len(gg), ma * 100, za, mb * 100, zb))
    for r in sorted(rows):
        print(f"  decile {r[0]:>2}  n={r[1]:>7,}  2a {r[2]:+6.2f} pp "
              f"(z {r[3]:+6.2f})   2b {r[4]:+6.2f} pp (z {r[5]:+6.2f})")
    miss_diag['by_decile'] = [
        {'decile': r[0], 'n': r[1], 'bullish_pp': r[2], 'bullish_z': r[3],
         'bearish_pp': r[4], 'bearish_z': r[5]} for r in rows]

    print('\nBy market-direction regime (already computed above):')
    for rg, v in regime_detail.items():
        print(f"  {rg:<9} n={v['n_episodes']:>7,}  2a {v['bullish_margin_pp']:+6.2f} pp "
              f"(z {v['bullish_z']:+6.2f})   2b {v['bearish_margin_pp']:+6.2f} pp "
              f"(z {v['bearish_z']:+6.2f})")
    miss_diag['by_regime'] = regime_detail

    print('\nBy year:')
    rows = []
    for yr, gg in t2.groupby(t2['DlyCalDt'].dt.year):
        mb, zb, _, _ = contrast(gg, 'bearish', 'dn_first_2atr')
        ma, za, _, _ = contrast(gg, 'bullish', 'up_first_2atr')
        rows.append((int(yr), len(gg), ma * 100, za, mb * 100, zb))
    for r in rows:
        print(f"  {r[0]}  n={r[1]:>7,}  2a {r[2]:+6.2f} pp (z {r[3]:+6.2f})   "
              f"2b {r[4]:+6.2f} pp (z {r[5]:+6.2f})")
    miss_diag['by_year'] = [
        {'year': r[0], 'n': r[1], 'bullish_pp': r[2], 'bullish_z': r[3],
         'bearish_pp': r[4], 'bearish_z': r[5]} for r in rows]

# --------------------------------------------------------------------------
# 10. Episode composition
# --------------------------------------------------------------------------
print('\n' + '-' * 88)
print('EPISODE COMPOSITION (prereg section 12)')
print('-' * 88)
comp_year = ep_primary.groupby(ep_primary['DlyCalDt'].dt.year).size()
print('By year:')
for y, n in comp_year.items():
    print(f"  {int(y)}: {n:>7,}")
print('By gsector:')
for s, n in ep_primary['gsector'].value_counts().sort_index().items():
    print(f"  {int(s) if pd.notna(s) else 'NA':>4}: {n:>7,}")
print('By size decile:')
for s, n in ep_primary['decile'].value_counts().sort_index().items():
    print(f"  decile {int(s)}: {n:>7,}")
print('\nDrops:')
for k, v in drops.items():
    print(f"  {k:<45} {v:>9,}")
print('Barrier tie / no-touch counts:')
for k, v in tie_counts.items():
    print(f"  {k:<45} {v:>9,}")

# --------------------------------------------------------------------------
# 11. Save JSON
# --------------------------------------------------------------------------
result = {
    'phase': 'E1 - EMA-ribbon residency',
    'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'script': 'src/46_E1_tests.py',
    'pre_registration': 'results/prereg_E1.md (LOCKED 2026-07-30)',
    'window': f'DEV {DEV_START.date()} to {DEV_END.date()} - no holdout path',
    'holdout_disclosure': (
        '2022-2025 already spent as holdout for R1, R2, V1, V2, K1. E1 makes '
        'no holdout claim. A future E1 holdout would be a sixth look at the '
        'same calendar period, not a fresh out-of-sample test.'),
    'implementation_decisions': {
        'D1_entry_convention': 'entry at open(t+1); h-session return = close(t+h)/open(t+1)-1',
        'D2_barrier_ties': 'same-session double touch excluded from that race and counted',
        'D3_fm_standardization': 'signal terms z-scored cross-sectionally within date',
        'D4_date_constant_controls': 'market-vol regime absorbed by FM per-date intercept',
        'D5_min_cross_section': MIN_XSEC,
        'D6_test1b_estimator': 'per-episode y - matched cell rate, date-clustered SE',
        'D7_breakout_reference': 'D uses episode running max; A/B/C use trailing 10-session high; D_ALT diagnostic uses trailing 10',
        'D8_trigger_window': TRIGGER_WINDOW,
        'D9_compression_criterion': 'most-compressed cross-sectional decile (V1 convention)',
        'D10_winsorization': 'none',
        'D11_forward_vol': 'DlyRet std over window, annualized sqrt(252)',
    },
    'sample': {
        'primary_episodes_before_common_start': n_before,
        'excluded_by_common_start_rule': n_excl,
        'primary_episodes_used': int(len(ep_primary)),
        'unique_permnos': int(ep_primary['PERMNO'].nunique()),
        'date_range': [str(ep_primary['DlyCalDt'].min().date()),
                       str(ep_primary['DlyCalDt'].max().date())],
        'drops': drops,
        'barrier_counts': tie_counts,
    },
    'test1': {
        'fama_macbeth_outcome': ycol,
        'amendment': 'AMENDED 2026-07-30 (prereg section 13 item 11): 1a '
                     'interaction is compression_ratio x '
                     'continuous_residency_depth, expected sign negative; '
                     'residency_10 dropped from the model (constant on the '
                     'primary episode set by construction).',
        'coefficients': t1_res,
        '1a': {'nw_t': t1a_val, 'bar': BAR_1A_T, 'expected_sign': 'negative',
               'met': t1a_pass},
        '1b': {'episode_rate_pct': ep_rate * 100, 'control_rate_pct': ct_rate * 100,
               'margin_pp': t1b_margin_pp, 'z': t1b_z, 'n': t1b_n,
               'bar_pp': BAR_1B_PP, 'bar_z': BAR_1B_Z, 'met': t1b_pass},
        'verdict': 'PASS' if test1_pass else 'FAIL',
    },
    'test2': {
        'state_counts': ep_primary['state'].value_counts().to_dict(),
        'state_barrier_rates': state_rates,
        '2a': {'margin_pp': m2a_pp, 'z': z2a, 'bar_pp': BAR_2_PP,
               'bar_z': BAR_2_Z, 'met': t2a_pass},
        '2b_pooled': {'margin_pp': m2b_pp, 'z': z2b, 'bar_pp': BAR_2_PP,
                      'bar_z': BAR_2_Z, 'met': t2b_pooled_pass},
        '2b_flat_load_bearing': {'margin_pp': m2bf_pp, 'z': z2bf,
                                 'bar_pp': BAR_2BFLAT_PP, 'bar_z': BAR_2BFLAT_Z,
                                 'met': t2bflat_pass},
        '2b_combined': t2b_pass,
        'regime_detail': regime_detail,
        'verdict': 'PASS' if test2_pass else 'FAIL',
    },
    'test3': {
        'arms': arm_stats,
        '3a': {'mean_daily_ret': t3a_mu, 'nw_t': t3a_t, 'bar': BAR_3A_T,
               'met': t3a_pass},
        '3b': {'mean_trade_ret': d_trade_mean, 'cost_bps': COST_ROUNDTRIP_BPS,
               'net': net_d, 'met': t3b_pass,
               'note': 'cost is a PLACEHOLDER per prereg section 7'},
        '3c': {'margin_bps': m3c_bps, 't': t3c, 'n_common_days': n3c,
               'bar_bps': BAR_3C_BPS, 'bar_t': BAR_3C_T, 'met': t3c_pass},
        'verdict': 'PASS' if test3_pass else 'FAIL',
    },
    'sensitivity_grid_no_authority': grid,
    'miss_diagnostics': miss_diag,
    'verdict': verdict,
    'tests_passing': n_tests_pass,
}
out_json.write_text(json.dumps(result, indent=2, default=str))
print(f"\n[OK] Saved {out_json}")

# --------------------------------------------------------------------------
# 12. Append-only, marker-guarded gate_log entry
# --------------------------------------------------------------------------
MARKER = '## E1 EMA-ribbon residency - Tests 1-3 (DEV only)'
log_text = log_path.read_text(encoding='utf-8', errors='replace')
if MARKER in log_text:
    print('[SKIP] gate_log.md already carries the E1 block; not duplicating.')
else:
    bak = log_path.with_suffix('.md.bak_before_e1_block')
    shutil.copy2(log_path, bak)
    print(f"[safety] gate_log.md backed up to {bak.name}")
    L = [
        f"\n---\n\n{MARKER}",
        "",
        f"Run {result['generated']} - script src/46_E1_tests.py - "
        f"pre-registration results/prereg_E1.md (LOCKED 2026-07-30). "
        f"DEV {DEV_START.date()} to {DEV_END.date()}, no holdout code path. "
        f"Machine-readable copy: results/46_E1_tests.json.",
        "",
        "```",
        f"primary episodes (thr025, W=10):        {n_before:,}",
        f"excluded by common-start rule:          {n_excl:,}",
        f"episodes used:                          {len(ep_primary):,}",
        f"unique PERMNOs:                         {ep_primary['PERMNO'].nunique():,}",
        "",
        "TEST 1 - MAGNITUDE",
        f"  interaction mean coef:                {t1_res['interaction']['mean_coef']:+.6f}",
        f"  1a  interaction NW t:                 {t1a_val:+.3f}   "
        f"(bar >= {BAR_1A_T})  {'MET' if t1a_pass else 'NOT MET'}",
        f"  1b  episode rate:                     {ep_rate*100:.2f}%",
        f"  1b  matched control rate:             {ct_rate*100:.2f}%",
        f"  1b  margin:                           {t1b_margin_pp:+.2f} pp   "
        f"z {t1b_z:+.3f}   (bar >= {BAR_1B_PP} pp, |z| >= {BAR_1B_Z})  "
        f"{'MET' if t1b_pass else 'NOT MET'}",
        f"  TEST 1:                               {'PASS' if test1_pass else 'FAIL'}",
        "",
        "TEST 2 - DIRECTION",
        f"  2a  margin:                           {m2a_pp:+.2f} pp   z {z2a:+.3f}   "
        f"(bar >= {BAR_2_PP} pp, |z| >= {BAR_2_Z})  {'MET' if t2a_pass else 'NOT MET'}",
        f"  2b  pooled margin:                    {m2b_pp:+.2f} pp   z {z2b:+.3f}   "
        f"(bar >= {BAR_2_PP} pp, |z| >= {BAR_2_Z})  {'MET' if t2b_pooled_pass else 'NOT MET'}",
        f"  2b-flat (load-bearing) margin:        {m2bf_pp:+.2f} pp   z {z2bf:+.3f}   "
        f"(bar >= {BAR_2BFLAT_PP} pp, |z| >= {BAR_2BFLAT_Z})  "
        f"{'MET' if t2bflat_pass else 'NOT MET'}",
        f"  TEST 2:                               {'PASS' if test2_pass else 'FAIL'}",
        "",
        "TEST 3 - BREAKOUT TRIGGER",
    ]
    for a in ['A', 'B', 'C', 'D']:
        s = arm_stats.get(a)
        if s:
            L.append(f"  arm {a}: trades {s['n_trades']:,}  mean daily "
                     f"{s['mean_daily_ret']*100:+.4f}%  NW t {s['nw_t']:+.3f}")
    L += [
        f"  3a  arm D NW t:                       {t3a_t:+.3f}   "
        f"(bar >= {BAR_3A_T})  {'MET' if t3a_pass else 'NOT MET'}",
        f"  3b  arm D net of {COST_ROUNDTRIP_BPS:.0f} bps:            "
        f"{net_d*100:+.3f}%   {'MET' if t3b_pass else 'NOT MET'}",
        f"  3c  arm D - arm A:                    {m3c_bps:+.2f} bps   t {t3c:+.3f}   "
        f"(bar >= {BAR_3C_BPS:.0f} bps, |t| >= {BAR_3C_T})  "
        f"{'MET' if t3c_pass else 'NOT MET'}",
        f"  TEST 3:                               {'PASS' if test3_pass else 'FAIL'}",
        "",
        f"TESTS PASSING:                          {n_tests_pass}/3",
        f"E1 VERDICT:                             {verdict}",
        "```",
        "",
    ]
    with log_path.open('a', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print(f"[OK] Appended E1 block to {log_path} (prior entries untouched)")

print('\n' + '=' * 88)
print(f'E1 TESTS COMPLETE - verdict {verdict}')
print('=' * 88)
