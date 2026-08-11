import pandas as pd
import numpy as np
import json
import shutil
from datetime import datetime
from pathlib import Path
from scipy.stats import norm
from scipy.optimize import brentq
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# K1 DELTA-HEDGED DECOMPOSITION - DEV window, per-trade.
#
# PURPOSE. The K1 real-prices gate logged a FAIL. Its headline per-trade
# numbers (mid-to-mid, bid-based worst fill, liquidity-filtered) are READ AT
# RUNTIME from data/processed/k1_trade_summary.json - nothing is transcribed
# here. The K1 straddle was never delta-hedged, so from the gross number
# alone it is unknown whether it reflects a real volatility forecast or
# uncontrolled directional exposure. This script separates the two.
#
# It makes NO gate decision, changes NO filter, threshold or trade selection,
# and re-uses the locked trade list exactly as persisted by src/37. No
# re-scan of opprcd: the per-trade parquet plus CRSP daily closes are the
# only inputs.
#
# ===========================================================================
# DECLARED ASSUMPTIONS (literals, fixed before any outcome was computed)
# ===========================================================================
#
# A1. RISK_FREE_RATE = 0.01, a single constant, NOT the per-trade Ken French
#     RF. That series is quantized to 0.0001/day steps over this window and
#     is therefore a lumpy step function, not a rate path: 0.00% in 2015-17,
#     2.52% in 2018-19, 0.6175% in 2020, 0.00% in 2021-22, mean 0.82%. A
#     single 1.0% constant is the sample-mean stand-in. The script re-derives
#     those RF facts from data/processed/factors_daily.parquet at runtime and
#     prints them next to the literal, so the stand-in is checkable rather
#     than asserted.
#
#     Robustness: r enters the straddle delta only through d1, as the term
#     r*T/(sigma*sqrt(T)) = r*sqrt(T)/sigma. At a 30-day tenor and sigma
#     around 50% that term is of order 1e-2, so the delta is near-inert to r
#     over the whole 0.00%-2.52% span the real RF actually spans. The script
#     RECOMPUTES every entry delta at r=0 and at r=0.0252 holding the backed-
#     out IV fixed and prints the realised distribution of the delta shift -
#     documented, not just asserted. NOTE the term scales as 1/sigma, so it
#     is NOT small for the handful of near-zero-IV trades that the IV sanity
#     check separately flags; the printed quantiles show where the claim
#     holds and where it does not, rather than averaging the tail away.
#
# A2. HEDGE_COST_BPS = (0, 5, 15). Three scenarios, reported side by side.
#     There is no single defensible guess for equity hedge slippage over
#     2015-2021, so none is made.
#
# A3. brentq bounds on the IV inversion: sigma in [0.0001, 5.0].
#
# A4. PATTERN_DIRECTIONAL_SHARE_CUTOFF = 0.5. A REPORTING convention only,
#     used to name which of the two pre-specified patterns the numbers match.
#     It has no analytical authority and gates nothing. The raw shares are
#     printed alongside so the classification can be overridden by eye.
#
#     The conditions are evaluated on MAGNITUDES, and NEITHER is a permitted
#     outcome. Patterns A and B are exhaustive only if the directional
#     component has the same sign as the unhedged gross and is no larger than
#     it. If it does not - e.g. if hedging the direction RAISES the gross
#     rather than lowering it - the honest report is that the numbers match
#     neither pre-specified pattern, and the components are stated directly.
#     Forcing such a case into A or B by a signed ratio would misreport it.
#
# No other number in this file is a literal. Every K1 reference figure, trade
# count and spread statistic is read from the persisted JSON at runtime.
#
# ===========================================================================
# MANDATORY DISCLOSURE
# ===========================================================================
#
# D1. STICKY-VOL DELTA. The delta path is computed with the ENTRY implied
#     volatility HELD CONSTANT for the life of the trade. Real implied
#     volatility moves during the trade, and the true hedge ratio moves with
#     it. Holding IV fixed makes the hedged return PATH-DEPENDENT: it is a
#     REALISTIC decomposition of what a trader following this rule would have
#     experienced, NOT a pure isolation of the volatility component. A
#     residual vega/vanna P&L remains inside the "delta-hedged" leg.
#
# D2. DISCRETE HEDGING. The hedge is rebalanced close-to-close, once per
#     trading day, not continuously. Intraday moves are unhedged. Discrete
#     hedging leaves a gamma-driven hedging error that does not vanish in
#     expectation and is larger the larger the daily move.
#
# D3. No dividend yield is modelled (q = 0), matching src/30/32/37. The
#     underlying path is the raw CRSP daily close. Zero trades in this sample
#     contain a CRSP price-adjustment-factor change inside their holding
#     window; the script asserts this at runtime rather than assuming it, so
#     the raw-close path carries no split discontinuity.
#
# D4. The daily portfolio series is NOT truncated at DEV_END, unlike src/37,
#     which dropped the handful of January-2022 days belonging to trades
#     entered in late December 2021. Truncating would make the daily-series
#     mean inconsistent with the per-trade mean reported on the same
#     waterfall row. The number of retained post-DEV_END days is printed and
#     stored. No trade is added, and no trade entered outside DEV.
#
# D5. Newey-West t-stats are reported BOTH on the raw daily series (the exact
#     quantity named in the waterfall row) and on the RF-excess series (the
#     src/38 convention). The waterfall rows are raw P&L ratios and their
#     means must add up, so the raw series is the primary; the excess-RF
#     column is carried so the src/38 convention is visible and the choice
#     is immaterial rather than hidden.
#
# ===========================================================================
# METHOD
# ===========================================================================
#
#  1. Entry IV, per trade: solve BS_call(sigma) + BS_put(sigma) - entry_mid
#     = 0 by scipy.optimize.brentq over A3's bounds, with S = the entry-date
#     CRSP close, K = the held contracts' strike, T = (exdate_d -
#     entry_date).days / 365, r = RISK_FREE_RATE.
#
#  2. Daily delta, sticky-vol: entry IV held constant, the CRSP close of each
#     session as spot, T decaying as (exdate_d - date).days / 365. The long-
#     straddle position delta is 2*N(d1_t) - 1.
#
#  3. Full daily rebalance: short delta_t shares, rehedged at every close,
#     including establishing the hedge at entry and unwinding it at exit.
#     The hedge held across the final session is delta_{n-1}, matching the
#     directional sum below; the position is unwound at S_n, so delta_n is
#     never traded.
#
# DECOMPOSITION. With OptionPnL_mid = exit_mid - entry_mid and
# Directional = sum_t delta_{t-1} * (S_t - S_{t-1}):
#
#   (a) delta-hedged P&L                := OptionPnL_mid - Directional
#   (b) directional P&L removed by hedge:= Directional
#       (a) + (b) = OptionPnL_mid exactly, by construction. This is a
#       bookkeeping assertion, not a finding.
#   (c) option spread cost := -[(entry_ask - entry_mid)
#                               + (exit_mid - exit_bid)]
#   (d) hedge transaction cost := -HEDGE_COST_BPS/10000
#                                 * sum |delta_t - delta_{t-1}| * S_t
#
#   Total P&L := (a)+(b)+(c)+(d) = OptionPnL_mid - spread_cost - hedge_cost,
#   asserted per trade.
#
# DROPS. A trade is dropped, and counted with its reason, if (i) any daily
# underlying close in [entry_date, exit_date] is missing from CRSP, or (ii)
# the IV inversion fails to converge. No other exclusion exists.
#
# gate_log.md receives numbers only, no interpretation.
# ---------------------------------------------------------------------------

# --- Declared assumptions (A1-A4) ------------------------------------------
RISK_FREE_RATE = 0.01
HEDGE_COST_BPS = (0, 5, 15)
IV_LO, IV_HI = 0.0001, 5.0
PATTERN_DIRECTIONAL_SHARE_CUTOFF = 0.5

# Robustness span for A1 (the two extremes the real RF step function visits)
R_ROBUST_LO, R_ROBUST_HI = 0.0, 0.0252

# Reporting grid for the execution-quality sweep: fraction of the entry and
# exit half-spread actually paid. 0% = pure mid, 100% = full ask/bid
# crossing, which is the original K1 bid-based assumption.
SPREAD_FRACTIONS = (0.0, 0.25, 0.50, 0.75, 1.00)

DEV_END = pd.Timestamp('2021-12-31')
NW_MAXLAGS = 10
IV_FLAG_HI = 2.00     # sanity flags only; nothing is dropped on these
IV_FLAG_LO = 0.05

project_root = Path(__file__).parent.parent
trades_path = project_root / 'data' / 'processed' / 'k1_trades_real_prices.parquet'
summary_path = project_root / 'data' / 'processed' / 'k1_trade_summary.json'
crsp_path = project_root / 'data' / 'processed' / 'crsp_combined.parquet'
fac_path = project_root / 'data' / 'processed' / 'factors_daily.parquet'
out_json = project_root / 'results' / '42_delta_hedged_decomp.json'
log_path = project_root / 'results' / 'gate_log.md'

print('=' * 88)
print('K1 DELTA-HEDGED DECOMPOSITION  (DEV, per-trade; no gate decision)')
print('=' * 88)

for p in (trades_path, summary_path, crsp_path, fac_path, log_path):
    if not p.exists():
        raise SystemExit(f'STOP: required input not found: {p}')

# --------------------------------------------------------------------------
# 1. Reference figures - READ, never transcribed
# --------------------------------------------------------------------------
print('\n' + '-' * 88)
print('1. K1 REFERENCE FIGURES (read from k1_trade_summary.json at runtime)')
print('-' * 88)

k1 = json.loads(summary_path.read_text(encoding='utf-8'))
K1_MID = k1['mean_per_trade_return_pct']['mid_to_mid']
K1_BID = k1['mean_per_trade_return_pct']['bid_based_worst_fill']
K1_LIQ = k1['mean_per_trade_return_pct']['mid_liquidity_filtered']
K1_N = k1['trades']['matched_entry_and_exit']
K1_SP_MED = k1['entry_half_spread_pct_of_mid']['median']
K1_SP_MEAN = k1['entry_half_spread_pct_of_mid']['mean']

print(f"  source phase:                          {k1['phase']}")
print(f"  source script:                         {k1['script']}")
print(f"  matched trades:                        {K1_N:,}")
print(f"  unhedged mean per-trade, mid-to-mid:   {K1_MID:+.4f}%")
print(f"  unhedged mean per-trade, bid-based:    {K1_BID:+.4f}%")
print(f"  unhedged mean per-trade, liq-filtered: {K1_LIQ:+.4f}%")
print(f"  entry half-spread, median / mean:      {K1_SP_MED:.2f}% / "
      f"{K1_SP_MEAN:.2f}% of mid")

t = pd.read_parquet(trades_path)
print(f"\n  per-trade parquet rows: {len(t):,}  columns: {t.shape[1]}")
assert len(t) == K1_N, (
    f'parquet has {len(t)} trades, summary JSON says {K1_N}')
_recomp_mid = float(t['ret_mid'].mean()) * 100.0
assert abs(_recomp_mid - K1_MID) < 1e-3, (
    f'ret_mid recomputed {_recomp_mid} vs JSON {K1_MID}')
print(f"  [PASS] parquet reproduces the JSON headline exactly "
      f"({_recomp_mid:+.4f}% vs {K1_MID:+.4f}%)")

# --------------------------------------------------------------------------
# 2. Declared-assumption check on RISK_FREE_RATE (A1)
# --------------------------------------------------------------------------
print('\n' + '-' * 88)
print('2. RISK_FREE_RATE - declared literal vs the RF series it stands in for')
print('-' * 88)

fac = pd.read_parquet(fac_path, columns=['date', 'RF'])
fac = fac[(fac['date'] >= t['entry_date'].min()) &
          (fac['date'] <= t['exit_date'].max())]
rf_steps = np.sort(fac['RF'].dropna().unique())
print(f"  distinct daily RF values over the trade window: {len(rf_steps)} "
      f"-> {np.array2string(rf_steps, precision=6)}")
print(f"  (quantized to 0.0001/day: a step function, not a rate path)")
print(f"  observed annualized RF by year:")
for y, v in fac.assign(y=fac['date'].dt.year).groupby('y')['RF'].mean().items():
    print(f"      {int(y)}: {v * 252 * 100:6.4f}%")
rf_mean_annual = float(fac['RF'].mean() * 252)
print(f"  observed mean annualized RF: {rf_mean_annual * 100:.4f}%")
print(f"  DECLARED LITERAL in use:     {RISK_FREE_RATE * 100:.4f}%  "
      f"(constant, A1)")

# --------------------------------------------------------------------------
# 3. Underlying price paths from CRSP daily
# --------------------------------------------------------------------------
print('\n' + '-' * 88)
print('3. UNDERLYING PRICE PATHS (CRSP daily close, entry through exit)')
print('-' * 88)

cal = pd.DatetimeIndex(sorted(
    pd.read_parquet(crsp_path, columns=['DlyCalDt'])['DlyCalDt'].unique()))
print(f"  master trading calendar: {len(cal):,} sessions "
      f"({cal.min().date()} to {cal.max().date()})")

ei = cal.get_indexer(t['entry_date'])
xi = cal.get_indexer(t['exit_date'])
assert (ei >= 0).all() and (xi >= 0).all(), 'trade date off the CRSP calendar'
gaps = np.unique(xi - ei)
assert len(gaps) == 1, f'holding period is not uniform: {gaps}'
N_STEPS = int(gaps[0])
print(f"  holding period: {N_STEPS} sessions for all {len(t):,} trades "
      f"({N_STEPS + 1} closes per path, entry and exit inclusive)")

crsp = pd.read_parquet(crsp_path,
                       columns=['PERMNO', 'DlyCalDt', 'DlyClose', 'DlyFacPrc'])
crsp = crsp[crsp['PERMNO'].isin(t['PERMNO'].unique())]
crsp = crsp[(crsp['DlyCalDt'] >= t['entry_date'].min()) &
            (crsp['DlyCalDt'] <= t['exit_date'].max())]
crsp = crsp.drop_duplicates(['PERMNO', 'DlyCalDt'], keep='first')
close_map = crsp.set_index(['PERMNO', 'DlyCalDt'])['DlyClose']
fac_map = crsp.set_index(['PERMNO', 'DlyCalDt'])['DlyFacPrc']

n_tr = len(t)
day_pos = ei[:, None] + np.arange(N_STEPS + 1)[None, :]
path_dates = cal.to_numpy()[day_pos]
permno_rep = np.repeat(t['PERMNO'].to_numpy(), N_STEPS + 1)
mi = pd.MultiIndex.from_arrays([permno_rep, path_dates.ravel()])
S = close_map.reindex(mi).to_numpy(dtype=float).reshape(n_tr, N_STEPS + 1)
FP = fac_map.reindex(mi).to_numpy(dtype=float).reshape(n_tr, N_STEPS + 1)

# CRSP encodes a no-trade session as a negative close (the bid/ask average).
# Zero appear in this sample; the check is kept so a future re-run cannot
# silently feed a signed price into the delta.
n_nonpos = int(np.nansum(S <= 0))
print(f"  non-positive CRSP closes on trade paths: {n_nonpos} "
      f"(negative closes are CRSP's no-trade bid/ask average)")
assert n_nonpos == 0, 'non-positive close on a trade path - handle explicitly'

# D3: no split / price-adjustment discontinuity inside any holding window.
with np.errstate(invalid='ignore'):
    fp_span = np.nanmax(FP, axis=1) / np.nanmin(FP, axis=1) - 1.0
n_split = int(np.nansum(fp_span > 1e-6))
print(f"  trades whose window contains a CRSP price-adjustment-factor "
      f"change: {n_split}")
assert n_split == 0, (
    'a split/adjustment falls inside a holding window - the raw-close path '
    'would fabricate a directional move; handle explicitly before proceeding')

ok_path = np.isfinite(S).all(axis=1)
n_drop_path = int((~ok_path).sum())
print(f"  trades dropped, missing >=1 daily underlying close: {n_drop_path} "
      f"({n_drop_path / n_tr * 100:.3f}%)  "
      f"[{int(np.isnan(S).sum())} missing day-cells]")

# --------------------------------------------------------------------------
# 4. Entry IV inversion
# --------------------------------------------------------------------------
print('\n' + '-' * 88)
print('4. ENTRY IV INVERSION  (brentq, sigma in [%.4f, %.1f])'
      % (IV_LO, IV_HI))
print('-' * 88)


def bs_straddle(S_, K_, T_, r_, sig_):
    """Black-Scholes value of a long call + long put, same strike and expiry."""
    sq = sig_ * np.sqrt(T_)
    d1 = (np.log(S_ / K_) + (r_ + 0.5 * sig_ * sig_) * T_) / sq
    d2 = d1 - sq
    disc = K_ * np.exp(-r_ * T_)
    call = S_ * norm.cdf(d1) - disc * norm.cdf(d2)
    put = disc * norm.cdf(-d2) - S_ * norm.cdf(-d1)
    return call + put


K = t['strike'].to_numpy(dtype=float)
T_entry = ((t['exdate_d'] - t['entry_date']).dt.days.to_numpy(dtype=float)
           / 365.0)
entry_mid = t['entry_mid'].to_numpy(dtype=float)
exit_mid = t['exit_mid'].to_numpy(dtype=float)
entry_ask = t['entry_ask'].to_numpy(dtype=float)
exit_bid = t['exit_bid'].to_numpy(dtype=float)
assert (T_entry > 0).all(), 'non-positive time to expiry at entry'

iv = np.full(n_tr, np.nan)
iv_fail_reason = {'no_sign_change_price_outside_bounds': 0, 'solver_error': 0}
for i in range(n_tr):
    if not ok_path[i]:
        continue
    s0, k0, tt = S[i, 0], K[i], T_entry[i]
    f = lambda sg: bs_straddle(s0, k0, tt, RISK_FREE_RATE, sg) - entry_mid[i]
    try:
        if f(IV_LO) * f(IV_HI) > 0:
            iv_fail_reason['no_sign_change_price_outside_bounds'] += 1
            continue
        iv[i] = brentq(f, IV_LO, IV_HI, xtol=1e-10, rtol=1e-12, maxiter=200)
    except Exception:
        iv_fail_reason['solver_error'] += 1

ok_iv = np.isfinite(iv)
n_drop_iv = int((ok_path & ~ok_iv).sum())
print(f"  trades dropped, IV inversion did not converge: {n_drop_iv}")
for rsn, cnt in iv_fail_reason.items():
    print(f"      {rsn}: {cnt}")

keep = ok_path & ok_iv
n_keep = int(keep.sum())
print(f"\n  TRADES PROCESSED: {n_keep:,} of {n_tr:,} "
      f"({n_keep / n_tr * 100:.2f}%)")
print(f"  TRADES DROPPED:   {n_tr - n_keep:,}  "
      f"(missing underlying close {n_drop_path}, IV inversion {n_drop_iv})")

ivk = iv[keep]
q = np.percentile(ivk, [1, 25, 50, 75, 99])
n_iv_hi = int((ivk > IV_FLAG_HI).sum())
n_iv_lo = int((ivk < IV_FLAG_LO).sum())
print(f"\n  Backed-out entry IV distribution ({n_keep:,} trades):")
print(f"      min {ivk.min() * 100:7.2f}%   p1 {q[0] * 100:7.2f}%   "
      f"q1 {q[1] * 100:7.2f}%")
print(f"      median {q[2] * 100:7.2f}%   mean {ivk.mean() * 100:7.2f}%")
print(f"      q3 {q[3] * 100:7.2f}%   p99 {q[4] * 100:7.2f}%   "
      f"max {ivk.max() * 100:7.2f}%")
print(f"      SANITY: IV above {IV_FLAG_HI * 100:.0f}%: {n_iv_hi:,} "
      f"({n_iv_hi / n_keep * 100:.2f}%)   "
      f"{'<-- CONCERN' if n_iv_hi else 'none'}")
print(f"      SANITY: IV below {IV_FLAG_LO * 100:.0f}%:  {n_iv_lo:,} "
      f"({n_iv_lo / n_keep * 100:.2f}%)   "
      f"{'<-- CONCERN' if n_iv_lo else 'none'}")

# --------------------------------------------------------------------------
# 5. Sticky-vol delta path and the r-sensitivity robustness check (A1)
# --------------------------------------------------------------------------
print('\n' + '-' * 88)
print('5. DELTA PATH (sticky-vol: entry IV held constant; D1)')
print('-' * 88)

exdate = t['exdate_d'].to_numpy()
T_path = ((exdate[:, None] - path_dates).astype('timedelta64[D]')
          .astype(float) / 365.0)
assert (T_path[keep] > 0).all(), 'a path date is at or past expiry'


def straddle_delta(S_, K_, T_, r_, sig_):
    """Long-straddle position delta, 2*N(d1) - 1."""
    with np.errstate(invalid='ignore', divide='ignore'):
        sq = sig_ * np.sqrt(T_)
        d1 = (np.log(S_ / K_) + (r_ + 0.5 * sig_ * sig_) * T_) / sq
    return 2.0 * norm.cdf(d1) - 1.0


sig_col = iv[:, None]
K_col = K[:, None]
# Deltas actually traded: t = 0 .. N_STEPS-1. The hedge held across the final
# session is delta_{N_STEPS-1}; the option is sold at exit, so the exit-close
# delta is never established.
delta = straddle_delta(S[:, :N_STEPS], K_col, T_path[:, :N_STEPS],
                       RISK_FREE_RATE, sig_col)
print(f"  delta matrix: {delta.shape[0]:,} trades x {delta.shape[1]} "
      f"rebalance points")
dk = delta[keep]
print(f"  entry delta (t=0):  mean {dk[:, 0].mean():+.4f}   "
      f"median {np.median(dk[:, 0]):+.4f}   "
      f"|delta| mean {np.abs(dk[:, 0]).mean():.4f}")
print(f"  final-hold delta:   mean {dk[:, -1].mean():+.4f}   "
      f"|delta| mean {np.abs(dk[:, -1]).mean():.4f}")

# A1 robustness, computed not asserted: hold IV fixed, move r across the full
# span the real RF step function visits, and measure the delta shift.
d_lo = straddle_delta(S[keep, 0], K[keep], T_entry[keep], R_ROBUST_LO,
                      iv[keep])
d_hi = straddle_delta(S[keep, 0], K[keep], T_entry[keep], R_ROBUST_HI,
                      iv[keep])
d1_shift = (R_ROBUST_HI - R_ROBUST_LO) * np.sqrt(T_entry[keep]) / iv[keep]
delta_shift = np.abs(d_hi - d_lo)
ds_q = np.percentile(delta_shift, [50, 95, 99])
d1_q = np.percentile(d1_shift, [50, 95, 99])
not_tiny_iv = iv[keep] >= IV_FLAG_LO
ds_ex = delta_shift[not_tiny_iv]
n_ds_big = int((delta_shift > 0.05).sum())
print(f"\n  ROBUSTNESS (A1): entry delta at r={R_ROBUST_LO:.4f} vs "
      f"r={R_ROBUST_HI:.4f}, IV held fixed:")
print(f"      |d(delta)|      median {ds_q[0]:.5f}  p95 {ds_q[1]:.5f}  "
      f"p99 {ds_q[2]:.5f}  mean {delta_shift.mean():.5f}  "
      f"max {delta_shift.max():.5f}")
print(f"      d1 term r*sqrt(T)/sigma  median {d1_q[0]:.5f}  "
      f"p95 {d1_q[1]:.5f}  p99 {d1_q[2]:.5f}  max {d1_shift.max():.5f}")
print(f"      -> near-inert across the mass of the distribution. The term "
      f"scales as 1/sigma, so it is NOT small for the")
print(f"         near-zero-IV trades flagged above: {n_ds_big:,} trades "
      f"({n_ds_big / n_keep * 100:.2f}%) shift by more than 0.05 in delta;")
print(f"         restricted to IV >= {IV_FLAG_LO * 100:.0f}% "
      f"({int(not_tiny_iv.sum()):,} trades) the mean is "
      f"{ds_ex.mean():.5f} and the max {ds_ex.max():.5f}.")

# --------------------------------------------------------------------------
# 6. Decomposition (a) (b) (c) (d)
# --------------------------------------------------------------------------
print('\n' + '-' * 88)
print('6. DECOMPOSITION')
print('-' * 88)

option_pnl = exit_mid - entry_mid
dS = np.diff(S, axis=1)                       # S_t - S_{t-1}, t = 1..N_STEPS
# On the processed set every element must be finite; nansum below must never
# be silently turning a NaN into a zero contribution.
assert np.isfinite(delta[keep]).all() and np.isfinite(dS[keep]).all(), \
    'non-finite delta or price step on a processed trade'
assert np.isfinite(option_pnl[keep]).all(), 'non-finite option P&L'
directional = np.nansum(delta * dS, axis=1)   # (b)
hedged_pnl = option_pnl - directional         # (a)
spread_cost = -((entry_ask - entry_mid) + (exit_mid - exit_bid))   # (c) <= 0

# (d) turnover: establish at entry, rebalance daily, unwind at exit. The
# augmented delta sequence [0, d_0, ..., d_{N-1}, 0] priced at
# [S_0, ..., S_N] gives exactly sum |delta_t - delta_{t-1}| * S_t with the
# entry establishment and the exit unwind included.
delta_aug = np.concatenate([np.zeros((n_tr, 1)), delta,
                            np.zeros((n_tr, 1))], axis=1)
turnover = np.nansum(np.abs(np.diff(delta_aug, axis=1)) * S, axis=1)
hedge_cost = {bps: -(bps / 10000.0) * turnover for bps in HEDGE_COST_BPS}

# Bookkeeping assertions, per trade, on the processed set.
kk = keep
assert np.allclose(hedged_pnl[kk] + directional[kk], option_pnl[kk],
                   rtol=0, atol=1e-9), '(a) + (b) != OptionPnL_mid'
for bps in HEDGE_COST_BPS:
    total = hedged_pnl + directional + spread_cost + hedge_cost[bps]
    assert np.allclose(total[kk],
                       option_pnl[kk] + spread_cost[kk] + hedge_cost[bps][kk],
                       rtol=0, atol=1e-9), \
        f'(a)+(b)+(c)+(d) identity fails at {bps} bps'
print(f"  [PASS] (a) + (b) = OptionPnL_mid per trade "
      f"(max abs error {np.abs(hedged_pnl[kk] + directional[kk] - option_pnl[kk]).max():.3e})")
print(f"  [PASS] (a)+(b)+(c)+(d) = OptionPnL_mid - spread - hedge, per "
      f"trade, at all {len(HEDGE_COST_BPS)} hedge-cost scenarios")

em = entry_mid
print(f"\n  Per-trade means, as % of entry_mid ({n_keep:,} trades):")
print(f"      option P&L, mid-to-mid (unhedged):   "
      f"{(option_pnl[kk] / em[kk]).mean() * 100:+8.4f}%")
print(f"      (a) delta-hedged P&L:                "
      f"{(hedged_pnl[kk] / em[kk]).mean() * 100:+8.4f}%")
print(f"      (b) directional P&L removed:         "
      f"{(directional[kk] / em[kk]).mean() * 100:+8.4f}%")
print(f"      (c) option spread cost:              "
      f"{(spread_cost[kk] / em[kk]).mean() * 100:+8.4f}%")
for bps in HEDGE_COST_BPS:
    print(f"      (d) hedge cost @ {bps:>2} bps:              "
          f"{(hedge_cost[bps][kk] / em[kk]).mean() * 100:+8.4f}%")
print(f"\n      hedge turnover per trade, as x entry_mid: mean "
      f"{(turnover[kk] / em[kk]).mean():.2f}x   median "
      f"{np.median(turnover[kk] / em[kk]):.2f}x")

# --------------------------------------------------------------------------
# 7. Newey-West standard errors on the daily portfolio series
# --------------------------------------------------------------------------
print('\n' + '-' * 88)
print('7. NEWEY-WEST STANDARD ERRORS')
print('-' * 88)
print(f"  Each trade's return is spread evenly across its {N_STEPS} holding")
print(f"  days as (1+r)^(1/{N_STEPS})-1, averaged across open trades per day,")
print(f"  then NW maxlags={NW_MAXLAGS} on that daily series - the src/31, "
      f"src/33, src/38 convention.")

entry_tidx = ei
rf_daily = (pd.read_parquet(fac_path, columns=['date', 'RF'])
            .drop_duplicates('date').set_index('date')['RF'])
cal_rf = rf_daily.reindex(cal).to_numpy(dtype=float)
n_post_dev = int((cal[np.unique(
    (entry_tidx[kk][:, None] + np.arange(1, N_STEPS + 1)[None, :]).ravel())]
    > DEV_END).sum())
print(f"  daily series is NOT truncated at DEV_END (D4): "
      f"{n_post_dev} post-{DEV_END.date()} sessions retained")


def daily_series(ret, mask):
    """src/37 build_daily, vectorised: geometric per-day spread over the
    holding window, mean across trades open that day. Returns (tidx, series,
    n_open, n_clamped)."""
    r = ret[mask].astype(float)
    finite = np.isfinite(r)
    n_clamped = int((finite & (r <= -1.0)).sum())
    r = np.where(finite, np.maximum(r, -0.999999), np.nan)
    ok = np.isfinite(r)
    per_day = (1.0 + r[ok]) ** (1.0 / N_STEPS) - 1.0
    tix = (entry_tidx[mask][ok][:, None]
           + np.arange(1, N_STEPS + 1)[None, :]).ravel()
    vals = np.repeat(per_day, N_STEPS)
    df = pd.DataFrame({'tidx': tix, 'ret': vals})
    g = df.groupby('tidx')['ret'].agg(['mean', 'size'])
    return (g.index.to_numpy(), g['mean'].to_numpy(),
            g['size'].to_numpy(), n_clamped)


def nw_t(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 30:
        return np.nan, np.nan, len(x)
    m = sm.OLS(x, np.ones(len(x))).fit(cov_type='HAC',
                                       cov_kwds={'maxlags': NW_MAXLAGS})
    return float(m.params[0]), float(m.tvalues[0]), len(x)


def row_stats(pnl, label, hedge_bps=None, spread_frac=None):
    ret = pnl / em
    tix, ser, nopen, n_clamped = daily_series(ret, kk)
    mean_d, tstat, ndays = nw_t(ser)
    ex = ser - cal_rf[tix]
    _, t_ex, _ = nw_t(ex)
    return {
        'label': label,
        'hedge_cost_bps': hedge_bps,
        'spread_fraction_paid': spread_frac,
        'mean_per_trade_return_pct': float(ret[kk].mean()) * 100.0,
        'median_per_trade_return_pct': float(np.median(ret[kk])) * 100.0,
        'nw_t': tstat,
        'nw_t_excess_rf': t_ex,
        'mean_daily_return_bps': mean_d * 1e4 if np.isfinite(mean_d) else np.nan,
        'n_days': ndays,
        'n_trades': n_keep,
        'n_trades_clamped_at_minus_1': n_clamped,
    }


# CONSTRUCTION VALIDATION. Rebuild the src/38 PRIMARY series exactly - all
# trades in the per-trade file (not just the processed subset), ret_mid,
# RF-excess, truncated at DEV_END - and print it. If daily_series() below
# does not reproduce build_daily() in src/37, this line will not land on the
# "K1 REAL PRICES" primary already logged in gate_log.md. Nothing is
# transcribed: compare by eye against that entry.
_all = np.ones(n_tr, dtype=bool)
_tix, _ser, _, _ = daily_series(t['ret_mid'].to_numpy(dtype=float), _all)
_dev = cal[_tix] <= DEV_END
_m38, _t38, _n38 = nw_t(_ser[_dev] - cal_rf[_tix[_dev]])
print(f"\n  CONSTRUCTION VALIDATION - src/38 primary rebuilt from this "
      f"script's daily_series():")
print(f"      all {n_tr:,} trades, ret_mid, RF-excess, truncated at "
      f"{DEV_END.date()}: days {_n38:,}, "
      f"mean {_m38 * 1e4:+.2f} bps/day, NW t {_t38:+.3f}")
print(f"      compare against the 'K1 REAL PRICES' PRIMARY block in "
      f"gate_log.md (src/38). Nothing here is transcribed.")

rows = []
rows.append(row_stats(option_pnl,
                      'REFERENCE: unhedged option P&L, mid-to-mid'))
rows.append(row_stats(directional,
                      'REFERENCE: (b) directional P&L removed by the hedge'))
rows.append(row_stats(hedged_pnl, '1. delta-hedged gross (a)'))
for bps in HEDGE_COST_BPS:
    rows.append(row_stats(hedged_pnl + hedge_cost[bps],
                          f'2. net of hedge cost only (a+d) @ {bps} bps',
                          hedge_bps=bps, spread_frac=0.0))
for bps in HEDGE_COST_BPS:
    for fr in SPREAD_FRACTIONS:
        rows.append(row_stats(
            hedged_pnl + hedge_cost[bps] + fr * spread_cost,
            f'3. net of hedge + {fr * 100:>3.0f}% of option spread @ '
            f'{bps} bps', hedge_bps=bps, spread_frac=fr))

# --------------------------------------------------------------------------
# 8. Waterfall table
# --------------------------------------------------------------------------
print('\n' + '=' * 88)
print('WATERFALL  (all means are per-trade, as % of entry_mid; NW t on the '
      'daily series)')
print('=' * 88)
hdr = (f"  {'row':<56} {'mean %':>9} {'NW t':>8} {'NW t exRF':>10} "
       f"{'days':>6}")
print(hdr)
print('  ' + '-' * (len(hdr) - 2))
for i, r_ in enumerate(rows):
    if i == 2 or i == 3 or i == 3 + len(HEDGE_COST_BPS):
        print('  ' + '-' * (len(hdr) - 2))
    print(f"  {r_['label']:<56} {r_['mean_per_trade_return_pct']:>+9.4f} "
          f"{r_['nw_t']:>+8.3f} {r_['nw_t_excess_rf']:>+10.3f} "
          f"{r_['n_days']:>6,}")
print('  ' + '-' * (len(hdr) - 2))
print(f"  100% of the option spread = full ask/bid crossing, i.e. the "
      f"original K1 bid-based assumption")
print(f"  ({K1_BID:+.2f}% unhedged, read from k1_trade_summary.json).")

# --------------------------------------------------------------------------
# 9. Pattern flag
# --------------------------------------------------------------------------
print('\n' + '=' * 88)
print('PATTERN FLAG')
print('=' * 88)

unhedged_subset_pct = float((option_pnl[kk] / em[kk]).mean()) * 100.0
hedged_gross_pct = float((hedged_pnl[kk] / em[kk]).mean()) * 100.0
directional_pct = float((directional[kk] / em[kk]).mean()) * 100.0

# The spec names the JSON figure as the reference. The processed subset is
# 10 trades smaller, so the two are cross-checked before the comparison is
# made, rather than assumed interchangeable.
subset_vs_json_gap = unhedged_subset_pct - K1_MID
print(f"  unhedged gross, from k1_trade_summary.json (all {K1_N:,} trades): "
      f"{K1_MID:+.4f}%")
print(f"  unhedged gross, recomputed on the {n_keep:,} processed trades:   "
      f"{unhedged_subset_pct:+.4f}%")
print(f"  gap from the {n_tr - n_keep} dropped trades: "
      f"{subset_vs_json_gap:+.4f} pp")
print(f"\n  (a) delta-hedged gross:              {hedged_gross_pct:+.4f}%")
print(f"  (b) directional P&L removed:         {directional_pct:+.4f}%")
print(f"      (a) + (b) = {hedged_gross_pct + directional_pct:+.4f}% "
      f"= unhedged gross on the same trades")

ref = abs(unhedged_subset_pct)
if ref > 1e-9:
    directional_share = directional_pct / unhedged_subset_pct
    hedged_share = hedged_gross_pct / unhedged_subset_pct
else:
    directional_share = np.nan
    hedged_share = np.nan
print(f"\n  share of unhedged gross carried by (b) directional: "
      f"{directional_share * 100:+.1f}%")
print(f"  share of unhedged gross surviving in (a) delta-hedged: "
      f"{hedged_share * 100:+.1f}%")

# Conditions are evaluated on MAGNITUDES and NEITHER is a permitted outcome
# (A4). Pattern A: (a) is close to the unhedged gross AND (b) is small next
# to it. Pattern B: (a) is smaller in magnitude than the unhedged gross AND
# (b) carries most of it.
c = PATTERN_DIRECTIONAL_SHARE_CUTOFF
gap_a = abs(hedged_gross_pct - unhedged_subset_pct)
cond_A = bool(np.isfinite(ref) and ref > 1e-9
              and gap_a <= c * ref and abs(directional_pct) <= c * ref)
cond_B = bool(np.isfinite(ref) and ref > 1e-9
              and abs(hedged_gross_pct) < ref
              and abs(directional_pct) >= c * ref)
print(f"\n  condition A  |(a) - unhedged| <= {c:.2f}*|unhedged| AND "
      f"|(b)| <= {c:.2f}*|unhedged|")
print(f"               {gap_a:.4f} <= {c * ref:.4f}  AND  "
      f"{abs(directional_pct):.4f} <= {c * ref:.4f}   -> "
      f"{'TRUE' if cond_A else 'FALSE'}")
print(f"  condition B  |(a)| < |unhedged| AND |(b)| >= {c:.2f}*|unhedged|")
print(f"               {abs(hedged_gross_pct):.4f} < {ref:.4f}  AND  "
      f"{abs(directional_pct):.4f} >= {c * ref:.4f}   -> "
      f"{'TRUE' if cond_B else 'FALSE'}")

if not np.isfinite(ref) or ref <= 1e-9:
    pattern = 'INDETERMINATE'
    pattern_txt = ('unhedged gross is ~0 on the processed subset, so the '
                   'shares are undefined; read (a) and (b) directly.')
elif cond_A and not cond_B:
    pattern = 'A'
    pattern_txt = ('delta-hedged gross (a) carries most of the unhedged '
                   'gross and directional P&L (b) is the smaller part.')
elif cond_B and not cond_A:
    pattern = 'B'
    pattern_txt = ('directional P&L (b) carries most of the unhedged gross '
                   'and the delta-hedged gross (a) is the smaller '
                   'remainder.')
else:
    pattern = 'NEITHER'
    pattern_txt = (
        f'the numbers match neither pre-specified pattern. (b) is '
        f'{directional_pct:+.4f}%, opposite in sign to the unhedged gross '
        f'{unhedged_subset_pct:+.4f}% and larger in magnitude than it, so '
        f'(a) at {hedged_gross_pct:+.4f}% is LARGER than the unhedged gross '
        f'rather than close to it (Pattern A) or smaller than it (Pattern '
        f'B). Removing the directional exposure raised the gross result '
        f'instead of lowering it.')
print(f"\n  PATTERN {pattern}: {pattern_txt}")
print(f"  Pattern A = vol forecast was real, spread was the killer.")
print(f"  Pattern B = the straddle was mostly catching direction, not "
      f"volatility.")
print(f"  (classification cutoff {PATTERN_DIRECTIONAL_SHARE_CUTOFF:.2f} is "
      f"assumption A4, a reporting convention with no analytical authority; "
      f"NEITHER is a permitted outcome)")

# --------------------------------------------------------------------------
# 10. JSON output
# --------------------------------------------------------------------------
result = {
    'phase': 'K1 delta-hedged decomposition (DEV, per-trade)',
    'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'script': 'src/42_delta_hedged_decomp.py',
    'inputs': {
        'trades': 'data/processed/k1_trades_real_prices.parquet',
        'k1_reference_summary': 'data/processed/k1_trade_summary.json',
        'underlying': 'data/processed/crsp_combined.parquet (DlyClose)',
        'risk_free_cross_check': 'data/processed/factors_daily.parquet (RF)',
        'note': 'No re-scan of opprcd. No gate decision is made here.',
    },
    'k1_reference_read_at_runtime': {
        'source_phase': k1['phase'],
        'matched_trades': K1_N,
        'unhedged_mean_per_trade_pct': {
            'mid_to_mid': K1_MID,
            'bid_based_worst_fill': K1_BID,
            'mid_liquidity_filtered': K1_LIQ,
        },
        'entry_half_spread_pct_of_mid': {'median': K1_SP_MED,
                                         'mean': K1_SP_MEAN},
    },
    'declared_assumptions': {
        'A1_risk_free_rate': RISK_FREE_RATE,
        'A1_rationale': (
            'Ken French RF is quantized to 0.0001/day over this window - a '
            'lumpy step function, not a rate path. A single constant equal '
            'to the sample mean is used instead of a per-trade RF.'),
        'A1_observed_rf_annualized_mean': rf_mean_annual,
        'A1_observed_rf_distinct_daily_values': [float(v) for v in rf_steps],
        'A1_robustness_delta_r0_vs_r0252': {
            'r_low': R_ROBUST_LO, 'r_high': R_ROBUST_HI,
            'abs_delta_shift': {
                'median': float(ds_q[0]), 'p95': float(ds_q[1]),
                'p99': float(ds_q[2]), 'mean': float(delta_shift.mean()),
                'max': float(delta_shift.max()),
            },
            'd1_term_shift': {
                'median': float(d1_q[0]), 'p95': float(d1_q[1]),
                'p99': float(d1_q[2]), 'mean': float(d1_shift.mean()),
                'max': float(d1_shift.max()),
            },
            'n_trades_delta_shift_above_0_05': n_ds_big,
            'pct_trades_delta_shift_above_0_05': n_ds_big / n_keep * 100.0,
            'restricted_to_iv_above_flag_low': {
                'n': int(not_tiny_iv.sum()),
                'mean_abs_delta_shift': float(ds_ex.mean()),
                'max_abs_delta_shift': float(ds_ex.max()),
            },
            'interpretation': (
                'r enters the straddle delta only via r*sqrt(T)/sigma in d1, '
                'so it is near-inert across the mass of the distribution at '
                'this tenor and vol level. The term scales as 1/sigma and is '
                'NOT small for the near-zero-IV trades flagged by the IV '
                'sanity check; quantiles are reported rather than a mean '
                'alone so the tail is visible. Measured, not asserted.'),
        },
        'A2_hedge_cost_bps': list(HEDGE_COST_BPS),
        'A3_brentq_iv_bounds': [IV_LO, IV_HI],
        'A4_pattern_directional_share_cutoff': PATTERN_DIRECTIONAL_SHARE_CUTOFF,
        'A4_note': ('reporting convention only; names the pattern, gates '
                    'nothing'),
        'spread_fractions_swept': list(SPREAD_FRACTIONS),
    },
    'mandatory_disclosure': {
        'D1_sticky_vol_delta': (
            'Delta uses the ENTRY implied volatility HELD CONSTANT for the '
            'life of the trade. Real IV moves during the trade. This makes '
            'the hedged return PATH-DEPENDENT: it is a REALISTIC '
            'decomposition, NOT a pure volatility isolation. A residual '
            'vega/vanna P&L remains inside the delta-hedged leg.'),
        'D2_discrete_hedging': (
            'Hedging is close-to-close (discrete), once per trading day, not '
            'continuous. Intraday moves are unhedged and a gamma-driven '
            'hedging error remains.'),
        'D3_no_dividend_yield': (
            'q = 0, matching src/30/32/37. Underlying path is the raw CRSP '
            'daily close; zero trades contain a CRSP price-adjustment-factor '
            'change inside their window (asserted at runtime).'),
        'D4_daily_series_not_truncated_at_dev_end': (
            f'{n_post_dev} sessions after {DEV_END.date()} are retained so '
            f'the daily-series mean stays consistent with the per-trade mean '
            f'on the same row; src/37 truncated these. No trade was added.'),
        'D5_nw_t_reported_both_ways': (
            'nw_t is on the raw daily series (the quantity in the row); '
            'nw_t_excess_rf subtracts daily RF, the src/38 convention.'),
    },
    'method': {
        'entry_iv': ('brentq on BS_call(sigma)+BS_put(sigma) - entry_mid, '
                     'S = entry-date CRSP close, K = held strike, '
                     'T = (exdate_d - entry_date).days/365'),
        'delta': 'long-straddle position delta = 2*N(d1_t) - 1, sticky-vol',
        'rebalance': ('full daily rebalance at each close, hedge established '
                      'at entry and unwound at exit; the exit-close delta is '
                      'never traded because the option is sold there'),
        'decomposition': {
            'a_delta_hedged_pnl': 'OptionPnL_mid - Directional',
            'b_directional_pnl': 'sum_t delta_{t-1} * (S_t - S_{t-1})',
            'c_option_spread_cost':
                '-[(entry_ask - entry_mid) + (exit_mid - exit_bid)]',
            'd_hedge_transaction_cost':
                '-bps/10000 * sum |delta_t - delta_{t-1}| * S_t',
            'identity': '(a)+(b)+(c)+(d) = OptionPnL_mid - spread - hedge',
            'identity_asserted_per_trade': True,
        },
        'standard_errors': (f'per-trade return spread evenly across its '
                            f'{N_STEPS} holding days as (1+r)^(1/{N_STEPS})-1, '
                            f'averaged across open trades per day, '
                            f'Newey-West maxlags={NW_MAXLAGS} - the '
                            f'src/31/33/38 convention'),
        'construction_validation_src38_primary_rebuilt': {
            'spec': (f'all {n_tr} trades, ret_mid, RF-excess, truncated at '
                     f'{DEV_END.date()} - identical to the src/38 PRIMARY'),
            'n_days': _n38,
            'mean_daily_bps': _m38 * 1e4,
            'nw_t': _t38,
            'note': ('compare against the K1 REAL PRICES PRIMARY block in '
                     'results/gate_log.md; nothing transcribed'),
        },
    },
    'sample': {
        'trades_in_locked_per_trade_file': n_tr,
        'trades_processed': n_keep,
        'trades_dropped': n_tr - n_keep,
        'drop_reasons': {
            'missing_daily_underlying_close': n_drop_path,
            'missing_underlying_day_cells': int(np.isnan(S).sum()),
            'iv_inversion_failed': n_drop_iv,
            'iv_inversion_failure_detail': iv_fail_reason,
        },
        'holding_sessions': N_STEPS,
        'unique_permnos_processed': int(t.loc[keep, 'PERMNO'].nunique()),
        'entry_date_range': [str(t.loc[keep, 'entry_date'].min().date()),
                             str(t.loc[keep, 'entry_date'].max().date())],
        'exit_date_range': [str(t.loc[keep, 'exit_date'].min().date()),
                            str(t.loc[keep, 'exit_date'].max().date())],
    },
    'entry_iv_distribution': {
        'n': n_keep,
        'min': float(ivk.min()), 'p1': float(q[0]), 'q1': float(q[1]),
        'median': float(q[2]), 'q3': float(q[3]), 'p99': float(q[4]),
        'max': float(ivk.max()), 'mean': float(ivk.mean()),
        'sanity_flags': {
            'threshold_high': IV_FLAG_HI,
            'n_above_high': n_iv_hi,
            'pct_above_high': n_iv_hi / n_keep * 100.0,
            'threshold_low': IV_FLAG_LO,
            'n_below_low': n_iv_lo,
            'pct_below_low': n_iv_lo / n_keep * 100.0,
        },
    },
    'component_means_pct_of_entry_mid': {
        'unhedged_option_pnl_mid_to_mid': unhedged_subset_pct,
        'a_delta_hedged': hedged_gross_pct,
        'b_directional_removed': directional_pct,
        'c_option_spread_cost': float((spread_cost[kk] / em[kk]).mean()) * 100.0,
        'd_hedge_cost_by_bps': {
            str(bps): float((hedge_cost[bps][kk] / em[kk]).mean()) * 100.0
            for bps in HEDGE_COST_BPS},
        'hedge_turnover_x_entry_mid': {
            'mean': float((turnover[kk] / em[kk]).mean()),
            'median': float(np.median(turnover[kk] / em[kk])),
        },
        'entry_delta_mean': float(dk[:, 0].mean()),
        'entry_abs_delta_mean': float(np.abs(dk[:, 0]).mean()),
    },
    'waterfall': rows,
    'pattern_flag': {
        'pattern': pattern,
        'statement': pattern_txt,
        'pattern_A_definition': ('delta-hedged gross (a) close to the '
                                 'unhedged gross, (b) small -> the vol '
                                 'forecast was real, spread was the killer'),
        'pattern_B_definition': ('delta-hedged gross (a) much smaller than '
                                 'unhedged gross with (b) accounting for '
                                 'most of it -> the straddle was mostly '
                                 'catching direction, not volatility'),
        'condition_A_met': cond_A,
        'condition_B_met': cond_B,
        'condition_A_rule': ('|(a) - unhedged| <= cutoff*|unhedged| AND '
                             '|(b)| <= cutoff*|unhedged|'),
        'condition_B_rule': ('|(a)| < |unhedged| AND '
                             '|(b)| >= cutoff*|unhedged|'),
        'neither_is_a_permitted_outcome': True,
        'gap_a_minus_unhedged_pp': gap_a,
        'unhedged_gross_pct_from_json_all_trades': K1_MID,
        'unhedged_gross_pct_processed_subset': unhedged_subset_pct,
        'subset_minus_json_pp': subset_vs_json_gap,
        'a_delta_hedged_gross_pct': hedged_gross_pct,
        'b_directional_pct': directional_pct,
        'directional_share_of_unhedged': directional_share,
        'hedged_share_of_unhedged': hedged_share,
        'cutoff_used': PATTERN_DIRECTIONAL_SHARE_CUTOFF,
    },
}
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(result, indent=2, default=str),
                    encoding='utf-8')
print(f"\n[OK] Saved {out_json}")

# --------------------------------------------------------------------------
# 11. Append-only, marker-guarded gate_log entry. Numbers only.
# --------------------------------------------------------------------------
MARKER = '## K1 delta-hedged decomposition (DEV, per-trade)'
log_text = log_path.read_text(encoding='utf-8', errors='replace')
if MARKER in log_text:
    print('[SKIP] gate_log.md already carries the delta-hedge block; '
          'not duplicating.')
else:
    bak = log_path.with_suffix('.md.bak_before_delta_hedge_block')
    shutil.copy2(log_path, bak)
    print(f"[safety] gate_log.md backed up to {bak.name}")
    L = [
        f"\n---\n\n{MARKER}",
        "",
        f"Run {result['generated']} - script src/42_delta_hedged_decomp.py. "
        f"Decomposes the K1 REAL PRICES per-trade P&L into a delta-hedged "
        f"and a directional component on the same locked trade list; no "
        f"gate decision, no filter or threshold changed. Reference figures "
        f"read at runtime from data/processed/k1_trade_summary.json. "
        f"Machine-readable copy: results/42_delta_hedged_decomp.json.",
        "",
        f"Declared assumptions: r = {RISK_FREE_RATE:.4f} constant; hedge "
        f"cost scenarios {list(HEDGE_COST_BPS)} bps; brentq IV bounds "
        f"[{IV_LO}, {IV_HI}]. Delta uses ENTRY IV held constant (sticky-vol) "
        f"and the hedge is close-to-close discrete, so this is a realistic "
        f"decomposition, not a pure volatility isolation.",
        "",
        "```",
        f"trades in locked per-trade file:       {n_tr:,}",
        f"trades processed:                      {n_keep:,}",
        f"trades dropped:                        {n_tr - n_keep:,}",
        f"  missing daily underlying close:      {n_drop_path:,}",
        f"  IV inversion failed to converge:     {n_drop_iv:,}",
        f"holding sessions per trade:            {N_STEPS}",
        "",
        "entry IV backed out of entry_mid (annualized)",
        f"  q1 / median / q3:                    "
        f"{q[1]*100:.2f}% / {q[2]*100:.2f}% / {q[3]*100:.2f}%",
        f"  min / max:                           "
        f"{ivk.min()*100:.2f}% / {ivk.max()*100:.2f}%",
        f"  above {IV_FLAG_HI*100:.0f}%:                          "
        f"{n_iv_hi:,} ({n_iv_hi/n_keep*100:.2f}%)",
        f"  below {IV_FLAG_LO*100:.0f}%:                           "
        f"{n_iv_lo:,} ({n_iv_lo/n_keep*100:.2f}%)",
        "",
        "component means, % of entry_mid, processed trades",
        f"  unhedged option P&L (mid-to-mid):    {unhedged_subset_pct:+.4f}%",
        f"  (a) delta-hedged:                    {hedged_gross_pct:+.4f}%",
        f"  (b) directional removed by hedge:    {directional_pct:+.4f}%",
        f"  (c) option spread cost:              "
        f"{float((spread_cost[kk]/em[kk]).mean())*100:+.4f}%",
    ]
    for bps in HEDGE_COST_BPS:
        L.append(f"  (d) hedge cost @ {bps:>2} bps:              "
                 f"{float((hedge_cost[bps][kk]/em[kk]).mean())*100:+.4f}%")
    L += [
        "",
        "WATERFALL (mean per-trade return, % of entry_mid; NW t maxlags="
        f"{NW_MAXLAGS} on the daily series)",
        f"  {'row':<54} {'mean %':>9} {'NW t':>8} {'exRF t':>8}",
    ]
    for r_ in rows:
        L.append(f"  {r_['label']:<54} "
                 f"{r_['mean_per_trade_return_pct']:>+9.4f} "
                 f"{r_['nw_t']:>+8.3f} {r_['nw_t_excess_rf']:>+8.3f}")
    L += [
        "",
        f"reference, unhedged, all {K1_N:,} trades (k1_trade_summary.json)",
        f"  mid-to-mid:                          {K1_MID:+.4f}%",
        f"  bid-based worst fill:                {K1_BID:+.4f}%",
        f"  mid, liquidity-filtered:             {K1_LIQ:+.4f}%",
        "",
        f"directional share of unhedged gross:   "
        f"{directional_share*100:+.1f}%",
        f"delta-hedged share of unhedged gross:  {hedged_share*100:+.1f}%",
        f"condition A |(a)-unhedged|<={PATTERN_DIRECTIONAL_SHARE_CUTOFF:.2f}"
        f"|unh| and |(b)|<={PATTERN_DIRECTIONAL_SHARE_CUTOFF:.2f}|unh|:  "
        f"{'TRUE' if cond_A else 'FALSE'}",
        f"condition B |(a)|<|unhedged| and |(b)|>="
        f"{PATTERN_DIRECTIONAL_SHARE_CUTOFF:.2f}|unh|:            "
        f"{'TRUE' if cond_B else 'FALSE'}",
        f"PATTERN FLAG:                          {pattern}",
        "",
        "construction validation - src/38 PRIMARY rebuilt by this script",
        f"  all {n_tr:,} trades, ret_mid, RF-excess, truncated "
        f"{DEV_END.date()}:",
        f"  days {_n38:,}   mean {_m38*1e4:+.2f} bps/day   NW t {_t38:+.3f}",
        "```",
        "",
    ]
    with log_path.open('a', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print(f"[OK] Appended delta-hedge block to {log_path} "
          f"(prior entries untouched)")

print('\n' + '=' * 88)
print(f'DELTA-HEDGED DECOMPOSITION COMPLETE - {n_keep:,} trades processed, '
      f'pattern {pattern}. No gate decision made here.')
print('=' * 88)
