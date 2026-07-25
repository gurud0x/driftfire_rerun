# Pre-Registration — K1: Straddle on Compression-Predicted Volatility
## DriftFire Rerun, Phase K1

**Status:** LOCKED upon git commit. This is the project's first ALPHA
claim (a real position, in option space, with real costs). Nothing here
is amended after commit; a change in results does not retroactively
justify a change in design.

---

## 1. Hypothesis

Buying an ATM straddle on stocks in V1's most-compressed decile (decile
1), when TRAIL20's forecast of forward realized volatility exceeds the
market's current implied volatility by a pre-committed margin, produces
a positive net return after realistic option costs. This is the first
test in the project that could plausibly generate tradeable alpha; V1
and V2 established the underlying forecasting relationship but were
explicitly not alpha claims.

## 2. What is locked in from prior phases (not re-derived, not re-tested)

- Universe and signal: V1's compression decile, computed exactly as in
  `src/10_compression_signal_v1.py`. No re-tuning of the compression
  definition here.
- Forecaster: **TRAIL20** (trailing 20-day realized volatility). Winner
  of the pre-registered K0 horse race at the 10-day horizon (MAE 0.1751
  vs GARCH(1,1) 0.1810 vs compression-decile-mean 0.1980) and confirmed
  competitive at 30 days (MAE 0.1644 vs GARCH's 0.1621 — GARCH's small
  edge is smaller than its in-sample fitting advantage and is not
  treated as a genuine win; see `src/26_forecaster_horse_race_30d.py`).
  No new forecaster search occurs in K1. A future, separately
  pre-registered phase may re-run the horse race with additional
  candidates; K1 does not anticipate or lean toward that outcome.
- Tenor: **30 calendar days**, not 10. Justified by two independent,
  pre-K1 findings that point the same direction: (a) OptionMetrics
  coverage in this universe is concentrated at the 30-day tenor (90.6%
  of valid near-ATM points vs ~9.4% at 10 days — small-caps rarely have
  listed expirations near 10 days); (b) the compression signal's
  forecasting power does not decay by 30 days — it strengthens (NW
  t = -8.18 at both 20d and 30d, vs -6.88 at 10d; see
  `src/25_compression_decay_check.py`). Tenor was chosen before this
  design doc was written and is not revisited based on K1's results.

## 3. OptionMetrics data and known constraints (disclosed up front)

- Volatility surface data covers 2015-01-02 through 2025-08-29 (not
  through 2025-12-31 like other project data). K1's sample ends at the
  data's actual end date; no extrapolation past it.
- Coverage of V1 decile-1 stock-days within the covered window: ~90%
  overall (84.4% including the post-August-2025 gap). Stock-days with
  no matching near-ATM IV point are EXCLUDED from the tradeable
  universe, not filled, proxied, or estimated.
- Coverage is uneven by size: decile 6 90.8%, decile 7 87.8%, decile 8
  78.7% (`src/28_validate_vol_surface_final.py`). K1's traded universe
  is therefore mildly tilted toward the larger end of deciles 6-8
  relative to V1's full decile-1 sample. This composition shift is
  reported in any write-up of K1, not hidden.
- Near-ATM defined as delta in [0.35, 0.65] (calls) matched with the
  put side at the same date/tenor; both legs are confirmed present for
  100% of populated secid-dates, so straddle pricing is never a
  single-leg problem where coverage exists at all.
- Known sentinel handled upstream: CUSIP `99999999` (OptionMetrics
  placeholder for unidentified/index instruments) is excluded from the
  security-linking join before any K1 computation.

## 4. Context disclosed, not acted on

Matched-sample mean implied volatility (0.61 annualized) exceeds V1
decile-1's mean realized volatility in the dev window (0.4656). This
suggests options on these names may carry a systematic volatility risk
premium — i.e., IV may on average overstate subsequent realized moves,
independent of any signal. K1's entry threshold (Section 6) is
DESIGNED to filter for cases where this premium is plausibly absent or
reversed, but the premium's existence is disclosed here, before seeing
any P&L, as a reason the trade may not be profitable on average even if
the underlying volatility forecast is accurate.

## 5. Trade construction

- Instrument: ATM straddle (1 call + 1 put, same strike, same
  expiration), 30-day tenor, near-ATM (delta band as in Section 3).
- Entry: next trading day after the compression signal and entry
  condition (Section 6) are both satisfied, at the mid price (last
  quoted bid + ask, divided by 2) for each leg.
- Exit: held to the earlier of (a) 10 trading days after entry — the
  horizon the underlying compression signal was gated on — or (b)
  option expiration, whichever comes first, closed at mid price. This
  is locked now, before data contact, specifically to avoid holding
  past the horizon the forecast actually covers.
- Position sizing: one straddle per qualifying stock-day, equal
  notional across all qualifying positions on a given day (no
  volatility-scaled or conviction-scaled sizing in this first pass).

## 6. Entry condition (the alpha claim, locked before data contact)

**Design note, superseding an earlier draft of this section:** the
original draft compared TRAIL20's forecast to IV as a raw volatility-point
gap. That is replaced here with a theoretical-price-vs-market-price
comparison: a fixed vol-point gap means something different on a cheap
option than an expensive one, and pricing the edge through the same
convex function the options market itself uses captures that
nonlinearity. TRAIL20 remains the forecaster — settled by the K0 horse
race (`src/21_forecaster_horse_race.py`, confirmed at 30d by
`src/26_forecaster_horse_race_30d.py`). No forecaster search is reopened
here; a Kronos-vs-TRAIL20 head-to-head, if it happens, is a separate,
future, independently pre-registered phase, run only after Kronos has its
own environment — not substituted in here on the strength of a name.

- **Market straddle value** = matched call-leg `impl_premium` + matched
  put-leg `impl_premium`, summed directly from the near-ATM 30-day
  surface points already established in Section 3. No new computation:
  these are OM's own fitted-surface premiums at the matched grid point,
  the same points used throughout K1's coverage validation.
- **Theoretical straddle value** = TRAIL20's forecast volatility, priced
  through standard Black-Scholes:
  - `S` = DlyClose on the signal evaluation day (day t) — the same day
    TRAIL20 and the matched IV are computed from; no look-ahead.
  - `K` = average of the matched call-leg and put-leg `impl_strike`
    (conventionally near-identical; averaging yields one strike for the
    straddle, matching Section 5's "same strike, same expiration").
  - `T` = 30/365 (a year-fraction from the 30-calendar-day tenor in
    Section 5, not trading days).
  - `r` = daily RF from `data/processed/factors_daily.parquet` on day t,
    annualized as `RF * 252` — the same annualization convention this
    project already uses for realized vol throughout V1/V2.
  - `sigma` = TRAIL20's forecast (annualized decimal), applied to BOTH
    legs — one forecast, no smile. Deliberate simplification, disclosed
    here.
  - `q` = 0 (no dividend adjustment). Disclosed simplification, same
    status as Section 7's no-commission disclosure.
  - `d1 = [ln(S/K) + (r + 0.5*sigma^2)*T] / (sigma*sqrt(T))`,
    `d2 = d1 - sigma*sqrt(T)`.
    `Call = S*N(d1) - K*exp(-r*T)*N(d2)`,
    `Put = K*exp(-r*T)*N(-d2) - S*N(-d1)`.
    Theoretical straddle = Call + Put.
- **Entry rule**: only enter if theoretical straddle value exceeds market
  straddle value by at least **T = 25%** of market value, i.e.
  `(theoretical - market) / market >= 0.25`.
- **How T was set, not calibrated against any K1 result**: Section 4
  already disclosed, before this section was rewritten, that
  matched-sample mean IV (0.6105) exceeds V1 decile-1's dev-window mean
  realized vol (0.4656) — an average structural premium of
  `(0.6105 - 0.4656) / 0.6105 = 23.7%`. T = 25% is that figure rounded
  UP, so the threshold requires more forecast edge than the average
  premium, not merely enough to match it. This derives from data already
  on the page before this rewrite, not from any K1 trade outcome.
- **Zero-candidate days**: unchanged — if no decile-1 stock clears the
  threshold on a given day, no trade is placed. No fallback, no relaxed
  threshold on thin days.
- **Relation to Section 7's costs**: this comparison uses OM's
  surface-implied premiums only to decide WHETHER to enter. Once
  triggered, the trade executes and is costed per Section 7 (actual
  quoted mid +/- half-spread) — a decision filter and an executed fill,
  not duplicate accounting of the same number.

## 7. Costs

- Entry and exit costs both legs: mid price minus/plus half the quoted
  bid-ask spread (buy at mid + half-spread, sell at mid - half-spread),
  applied on both entry and exit.
- No separate commission model in this first pass (disclosed as a
  simplification; real per-contract commissions would add further
  drag, meaning any positive result here is an upper bound on realistic
  performance, not a conservative estimate).
- **Bid/ask spread data does not exist in the pulled OptionMetrics
  files.** `vol_surface.csv` and `om_security_names.csv` — the only two
  OptionMetrics files pulled for this project — carry `impl_premium`, a
  single fitted theoretical price per leg from OM's surface model, but
  no quoted bid or ask. The "mid minus/plus half-spread" design above
  cannot be computed without that data. K1's implementation uses
  `impl_premium` directly as the transacted price for both entry and
  exit, applying NO explicit spread cost. Disclosed here, discovered
  during implementation, not silently substituted mid-run: this joins
  the no-commission disclosure above as a second, larger source of cost
  understatement, and makes any positive K1 result more strongly an
  upper bound than originally anticipated. A future phase that pulls
  actual quoted bid/ask could retest K1 faithfully to this section's
  original intent; this phase does not wait for that pull.

## 8. Test methodology and gate

- Dev window: 2015-01-01 to 2021-12-31 (identical split to every prior
  phase). **The full dev window is the PRIMARY gate — it alone decides
  PASS/FAIL**, per the criterion below.
- Metric: net return per trade and portfolio-level daily net return
  (equal-weighted across concurrent open positions), Newey-West t-test
  on the daily return series, maxlags=10.
- PASS requires: mean daily net return positive AND Newey-West t-stat
  >= 2.0 (the project's standard alpha-claim bar, matching R1/R2's
  gate, not V1/V2's higher exploratory-forecast bar).
- **2020 stratification, disclosed before any K1 return or P&L has been
  computed:** `src/29_k1_threshold_calibration.py`'s count-only
  calibration check found that 44% of dev-window qualifying trade-days
  under the locked T=25% rule (2,391 of 5,426) fall in calendar year
  2020 alone — far above 2020's ~1/7 share of the dev window. Two
  additional views are computed and reported alongside the primary
  result, **with explicitly NO gate authority**: (a) dev window
  EXCLUDING 2020, (b) 2020 ALONE. The primary gate decision uses the
  full dev window regardless of what either diagnostic view shows. This
  split is locked now, before any K1 return has been computed, so that
  a result driven disproportionately by one calendar year is visible
  rather than hidden inside a single aggregate statistic.
- Holdout: 2022-01-01 through the data's actual end (2025-08-29), ONE
  pass, only after a logged dev PASS, same enforcement pattern as every
  prior phase — no holdout code path in the dev script. **The same
  three-way reporting split — full holdout, holdout excluding its most
  concentrated year (if one comparably concentrated year emerges), and
  that year alone — is reported for holdout too**, for the same
  transparency reason, also with NO gate authority beyond the single
  sign-consistency rule already in force. Holdout PASS/FAIL is decided
  on the full holdout window only.

## 9. Outcome commitments

A FAIL is written up with the same rigor as every prior phase: trade
count, win rate, the IV-RV premium's apparent role, and cost sensitivity
(a stress case at double the assumed spread cost). Nothing here is
amended after data contact. If K1 fails, the project's honest position
is that compression-based forecasting is real (V1/V2) but this specific
options expression of it, at this threshold, does not clear the
options-space alpha bar — a legitimate and reportable outcome, not a
reason to retune the threshold and re-run.
