# Phase V1 — Volatility Compression as Expansion Precursor (pre-registration)

## Hypothesis
Stocks with abnormally LOW recent trading activity (compressed volume 
relative to their own longer-run baseline) exhibit higher REALIZED 
VOLATILITY in the subsequent period than stocks without such compression. 
This is a forecasting/IC test, not an alpha claim — no position is taken, 
no direction is predicted, only magnitude of movement.

## Data and universe
Reuse data/processed/universe_membership.parquet unchanged (deciles 6-8, 
point-in-time, already validated). Reuse data/processed/crsp_combined.parquet 
for DlyRet, DlyPrcVol, DlyCalDt, PERMNO.

## Compression score (locked definition)
For each PERMNO, each trading day t:
- vol20 = mean(DlyPrcVol) over the 20 trading days ending t-1
- vol60 = mean(DlyPrcVol) over the 60 trading days ending t-1
- compression_ratio = vol20 / vol60 (lower = more compressed, i.e. recent 
  activity is below the longer-run baseline)
- Require at least 15 of 20 days in the vol20 window and 45 of 60 days in 
  the vol60 window to have non-null DlyPrcVol; else the score is null for 
  that day (no substitution, no shorter fallback window).
- Cross-sectional decile rank within the in-universe panel each day: 
  decile 1 = most compressed (lowest ratio).

## Forward target (locked definition)
- realized_vol_fwd_10d = annualized standard deviation of DlyRet over the 
  10 trading days starting at t+1 (sqrt(252) scaling). This is the 
  PRIMARY target.
- realized_vol_fwd_5d and realized_vol_fwd_20d computed identically as 
  SECONDARY/diagnostic targets, same status as R1/R2's secondary horizons: 
  reported, no gate authority.

## Test methodology (locked)
Fama-MacBeth style, not a pooled panel regression (pooled panel would 
understate standard errors given cross-sectional and serial correlation):
- Each trading day t, run a cross-sectional regression: 
  realized_vol_fwd_10d ~ compression_decile_rank, across all in-universe 
  stocks with a valid score that day.
- Collect the daily slope coefficient for every day t.
- Test the time series of daily coefficients with a Newey-West adjusted 
  t-test, maxlags = 10 (matching the 10-day forecast horizon, since 
  overlapping windows induce autocorrelation in the coefficient series).

## Gate (binding numbers)
- Dev window: 2015-01-01 to 2021-12-31 (identical split to R1/R2).
- PASS requires: mean daily coefficient is NEGATIVE (lower decile rank / 
  more compression -> higher forward realized vol) AND Newey-West t-stat 
  on the coefficient series has |t| >= 3.0 (higher bar than the 2.0 used 
  for alpha claims, since this is an exploratory forecasting test without 
  the discipline of a cost model attached to it).
- Additional required evidence, not a numeric gate but must be reported: 
  monotonicity check — mean realized_vol_fwd_10d by compression decile, 
  printed as a table, should be roughly monotonic (decile 1 highest, 
  decile 10 lowest) if the mechanism is real, not just a linear artifact.
- Holdout: 2022-01-01 to 2025-12-31, ONE pass, only after a logged dev 
  PASS, same enforcement pattern as R1/R2 (holdout code path must not 
  exist in the same script that touches dev data).

## What happens after a PASS (not built yet, future phase)
Kronos volatility forecasts get evaluated ONLY if this phase passes, and 
ONLY as a horse race against named baselines (trailing realized vol, 
GARCH(1,1)) — Kronos must beat both baselines on the dev window or the 
project defaults to the simplest baseline that passed. Options/OptionMetrics 
data is not pulled until that horse race is won. This section is dormant.

## Outcome commitments
A FAIL is written up with the same rigor as any other phase in this 
project: the decile table, the coefficient time series summary, and a 
stated reason. Nothing here is amended after data contact.
