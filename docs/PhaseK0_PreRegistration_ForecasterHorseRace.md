# Phase K0 — Volatility Forecaster Horse Race (pre-registration)

**Status:** LOCKED upon git commit. Never edited after data contact.
**Purpose:** determine the sanctioned E[RV] forecaster BEFORE any
OptionMetrics contact, per the locked sequencing in Phase V1's
"What happens after a PASS" section. This IS the horse race that section
requires; its winner is what unlocks Phase K1 (options work).

## Candidates (dev window 2015-01-01 to 2021-12-31 only)

All three forecast the same target: realized_vol_fwd_10d as defined in
Phase V1 (annualized sample std of DlyRet over trading days t+1..t+10).
All forecasts are made as of day t using information available at the
close of day t.

1. **Trailing realized vol (TRAIL20).** Same-stock annualized std of
   DlyRet over the 20 trading days ending at day t (inclusive),
   rolling(20, min_periods=15), sqrt(252) scaling. Pure past-data
   forecaster, no calibration.

2. **GARCH(1,1) (GARCH11).** Fit PER STOCK — not pooled — on that
   stock's dev-window daily returns, because unconditional volatility
   levels differ by an order of magnitude across names and a pooled fit
   would forecast a meaningless cross-sectional average. Constant-mean
   GARCH(1,1), parameters estimated ONCE per stock on the full dev
   window. Forecast at day t: h_{t+k} iterated analytically from the
   fitted conditional variance, E[RV] = sqrt(mean of h_{t+1..t+10}) *
   sqrt(252). DISCLOSED LIMITATION: parameters are in-sample for the dev
   window (fit on 2015-2021, forecasts evaluated within 2015-2021).
   This is accepted because candidate 3 is calibrated on the same dev
   window, keeping the race internally consistent; the race is a
   dev-window model selection, not an out-of-sample claim. Stocks with
   fewer than 250 dev-window return observations are not fitted (their
   stock-days simply lack a GARCH forecast and fall out of the common
   evaluation sample).

3. **Compression-decile forecaster (COMPDEC).** V1's compression decile
   as a categorical forecaster: each decile's forecast value is the
   dev-window average realized_vol_fwd_10d of that decile. This
   operationalizes the "simplest baseline that passed" language from
   V1's Section on post-PASS sequencing. In-sample dev calibration,
   same status as GARCH's.

## Evaluation (locked)

- Sample: in-universe stock-days from data/processed/
  compression_signal_v1.parquet, dev window only, restricted to the
  COMMON sample where the target and ALL THREE forecasts are non-null
  (so every forecaster is scored on identical stock-days).
- Metric: MAE = mean |forecast - realized_vol_fwd_10d|, in annualized
  vol units.
- Decision rule: lowest MAE wins and becomes the sanctioned E[RV]
  forecaster. Ties (MAE equal to 4 decimal places) broken toward
  simplicity: TRAIL20 > COMPDEC > GARCH11.
- Kronos is NOT built in this phase. If a Kronos environment is stood
  up later, it competes one-on-one against this phase's winner (via the
  data/processed/kronos_forecasts.parquet contract), not against all
  three afresh.

## Outcome commitments

- The full MAE table is appended to results/gate_log.md with the winner
  and reasoning, whatever the ordering turns out to be.
- The winner unlocks K1 (OptionMetrics contact) per V1's sequencing.
  Nothing else about K1 is decided here.
- Nothing in this document is amended after the race runs.
