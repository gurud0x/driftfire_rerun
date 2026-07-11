# Pre-Registration — Long-Only Small-Cap Short-Term Reversal
## DriftFire Rerun, Phase R1

**Status:** LOCKED upon git commit. Any change after first data contact
requires a new phase document; this one is never edited.
**Author:** Aarav
**Date committed:** [git timestamp is authoritative]

---

## 1. Hypothesis

Stocks in the small-cap segment (NYSE size deciles 6–8) that experienced the
largest negative returns over the prior 5 trading days earn positive
abnormal returns over the subsequent 5 trading days, as compensation for
providing liquidity where arbitrage capital is capacity-constrained
(Nagel 2012; Avramov, Chordia & Goyal 2006). The claim is tested long-only
and must survive neutralization against standard factors including the
short-term reversal factor itself — i.e., the portfolio must exhibit alpha
beyond passive exposure to the published ST_Rev factor.

## 2. Data

- CRSP daily (CIZ format), 2015-01-01 to 2025-12-31:
  `raw/crsp/crsp_daily.parquet`, `crsp_names.parquet`.
- Factors: Fama-French 5 (daily), Momentum (daily), Short-Term Reversal
  (daily) from Ken French library: `raw/factors/*.csv`.
- Delisting returns from CRSP delisting fields are incorporated into all
  return series. A universe-year with zero delistings is treated as a
  pipeline error, not a result.

## 3. Universe (point-in-time, monthly refresh)

- NYSE-breakpoint market-cap deciles 6, 7, 8 (decile 1 = largest).
- Month-end market cap = |price| × shares outstanding; decile assignment
  from month t−1 applies throughout month t.
- Filters: price >= $5 at assignment; share codes 10, 11; exchanges
  NYSE / AMEX / NASDAQ; ordinary common shares only.
- Sanity bound: 800–1,500 names per month. Sustained deviation > 30% from
  trailing 12-month mean triggers a data audit before any analysis.

## 4. Signal (signal layer)

- SIG_t = return over days [t−5, t−1], computed with shift(1); the return
  on day t itself is never included.
- Cross-sectional decile rank of SIG within the universe each day.
- Position rule: LONG ONLY, bottom SIG decile (largest 5-day losers).
- No other features. No short leg. Kronos volatility forecasts, if present
  in `processed/kronos_forecasts.parquet`, are carried as columns for the
  expression layer only and have no role in selection or ranking.

## 5. Portfolio construction

- Entry: next-day open (signal on day t, fill at open of t+1).
- **Primary holding period: 5 trading days**, implemented as 5 overlapping
  daily tranches, each 1/5 of capital, equal-weighted within tranche.
- **Pre-registered secondary horizons: 1, 3, and 10 trading days.**
  These are computed and reported as a horizon-decay exhibit. They carry
  NO gate authority. The gate binds on the 5-day primary only. This is
  declared here, in advance, to preclude horizon selection after results.
- Exits are calendar-based only. No stop-losses, no recovery-based exits,
  no discretionary re-entry.

## 6. Costs

- Base case: 15 bps per side. Stress case: 30 bps per side.
- Annualized turnover reported alongside every result table.
- All gated statistics are NET of base-case costs; stress case reported.

## 7. Alpha isolation

- Daily portfolio returns (net) regressed on: Mkt-RF, SMB, HML, RMW, CMA,
  MOM, ST_Rev. Newey-West standard errors, 10 lags.
- The claim is the INTERCEPT. Raw returns, Sharpe ratios, and equity curves
  are descriptive only.

## 8. Gate (binding numbers)

- Dev window: 2015-01-01 to 2021-12-31.
- PASS requires, on the 5-day primary, net of base-case costs:
  intercept t-stat >= 2.0 AND positive intercept.
- Holdout window: 2022-01-01 to 2025-12-31. ONE pass, run only after a
  dev PASS is logged. Holdout PASS requires positive intercept
  (sign consistency). No retuning, re-splitting, or re-running under any
  circumstance; a holdout failure closes the phase as a documented null.
- All gate evaluations are appended to `results/gate_log.md` by
  `src/gate_check.py`; the backtest module remains locked until PASS.

## 9. Expression layer (DORMANT — separate pre-commitment)

Activates only after a full gate PASS (dev + holdout). Until then no
OptionMetrics data is pulled and no options code runs.

- Per gated long signal: express as 1 ATM call (30–45 DTE) iff
  Kronos E[RV21] / IV30 >= **[FILL IN — locked before any OptionMetrics
  contact]**; otherwise express as shares.
- Option costs: mid minus half the quoted spread.
- Kill-switch: if Kronos E[RV21] does not beat the trailing-vol baseline
  on dev-window MAE, the expression layer defaults to shares-only and the
  threshold field is voided.
- Evaluation metric: net Sharpe of expressed portfolio vs shares-only,
  descriptive comparison; no new alpha claim is made at this layer.

## 10. Outcome commitments

- A dev-window FAIL is written up with the same rigor as a pass: horizon
  decay exhibit, cost sensitivity, factor loadings, and a stated reason
  the mechanism did not survive in this segment.
- Nothing in this document is amended post hoc. Extensions (new universe,
  new lookback, PEAD event conditioning, OptionMetrics signal features)
  are new phase documents.

## References

Nagel (2012), Evaporating Liquidity, RFS. Avramov, Chordia & Goyal (2006),
Liquidity and Autocorrelations in Individual Stock Returns, JF. McLean &
Pontiff (2016), Does Academic Research Destroy Stock Return Predictability?,
JF. Lou, Polk & Skouras (2019), A Tug of War: Overnight vs Intraday Expected
Returns, JFE. Grinold & Kahn, Active Portfolio Management.
