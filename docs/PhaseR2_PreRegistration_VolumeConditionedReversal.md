# Pre-Registration — Volume-Conditioned Long-Only Small-Cap Reversal
## DriftFire Rerun, Phase R2

**Status:** LOCKED upon git commit. Any change after first data contact
requires a new phase document; this one is never edited.
**Author:** Aarav (design), drafted with agent assistance
**Date committed:** [git timestamp is authoritative]
**Predecessor:** Phase R1 (FAIL, dev, logged 2026-07-11 in results/gate_log.md)

---

## 1. Hypothesis

Phase R1 established that the unconditional bottom-decile 5-day loser
portfolio in NYSE size deciles 6-8 carries NO factor-adjusted alpha even
gross of costs (dev-window gross intercept -3.6%/yr, t = -0.98); its raw
gross return was fully explained by passive ST_Rev, SMB, and market
loadings.

R2 tests the sharper claim from the liquidity-provision literature
(Nagel 2012; Avramov, Chordia & Goyal 2006): the reversal premium
concentrates in losers whose decline was accompanied by abnormal trading
volume — a proxy for forced, non-informational selling. R2 FILTERS the R1
signal; it does not replace it. The claim is again long-only alpha beyond
the published factor set, including ST_Rev.

## 2. Data

- Same frozen CRSP pulls as R1: `data/raw/crsp/*.parquet` (CIZ format).
- Volume measure: **daily dollar volume `DlyPrcVol`** (price x volume),
  chosen over share volume because it is robust to splits within the
  baseline window. Decided before any R2 data contact.
- Factors: daily FF5 + MOM + ST_Rev, `data/processed/factors_daily.parquet`
  (re-pulled daily files, 2026-07-11; monthly mispulls quarantined in
  `data/raw/factors/*.WRONG_*`).

## 3. Universe

Identical to R1 in every respect. `data/processed/universe_membership.parquet`
is reused unchanged: NYSE-breakpoint deciles 6-8, monthly point-in-time,
price >= $5, ordinary common (CIZ mapping), NYSE/AMEX/NASDAQ, sanity bound
800-1,500 names/month (verified PASS in R1).

## 4. Signal (signal layer)

Base signal identical to R1: SIG_t = return over days [t-5, t-1] computed
with shift(1); cross-sectional decile rank within the full universe each
day; bottom decile (rank 1) = largest 5-day losers.

**New volume condition.** Let w = the day within [t-5, t-1] with the most
negative DlyRet for that stock. A rank-1 stock is a long candidate on day t
iff ALL of:

1. DlyRet_w < 0 (the worst day is an actual decline);
2. DlyPrcVol_w / mean(DlyPrcVol over the 20 trading days ending w-1) >= **2.0**;
3. at least **15 of the 20** baseline days have non-null DlyPrcVol
   (data-sufficiency; a stock failing this is ineligible that day — no
   discretionary relaxation on thin days, locked now).

`is_long_candidate_r2 = (R1 rank == 1) AND volume condition.`

Expected candidate pool ~15-30 names/day (R1: ~103/day). A pool wildly
outside this range triggers a data audit before analysis proceeds.

**Zero-candidate days:** the tranche formed that day is 0% invested (cash).
Never filled with second-best candidates. Locked before data contact.

No other features. No short leg. Kronos forecasts, if present, are carried
for the expression layer only.

## 5. Portfolio construction

- Entry: next-day open (signal day t, fill at open of t+1).
- **Primary holding period: 5 trading days** (gate authority), overlapping
  daily tranches, each 1/5 of capital, equal-weighted within tranche.
- **Pre-registered secondary horizons: 10 and 20 trading days.** Decay
  exhibit only, NO gate authority. 1-day and 3-day are dropped: R1
  established they are the most cost-poisoned; retesting them invites
  horizon-shopping.
- Exits calendar-based only. No stops, no recovery exits, no re-entry.

## 6. Costs

- Base 15 bps per side; stress 30 bps per side.
- REALIZED annualized turnover reported with every result table (with
  empty-tranche days, realized turnover is below the 2/H mechanical rate).
- All gated statistics NET of base-case costs.

## 7. Alpha isolation

Identical to R1: daily net portfolio returns in excess of RF regressed on
Mkt-RF, SMB, HML, RMW, CMA, MOM, ST_Rev; Newey-West standard errors,
10 lags. The claim is the INTERCEPT.

## 8. Gate (binding numbers — unchanged from R1; the bar does not move)

- Dev window: 2015-01-01 to 2021-12-31.
- PASS requires, on the 5-day primary, net of base-case costs:
  intercept t-stat >= 2.0 AND positive intercept.
- Holdout: 2022-01-01 to 2025-12-31. ONE pass, only after a logged dev
  PASS; holdout PASS = positive intercept. A holdout failure closes the
  phase as a documented null. No retuning or re-running.
- Evaluations appended to `results/gate_log.md`; backtest module remains
  locked until PASS.

## 9. Pre-registered exhibits (descriptive, no gate authority)

- Daily candidate-count time series: min/max/mean and count of
  zero-candidate days.
- Gross (pre-cost) alpha regression on the same factor set — reported so
  a FAIL can distinguish "no alpha" from "alpha eaten by costs."
- Horizon decay: the 10d and 20d secondary regressions.

## 10. Outcome commitments

Identical to R1 Section 10: a FAIL is written up with full rigor (decay
exhibit, cost sensitivity, loadings, mechanism discussion). Nothing here
is amended post hoc; extensions are new phase documents. The single
threshold 2.0x in Section 4 was fixed before any R2 data contact and is
not tunable; testing other thresholds requires a new phase.

## References

Nagel (2012), Evaporating Liquidity, RFS. Avramov, Chordia & Goyal (2006),
Liquidity and Autocorrelations in Individual Stock Returns, JF. Campbell,
Grossman & Wang (1993), Trading Volume and Serial Correlation in Aggregate
Stock Returns, QJE. McLean & Pontiff (2016), JF. Grinold & Kahn, Active
Portfolio Management.
