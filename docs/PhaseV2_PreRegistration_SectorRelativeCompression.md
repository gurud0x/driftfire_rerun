# Phase V2 — Sector-Relative Volatility Compression (pre-registration)

## Hypothesis
Stocks quiet relative to their OWN SECTOR's current activity level show 
higher forward realized volatility than stocks only quiet relative to 
their own history (V1's finding). Sector-relative compression should 
separate a single stock going quiet from its whole sector cooling 
together.

## Data source and known limitation (stated up front, not discovered 
post-hoc)
- GICS sector comes from data/raw/compustat/ccm_link_gics.csv, joined to 
  CRSP via lpermno, using linktype IN (LC, LU) and linkprim IN (P, C) 
  only, filtered to rows where DlyCalDt falls between linkdt and 
  linkenddt ('E' in linkenddt treated as still-active, i.e. no upper 
  bound).
- Where both ccm_link_gics.csv and compustat_gics_names.csv have a 
  gsector value for the same gvkey, the link file is authoritative (20 
  known exceptions out of 10,615 shared gvkeys, logged not resolved).
- LIMITATION, ACCEPTED KNOWINGLY: gsector in both source files is static 
  (backfilled current classification, 0% of gvkeys show a historical 
  change in either pull). This introduces mild look-ahead: a stock's 
  sector membership reflects its most recent classification, not its 
  true historical one at the time. True point-in-time GICS (co_hgic) was 
  investigated and is not accessible through this WRDS subscription's 
  query interface. This limitation is disclosed in any write-up of this 
  phase, pass or fail, and is not corrected after the fact.

## Universe and coverage
- Reuse data/processed/universe_membership.parquet unchanged.
- Expected coverage: ~98.4% of universe PERMNOs link via CCM; further 
  attrition from GICS nulls (~13.4% null rate in the link file) means 
  effective coverage will be lower — the actual number is computed and 
  reported by the ingest script, not assumed.
- Stock-days with no valid sector match are excluded from V2 entirely 
  (not assigned a synthetic sector, not carried forward from a stale 
  value).

## Sector definition (locked)
- gsector as pulled (11 standard GICS sectors), used directly — no 
  further collapsing needed, since GICS sectors are already a clean, 
  standard broad-bucket scheme (unlike the SIC alternative considered 
  earlier).
- Sector-day sufficiency requirement: each sector-day used in the signal 
  must have at least 20 in-universe stocks with valid compression_ratio 
  and gsector that day; sector-days below this are excluded and the 
  exclusion count is reported.

## Signal (locked)
- Reuse V1's compression_ratio (vol20/vol60 on DlyPrcVol) exactly as 
  computed in src/10_compression_signal_v1.py — not recomputed.
- sector_relative_compression = stock's own compression_ratio MINUS the 
  same-day median compression_ratio across other in-universe stocks in 
  the same gsector.
- Cross-sectional decile rank of sector_relative_compression within the 
  full universe each day (comparable structure to V1).

## Forward target (identical to V1)
- realized_vol_fwd_10d primary; fwd_5d and fwd_20d secondary/diagnostic.

## Test methodology (identical to V1)
- Fama-MacBeth: daily cross-sectional OLS of realized_vol_fwd_10d on 
  sector_relative_compression decile.
- Newey-West t-test, maxlags=10, on the daily coefficient series.

## Gate (locked)
- Dev window: 2015-01-01 to 2021-12-31.
- PASS requires: mean daily coefficient negative AND |Newey-West t-stat| 
  >= 3.0 (identical bar to V1).
- Required reported evidence: 10-decile monotonicity table, AND explicit 
  side-by-side comparison of V2's dev t-stat against V1's logged dev 
  t-stat (-6.88), printed in the same output.
- Holdout: 2022-01-01 to 2025-12-31, ONE pass, only after logged dev 
  PASS, same enforcement pattern (no holdout code path in the dev 
  script).

## Outcome commitments
A FAIL is written up with the same rigor as any phase: decile table, 
coefficient summary, comparison to V1, and the static-GICS limitation 
restated as context. Nothing here is amended after data contact.
