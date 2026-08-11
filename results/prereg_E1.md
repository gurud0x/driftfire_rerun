# Phase E1 Pre-Registration — EMA-Ribbon Residency

**Status: LOCKED — 2026-07-30, AMENDED — 2026-07-30 (condition 1a only; see
§13 item 11).** Bars confirmed by the project owner on 2026-07-30 after panel
verification. No E1 outcome (forward return, barrier probability, regression,
or arm comparison) had been computed at the time of the original lock. The 1a
amendment was written after a first `src/46` run found condition 1a
structurally inestimable — before any 1a coefficient, sign, or magnitude
existed, and before Tests 1b/2/3 had been run — so it is a fix to an
inestimable condition, not a reaction to an unfavorable result. See §13 item
11 for the full record.
**Phase type:** exploratory (E-series), DEV window only.
**Scripts:** signal construction `src/45_E1_signal_construction.py` (built,
run, QA'd via `src/45b_E1_signal_qa.py`, unchanged by the 1a amendment); tests
`src/46_E1_tests.py` (built; computes the amended `continuous_residency_depth`
term itself, without modifying `src/45` or its locked output).
**Signal artifact:** `data/processed/e1_signal_panel.parquet`
(151,752 rows × 54 columns, 3,032 PERMNOs, 2015-02-17 → 2021-12-31, verified on
disk 2026-07-30).

---

## 1. Purpose

Test whether volume compression combined with price residency near the EMA8–EMA21
ribbon predicts:

1. **Magnitude** — greater subsequent absolute stock movement;
2. **Direction** — the direction of the subsequent breakout;
3. **Trigger** — faster upside movement after a bullish volume-expansion trigger.

Each is a separate test with its own bar. The combination rule in §9 decides the
phase verdict.

---

## 2. Universe and sample

Identical to R1 / R2 / V1 / V2 / K0 / K1. No new universe is constructed for E1.

- Source: `data/processed/universe_membership.parquet`, built by
  `src/02_universe.py`.
- NYSE-breakpoint size **deciles 6–8**, monthly point-in-time. Breakpoints from
  NYSE ordinary common only (Fama-French convention). Month *t−1* data applies to
  month *t*.
- Ordinary common filter: `SecurityType='EQTY'` & `SecuritySubType='COM'` &
  `ShareType='NS'` & `IssuerType='CORP'` & `ShrAdrFlg='N'`.
- Exchange: `PrimaryExch in ('N','A','Q')`. Price: `|DlyClose| >= $5`.
- Universe size: 803,000 PERMNO-months total, 136,014 in-universe (16.94%).

Price data: `data/processed/crsp_combined.parquet` (CRSP CIZ / Flat File 2.0;
`DlyCalDt`, `PERMNO`). Duplicate PERMNO-days deduped `keep='first'` after
verifying no conflicting OHLC values, matching every prior signal script.

---

## 3. Window: DEV only

| | |
|---|---|
| **DEV** | 2015-01-01 → 2021-12-31 (1,763 trading sessions) |
| HOLDOUT | 2022-01-01 → 2025-12-31 — **not used in E1** |

**E1 runs on DEV only. No E1 script contains a holdout code path.**
`src/45` hard-filters its output to DEV before saving.

### 3.1 Holdout disclosure — mandatory, recorded now

The 2022-01-01 → 2025-12-31 window **has already served as the holdout for five
phases: R1, R2, V1, V2, and K1.** Per `CLAUDE.md`, that holdout is spent.

E1 makes **no holdout claim of its own in this phase.** Every E1 result is a
DEV-window result and must be described as such.

If a future E1 holdout test is ever proposed, it must be disclosed at that time as
a **sixth look at the same calendar period — not a fresh out-of-sample test.** The
statistical guarantee normally attached to a holdout does not survive repeated
reuse, and the reuse count is already five before E1 begins. This limitation is
written down here, before it is needed, so that it cannot later be presented as a
detail discovered after the fact.

---

## 4. Warmup and usable date range

### 4.1 Warmup chain, in per-PERMNO sessions

Session 1 = a PERMNO's first CRSP daily observation. Each feature is `shift(1)`
aligned, so a feature "usable at session *n*" is computed from information through
session *n−1*. EMA warmup is enforced explicitly in `src/45` via
`min_periods=span` on the `ewm` calls (early unconverged EMA values are null, not
emitted).

| step | requirement | usable at |
|---|---|---|
| EMA8 | 8 closes | session 9 |
| **EMA21** | **21 closes** | **session 22** |
| **ATR14** | 14 true ranges (TR starts session 2) | **session 16** |
| `normalized_body_distance` | EMA21 **and** ATR14 | **session 22** |
| residency window *W* | *W* consecutive resident flags | **session 22 + W − 1** |

Episode definition uses **W = 10 for every threshold variant** (§6), so the
episode warmup is **session 31** — confirmed by the panel's first row landing on
**2015-02-17**, which is exactly session 31 of the CRSP calendar for names present
from 2015-01-02. Residency *scores* extend to W = 25, usable at **session 46**
(**2015-03-10** for names present from the start).

### 4.2 LOCKED: common evaluation start — enforced per PERMNO, mechanically

**An episode row enters the evaluation sample (primary spec and every grid cell
alike) only if `residency_25` is non-null on that row.** `residency_25` has the
longest warmup of any feature in the panel (session 46), so this single condition
guarantees every feature used anywhere in E1 is fully warmed on every evaluated
episode, per PERMNO — not via a global date cutoff, which would silently admit
under-warmed features for later-listing names.

For names present from 2015-01-02 this is equivalent to an evaluation start of
**2015-03-10**. Rationale for a common start: grid cells fitted on different
samples would confound parameter sensitivity with sample composition. The cost
(episodes between 2015-02-17 and 2015-03-09 for early-listed names) is accepted.
`src/46` must report the count of episodes excluded by this rule.

### 4.3 LOCKED: forward-window truncation at the DEV end

Outcomes requiring a complete forward window are computed only where the full
window fits inside DEV. Episodes with an incomplete window are **dropped for that
outcome and counted**, never truncated or padded. Last eligible episode-start
dates, from the CRSP calendar:

| forward horizon | last eligible episode-start |
|---|---|
| 5 sessions | 2021-12-23 |
| 10 sessions | 2021-12-16 |
| 20 sessions | 2021-12-02 |

`src/46` must re-derive every date in §4 from the CRSP calendar at runtime and
assert agreement with the values above, per the no-transcribed-numbers rule. The
dates here are documentation, not the source of truth.

---

## 5. Locked primary specification

**Not adjustable after seeing any result.**

| parameter | locked value |
|---|---|
| Residency window | **10 sessions** |
| Residency condition | **≥ 6 of 10** sessions with `normalized_body_distance <= 0.25` ATR |
| Volume-expansion trigger | **2.0×** trailing 20-day average volume |
| Ribbon | EMA8 / EMA21 on close, `shift(1)` aligned, `min_periods` enforced |
| ATR | ATR14, `shift(1)` aligned, `min_periods=14` |
| Episode definition | first day of each run of the residency condition (non-overlapping by run) |
| Signal construction | exactly as built in `src/45_E1_signal_construction.py` |
| Compression score | `compression_ratio` from V1, joined as-is (already pre-lagged) |
| Residency score | `residency_10` (fraction of the 10-session window resident, 0–1) |
| Entry convention | **next day's open** |

The primary episode set is the panel's `episode_start` column (threshold 0.25,
W = 10): **69,126 episodes** before the §4.2 common-start filter and Test-specific
drops (count verified from the artifact at write-time; `src/46` re-derives it at
runtime).

### 5.1 Additional locked definitions

- **EMA8 slope** := `ema8 − ema8_lag1` — the **1-session change** in the
  `shift(1)`-aligned EMA8, exactly as built and verified in the panel
  (`ema8_slope` column; equality `ema8_slope == ema8 − ema8_lag1` checked against
  the artifact). *Disclosure: an earlier draft of this document proposed a
  5-session change. The as-built 1-session definition supersedes it and is locked
  here; the panel carries `ema8` and `ema8_lag1` explicitly so the definition is
  mechanically auditable. Slope `> 0` = rising.*
- **Episode high** (Test 3 trigger reference) := the running maximum of `DlyHigh`
  from the episode-start day through session *s−1*, evaluated for a trigger on
  session *s* — computed in `src/46` by rejoining the daily panel. *Naming
  caution: the panel column `episode_high` holds only the episode-start day's own
  high (the day-1 seed of the running max), not the running max itself.*
- **Trailing 20-day average volume** := `DlyVol` rolling mean over 20 sessions,
  `min_periods=20`, `shift(1)` aligned (computed in `src/46`).
- **Upper 25% of daily range** := `(DlyClose − DlyLow) / (DlyHigh − DlyLow) >=
  0.75` on the trigger day; if `DlyHigh == DlyLow` the trigger does not fire.
  (Trigger-day close/range are same-day information; entry is next day's open, so
  no lookahead enters the entry price.)
- **Prior realized vol** (control) := annualized stdev of `DlyRet` over the 20
  sessions ending *t−1*.
- **Market-vol regime** (control, Test 1) := cross-sectional median of prior
  realized vol across the in-universe panel on the episode-start date.
- **Market-direction regime** (reporting split, Test 2 — see §7 Test 2):
  trailing 10-session compounded return of **`vwretd`** (CRSP value-weighted
  market index, already present in `crsp_combined.parquet`; verified zero nulls
  over DEV), `shift(1)` aligned (window ends at *t−1*). Buckets, locked as
  declared assumptions: **mkt-up** if ≥ +1.0%, **mkt-down** if ≤ −1.0%,
  **mkt-flat** otherwise. `vwretd` is chosen over `sprtrn` because the E1
  universe is deciles 6–8 (mid-caps): the S&P 500 is a large-cap subset, while
  `vwretd` covers the full tape including the sample's own size range. At these
  thresholds the DEV split is roughly 46% / 21% / 33% up/down/flat (verified at
  write-time; re-derived at runtime).

### 5.1a `continuous_residency_depth` — AMENDED definition, locked 2026-07-30 (§13 item 11)

`continuous_residency_depth` := mean of `normalized_body_distance` over the 10
trading sessions ending at, and including, the episode-start day, computed in
`src/46` by rejoining `crsp_combined.parquet` and recomputing the daily
ribbon/ATR/body-distance primitives exactly as `src/45` defines them (verified
by provenance check against the panel). Because `normalized_body_distance`
itself is already `shift(1)`-aligned (built from information through
session *t−1*), averaging 10 already-lagged daily values introduces no further
lookahead.

**Lower value = tighter/deeper residency** (price stayed closer to the ribbon
on average over the window). This is the opposite direction from
`residency_10` (§5, where higher = more resident), and is the reason this
measure's sign is not interchangeable with `residency_10`'s in a regression —
see §7 Test 1 for the resulting sign convention.

### 5.2 Missing-control policy

Episodes with a null control (`gsector` 1.41% of panel rows from CCM link
attrition; `DlyCap` / `DlyVol` 0.02%) are **dropped from the affected model and
counted, never imputed** — consistent with the project-wide drop policy.

---

## 6. Declared sensitivity grid — exploratory only, restated to match the built artifact

The grid has three axes. **The draft version of this document declared a full
5 × 3 × 3 = 45-cell factorial; the built panel does not support that factorial,
and the grid is narrowed here, before any outcome is computed, to what the
artifact actually contains:**

| axis | values | what varies | available as |
|---|---|---|---|
| **T** — episode threshold | {0, 0.25, 0.5} ATR | the episode *definition* (W = 10 in all cases) | `episode_start_thr0 / _thr025 / _thr05` |
| **W** — residency-score window | {5, 10, 15, 20, 25} | the residency *score* used in Test 1's model, on the primary (thr 0.25) episode set | `residency_5 … residency_25` |
| **V** — volume multiplier | {1.5, 2.0, 3.0}× | Test 3's trigger | computed in `src/46` |

Reported cells: Test 1 across W (5 cells) and across T (3 cells); Test 2 across
T (3 cells); Test 3 across V × T (9 cells). **Episode redefinition at residency
windows other than 10 (e.g. W = 25 episodes) is not constructible from this panel
and is not part of E1.** If a window-variant episode definition is ever wanted,
it is a future, separately pre-registered phase.

**Only the primary spec — (W = 10, 0.25 ATR, 2.0×) — determines the verdict.**
Every sensitivity cell is reported for transparency and cannot change the
verdict, regardless of outcome. **Any notably different result among the other
grid cells is reported as a candidate for a future separate pre-registered phase
— never as supporting evidence for E1 itself.** Nothing in the grid may be
promoted to primary status in this phase. The grid carries no inferential weight
and no multiplicity correction is applied to it, because it decides nothing.

---

## 7. Tests

Standard-error convention, chosen because episodes overlap in calendar time
across names and forward windows overlap within a name:

- **Cross-sectional coefficient tests:** Fama-MacBeth by episode-start date,
  Newey-West on the coefficient time series, `maxlags = 20`.
- **Pooled proportion / rate tests:** two-proportion z-test, standard errors
  clustered by date.
- **Time-series mean returns (Test 3):** Newey-West, `maxlags = 10`.

### Test 1 — MAGNITUDE

**Outcomes per episode-start:** 5-, 10-, 20-day absolute return; 5-, 10-, 20-day
forward realized volatility; MFE and MAE; time to leave the EMA ribbon; +1 ATR vs
−1 ATR first; +2 ATR vs −2 ATR first; probability of a 3% move, a 5% move, and a
2-ATR move.

**Model (AMENDED 2026-07-30, §13 item 11):** regress each magnitude outcome on
`compression_ratio`, `ribbon_width`, `continuous_residency_depth` (§5.1a), and
the **`compression_ratio × continuous_residency_depth` interaction**,
controlling for prior realized vol, market cap (`DlyCap`), price, dollar
volume, sector (`gsector` fixed effects), recent absolute return, longer-term
trend, and market-vol regime. `residency_10` is **not** a separate regressor
here: on the primary episode set it is constant at exactly 0.6 by
construction (§13 item 11), perfectly collinear with the intercept.
`continuous_residency_depth` replaces it, both as the model's residency main
effect and inside the interaction.

**LOCKED bar — both required:**

- **1a.** Interaction term (`compression_ratio × continuous_residency_depth`)
  on the 10-day absolute-return specification (the designated primary
  outcome): **|t| ≥ 3.0**, **expected sign NEGATIVE**, stated here before any
  coefficient was computed under this definition. Reasoning: the
  pre-amendment interaction used `residency_10` (higher = more resident) and
  was hypothesized positive; `continuous_residency_depth` is coded in the
  opposite direction (lower = more resident), so replacing one with the
  other while holding `compression_ratio`'s coding fixed flips the
  hypothesized sign of the product from positive to negative. *(3.0 inherited
  from V1/V2, the project's prior cross-sectional forecasting phases.)*
- **1b.** P(|move| > 2 ATR within 10 days) for episodes vs matched controls:
  margin **≥ +3.0 percentage points**, **|z| ≥ 3.0**. Matched controls =
  non-episode in-universe PERMNO-days matched on date, size decile, `gsector`,
  and prior-realized-vol quintile.

*The 1b margin was relaxed from the draft's 5.0 pp to 3.0 pp at lock:
the stock-level signal is a smaller precursor to a larger options-based trade,
not the primary edge source.*

### Test 2 — DIRECTION

**Episode classification (locked):**

| state | condition |
|---|---|
| bullish | `ema8 > ema21` **and** `ema8_slope > 0` |
| bearish | `ema8 < ema21` **and** `ema8_slope < 0` |
| neutral | everything else |

Report upside-first and downside-first barrier probabilities per state.

**LOCKED bar — both required (symmetric):**

- **2a.** P(+2 ATR first | bullish) − P(+2 ATR first | neutral)
  **≥ +3.0 pp**, **|z| ≥ 3.0**.
- **2b.** P(−2 ATR first | bearish) − P(−2 ATR first | neutral)
  **≥ +3.0 pp**, **|z| ≥ 3.0**, computed pooled across all regimes.
- **2b-flat.** The **same** contrast, computed **within the mkt-flat bucket
  alone**: P(−2 ATR first | bearish, mkt-flat) − P(−2 ATR first | neutral,
  mkt-flat) **≥ +1.5 pp** (half the pooled bar), **|z| ≥ 2.0**.

*Both sides are required: a bullish-only result over 2015–2021 is consistent with
generic upward drift and would not demonstrate directional information in the
ribbon state. Margins relaxed from 5.0 pp to 3.0 pp at lock, same reasoning as
1b.*

**2b-flat is load-bearing, not diagnostic.** If 2b-flat fails, **2b fails
regardless of the pooled number.** Rationale: cross-sectional correlations rise
during market-wide selloffs, so a bearish barrier result concentrated in the
mkt-down bucket is consistent with a market-wide effect rather than
stock-specific information in the ribbon state. Requiring the contrast to survive
in the flat-market bucket — where there is no market-wide directional push — is
the discriminating condition. A bearish result that appears only when the whole
market is falling is not evidence that the ribbon signal adds anything.

**Regime-split reporting (no gate authority, except 2b-flat above):** Test 2's
barrier probabilities are additionally reported broken out by the
market-direction regime of §5.1 (**mkt-up / mkt-down / mkt-flat**, trailing
10-session `vwretd`, ±1.0% buckets). Apart from the load-bearing 2b-flat
condition, the regime split is diagnostic, showing whether any directional result
is market-backdrop-dependent. The mkt-up and mkt-down splits carry no verdict
authority for either 2a or 2b, and no regime split carries authority for 2a.

### Test 3 — BREAKOUT TRIGGER

**Primary trigger (bullish episodes only):** (1) close above the high of the
active episode (running max per §5.1); (2) daily volume ≥ **2.0×** trailing
20-day average; (3) close in the upper 25% of the daily range. **Entry at next
day's open.** Report 5- and 10-day forward return, MFE, MAE, and P(+2 ATR before
−1 ATR).

**Four arms:**

| arm | definition |
|---|---|
| A | ordinary breakout, no compression or residency filter |
| B | compression breakout, no residency filter |
| C | residency breakout, no compression filter |
| D | full setup: compression + residency + breakout |

**LOCKED bar — all three required:**

- **3a.** Arm D mean 10-day forward return: positive, **NW t ≥ 3.0**
  (`maxlags = 10`).
- **3b.** Arm D mean 10-day forward return remains **positive after 30 bps
  round-trip cost**. **The 30 bps figure is a placeholder** (the project's stress
  assumption: 15 bps/side base, 30 bps stress); it is retained for the verdict
  as written, and flagged so that any future cost calibration for this universe
  is a disclosed revision, not a silent one.
- **3c.** Arm D beats arm A on mean 10-day forward return by **≥ 50 bps**,
  difference-in-means **|t| ≥ 2.0**. *(Level test at 3.0; difference test at 2.0
  because a difference of two noisy means is noisier than either level. Costs
  cancel in D − A, which is why 3b is stated against arm D's own level.)*

---

## 8. Summary of locked bars

| test | condition | locked bar |
|---|---|---|
| 1a | `compression_ratio × continuous_residency_depth`, 10-day abs return (AMENDED, §13 item 11) | \|t\| ≥ 3.0, **negative** |
| 1b | P(\|move\| > 2 ATR, 10d) vs matched controls | ≥ +3.0 pp, \|z\| ≥ 3.0 |
| 2a | P(+2ATR first \| bullish) − neutral | ≥ +3.0 pp, \|z\| ≥ 3.0 |
| 2b | P(−2ATR first \| bearish) − neutral, pooled | ≥ +3.0 pp, \|z\| ≥ 3.0 |
| 2b-flat | same contrast **within mkt-flat only** (load-bearing) | ≥ +1.5 pp, \|z\| ≥ 2.0 |
| 3a | arm D mean 10d return | positive, NW t ≥ 3.0 |
| 3b | arm D mean 10d return net of 30 bps (placeholder) | > 0 |
| 3c | arm D − arm A mean 10d return | ≥ +50 bps, \|t\| ≥ 2.0 |

A test **passes** only if **all** of its listed conditions hold. Test 2 therefore
requires 2a **and** 2b **and** 2b-flat.

---

## 9. Combination rule

- **PASS** — all three tests meet their bars.
- **NEAR-PASS** — at least two tests meet their bars, **and** the third is
  *directionally consistent at the looser threshold*: every primary quantity in
  that test carries the hypothesized sign, **and** each significance condition
  reaches **|t| or |z| ≥ 2.0**, **and** each margin condition reaches **at least
  50% of the stated margin** (≥ 1.5 pp where 3.0 pp is required; ≥ 25 bps where
  50 bps is required). For **2b-flat**, whose bar is already at the NEAR-PASS
  significance floor, the relaxed form is **≥ +0.75 pp with |z| ≥ 2.0** — the
  significance requirement does not relax further.
- **FAIL** — anything else.

A wrong-signed result is never a NEAR-PASS, at any significance level.

**Diagnostic requirement on any miss:** whenever a test fails its bar (including
the NEAR-PASS case), `src/46` must report which specific control, subgroup,
sector, or regime most explains the miss — e.g. the sector fixed effect,
size-decile cell, vol-regime bucket, or market-direction regime whose removal or
isolation moves the failed quantity the most — not just the pass/fail flag. This
is reporting, not re-testing: the verdict is computed on the full locked
specification only.

**Only PASS or NEAR-PASS makes E1 a candidate for a tradeable strategy.**
Anything else is a research finding, not a strategy, and is to be written up as
such — regardless of how interesting any single test looks in isolation. A
single strongly-passing test alongside two failures is a FAIL and is reported as
a FAIL.

---

## 10. Signal artifact: verification record and bugfix disclosure

### 10.1 Bugfix disclosure — episode-count swing

The first build of `src/45` contained a boolean-dtype bug in the episode-start
crossing logic (bitwise NOT applied to an object-typed shifted series), which
generated spurious repeated episode starts. After the fix, the panel row count
moved from **1,395,853 to 151,752 rows** — a ~89% reduction. **Source of
record: `results/45_bugfix_note.md`.** The fix was independently verified with 5
fresh spot-checks against hand-counted prior-10-session windows (all consistent;
`residency_10_hits ≥ 6` on every `episode_start` row — re-verified against the
artifact at lock time, min = 6.0). The QA script (`src/45b`) had its own
independent indexing bug, also fixed. This disclosure exists so the magnitude of
the correction is on the record inside the pre-registration itself.

### 10.2 Panel contents (verified on disk at lock time)

`data/processed/e1_signal_panel.parquet`: 151,752 rows × 54 columns (union of
episode starts across the three threshold variants, `episode_start_any`), 3,032
PERMNOs, 2015-02-17 → 2021-12-31, of which **69,126** are primary
(`episode_start`, thr 0.25) episodes. Columns include: `PERMNO`, `DlyCalDt`,
full OHLC (`DlyOpen/High/Low/Close`), `DlyVol`, `DlyCap`, `gsector`,
`daily_range`, `episode_high`, `ema8`, `ema8_lag1`, `ema8_slope`, `ema21`,
`atr14`, `ribbon_low/high`, `body_low/high`, `body_to_ribbon_distance`,
`normalized_body_distance`, `close_to_ribbon_distance(_norm)`, `ribbon_width`,
`residency_5/10/15/20/25`, `residency_10_hits`, threshold-grid variants
(`is_resident_thr*`, `residency_10_thr*`, `primary_hits_thr*`,
`primary_residency_condition_thr*`, `episode_start_thr*`, `episode_seq_thr*`),
`episode_start_any`, and `compression_ratio`.

All nine gaps flagged in the pre-lock feasibility review are resolved: volume ✔,
OHLC ✔, `DlyCap`/`gsector` ✔ (sector joined from the V2 CCM-linked panel),
`ema8_slope` ✔ (with `ema8_lag1` for auditability), `residency_15/25` ✔,
threshold variants ✔, EMA warmup enforced via `min_periods` ✔, panel exists and
is DEV-filtered ✔. Forward paths remain absent **by design** — `src/46` rejoins
`crsp_combined.parquet` for all forward outcomes.

**Lookahead review of the rebuilt `src/45` (re-done at lock):** `ema8`/`ema21`
are `ewm(min_periods=span).mean().shift(1)`; `ema8_lag1` is a further shift (so
`ema8_slope` compares information through *t−1* vs *t−2*); `atr14` is
`rolling(14).mean().shift(1)`; body geometry uses lagged open/close. All feature
inputs at row *t* are information through *t−1*. The same-day columns
(`DlyOpen/High/Low/Close`, `DlyVol`, `daily_range`, `episode_high`) are carried
for `src/46`'s outcome and trigger construction and **must not** be used as
features dated *t*; §5.1 defines their permitted uses. No lookahead found.

### 10.3 Run hygiene

Run logs archived to `results/logs/`. No stale temporary scripts;
`e1_signal_panel.parquet` is the sole E1 parquet in `data/processed/` and is
gitignored per project policy.

---

## 11. Provenance and prior-work boundary

Sector and market cap for E1 come from the **current** CCM-linked Compustat and
CRSP data only:

- `gsector` via the V2 sector panel (`sector_compression_signal_v2.parquet`),
  itself built from `data/raw/compustat/ccm_link_gics.csv`, date-windowed on
  `LINKDT`/`LINKENDDT`, `LINKTYPE ∈ {LC, LU}`, `LINKPRIM ∈ {P, C}`, P-over-C then
  gvkey tiebreak (`src/17` logic). Known attrition ~2% of PERMNO-days; 1.41% of
  E1 panel rows carry null `gsector` and are dropped-and-counted where sector is
  required (§5.2).
- `DlyCap` native in `crsp_combined.parquet` (units: \$ thousands); 0.02% null on
  the panel.

A separate earlier project (repo `driftfire`, yfinance data, 2018–2024) built a
`sector_map.csv` and a `vol_ratio` metric. **That project is not this project and
none of its data, universe, or date range is used here.** No `sector_map.csv`
exists in this repo. Method observation only: SPDR-sector-ETF assignment maps to
11 buckets via *current* membership — a point-in-time violation — whereas the CCM
link is date-windowed; the ETF approach is not a substitute and is not used.
*(The `vol_ratio` at `src/07_signal_r2.py:68` is R2's dollar-volume ratio — an
unrelated name collision.)*

---

## 12. Reporting commitments

- Every sensitivity cell of §6 reported alongside the primary, labeled as
  carrying no verdict authority; notable deviations labeled as future-phase
  candidates only.
- Episode counts by year, `gsector`, and size decile; drop counts with reasons
  (warmup/common-start rule, incomplete forward window, missing control).
- Verdict reported as PASS / NEAR-PASS / FAIL per §9, with each §8 condition
  shown individually as met / not met, plus the §9 diagnostic on any miss.
- A FAIL is logged with the same completeness as a PASS.
- `results/gate_log.md` is written (append-only, marker-guarded, backed up
  first) only after `src/46` runs — numbers only, no interpretation.

---

## 13. Record of confirmed decisions (2026-07-30)

1. Bars 1a / 3a / 3c as originally proposed; **1b, 2a, 2b margins relaxed
   5.0 pp → 3.0 pp** (|z| ≥ 3.0 unchanged) — reasoning: the stock-level signal
   is a smaller precursor to a larger options-based trade, not the primary edge
   source.
2. **2b (bearish symmetry) required**, at the same relaxed margin.
3. **Regime-split reporting** added to Test 2: `vwretd` trailing 10-session
   return, ±1.0% buckets, diagnostic only.
4. **3b fixed at 30 bps, flagged as placeholder.**
5. **NEAR-PASS diagnostic requirement** added (§9).
6. **Sensitivity grid narrowed to the artifact-supported axes** (§6); grid
   deviations are future-phase candidates, never E1 evidence.
7. **EMA8 slope = 1-session change**, as built (supersedes the draft's
   5-session proposal).
8. **Common evaluation start** enforced per PERMNO via the `residency_25`
   non-null rule (§4.2).
9. Holdout disclosure retained unchanged (§3.1).
10. **2b-flat added as a load-bearing condition on Test 2** (§7, §8): the
    bearish contrast must **also** hold within the mkt-flat regime bucket alone
    at **≥ +1.5 pp, |z| ≥ 2.0**, and 2b fails regardless of its pooled value if
    2b-flat fails. Reasoning: cross-sectional correlations are known to rise
    during market-wide selloffs, so a bearish result concentrated in mkt-down
    would be consistent with a market-wide effect rather than stock-specific
    ribbon information. This promotes one regime bucket from diagnostic to
    gate-relevant; the mkt-up and mkt-down splits remain diagnostic, and no
    regime split is gate-relevant for 2a. Added at lock time, before any Test 2
    outcome was computed.

11. **1a amended: `residency_10` → `continuous_residency_depth` (§5.1a,
    §7 Test 1, §8).** Amendment written 2026-07-30, after a first `src/46`
    run and before any Tests 1–3 outcome was computed.

    **What was found.** On the locked primary episode set (69,126 episodes,
    67,224 after the §4.2 common-start filter), `residency_10` is constant at
    exactly **0.6** on every single row (`residency_10_hits == 6.0` for all
    of them; std ≈ 1.1e-16). This holds across every threshold grid cell
    (thr0, thr025, thr05) as well.

    **Why it is structural, not a data defect.** §5 defines an episode as the
    *first* session of a run in which the rolling 10-session count of
    resident sessions reaches ≥ 6. A rolling count of a 0/1 flag can only
    change by at most 1 from one session to the next, so the session on
    which it *first* reaches 6 necessarily has a count of *exactly* 6 — not
    7, not 8. Every episode-start row is therefore pinned to
    `residency_10_hits = 6` (`residency_10 = 0.6`) by the mechanics of the
    episode definition itself, independent of the underlying data.
    Consequence: `residency_10`, and any term built from it, has zero
    cross-sectional variance on the episode set and is exactly collinear
    with the regression intercept — condition 1a as originally written
    (interaction of `compression_ratio × residency_10`) cannot be estimated
    at any sample size. §5 (episode = run start) and the original §7 Test 1a
    were in direct conflict. Sections 1b, 2, and 3 do not use `residency_10`
    in a regression and were unaffected.

    **Non-post-hoc confirmation.** At the point this was discovered, `src/46`
    halted before computing any Test 1 coefficient (verified: no
    `results/46_E1_tests.json` was written, no `gate_log.md` append
    occurred, and the halt was on a pre-flight variance check performed
    before the Fama-MacBeth loop executed). No 1a sign, magnitude, or
    t-statistic existed at any point before this amendment was written.

    **Why option C over A/B/D.** Four repairs were considered:
    - *(A) use `residency_25` or `residency_5` in place of `residency_10`* —
      rejected: this substitutes the primary spec's stated window (10) with
      a different one already reserved for the §6 sensitivity grid, and
      partially collapses the grid's own W-axis into the primary cell.
    - *(B) redefine episodes as every in-episode day, not just run-starts* —
      rejected: `residency_10` would regain variation, but this changes §5's
      episode definition and the episode count wholesale, well beyond a
      Test 1 fix.
    - *(C) replace the residency term with a continuous residency-depth
      measure* — **adopted**. `continuous_residency_depth` (§5.1a) := mean of
      `normalized_body_distance` over the 10 sessions ending at, and
      including, the episode-start day. This varies continuously across
      episodes (it is not gated by the ≥6-hits threshold that pins
      `residency_10`), requires no change to `src/45`'s locked output or the
      episode definition in §5, and is arguably the more faithful reading of
      "residency" the phase's §1 purpose describes — how *deep* into the
      ribbon price sat, not merely whether a threshold count was crossed.
    - *(D) drop 1a and rest Test 1 on 1b alone* — rejected: weakens Test 1 to
      a single condition when a defensible continuous alternative (C)
      exists.

    **What changed, precisely.** `residency_10` is dropped from the Test 1
    regression entirely (main effect and interaction) — keeping it as a
    regressor while also adding `continuous_residency_depth` would leave a
    constant column in the design matrix, recreating the identical
    rank-deficiency this amendment exists to fix. `continuous_residency_depth`
    takes over both roles. Test 1a's hypothesized sign flips from positive to
    **negative**, stated in §7 before computation, because the new term is
    coded in the opposite direction (lower = more resident) from the term it
    replaces (higher = more resident); this is a direct consequence of the
    substitution, not a new, independently chosen sign. Bar magnitude
    (|t| ≥ 3.0) is unchanged. 1b, Test 2 (including the mkt-flat load-bearing
    condition 2b-flat added earlier at lock time), and Test 3 are unaffected
    and unchanged by this amendment.

**Locked (with the 1a amendment above). `src/46_E1_tests.py` is built and
implements 1a as amended, 1b, Test 2 (with 2b-flat), and Test 3.**
