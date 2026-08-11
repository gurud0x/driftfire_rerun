# V4 §4.2 Addendum — Gate 6 Benchmark Volatility Forecast Model

**DRAFT — NOT LOCKED. No data contact. No feature-construction code written.**
This document is Part 2 Step 1 only: specify before building. Several items
below are **flagged, not resolved** — this draft stops at exactly those
points rather than guessing past them, per instruction. **Do not proceed to
Step 2 (data contact / feature construction) until the flags below are
resolved and this addendum is confirmed.**

One piece of code was read to answer the Step 2 guardrail properly
(`src/10_compression_signal_v1.py`) — a structural/schema check on an
existing script's construction, the same category of check V3 made before
locking its own IV method (inspecting what a file or script contains, not
computing a result from return/price data). No return, price, or IV panel
was touched.

---

## 0. What this supersedes, stated plainly

`prereg_V4.md` §4.1 already contains a **Model B** (compression-free
benchmark) from the earlier return-blind audit (item 9):
`IV², PriorRV², EarningsBucket, controls` — no compression term. §11 item 6
already locks a **portfolio-level** gate on it: `mean(Diff) > 0` and
`NW t(Diff) ≥ 2.0`, `Diff(t) = Return_A(t) − Return_B(t)`.

**This addendum's 10-feature benchmark is a materially richer model than
that earlier Model B**, and is explicitly meant to supersede it as the
benchmark's *feature content* — the task frames this as extending V3's
own logic ("compression adds information beyond IV alone") to "a fuller,
standard volatility-forecasting feature set." If this addendum is
confirmed, §4.1's Model B specification needs a dated amendment (same
non-silent-rewrite convention just used for §7.7) replacing its five-term
form with the 10-feature set below. **Not yet applied — flagged for your
confirmation alongside everything else here**, since editing §4.1 before
this addendum is settled would be the same mistake §7.7 needed correcting
for.

---

## 1. Functional form and estimation — reused, not reinvented

~~**Fama-MacBeth daily cross-sectional OLS, identical machinery to §4.2's
already-locked scheme** — expanding window, monthly refit, 24-month
burn-in, first score 2017-01, the row-wise embargo assertion
(`t' + 30 calendar days <= tau - 1 trading day`), FM-mean coefficient
aggregation, `MIN_XSEC = 30`. No new estimator is introduced, per the
task's own instruction not to add model complexity without separate
justification.~~ **SUPERSEDED — see dated correction immediately below.**

> **[2026-08-09 — ESTIMATOR CORRECTION, applied before `src/66` was run, not
> discovered after.]** The struck text above specified reusing §4.2's
> **expanding-window walk-forward** scheme. On review, that scheme exists
> to build **out-of-sample `Score_{i,t}` for live trading decisions**,
> feeding Gate 6b's portfolio (§11 item 6) — a forecasting/scoring
> construction. Gate 6a's actual purpose is different: it is the same kind
> of test **V3 itself already ran** (`src/49`) — a **single joint
> Fama-MacBeth cross-sectional regression over the full DEV sample**,
> testing whether a coefficient survives controlling for other regressors,
> not a walk-forward forecast validation.
>
> **Gate 6a was built and run using the single-joint-regression form**,
> not the expanding-window scheme: `src/49`'s exact `fama_macbeth()` /
> `nw_ols_const()` design, reused directly — daily cross-sectional OLS
> across all DEV dates at once, `MIN_XSEC = 30`, NW `maxlags = 21`. No new
> estimator was introduced; the correction is *which already-established
> project estimator* applies, not a departure from "keep this comparable
> in spirit to V1-V3's methodology" — if anything, the single joint
> regression is the more literal match to that instruction, since it is
> exactly V1/V2/V3's own construction, unmodified.
>
> **Why this matters enough for its own entry, not just a run-log
> mention:** the expanding-window scheme and the single joint regression
> produce materially different objects (one estimates a coefficient once,
> pooling all DEV cross-sections; the other refits monthly and only uses
> each window's *training* data) — a reader relying on the written spec
> above without this correction would build a different, non-comparable
> test. `results/66_v4_gate6a_regression.json` and its `gate_log.md` entry
> already carried this disclosure at run time; this section is the
> addendum's own visible record of the same correction.

Two models, same naming convention as §4.1:

- **Model B (benchmark)**: the 10 features below, no compression term.
- **Model A (augmented)**: the same 10 features **plus**
  `Compression_{i,t}` (V1's `compression_decile`, unchanged from §4.1).

**Newey-West inference**: `maxlags = 21`, matching §4.2/§11's already-locked
primary-horizon convention (V3's own correction logic, applied without
re-deriving it).

**Dependent variable — FLAGGED, recommend resolving before Step 2.** The
task specifies "forward realized volatility over the same horizon already
locked in V3 (30 calendar days)." V3's and V4's own locked target is
`RV²_{i,t,t+30cal}` — **variance**, not volatility (V3 §3.2, V4 §4.1). The
economic-size reporting convention (§1.5) converts variance to volatility
*for narrative purposes only*, never as the regression target itself.
**Recommendation: keep `RV²` (variance) as the actual regression DV**,
reading "volatility" in the task as the colloquial usage common in this
literature — but this is not certain, and switching to `sqrt(RV²)` as a
literal volatility DV is a different specification with different
statistical properties (linear model on a variance vs. a standard-deviation
target). **Flagged for explicit confirmation, not silently decided.**

---

## 2. The ten features

Every feature is `shift(1)`-aligned: computed from information through
session `t−1`, used to forecast `RV²_{i,t,t+30cal}`. All realized-measure
features (1, 3–6) are annualized-variance-units, matching the DV; the
IV-based features (7–9) are level (volatility) units, matching how implied
vol is quoted — units documented per feature, not left ambiguous.

### Feature 1 — 1-day realized volatility
`RV1_{i,t} := DlyRet_{i,t-1}² × 252` — a single day's squared return,
annualized. The shortest-horizon rung of a HAR-RV-style multi-scale
structure (Corsi 2009 uses daily/weekly/monthly realized-vol rungs; this
feature set's 1-day / short-window-autocorrelation / 20-60-day structure
is in that spirit, though not identical to HAR-RV's specific construction
— noted for context, not adopted as a citation this project is making).

### Feature 2 — Persistence (short-window autocorrelation of daily RV)
**Window locked at 20 trading days** — reusing `PRIOR_VOL_WIN = 20`
already established by E1/V3 (`src/46`), rather than introducing a new
arbitrary parameter. Construction: rolling lag-1 autocorrelation of the
`RV1` series (feature 1) over the trailing 20 sessions ending `t-1`,
computed via `pandas.Series.rolling(20).apply` on the autocorrelation
statistic, `min_periods=20` (no fallback window, matching project
convention).

### Feature 3 — Longer-window realized volatility (20-day, 60-day)
`RV20_{i,t} := Var(DlyRet over trailing 20 sessions, ddof=1) × 252`,
`RV60_{i,t}` likewise over 60 sessions — both `shift(1)`-aligned,
`min_periods` matching E1's own `PriorRV²` convention (15/45, per
`src/10`'s `min_periods=15`/`45` pattern for its own 20/60-day windows,
reused for consistency rather than picking new floors). Both reported;
RV60 doubles as an extension of §3.4's existing `PriorRV²` control, so V4
already uses one of these two windows implicitly — stated here explicitly
rather than left as an unstated overlap.

**GUARDRAIL — resolved by construction, verified by reading
`src/10_compression_signal_v1.py` directly (not by data contact):**

```
compression_ratio = vol20 / vol60
vol20 = DlyPrcVol.rolling(20, min_periods=15).mean().shift(1)
vol60 = DlyPrcVol.rolling(60, min_periods=45).mean().shift(1)
```

`compression_ratio` is a ratio of two rolling **means of dollar volume**
(`DlyPrcVol`). `RV20`/`RV60` above are rolling **variances of daily
returns** (`DlyRet`). The two features share window *lengths* (20, 60
days) but are built from **entirely different raw input series** — volume
vs. returns — confirmed by reading the construction, not asserted from
memory. This is the "by construction differences" resolution the guardrail
asks for, not an ablation.

**Honest caveat, not swept under the resolution above:** mechanical
independence of construction does not imply zero *empirical* correlation.
Volume and volatility are well documented as positively correlated in
market microstructure (the "mixture of distributions hypothesis" — Clark
1973, Tauchen & Pitts 1983), so some correlation between `RV20`/`RV60` and
`compression_ratio` is expected and unsurprising, not a red flag on its
own. **Proposed confirmatory step, deferred to Step 2** (since it requires
touching the DEV panel): report `corr(RV20, compression_ratio)` and
`corr(RV60, compression_ratio)` on the DEV panel as a printed diagnostic
before the Gate 6 regression is fit, so the magnitude is visible rather
than assumed. Not run here — this is data contact, out of scope for Step 1.

### Feature 4 — Downside variance (5-day)
`DSV5_{i,t} := [ Σ_{s ∈ (t-5,t-1]} DlyRet_{i,s}² · 1{DlyRet_{i,s}<0} ] ×
(252/5)` — sum of squared *negative* daily returns only, over the trailing
5 sessions, annualized by `252/5` (treating the sum as an average per-day
semi-variance, consistent with how a 5-day realized variance would
annualize if all 5 days contributed). `shift(1)`-aligned,
`min_periods = 5` (no fallback).

### Feature 5 — Jump variance (daily-frequency proxy, limitation disclosed)
`RV20` (feature 3) less a daily-frequency bipower variation estimate over
the same 20-day window:

```
BPV20_{i,t} := (pi/2) * Σ_{s=2}^{20} |DlyRet_{i,s}| * |DlyRet_{i,s-1}|  (annualized x252/20)
JumpVar_{i,t} := max(RV20_{i,t} - BPV20_{i,t}, 0)
```

**Disclosed limitation, stated as instructed rather than presented as a
true jump measure.** Bipower variation's statistical property of isolating
continuous from jump variation is established in the **high-frequency
(intraday) asymptotic limit** (Barndorff-Nielsen & Shephard 2004); this
project has only **daily** CRSP data, no intraday. Applying the BPV
*formula* to a 20-day sequence of daily returns is mechanically valid but
statistically weak — with only 20 observations and daily (not intraday)
sampling, this is a coarse proxy for "variance not explained by a
first-order smoothed estimate," not a rigorously identified jump process.
The literature this project is drawing on (Branger, Hülsbusch &
Middelhoff 2017 and the broader jump-variation literature) generally uses
5-minute or similar intraday data for this estimator — a materially
different data regime. This limitation is carried into any Gate 6 write-up
that reports this feature, not just noted here and dropped.

### Feature 6 — Volatility-of-volatility
`VoV20_{i,t} := StDev(|DlyRet| over trailing 20 sessions, ddof=1)`,
`shift(1)`-aligned, `min_periods=20`. Window reuses the same 20-day
convention as features 2/3 for consistency. **Analogous in spirit to
IVOLVOL** (Branger, Hülsbusch & Middelhoff 2017) but built from **realized**
(daily-return) inputs, not option-implied inputs — the two are not the
same quantity and this document does not conflate them in any write-up,
per instruction.

### Feature 7 — 30-day ATM implied volatility
Reused verbatim from V3 §3.3-F Resolution's already-locked construction:
`IV_30`, the mean of call-side and put-side ATM `impl_volatility` at the
surface's native 30-day tenor (minimum `|delta/100 − 0.50|` selection, both
sides required). **Level units** (not squared), unlike this project's usual
`IV²` regressor — kept in level form here to match features 8/9's own
level-based construction (term structure and skew are natural differences
of IV levels, not variances).

### Feature 8 — IV term structure (30d − 60d) — **FLAGGED, DATA DOES NOT EXIST AS SPECIFIED**

**`vol_surface.csv` carries exactly two standardized tenors: 10 and 30
calendar days** — this is not a new finding, it is V3 §3.3's own confirmed,
full-file-scanned fact ("carries exactly two standardized tenors in the
`days` column: 10 and 30, both calendar days"). **There is no 60-day tenor
in the file this project has actually pulled.** Feature 8 as literally
specified (30d − 60d) is **not constructible** from `vol_surface.csv`.

Two paths, neither adopted here:
- **(A) Substitute 10d − 30d term structure.** Constructible immediately,
  no new pull — but inherits V3 §3.3-F's own finding that the 10-day
  tenor's usable-both-sides coverage is **4.30%** of the universe (the same
  sparsity problem that forced V3's own primary horizon to move away from
  10 days in the first place). A term-structure feature built on 4.3%
  coverage would itself be mostly missing, and a missing-feature row policy
  would need to be decided (drop-and-count, matching project convention, or
  impute — imputation is against this project's standing rule).
- **(B) Check `vol_surface_full_grid.csv`** — noted to exist in the local
  staging folder (`~/Downloads/quantdata/driftfire/raw/optionmetrics/`,
  109 GB, same columns as `vol_surface.csv` per an earlier header check
  this session) but **never scanned for its actual tenor grid**. It may or
  may not carry a genuine 60-day point. Checking would require a **new,
  narrowly-scoped structural pull** (tenor/coverage counts only, the same
  category of check V3 §6(b) ran before locking its own thresholds) —
  **data contact requiring authorization**, not yet requested.

**Not resolved here. Recommend (B) if a genuine 60-day point turns out to
exist and clears a reasonable coverage bar; otherwise (A) with the coverage
cost disclosed plainly.** Flagged for your decision.

### Feature 9 — Skew / fear premium (25-delta put IV − ATM IV, 30-day tenor) — **FLAGGED, COVERAGE UNVERIFIED**

Planned construction: at the 30-day tenor, select the row nearest
`delta = −25` (on OM's `delta/100` scale, matching V3's own ATM selection
logic applied to a different delta target) for `cp_flag='P'`, subtract
`IV_30` (feature 7). This is mechanically the same per-chunk
nearest-delta selection V3 §3.3 already uses for ATM (`dpen = |delta/100 −
0.50|`), retargeted to `|delta/100 − (−0.25)|` for puts.

**Not yet verified: does `vol_surface.csv` carry usable coverage near
`delta ≈ −25` at the 30-day tenor, at a rate that isn't another 4.3%-style
sparsity problem?** This is unknown until checked. **Proposed: a
structural coverage check** (row counts near the 25-delta point, by year
and decile — the same category of check V3 §6(b) and this session's V3-short
audit both ran, descriptive of data availability, not a computed result)
**before Step 2 begins**, not assumed to work. Flagged, not assumed.

### Feature 10 — Market (systematic) realized volatility
`MktRV5_{t} := Var(vwretd over trailing 5 sessions, ddof=1) × 252`,
`shift(1)`-aligned — one value per date, applied to every stock that day.
**`vwretd` chosen over `sprtrn`, reusing E1 §5.1's own precedent and stated
reason verbatim**: "the E1 universe is deciles 6–8 (mid-caps): the S&P 500
is a large-cap subset, while `vwretd` covers the full tape including the
sample's own size range." V4 uses the identical universe, so the identical
reasoning applies without re-deriving it. `vwretd` already confirmed
zero-null over DEV by E1's own construction.

---

## 3. Missing-feature policy

Reused verbatim from project convention (V3 §3.3, `src/43`): a row missing
any of the 10 features (most likely feature 8 or 9, per the flags above) is
**dropped from that model's estimation and counted**, never imputed or
filled. Reported by feature, by year, matching every prior phase's
disclosure standard.

---

## 4. Required console output before any regression is fit

Per standing project convention (not a new requirement): for each of the 10
features, print mean, std, min, max, and % missing over the DEV panel
**before** the Gate 6 regression runs — real inspected numbers, not
assumed-correct construction.

---

## 5. Holdout — **FLAGGED, DIRECT CONFLICT WITH AN ALREADY-LOCKED RULE, NOT RESOLVED HERE**

**This is the most consequential flag in this document and needs an
explicit decision before Step 2, not a default.**

The task's Step 3 asks: "Repeat both fits on holdout, report the same
statistics." `prereg_V4.md` §3 currently states, unconditionally:

> "V4 runs on DEV only. **No V4 script may contain a holdout code path.**"
> "...the 2022–2025 window has already served as the holdout for **five**
> phases: R1, R2, V1, V2, and K1... V4 makes **no holdout claim**. If a V4
> holdout pass is ever proposed it must be disclosed **at that time** as a
> **sixth look at the same calendar period, not a fresh out-of-sample
> test**."

Running Gate 6 on holdout, as literally requested, would be exactly the
scenario §3 already anticipated and required special disclosure for — and
`docs/CLAUDE.md` rule 3 states project-wide: "Holdout is spent once. Never
re-run, retune, or re-split after a holdout pass."

**This document does not build a holdout code path and will not, absent an
explicit instruction to do so that also amends §3 with the required
sixth-look disclosure.** Three ways this could be resolved, none chosen
here:

1. **Run Gate 6 on DEV only**, matching every other locked V4 component,
   and drop the holdout comparison from Step 3's reporting.
2. **Explicitly authorize a sixth holdout look**, with the same dated,
   disclosed treatment E1's own precedent requires — this would need a
   parallel amendment to §3, not a silent addition inside this addendum.
3. **Reserve holdout for a later, single, pre-committed V4 promotion
   pass** (matching how V1/V2/K1 each spent their own single holdout look
   only after a logged DEV PASS) — meaning Gate 6, like every other V4
   gate, is evaluated on DEV now, and holdout is saved for one full,
   final V4 promotion decision later, not spent piecemeal on individual
   gates.

**Flagged. Awaiting your decision. No holdout code will be written until
one of these is chosen and, if (2), §3 is amended accordingly.**

---

## 6. What "Gate 6" means here — **FLAGGED, NAMING AMBIGUITY, NOT RESOLVED**

§11 item 6, already locked, is a **portfolio-level** test: `Diff(t) =
Return_A(t) − Return_B(t)`, actual traded returns of two full delta-hedged
books, requiring the complete hedging/execution/sizing machinery this
project has explicitly not built yet.

This task's Step 3 describes a **regression-level** test: the compression
coefficient's own NW t-stat in an augmented forecasting regression, plus
incremental R² — no traded position, no hedging, no P&L, computable from
forecasts alone. Step 3 point 4 asks to "state plainly whether NW t(Diff)
≥ 2.0 is met" — reusing §11 item 6's exact notation and threshold for a
different quantity (a regression coefficient's t-stat, not a portfolio
return-difference's t-stat).

**Proposed resolution, flagged for confirmation, not applied silently:**
treat this regression-level test as a **pre-gate diagnostic** — call it
**"Gate 6-forecast"** — that determines whether it is worth building the
expensive portfolio simulation for the actual locked **Gate 6
(§11 item 6, portfolio-level)** at all. If compression does not survive
this richer benchmark at the forecasting level, that is strong information
before committing to a hedging-simulation build; if it does survive, §11
item 6 remains the criterion that actually decides a V4 PASS. **This task's
"NW t(Diff) ≥ 2.0" language would then refer to Gate 6-forecast's own
compression coefficient t-stat**, a distinct number from §11 item 6's
future portfolio-level `Diff` t-stat, and the two should never be reported
under the same symbol without this distinction stated. Not adopted without
your confirmation — an equally valid alternative is that you intend this
regression test to **replace** §11 item 6 outright, which would itself be a
locked-document amendment, not an addition alongside it.

---

## 7. Hou-Loh decomposition (Step 4) — status, not built

Understood and accepted as described: a **diagnostic, not a new
pre-registered gate**, run regardless of Gate 6-forecast's outcome, logged
to `gate_log.md` as diagnostic per the task's own instruction. No open
flags on this step's role — it is correctly scoped as reporting on data
Step 3 already produces, not a new data contact or a new gate. Not built
yet; sequenced after Step 3 resolves.

---

## 8. Summary — six items block Step 2

| # | flag | recommendation | status |
|---|---|---|---|
| 1 | §4.1's existing Model B is superseded by this richer 10-feature set | apply as a dated amendment once this addendum is confirmed | open |
| 2 | DV: variance vs. volatility | keep `RV²` (variance), matching V3/V4's existing target | open |
| 3 | Feature 3 vs. compression overlap | resolved by construction (verified: different raw inputs); correlation check deferred to Step 2 | **resolved**, one deferred confirmatory step |
| 4 | Feature 8: no 60-day tenor in `vol_surface.csv` | choose 10d/30d (inherits 4.3% coverage) or authorize a `vol_surface_full_grid.csv` structural check | open |
| 5 | Feature 9: 25-delta put coverage unverified | run a structural coverage check before Step 2 | open |
| 6 | Holdout requested, conflicts with locked §3 | choose DEV-only / explicit sixth-look amendment / reserve for a later single promotion pass | open |
| 7 | "Gate 6" naming — regression-level vs. portfolio-level | treat regression test as a pre-gate diagnostic ("Gate 6-forecast"), §11 item 6 remains the actual PASS criterion | open |

**No feature-construction code will be written and no data will be touched
beyond the one code-read already disclosed in this document's header until
these are resolved.**

---

## Addendum — dated resolution of flagged items (2026-08-04)

**Original flags above are left unedited, not rewritten** — this section
resolves them, matching the non-rewrite amendment convention just applied
to `prereg_V4.md` §7.7. Six items resolved; Step 2 proceeds after this
block on the items that are fully closed, with the two coverage-dependent
items (4 and 5) closed using the measured numbers below.

### Item — §7.3 (Rehedging), stale "through expiration" language

**No fix now.** Confirmed out of scope until the hedging-simulation phase
begins — no computed V4 result depends on §7.3 as of this date, and fixing
it now would be premature since the hedging simulation that would exercise
it doesn't exist yet. Remains flagged in `gate_log.md`'s 2026-08-04 entry.

### Item 4 (table) / Feature 8 — 60-day IV term structure, coverage measured

**Inspection run, authorized as inspection-only (not hypothesis-relevant
data contact):** `src/62_v4_feature8_tenor_coverage_check.py`, full scan of
`vol_surface_full_grid.csv` (109 GB, staging, secid/date/days only for the
existence check, delta/cp_flag added only for the follow-on ATM-coverage
pass once existence was confirmed).

**Finding: the full grid carries 11 standardized tenors** (10, 30, 60, 91,
122, 152, 182, 273, 365, 547, 730 calendar days) — **a 60-day point does
exist**, unlike the two-tenor `vol_surface.csv` this project has actually
been using (`KEEP_DAYS=(10,30)`, per V3 §3.3).

**A real bug was caught and fixed mid-check, disclosed rather than
smoothed over.** The first pass (`src/62`) measured coverage as "a row
with a valid delta label exists" and found an identical **92.05%** for
all three tenors checked (10, 30, 60) — suspiciously uniform. A spot check
(secid 8170, 2015-01-02) found rows with valid delta labels at `days=10`
whose `impl_volatility` was **NaN**: `vol_surface_full_grid.csv` carries a
**complete synthetic grid skeleton** — a row for every standard
tenor/delta combination — independent of whether OM's surface model
actually fit a usable value there. The 92.05% figure measured skeleton
existence, not usable IV, and would have materially overstated coverage.
**Corrected** (`src/62b_v4_feature8_iv_coverage_corrected.py`, adds the
missing `impl_volatility.notna() & > 0` filter) and re-run in full:

```
days=10: 4.30% of base universe  [EXACT match to V3 3.3-F's own cited 10d figure - cross-validates the method]
days=30: 91.10% of base universe [V3 3.3-F reference: 91.45% - same ballpark, same file, independent construction]
days=60: 91.10% of base universe [identical to the 30-day figure]
```

The corrected 10-day figure landing on **exactly** V3's own independently-
measured 4.30% is strong evidence the corrected methodology is measuring
the same real quantity, not an artifact of this specific script.

**Decision, per the pre-authorized rule: BUILD feature 8.** 91.10% is
comparable to the 30-day node (91.45%), nowhere near the thin 10-day node
(4.30%) — the 60-day point is genuinely, usably populated at essentially
the same rate as the 30-day point already relied on throughout this
project. Construction: a **new, narrowly-scoped pull** of just the
30d/60d ATM rows (both sides, DEV-window base universe only) from
`vol_surface_full_grid.csv` — not the entire 109 GB file retained — feeding
`IV_ATM_var` (feature 7) and `IVTermStruct_var` (feature 8) together from
one extraction.

### Item 6 (table) / Feature 9 — 25-delta put skew, resolved with a documented substitution

**Inspection run, authorized as inspection-only:**
`src/63_v4_feature9_skew_coverage_check.py`, full scan of the
already-pulled `vol_surface.csv` (no new pull).

**Finding: `vol_surface.csv`'s put-side delta grid at the 30-day tenor is
a hard boundary, not sparse coverage.** The file carries put deltas only
at `{-35, -40, -45, -50, -55, -60, -65}` — confirmed by direct inspection
of the distinct values present, not inferred. **There is no delta point
anywhere near -25**; every 25-delta candidate row found was exactly 10
delta points from target (i.e., resolving to -35), with **zero** coverage
within even a generous ±5-point tolerance of -25.

**Follow-up check, same authorization (inspection-only, no hypothesis
touched):** coverage at the nearest *actually available* grid point,
`delta = -35` exactly: **91.45%** of the base universe — identical to the
"any 30-day put quote present" baseline. This is not a degraded substitute;
`-35` is evidently a second standardized grid point in this file, exactly
as reliably populated as the ATM point itself.

**Decision, documented per the delegated "decide on a substitute delta...
documented either way":** **Feature 9 is redefined as `IV(put, 30d,
delta=-35) − IV(ATM, 30d)`** — a moderate-OTM put skew at the nearest
grid point this data actually supports, not the literal 25-delta. This is
a **substitution of the delta target, not an abandonment of the feature**:
the economic content (put-side skew / crash-risk premium) is preserved;
only the specific moneyness point moves from a value this data cannot
produce to the nearest one it can, fully covered (91.45%, matching feature
7's own ATM coverage almost exactly — no incremental missing-row cost from
this substitution).

### Item — Holdout

**DEV-only, confirmed, no holdout code path built or planned for this
diagnostic.** Holdout remains reserved for a single, later, full V4
promotion pass (matching V1/V2/K1's own precedent), evaluated and
disclosed separately when that time comes — not spent piecemeal across
individual gates. `prereg_V4.md` §3's existing prohibition stands
unchanged; no amendment to §3 is needed since this resolution doesn't
touch holdout at all.

### Item — Naming: Gate 6a / Gate 6b

**Confirmed and locked.** The regression-level, DEV-only, forecast-quality
test built by this addendum is **Gate 6a** (signal-level: does compression
carry incremental predictive power over the 10-feature benchmark). The
already-locked `prereg_V4.md` §11 item 6 — portfolio-level, full-cost,
`mean(Diff)>0` and `NW t(Diff)≥2.0` on traded book returns — is **Gate 6b**
and is **unchanged**. Gate 6a's result will be logged in `gate_log.md` as
its own numbered entry, cross-referenced to Gate 6b, never substituted for
it. A Gate 6a pass does not constitute a V4 PASS by itself; it is a
go/no-go signal for whether building Gate 6b's full hedging simulation is
worthwhile.

### Item — DV units (variance, with an explicit squaring step)

**Confirmed: the regression DV is `RV²` (variance)**, matching V3's and
V4's own locked target — resolving §1's flag in favor of the existing
project convention, not a literal volatility DV.

**Features 7–9 (ATM IV, term structure, skew) are constructed in Black-
Scholes volatility units first, then squared into variance units as their
own explicit, separately visible step** — not silently folded inside a
single feature function:

```
IV_ATM_var_{i,t}   := IV_ATM_{i,t}²            (feature 7, squared)
IVTermStruct_var    := (IV_30 - IV_60)²  x sign(IV_30 - IV_60)   [signed-square,
                        preserving the sign of the level difference rather than
                        destroying it - a plain square would make a positive
                        (upward-sloping) and negative (inverted) term structure
                        indistinguishable, which defeats the feature's purpose]
IVSkew_var_{i,t}    := (IV_put35 - IV_ATM)²  x sign(IV_put35 - IV_ATM)   [same
                        signed-square treatment, same reasoning]
```

**Disclosed departure from a naive squaring, stated now rather than
discovered later:** features 8 and 9 are *differences* of levels, and a
plain square of a difference discards its sign — economically, an inverted
term structure and a steep upward one are different phenomena a plain
square would conflate. The **signed square** (`x² · sign(x)`, sometimes
called the "signed squared" transform) preserves direction while still
expressing the feature in variance-scaled units, keeping it comparable in
magnitude to the DV and to feature 7. Feature 7 itself (a level, not a
difference) uses a plain square since it has no sign to preserve.

---

## Companion amendment applied to `prereg_V4.md` §4.1

Proceeding to Step 2 with the 10-feature benchmark set above operationally
confirms superseding §4.1's earlier, simpler Model B (item 1 of the
original flag table) — a dated, non-rewrite amendment has been applied
directly to `prereg_V4.md` §4.1 pointing here, consistent with the same
convention used throughout this document and §7.7.

**Status: all seven original flags now closed** (feature 3 resolved in the
original draft; §7.3, feature 9, holdout, naming, and DV-units resolved
above; feature 8 resolved above with a corrected coverage measurement,
91.10% at the 60-day tenor — **build**, not drop). Step 2 (feature
construction) proceeds for all 10 features.

---

## Addendum — Gate 6a result qualification (2026-08-10)

**⚠ FLAGGED FOR WHOEVER DRAFTS THE PAPER SECTION ON GATE 6a — READ BEFORE
CITING GATE 6a'S HEADLINE NUMBERS.**

Gate 6a's logged result (`results/66_v4_gate6a_regression.json`,
`gate_log.md`) is: compression coefficient −6.312308e-03, NW t = −9.3978,
incremental R² = +0.003489, in the full 10-feature benchmark model. A
follow-up investigation (`src/67`–`src/69`, `gate_log.md`'s "V4 Gate 6a -
correlation and regression-order check" entry, now marked **RESOLVED**)
found and confirmed a **suppression effect** between `compression_decile`
and `IV_ATM_var` (30-day ATM implied variance): the two are weakly
correlated on their own (pooled Pearson r = −0.0177) but substantially
more correlated once both are residualized against the other 9 benchmark
features (partial r = −0.1220, roughly 7× larger). Removing `IV_ATM_var`
alone from the benchmark set moves compression's NW t-stat from −9.40 to
−13.43 — by far the largest shift from removing any single feature — and
the leave-one-out decomposition attributed more than 100% of compression's
incremental R² to `IV_ATM_var` in isolation (offset by a large negative
residual from the other features), the signature of overlapping rather
than additive explanatory content.

**The honest characterization, stated plainly: Gate 6a's incremental R² is
real — the augmented model does fit better than the benchmark-only model,
and the effect is not an artifact of a coding error or a spurious
collinearity (both were checked and ruled out) — but it is *not fully
independent* of `IV_ATM_var`.** Part of what the full-model regression
attributes to compression is compression's own overlap with implied
variance, mediated through its shared variance with the other 9 benchmark
features, not information wholly separate from what implied variance
already captures. This is conceptually the same kind of caution §1.1
already states about V3 itself (incremental information is not the same
claim as an independent or orthogonal one) — Gate 6a's result should be
read with the same care, not presented as a clean, unqualified orthogonal
finding.

**Do not silently report the −9.3978 / +0.003489 headline pair as a
self-contained, fully independent result without this qualification
attached.** Full investigation trail: `results/67_v4_gate6a_sensitivity_decomposition.json`
(leave-one-out and decomposition), `results/68_v4_gate6a_correlation_ordering_check.json`
(pooled correlation and regression-order dependence),
`results/69_v4_gate6a_partial_correlation_check.json` (per-date correlation
and partial correlation) — and the corresponding `gate_log.md` entry,
now closed as RESOLVED.

**Not yet decided, and explicitly out of scope for this addendum:** whether
an orthogonalized compression measure, a revised benchmark specification,
or some other follow-up is warranted before Gate 6a is treated as a
completed, final result. That decision is the owner's, not made here.

---

## Final consolidation for write-up (2026-08-11)

**(a) Raw compression result** (`results/66_v4_gate6a_regression.json`, DEV
only, single joint Fama-MacBeth, full 10-feature benchmark):

```
compression_decile coefficient: -6.312308e-03
NW t-stat (maxlags=21):         -9.3978
incremental R^2:                +0.003489
```

**(b) Orthogonalized compression result**
(`results/70_v4_gate6a_orthogonalized_compression.json`). Follow-up ordered
after the RESOLVED suppression finding: `compression_orthogonal :=
resid(compression_decile ~ const + other 9 benchmark features)`, pooled
OLS, then the same Gate 6a regression re-run with `compression_orthogonal`
in place of the raw measure, same 10-feature benchmark:

```
compression_orthogonal coefficient: -6.312308e-03
NW t-stat (maxlags=21):              -9.3978
incremental R^2:                     +0.003489
```

**⚠ These numbers are IDENTICAL to (a), and that identity is not
independent confirmation — read the mechanism before citing this as a
robustness check.** `compression_orthogonal` was residualized against the
**same 9 features that remain in the regression it was tested in**
(everything in the 10-feature benchmark except `IV_ATM_var` itself). By
the Frisch-Waugh-Lovell theorem, orthogonalizing a regressor against
controls that are *already in the model* cannot change that regressor's
coefficient, t-stat, or the model's R² — the full model already performs
this projection internally during OLS fitting. **This follow-up was
mathematically guaranteed to reproduce (a) exactly, by construction, before
a single number was computed.** It therefore does **not** test whether
compression's effect is independent of `IV_ATM_var` specifically — the one
relationship the RESOLVED suppression finding actually flagged. A test
that speaks to that would require residualizing compression against **all
10** benchmark features, including `IV_ATM_var` — a different, materially
harder specification (the incremental R² of a variable orthogonalized
against everything already in its own model is mechanically zero by the
same theorem, so that version would need to be evaluated as a *standalone*
univariate forecast of `RV²`, not as an addition to the 10-feature model).
~~Not run. Flagged as the follow-up that would actually close this
question, not the one that was run.~~ **[2026-08-11 — now run, see (d) below.]**

**(c) Honest one-paragraph conclusion for a paper draft.** V4's
compression signal shows a real, non-spurious incremental relationship
with forward realized variance beyond a ten-feature standard volatility-
forecasting benchmark (NW t = −9.40, incremental R² = +0.0035, DEV-only,
single joint Fama-MacBeth) — the effect survives a leave-one-out check
against nine of the ten benchmark features individually, is not an
artifact of the market-level `MktRV5` collinearity bug caught during
construction, and is not explained by a raw correlation with any single
feature (all pooled and per-date correlations with the ten benchmark
features are small). It is, however, **partially entangled specifically
with 30-day ATM implied variance**: a confirmed suppression effect (raw
Pearson r = −0.018, partial r after conditioning on the other nine
features = −0.122, roughly 7× larger) means part of what this regression
attributes to compression is compression's shared variance with implied
volatility, not information wholly separate from what the options market
already prices — and the one orthogonalization test that could cleanly
settle how much is **not yet run**, for the algebraic reason stated above.
**Monetization has not been tested at all**: Gate 6a is a signal-level,
cost-free, DEV-only forecasting diagnostic — Gate 6b (the portfolio-level,
full-cost test, §11 item 6), the hedging simulation, and any P&L
construction remain entirely future work, not started at any point in this
investigation.

---

**(d) FINAL closing test — `compression_orthogonal_full`, standalone
univariate forecast (2026-08-11)**
(`results/71_v4_gate6a_final_standalone_orthogonal.json`,
`src/71_v4_gate6a_final_standalone_orthogonal.py`). This is the test
flagged as not-yet-run in (b): `compression_orthogonal_full :=
resid(compression_decile ~ const + ALL 10 benchmark features, including
`IV_ATM_var`)`, pooled OLS, then tested as a **standalone univariate**
Fama-MacBeth forecaster of `RV2_primary` — not re-inserted into the
10-feature model, since that would be mechanically zero-incremental by
Frisch-Waugh-Lovell (already established in (b)).

```
compression_decile ~ all 10 benchmark features (pooled OLS): R2 = 0.091450
  (n = 1,585,103; for reference, R2 vs the other 9 only, (b)/src/69-70: 0.077720)

STANDALONE FORECAST: RV2_primary ~ const + compression_orthogonal_full
  n_dates: 1,701 (dropped 0)   mean_xsec_n: 927.6
  R^2:          0.008278
  coefficient:  -4.747852e-03
  NW t-stat:    -3.2787
```

Reported plainly, no reframing: the NW t-stat is smaller in magnitude than
the raw/(a) result (−3.28 vs −9.40) but still exceeds conventional
significance thresholds (|t| > 2.58 at the 1% level). The coefficient sign
is unchanged. This is a materially different number from (a) and (b) — not
identical, unlike (b), since this orthogonalization was run against
`IV_ATM_var` too and is therefore not mechanically constrained by FWL to
reproduce the raw result.

**(e) Updated honest conclusion — supersedes (c) above.** (c)'s open
question — how much of compression's incremental R² in the 10-feature
model reflects information not already priced by 30-day ATM implied
volatility — is now answered directly rather than left as an algebraic
gap: after stripping out everything correlated with the full 10-feature
benchmark, including `IV_ATM_var` itself, compression still forecasts
forward realized variance on its own (NW t = −3.28, own R² = 0.0083),
though the statistical strength drops considerably from the raw/joint-model
result (NW t = −9.40). This does not overturn (c)'s qualitative reading —
the signal is real and not fully explained by any single benchmark
feature, including implied volatility — but it sharpens it: the honest
characterization is no longer "the size of the independent effect is
untested," it is "the independent effect is real but roughly a third the
statistical magnitude of the number that would appear in a paper draft
citing the raw or 9-feature-orthogonalized regression." A paper draft
citing Gate 6a's forecasting result should cite (d)'s t = −3.28, own R² =
0.0083 as the defensible lower-bound, IV-independent number, not (a)'s
t = −9.40, which is now known to be partly attributable to shared variance
with `IV_ATM_var`. **Monetization remains completely untested**: Gate 6b,
hedging, and P&L construction have not been started at any point in this
investigation.
