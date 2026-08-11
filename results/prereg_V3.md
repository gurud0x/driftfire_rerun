# Phase V3 Pre-Registration — Incremental Variance Information

**Status: LOCKED — 2026-08-02.** All sections are locked, including §6(b)'s
numeric liquidity thresholds, §3.3-F's horizon resolution, and the two
follow-on items §3.3-F's resolution disclosed. §3.3-F (the missing-tenor
problem found during §6(b)'s feasibility pass) is resolved: **V3's primary
comparison horizon moves from 10 trading days to 30 calendar days**,
matching the tenor where usable IV coverage actually exists (91.45% vs
4.30%) — decided on data-availability grounds alone, before any V3 outcome
existed. The original 10-trading-day construction is retained as a labeled,
no-verdict-authority sensitivity arm (§3.3-F Resolution, §4). Both items
§3.3-F's resolution left open are now closed, same day, same
outcome-blind standard: **§5's `maxlags` is corrected to 21 for the primary
/ 10 for the sensitivity arm** (matching each spec's own horizon, V1/V2's
original logic just corrected for the new horizon); **the ACT/365 day-count
assumption could not be verified against any locally available OM
documentation, so instead of a deferred sensitivity promise, a precise
result is locked** — because `IV²` is the only model term the assumption
touches and it enters linearly, `b3` is provably invariant to ACT/365 vs
ACT/360, and the eventual build script is required to assert this holds.
This phase is now fully runnable.

No coefficient, correlation, or regression on real return/IV data has been
computed anywhere in the production of this document, at any point across
its drafting, feasibility, or resolution passes. The real-data facts used
are **coverage/structural/calendar** ones — which tenors and which option
chains exist, for how many stock-days, and how many trading sessions fall in
a fixed calendar window — the same category of fact `src/43`'s "RDQ SOURCE
DISCOVERY" step inspected before locking that phase's method. No V3 outcome
exists, so nothing here can have been tuned to a result. One figure was
produced to support the horizon decision — IV coverage by tenor, year, and
decile (`results/figures/v3_iv_tenor_coverage.png`/`.pdf`) — purely
descriptive of data availability; no plot of coefficients, t-stats, or
predicted results across candidate horizons was produced or requested, per
this project's rule against choosing a parameter by its performance (E1 §6).
`gate_log.md` has not been touched.

**Phase type:** exploratory (V-series extension), DEV window only.
**Scripts:** no V3 regression script built yet — the pre-registration is
complete and locked, construction is not. `src/47_v3_data_feasibility.py`
(run 2026-08-02, extended across four steps) and
`src/48_v3_iv_tenor_coverage_figure.py` (the coverage figure) are both
built and run. Artifacts: `results/47_v3_liquidity_coverage.json`,
`results/47_v3_iv_tenor_coverage.json`, the durable cache
`data/processed/opprcd_liquidity_daily.parquet` (3,129,623 secid-days, so no
future run re-scans the 75 GB source), and
`results/figures/v3_iv_tenor_coverage.png`/`.pdf`.
**Lineage:** corrects the RV-IV framing per the reviewer critique. Builds on
locked results from V1 (`docs/PhaseV1_PreRegistration_VolatilityCompression.md`,
commit 4474312), V2 (`docs/PhaseV2_PreRegistration_SectorRelativeCompression.md`,
commit 3fdac95), `src/43_earnings_confound.py` (earnings-timing confound,
COMPLETE), and E1 (`results/prereg_E1.md`, control-variable precedent).

---

## 1. Purpose

V1/V2 show compression predicts 10-trading-day forward realized volatility
(locked primary horizon). `src/43` shows this survives an earnings-timing
confound at roughly half coefficient magnitude (V1 retains 50.3%, V2 retains
42.2%) while remaining significant, with a significant compression ×
earnings-dummy interaction. K1 attempted to monetize a related but
**not equivalent** concept (~30-day straddles) and failed on spread cost;
`src/42`'s delta-hedge decomposition found a nominally positive but
statistically unreliable edge once directional exposure is removed.

None of this has tested whether compression predicts variance **beyond what
the options market already prices**. If implied variance already reflects
the compression pattern, V1/V2's forecasting result is real but
commercially inert — the market has already marked it in. V3 tests this
directly: does compression carry information about forward realized
variance **incremental to** contemporaneous implied variance and recent
realized variance, and if so, is that information structural, event-driven,
or confined to stocks whose options can't actually be traded.

---

## 2. Universe and window

Identical DEV-window discipline as every prior phase: **2015-01-01 to
2021-12-31**, holdout untouched, no holdout code path in any V3 script when
built. The 2022-2025 holdout is already spent for V1 and V2 (single
pre-committed passes, both logged PASS) — V3 makes no holdout claim.

Base universe: `data/processed/universe_membership.parquet`, NYSE-breakpoint
size deciles 6-8, monthly point-in-time — identical construction used by
R1/R2/V1/V2/K0/K1/E1/`src/43`. No new universe is built. Two research
universes are carved from this base for reporting (§6); neither replaces it.

---

## 3. Signal, target, and control definitions

### 3.1 Compression regressor — both V1 and V2, run separately

`Compression_{i,t}` is **not** a new construction. Two variants, each
already validated as the primary regressor of its own locked phase, are run
as two separate instances of the full model in §4 — exactly the "both
reported, neither privileged" convention `src/43` used for V1 vs V2:

- **V1 variant:** `compression_decile` from `compression_signal_v1.parquet`
  (1-10 rank, 1 = most compressed). Reused verbatim.
- **V2 variant:** `sector_rel_decile` from
  `sector_compression_signal_v2.parquet` (1-10 rank within the sector-relative
  measure). Reused verbatim.

Both enter the regression **un-standardized**, on the same 1-10 decile scale
V1/V2's own gates used. This is deliberate: rescaling the regressor would
make `b3` here incomparable to V1/V2's own logged slope and to `src/43`'s
`beta_compression(i)`/`beta_compression(ii)`, breaking the chain of
comparison this phase depends on.

No sector fixed effects are added to the V2 specification: V2's own
construction (`src/17`) already nets out the same-day sector median, so a
sector dummy would partially absorb the signal being tested, not just
nuisance variation.

### 3.2 Realized variance target, RV²

`RV²_{i,t,t+10} := (realized_vol_fwd_10d_{i,t})²`

Reused verbatim from `compression_signal_v1.parquet` (V1's own column,
identical for the V2 run since V2 reuses V1's forward-return construction
unchanged), squared — not recomputed. This is annualized variance on a
**252-trading-day** basis (V1's construction: `std(DlyRet over 10 sessions)
* sqrt(252)`, then squared here). Window: sessions t+1..t+10, `shift(-10)`
of a backward-looking rolling window — no look-ahead, per V1's own
look-ahead confirmation.

### 3.3 Implied variance term, IV² — LOCKED NOW, PRECISELY

This is the section the task calls "critical." Structural facts about the
surface file were checked before locking this (schema/coverage, not a
regression result — see the note at the top of this document):

**Surface tenor facts (confirmed, full-file scan, not a sample):**
`data/raw/optionmetrics/vol_surface.csv` (78,796,690 rows) carries **exactly
two** standardized tenors in the `days` column: **10 and 30**, both
**calendar** days to expiration (not trading days — confirmed by K1's own
code, which treats `days == 30` as matching `entry_date + Timedelta(days=30)`
calendar-day target expirations). Population share: `days=10` is 6.88% of
rows; `days=30` is 93.12%. `src/27`'s `KEEP_DAYS = (10, 30)` filter is the
reason no other tenor survives in the repo file — the original full grid
(if it still exists, only in the local Downloads staging folder, not
verified here) may carry more, but re-scanning 100+ GB to find out is not
a prerequisite for locking the method below, for the reason given in the
fallback rule.

**Why this is good news, and why interpolation is still genuinely required:**
V1/V2's target horizon is **10 trading days**, which spans **14 to 18
calendar days** depending on holidays (confirmed by direct calendar lookup
at four sample positions — never a fixed constant). This falls **strictly
between** the surface's two available tenors (10 and 30 calendar days), so
linear interpolation between exactly those two points is well-posed on the
data actually in the repo. No third tenor and no re-scan of the staging
grid is needed for the method itself.

**Step-by-step, locked:**

1. **ATM selection per tenor, per side.** For each `(secid, date, days)`
   with `days ∈ {10, 30}`, select the row with minimum
   `dpen = |delta|/100 - 0.50|` separately for `cp_flag='C'` and
   `cp_flag='P'` — the exact selection rule already established in
   `src/28`/`src/29`/`src/30`/`src/32`/`src/37`/`src/47` (reused, not
   reinvented). A tenor's ATM IV is used **only if both sides are present**
   (matching `src/47`'s own `has_any_atm_iv` bitmask==3 definition, so this
   pre-registration's availability criterion is the same one the
   feasibility script already checked): `IV_h := mean(impl_volatility_C,
   impl_volatility_P)` at that tenor's ATM row, for `h ∈ {10, 30}`.

2. **Convert each tenor's ATM IV to total (non-annualized) variance,**
   using the surface's own calendar-day tenor and the DECLARED ASSUMPTION
   below on OM's annualization convention:
   `TotalVar_h = IV_h² × (h / 365)`, for `h ∈ {10, 30}`.

3. **Compute the actual calendar-day span for each stock-day**, not a fixed
   constant: `h*_{i,t} = ` the real number of calendar days from date `t` to
   the date reached by V1's own 10-trading-day forward index
   (`cal[pos_t + 10] - date_t`), using the same trading-calendar position
   lookup this project uses throughout (E1's `build_paths`,
   K1's `tidx`/`cal` indexing). Confirmed by direct check: this ranges
   14-18 days across the DEV window, always strictly inside `[10, 30]`.

4. **Linear interpolation in TOTAL variance, over calendar time:**
   ```
   TotalVar(h*) = TotalVar_10 + (TotalVar_30 - TotalVar_10) * (h* - 10) / (30 - 10)
   ```
   Total variance, not the annualized rate, is interpolated linearly in
   time — this is the standard variance term-structure convention (the
   quantity that is additive under the standard forward/local-variance
   decomposition; the same logic behind CBOE's own near/next-term VIX
   interpolation). Interpolating the annualized rate directly instead would
   implicitly assume a different, non-standard term-structure shape and is
   not used.

5. **Re-annualize onto RV²'s own basis, not IV's.** This is the step most
   likely to be silently gotten wrong, so it is spelled out in full. RV² (§3.2)
   is annualized on a **252-trading-day** basis over the actual **10-trading-day**
   window. `TotalVar(h*)` from Step 4 is a basis-free absolute variance —
   the amount of variance expected to accumulate over that specific 10-trading-day
   span, however it is later expressed as a rate. To make `IV²_{i,t}`
   directly comparable to `RV²_{i,t,t+10}` (same units, same window), it is
   converted back to a rate using RV²'s own convention — 252 trading
   days/year and the 10-trading-day window, **not** `h*`/365:
   ```
   IV²_{i,t} := TotalVar(h*_{i,t}) / (10 / 252)
   ```
   Using `h*/365` here instead (i.e., staying on the options-market
   calendar-day convention throughout) would introduce a silent ~1.44x
   scale mismatch between `RV²` and `IV²` before a single coefficient is
   estimated, biasing `b1` away from 1 and biasing the `RV²-IV²` gap by a
   constant factor unrelated to any real economic difference. Locking this
   conversion now, rather than after `b1` comes out looking strange, is the
   entire point of this section.

**DECLARED ASSUMPTION (a literal, not a result — matching K1's
`RISK_FREE_RATE` precedent):** OM's `impl_volatility` is assumed annualized
on an ACT/365 calendar-day basis, consistent with `days` being calendar
days to expiration. This is the standard market/OptionMetrics convention
but is **not independently re-derived from option prices** in this
document, and is not confirmed against OM's own technical documentation
(not available in this repo). If it is wrong (e.g., ACT/360), every
`IV²_{i,t}` figure is off by a constant, known scale factor
`(assumed_days_per_year / true_days_per_year)`, which rescales `b1` but does
**not** change `b3`'s sign or its significance test, since `b3` is
identified off cross-sectional variation in compression conditional on
`IV²`, not off `IV²`'s absolute level. **Before this becomes load-bearing**:
verify against OM documentation if available; if not resolvable, run the
joint regression under both ACT/365 and ACT/360 as a sensitivity check and
report both — the same treatment K1 gave its own `RISK_FREE_RATE`
assumption (r=0 vs r=0.0252 sensitivity print).

> **[2026-08-02 — verification attempted, read with §3.3-F Resolution
> below.]** No OM documentation exists anywhere in this environment (`docs/`,
> `data/raw/optionmetrics/`, the local staging folder, and a repo-wide
> search all checked) — ACT/365 remains a declared assumption, not a
> confirmed citation, for both this sensitivity-arm construction and the
> primary's. §3.3-F Resolution turns the informal claim in this paragraph
> ("rescales `b1` but does not change `b3`") into a precise, derived result
> — `b3` is *provably* identical under ACT/365 vs ACT/360, not merely
> expected to be — and locks a required, falsifiable assertion the eventual
> build script must run and pass, rather than leaving the check as an
> unscheduled "before this becomes load-bearing" promise.

**Missing-tenor fallback — LOCKED, not extrapolated.** If a stock-day's
`(secid, date)` lacks a usable ATM IV at **either** `days=10` or `days=30`
(both sides present, per Step 1), `IV²_{i,t}` is undefined for that row and
the row is **dropped from the primary joint regression, counted and
disclosed** — not filled by extrapolating from the single available tenor,
and not filled by carrying forward a stale prior-day surface reading. This
mirrors `src/43`'s refusal to substitute `datadate` for RDQ: a method that
silently patches a missing input with a biased proxy cannot be told apart
from "no confound found" if the patch itself absorbs the effect. The exact
row-count cost of this rule is an empirical question for the feasibility
step (§6), not decided here.

### 3.3-F. FEASIBILITY FINDING (2026-08-02) — the missing-tenor fallback is not viable as written. DECISION REQUIRED.

§3.3's fallback rule above ends with: *"The exact row-count cost of this rule
is an empirical question for the feasibility step (§6), not decided here."*
That measurement has now been made. **The cost is 95.70% of the universe.**

ATM IV availability (both call and put sides present), measured against all
1,733,857 V1/V2 universe stock-days (decile 6/7/8, DEV, compression defined):

| tenor | coverage |
|---|---|
| 30-day tenor available | **91.45%** |
| 10-day tenor available | **4.30%** |
| **both 10d and 30d — what §3.3 Step 4 requires** | **4.30%** |

10-day availability is a strict *subset* of 30-day (whenever 10d exists, 30d
does too), so the joint requirement collapses to the 10-day rate. Worse, it
is severely unbalanced across the size band — by-decile both-tenor coverage
runs 0.95%–7.68%, thinnest exactly in decile 8, so the surviving 4.30% would
not be a scaled-down copy of the universe but a liquidity-skewed corner of it.

This is a **structural property of OptionMetrics' surface**, not an artifact
of `src/27`'s `KEEP_DAYS=(10,30)` filter: the 10-day standardized point is
only computed where sufficient short-dated option activity exists, and it is
6.88% of surface rows file-wide. Re-pulling the full grid from staging would
add *longer* tenors (60/91/…), which cannot help — the interpolation needs a
point at or below the ~14–18 calendar-day target, and 10-day is the only one
that exists.

**Consequence.** §3.3 as locked is internally consistent but empirically
unusable: it would run the primary specification on 4.30% of the universe.
Every option for resolving this trades against something §1 or §3.3 locked
deliberately, so none is a free fix and **none has been selected here**:

- **(A) Use the 30-day tenor directly, no interpolation.** Coverage 91.45%.
  Cost: the IV horizon (30 calendar days) no longer matches the RV horizon
  (~14–18 calendar days). §1 forbids this as a *silent* substitution; doing
  it openly, with the mismatch named in every table and the phase relabelled
  accordingly, is a different act — but it does weaken the "matched-horizon"
  claim that is this phase's headline.
- **(B) Extrapolate to h\* from the 30-day point** using a term-structure
  slope estimated on the 4.30% where both tenors exist. Cost: §3.3
  explicitly refused extrapolation, and the estimating subset is precisely
  the liquidity-skewed corner described above, so the correction would be
  fitted where it is least representative.
- **(C) Move the target horizon to ~30 calendar days** so RV and IV match at
  the tenor that actually exists. Cost: abandons V1/V2's locked 10-trading-day
  primary, which §1 requires and which is the entire basis for comparing `b3`
  back to V1/V2 and `src/43`. Note this is *not* the same as adopting K1's
  tenor — it would be a horizon-matched 30-day test — but it is a different
  question from the one V1/V2 asked.
- **(D) Restrict the phase to the 4.30% both-tenor sample** and report it as
  a narrow, explicitly non-representative test. Cost: ~74,500 stock-days,
  decile-8 coverage under 1% in some years; §6(b)'s liquid subset would then
  be a subset of an already-skewed 4.30%, and the two-universe design of §6
  would lose most of its meaning.

**This is an owner decision and is deliberately left open.** It parallels
E1's condition-1a precedent exactly: the defect was found *before* any
coefficient existed, so amending in response to it is a fix to an unusable
construction, not a reaction to an unfavourable result — and, as there, the
record of what was found and when is written down before anything is changed.

### 3.3-F RESOLUTION (2026-08-02) — option (C) selected: primary horizon moves to 30 calendar days

**Decision.** V3's primary comparison horizon moves from **10 trading days**
to **30 calendar days**, matching the surface tenor where usable IV coverage
actually exists (91.45% vs 4.30%).

**This was decided on data-availability grounds only, before any V3
regression ran.** No `b3`, no coefficient, no t-stat, no predicted-outcome
comparison across candidate horizons exists anywhere in this project at the
time of this decision — the only inputs to it are the coverage percentages
in the table above and the figure below, both descriptive of which surface
tenors exist, not of any forecasting result. The one figure produced for
this decision (`results/figures/v3_iv_tenor_coverage.png`/`.pdf`, data in
`results/47_v3_iv_tenor_coverage.json`) shows IV coverage by tenor, by year,
by decile — nothing else. No plot of coefficients, t-stats, or predicted
results across candidate horizons was produced or requested, per this
project's standing rule that a parameter is never chosen by looking at
performance (E1 §6; the same reason E1's sensitivity grid carries no
verdict authority).

**Relabeling, not deletion.** Sections 3.2 and 3.3 above are left exactly as
written — nothing in their text is edited. What changes is their *role*:
- §3.2's `RV²_{i,t,t+10}` (V1's reused column) and §3.3's interpolation
  method (steps 1–5) now describe the construction used **only** by the
  sensitivity arm defined below, not the primary.
- The primary's `RV²` and `IV²` are defined fresh, immediately below, and do
  not reuse either.

**Primary target, recomputed fresh — `RV²_{i,t,t+30cal}`.** NOT V1/V2's
column; reusing it would reintroduce the exact horizon mismatch this
resolution exists to remove.

1. For stock-day *t*, let `n_t` = the number of trading sessions strictly
   after *t* and on/before *t*+30 calendar days (a trading-calendar position
   lookup, the same convention as E1's `build_paths` and K1's `tidx`/`cal`
   indexing — inverted from the original §3.3 construction: there, the
   *session count* was fixed at 10 and the calendar span was measured; here
   the *calendar span* is fixed at 30 and the session count is measured).
2. Confirmed by direct calendar computation (pure calendar structure, no
   return/IV data touched — the same category of check as the original
   14–18-day-span confirmation): across all 1,763 DEV trading days, `n_t`
   ranges **17–22 sessions**, mean **20.67**, mode 21 (705 of 1,763 days).
   Every DEV row has a complete window available (the full trading calendar
   extends to 2025-12-31, past DEV_END+30 days).
3. `RV²_{i,t,t+30cal} := Var(DlyRet over the n_t sessions in (t, t+30cal],
   ddof=1) × 252` — V1's own `std(...) × sqrt(252)`, squared, construction
   pattern exactly, with the window boundary calendar-anchored instead of
   session-count-anchored. A window is used only if `n_t` sessions are fully
   observed (no gap); rows failing this are dropped and counted, matching
   V1's "no fallback window" discipline.

**Primary implied variance — `IV²_{i,t}`, now a direct tenor read, no
interpolation.** Since the target calendar span (30 days) now *equals* the
surface's native 30-day tenor exactly, §3.3's interpolation machinery is not
needed for the primary at all — the entire reason interpolation existed was
to bridge a 14–18-day target sitting between two tenors, and that gap no
longer exists.

1. ATM selection: identical rule to §3.3 Step 1 (minimum `dpen`, both call
   and put sides required) — applied at `days=30` only.
   `IV_30 := mean(impl_volatility_C, impl_volatility_P)` at that tenor's ATM
   row.
2. `TotalVar_30 := IV_30² × (30/365)` — same DECLARED ASSUMPTION as §3.3
   (OM's `impl_volatility` annualized ACT/365). **Verification attempted,
   2026-08-02: not resolvable.** Searched for an OptionMetrics/IvyDB data
   dictionary or manual in `docs/`, `data/raw/optionmetrics/`, the local
   Downloads staging folder, and a repo-wide search for day-count terms —
   no such document exists anywhere in this environment. ACT/365 remains a
   DECLARED ASSUMPTION, not a confirmed citation, exactly as before.

   **What is now locked instead: a precise, required check, not a deferred
   promise.** ACT/365 vs ACT/360 is a single constant multiplier
   (365/360 ≈ 1.013889) applied identically to `TotalVar_30` on every row —
   it does not vary by secid, date, or `n_t`. `IV²_{i,t}` is the **only**
   term in §4's model touched by this assumption; every other term (`RV²`,
   `PriorRV²`, `Compression`, `EarningsBucket`, all controls) is built
   purely from CRSP daily prices and carries no OM day-count dependency.
   Because `IV²` enters §4 linearly with its own free coefficient `b1`, and
   because rescaling one regressor column of a linear (or Fama-MacBeth
   cross-sectional) regression by a global constant `c` changes only that
   column's own coefficient (`b1' = b1/c`) while leaving every other
   coefficient, every residual, and hence every other coefficient's entire
   daily time series **exactly** unchanged — this is standard OLS/FM linear
   algebra, not an empirical claim to be discovered — it follows that:
   - **`b3` (and its NW t-stat, and `b2`/`b4`/`b5`/controls/alpha) are
     provably identical, to floating-point precision, whether ACT/365 or
     ACT/360 is used.** Not "expected to move little" — proven not to move
     at all, given §4's linear specification.
   - `b1` itself rescales exactly: `b1_ACT360 = b1_ACT365 × (360/365)`
     ≈ `b1_ACT365 × 0.986301`.

   **REQUIRED in the eventual V3 results reporting, not optional and not
   deferred:** the build script must compute `b3` (and its NW t-stat) under
   both conventions and assert they match to numerical tolerance, and report
   `b1` under both, confirming the 360/365 ratio holds. This is a built-in
   validation, not a hedge — if the assertion ever fails, that reveals a
   coding error (the ACT/365 assumption leaking into some part of the
   pipeline this analysis did not account for), and no V3 result may be
   reported until that is found and fixed.

   **Scope limit, stated plainly.** This invariance protects `b3` — the
   load-bearing coefficient — from the day-count choice specifically. It
   does **not** verify that either convention is correct in absolute terms,
   and it does **not** protect `b1`'s own interpretability: if `b1`'s
   magnitude is ever used for anything beyond "control for `IV²`" (e.g. "is
   realized variance explained roughly 1:1 by implied variance"), the
   ACT/365 assumption remains a real, unresolved limitation on that specific
   reading.
3. Re-annualize onto `RV²`'s basis using the **same row-specific `n_t`** the
   primary `RV²` for that stock-day uses:
   `IV²_{i,t} := TotalVar_30 / (n_t / 252)`.
   This is a refinement on §3.3's original re-annualization (which divided
   by a *fixed* 10 despite the interpolation target `h*` varying 14–18
   days) — here both sides of the comparison share the exact same
   realized session count for every row, so `RV²` and `IV²` are on a
   perfectly matched basis with no residual approximation from that source.

**Missing-tenor fallback, now much smaller.** A row lacking a usable 30-day
ATM IV (both sides) is dropped from the primary, counted and disclosed —
same discipline as before, same reasoning (no extrapolation, no stale
carry-forward). Expected cost is **~8.55%** of the universe (100% −
91.45%), not 95.70% — the change that makes the primary specification
actually runnable on close to the full universe.

**The original 10-trading-day spec is retained — as a labeled, no-verdict
sensitivity arm.** Full construction unchanged from §3.2/§3.3 as originally
written (V1's `RV²_{i,t,t+10}` column, the Step 1–5 interpolation, the fixed
`/(10/252)` re-annualization). Run on the **4.30%** both-tenor sample only.
Reported alongside the primary in every table §4 produces. **Carries no
authority over any §7 branch** — it cannot promote a primary FAIL to a pass,
and it cannot be substituted for the primary if the two disagree. This is
the identical treatment E1 gave its own sensitivity grid (prereg_E1.md §6):
reported for transparency, explicitly barred from deciding anything.

**Consequential edits made elsewhere, flagged rather than silent.** §4's
equation notation (`RV²_{i,t,t+10}`) is factually superseded by the above —
leaving it unedited would make the document self-contradictory, since it
would name a target this section just relabeled as sensitivity-only while
calling it primary. The **only** change made to §4 is the horizon subscript
in the primary equation (`t+10` → `t+30cal`), plus the addition of the
sensitivity arm's own equation immediately below it as a labeled secondary
spec. No control, no coefficient, no joint-vs-sequential reasoning, and no
part of §4's actual design was reopened or reconsidered.

**One tension was disclosed here, then closed (2026-08-02, same day).** §5
originally locked Newey-West `maxlags=10` for the whole section, whose own
stated rationale (matching V1/V2's convention) is "matching the forecast
horizon" — no longer true of a primary spec at ~20–21 trading sessions. §5
has since been corrected: `maxlags=21` for the primary, `maxlags=10`
retained for the sensitivity arm (matching its own unchanged 10-trading-day
horizon). See §5 for the full correction and its dated note.

**K1's tenor is explicitly NOT used here.** K1's straddle used a ~30-day
(nominal calendar) tenor selected against a different rule (nearest
expiration to entry+30 outliving a 10-trading-day hold). V3's target
horizon is V1/V2's **10 trading days**, full stop. If a 30-day-matched
comparison is ever wanted — e.g. to connect back to K1's straddle economics
— it is a **separate, explicitly labeled secondary test**, built and run
only on explicit instruction, never silently blended into the primary
10-trading-day specification above.

> **[2026-08-02 — read with §3.3-F Resolution below.]** "V3's target horizon
> is V1/V2's 10 trading days, full stop" was true when written and remains
> true of the construction described in this section — which, per the
> Resolution below, is now the **sensitivity arm's** construction, not the
> primary's. It is still not K1's tenor: K1 selected an actual traded
> contract's expiration against a hold-period rule; V3's new 30-calendar-day
> primary (§3.3-F Resolution) reads the surface's native 30-day standardized
> point directly. Two different 30-day things, neither silently substituted
> for the other, exactly as this paragraph originally insisted.

### 3.4 Prior realized variance control, PriorRV²

`PriorRV²_{i,t} := (prior_vol_{i,t})²`, reusing E1's exact definition
(`src/46`): trailing 20-trading-day standard deviation of `DlyRet`,
annualized ×√252, `shift(1)`-aligned to end at `t-1`. Not rebuilt with a new
10-day window. This is a **regime-level control**, not itself subject to
the horizon-matching requirement in §1 (that requirement binds the
`RV²`/`IV²` target construction, not a right-hand-side control) — a
deliberate, disclosed choice, made for continuity with the one trailing-vol
definition this project has already established, rather than inventing a
second one with different parameters.

### 3.5 Earnings buckets

`EarningsBucket_{i,t} ∈ {1-5 trading days, 6-10 trading days, 11-20 trading
days, none within 20 trading days}`, relative to the next RDQ from `t+1`.

**Reused without rebuild:** the CCM link (`ccm_link_gics.csv`, LINKTYPE
LC/LU, LINKPRIM P/C, non-null `lpermno`, date-windowed on
`LINKDT`/`LINKENDDT`, P-over-C tiebreak) and the `(gvkey, rdq)` pairs from
`data/raw/compustat/rdq_pull_fundq_2014_2026.parquet` — exactly `src/43`'s
join, unchanged.

**New, built on that join:** `src/43` computed one binary 10-trading-day
flag. V3 needs the trading-day distance to the next RDQ over a 20-session
forward window, bucketed into four categories — this bucketing logic is new
code operating on the same underlying `(gvkey, rdq)` pairs and the same
link, consistent with the task's "no rebuild needed there" referring to the
join itself, not a bucketing function that did not previously exist.

**Coverage gaps are dropped, not folded into "none."** Stock-days whose
linked gvkey has no RDQ history at all (`src/43` found this for 0.76% of V1
rows) are **excluded and counted**, exactly as `src/43` did — never
recoded as "none within 20 days." Those are two different facts: "we
checked and there is no earnings event" is not the same claim as "we don't
know," and conflating them would quietly mismeasure the bucket a firm
without options-relevant Compustat coverage falls into.

### 3.6 Other controls

Reused verbatim from E1's control set (`src/46`, D-series), not
reinvented: `log_cap = log(DlyCap)`, `log_price = log(|DlyClose|)`,
`log_dvol = log(DlyPrcVol)`, `recent_absret` (20-day trailing |return|,
`shift(1)`), `trend` (120-day trailing return, `shift(1)`). Identical
window parameters as E1 (`PRIOR_VOL_WIN=20`, `RECENT_ABSRET_WIN=20`,
`TREND_WIN=120`).

---

## 4. Primary specification — one joint regression

**Notation note (2026-08-02, forced by §3.3-F, nothing else in this section
reopened):** the horizon subscript below reads `t+30cal`, not the original
`t+10`, because §3.3-F moved the primary target to 30 calendar days. `RV²`
and `IV²` are as newly defined there. No control, coefficient, or the
joint-vs-sequential design below was reconsidered.

**Compact form, PRIMARY (30 calendar days):**

```
RV²_{i,t,t+30cal} = alpha + b1*IV²_{i,t} + b2*PriorRV²_{i,t} + b3*Compression_{i,t}
                   + b4*EarningsBucket_{i,t} + b5*(Compression x EarningsBucket_{i,t})
                   + controls + error
```

**Expanded, fully specified form** (what is actually estimated — the
compact form's `b4`/`b5` are vectors over 3 dummies, reconciled here so
nothing is left implicit):

```
RV²_{i,t,t+30cal} = alpha
   + b1 * IV²_{i,t}
   + b2 * PriorRV²_{i,t}
   + b3 * Compression_{i,t}                                    <- reference case: no earnings within 20td
   + b4a * D_earn_1to5_{i,t}   + b4b * D_earn_6to10_{i,t}   + b4c * D_earn_11to20_{i,t}
   + b5a * (Compression_{i,t} * D_earn_1to5_{i,t})
   + b5b * (Compression_{i,t} * D_earn_6to10_{i,t})
   + b5c * (Compression_{i,t} * D_earn_11to20_{i,t})
   + controls (log_cap, log_price, log_dvol, recent_absret, trend)
   + error
```

`D_earn_none` (no RDQ within 20 trading days) is the omitted reference
category. `b3` is therefore the compression effect **specifically outside
any earnings window** — the pure structural effect — and `b3 + b5k` is the
total compression effect **within** earnings bucket `k`. This mapping is
what §7's decision tree tests against directly.

**SENSITIVITY ARM (10 trading days, 4.30% both-tenor sample, NO VERDICT
AUTHORITY — §3.3-F).** Identical structure, `RV²_{i,t,t+10}` and its `IV²`
as originally defined in §3.2/§3.3, on the liquidity-skewed 4.30% sample:

```
RV²_{i,t,t+10} = alpha + b1*IV²_{i,t} + b2*PriorRV²_{i,t} + b3*Compression_{i,t}
                + b4*EarningsBucket_{i,t} + b5*(Compression x EarningsBucket_{i,t})
                + controls + error          [reported alongside the primary;
                                              cannot override any §7 branch]
```

Two full instances of the PRIMARY model are run — one per compression
variant (§3.1) — never merged into a single combined regressor. The
sensitivity arm is likewise run for both variants and reported alongside,
same non-authority in both cases.

**Why one joint model, not sequential earnings-only and IV-only scripts:**
a sequential approach (fit compression alone, separately fit an
earnings-only model, separately an IV-only model) cannot detect a
compression × earnings interaction unless it is explicitly built to look
for one. `src/43` already found exactly such an interaction is real and
significant (V1 interaction NW t = -3.191; V2 interaction NW t = -2.534),
so a sequential design here would risk repeating the same blind spot
`src/43` was built to close for the univariate case. A single joint
specification with the interaction term included by construction cannot
miss it.

**The load-bearing coefficient is `b3`.** Both its magnitude (in variance
units — annualized-variance-squared-return terms, i.e. the same units as
`RV²`/`IV²` themselves) and its NW/clustered t-stat are reported together;
significance alone does not settle anything. Magnitude is additionally
translated to a plain-language economic size: the model's implied change in
forward-10-trading-day realized volatility (not variance) for a one-decile
move in compression, holding `IV²` fixed — i.e. `sqrt(current level + b3)
- sqrt(current level)` evaluated at the sample median `RV²`, so a reader
without a variance-units intuition can still judge the size.

---

## 5. Standard errors and significance bar

**Estimator:** Fama-MacBeth — a daily cross-sectional OLS of the full
expanded specification in §4, one regression per trading day, collecting
the daily coefficient series per term. Not a pooled panel regression, for
the same reason V1/V2's own gate scripts give explicitly ("pooled panel
would understate standard errors given cross-sectional and serial
correlation") and the same design E1's Test 1 used for its own multivariate
specification.

**Inference:** Newey-West HAC t-test — **maxlags=21 for the PRIMARY
(30-calendar-day) specification**, **maxlags=10 for the SENSITIVITY ARM
(10-trading-day, §3.3-F)** — each on its own daily coefficient series,
each matching its own forecast horizon.

> **[CORRECTED 2026-08-02, alongside §3.3-F, dated together.]** This bullet
> originally read a single `maxlags=10` for the whole section, identical to
> V1/V2/`src/43`/E1 Test 1 — correct for the horizon this document had at
> the time (10 trading days everywhere). §3.3-F moved the primary horizon to
> ~20–21 trading sessions; leaving `maxlags=10` on the primary would apply
> the OLD horizon's justification to a spec that no longer has that horizon.
> This is a parameter correction to match new information about the
> forecast horizon, made before any V3 coefficient exists — not a response
> to any result. The corrected value, **21**, follows the exact same "match
> the forecast horizon" logic V1/V2 originally stated, applied to the
> horizon §3.3-F actually locked: `n_t` (§3.3-F Resolution) ranges 17–22
> sessions across DEV, mode **21** (705 of 1,763 days), mean 20.67 — 21 is
> both the mode and the mean rounded, making it the natural single scalar to
> lock for a hyperparameter that must be one fixed number applied to the
> whole coefficient time series, not a per-row quantity. The sensitivity
> arm keeps `maxlags=10` unchanged, since it still tests the original
> 10-trading-day horizon and should be internally consistent with itself,
> not with the primary.
>
> **Checked, not adopted:** E1's own Fama-MacBeth test (`src/46`, the same
> estimator design this section cites as precedent) used `maxlags=20` at
> its own 10-trading-day primary horizon — a more conservative 2×-horizon
> convention, not a 1:1 match. This section follows V1/V2's original 1:1
> "matches the horizon" logic, corrected for the new horizon, per explicit
> instruction — not E1's different convention.

A date enters the average only with a full-column-rank design
matrix and `n >= MIN_XSEC` (locked at 30, matching `src/43`'s own bar,
itself matching `src/11`/`src/18`'s established minimum).

**Significance bar:** `|NW t| >= 3.0` for `b3` and for each within-bucket
total `b3 + b5k`. This matches V1/V2's own bar and its stated rationale —
"higher bar... exploratory forecasting test without the discipline of a
cost model attached to it" — and E1's Magnitude/Direction bars. V3 is
still an information test, not yet a cost-bearing trading claim; the
eventual trading-test branch of §7 would carry its own, separate economic
bar, not decided here.

---

## 6. Two universes

### 6(a) Full V1/V2 research universe — LOCKED

The base universe from §2, unrestricted by any options-liquidity criterion.
This is the universe V1/V2 themselves were gated on.

### 6(b) Ex-ante liquid-options subset — LOCKED 2026-08-02

**Feasibility source.** `src/47_v3_data_feasibility.py`, extended and run
2026-08-02. A pre-aggregated liquidity panel was searched for first, by
column content rather than filename, across all 15 files in
`data/processed/` and `data/raw/optionmetrics/`: none exists. The single
near-match, `k1_trades_real_prices.parquet`, carries `liq_ok`/bid/ask but
only for the 5,356 stock-days K1's own straddle-selection rule picked — not
a general daily panel — and was rejected. A targeted pull was therefore
authorized and run: **one sequential pass over the 75.02 GB `opprcd.csv`
reading only `secid`, `date`, `volume`, `open_interest`** — 886,525,545 rows
scanned in 11.1 minutes, 467,078,693 kept in scope, aggregated to 3,129,623
secid-days and cached to `data/processed/opprcd_liquidity_daily.parquet` so
this is never re-scanned. Full table in
`results/47_v3_liquidity_coverage.json`.

**Measured coverage** against all 1,733,857 universe stock-days (a stock-day
with no chain at all counts as *not liquid*, not as missing — the correct
denominator here):

| criterion | stock-days | % of universe |
|---|---|---|
| any nonzero open interest | 1,596,027 | 92.05% |
| any nonzero option volume | 1,259,825 | 72.66% |
| nonzero volume **and** underlying dvol ≥ same-day median | **778,354** | **44.89%** |
| nonzero volume and dvol ≥ 60th pct | 639,866 | 36.90% |
| nonzero volume and dvol ≥ 75th pct | 414,261 | 23.89% |

**LOCKED DEFINITION.** A stock-day is in universe 6(b) if **both**:

1. **At least one listed option contract on that underlying traded that day**
   — i.e. some contract in the chain had `volume > 0` on date *t*; and
2. **The underlying's dollar volume `DlyPrcVol` is at or above the
   cross-sectional median of the V1/V2 universe on that same day** — a
   point-in-time rank computed within date *t* only.

**Reasoning, stated plainly.**

- *Why option volume, not open interest.* Nonzero OI retains 92.05% and
  excludes almost nothing — a near-vacuous filter, and the "too loose to
  mean anything" failure mode this section was told to avoid. Open interest
  can also persist on a chain nobody has traded for weeks. `volume > 0` is
  the stricter and more honest reading of "could this have been transacted
  on the day the signal fires," which is the question a liquid-universe
  screen exists to answer.
- *Why add an underlying dollar-volume cut.* Option volume alone still keeps
  72.66% — usable, but loose enough that the "liquid" and "full" universes
  would be near-duplicates, and §7's decision tree depends on those two
  universes being able to disagree. Adding the same-day median `DlyPrcVol`
  cut brings it to 44.89%: not near-0%, not near-100%, and unambiguously a
  different population from 6(a).
- *Why the median specifically, not the 60th or 75th.* The median is the
  least aggressive cut that achieves a genuinely distinct universe, and it
  is the only one of the three that is not itself a tunable dial — "the
  liquid half of the universe, by its own daily standard" is a criterion
  fixed by definition rather than chosen from a menu. The 60th and 75th
  percentiles are recorded above as pre-committed sensitivity variants with
  **no authority over any §7 branch**; they exist so a reader can see the
  threshold's neighbourhood, not so a better-looking one can be adopted later.
- *Why point-in-time and cross-sectional.* A fixed dollar level (e.g.
  "≥ $5M/day") would drift with the market over 2015–2021 and mechanically
  grow the liquid subset in later years. A same-day cross-sectional rank is
  stable by construction and uses only information available at *t*, matching
  how V1 and V2 already define every decile in this project.

**Non-vacuity confirmed, and stable.** By year the subset holds 42.57%
(2015) to 46.52% (2021) — a 3.95 pp spread across seven years, so it is not
an artifact of any one period. By size decile it is 69.62% / 48.12% / 24.01%
for deciles 6 / 7 / 8: the filter genuinely discriminates on liquidity, as
intended, while still retaining 162,186 decile-8 stock-days, so no decile
collapses toward zero. The thinnest single year×decile cell retains 22.22%
(18,149 stock-days), comfortably estimable under §5's `MIN_XSEC = 30`.

**No outcome could have influenced this.** No V3 regression has been run and
no `b3` exists. The threshold was selected against coverage counts only, on
the single stated criterion of non-vacuity.

**One locked category was dropped: the bid-ask spread bucket.** The draft
§6(b) named a maximum spread bucket as a third criterion. The authorized
pull was explicitly limited to `secid, date, volume, open_interest`, so
`best_bid`/`best_offer` were not read and no spread quantity exists to
threshold on. Rather than widen the authorized scope unilaterally or quietly
leave a criterion in the definition that nothing computes, it is **removed
from the 6(b) definition and recorded here as removed**. This is also the
more defensible placement on the merits: a spread screen is a *transaction-cost*
filter, and V3 is an information-content test that charges no costs — spread
belongs in the eventual trading test of §7's first branch, where K1 and
`src/42` already showed it is decisive. Adding it back would require a
further authorized pull and a dated amendment to this section.

---

## 7. Outcome decision tree — locked now, mapped to exact model quantities

Each branch below is tied to a specific, named coefficient or coefficient
combination from §4, so no reinterpretation is possible after the numbers
exist.

| Outcome | Tested quantity | Consequence |
|---|---|---|
| **b3 survives in both universes** | `b3` significant (|NW t| ≥ 3.0, correct sign) with `IV²` controlled, in **both** 6(a) and 6(b) | Proceed to a real trading test: rank by predicted `RV² - IV²`, delta-hedged calls/puts traded **separately** (per K1/`src/42`'s finding that the straddle's directional exposure, not its volatility bet, was doing most of the work) — no return to straddles. |
| **b3 survives only in the no-earnings bucket** | `b3` significant, but the within-bucket totals `b3 + b5a`, `b3 + b5b`, `b3 + b5c` are **not** significant | Compression is a structural, non-event volatility finding. Reported as such. Does **not** proceed to a trading test without further work characterizing the structural mechanism. |
| **b3 survives only in the illiquid subset** | `b3` significant in 6(a) but **not** in 6(b) | Real but inaccessible finding. Close the volatility-trading branch. Report honestly. No strategy is built on it. |
| **b3 does not survive controlling for IV²** | `b3` not significant (or wrong sign) once `IV²` is in the model, in either universe | The options market already prices this. Close the pure long-volatility branch entirely. **This does not close the separate EMA-entanglement (E1) directional branch**, which does not depend on this result and is evaluated on its own terms. |

If more than one row's condition is literally satisfied (e.g. b3 fails
everywhere, which trivially satisfies both "illiquid-only" read as
vacuously false and "does not survive" as true), the **last-listed
applicable row takes precedence** — i.e. "does not survive controlling for
IV²" is checked first and, if triggered, ends the evaluation without
consulting the bucket- or liquidity-conditioned rows, since a coefficient
that is not significant in the pooled full-universe test cannot be
meaningfully partitioned further. This ordering is stated now specifically
to avoid an after-the-fact choice among rows once real numbers exist.

---

## 8. Non-actions, disclosures, and status

- No coefficient, correlation, p-value, or t-stat computed on real
  return, volatility, or IV data anywhere in producing this document, at
  any point across its drafting, feasibility, or resolution passes. Every
  real-data fact used is a coverage/schema/calendar fact: surface tenor
  values and their row-share, the liquidity pull's coverage counts, and
  two pure trading-calendar computations (the 14–18 calendar-day span of a
  10-trading-day window; the 17–22 session count of a 30-calendar-day
  window). None depends on a return, an RV, an IV level, or a regression
  outcome.
- `gate_log.md` was not opened for writing at any point in this document's
  production. `src/47`/`src/48` compute coverage counts and a descriptive
  figure only.
- **Resolved:**
  1. §6(b)'s liquidity thresholds — locked on real coverage data (§6(b)).
  2. §3.3-F, the missing-tenor problem — resolved 2026-08-02 by moving the
     primary horizon to 30 calendar days, decided on data-availability
     grounds alone, before any V3 outcome existed (§3.3-F Resolution). The
     original 10-trading-day construction is retained as a labeled,
     no-verdict sensitivity arm, matching E1's sensitivity-grid treatment.
  3. **§5's `maxlags` — corrected 2026-08-02, alongside §3.3-F.** Primary
     now `maxlags=21` (matching the ~20–21 session horizon §3.3-F locked;
     21 is both the mode and rounded mean of the `n_t` distribution),
     sensitivity arm keeps `maxlags=10` (matching its own 10-trading-day
     horizon). Same "matches the forecast horizon" logic V1/V2 originally
     stated, corrected for the horizon that logic now applies to — not a
     response to any result, since no coefficient exists yet.
  4. **ACT/365 — verification attempted 2026-08-02, not resolvable; a
     precise required check locked instead of a deferred promise.** No OM
     documentation exists anywhere in this environment (checked: `docs/`,
     `data/raw/optionmetrics/`, the local staging folder, a repo-wide
     search). ACT/365 remains a declared assumption. But because `IV²` is
     the only §4 term the assumption touches, and it enters linearly with
     its own free coefficient `b1`, standard OLS/Fama-MacBeth linear
     algebra makes it a *provable* fact — not an empirical one to be
     discovered — that `b3` (and its NW t-stat, and every other coefficient)
     is exactly identical under ACT/365 vs ACT/360; only `b1` rescales, by
     exactly `360/365`. The eventual build script is **required** to compute
     both and assert the match — a built-in validation, not a hedge: a
     failed assertion would reveal a coding error, not a modeling choice.
     Full derivation in §3.3-F Resolution.
- **Disclosed, correctly left open — neither is an outcome-blind decision
  requiring a lock, unlike the four items above:**
  1. If a spread criterion is wanted back in §6(b), it needs a further
     authorized `opprcd` pull for `best_bid`/`best_offer` (see §6(b)).
  2. Universe 6(a) is the full V1/V2 universe as specified, but any row
     lacking a usable primary `IV²` is dropped (§3.3-F: ~8.55% expected, a
     structural improvement on the pre-resolution 95.70%). The *estimable*
     6(a) sample is therefore smaller than 1,733,857; report both counts
     side by side when V3 runs, so the attrition stays visible.
- **Status: LOCKED, 2026-08-02.** Every section is locked, including both
  items resolved in this pass. §3.2's original text is unedited and now
  describes the sensitivity arm; §3.3's original text is unedited, with two
  dated pointer annotations (ACT/365 verification, K1-tenor clarification)
  added rather than any rewrite; §4 carries the one forced notation change
  (§3.3-F); §5 now carries two horizon-matched `maxlags` values instead of
  one. This phase is now fully runnable — nothing is pending an owner
  decision.
