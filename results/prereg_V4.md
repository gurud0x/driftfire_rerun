# Phase V4 Pre-Registration — Monetization Test (Pure-Volatility Branch)

**Status: LOCKED — 2026-08-03.** The final-reconciliation decisions are
applied throughout (fixed 30-calendar-day exit, [40,60] DTE band at anchor
50, $100,000 NAV, 10% spread ceiling, calls-primary/puts-secondary), and
both items that previously blocked lock — §7.8's pre-entry quote rule and
§8.2's dynamic liquidity cap — are resolved and confirmed (§14). No V4
script exists. No return, coefficient, P&L, t-stat, or trade has been
computed anywhere in this document's production. `results/gate_log.md` has
not been opened for writing. The eventual build is gated on §13's data pull
being authorized, not on anything further in this document.

**Phase type:** monetization / alpha claim (K-series lineage, V-series
signal), DEV window only.
**Scripts:** none built. The eventual build is expected to occupy
`src/50_*` onward and remains **blocked until the data pull in §13 is
authorized** — this document's own lock is no longer the blocker.
**Lineage:** V3 (`results/prereg_V3.md`, LOCKED 2026-08-02;
`src/49_v3_incremental_variance_test.py`, run 2026-08-03, logged PASS —
decision-tree **row 1**, both compression variants, both universes). Also
inherits from K1 (`docs/PhaseK1_PreRegistration_StraddleOnCompression.md`,
GATE FAIL dev), `src/42_delta_hedged_decomp.py`, `src/41_spread_terminology.py`,
`src/43_earnings_confound.py`, and E1 (`results/prereg_E1.md`).

**Real-data contact in producing this document — disclosed, and it is one
item.** Two CSV *headers* were read with `nrows=0`
(`opprcd.csv`, `vol_surface_full_grid.csv`), returning column names and
zero rows. This is the same category of check V3 §3.3 made before locking
its own IV construction ("structural facts about the surface file were
checked before locking this — schema/coverage, not a regression result").
It was done because §6's delta-source lock is not writable without knowing
whether the contract-level file carries a vendor delta. **It does not** —
see §6.1. No other file was opened. Nothing else in this document rests on
a number that was not already logged in `results/` or `docs/` before today.

---

## 1. Background — what is actually established, stated precisely

This section exists because three prior findings have been described
loosely elsewhere in the project's history, and V4's design depends on all
three being stated correctly. Each correction below is a factual fix, not a
framing preference.

### 1.1 V3 established INCREMENTAL INFORMATION, not mispricing

V3's PASS means: controlling for contemporaneous implied variance `IV²`,
prior realized variance, earnings-timing buckets, and the D-series
controls, the compression decile still carries a significant, correctly
signed coefficient on forward 30-calendar-day realized variance.

| spec | universe | `b3` | NW t (maxlags 21) | dates | mean cross-section |
|---|---|---|---|---|---|
| V1 `compression_decile` | 6(a) full | −0.005608 | −7.051 | 1,568 | 908.5 |
| V1 `compression_decile` | 6(b) liquid | −0.005459 | −5.844 | 1,424 | 445.3 |
| V2 `sector_rel_decile` | 6(a) full | −0.005400 | −7.490 | 1,568 | 888.0 |
| V2 `sector_rel_decile` | 6(b) liquid | −0.005110 | −6.024 | 1,419 | 438.8 |

**What that does and does not license.** A significant compression
coefficient *conditional on measured `IV²`* means compression contains
predictive content about forward realized variance **that is not summarized
by measured implied variance**. It does **not**, by itself, establish that
the options market misprices anything. At least four readings survive V3
intact:

1. `IV²` is measured with error — it is read off OptionMetrics' fitted
   standardized 30-day surface point (V3 §3.3-F Resolution), not off a
   traded contract. Classical measurement error in a regressor attenuates
   its own coefficient and leaves room for a correlated regressor to pick
   up the residual signal, with no mispricing involved.
2. Quotes can be stale or wide on exactly the names where compression is
   most extreme, so "the market has not priced it" and "the market has not
   quoted it tightly" are not separable in V3's design.
3. The relationship may be a **volatility risk premium** that varies with
   compression — a compensated exposure, not an error. K1 §4 already
   disclosed the unconditional version of this: matched-sample mean IV
   0.6105 against V1 decile-1 mean realized vol 0.4656, a ~23.7% structural
   premium. V3 did not test whether the premium *varies* with compression.
4. Even a genuine forecast improvement need not survive the cost of
   expressing it. K1 is this project's own demonstration of that.

**Therefore: "incremental information" is the only claim V3 supports, and
this document uses that phrase for V3 throughout. The words "mispricing",
"tradeable edge", and "alpha" are reserved exclusively for what V4 itself
might establish, and appear nowhere in this document as a description of
V3.**

### 1.2 `src/42`'s delta-hedge decomposition — stated correctly

This was previously mischaracterized. The correct statement, from
`results/42_delta_hedged_decomp.json` (5,327 trades processed of the 5,356
locked K1 trades; means as % of entry mid):

| component | value |
|---|---|
| unhedged option P&L, mid-to-mid | **+3.8345%** |
| (a) delta-hedged gross | **+8.3636%** |
| (b) directional P&L *removed* by the hedge | **−4.5291%** |
| (c) option spread cost | **−62.7408%** |
| (d) hedge cost @ 15 bps | −2.8002% |

**Delta hedging INCREASED K1's average return, from +3.83% unhedged to
+8.36% delta-hedged.** The directional component the hedge removed was a
**drag of −4.53%** — opposite in sign to the unhedged gross and larger in
magnitude than it. `src/42`'s own pre-specified pattern flag recorded
**NEITHER**: neither "the vol forecast was real, spread was the killer"
(Pattern A) nor "the straddle was mostly catching direction" (Pattern B).
Directional share of unhedged gross: **−118.1%**. Hedged share: **+218.1%**.

**K1 did not fail because it was secretly a directional bet.** It failed
for two other reasons, both visible in the same artifact:

- The delta-hedged calendar-time return was **statistically unreliable**:
  NW t = **−0.409** on the daily series (−0.438 in RF-excess terms), i.e.
  a nominally positive +8.36% per-trade mean that is indistinguishable
  from zero once the overlapping daily series is used for inference. The
  per-trade mean and the calendar-time t-stat point in opposite directions
  — which is exactly why §9 of this document reports trade-level and
  portfolio-level results separately and never conflates them.
- It was **overwhelmed by option spreads**. `src/41_spread_terminology.py`
  established the authoritative figure and its meaning: the entry-side
  **half**-spread was a median **18.75%** of mid (mean 29.80%), i.e. a
  median **full** spread of ~37.5% of mid, at the combined straddle level,
  paid *twice* on a round trip. Charging just 25% of that spread turns the
  +8.36% into **−7.32%** (NW t −10.69); charging all of it gives −54.38%.

Both facts drive V4's design directly: §9 makes a conservative execution
tier the pass criterion rather than a diagnostic, and §5.3 eliminates the
exit-side option spread entirely by holding to expiration.

### 1.3 Delta-hedged calls and delta-hedged puts are NOT a directional strategy

A delta-hedged long call and a delta-hedged long put each have most of
their first-order directional exposure removed by construction. Under
put–call parity they are, at the same strike and expiration, closely
related volatility trades: the parity relation `C − P = S − Ke^{−rT}` is
itself a pure stock-plus-bond position, so the two hedged option positions
differ only by that (hedged-away) linear piece plus the frictions of
actually trading them.

**V4 tests them separately, and the justification is explicitly NOT "calls
are bullish, puts are bearish."** It is:

- **Skew.** Equity index and single-name volatility surfaces are not flat
  in strike. At matched |delta| the call side and put side sit at different
  points on the smile and therefore embed different implied variance, so
  the same score can correspond to a different actual edge on each side.
- **Spread.** Relative bid–ask spread differs systematically between the
  call and put side of the same chain. Given §1.2, spread is the single
  most decisive cost in this universe, so a design that averaged the two
  sides would hide the one quantity most likely to determine the verdict.
- **Liquidity.** Volume and open interest are not symmetric across the
  chain; the binding liquidity screens in §6.3 will admit different contract
  populations on each side.
- **Quote quality.** `opprcd`'s whole-file diagnostics already logged 23.07%
  of contract-days with a **zero bid** and 49.78% with zero volume *and*
  zero open interest. There is no reason to assume that defect rate is
  symmetric across call and put.

**V4 is the PURE-VOLATILITY branch.** A future **DIRECTIONAL branch** —
unhedged calls or puts entered after a validated breakout trigger, which
is a different instrument from a delta-hedged option, not a variation on
one — is a separate phase with its own pre-registration. It is not
anticipated, leaned toward, or blended into V4's pass criteria or language
anywhere in this document. Nothing in V4's verdict, whatever it is, may be
cited as evidence for or against that branch.

### 1.4 Why straddles are excluded — corrected rationale

Straddles are not used in V4. The reason is **not** that the straddle was
"contaminated by direction" — §1.2 shows it was not; removing the
directional component *improved* the result. The reasons are:

1. **Two-leg trading cost.** A straddle pays the bid–ask spread on two
   contracts at entry. §1.2 shows spread was decisive at K1's scale, so
   halving the number of legs is the single largest controllable cost
   reduction available.
2. **The K1 spread was structurally uneconomic for the tested universe.** A
   median full spread of ~37.5% of mid on the combined straddle, paid
   round-trip, is not a cost that a volatility forecast of the magnitude
   V3 measured can plausibly overcome. §6.3's contract-level spread cap
   (C5) and §9.1's ex-ante cost filter exist specifically so V4 never
   enters a position whose modeled cost already exceeds its modeled edge.
3. **Separating the legs is informative.** Running calls and puts as
   separate books lets V4 observe **where the variance information actually
   resides** — call side, put side, or both. A straddle averages that away
   by construction and cannot answer it.

### 1.5 Economic size of V3's coefficient — correct nonlinear conversion

Any plain-language restatement of a variance coefficient in volatility
points must use

```
delta_sigma = sqrt(sigma^2 + delta_var) - sigma
```

evaluated at a **named** representative volatility level `sigma`, where
`delta_var` is the change in annualized variance. A variance coefficient is
never linearly restated as volatility points, and is never reported as one
blended number standing in for the whole universe.

Three named levels are used throughout this document, each anchored to a
number this project has already logged:

| label | `sigma` | anchor |
|---|---|---|
| quiet | 0.25 | low end of the mid-cap range in this universe |
| typical | 0.40 | ≈ sqrt(median RV²) in V3's 6(b) liquid universe = 0.4058 |
| high-vol | 0.65 | ≈ K1's matched-sample mean IV 0.6105; `src/42` entry-IV median 0.579, mean 0.667 |

**V3's `b3`, converted (one decile toward more compressed → variance rises
by |b3|):**

| spec | quiet 0.25 | typical 0.40 | high-vol 0.65 |
|---|---|---|---|
| V1, 6(a) full (`b3` = −0.005608) | +1.098 pp | +0.695 pp | +0.430 pp |
| V1, 6(b) liquid (`b3` = −0.005459) | +1.069 pp | +0.677 pp | +0.419 pp |
| V2, 6(a) full (`b3` = −0.005400) | +1.058 pp | +0.669 pp | +0.414 pp |
| V2, 6(b) liquid (`b3` = −0.005110) | +1.002 pp | +0.634 pp | +0.392 pp |

**Full decile-10 → decile-1 span (9 deciles, variance rises by 9·|b3|):**

| spec | quiet 0.25 | typical 0.40 | high-vol 0.65 |
|---|---|---|---|
| V1, 6(a) full | +8.612 pp | +5.878 pp | +3.773 pp |
| V1, 6(b) liquid | +8.411 pp | +5.730 pp | +3.675 pp |
| V2, 6(a) full | +8.332 pp | +5.673 pp | +3.637 pp |
| V2, 6(b) liquid | +7.938 pp | +5.386 pp | +3.447 pp |

**Why the transform is not cosmetic.** At the one-decile increment the
nonlinearity is negligible (the increment is small relative to `sigma²`),
but at the decile-spread magnitude that actually corresponds to a
long/short sort it is not. Scaling the one-decile volatility figure by 9 —
the linear restatement — overstates the true span effect by **+1.209 pp
(14.4%) at `sigma` = 0.25**, +0.359 pp (6.3%) at 0.40, and +0.092 pp (2.5%)
at 0.65 (V1, 6(b)). The error is largest exactly where volatility is
lowest, which is where the compression signal is by construction most
concentrated. A separate and worse error — reading `b3` = −0.0055 as "0.55
volatility points" — is dimensionally wrong and is not used anywhere.

**V3's own logged figure, placed correctly.** V3 reported a single
`delta_vol_annualized` per cell (e.g. −0.008050 for V1 6(a)), evaluated at
`sqrt(median RV²)` = 0.3524 — a correct application of the same transform,
but at one blended sample-median level. V4 supersedes that presentation
with the named-level grid above, and requires the same treatment for every
economic-size statement V4 itself produces.

---

## 2. Central question

> **Among liquid option contracts, do stocks with the greatest predicted
> realized-variance excess over implied variance generate positive
> delta-hedged returns after option execution and stock-hedging costs?**

This is the first question in the project that could establish a
**tradeable edge** in the compression signal. V1/V2 established forecasting
power; `src/43` showed it survives an earnings-timing confound at roughly
half magnitude; V3 established information incremental to implied variance
(§1.1); K1 attempted monetization through a different instrument and failed
(§1.2). V4 is the pure-volatility monetization test.

---

## 3. Window and holdout disclosure

| | |
|---|---|
| **DEV** | 2015-01-01 → 2021-12-31 |
| HOLDOUT | 2022-01-01 → 2025-12-31 — **not used in V4** |

**V4 runs on DEV only. No V4 script may contain a holdout code path.**

**Holdout disclosure, recorded now (E1 §3.1 precedent).** The 2022–2025
window has already served as the holdout for **five** phases: R1, R2, V1,
V2, and K1. Per `docs/CLAUDE.md` that holdout is spent. V4 makes **no
holdout claim**. Every V4 result is a DEV-window result and must be
described as such. If a V4 holdout pass is ever proposed it must be
disclosed at that time as a **sixth look at the same calendar period, not a
fresh out-of-sample test**.

**Base universe:** `data/processed/universe_membership.parquet`, NYSE
size deciles 6–8, monthly point-in-time — identical to
R1/R2/V1/V2/K0/K1/E1/`src/43`/V3. No new universe is constructed.

**Options-data end date.** OptionMetrics coverage runs 2015-01-02 →
2025-08-29 (K1 §3). This is irrelevant to a DEV-only phase but is recorded
so a future holdout proposal does not discover it late.

---

## 4. Signal construction — out-of-sample by construction

### 4.1 The forecast is of `RV²`, and it must never be fit on the rows it scores

V4 forecasts `RV²_{i,t,t+30cal}` — V3's primary target, constructed exactly
as V3 §3.3-F Resolution defines it (calendar-anchored 30-day window,
`n_t` sessions, `Var(DlyRet, ddof=1) × 252`, complete windows only). No new
target is built.

**Model, locked:** V3's own primary specification, unchanged in form —

```
RV2_{i,t,t+30cal} = alpha
   + b1 * IV2_{i,t}
   + b2 * PriorRV2_{i,t}
   + b3 * Compression_{i,t}
   + b4a * D_earn_1to5 + b4b * D_earn_6to10 + b4c * D_earn_11to20
   + b5a * (Compression x D_earn_1to5)
   + b5b * (Compression x D_earn_6to10)
   + b5c * (Compression x D_earn_11to20)
   + controls (log_cap, log_price, log_dvol, recent_absret, trend)
   + error
```

Every term reuses V3's locked definition verbatim (V3 §3.1, §3.3-F, §3.4,
§3.5, §3.6). No regressor is added, removed, rescaled, or re-parameterised.
`IV²` is retained as a forecast input despite also appearing in the score;
this is deliberate and is discussed in §4.4.

**Compression variant — one primary, locked now.** V3 ran V1's
`compression_decile` and V2's `sector_rel_decile` as co-equals ("both
reported, neither privileged") because it was an information test. V4 is a
verdict-bearing trading test, and running two co-primary variants would be
two shots at the same bar. **V1's `compression_decile` is the V4 primary**,
because it is the original locked signal and is the one K1 and `src/42`
were built on, preserving comparability with the project's only prior
monetization attempt. **V2's `sector_rel_decile` is a labeled sensitivity
arm with no verdict authority**, reported alongside the primary in every
table, and unable to promote a primary FAIL to a pass — the identical
treatment E1 §6 and V3 §3.3-F gave their own sensitivity arms.

~~**Model B — compression-free benchmark, REQUIRED (item 9 of the
return-blind audit, `results/52_v4_compression_benchmark_design.md`, full
specification there).** A second forecast, fit through the identical
expanding-window machinery of §4.2, on the identical training rows, with
`Compression` and its three earnings interactions (`b3`, `b5a`, `b5b`,
`b5c`) **dropped from the design matrix before fitting** — not Model A's
coefficients with those terms zeroed post-hoc:~~

```
[SUPERSEDED FORM - RV2 = alpha + c1*IV2 + c2*PriorRV2 + c4a/b/c*EarnBucket
 + controls(log_cap,log_price,log_dvol,recent_absret,trend) + error]
```

> **[2026-08-04 — AMENDMENT.]** Model B's five-term form above is
> **superseded** by the 10-feature benchmark specified in
> `results/61_v4_gate6_benchmark_model_spec.md` (1-day RV, 20-session
> persistence/autocorrelation, 20/60-day RV, 5-day downside variance,
> daily-proxy jump variance, vol-of-vol, ATM IV, IV term structure [pending
> feature-8 coverage confirmation], and 30-35-delta put skew — squared into
> variance units per that document's explicit signed-square step). This is
> not a narrower revision — the new Model B is a materially richer,
> standard volatility-forecasting feature set, superseding the five-term
> placeholder everywhere the term "Model B" is used in this document,
> including as the input to **Gate 6b (§11 item 6, unchanged, portfolio-
> level, full-cost)**. `Score_B_{i,t} := predicted_RV2_B - IV2_{i,t}`
> (§4.3's basis) is retained unchanged as a formula; only what feeds
> `predicted_RV2_B` changes.
>
> **A new, prior gate is added ahead of Gate 6b: Gate 6a** (signal-level,
> DEV-only — does compression carry incremental predictive power, NW
> t-stat and incremental R², over this same 10-feature benchmark, with no
> portfolio simulation required). Gate 6a is a go/no-go diagnostic for
> whether building Gate 6b's full hedging simulation is worthwhile; it does
> not itself constitute a V4 PASS and cannot substitute for Gate 6b.
>
> Model B still shares every regressor structurally analogous to Model A's
> non-compression terms — the reasoning in the paragraph below (why
> sharing regressors isolates compression's own contribution) is unchanged
> in spirit and now applies to the larger shared set.

Model B shares every non-compression regressor with Model A —
including `log_cap`, `log_dvol`, `log_price`, and `IV2` itself — precisely
so that any tendency of the score to favor cheap or liquid names for
reasons unrelated to compression is captured identically by both models;
see the design doc §3 for the full argument. Model B feeds a second,
otherwise-identical run of §6–§9's entire pipeline (same universe,
contract selection, hedging, execution, sizing, portfolio limits — the
design doc §2 confirms this item by item), producing a second portfolio,
Book B. **§11 item 6 (Gate 6b) makes Book A's incremental performance over
Book B a required gate, not a diagnostic.**

### 4.2 Estimation scheme — LOCKED before any scoring

**Fitting on the full DEV sample and scoring those same observations is
explicitly forbidden.** That would make V4 an in-sample test wearing a
trading test's clothes, and no result produced that way may be reported as
a V4 outcome.

Locked scheme:

| element | locked value |
|---|---|
| scheme | **expanding window** |
| refit frequency | **monthly**, on the first trading day of each calendar month |
| burn-in | **24 months** of training data before the first forecast |
| first scored date | first trading day of **2017-01** (first refit after burn-in) |
| last scored date | last DEV entry date with a complete 30-calendar-day forward window inside DEV |
| coefficient used | **mean of the daily Fama-MacBeth cross-sectional coefficient vectors** over the training window |
| embargo | see below — mandatory |

**The embargo is the part most easily gotten wrong, so it is spelled out.**
`RV²_{i,t,t+30cal}` is not observable until 30 calendar days after `t`. A
model refit on date `tau` may therefore train only on stock-days `t'`
satisfying

```
t' + 30 calendar days  <=  tau - 1 trading day
```

Any training row whose forward window extends to or past `tau` is excluded.
Without this, a model refit on `tau` would be fit on realized variance
that had not yet happened at `tau` and every downstream "out-of-sample"
claim in this phase would be false. **The build script must assert this
condition row-wise at every refit and halt on violation** — a required,
falsifiable check in V3 §3.3-F's style, not a comment.

**Why expanding, monthly, FM-mean:**
- *Expanding rather than rolling*: the daily FM cross-sectional coefficients
  are noisy (V3's own daily series is reported in
  `results/49_v3_incremental_variance_test.json`); an expanding window uses
  all admissible history and produces a more stable coefficient vector.
  A **60-month rolling window** is a labeled robustness arm with no verdict
  authority.
- *Monthly refit*: matches the monthly point-in-time cadence of the base
  universe, and refitting daily would multiply compute for a coefficient
  vector that is an average over hundreds of cross-sections and moves
  slowly by construction.
- *FM-mean rather than pooled OLS*: reuses V3's estimator exactly, so the
  forecast coefficient vector is directly comparable to V3's logged `b`
  vector, and avoids the pooled-panel standard-error objection V1/V2 stated
  explicitly. **Pooled OLS is a labeled robustness arm with no verdict
  authority.**
- *24-month burn-in*: gives roughly 500 daily cross-sections before the
  first forecast, and leaves 2017-01 → 2021-12 (~5 years, ~1,250 trading
  days) as the scored period. Chosen for estimation stability, before any
  V4 return exists; not swept.

A training cross-section enters the FM mean only with a full-column-rank
design matrix and `n >= MIN_XSEC = 30` — V3 §5's own bar, unchanged.

### 4.3 The score

```
Score_{i,t} := predicted_RV2_{i,t,t+30cal} - IV2_{i,t}
```

Both terms are annualized variance on the **252-trading-day basis over the
row's own `n_t`-session window**, per V3 §3.3-F Resolution step 3
(`IV2 = TotalVar_30 / (n_t / 252)`). They are on a matched basis by
construction; the build script must assert the basis alignment.

`Score` is signed and is used for three distinct purposes, which are kept
separate throughout:

1. **Ranking** candidates for the traded book (§8).
2. **Bucketing** for the high-vs-low contrast (§11 item 3) and the
   monotonicity report (§12).
3. It is **not** the eligibility test. Eligibility is §9.1's cost filter,
   which uses the traded contract's own quoted IV, not the surface `IV²`.

**V4 trades the LONG-volatility side only.** The central question asks
about the *greatest predicted excess* of realized over implied variance,
which is a long-volatility position. A short-volatility book (selling
delta-hedged options in the bottom score bucket) is **not** traded in V4:
it is short gamma with unbounded loss, carries margin and assignment
mechanics this project has never modeled, and would be a different risk
object requiring its own pre-registration. The bottom bucket appears only
as a *reported* comparison in §11 item 3 and §12, never as a traded
position. A short-volatility phase, if ever wanted, is separate.

### 4.4 Disclosed: `IV²` appears on both sides, and what that does

`IV²` is a forecast input (with free coefficient `b1`) and is also
subtracted to form the score. This is intentional and its consequences are
stated now, before any result:

- V3 measured `b1` ≈ 0.49 (6(a)) to 0.61 (6(b)) — well below 1. Implied
  variance therefore over-predicts realized variance in this universe, the
  regression-based expression of the volatility risk premium K1 §4
  disclosed unconditionally.
- Consequently `Score` will be **negative for most rows**, and *strongly*
  negative for high-IV rows, purely from `b1 < 1`. The score's
  cross-sectional ranking is therefore partly a low-IV tilt, not purely a
  compression signal.
- This is not a defect to be corrected mid-phase — it is the honest output
  of V3's own locked model. But it means the top score bucket may be
  populated by low-IV names, and §9 therefore **requires** reporting the
  entry-IV distribution and the compression-decile distribution of the
  traded book, so that a passing result can be attributed and a reader can
  see whether V4 monetized compression or merely monetized low implied
  volatility.
- **SUPERSEDED, item 9 of the return-blind audit
  (`results/52_v4_compression_benchmark_design.md`).** An earlier draft of
  this bullet specified a no-authority diagnostic built by forcing a
  fitted `b3` to zero post-hoc. That is replaced by a properly separate
  benchmark model — see §4.1's Model B and §11 item 6, which is now a
  **required gate, not a diagnostic**: zeroing one coefficient after
  fitting leaves every other term frozen at values estimated *with*
  compression present, which understates how much of Model A's edge a
  genuinely compression-free model could recover through its shared
  controls. Model B is fit independently instead, so it is not vulnerable
  to that understatement.

---

## 5. Holding period — V3's 10-day arm addressed directly

### 5.1 What V3's 10-day arm actually shows

V3's sensitivity arm at the 10-trading-day horizon was **not significant**.
The full numbers, which matter here:

| arm | `b3` | NW t | dates | mean cross-section | implied NW SE |
|---|---|---|---|---|---|
| V1 primary, 30 cal, 6(a) | −0.005608 | −7.051 | 1,568 | 908.5 | 0.000795 |
| V1 sensitivity, 10 td, 6(a) | **−0.006796** | −0.918 | 424 | 45.2 | 0.007405 |
| V2 primary, 30 cal, 6(a) | −0.005400 | −7.490 | 1,568 | 888.0 | 0.000721 |
| V2 sensitivity, 10 td, 6(a) | **−0.007567** | −1.123 | 422 | 44.8 | 0.006740 |

**The 10-day arm's point estimate is LARGER in magnitude than the
primary's, not smaller.** It is insignificant because its standard error is
roughly **9.3×** wider (V1: 0.007405 vs 0.000795; V2: 0.006740 vs 0.000721),
which is what a sample carrying **1.35%** of the primary's observations
(424 × 45.2 vs 1,568 × 908.5) produces. That arm ran on the 5.35%
both-tenor subsample V3 §3.3-F documented as severely liquidity-skewed
(by-decile coverage 0.95%–7.68%, thinnest in decile 8).

**Correct reading, stated before any V4 result exists:** the 10-day arm is
**uninformative about the 10-day horizon**. It is not evidence that the
effect is absent at 10 days, and it is not evidence that it is present.
V3's own §3.3-F Resolution gave it no verdict authority for exactly this
reason. Treating it as a negative finding about short horizons would be
reading a precision failure as a substantive one.

### 5.2 Locked holding period, and the reasoning for not adding a short arm

**PRIMARY holding period: a FIXED 30 calendar days (§5.3), exiting at the
last trading session on or before entry + 30 calendar days.** The realized
hold is therefore **17–22 trading sessions** (mean 20.67, mode 21) — V3's
own `n_t` distribution, matched exactly. This is the horizon at which V3's
evidence exists, and V4 matches it.

> **REVISED at the final reconciliation.** This paragraph previously read
> "hold the option to its expiration, where the contract is selected to
> expire as near as possible to entry + 30 calendar days, within [25, 38]
> calendar days." Both halves are superseded: the exit is now
> calendar-fixed rather than expiry-driven (§5.3), and the entry band is
> [40, 60] (§7.1). The *holding period* is unchanged in length and still
> matches V3's forecast horizon; what changed is that the option is now
> **sold** at that point rather than held to expiry.

**No shorter-horizon trading arm is included. Reasoning, stated now:**

1. **There is no validated shorter horizon to test.** V1/V2 established
   10-day forecasting power, but V3 is the phase that established
   information *incremental to implied variance* — and it established it at
   30 calendar days only. §5.1's arm establishes nothing either way. A
   10-day trading arm would therefore be monetizing a relationship no
   locked phase has demonstrated survives the `IV²` control.
2. **A second holding period is a second verdict-bearing specification** in
   this signal's first monetization test. That is precisely the multiplicity
   that pre-registration exists to prevent.
3. **The cost asymmetry runs the wrong way and makes a short arm dangerous
   to leave available.** A shorter hold pays the same entry spread over
   fewer days, so a shorter horizon is mechanically more attractive on a
   per-day basis and would be tempting to promote after seeing results.
   Excluding it now, rather than carrying it as an unweighted option, is
   the point.
4. **Holding to expiry eliminates the exit-side option spread entirely** —
   the single largest cost improvement available given §1.2. A shorter hold
   would reintroduce it. This is a design reason, not a results reason, and
   it is on the record before any V4 P&L exists.

**But the underlying question — when does the effect arrive — is answered
by a diagnostic instead, and that diagnostic is REQUIRED.** The build
script must report the **mean and median cumulative delta-hedged net P&L by
day-in-trade** (day 1 … day `n`), per side, per execution tier. This shows
whether the P&L accrues evenly, arrives early, or arrives only near expiry,
without creating a second tradeable specification. It carries **no verdict
authority**. If it shows the P&L is complete well before expiry, that is a
**named candidate for a future, separately pre-registered shorter-horizon
phase** — never retrofitted into V4.

### 5.3 Exit rule — FIXED 30-CALENDAR-DAY EXIT (REVISED, final reconciliation)

**This section supersedes an earlier hold-to-expiry design.** The earlier
draft read "Exit is at expiration and at no other time," held the contract
to expiry, settled at intrinsic, and charged **no option-side exit spread**.
That design is withdrawn. Exit is now at a **fixed 30 calendar days** after
entry, which necessarily means the option is **sold** before expiry. Every
consequence of that change is carried through §7.1, §7.6, §9, and §9.1
rather than left implicit.

- The **exit session** is the last trading session on or before
  `entry_date + 30 calendar days` — V3 §3.3-F's own `n_t` convention,
  reused unchanged. Measured across DEV, the hold spans **17–22 trading
  sessions** (mean 20.67, mode 21).
- The position is **closed by selling the contract** at the exit session,
  at the execution tier's exit price (§9). It is **not** held to
  expiration: under §7.1's locked [40,60] band the contract still carries a
  median of **21 days** of remaining life at exit (minimum observed across
  the full DEV sample: **13 days**), so it is sold as a live, still-quoted
  option rather than as an expiring stub.
- **An option-side exit spread IS charged.** This is the single most
  consequential cost consequence of the change: the superseded design paid
  one half-spread (buy at ask, settle at intrinsic); this design pays a
  **full round trip** (buy at ask, sell at bid). §6.3's spread ceiling is
  tightened to 10% partly in response — see §6.3 C5.
- **Missing-quote handling at exit is governed by §7.8**, not here. A valid
  bid is present on the scheduled exit session for **91.75% of calls /
  94.47% of puts** (measured, `results/55_v4_design_reconciliation.json`),
  so the fallback path is a live branch, not a formality.
- **No stop-loss and no profit target.** Unchanged, and for the unchanged
  reason: either would be a tuned exit parameter with no pre-registered
  justification, and both would create a path-dependent selection this
  design has no way to validate.
- If the exit session would fall after DEV_END (2021-12-31), the trade is
  **not entered**. Trades with an incomplete window are never truncated or
  padded (E1 §4.3 convention).

---

## 6. Universe — two tiers, defined before any return is examined

### 6.1 Schema fact that constrains everything below

`opprcd.csv` (80.55 GB, staging) carries exactly these columns:

```
secid, date, exdate, cp_flag, strike_price, best_bid, best_offer,
volume, open_interest, impl_volatility, optionid, index_flag,
issuer, exercise_style
```

Consequences, each load-bearing:

- **There is no vendor delta for a traded contract.** The `delta` column
  exists only on the *surface* files, where it labels a standardized
  constant-maturity grid point, not a listed contract. §7.2's delta source
  is therefore forced, not chosen.
- **There is a per-contract, per-day `impl_volatility`.** This is what makes
  a daily-updated hedge possible at all (§7.2).
- **There is no `cfadj`, no dividend field, no settlement-type field, and no
  contract-size field.** `q = 0` and a 100-share contract multiplier are
  declared assumptions (§7.5), matching `src/30`/`32`/`37` and `src/42`'s
  disclosure D3.
- **`exercise_style` is present**, so the American/European split is
  observable and must be reported (§7.1).
- **`best_bid`/`best_offer` are present** — the quantities K1 §7 discovered
  it did not have, and which `src/37` later used. V4's cost model is
  therefore built on observed quotes, not on an assumed spread.

### 6.2 Universe 6(a) — full optionable universe, DIAGNOSTIC ONLY

V3's universe 6(a) (the full V1/V2 base universe) restricted to stock-days
with (i) a usable primary `IV²` per V3 §3.3-F, and (ii) at least one
contract satisfying the instrument definition in §7 with a two-sided quote.
No liquidity screen beyond that. **Carries no verdict authority.** It exists
so that a "real but inaccessible" outcome (V3 decision-tree row 3's trading
analogue) is visible rather than invisible.

### 6.3 Universe 6(b) — ex-ante liquid-options subset, PRIMARY TRADING UNIVERSE

**V3's §6(b) is reused as the stock-day gate, unchanged and not re-tuned:**

1. At least one listed option contract on that underlying **traded** that
   day (`volume > 0` somewhere in the chain); **and**
2. the underlying's `DlyPrcVol` is at or above the **same-day
   cross-sectional median** of the V1/V2 universe.

Measured membership: **44.89%** of universe stock-days
(`results/47_v3_liquidity_coverage.json`); stable by year (42.57%–46.52%),
discriminating by decile (69.62% / 48.12% / 24.01% for deciles 6/7/8).

**V4 needs additional contract-level thresholds, and here is why.** V3's
6(b) was built for an *information* test that charged no costs and executed
nothing; it screens the **underlying's stock-day**, and it deliberately
dropped a spread criterion because V3 "charges no costs — spread belongs in
the eventual trading test of §7's first branch" (V3 §6(b), final
paragraph). V4 **is** that trading test. It executes a specific contract at
a specific quote, so a stock-day screen is necessary but nowhere near
sufficient: a name can clear both V3 conditions on the strength of a
heavily traded near-dated chain while the specific ~30-day ATM contract V4
wants has a zero bid. The following contract-level thresholds are therefore
**added**, not substituted, and all are evaluated **at entry, on
information available at entry**:

| # | criterion | locked threshold |
|---|---|---|
| C1 | valid two-sided quote | `best_bid > 0` **and** `best_offer > best_bid`, on the entry date |
| C2 | no static-arbitrage violation | call: `max(S − K·e^{−rT}, 0) <= mid <= S`; put: `max(K·e^{−rT} − S, 0) <= mid <= K·e^{−rT}` |
| C3 | IV sanity | `impl_volatility` non-null and within **[0.05, 2.00]** |
| C4 | minimum option price | `mid >= $0.50` **and** `best_bid >= $0.20` |
| C5 | maximum relative spread | `(ask − bid) / mid <= 0.10` (10% **full** spread) — REVISED from 15% |
| C6 | minimum open interest | `open_interest >= 100` contracts on the entry date |
| C7 | minimum recent option volume | contract `volume >= 10` on the entry date **and** trailing 5-trading-day total contract volume `>= 50` |
| C8 | minimum underlying dollar volume | V3 6(b) condition 2, reused verbatim |
| C9 | underlying data completeness | valid CRSP `DlyClose` on the entry date and on every session of the hold |

**Reasoning for each threshold, stated before any V4 return exists:**

- **C1** is the binding structural filter: `opprcd`'s whole-file diagnostic
  already logged **23.07% of contract-days with a zero bid**. A zero bid
  means the position cannot be exited at any price, so it is not tradeable
  in any meaningful sense.
- **C2** is a correctness filter, not a liquidity one. A quote violating
  static arbitrage bounds is a data error; letting one through would let a
  data error masquerade as an edge.
- **C3** reuses the exact sanity thresholds `src/42` already measured on
  this data (`threshold_high = 2.0`, 1.07% of trades; `threshold_low =
  0.05`, 0.58%). `src/42` used them as flags; V4 uses them as filters. That
  is a change in role, disclosed here, and the numbers themselves are not
  new dials.
- **C4** exists because OptionMetrics quotes in cents and the exchange
  minimum tick is 5¢ for options under $3. Below ~$0.50 the tick alone is
  ≥10% of the price, so C5's relative-spread measurement stops describing
  liquidity and starts describing tick granularity. The `best_bid >= $0.20`
  leg additionally requires the exit side to be genuinely marketable.
- **C5 is the most consequential threshold in this document and is
  deliberately aggressive. REVISED from 15% to 10% at the final
  reconciliation.** `src/41` established K1's entry-side **half**-spread at
  a median 18.75% of mid — a median **full** spread of ~37.5% at the
  straddle level. A 10% full-spread cap is roughly **3.75× tighter** than
  K1's typical traded contract. Given that §1.2 shows spread is what killed
  K1, admitting K1-like spreads would guarantee the same outcome and learn
  nothing. **This cap will cut the sample sharply and its cost is reported,
  never used as grounds to relax it.** If 6(b) turns out to be near-empty
  under C1–C9, that emptiness is itself the reportable finding — the trading
  analogue of V3's "real but inaccessible" row — and the phase reports
  FAIL-by-infeasibility rather than loosening a threshold.

  **Why 10% and not 15%, and the utilization consequence stated plainly.**
  §5.3's fixed-exit design pays a **round-trip** option spread where the
  superseded hold-to-expiry design paid one half-spread. Spread therefore
  costs roughly twice as much per completed trade as it did when the 15%
  figure was set, which is the direct reason for tightening. Measured at
  [40,60] / $100,000 NAV, the 10% ceiling clears both stated minimums on
  both sides (calls 11 median filled positions / 1,058 invested days; puts
  9 / 985).

  **The 20% median invested-day utilization figure is NOT a hard
  requirement and is not claimed to be met.** Under the 10% ceiling,
  utilization is **17.44% (calls) / 18.59% (puts)** at $100,000, and **no
  tested NAV reaches 20%.** That 20% figure originated as a byproduct of the
  now-superseded one-half-spread cost structure, not as a validity
  condition; it is retired as a binding criterion here rather than carried
  forward as an unmet target. Utilization is reported as a capacity
  diagnostic (§8.1) and gates nothing. **This document does not claim the
  10% ceiling and a ≥20% utilization standard are simultaneously
  satisfied — they are not, and 10% is adopted knowing that.**
- **C6/C7** distinguish "a contract exists" from "a contract trades."
  V3 §6(b) already rejected open interest alone as near-vacuous (92.05%
  retention). OI ≥ 100 plus same-day volume ≥ 10 plus 5-day volume ≥ 50 is
  the contract-level version of the same "could this have been transacted
  on the day the signal fires" test, at a scale consistent with the
  position sizes §8 permits.
- **C8** is reused verbatim so V4's stock-day screen is identical to the one
  V3 gated its own liquid-universe result on. It is not re-tuned; V3's own
  60th/75th-percentile variants remain no-authority sensitivities there and
  are **not** carried into V4.
- **C9** is a hedging prerequisite: the hedge is marked at each session's
  close, so a missing close makes the hedge unmarkable.

**Coverage under C1–C9 is unknown and cannot be pre-verified.** The cached
`data/processed/opprcd_liquidity_daily.parquet` holds only
`secid/date/volume/open_interest` aggregates — no bid, no ask, no strike, no
expiration, no IV. Measuring C1–C7 requires the pull in §13. **No threshold
above may be revised after that pull returns a coverage number**; the pull
is authorized to measure the cost of the locked thresholds, not to choose
them. This is the identical discipline V3 §6(b) applied to its own
feasibility pass, with one difference recorded honestly: V3 locked its
thresholds *after* seeing coverage, on a stated non-vacuity criterion; V4
locks them *before*, because the relevant costs (C4, C5) are grounded in
already-logged K1 facts rather than in unmeasured coverage.

---

## 7. Instruments and hedging specification

### 7.1 Instrument definition

**Two instruments, tested as two separate books:** delta-hedged **calls**
and delta-hedged **puts**. Both are volatility-branch instruments (§1.3).
Neither is a directional position, and neither may be described as one.

| parameter | locked value |
|---|---|
| moneyness convention | **delta-based**, not fixed % OTM |
| delta band | `|delta| ∈ [0.40, 0.60]`, target 0.50 |
| days to expiration at entry | **[40, 60] calendar days**, target 50 (REVISED — see below) |
| expiration constraint | the **exit session** (§5.3), not the expiration date, must fall inside DEV |
| positions per underlying | **maximum one active position per PERMNO**, across both books combined |
| earnings-window treatment | **excluded** — see §7.7 |
| contract multiplier | 100 shares (declared assumption, §6.1) |

**Why delta-based moneyness, per the prior correction:** a fixed
percentage-OTM rule means something different on a 20%-vol name than on a
90%-vol name — the same 5% OTM strike is ~2 standard deviations away on one
and ~0.4 on the other, so a fixed-% rule silently sorts on volatility
rather than on option characteristics. Delta normalizes for this. The
[0.40, 0.60] band is tighter than K1's [0.35, 0.65] because V4 executes and
hedges a **single** contract rather than a two-leg straddle whose combined
delta is near zero by construction, so per-contract delta precision matters
more here.

**Why [40,60] on a fixed 30-day hold, and the mismatch it creates — stated
here rather than left to be discovered later.** The earlier draft locked
[25,38] and justified it as "~30 DTE held to expiry… it removes the
exit-side option spread." **Both halves of that justification are
withdrawn:** the design no longer holds to expiry (§5.3), so no exit-side
spread is removed; and [25,38] is measurably incompatible with a fixed
30-day exit.

**Measured, on the full DEV sample under the fixed 30-day exit**
(`results/55_v4_design_reconciliation.json`; three bands compared):

| | [25,38] C/P | **[40,60] C/P (locked)** | [45,60] C/P |
|---|---|---|---|
| % expiring **before** the scheduled exit | **9.28% / 9.45%** | **0.00% / 0.00%** | 0.00% / 0.00% |
| remaining DTE at exit, median | **1 day** | **21 days** (min 13) | 22 days |
| hold path fully quoted | 55.95% / 60.83% | 89.66% / 93.48% | 90.96% / 94.57% |
| bid present at exit session | 58.19% / 62.53% | 91.75% / 94.47% | 92.84% / 95.48% |
| selected candidates | 35,178 / 24,826 | 37,410 / 22,212 | 27,594 / 16,192 |

[25,38] fails the no-expiration-before-exit requirement outright, and even
its survivors are unusable: a **median of 1 day** of remaining life at exit
is not a sellable instrument, which is exactly why its exit-session bid
availability collapses to 58%/63%. [40,60] is chosen over the equally
compliant [45,60] because it carries **~36% more call candidates** for only
~1.3pp less quote survival.

**The mismatch this creates, disclosed:** the *holding period* still matches
V3's 30-calendar-day forecast horizon exactly — that is the horizon the
forecast is actually about. The *instrument's tenor* (40–60 days) does
**not** match V3's 30-day standardized IV tenor, so the option's vega sits
at a longer tenor than the signal was measured at. This is a real mismatch,
accepted deliberately because the fixed-exit requirement makes it
unavoidable: any band short enough to match the 30-day tenor permits
expiration before the scheduled exit.

**Tie-breaking when multiple contracts qualify — deterministic, and never
by expected return.** Applied in strict order:

1. smallest `|delta − 0.50|`;
2. if tied within 0.01, smallest `|DTE − 50|` (anchor revised with the band);
3. if still tied, smallest relative spread `(ask − bid)/mid`;
4. if still tied, largest `open_interest`;
5. if still tied, smallest `optionid` — a deterministic, data-independent
   final tie-break so the selection is reproducible.

Ranking by any expected-return or score quantity at this stage is
explicitly forbidden: the contract choice must not be a second, hidden
signal.

### 7.2 Delta source — recomputed, and the choice is forced

**Locked: Black–Scholes delta, recomputed. Vendor delta is not available
(§6.1) and the surface `delta` is not the traded contract's delta.**

```
delta_call = N(d1),   delta_put = N(d1) - 1
d1 = [ ln(S/K) + (r + 0.5*sigma^2) * T ] / (sigma * sqrt(T))
S = CRSP DlyClose on the valuation date
K = strike_price / 1000            (OptionMetrics x1000 convention)
T = (exdate - valuation date) / 365
r = 0.01                           (§7.5)
q = 0                              (§7.5)
sigma = the CONTRACT'S OWN opprcd impl_volatility on the valuation date
```

**IV is updated daily.** `sigma` on each hedge date is that date's own
quoted `impl_volatility` for that `optionid`, not the entry value.

**Justification against K1/`src/42`'s sticky-entry-IV approach.** `src/42`'s
mandatory disclosure D1 states it plainly: holding entry IV constant "makes
the hedged return PATH-DEPENDENT: it is a REALISTIC decomposition, NOT a
pure volatility isolation. A residual vega/vanna P&L remains inside the
delta-hedged leg." V4's central question is specifically about volatility
P&L, so leaving vega leakage inside the primary would mean the primary
measures something other than what it claims. **The sticky-entry-IV variant
is retained as a labeled robustness case with no verdict authority**, so
V4's numbers remain comparable to `src/42`'s decomposition on that
construction's own terms.

**Declared assumption, with its consequence named.** OptionMetrics computes
`impl_volatility` for American-style equity options from a binomial model;
V4's delta is Black–Scholes. For an ATM option with `q = 0` and ~30 days to
expiry the early-exercise premium is very small, and the `|delta| ∈ [0.40,
0.60]` band excludes the deep-ITM puts where it is largest. Two required
checks make this falsifiable rather than asserted:

- The build script must report the distribution of
  `|BS_price(sigma_OM) − mid| / mid` at entry across all selected contracts,
  broken out by `exercise_style`.
- **Assertion:** the median of that distribution must be **below 5%**. If it
  is not, the run halts and no V4 result is reported until the cause is
  found — the same standing this project gave V3's ACT/365 invariance
  assertion (a failed assertion reveals a coding or data-understanding
  error, not a modeling preference).
- Contracts with a relative pricing error **above 25%** at entry are
  **dropped and counted**, with the drop rate reported by `exercise_style`.

### 7.3 Rehedging

**PRIMARY: full rehedge daily at the close**, on every trading session from
entry through expiration. This matches `src/42`'s D2 convention ("close-to-
close (discrete), once per trading day") and preserves comparability.

`src/42`'s D2 disclosure carries forward unchanged and applies to V4:
intraday moves are unhedged and a gamma-driven hedging error remains. This
biases nothing systematically but adds variance to every per-trade number.

**Robustness arm, no verdict authority: threshold rehedging** — rehedge only
when `|delta_t − delta_last_hedged| > 0.10`. This *reduces* hedge cost, so
it can only make results look better; it is therefore explicitly
**non-promotable** — a threshold-rehedge result may never be substituted
for the primary, regardless of how the two compare.

### 7.4 Execution prices and timing

- **Signal → entry timing.** `Score_{i,t}` uses information through the
  close of `t`. Entry occurs at the **close of `t+1`**, on `t+1`'s option
  quote and `t+1`'s stock close. This satisfies `CLAUDE.md` rule 5
  ("signals computed on day t trade day t+1").
- **Disclosed deviation from `CLAUDE.md`'s "entries fill at next-day
  open".** `opprcd` carries **closing** quotes only; there is no opening or
  intraday option quote in this data. Filling the stock leg at the open
  while filling the option leg at the close would create an artificial
  cross-leg timing gap of a full session. Both legs are therefore executed
  at the same `t+1` close, which keeps them synchronous. This is a
  deviation from the letter of rule 5's second sentence and is recorded
  here rather than absorbed silently.
- **Stock execution price:** CRSP `DlyClose` on the execution date, for the
  entry hedge, every rehedge, and the terminal unwind.

### 7.5 Rates, dividends, and corporate actions

- **Risk-free rate: `r = 0.01` annualized, constant.** Reused verbatim from
  `src/42`'s declared assumption A1 and its stated rationale: "Ken French
  RF is quantized to 0.0001/day over this window — a lumpy step function,
  not a rate path." `src/42` also *measured* the sensitivity: moving `r`
  from 0 to 0.0252 shifts the straddle delta by a median 0.0095 (p95
  0.0255), because `r` enters `d1` only via `r·sqrt(T)/sigma` and is near-
  inert at this tenor and volatility level. That measurement is cited, not
  re-derived.
- **Dividends: `q = 0`.** Declared assumption, matching `src/30`/`32`/`37`
  and `src/42`'s D3. Consequence, stated with its direction: on a
  dividend-paying underlying this slightly overstates call delta and
  understates put delta, and a cash dividend paid during the hold on a
  **short** hedge leg (which the call book carries) is an out-of-pocket cost
  not modeled. **The call book's result is therefore biased mildly
  favorably by this assumption, and that must be stated wherever the call
  result is reported.** No dividend field exists in the pulled data.
- **Corporate actions.** Trades whose hold window contains a change in the
  CRSP price-adjustment factor (splits and stock distributions — *not*
  ordinary cash dividends) are **dropped and counted**. `src/42` asserted
  zero such changes inside its 10-session windows; V4's windows are roughly
  twice as long, so the assertion is promoted to a drop rule rather than
  left as an assertion that will now sometimes fail. Ordinary cash
  dividends are handled by the `q = 0` disclosure above, not by this rule.

### 7.6 Costs

**REVISED (item 6 of the return-blind audit — `results/51_v4_borrow_data_check.md`).**
The borrow rate calibration, the dividends-owed-on-short component, and
symmetric hedge financing below replace the earlier draft's 50 bps / 0 bps
/ 200 bps treatment. Confirmed first, before locking numbers: no point-in-
time securities-lending source (borrow fee, hard-to-borrow flag, share
availability, or lending utilization) exists anywhere in this project's
data — CRSP, Compustat, OptionMetrics, and Fama-French factors were all
checked; none carries one. `results/51_v4_borrow_data_check.md` has the
full inventory. Everything in this section is therefore necessarily a
flat-rate model, disclosed as such.

| cost | locked treatment |
|---|---|
| option entry | per the execution tier in §9 |
| option exit | per the execution tier in §9 — a **full round-trip** spread is charged (§5.3, REVISED; the earlier "no option-side exit cost" is withdrawn) |
| equity slippage/spread | **15 bps per side**, on every stock trade (entry hedge, each rehedge, terminal unwind) |
| equity stress case | **30 bps per side**, reported alongside, no verdict authority |
| short borrow (PRIMARY) | **10% annualized**, accrued daily on short stock notional only |
| short borrow (sensitivity) | **3% annualized**, no verdict authority |
| short borrow (stress) | **25% annualized**, no verdict authority |
| dividends owed on short shares | `DisDivAmt × shares_short` on each `DisExDt` falling inside the holding window, charged to the call book's hedge leg |
| short rebate | **none credited** (conservative) |
| long-hedge financing | **`r` = 0.01 annualized cost** on long stock-hedge notional (put book) — symmetric with the short side's no-rebate treatment; both hedge directions are charged, neither is credited |
| commissions | **not modeled** — disclosed, see below |

**Equity cost.** 15 bps/side base and 30 bps/side stress are `CLAUDE.md`
rule 6's own numbers, unchanged. This also sits at the top of `src/42`'s
swept hedge-cost range (0 / 5 / 15 bps), so V4's primary equity-cost
assumption is no more optimistic than `src/42`'s most conservative
hedge-cost case.

**Short borrow — recalibrated to 10% primary, materially higher than the
superseded 50 bps.** Hedging a long **call** requires shorting the
underlying; hedging a long **put** requires buying it. The borrow charge
therefore applies to the call book and, transiently, to any put position
whose delta hedge goes short. The earlier 50 bps figure was pitched at a
general-collateral large-cap rate; V4's universe is deciles 6–8
(mid/small-cap), where GC borrow is typically higher and a meaningful
share of names carry at least occasional special/hard-to-borrow pricing.
**10% annualized is the new primary** — a materially more conservative
starting point than a pure-GC assumption, chosen precisely because no
name-level data exists to tell primary-book names apart from occasionally-
special ones, and the call book (the primary verdict arm, §11.1) should
not be flattered by an optimistic default. The 3% sensitivity brackets a
near-GC case; the 25% stress brackets specials-adjacent pricing. **None of
the three is a specials-AWARE model** — each is a single flat number
applied uniformly across every name and every day, which cannot
distinguish a persistently-hard-to-borrow name from an always-easy one.

Order of magnitude, so the primary assumption's weight is visible: at
`delta` ≈ 0.5, `S` = $50, 30 days, 10%/yr, the borrow cost is
`0.5 × $50 × 0.10 × 30/365` ≈ **$0.205 per share**, against an ATM premium
of roughly $2.85 per share at 50% vol — about **7.2% of premium**. This is
no longer a small line item (the superseded 50 bps figure was ~0.36% of
premium) — at the 25% stress tier it is roughly **18% of premium**. Borrow
cost is now a first-order consideration for the call book's viability, not
a footnote, and must be reported with that weight in any V4 result.

**Availability is not the same question as cost, and this model answers
only the second one.** A flat borrow rate says nothing about whether
shares are actually obtainable, in the required quantity, on the specific
name and date the hedge needs them. Every call-book simulation this
project can build implicitly assumes the short leg is always executable —
if a name is in practice occasionally or persistently unborrowable, the
simulated result reports a hedge that could not have been placed, a more
severe failure mode than a mispriced one. This caveat must be disclosed
alongside any call-book result as prominently as the rate figure itself,
not folded into the same sentence as if a rate bracket already covers it
(`results/51_v4_borrow_data_check.md` §3).

**Commissions are not modeled**, inheriting K1 §7's disclosure. Real
per-contract commissions and per-share equity commissions would add further
drag, so **any positive V4 result is an upper bound on realistic
performance, not a conservative estimate.** With daily rehedging over ~21
sessions, the omitted per-share equity commission is a larger omission here
than it was in K1.

### 7.7 Earnings-window treatment — excluded

~~**A candidate is ineligible if any RDQ falls inside its holding window
(entry date through expiration date, inclusive).**~~ **SUPERSEDED — see
dated amendment immediately below. Original text left visible, not
deleted, matching this document's own non-rewrite amendment convention
(V3 §3.3's "original text is unedited... dated pointer annotations added
rather than any rewrite").**

> **[2026-08-04 — AMENDMENT, discovered during `src/60`'s C1-C9 funnel
> construction, patched before any further V4 script is built.]** The
> struck sentence above is stale. It was written when V4's design held the
> option **to expiration** (the pre-final-reconciliation draft). §5.3 was
> revised to a **fixed 30-calendar-day exit** — the position is sold, not
> held to expiration — and that revision was propagated to §5.2, §5.3,
> §7.1, §7.6, §9, and §9.1 during the final reconciliation, but §7.7 was
> missed. Left as originally written, §7.7 would exclude candidates for
> earnings exposure they are never actually exposed to, since expiration
> sits 10–30+ days past the actual exit at the locked [40,60] entry band.
>
> **CORRECTED LANGUAGE, LOCKED NOW:** a candidate is ineligible if any RDQ
> falls inside its holding window, defined as **(entry date, exit date]**
> — the exit date being §5.3's fixed-30-calendar-day exit session, **not**
> the contract's expiration date. This is the position's actual risk
> window under the locked exit design, not a new interpretation.
>
> **Confirmed consistent with what was actually built, not reconciled
> after the fact.** `src/60_v4_c1c9_funnel.py` (STEP 4d) already
> implemented `(entry, exit]` — computing `exit_dates` from each row's own
> `entry_pos + n_hold` and testing whether any RDQ falls strictly after
> entry and on/before that exit date — before this amendment was written.
> The script was correct; the locked document's text was stale. No
> discrepancy exists between the two as of this amendment, so nothing in
> `src/60`'s output changes and no re-run is required.

Rationale: V3's `b3` — the coefficient V4 monetizes — is defined against
the **omitted reference category** "no RDQ within 20 trading days" (V3 §4).
It is, by construction, the compression effect *outside* earnings windows.
V3 also found the within-bucket totals `b3 + b5k` behave differently, and in
the 6(b) liquid universe most were not significant. Trading through
earnings would therefore monetize a different coefficient than the one that
passed, and would additionally expose a delta hedge to overnight jump risk
that daily rehedging cannot hedge by construction.

Construction: `EarningsBucket`'s underlying `(gvkey, rdq)` pairs and CCM
link, reused verbatim from `src/43`/V3 §3.5. Stock-days whose linked gvkey
has no RDQ history at all are **excluded and counted**, never recoded as
"no earnings" — V3 §3.5's rule, unchanged.

**Two disclosures, both recorded now:**

1. **Attrition will be large.** A ~30-calendar-day window covers roughly a
   third of a quarterly reporting cycle, so this filter plausibly removes
   on the order of a third of otherwise-eligible candidates. The exact cost
   is reported; it is not grounds to relax the filter.
2. **Inherited mild look-ahead, named.** Compustat's `rdq` is the *realized*
   announcement date. Firms usually pre-announce their reporting date
   weeks ahead, but not universally, so using `rdq` at entry assumes
   slightly more foreknowledge than a trader always has. This is the same
   construction V3 and `src/43` used, and no I/B/E/S expected-report-date
   file exists in this repo to resolve it. It acts on **eligibility**, not
   on the return of any trade that is taken. **Required no-authority arm:**
   the primary is also reported with earnings-window trades **included**, so
   the filter's effect is visible rather than assumed benign.

### 7.8 Missing data during a trade

> ## SECTION STATUS: **LOCKED** (was PENDING; §7.8's pre-entry rule confirmed 2026-08-03)

**Mid-hold rules — settled (REVISED at the final reconciliation).** The
earlier draft's "drop the entire trade after 3 interior missing-quote days"
convention is **withdrawn** and replaced:

| situation | rule |
|---|---|
| missing/invalid contract quote on the **entry** date | candidate ineligible, counted |
| missing CRSP close on **any** session of the hold | trade **dropped and counted** (the hedge cannot be marked) |
| **1 missing contract quote mid-hold** | **carry the last valid delta forward for at most one session**; no rebalance, no cost charged |
| **2 consecutive missing quotes mid-hold** | **exit at the first subsequent valid bid** |
| **no bid returns before expiration** | **settle at intrinsic value** |

**A trade is NEVER dropped solely because quotes went missing mid-hold.**
This deliberately departs from the "drop and count" convention used
elsewhere in this project (V3 §3.3, `src/43`), and the reason is specific:
**mid-hold gaps in option quotes are a feature of the exact liquidity tier
V4 is studying, not a data defect.** Dropping those trades would bias the
sample toward only the smoothest-quoted names and would systematically
overstate the tradability of the universe — the opposite of what a
cost-bearing monetization test should do. Pre-entry gaps are treated
differently, and may exclude a candidate, precisely because they are
observable **before** committing capital; mid-hold gaps are not.

**Measured incidence** (band [40,60], full DEV,
`results/55_v4_design_reconciliation.json`): hold path fully quoted
**89.66% calls / 93.48% puts**; worst consecutive gap median 0, p95 4
(calls) / 1 (puts); valid bid present on the scheduled exit session
**91.75% / 94.47%**.

All drop counts are reported by reason (V3 §8, E1 §12 convention).

### Pre-entry eligibility rule — LOCKED, listing-adjusted form (CONFIRMED 2026-08-03)

**What the measurement found, and why the original rule is not what gets
locked.** A candidate rule of "valid quotes on ≥90% of the prior 20
sessions, ≤1 consecutive missing session," evaluated unconditionally over a
fixed 20-session lookback, was measured
(`src/56_v4_pending_measurements.py`, keyed on each contract's first
observed quote date, full DEV, both opprcd passes) against the 38,627
selected candidates:

| | calls | puts |
|---|---|---|
| fail rate, unconditional 20-session form | **39.22%** (9,175) | **32.83%** (5,002) |
| **of those failures, dominated by NOT-YET-LISTED** | **98.26%** | **99.14%** |
| of those failures, dominated by LISTED-BUT-UNQUOTED | 1.74% | 0.86% |
| mean lookback sessions before first listing | 4.41 of 20 | 3.55 of 20 |
| **mean lookback sessions listed-but-unquoted** | **0.05** | **0.02** |
| pass rate evaluated only from the listing date (no history floor) | 98.91% | 99.54% |

**The unconditional 20-session form was overwhelmingly measuring contract
age, not illiquidity, and is REJECTED.** Genuine listed-but-unquoted
illiquidity averages **0.05 sessions out of 20 for calls and 0.02 for
puts** — effectively nil. **This near-zero residual is expected, not
evidence the rule should be dropped**: C6/C7 (§6.3 — minimum open interest,
minimum same-day and trailing-5-day volume) are already the primary
liquidity gate, applied before this rule is ever reached. What §7.8's
pre-entry rule catches is the narrow residual case C6/C7 cannot see —
sporadic quote gaps on a contract that otherwise clears the liquidity
floor. A near-zero rate on that narrow case is exactly what a working
secondary safety net should show, not a sign it is redundant.

**LOCKED RULE, final form:**

1. Let `n_since_listing` := the number of trading sessions between the
   contract's first observed quote (inclusive) and the entry date
   (exclusive).
2. **Minimum listing history: `n_since_listing ≥ 10`.** A contract with
   fewer than 10 sessions of history is **entry-ineligible, counted as
   "insufficient listing history"** — regardless of how clean those few
   sessions look. This exists so a just-listed contract cannot trivially
   pass on a near-empty window; it is a validity floor, not a liquidity
   measurement.
3. For candidates clearing step 2, the evaluation window is
   `W = min(20, n_since_listing)` sessions immediately preceding entry —
   by construction entirely **post-listing**, never reaching before the
   contract existed.
4. **Entry-eligible only if:** valid quotes exist on **≥90% of `W`** AND
   **no more than 1 consecutive missing session within `W`**. Otherwise
   **entry-ineligible, counted as "quote continuity failure."**

This is the measured "evaluated only from the listing date" construction
(§ table above), **with the added ≥10-session floor now locked on top of
it.** The floor was not present in the 98.91%/99.54% measurement, so the
**exact pass rate of the final, floored rule has not been separately
measured** — it will run somewhat below the unfloored figures once
newly-listed candidates are excluded by step 2. This is disclosed as an
unmeasured refinement, not silently assumed: the floor only removes
candidates, so 98.91%/99.54% are valid **upper bounds** on the final pass
rate, and the reduction from a 10-session floor on a mean pre-listing gap
of 4.41/3.55 sessions (measured on the unconditional rule's failures) is
expected to be modest. No further return-blind pass is required before
this locks — the floor is a monotonic tightening of an already-measured,
very high pass rate, confirmed by the owner as the intended design rather
than a data-driven parameter search.

Full measurement detail: `results/58_v4_pending_measurements_note.md`.

---

## 8. Portfolio construction

**REVISED (item 3 of the return-blind audit, position count and sizing
convention corrected).** An earlier draft of this section locked 50
maximum positions with equal-premium sizing as primary. Both are
superseded here: **maximum concurrent positions is 20, and equal-vega is
the PRIMARY sizing convention** — the reverse of the earlier draft's
ranking. This was confirmed before item 3's own capacity audit was run
(§8.1 below), specifically so that audit measured the design actually
being locked, not a superseded one.

| element | locked value |
|---|---|
| positions per underlying | **1 active**, across both books combined |
| overlapping signal on an open name | **skipped and counted** — no pyramiding, no roll |
| maximum concurrent positions | **20** |
| minimum for an "invested day" | **5** — a reporting/diagnostic threshold (§8.1), not an entry gate: a day with 1–4 positions still trades whatever is eligible, but is classified as thin/under-invested in capacity reporting, so a strategy that spends much of its time below this floor is visible as such |
| slot filling | descending `Score`; ties broken by §7.1's contract chain |
| sizing convention | **equal vega** (see below) — PRIMARY, reversed from the earlier draft |
| capital denominator | **$100,000 fixed** (REVISED from $2,000,000) |
| unused capital | earns `r` = 0.01 annualized (consistent with §7.5) |
| capacity cap (sizing) | contracts ≤ min(vega-implied count, **10% of entry open interest**, **20% of trailing-5-day contract volume sum**) — **retained as PRIMARY for this lock**; the stricter dynamic alternative is a reported sensitivity only, see §8.2 |
| minimum position | 1 contract; if the 1-contract premium alone exceeds **2× the nominal per-slot capital** ($10,000 on $100k/20), the candidate is **skipped and counted** as premium-constrained |
| sector concentration | **≤ 4 of 20 open slots** in any one GICS `gsector` — rescaled from the superseded 10-of-50 at the same 20% ratio |
| aggregate vega cap | portfolio dollar vega **≤ 0.5% of capital per 1 volatility point** (**≤ $500/pt on $100k**) — also anchors the per-position sizing target |
| aggregate gamma cap | a simultaneous **+1%** move in all underlyings must change portfolio delta by **≤ 5% of capital** (**$5,000 on $100k**) |
| rebalancing | **none** — option positions are never resized; the only daily action is the delta hedge (§7.3) |
| exit | **fixed 30-calendar-day exit (§5.3)**; the slot frees the next trading day |

**Capital denominator — $100,000, and what that choice is and is not.**
Measured across five NAV levels at the locked [40,60] band (§8.1),
$100,000 is the smallest level tested and the only one where both books
clear a 20% median invested-day utilization at the former 15% ceiling.
Every cap above keeps its existing **fractional** definition; only the
absolute figures rescale, so nothing about the risk structure changes.

**Stated explicitly, because it is easy to over-read: this is a
CAPACITY-FEASIBILITY choice, not a RISK-SIZING choice.** It answers only
"can the book stay adequately busy and deploy capital without exceeding
participation limits." It does **not** answer "is $100,000 the right amount
of capital given the edge's actual magnitude" — that requires the edge's
size and volatility, which are returns, which the return-blind audits
behind this figure cannot see by design. **The NAV must be revisited once
real V4 returns exist.** A capacity-feasible $100,000 book could still be
far too large or far too small relative to a real edge, and nothing
measured so far speaks to that.

**Sizing — equal vega, and why the ranking reversed.** Each position
targets equal **dollar vega per 1 volatility point**, not equal premium.
Target per position, at full capacity: `target_vega_per_position = (0.5% ×
capital) / max_positions = $500 / 20 = $25 per vol point` on the locked
$100,000 denominator, i.e. the
same aggregate vega cap this section already locked now doubles as the
sizing anchor when the book is fully invested, rather than introducing a
second, independent parameter. Per-candidate contract count:

```
contract_vega_per_pt = 100 * S * phi(d1) * sqrt(T) * 0.01     [BS vega, per 1 vol POINT, per contract]
contracts = round( target_vega_per_position / contract_vega_per_pt )
```

subject to a **1-contract floor** (a candidate whose vega-implied count
rounds below 0.5 is either sized at 1 contract, flagged **oversized
relative to its vega target**, or skipped as **whole-contract-rounding-
constrained** — §8.1 reports which, and how often each occurs) and the
capacity cap above.

**Why equal vega is now primary, reversing the earlier draft's reasoning.**
The earlier draft rejected equal vega specifically because it "puts a
model output inside the position size, coupling sizing to the same BS/IV
machinery V4 is testing." That concern is real but was weighed against the
wrong alternative: equal-premium sizing does not avoid a model dependency
either — it just hides a different one, since a fixed dollar amount of
premium buys wildly different amounts of *risk* (vega, gamma, delta-hedge
turnover) depending on the option's own price level, moneyness, and IV,
so an equal-premium book is not actually a risk-equalized book at all. V4
is a volatility-branch strategy whose entire thesis is expressed through
vega exposure (§1.3); sizing every position to equal premium while letting
vega exposure vary freely by whatever the option happens to cost is a
mismatch between what the strategy is betting on and what it is
equalizing. Equal vega sizes positions on the dimension the strategy
actually trades. The vega-computation dependency this introduces is the
same BS machinery already used for the hedge (§7.2) and the entry filter
(§9.1) — it adds no new model risk beyond what V4 already carries
end-to-end.

**Equal premium is retained as a labeled robustness arm, reversed from its
former primary status, with no verdict authority** — reported alongside so
a reader can see whether the verdict is sensitive to the sizing
convention. Equal forecast-risk is still not run: it would require a
second pre-registered risk model that does not exist in this project.

**Concentration and Greek caps.** Unchanged in mechanism from the earlier
draft, only rescaled to 20 positions: when the sector cap or either Greek
cap binds, the **lowest-Score** qualifying candidate is skipped and
counted, so the binding rule never removes the strongest signal. The build
script must report **how often each cap binds**. `gsector` comes from the
V2 CCM-linked panel (E1 §11 provenance); rows with null `gsector` are
dropped from the sector cap's accounting and counted, never bucketed as a
residual sector.

**Capacity, per §8.1's measured findings, not a hypothetical example.**
§8.1 is the required, return-blind capacity audit of this specification
(20 positions, equal-vega sizing) run against real `opprcd` quotes at V4's
locked instrument definition (§7.1) and liquidity universe (§6.3). Its
findings are load-bearing for this section, not illustrative.

### 8.1 Portfolio breadth and capacity — return-blind audit (item 3)

**Scope, exactly as specified, no more.** This audit reports capacity
only — median and maximum contract count per position, median effective
number of positions, the share of days below the 5-position "invested"
floor, and the share of candidates constrained by each named factor
(option volume, premium, vega, gamma, sector, whole-contract rounding),
plus projected capital utilization. **No portfolio return, P&L, or Score-
based ranking was computed anywhere in this audit** — Score requires the
expanding-window forecast (§4.2), which is a separate, not-yet-audited
component (item 9 addresses its design, not its output). Where a
day's eligible-candidate count exceeds 20, this audit fills slots by a
**neutral, deterministic, Score-independent order** (ascending PERMNO) —
disclosed explicitly because the specific candidates chosen when eligible
count exceeds capacity does not affect the breadth statistics this item
asks for (which depend on counts, not identities), and using a neutral
order avoids implicitly pre-testing "how liquid are the compression
signal's favorites specifically," a different question this audit was not
asked to answer.

**Results — the five-NAV reconciliation supersedes the earlier single-NAV
run.** An earlier version of this section reported a $2,000,000-only audit
at the superseded [25,38] band (`results/53_v4_portfolio_breadth_capacity.md`,
retained as the historical record). The current figures come from
`src/55_v4_design_reconciliation.py` at the **locked [40,60] band**, full
DEV, 886,525,545 opprcd rows per pass, at the former 15% ceiling:

| NAV | median filled (C/P) | median eff-HHI (C/P) | invested days (C/P) | median invested-day utilization (C/P) | median contracts/position (C/P) |
|---|---|---|---|---|---|
| **$100k (LOCKED)** | 17 / 13 | 14.56 / 10.53 | 1,153 / 1,086 | **22.76% / 24.38%** | 8 / 8 |
| $250k | 17 / 13 | 13.92 / 9.71 | 1,153 / 1,086 | 20.51% / 19.87% | 16 / 15 |
| $500k | 17 / 13 | 12.38 / 8.25 | 1,153 / 1,086 | 16.82% / 14.81% | 23 / 20 |
| $1M | 17 / 13 | 10.34 / 6.88 | 1,153 / 1,086 | 12.49% / 10.03% | 30 / 22 |
| $2M | 17 / 13 | 8.25 / 5.67 | 1,153 / 1,086 | 8.13% / 6.33% | 33 / 23 |

**At the locked 10% ceiling** (§6.3 C5), the same $100,000 book gives
**11 / 9** median filled positions, **1,058 / 985** invested days, and
**17.44% / 18.59%** median invested-day utilization. Per §6.3, the 20%
utilization figure is **not** a requirement and is not claimed to be met.

**Two findings that shape how this section should be read:**

1. **The anticipated small-NAV whole-contract-rounding non-linearity does
   not occur.** `rounding_up_forced` is **0.00%** at $100,000 — no position
   was force-rounded up from a sub-half-contract vega target — and median
   filled positions is **identical (17 calls / 13 puts) at every NAV
   level**, because what fills a slot is candidate availability and the
   20-position cap, not capital.
2. **The real non-linearity is at the large end, and it is the binding
   capacity fact about this strategy.** Median contracts per position runs
   8 → 16 → 23 → 30 → 33 from $100k to $2M: a **20× increase in NAV buys
   only ~4.1× the contracts**, because the absolute OI/volume
   participation caps bind on 2.0% of call candidates at $100k but **48.4%
   at $2M** (puts 4.1% → 69.0%). Capital above roughly $250,000 is
   substantially undeployable at this liquidity tier. **This book does not
   scale, and the constraint is the underlying option market, not the
   sizing rule.**

Aggregate vega and gamma caps never bind at any NAV tested (0.00%), so the
portfolio-level Greek limits are not what constrains size; the §6.3
liquidity thresholds are. This is reported as a finding, not grounds to
loosen those thresholds.

### 8.2 Dynamic liquidity cap — RESOLVED: not adopted, retained as a disclosed sensitivity

> ## SECTION STATUS: **RESOLVED — NOT ADOPTED** (was PENDING; combined measurement completed 2026-08-03)
>
> The dynamic cap below is **not** the primary sizing rule. The fixed cap
> in §8's table remains primary. The combined measurement that this section
> previously awaited is complete and is reported below; it resolved in
> favour of the status quo, so **no further decision is required here** and
> this section no longer blocks lock.

**The alternative rule:** `max contracts = min(5% of trailing 20-day
average contract volume, 1% of open interest)`.

**Measured against the incumbent fixed cap** (band [40,60], full DEV):

| rule | median cap (contracts) | cap ≥ 1 contract |
|---|---|---|
| fixed cap (10% OI, 20% of 5-day volume sum) — **PRIMARY** | **31** | 100.00% |
| dynamic cap (1% OI, 5% of trailing-20d avg volume) — sensitivity | **1** | **74.40%** |

Effect on the book at the 15% ceiling: median invested-day utilization
falls from 22.76% / 24.38% to **9.41% / 7.07%** at $100,000, and to
0.69% / 0.48% at $2,000,000.

**Are C6/C7's absolute minimums redundant with the dynamic rule? No.** The
dynamic rule is a `min()` of a volume leg and an OI leg, and **the OI leg
can be satisfied by stale open interest with no recent trading at all** — a
contract with 100,000 open interest and zero recent volume yields a
1,000-contract dynamic cap while failing C7's actual-trading floor
entirely. C6/C7 encode "this contract genuinely trades"; the dynamic rule
encodes "given that it trades, how much can I take." Only the second is a
capacity formula. **C6 and C7 are therefore retained unchanged as
eligibility screens regardless of what happens to the sizing cap.**

**Honest statement of the case against the incumbent:** taking 10% of a
contract's entire open interest in a single trade is not a realistic
participation assumption, and the fixed cap's permissiveness is why §8.1's
utilization figures look as healthy as they do. The dynamic rule is the
more realistic model of day-to-day tradability. It is not adopted now only
because its cost has not been fully measured in combination with the
locked 10% ceiling.

**MEASURED 2026-08-03 — the combined case now exists, and it fails the
effective-breadth floor on the put book.** The required combined
measurement (`src/56_v4_pending_measurements.py`, locked band, locked 10%
ceiling, both caps through identical machinery, full DEV):

| at $100,000 NAV | calls FIXED | calls DYNAMIC | puts FIXED | puts DYNAMIC |
|---|---|---|---|---|
| median filled positions | 11 | 9 | 9 | **6** |
| **median effective breadth (inverse-HHI)** | 10.11 | 5.76 | 7.67 | **3.52** |
| invested days | 1,058 | 992 | 985 | 806 |
| median invested-day utilization | 17.51% | **7.62%** | 18.75% | **6.17%** |
| median contracts per position | 8 | 3 | 8 | 2 |

Utilization under the dynamic cap falls to roughly **a third** of the fixed
cap's at every NAV ($250k: 4.13%/3.20%; $2M: 0.57%/0.42%). The earlier ~7%
extrapolation is superseded by the measured 7.62% / 6.17%.

**DECISION: the fixed cap remains PRIMARY for this lock.** The decisive
finding is breadth, not utilization — under the combined case the put
book's **effective breadth falls to 3.52, below the 5-position floor this
design already committed to**, on a book whose capacity is thin to begin
with.

**What this costs, stated rather than buried.** The gap between the two
caps is not a modelling detail: it is the difference between deploying
~18% of capital and ~7%, and between a put book with ~8 effective
positions and one with ~3.5. **§8.1's utilization figures are therefore an
upper bound, and any eventual V4 result computed under the fixed cap
inherits that optimism.** If V4 passes its gates under the fixed cap with
an edge that is small relative to the ~2.5× position-size difference
between the two rules, that result should be treated as fragile.

**Suggested, NOT pre-registered:** if V4 ever produces a passing result
under the fixed cap, re-running its P&L under the dynamic cap is the
natural robustness check — but that is a post-result test outside this
return-blind cycle and carries no authority here. Full detail:
`results/58_v4_pending_measurements_note.md`.

---

## 9. Execution — three named tiers, one of them primary

**REVISED at the final reconciliation.** All three tiers are computed on the
**same** trade list, hedge path, and equity costs. Only the option **entry
and exit** prices differ. The position is sold at the fixed 30-day exit
(§5.3), so **every tier carries an option-side exit price** — superseding
the earlier draft's statement that none did.

| tier | option entry price | option exit price | status |
|---|---|---|---|
| **Midpoint** | `(best_bid + best_offer) / 2` | `(best_bid + best_offer) / 2` | **frictionless diagnostic only — explicitly NOT the pass criterion** |
| **Partial-spread** | `mid + 0.50 × (ask − mid)` | `mid − 0.50 × (mid − bid)` | realistic sensitivity; assumed fraction = 0.50 of the half-spread on each side |
| **Ask-to-bid** | `best_offer` | `best_bid` | conservative, fully marketable in both directions |

**PRIMARY for the pass/fail verdict: the ask-to-bid tier.**

Reasoning: a midpoint-only pass is not a tradeable result — that is the
whole lesson of §1.2, where K1's delta-hedged +8.36% mid-based figure turned
into −7.32% at just 25% of the spread. Buying at the offer and selling at
the bid is what marketable limit orders on both legs actually achieve.

**The earlier claim that this tier charges "precisely one half-spread over
the life of the position" is withdrawn** — that was true only of the
superseded hold-to-expiry design. The ask-to-bid tier now charges a **full
round-trip spread**, which at §6.3's 10% ceiling is up to ~10% of mid per
completed trade. This is a materially heavier cost assumption than the
earlier draft carried, and it is the honest one for a design that must sell
the position 30 days later.

The partial-spread fraction convention (0.50 of `ask − mid`) is `src/42`'s
own `spread_fraction_paid` convention, reused so the two phases' cost
ladders are directly comparable.

### 9.1 Entry cost filter — no position is entered that is already expected to lose

A candidate is eligible **only if**, using information available at entry:

```
ExpectedGrossEdge  >  1.5 * ( ExpectedOptionCost + ExpectedHedgeCost )
        AND
ExpectedGrossEdge  -  ( ExpectedOptionCost + ExpectedHedgeCost )  >=  $5.00 per contract
```

**`ExpectedGrossEdge`** — the forecast priced through the same convex
function the market uses, i.e. K1 §6's construction reused on a single leg,
with K1's in-sample TRAIL20 replaced by V4's out-of-sample forecast:

```
ExpectedGrossEdge = 100 * [ BS(S, K, T, r, q=0, sigma = sqrt(predicted_RV2))
                          - BS(S, K, T, r, q=0, sigma = IV_contract) ]
```

evaluated as a call or a put to match the instrument. `IV_contract` is the
**traded contract's own** `impl_volatility`, **not** the surface `IV²` used
in `Score`. This distinction is deliberate: `Score` ranks candidates on
V3's validated, horizon-matched surface quantity; the eligibility filter
must charge against the IV actually embedded in the price being paid.

K1 §6's rationale for pricing the gap rather than comparing volatility
points carries over unchanged: a fixed vol-point gap is worth different
dollars on a cheap option than an expensive one, and the BS difference
captures that nonlinearity.

**`ExpectedOptionCost`** = `100 × [(ask − mid) + (mid − bid)]` = `100 ×
(ask − bid)` per contract — the **full round-trip** spread under the
primary ask-to-bid tier (§9). **REVISED:** the earlier draft's "No
exit-side term" is withdrawn with the hold-to-expiry design (§5.3). Because
§9.1 requires expected edge to exceed `1.5 × (option cost + hedge cost)`,
doubling the option-cost term materially raises the entry bar — the
intended, conservative direction.

**`ExpectedHedgeCost`**, closed-form and computable at entry:

```
E[ sum |d_delta| ] ~= Gamma_0 * S_0 * sigma_f * sqrt(2/pi) * n / sqrt(252)
ExpectedHedgeCost   = 100 * c_eq * S_0 * E[ sum |d_delta| ]
```

where `Gamma_0` is the entry BS gamma, `sigma_f = sqrt(predicted_RV2)`, `n`
is the number of trading sessions from entry to expiration, and `c_eq` =
0.0015. The `sqrt(2/pi)` factor is the mean of `|Z|` for a standard normal
— the expected absolute daily delta change under a BS delta with gamma held
at its entry value.

**Disclosed approximation:** holding gamma fixed at its entry value
understates turnover when the option stays near the money and overstates it
when the underlying drifts away. This formula is **only the eligibility
screen**; the **realized** hedge cost is charged separately, path by path,
in the actual P&L (§7.6).

**Safety margin — the 1.5× multiplier and the $5 floor.** The multiplier
requires modeled edge to exceed modeled cost by 50%, not merely to match it.
It follows K1 §6's own precedent for setting a threshold against a known
headwind: K1 set `T = 25%` as the measured average volatility risk premium
(23.7%) *rounded up*, so the threshold demanded more than the headwind
rather than exactly the headwind. Here the known headwind is measured cost,
and 50% over it is the analogue. The $5-per-contract absolute floor stops
mathematically-qualifying but economically-trivial trades on very cheap
options from entering. **Neither number is swept, and neither may be
revised after any V4 result exists.**

**Basis note, with a required check.** `sqrt(predicted_RV2)` is annualized
on the 252-trading-day basis (V3 §3.3-F step 3), while BS `T` is a
calendar-year fraction `DTE/365`. An annualized volatility is basis-free
once annualized, so this is internally consistent — but the build script
must **report** how many candidates change eligibility if `T = n/252` is
used instead, so the assumption's practical weight is measured rather than
asserted.

---

## 10. Primary outcomes — trade-level and portfolio-level, reported SEPARATELY

`src/42` is the reason this separation is mandatory rather than stylistic:
its delta-hedged leg showed **+8.36% mean per trade** alongside **NW t =
−0.409** on the calendar-time daily series. Those two numbers describe the
same positions and point in opposite directions. Reporting either alone
would misrepresent the result. **No V4 table may combine them, and no V4
sentence may report a per-trade mean without its calendar-time counterpart
adjacent to it.**

**Reported for each of: {call book, put book} × {6(a) diagnostic, 6(b)
primary} × {midpoint, partial-spread, full-ask} —**

**Trade level**
- mean and median per-trade net P&L, in dollars and as % of entry premium
- number of trades; unique PERMNOs; win rate
- distribution: p1 / q1 / median / q3 / p99
- share of total P&L from the **top 1% / 5% / 10%** of trades
- entry-IV distribution and compression-decile distribution of the traded
  book (§4.4 attribution requirement)
- realized hedge turnover, and realized-vs-modeled hedge cost
- mean and median **cumulative delta-hedged net P&L by day-in-trade**
  (§5.2's required diagnostic)

**Portfolio level**
- daily calendar-time portfolio return on the $100,000 denominator (§8), with
  its own **Newey–West t-stat, `maxlags = 21`**
- cumulative return; annualized return; annualized volatility
- **maximum drawdown**
- **turnover** (option premium traded and stock notional traded, per
  `CLAUDE.md` rule 6)
- **worst month**; best month
- **performance by year** (2017–2021, the scored period per §4.2)
- **result excluding calendar year 2020 specifically**, named — the known
  volatility-regime outlier, following K1 §8's 2020 stratification
- 2020 alone
- **capital utilization**: mean and median deployed fraction; frequency at
  the 20-position cap (§8, REVISED from 50); frequency each capacity/sector/Greek cap binds

**Required no-authority arms, all reported, none able to change a verdict:**
V2 compression variant (§4.1); rolling-60-month and pooled-OLS estimation
(§4.2); sticky-entry-IV hedge (§7.2); threshold rehedging (§7.3); 30 bps
equity stress and 3%/25% borrow bracket (§7.6, REVISED); earnings-included
arm (§7.7); zero-tolerance missing-quote arm (§7.8); **equal-premium
sizing** (§8, REVISED — this is now the reversed robustness arm since
equal-vega became primary); dynamic liquidity cap (§8.2).

**§4.4's former `b3`-zeroed placebo book is superseded, not additionally
required** — Model B (§4.1, §11 item 6) replaces it with a properly
separate fit, per §4.4's own disclosure. It is not listed twice.

A **FAIL is written up with the same completeness as a PASS** — trade
count, win rate, the cost waterfall, and where the edge died — matching K1
§9 and E1 §12.

---

## 11. Locked PASS bar — all six required (REVISED, item 9 of the return-blind audit adds item 6)

Evaluated on the **6(b) liquid universe**, under the **full-ask** execution
tier, on the **V1 compression variant**, for the **call book** (§11.1 —
calls are the primary arm; the put book is evaluated against the same six
items but its result is secondary, never gating, per §11.1).

| # | condition | locked threshold |
|---|---|---|
| **1** | positive net calendar-time return under the PRIMARY execution tier | mean daily net portfolio return **> 0** under full-ask; **midpoint does not satisfy this item under any circumstance** |
| **2** | statistically significant at the portfolio level | **Newey–West t ≥ 2.0**, `maxlags = 21`, on the daily calendar-time series |
| **3** | positive incremental performance, high-score vs low-score | **Q5 − Q1 > 0** and **\|NW t\| ≥ 2.0** on the daily Q5−Q1 difference series, where Q1/Q5 are **quintiles of `Score`** formed **within each entry date** on the cost-filter-free eligible set (see below) |
| **4** | no dependence on a single year or a small handful of trades | **(i)** no single calendar year contributes **> 50%** of total P&L; **(ii)** the **top 1% of trades by P&L** contribute **≤ 50%** of total P&L; **(iii)** the result **excluding 2020** remains **positive in sign** (sign only — no significance requirement) |
| **5** | survives in the liquid-options subset specifically | items 1–4 hold on **6(b)**; 6(a) is diagnostic only |
| **6** | **beats the compression-free benchmark (Model B)** | `mean(Diff) > 0` **and** `NW t(Diff) ≥ 2.0` (`maxlags = 21`), `Diff(t) := Return_A(t) − Return_B(t)` — §4.1's Model B, full specification `results/52_v4_compression_benchmark_design.md` |

**Item 2's threshold, and why 2.0 and not 3.0.** This project uses two bars,
and K1 §8 states the distinction explicitly: **2.0** is "the project's
standard alpha-claim bar, matching R1/R2's gate, not V1/V2's higher
exploratory-forecast bar." V1/V2/V3 used 3.0 because they were forecasting
tests "without the discipline of a cost model attached" (V3 §5). V4 is a
cost-bearing trading claim with a cost model attached, so it takes the
alpha-claim bar of 2.0. `maxlags = 21` matches V3's own primary-horizon
convention and the ~21-session overlap of concurrent V4 positions.

**Item 3's sample, stated precisely.** The score-spread test is computed on
the **cost-filter-free eligible set** — every contract passing §6's universe
screens and §7's instrument definition, but **before** §9.1's edge-vs-cost
filter. This is necessary because §9.1 is designed to remove low-score
candidates, so applying it first would leave no bottom quintile to compare
against. Item 3 therefore measures whether the score *sorts* delta-hedged
returns; items 1, 2, 4 measure whether the *traded book* makes money. Both
are reported. A date enters the quintile test only with **≥ 25 eligible
candidates** (5 per bucket); dates below that are dropped and counted.

**Item 4's (iii) is a sign test only, deliberately.** A single-year
exclusion cuts roughly 20% of the scored period, and demanding significance
on the remainder would let ordinary loss of power register as
year-dependence. Sign consistency is the discriminating condition; the
excluding-2020 t-stat is reported alongside without gate authority.

**Item 6 — REQUIRED, not diagnostic, added by the return-blind audit
(item 9, `results/52_v4_compression_benchmark_design.md`).** A call book
that clears items 1–5 but fails item 6 does **not** produce a primary V4
PASS: passing the five-item bar shows the traded book made money, but not
that compression specifically is why, since Model B shares every
non-compression regressor (`IV2`, `PriorRV2`, earnings buckets, and every
control including the liquidity-adjacent `log_cap`/`log_dvol`/`log_price`)
and runs through the identical universe, contract-selection, hedging,
execution, sizing, and portfolio-limit machinery. Only `Compression` and
its earnings interactions differ between the two books. `NW t(Diff) ≥ 2.0`
is this document's own proposed calibration (matching item 2's alpha-claim
bar), flagged for owner confirmation since the reviewer's instruction
specified that the gate must exist and be required, not its exact
threshold.

### 11.1 Two books, one verdict — CALLS PRIMARY, PUTS SECONDARY (REVISED, item 4 of the return-blind audit)

**Superseded.** An earlier draft of this section made calls and puts
co-equal, with a one-sided-pass path available to either side. That is
replaced here with an asymmetric structure, resolving what was flagged in
§14 as the single most consequential open item in this document:

- **Delta-hedged calls are the PRIMARY V4 arm.** A primary **V4 PASS**
  requires the **call book alone** to clear every item of the §11 five-item
  bar, on 6(b), under full-ask execution, on the V1 compression variant —
  exactly as §11 states, with no reference to the put book's own result.
- **Delta-hedged puts are a SECONDARY confirmation and attribution arm.**
  The put book is evaluated against the identical five-item bar and fully
  reported (§10's trade-level and portfolio-level tables, §12's
  monotonicity report — all unchanged, both books, per §10's standing
  requirement). But the put book's own result **carries no authority over
  the primary V4 verdict** in either direction: a passing put book cannot
  turn a failing call book into a PASS, and a failing put book cannot turn
  a passing call book into a FAIL.
- **A put-only pass is reported as a secondary put-side result, not a
  primary V4 PASS.** If the put book clears all five items while the call
  book does not, the write-up states exactly that — "put book clears the
  five-item bar; primary V4 verdict is FAIL because the call book does
  not" — and does **not** use the word "PASS" for V4 itself under that
  outcome.
- **Both books passing is labeled stronger cross-side confirmation of the
  primary verdict**, not a separate, stronger category of pass. The
  primary verdict is still simply "V4 PASS" (on the call book); the put
  book's own pass is additional evidence reported alongside it, not a
  second thing being decided.

**Why calls are primary — an implementation choice, not a directional
claim, restated explicitly because it is easy to misread.** Nothing about
naming the call book primary reflects a view that a delta-hedged call is a
bullish strategy, or that calls are expected to outperform puts on
volatility grounds. §1.3 already establishes that both are volatility-
branch instruments with most of their directional exposure hedged away.
The call book is named primary because it is the intended instrument
implementation for this phase — the specific, singular claim V4 is built
to test — and reporting a single named primary arm, with the other book as
disclosed confirmation, avoids exactly the multiplicity problem the
earlier co-equal structure existed to solve (testing two books is two
shots at the same bar) while being simpler and more conservative than the
one-sided-pass path it replaces: under the old rule a put-only result
could itself be called "V4 PASS, put side"; under this revision it cannot
be called a V4 PASS at all.

**What is unchanged.** §10's requirement that both books be reported in
full, separately, at both trade and portfolio level, stands exactly as
written — this revision changes which result the word "PASS" attaches to,
not what gets measured or disclosed. §12's monotonicity reporting is
likewise unchanged and still covers both books. The **SPLIT** language from
the superseded rule (a significant positive on one side against a
significant negative on the other) is retained as a **required disclosure
item**, not a verdict category: if the put book is negative with NW t ≤
−2.0 while the call book passes, that asymmetry must be stated plainly
in the write-up, since a real, unresolved cross-side divergence is exactly
the kind of fact this project does not bury inside a single headline
verdict.

---

## 12. Monotonicity — reported, not gated

Sort the cost-filter-free eligible set into **five score buckets
(quintiles)**, formed within each entry date, matching §11 item 3's
construction exactly. Report, per bucket, per book, per execution tier:
mean and median delta-hedged net return, trade count, mean entry IV, mean
compression decile, and the daily calendar-time mean with its NW t-stat.

**Monotonicity is diagnostic evidence of robustness. It is NOT a separate
pass/fail gate.** The five-item bar in §11 is what determines PASS/FAIL, and
nothing in this section can override it in either direction.

Explicitly: **a single working threshold without monotonicity is weaker
evidence, and must be flagged as such if it occurs.** If Q5 outperforms Q1
while the intermediate buckets are unordered or non-monotone, the write-up
must say so plainly and must not present the Q5−Q1 spread as though the
score were a graded signal. Conversely, clean monotonicity across all five
buckets alongside a §11 failure is **not** a pass and may not be reported
as one — it is a named candidate for a future, separately pre-registered
phase, matching E1 §6's treatment of its own grid.

---

## 13. Required data pull — authorization needed before any build

V4 cannot be built from cached artifacts. The cached
`data/processed/opprcd_liquidity_daily.parquet` (3,129,623 secid-days) holds
only `secid/date/volume/open_interest` — deliberately, because bid/ask "were
NOT authorized and are NOT read" in `src/47`. V4 needs the quote and
contract fields.

**Requested pull, scoped as narrowly as the task allows:**

- **Source:** `opprcd.csv` (80.55 GB, staging folder — not copied into the
  OneDrive-synced repo, per `src/36`'s stated reasoning).
- **Columns:** `secid, date, exdate, cp_flag, strike_price, best_bid,
  best_offer, volume, open_interest, impl_volatility, optionid, index_flag,
  exercise_style` (13 of 14; `issuer` not needed).
- **Row scope:** `index_flag == 0`; `secid` restricted to the V1/V2 universe
  linkage; `date` within DEV; DTE within [8, 90] calendar days at quote
  date — a superset of §7.1's [40, 60] band, widened at **both** ends: the
  upper end so a candidate's trailing 20-session quote/volume history is
  readable (a 60-DTE contract was ~88 DTE twenty sessions earlier, per
  §7.8's pending measurement), the lower end so the hold path of a
  40-DTE contract remains readable through its fixed 30-day exit and
  beyond (down to ~10 DTE).
- **Output:** a durable cached parquet, so this is scanned once and never
  re-scanned — the same discipline `src/47` applied to its own pull.
- **Purpose limit:** this pull measures the **cost** of the thresholds
  locked in §6.3. It does **not** authorize revising them. Any threshold
  change after this pull returns would be a post-hoc amendment and must be
  dated, justified, and disclosed as such — the standard E1 §13 item 11 and
  V3 §3.3-F set.

A second, smaller step reuses existing artifacts only:
`compression_signal_v1.parquet`, `sector_compression_signal_v2.parquet`,
`crsp_combined.parquet`, `universe_membership.parquet`,
`vol_surface.csv` (for V3's `IV²`), `ccm_link_gics.csv`, and
`rdq_pull_fundq_2014_2026.parquet`. No new pull is needed for any of these.

---

## 14. Non-actions, disclosures, and open items

**Non-actions in producing this document:**

- No return, P&L, coefficient, correlation, t-stat, or trade computed. No
  regression run. No V4 script written.
- `results/gate_log.md` not opened for writing.
- No holdout data touched, and no holdout code path will exist in any V4
  script.
- Two CSV headers read with `nrows=0` — disclosed at the top of this
  document and load-bearing only for §6.1/§7.2. Zero rows of data returned.

**Declared assumptions carried into V4, each inherited with its source:**

| assumption | value | source |
|---|---|---|
| risk-free rate | `r` = 0.01 constant | `src/42` A1 |
| dividend yield | `q` = 0 | `src/30`/`32`/`37`, `src/42` D3 |
| contract multiplier | 100 shares | market convention; no field in data (§6.1) |
| equity cost | 15 bps/side base, 30 bps stress | `CLAUDE.md` rule 6 |
| short borrow | **10%/yr primary** (3% sensitivity, 25% stress), no rebate | **new to V4** — no borrow data in repo (§7.6, `results/51_v4_borrow_data_check.md`) |
| commissions | not modeled | K1 §7 |
| OM IV day-count | ACT/365 | V3 §3.3; unresolvable in this environment |
| American vs European | BS delta on OM binomial IV | **new to V4** — §7.2, with a required assertion |

**Known biases in the primary, each named with its direction:**

1. `q = 0` biases the **call** book mildly **favorably**. The borrow rate
   is 10%/yr primary (§7.6), materially more conservative than the
   superseded 50 bps, but it remains a flat rate that **cannot represent
   borrow availability at all** — the more severe limitation.
2. Omitted commissions bias **both** books favorably, and more so than in
   K1 because of ~21 sessions of daily rehedging.
3. **The §8 fixed capacity cap (10% of open interest) is permissive and
   biases capacity favorably.** §8.2's measured dynamic alternative cuts
   utilization by roughly two-thirds and drops the put book's effective
   breadth below the 5-position floor; the fixed cap is primary for this
   lock, so **§8.1's utilization figures should be read as an upper
   bound**, not a central estimate.
4. Discrete daily hedging (`src/42` D2) adds variance without a systematic
   sign.
5. **§7.8's pre-entry rule, in its locked listing-adjusted form, has an
   unmeasured exact pass rate** — the 98.91%/99.54% figures are upper
   bounds measured without the confirmed ≥10-session minimum-listing-
   history floor; the floor can only tighten them further, by an amount
   expected to be modest but not separately quantified.

## Both former blocking items are RESOLVED — this document is LOCKED

Both were return-blind measurements; neither inspected returns or P&L.

| # | item | section | resolution |
|---|---|---|---|
| **O1** | Dynamic liquidity cap — combined measurement | §8.2 | **RESOLVED, not adopted.** Combined 10% ceiling + dynamic cap gives 7.62%/6.17% utilization at $100k, and the put book's effective breadth falls to **3.52 — below the 5-position floor**. Fixed cap stays primary. |
| **O2** | Pre-entry quote rule — listing-recency separation | §7.8 | **RESOLVED, confirmed 2026-08-03.** 98.26%/99.14% of the original rule's exclusion was contract age, not illiquidity. **Locked in listing-adjusted form with a ≥10-session minimum-listing-history floor** — evaluate over `min(20, sessions since listing)`, requiring ≥10 sessions of history to be eligible at all. Rationale: C6/C7 are the primary liquidity gate; this rule is a secondary safety net for sporadic quote gaps, and its near-zero residual (0.02–0.05 sessions of 20) confirms that division of labor rather than the rule's redundancy. |

**Still open, unchanged, and it is the only thing left before any V4
script may run:** §13's data pull must be authorized.

**Resolved and applied (final reconciliation + the earlier return-blind
audit; see `results/56_v4_final_reconciliation_memo.md` and
`results/54_v4_audit_decision_memo.md`):**

- **Exit design** — fixed 30-calendar-day exit, ask-to-bid round trip; all
  hold-to-expiry language removed (§5.3, §7.1, §7.6, §9, §9.1).
- **Entry DTE band** — **[40,60]**, anchor 50, replacing [25,38], which
  permits expiration before the scheduled exit (§7.1).
- **Capital denominator** — **$100,000**, replacing $2,000,000 (§8).
- **Spread ceiling** — **10%**, replacing 15%; the 20% utilization figure
  is retired as a criterion and explicitly **not** claimed to be met
  (§6.3 C5).
- **Position count / sizing** — 20 maximum, equal-vega primary (§8).
- **Verdict structure** — calls primary, puts secondary (§11.1).
- **Borrow** — 10%/yr primary, 3%/25% sensitivities, with the
  availability limitation disclosed (§7.6).
- **Compression benchmark** — Model B specified and made a required sixth
  gate (§4.1, §11 item 6); not yet run.
- **Pre-entry quote rule** — listing-adjusted form with a ≥10-session
  history floor, confirmed by the owner (§7.8).

**Status: LOCKED — 2026-08-03.** Every section above reflects a confirmed
decision. Nothing in this document is amended after this point without a
dated, disclosed amendment record, matching this project's standing
convention (V3 §3.3-F, E1 §13 item 11) — a change in results does not
retroactively justify a change in design. The only remaining gate before
any code is written is §13's data-pull authorization.
