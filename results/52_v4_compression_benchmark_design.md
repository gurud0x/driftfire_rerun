# V4 Audit Item 9 — Compression-vs-Benchmark Isolation Design

**Design specification only. No forecast fit, no score computed, no return
or P&L produced anywhere in this document.** This requires the full build
(§13's data pull, §4.2's expanding-window machinery) to run; nothing here
can execute until then. `gate_log.md` not touched.

This design **supersedes** `results/prereg_V4.md` §4.4's earlier "required
no-authority diagnostic" (compression coefficient forced to zero
post-fit). That construction is weaker than what follows: zeroing a
fitted `b3` after the fact leaves every other coefficient (`b1`, `b2`,
`b4`, the controls) frozen at values that were themselves estimated *with*
compression in the model, so the other terms never get to re-settle at
their own best fit in compression's absence — an omitted-variable
coupling the reviewer's version avoids by fitting two genuinely separate
models. §4.4 is revised in `prereg_V4.md` to reflect this.

---

## 1. The two forecasts

**Model A — full model (V4's existing primary, `prereg_V4.md` §4.1,
unchanged):**

```
RV2_{i,t,t+30cal} = alpha
   + b1 * IV2_{i,t} + b2 * PriorRV2_{i,t} + b3 * Compression_{i,t}
   + b4a*D_earn_1to5 + b4b*D_earn_6to10 + b4c*D_earn_11to20
   + b5a*(Compression x D_earn_1to5) + b5b*(Compression x D_earn_6to10)
   + b5c*(Compression x D_earn_11to20)
   + controls (log_cap, log_price, log_dvol, recent_absret, trend)
   + error
```

**Model B — benchmark, compression removed entirely:**

```
RV2_{i,t,t+30cal} = alpha
   + c1 * IV2_{i,t} + c2 * PriorRV2_{i,t}
   + c4a*D_earn_1to5 + c4b*D_earn_6to10 + c4c*D_earn_11to20
   + controls (log_cap, log_price, log_dvol, recent_absret, trend)
   + error
```

Model B is **not** Model A with `b3` and the interaction terms set to
zero after fitting — it is a **separately estimated** model with those
five terms (`Compression` plus its three earnings interactions) dropped
from the design matrix before fitting, so `c1`, `c2`, `c4a-c4c`, and the
control coefficients are each Model B's own best fit, not Model A's
coefficients with one term suppressed. Distinct coefficient names (`c`
instead of `b`) are used above specifically so the two are never
conflated in any downstream table.

Two scores follow directly, on the identical basis §4.3 already locks
(252-trading-day annualized, matched to each row's own `n_t`):

```
Score_A_{i,t} := predicted_RV2_A_{i,t,t+30cal} - IV2_{i,t}
Score_B_{i,t} := predicted_RV2_B_{i,t,t+30cal} - IV2_{i,t}
```

## 2. Everything else is byte-for-byte identical — confirmed item by item

| component | Model A | Model B | identical? |
|---|---|---|---|
| **scoring dates** | §4.2's expanding-window schedule (24-month burn-in, monthly refit, first score 2017-01, embargo rule) | same | **yes** — same code path, same refit calendar, same embargo assertion |
| **candidate contracts** | §7.1 delta [0.40,0.60] / DTE [25,38] tie-break | same | **yes** — contract selection depends only on delta/DTE/spread/OI/volume/optionid (§7.1's tie-break order), never on `Score`, so a name selected by both books on the same day resolves to the identical `optionid` |
| **earnings treatment** | §7.7 exclusion (any RDQ inside the hold window) | same | **yes** — eligibility rule, independent of which forecast produced the score |
| **liquidity rules** | §6.3 universe (C1–C9) | same | **yes** — a stock-day/contract either clears the liquid universe or it doesn't, before any score is consulted |
| **hedging** | §7.2–§7.5 (recomputed BS delta from the contract's own daily IV, daily rehedge, `r=0.01`, `q=0`, borrow per the revised §7.6) | same | **yes** — hedge mechanics never reference `Score` |
| **execution** | §9's three named tiers, full-ask primary | same | **yes** |
| **sizing** | §8 equal-vega, $2,000,000 NAV, target `$500/pt` per slot at full capacity | same | **yes** — the sizing formula is a function of the *selected* contract's own vega, not of which model selected it |
| **portfolio limits** | §8: 20 max positions, 1/underlying, sector ≤4/20, aggregate vega/gamma caps | same | **yes** — identical caps, applied independently to each book's own ranked list |

**The only thing that differs anywhere in this pipeline is which score
ranks candidates into the 20 slots (§8's "slot filling: descending
`Score`") and which score feeds §9.1's entry cost filter
(`ExpectedGrossEdge` uses `sqrt(predicted_RV2)` — Model A's or Model B's
own prediction, respectively).** Every other line of the eventual build
script is shared code, parameterized only by which `Score` column is
passed in — not two separate implementations that could silently drift
apart.

## 3. What this isolates, and why sharing the controls is what makes it work

Model B is not a naive "no signal" placebo — it retains **every other
regressor V4's own primary model uses**, including the three controls most
plausibly correlated with "cheap" or "liquid": `log_cap`, `log_dvol`
(dollar volume), and `log_price`. It also retains `IV2` itself, the term
most directly tied to how expensive an option is. This is deliberate and
is the entire reason the comparison works:

**If compression's apparent edge in Model A were actually just liquidity
or cheapness working through the score, Model B would already capture
that same channel** — `log_cap`, `log_dvol`, `log_price`, and `IV2` are
identical inputs in both models, so any tendency for high-score names to
be cheap or liquid *for reasons unrelated to compression* shows up in both
`Score_A` and `Score_B` equally. The only term present in Model A and
absent from Model B is `Compression` and its earnings interactions.
**Whatever incremental performance Book A produces over Book B is
therefore attributable to compression specifically — not to the shared
channels both models already have access to.** This is the precise sense
in which the design distinguishes "compression adds information" from
"the model is just picking cheap or liquid names," and it is why Model B
must be a properly refit model sharing every other regressor, rather than
a stripped-down or randomly-selected placebo.

## 4. The required incremental-performance gate — now locked in §11

Both books are run through the identical portfolio machinery (§8, §9,
primary tier = full-ask, universe = 6(b)) to produce two independent daily
calendar-time return series, `Return_A(t)` and `Return_B(t)`, on the same
$2,000,000 notional, over the same scored period.

```
Diff(t) := Return_A(t) - Return_B(t)
```

**Locked, required gate (added to `prereg_V4.md` §11 as item 6 — "all six
required," not five):** `mean(Diff) > 0` **and** `NW t(Diff) ≥ 2.0`
(`maxlags = 21`, matching §11 item 2's own convention and the ~21-session
overlap of concurrent positions). **This is required for a primary V4
PASS, not a diagnostic** — per the reviewer's instruction, a call book
that clears items 1–5 but fails this incremental test does **not**
produce a V4 PASS, because a passing book that cannot beat a same-universe,
same-execution, same-sizing, compression-free benchmark has not
demonstrated that compression is the source of its edge.

**The `NW t ≥ 2.0` threshold on `Diff` is this document's own proposed
number, not one the reviewer specified — flagged for owner confirmation**,
matching every other numeric bar in this project's convention of stating
rationale before adoption: 2.0 is chosen for consistency with §11 item 2's
own alpha-claim bar (V4 is a cost-bearing trading claim throughout, not an
exploratory forecast test, so the same bar applies to its incremental
claim), rather than inventing a separate threshold with no stated
justification.

**Applies to the call book (primary, §11.1) as the gating requirement.**
The put book's own `Diff` series is computed and reported identically
(§10's standing both-books-always-reported rule), but — consistent with
§11.1's calls-primary structure — the put-side incremental result carries
the same secondary, non-gating status as every other put-book statistic.

## 5. What this design does not resolve

- **Model B is not run here.** No forecast has been fit; this section
  specifies the construction so the eventual build script implements one
  agreed design, not two independently-improvised ones.
- **`c1`, Model B's own `IV2` coefficient, may differ materially from
  Model A's `b1`.** This is expected and not itself informative — Model
  B's other coefficients absorbing some of the variation `Compression`
  would otherwise explain is the entire mechanism this comparison relies
  on, not a symptom of a flawed benchmark.
- **This does not test V2's `sector_rel_decile` variant.** Per §4.1, V2
  remains a labeled sensitivity arm with no verdict authority; a Model-B
  analogue for V2 is not separately required, though it can be reported
  alongside with the same no-authority status if built.
