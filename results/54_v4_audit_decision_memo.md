# V4 Return-Blind Audit — Final Decision Memo

**Synthesizes items 3, 4, 6, 9 of the return-blind audit. No V4 trading
script has been written or run — this memo, and the audit it summarizes,
is exactly what item 10 asked to precede that.** `gate_log.md` was not
touched anywhere in this audit. Supporting documents: `results/51_v4_borrow_data_check.md`
(item 6), `results/52_v4_compression_benchmark_design.md` (item 9),
`results/53_v4_portfolio_breadth_capacity.md` + `.json` (item 3); item 4's
resolution is applied directly in `results/prereg_V4.md` §11.1.

---

## Recommended entry DTE range

**Keep the locked [25, 38] range (target 30), unchanged.** Item 3's audit
ran the full liquidity funnel at this exact band and found DTE width was
never the binding constraint — open interest (C6, 11.02% pass rate on the
DTE/universe-eligible population) and trailing volume (C7, 4.13%) are far
more restrictive than any DTE-driven effect. There is no count-only
evidence to revise the DTE band, and revising it now, after seeing
coverage, would violate this project's standing rule against choosing a
parameter by its own performance.

## Projected sample size

| | calls | puts |
|---|---|---|
| selected candidate stock-dates (full funnel + earnings exclusion, pre-Score) | **33,269** | **22,541** |
| unique underlyings represented | 1,460 (combined) | |
| trading days with ≥1 candidate, of 1,763 DEV days | 1,370 (77.7%) | 1,242 (70.4%) |

These are **eligible-universe** counts, not projected trade counts — the
actual traded sample depends on Score-based ranking into ≤20 daily slots
(§4.2's forecast, not yet built) and will be smaller. They are the correct
denominator for judging whether V4 has enough raw material to work with,
and the answer is yes, at a scale roughly comparable to — a bit larger
than — V3's own liquid-universe (6(b)) sample size.

## Recommended spread threshold, under the count-only selection rule

**Keep C5's locked 15% full-spread cap.** This is a count-only judgment
using data already collected in item 3's funnel pass-rates, not a new
sweep, for a specific reason: **C5 (spread, 23.56% pass) is not the
binding constraint.** C6 (open interest, 11.02% pass) and C7 (volume,
4.13% pass) are both substantially tighter than C5 on the identical
population. Loosening C5 to, say, 20% or 25% would admit more
spread-qualifying rows, but the great majority of those rows would still
fail C6/C7 and never reach eligibility — the marginal gain in coverage
from loosening spread specifically is small, while tightening it further
would needlessly shrink an already-thin pool without addressing the real
bottleneck. This reasoning is grounded entirely in coverage counts already
in hand (`results/53_v4_portfolio_breadth_capacity.md` §3's funnel table),
not in any return or performance figure — consistent with the "count-only"
instruction. If a literal multi-threshold count sweep is wanted for full
precision, it is a cheap addition on top of an already-extracted candidate
table but was not run separately here, since the marginal-bottleneck
reasoning above is sufficient to support the recommendation without it.

## Projected daily breadth

**Materially thinner than a $2,000,000/20-position design implies.**
Median 12 eligible candidates/day (calls) and 11 (puts), *on days that
have any candidate at all* — but **22.3% of all DEV days have zero call
candidates, and 29.6% have zero put candidates.** Correcting item 3's own
mid-run share to the honest full-DEV-window denominator: the book sits
below the 5-position "invested" floor on **50.1% of all DEV days (calls)**
and **54.9% (puts)**. Breadth is a real constraint on this design, not a
footnote.

## Whether daily midpoint-IV delta is feasible

**Numerically yes, in the sense that matters; full-hold coverage was not
directly measured and is flagged as an open gap.** Two distinct questions
sit under this phrase, and they have different answers:

1. **Does midpoint-based IV inversion numerically succeed when a valid
   quote exists?** Yes, essentially always — the V3-short audit's own
   Newton-Raphson inversion (`results/50_v3_short_data_feasibility.md`)
   converged on 99.94–99.95% of rows that reached the inversion step,
   across roughly 1.4 million selected stock-date-sides. Numerical
   convergence is not the limiting factor anywhere in this project's data.
2. **Does a valid quote exist on every day across a position's full ~21-
   session hold, not just at entry?** **Not directly measured by this
   audit.** Item 3's funnel (C1: valid two-sided quote, 70.73% pass on
   the eligible population) is an entry-day snapshot; no script in this
   audit tracked a selected contract's quote/IV availability across its
   own subsequent holding path. `prereg_V4.md` §7.2 already locks the
   PRIMARY delta source as opprcd's own `impl_volatility` field (not a
   fresh daily midpoint inversion) specifically to avoid this dependency
   — a choice this finding does not disturb, but does not yet fully
   validate either. **A full-hold quote-completeness check is a concrete,
   scoped follow-on this audit did not do and should precede the actual
   build**, since §7.8's missing-quote handling rule (hold the hedge
   unchanged, up to 3 interior days) currently rests on an assumption
   about how often that rule would actually fire, not a measurement.

## Whether borrow data exists

**No.** Confirmed by a full inventory of every raw data file in this
project (CRSP, Compustat, OptionMetrics, Fama-French factors) —
`results/51_v4_borrow_data_check.md` §1. No borrow fee, hard-to-borrow
flag, share-availability, or lending-utilization field exists anywhere.
§7.6 is revised to a flat-rate model (10% primary / 3% sensitivity / 25%
stress, plus dividends-owed-on-short and symmetric hedge financing), with
a standing disclosure that **a rate model cannot answer whether shares are
actually borrowable at all** — a structurally different and more severe
gap than a cost-calibration question.

## Specifications that cannot be implemented with current data

1. **Any specials-aware or availability-aware borrow model.** Only a flat
   annualized rate is possible; §3 above and `results/51_v4_borrow_data_check.md`
   §3 are the full statement of what this cannot capture.
2. **A validated full-hold quote/IV completeness rate.** Entry-day
   coverage is measured; hold-path coverage is not, per the delta-
   feasibility finding above.
3. **A specials-aware, binomial (rather than Black-Scholes) delta,
   matching OptionMetrics' own American-exercise convention exactly.**
   `prereg_V4.md` §7.2 already discloses this as an approximation with a
   required pricing-error assertion (median relative error < 5%); this
   audit did not add new information on its size, since doing so would
   require the full build's contract-selection and pricing path, not a
   return-blind check.
4. Everything already disclosed as infeasible in `prereg_V4.md` itself
   (commissions, unmodeled per-contract fees, delisting mechanics beyond
   CRSP's own price-adjustment-factor check) is unchanged by this audit
   and not restated in full here.

## What is now locked, and what remains before any script is written

**Locked by this audit, in `results/prereg_V4.md`:** §8/§8.1 (20 positions,
equal-vega primary, measured capacity), §11.1 (calls primary, puts
secondary), §11 item 6 (compression-vs-benchmark required gate), §7.6
(borrow calibration and its stated limits).

**Still open before a build begins:**
- §13's `opprcd` data-pull authorization (unchanged from the original
  draft — nothing in this audit substitutes for it).
- The full-hold quote-completeness check named above.
- Owner confirmation of this memo's two proposed-but-unconfirmed numbers:
  §11 item 6's `NW t(Diff) ≥ 2.0` threshold, and this memo's own C5
  recommendation (keep 15%, on count-only grounds).

**Per the standing instruction: no V4 trading script is written or run
until these items are confirmed.**
