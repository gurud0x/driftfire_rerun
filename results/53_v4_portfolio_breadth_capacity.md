# V4 Audit Item 3 — Portfolio Breadth and Capacity

**Capacity only. No portfolio return, P&L, or Score-based ranking computed
anywhere.** Script `src/53_v4_portfolio_breadth_capacity.py`, run
2026-08-03, full 886,525,545-row `opprcd.csv` scan. Machine-readable
detail: `results/53_v4_portfolio_breadth_capacity.json`. `gate_log.md` not
touched.

**Parameters:** $2,000,000 NAV, max 20 positions, equal-vega sizing
(target $500/pt per position at full capacity, i.e. the aggregate vega cap
of $10,000/pt ÷ 20), whole contracts, minimum 5 filled positions to
classify a day "invested." V4's own locked instrument definition
(`prereg_V4.md` §7.1: delta [0.40,0.60], DTE [25,38]) and full §6.3
liquidity funnel (C1–C9), on the same 1,733,857-stock-day base population
V3 and the V3-short audit both used. Score does not exist yet (§4.2's
forecast is a separate, not-yet-audited component) — slots are filled in a
neutral, deterministic, Score-independent order (ascending PERMNO); see
§2 below for why this doesn't distort the breadth numbers.

Each book (calls, puts) is run as its own **independent** hypothetical
$2M/20-slot portfolio — the real build's cross-book 1-position-per-
underlying rule needs the Score ranking this audit doesn't have, so
per-book capacity is measured in isolation rather than resolving an
invented cross-book tie-break.

---

## 1. Headline numbers

| metric | calls | puts |
|---|---|---|
| DEV trading days with ≥1 eligible candidate | 1,370 / 1,763 (77.7%) | 1,242 / 1,763 (70.4%) |
| **days with ZERO eligible candidates** | 393 (22.3% of all DEV days) | 521 (29.6% of all DEV days) |
| **days with <5 eligible candidates, of ALL 1,763 DEV days** | **884 (50.1%)** | **967 (54.9%)** |
| median eligible candidates, on days with any | 12 | 11 |
| median filled positions, on days with any | 10 | 9 |
| max filled positions (cap reached) | 20 | 20 |
| % of candidate-days classified "invested" (≥5 filled) | 64.2% | 64.1% |
| median contracts per position | 40 | 32 |
| mean contracts per position | 72.9 | 62.2 |
| max contracts per position | 1,104 | 1,054 |
| projected capital utilization (median / mean / max) | 3.99% / 5.97% / 25.82% | 4.60% / 5.86% / 32.75% |

**The 50.1%/54.9% "<5 eligible" figures are the correct answer to item
3's question and are NOT the same number the script's own console output
printed.** The script computed that share only among days that had *any*
candidate at all (35.8%/35.9%), silently excluding true zero-candidate
days from the denominator. Recomputed against the full 1,763-day DEV
window (the honest denominator, since a zero-candidate day trivially has
fewer than 5 eligible positions), the book is below the 5-position
"invested" floor on **roughly half of all DEV trading days** — a
materially different and more consequential finding than the script's raw
printout suggested. Both figures are reported here so neither reading is
silently chosen.

## 2. Why the neutral fill order doesn't distort the breadth numbers

Every statistic above except contract-count-per-position and capacity-
utilization depends only on the **count** of eligible candidates per day,
not on which ones are selected — the neutral ascending-PERMNO order used
in place of the not-yet-built Score exists only to pick a definite subset
when eligible count exceeds 20 slots, and does not change how many days
clear the 5-position floor or how many candidates exist in total.

## 3. Constraint attribution — corrected mid-audit

An earlier version of this script's capacity-cap accounting only counted
a constraint as "binding" when it reduced a position's contract count all
the way to zero (i.e., only when the position was skipped entirely) —
silently missing every case where the cap merely *shrank* an oversized
vega-implied position while still allowing it to be entered, which turned
out to be the common case at these contract counts. Fixed before the full
run; verified against synthetic cases reproducing the exact capping
arithmetic (a floor-level candidate correctly collapses from an
926-contract vega target down to 20, matching hand computation) and
against the dry-run's own before/after comparison, which confirmed the
fix changed only the *reporting* — the actual entered-position sizes were
unchanged, because the underlying capping arithmetic was already correct;
only its attribution to a named constraint was wrong.

| constraint | calls (% of candidates considered) | puts |
|---|---|---|
| **volume-capacity-constrained** (20% of trailing-5-day volume) | **39.48%** | **58.32%** |
| **OI-capacity-constrained** (10% of open interest) | **44.44%** | **59.37%** |
| sector-concentration-constrained | 11.39% | 12.10% |
| premium-constrained (position unaffordable even after sizing down) | 0.00% | 0.00% |
| whole-contract-rounding-constrained | 0.00% | 0.00% |
| aggregate-vega-cap-constrained | 0.00% | 0.00% |
| aggregate-gamma-cap-constrained | 0.00% | 0.00% |

**Reading this correctly: these are independent, non-exclusive rates** — a
single candidate can be both volume- and OI-constrained simultaneously (in
practice, most are, since C6/C7's own eligibility floor — OI ≥ 100,
volume ≥ 10/50 — is thin relative to what a large equal-vega-implied
contract count needs). **The dominant capacity constraint in this universe
is liquidity (OI and volume), not the portfolio-level Greek caps or
sector limits** — vega-cap-constrained and gamma-cap-constrained are both
exactly 0%, meaning the aggregate risk caps (§8: $10,000/pt vega,
$100,000 gamma) never bound anywhere in the full DEV sample. Whichever
Greek caps V4 ultimately locks, they are not what limits this book's size
— the underlying names' own option-market liquidity is.

**Premium-constrained at 0% is a real finding, not evidence the check is
inert.** Large contract counts (median 40, up to 1,104) are affordable in
dollar terms specifically because they arise from **cheap, low-vega
contracts** (a $5–15 stock at moderate IV can have a per-contract vega
under $1/point, so reaching a $500/pt target takes hundreds of cheap
contracts whose combined premium still stays well under the $200,000
ceiling). The premium check is doing real work in the code — it was
independently verified against synthetic near-floor cases — it simply
never binds *in this particular universe*, because the OI/volume caps
above already suppress most of these positions to a small fraction of
their vega-implied target before the premium check is ever reached.

## 4. What this means for the equal-vega sizing convention (§8)

The $500/pt-per-position target (derived from the aggregate $10,000/pt cap
÷ 20 positions) is **rarely achieved in practice** — median contracts per
position (40 calls / 32 puts) implies most positions are running at a
**fraction** of their vega target once the OI/volume capacity cap
compresses them (a candidate needing 40 contracts to hit $500/pt, capped
by a thin name's liquidity, might enter at far fewer). This is the
capacity ceiling item 3 exists to surface: **the strategy's realistic
capital deployment is well below its nominal $2,000,000, even before any
Score-based selection is applied** — median utilization sits at 4–5% of
NAV, meaning a fully-capacity-constrained day still deploys only a small
fraction of the notional capital this section's own arithmetic assumed.

## 5. Recommendation

This is reported as a finding, not grounds to loosen the C6/C7 liquidity
thresholds — per `prereg_V4.md` §6.3's own standing rule, those thresholds
were locked before any coverage number existed and may not be revised
after seeing one. The honest reading is that **V4's realistic capacity, at
its own locked instrument definition, is meaningfully below its nominal
$2,000,000 book**, and roughly half of all DEV trading days would leave
the book thinly invested (fewer than 5 positions) even before Score
selects among the candidates that do exist. This is exactly the kind of
capacity finding item 3 was designed to surface before any return is ever
computed.
