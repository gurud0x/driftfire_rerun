# V4 Pending Measurements — O1 and O2 Results

**RETURN-BLIND.** No return, P&L, forward realized variance, outcome-based
score, or test statistic computed. Forward-looking measurement is **quote
existence** and calendar/DTE arithmetic only. `DlyRet` never loaded.
`gate_log.md` not touched. No V4 trading script written or run.

**Source:** `src/56_v4_pending_measurements.py`, run 2026-08-03, two full
passes over `opprcd.csv` (886,525,545 rows each), at the **locked** design
(DTE [40,60] anchor 50, 10% spread ceiling, fixed 30-calendar-day exit,
$100,000 primary NAV, 20 positions, equal-vega). Detail:
`results/56_v4_pending_measurements.json`.

Selected candidate base: **38,627** (23,391 calls / 15,236 puts) after the
full C1–C9 funnel, the 10% ceiling, earnings exclusion over `(entry, exit]`,
and the §7.1 tie-break.

---

## O2 — Listing-recency separation: the pre-entry rule was measuring the wrong thing

**Result, and it is decisive.**

| | calls | puts |
|---|---|---|
| pass rate, rule as written | 60.78% | 67.17% |
| fail rate | **39.22%** (9,175) | **32.83%** (5,002) |
| **of those failures, dominated by NOT-YET-LISTED** | **98.26%** | **99.14%** |
| of those failures, dominated by LISTED-BUT-UNQUOTED | 1.74% | 0.86% |
| mean lookback sessions before the contract's first listing | 4.41 of 20 | 3.55 of 20 |
| mean lookback sessions listed-but-unquoted | **0.05** | **0.02** |
| pass rate if the rule runs only from the listing date | **98.91%** | **99.54%** |
| candidates rescued by that adjustment | 8,919 (**38.13%** of all selected) | 4,932 (**32.37%**) |

**The rule as drafted was almost entirely a listing-recency filter, not an
illiquidity filter.** Of the ~39%/33% it excluded, **98.3% / 99.1%** was
driven by sessions in which the contract **did not yet exist** — a 40–60
DTE contract had 68–88 DTE twenty sessions earlier and simply had not been
listed. Genuine listed-but-unquoted illiquidity averages **0.05 sessions
(calls) and 0.02 sessions (puts) out of 20** — effectively nil.

This confirms the inference flagged in the reconciliation memo (median 1.0
with Q1 0.50 being the signature of listing recency), and it converts that
inference into a direct measurement keyed on each contract's first observed
quote date.

**Consequence:** applying the rule as written would discard **~38% of calls
and ~32% of puts for having been listed recently**, which is not a
liquidity property at all, and would silently bias the traded universe
toward long-dated, monthly-cycle contracts on a basis nobody chose. Running
the identical rule only over sessions **at or after the contract's first
listing** raises pass rates to **98.91% / 99.54%** while still screening the
genuine illiquidity the rule was written to catch — because that genuine
component barely exists in this universe once listing is accounted for.

**RECOMMENDATION: lock the rule in its listing-adjusted form** — evaluate
the ≥90%-valid and ≤1-consecutive-gap conditions over the intersection of
the prior 20 sessions with the contract's post-listing life, requiring a
minimum number of effective sessions for the test to be meaningful. Since
the residual illiquidity signal is so small (0.02–0.05 sessions of 20), the
honest alternative — dropping the pre-entry rule entirely as non-binding —
is also defensible and is noted, but retaining it in adjusted form costs
almost nothing and preserves a genuine (if rarely-triggered) screen.

---

## O1 — Combined 10% ceiling + dynamic cap: fails the breadth floor on puts

Measured at the locked band and ceiling, both caps run through identical
machinery:

| NAV | | calls FIXED | calls DYNAMIC | puts FIXED | puts DYNAMIC |
|---|---|---|---|---|---|
| **$100k** | median filled | **11** | **9** | **9** | **6** |
| | median eff-HHI | 10.11 | 5.76 | 7.67 | 3.52 |
| | invested days | 1,058 | 992 | 985 | 806 |
| | **median invested-day utilization** | **17.51%** | **7.62%** | **18.75%** | **6.17%** |
| | median contracts/position | 8 | 3 | 8 | 2 |
| $250k | utilization | 15.83% | 4.13% | 15.74% | 3.20% |
| $500k | utilization | 13.30% | 2.25% | 12.20% | 1.69% |
| $1M | utilization | 9.98% | 1.15% | 8.48% | 0.85% |
| $2M | utilization | 6.70% | 0.57% | 5.33% | 0.42% |

**The extrapolation in the reconciliation memo was close but slightly
pessimistic for calls and optimistic for puts.** It projected ~7% combined;
measured is **7.62% (calls) / 6.17% (puts)** at $100,000. The projection is
now superseded by measurement.

**The decisive finding is breadth, not utilization.** Under the combined
case the **put book's median effective breadth falls to 3.52 by inverse-HHI
at $100,000** — below the 5-position floor — and median filled positions
drops to 6 (from 9). Invested days fall to 806 (from 985). Utilization at
$100,000 lands at 6–8%, roughly **a third** of the fixed cap's.

**RECOMMENDATION: retain the fixed cap as primary, as already locked for
this cycle.** The dynamic rule remains the more realistic model of
day-to-day participation — taking 10% of a contract's entire open interest
in one trade is not achievable — but adopting it now would push the put
book below the effective-breadth floor the design already committed to, on
a book whose capacity is thin to begin with.

**What this means, stated plainly rather than buried:** the gap between the
two caps is not a modelling detail. It is the difference between a book
that deploys ~18% of $100,000 and one that deploys ~7%, and between a put
book with 9 effective positions and one with 3.5. **§8.1's utilization
figures should continue to be read as an upper bound**, and any eventual V4
result computed under the fixed cap inherits that optimism. If the strategy
passes its gates under the fixed cap but its edge is small relative to the
~2.5× position-size difference, that result should be treated as fragile.

**Suggested (not adopted) follow-on:** if V4 ever produces a passing
result under the fixed cap, re-running the P&L under the dynamic cap would
be the natural robustness check — but that is a post-result test, outside
this return-blind cycle, and is not pre-registered here.

---

## Status

Both items that blocked lock are now measured. Neither required inspecting
a return.

- **O2 is resolved with a recommended rule change** (listing-adjusted
  evaluation window), which needs your confirmation because it changes the
  locked rule's text.
- **O1 is resolved in favour of the status quo** (fixed cap stays primary),
  which needs no design change — only the disclosure above.

`results/prereg_V4.md` §7.8 and §8.2 are updated with these findings and
remain marked PENDING pending your confirmation of O2's recommended form.
