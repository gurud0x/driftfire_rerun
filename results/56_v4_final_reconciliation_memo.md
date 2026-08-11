# V4 Final Design Reconciliation — Feasibility Decision Memo

**RETURN-BLIND.** No return, P&L, forward realized variance, outcome-based
score, or test statistic was computed anywhere in this exercise. The only
forward-looking quantities measured are **quote existence** and
**calendar/DTE arithmetic** across the holding path — "does a tradeable
quote exist on this date," never "what was it worth." `DlyRet` is never
loaded by the measurement script. `gate_log.md` not touched. **No V4
trading script has been written or run.**

**Source:** `src/55_v4_design_reconciliation.py`, run 2026-08-03. Two full
passes over `opprcd.csv` (886,525,545 rows each). Machine-readable detail:
`results/55_v4_design_reconciliation.json`.

---

## 0. The structural finding that drives everything below

**Items 1 and 6 together change V4's exit design, and the currently-locked
document does not reflect it.** `prereg_V4.md` §5.3 currently locks
**hold-to-expiry** ("Exit is at expiration and at no other time"). Items 1
and 6 specify a **fixed 30-calendar-day exit** and **ask-to-bid** execution
— which necessarily means the option is *sold* at the bid before expiry.
These are different designs. Your instruction set is internally consistent;
the *document* is stale.

**The consequence is a real cost increase, and it must not be buried.** The
superseded design charged **one half-spread** (buy at ask, settle at
intrinsic — no exit trade). The new design charges a **full round-trip
spread** (buy at ask, sell at bid). At the incumbent 15% ceiling that is
roughly **double** the option transaction cost; at a 10% ceiling it is
roughly **1.33×** the superseded cost rather than 2×. Two of
`prereg_V4.md`'s existing claims become false and are flagged for amendment:
§9's "there is no option-side exit price in any tier" and §7.1's "it removes
the exit-side option spread."

This is the single most consequential change in this reconciliation, and it
independently strengthens the case for the tighter spread ceiling (item 3).

---

## 1. Entry DTE band — [25,38] FAILS and must be revised

Measured on real quotes under the fixed 30-calendar-day exit (exit session
= last trading session on or before entry + 30 calendar days, V3's own
`n_t` convention; `n_hold` = 17–22 sessions, mean 20.67, mode 21).

| metric | [25,38] calls / puts | [40,60] calls / puts | [45,60] calls / puts |
|---|---|---|---|
| selected candidates | 35,178 / 24,826 | **37,410 / 22,212** | 27,594 / 16,192 |
| unique underlyings | — | 1,416 / 1,207 | — |
| **% expiring BEFORE the scheduled exit** | **9.28% / 9.45%** | **0.00% / 0.00%** | **0.00% / 0.00%** |
| remaining DTE at exit (median) | **1 day** | **21 days** | 22 days |
| remaining DTE at exit (min) | — | **13 days** | — |
| hold path fully quoted | 55.95% / 60.83% | 89.66% / 93.48% | 90.96% / 94.57% |
| bid present at exit session | 58.19% / 62.53% | 91.75% / 94.47% | 92.84% / 95.48% |

**[25,38] fails the no-expiration-before-exit requirement outright** — 9.3%
of selected contracts expire before the scheduled exit. But the more
damaging number is the one that is *not* an outright failure: **the median
surviving [25,38] contract has just 1 day of remaining life at exit.** A
one-day option is not a sellable instrument at any reasonable spread, which
is exactly why its "bid present at exit session" rate collapses to 58%/63%
against 92%/94% for [40,60]. The band is unusable under a fixed 30-day
exit, on these grounds specifically — independent of, and not contradicting,
the prior audit's separate finding that [25,38] was not the *sample-size*
bottleneck. Both findings stand; they concern different constraints.

**RECOMMENDATION: primary entry band [40, 60] calendar days, tie-break
anchor 50.** It satisfies the requirement absolutely (0.00% expiring before
exit; worst observed case still has 13 days of life left at exit), and it
carries **~36% more call candidates than [45,60]** (37,410 vs 27,594) while
giving up only ~1.3pp of quote survival. [45,60] is marginally cleaner but
materially smaller; the sample difference is the deciding factor.

**Disclosure that must accompany this change:** the traded instrument's
tenor (40–60 days) no longer matches V3's 30-day standardized IV tenor.
The *holding period* still matches V3's 30-calendar-day forecast horizon
exactly, which is what the forecast is about — but the option's own vega
now sits at a longer tenor than the signal was measured at. This is a real
mismatch, forced by the fixed-exit requirement, and it should be stated in
§7.1 rather than discovered later.

## 2. Capacity across five NAV levels — recommend $100,000

Band [40,60], 15% ceiling, equal-vega sizing, whole contracts, max 20
positions, min breadth 5. Effective breadth reported two ways: filled
position count, and premium-weighted inverse-HHI.

| NAV | median filled (C/P) | median eff-HHI (C/P) | invested days (C/P) | **median invested-day utilization (C/P)** | median contracts/position (C/P) |
|---|---|---|---|---|---|
| **$100k** | 17 / 13 | 14.56 / 10.53 | 1,153 / 1,086 | **22.76% / 24.38%** | 8 / 8 |
| $250k | 17 / 13 | 13.92 / 9.71 | 1,153 / 1,086 | 20.51% / **19.87%** | 16 / 15 |
| $500k | 17 / 13 | 12.38 / 8.25 | 1,153 / 1,086 | 16.82% / 14.81% | 23 / 20 |
| $1M | 17 / 13 | 10.34 / 6.88 | 1,153 / 1,086 | 12.49% / 10.03% | 30 / 22 |
| $2M | 17 / 13 | 8.25 / 5.67 | 1,153 / 1,086 | 8.13% / 6.33% | 33 / 23 |

**RECOMMENDATION: $100,000.** It is the smallest level tested and the only
one where **both** sides clear the 20% median invested-day utilization
floor (22.76% / 24.38%). $250k fails on the put side by 0.13pp (19.87%).
Breadth (17/13 filled, eff-HHI 14.6/10.5) and invested days (1,153/1,086)
clear their floors at every level — both are NAV-invariant here, because
what fills a slot is candidate availability and the 20-position cap, not
capital. No liquidity participation limit was relaxed to reach this.

**The $100k whole-contract rounding check: the anticipated non-linearity
was looked for and is NOT present.** `rounding_up_forced` is **0.00%** at
$100k — no position was force-rounded up from a sub-half-contract vega
target — and median filled positions is **identical (17 calls / 13 puts) at
every NAV level**, so effective position count does not diverge from
proportional scaling at the small end at all.

**The real non-linearity runs the opposite way, at the LARGE end.** Median
contracts per position goes 8 → 16 → 23 → 30 → 33 across $100k → $2M: a
**20× increase in NAV buys only ~4.1× the contracts.** The cause is the
absolute OI/volume participation caps, which bind on 2.0% of call
candidates at $100k but **48.4% at $2M** (puts: 4.1% → 69.0%). Capital
above roughly $250k is substantially unusable at this liquidity tier — it
cannot be deployed without exceeding the participation limits. Stated
plainly: **this book does not scale**, and the constraint is the underlying
option market, not the sizing rule.

**This is a CAPACITY-FEASIBILITY choice, not a RISK-SIZING choice.** It
answers only "can the book stay adequately busy and deploy its capital
without exceeding participation limits." It does **not** answer "is
$100,000 the right amount of capital given the edge's actual magnitude" —
that question requires the edge's size and volatility, which are returns,
which this exercise cannot see by design. **The recommended NAV must be
revisited once real V4 returns exist.** A $100k book that is capacity-
feasible could still be far too large or far too small relative to a real
edge, and nothing here speaks to that.

## 3. Spread ceiling — 10% passes the stated test, with a flagged tension

The decision rule as specified: *use 10% if both sides still meet the
locked sample and breadth minimums at item 2's chosen NAV; otherwise retain
15%.* Applying it literally at [40,60] / $100k:

| ceiling | calls: filled / invested days | puts: filled / invested days | verdict on stated test |
|---|---|---|---|
| 15% | 17 / 1,153 | 13 / 1,086 | passes |
| **10%** | **11 / 1,058** | **9 / 985** | **passes — both sides clear breadth ≥5 and ≥250 invested days** |

**RECOMMENDATION: adopt the 10% ceiling** — and note it is independently
reinforced by §0: now that the design pays a *round-trip* option spread
rather than one half-spread, spread cost matters roughly twice as much as
it did under the superseded design, and spread is precisely what killed K1
(`prereg_V4.md` §1.2). Tightening from 15% to 10% brings the round-trip
cost back to ~1.33× the superseded design's rather than ~2×.

**Flagged tension, stated rather than smoothed over.** At the 10% ceiling,
median invested-day utilization at $100k falls to **17.44% (calls) / 18.59%
(puts)** — below the 20% floor that selected $100k in the first place. The
20% test in item 2 was run at the incumbent 15% ceiling, before item 3
tightened it. Under a 10% ceiling, **no tested NAV reaches 20%
utilization**; $100k remains the closest. The two criteria are satisfiable
sequentially, as specified, but not jointly.

I recommend accepting the ~17–19% utilization and adopting 10%, because
utilization is a capacity-efficiency measure, not a validity or risk
measure, and missing it by ~2pp is far less consequential than paying 50%
more spread on every entry *and* every exit in a design whose one
documented prior failure was spread-driven. **This deviates from the
literal ≥20% floor and is therefore flagged for your decision rather than
taken silently.**

## 4. Dynamic vs fixed liquidity cap — supplement, do not replace

| rule | median cap (contracts) | cap ≥ 1 contract | candidates passing fixed eligibility |
|---|---|---|---|
| fixed cap (10% OI, 20% of 5-day volume sum) | **31** | 100.00% | 90.80% |
| dynamic cap (1% OI, 5% of trailing-20d avg volume) | **1** | **74.40%** | — |

The dynamic rule is roughly **30× stricter**. Its effect on the book,
measured directly ([40,60], 15% ceiling):

| NAV | utilization, fixed cap (C/P) | utilization, DYNAMIC cap (C/P) | median contracts/position, dynamic |
|---|---|---|---|
| $100k | 22.76% / 24.38% | **9.41% / 7.07%** | 2 / 2 |
| $500k | 16.82% / 14.81% | 2.70% / 1.93% | 3 / 2 |
| $2M | 8.13% / 6.33% | **0.69% / 0.48%** | 3 / 2 |

**Are the absolute OI/volume minimums redundant with the dynamic rule? No —
and the reason is specific.** The dynamic rule is a `min()` of a volume leg
and an OI leg. The OI leg can be satisfied by **stale open interest with no
recent trading at all**: a contract with 100,000 open interest and zero
recent volume yields a 1,000-contract dynamic cap while failing C7's actual-
trading floor entirely. C6/C7 encode "this contract genuinely trades"; the
dynamic rule encodes "given it trades, how much can I take." Those are
different questions, and only the second is a capacity formula.

**RECOMMENDATION: supplement, not replace.** Keep C6/C7 as eligibility
floors (unchanged, already locked), and **replace the current sizing cap
(10% OI / 20% of 5-day volume sum) with the dynamic rule** — because taking
10% of a contract's entire open interest in a single trade is not a
realistic participation assumption and never was.

**The cost of doing so must be stated at the same volume as the
recommendation:** adopting the dynamic cap cuts median invested-day
utilization to **9.41% / 7.07%** at $100k, and makes the 20% floor
unreachable at every NAV tested. **If both item 3's 10% ceiling and item
4's dynamic cap are adopted together, utilization falls further still. That
specific combination was NOT measured** (the dynamic cap was run only at
the 15% ceiling); scaling the two measured effects suggests roughly 7%, but
that is an extrapolation, not a measurement, and I recommend measuring it
before final lock if you adopt both.

## 5. Quote survival and the missing-quote rule

Measured across the full ~21-session hold and the prior 20 sessions, band
[40,60]:

| metric | calls | puts |
|---|---|---|
| hold path fully quoted (no missing day) | 89.66% | 93.48% |
| hold valid-quote fraction, p5 | 0.7895 | 0.9048 |
| hold worst consecutive gap: median / p95 / max | 0 / 4 / 22 | 0 / 1 / 19 |
| **bid present on the scheduled exit session** | **91.75%** | **94.47%** |
| pre-entry ≥90% of prior 20 sessions valid | 62.04% | 67.99% |
| pre-entry max consecutive gap ≤ 1 | 61.64% | 67.71% |
| **passing the full locked pre-entry rule (both)** | **61.62%** | **67.70%** |

The mid-hold rule matters: ~8% of calls and ~5.5% of puts have **no bid on
the scheduled exit day**, so the "exit at the first subsequent valid bid,
else settle at intrinsic" branch is a live path, not a formality.

**A caveat on the pre-entry rule that materially affects its interpretation
— flagged because it would otherwise be mistaken for pure illiquidity.**
The locked rule would exclude ~38% of calls and ~32% of puts. But my
measurement counts "no row in the data" as "missing quote," which conflates
two different things: a contract that **existed but was not quoted**
(genuine illiquidity), and a contract that **did not exist yet** (listing
recency). A 40–60 DTE contract had 68–88 DTE twenty sessions earlier, and
may simply not have been listed. The distributional shape supports this
reading: pre-entry valid fraction has **median 1.0 but Q1 = 0.50** (calls),
which is the signature of a cluster of contracts listed partway through the
lookback window, not of smooth sporadic illiquidity — a contract listed
exactly 10 sessions ago scores exactly 0.50. This is an **inference from
the distribution, not a direct measurement**; separating the two would need
another pass keyed on each contract's first-quote date.

I recommend **adopting the rule as locked, with this disclosure**, since
biasing toward established monthly-cycle contracts is defensible on its own
merits for a strategy that must sell the position 30 days later — but it
should be an acknowledged design choice, not an unexamined side effect.

## 6. Retained unchanged, confirmed

Calls primary / puts secondary; daily contemporaneous midpoint-IV delta as
primary hedge; 10% borrow primary with 3% / 25% sensitivities; fixed
30-calendar-day exit with no discretionary stop or target; ask-to-bid as
primary execution; paired compression-minus-benchmark gate at NW t(Diff) ≥
2.0. None of these is altered by anything above. Note that "ask-to-bid" is
now *load-bearing* in a way it was not before — see §0.

---

## Summary of recommendations

| item | recommendation | status |
|---|---|---|
| 1 | entry band **[40,60]**, anchor 50; [25,38] fails on expiration-before-exit | measured, decisive |
| 2 | NAV **$100,000**; capacity-feasibility only, revisit once returns exist | measured |
| 3 | spread ceiling **10%** | passes stated test; **utilization tension flagged for your decision** |
| 4 | dynamic cap **supplements** (replaces the sizing cap), C6/C7 retained | measured; **combined 10%+dynamic case unmeasured** |
| 5 | adopt the locked missing-quote rule | measured; **listing-recency caveat flagged** |
| 6 | retained unchanged | confirmed |

**Three items need your decision before I finalize anything:** (a) accept
~17–19% utilization under the 10% ceiling, or retain 15% to hold the 20%
floor; (b) adopt the dynamic sizing cap knowing it cuts utilization to
~7–9%, and whether to measure the combined 10%+dynamic case first; (c)
accept the pre-entry rule's listing-recency conflation, or commission the
extra pass to separate it.

**`results/prereg_V4.md` has NOT been modified.** Proposed amendments are
specified in `results/57_v4_proposed_amendments.md` and will be applied
only on your confirmation. The document is **not** marked LOCKED.
