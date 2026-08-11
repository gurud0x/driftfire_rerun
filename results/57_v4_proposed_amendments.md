# Proposed Amendments to `results/prereg_V4.md`

**NOT YET APPLIED.** `results/prereg_V4.md` is unmodified and is **not**
marked LOCKED. Every amendment below is specified precisely enough to apply
mechanically on your confirmation. Evidence: `results/56_v4_final_reconciliation_memo.md`,
`results/55_v4_design_reconciliation.json`.

Amendments are grouped by whether they are **forced** (the document is
currently factually wrong given the fixed-30-day exit) or **recommended**
(a design choice supported by measurement, which you may decline).

---

## GROUP A — FORCED: the fixed 30-calendar-day exit

These are not optional. With a fixed 30-day exit and ask-to-bid execution,
the current text asserts things that are false.

### A1. §5.3 Exit rule — replace the section body

**Current (excerpt):**
> Exit is at expiration and at no other time. Specifically:
> - The option is held to its expiration date. Terminal value is **intrinsic at the expiration-date underlying close** …
> - **No option-side exit spread is charged**, because no option-side exit trade occurs. This is a genuine structural advantage over K1 …

**Proposed replacement:**
> Exit is at a **fixed 30 calendar days** after entry, and at no other time.
> Specifically:
> - The **exit session** is the last trading session on or before
>   `entry_date + 30 calendar days` — V3 §3.3-F's own `n_t` convention,
>   reused unchanged. Measured over DEV, the hold spans **17–22 trading
>   sessions** (mean 20.67, mode 21).
> - The option position is **closed by selling the contract** at the exit
>   session, at the execution tier's exit price (§9). It is **not** held to
>   expiration; under §7.1's locked band the contract still has a median of
>   **21 days** of remaining life at exit (minimum observed: 13 days).
> - **An option-side exit spread IS charged.** This supersedes the earlier
>   draft's hold-to-expiry design, which charged only one half-spread. The
>   fixed-exit design charges a full round trip (buy at ask, sell at bid),
>   roughly doubling option transaction cost at a given ceiling — the single
>   most consequential cost consequence of this change, and the reason §6.3's
>   spread ceiling is tightened (Group B).
> - **Missing-quote handling at exit** is governed by §7.8, not by this
>   section: on the measured sample, a valid bid is present on the scheduled
>   exit session for **91.75% of calls / 94.47% of puts**, so the fallback
>   path is a live branch, not a formality.
> - **No stop-loss and no profit target.** Unchanged, and for the unchanged
>   reason: either would be a tuned exit parameter with no pre-registered
>   justification.
> - If the exit session would fall after DEV_END (2021-12-31), the trade is
>   **not entered**. Trades with an incomplete window are never truncated or
>   padded (E1 §4.3 convention).

### A2. §7.1 — delete the false justification sentence

**Current:**
> **Why ~30 DTE held to expiry:** it makes the instrument's tenor match the signal's horizon exactly (V3's 30-calendar-day primary), avoiding the signal/instrument mismatch V3 §3.3-F spent an entire section eliminating, **and** it removes the exit-side option spread (§5.3). Both reasons are design reasons, recorded before any V4 P&L exists.

**Proposed replacement:**
> **Why a 40–60 DTE contract on a fixed 30-day hold, and the mismatch this
> creates — stated rather than left to be discovered.** The *holding period*
> matches V3's 30-calendar-day forecast horizon exactly, which is the
> horizon the forecast is actually about. The *instrument's tenor* (40–60
> days) does **not** match V3's 30-day standardized IV tenor — the option's
> vega sits at a longer tenor than the signal was measured at. This is a
> real mismatch, accepted deliberately because the fixed-exit requirement
> (§5.3) makes it unavoidable: any band short enough to match the 30-day
> tenor permits expiration before the scheduled exit. The earlier draft's
> claim that this design "removes the exit-side option spread" is
> **withdrawn** — it was true only of the superseded hold-to-expiry design.

### A3. §9 — correct the no-exit-price claim and add the exit side

**Current:**
> All three tiers are computed on the **same** trade list, hedge path, and equity costs. Only the option entry price differs. Because the position is held to expiry, **there is no option-side exit price in any tier** (§5.3).

**Proposed replacement:**
> All three tiers are computed on the **same** trade list, hedge path, and
> equity costs. Only the option **entry and exit** prices differ. The
> position is sold at the fixed 30-day exit (§5.3), so **every tier carries
> an option-side exit price** — superseding the earlier draft's
> hold-to-expiry statement that none did.

**And the tier table becomes:**

| tier | option entry price | option exit price | status |
|---|---|---|---|
| Midpoint | `(bid+offer)/2` | `(bid+offer)/2` | frictionless diagnostic only — NOT the pass criterion |
| Partial-spread | `mid + 0.50×(ask−mid)` | `mid − 0.50×(mid−bid)` | realistic sensitivity |
| **Ask-to-bid** | **`best_offer`** | **`best_bid`** | **PRIMARY** — conservative, fully marketable both ways |

Plus the amended reasoning: the full-ask tier no longer charges "precisely
one half-spread over the life of the position" (that sentence is
withdrawn); it charges a **full round-trip spread**, which at §6.3's
proposed 10% ceiling is ~10% of mid per completed trade.

### A4. §9.1 — `ExpectedOptionCost` must include both sides

**Current:**
> **`ExpectedOptionCost`** = `100 × (entry price under the PRIMARY tier − mid)` = `100 × (ask − mid)` per contract. No exit-side term.

**Proposed replacement:**
> **`ExpectedOptionCost`** = `100 × [(ask − mid) + (mid − bid)]` = `100 ×
> (ask − bid)` per contract — the **full round-trip** spread under the
> primary ask-to-bid tier. The earlier draft's "No exit-side term" is
> withdrawn (§5.3). Since §9.1's filter requires expected edge to exceed
> `1.5 × (option cost + hedge cost)`, this materially raises the entry bar
> — which is the intended, conservative direction.

### A5. §7.6 costs table — add the option exit cost row

Add: `| option exit | per the execution tier in §9 — a full round-trip
spread is now charged (§5.3) |`, and delete the current row's trailing
"no option-side exit cost (§5.3)".

### A6. §7.8 — replace the missing-data rule

**Replace the current table and its "drop the entire trade" convention
with the locked rule:**

> | situation | rule |
> |---|---|
> | pre-entry quote continuity | **entry-eligible only if valid quotes exist on ≥ 90% of the prior 20 sessions AND no more than 1 consecutive missing session** |
> | missing/invalid quote on the **entry** date | candidate ineligible, counted |
> | missing CRSP close on any hold session | trade dropped and counted (the hedge cannot be marked) |
> | **1 missing contract quote mid-hold** | **carry the last valid delta forward for at most one session**; no rebalance, no cost charged |
> | **2 consecutive missing quotes mid-hold** | **exit at the first subsequent valid bid** |
> | **no bid returns before expiration** | **settle at intrinsic value** |
>
> **A trade is NEVER dropped solely because quotes went missing mid-hold.
> This deliberately replaces the "drop and count" convention used elsewhere
> in this project (V3 §3.3, `src/43`), and the reason is specific:** mid-hold
> gaps in option quotes are a *feature of the exact liquidity tier V4 is
> studying*, not a data defect. Dropping those trades would bias the sample
> toward only the smoothest-quoted names and would systematically overstate
> the tradability of the universe — the opposite of what a cost-bearing
> monetization test should do. Pre-entry gaps are treated differently, and
> may exclude a candidate, because they are observable **before** committing
> capital; mid-hold gaps are not.
>
> **Measured incidence (`results/56_v4_final_reconciliation_memo.md` §5):**
> hold path fully quoted 89.66% calls / 93.48% puts; worst consecutive gap
> median 0, p95 4 (calls) / 1 (puts); valid bid present on the scheduled
> exit session 91.75% / 94.47%; passing the full pre-entry rule 61.62% /
> 67.70%.
>
> **Disclosed limitation of the pre-entry rule.** The measurement counts "no
> row in the data" as "missing quote," conflating a contract that existed
> but was unquoted (illiquidity) with one that **did not exist yet**
> (listing recency) — a 40–60 DTE contract had 68–88 DTE twenty sessions
> earlier. The distribution (median 1.0, Q1 0.50) is the signature of the
> latter. This is an inference from shape, not a direct measurement;
> separating the two needs a further pass keyed on each contract's
> first-quote date. The rule is adopted with this disclosure, accepting that
> it biases toward established monthly-cycle contracts.

---

## GROUP B — RECOMMENDED: measured design choices

### B1. §7.1 — entry DTE band [25,38] → [40,60]

| row | current | proposed |
|---|---|---|
| days to expiration at entry | **[25, 38] calendar days**, target 30 | **[40, 60] calendar days**, target 50 |

And tie-break rule 2 changes from `smallest |DTE − 30|` to `smallest
|DTE − 50|`.

**Evidence:** [25,38] permits **9.28% / 9.45%** (calls/puts) of selected
contracts to expire before the scheduled exit and leaves the median
survivor with **1 day** of life at exit; [40,60] gives **0.00%** and a
median of **21 days** (min 13). [40,60] carries ~36% more call candidates
than [45,60] at ~1.3pp lower quote survival.

### B2. §6.3 C5 — spread ceiling 15% → 10%

| criterion | current | proposed |
|---|---|---|
| C5 maximum relative spread | `(ask − bid)/mid ≤ 0.15` | `(ask − bid)/mid ≤ 0.10` |

**Evidence:** at [40,60] / $100k both sides still clear the breadth and
invested-day minimums (calls 11 filled / 1,058 invested days; puts 9 /
985), which is the stated decision test. Independently reinforced by A3:
the design now pays a round-trip spread, so spread cost matters ~2× more
than under the superseded design.

**⚠ FLAGGED — this is the amendment most in tension with another
criterion.** Adopting 10% drops median invested-day utilization at $100k to
**17.44% / 18.59%**, below the 20% floor used to select $100k. Under a 10%
ceiling **no tested NAV reaches 20%**. Recommended anyway (see memo §3), but
this is a genuine deviation and is yours to accept or decline.

### B3. §8 — NAV $2,000,000 → $100,000

| row | current | proposed |
|---|---|---|
| capital denominator | **$2,000,000 fixed** | **$100,000 fixed** |
| aggregate vega cap | ≤ $10,000/pt on $2M | ≤ **$500/pt** on $100k (same 0.5% of capital) |
| aggregate gamma cap | $100,000 on $2M | **$5,000** on $100k (same 5% of capital) |
| target vega per position | $500/pt | **$25/pt** (same 0.5%/20 formula) |

All caps keep their existing *fractional* definitions; only the absolute
figures rescale. Max 20 positions, sector cap 4/20, and the min-5 invested
day threshold are unchanged.

**Evidence:** $100k is the smallest tested NAV where both sides clear the
20% utilization floor (22.76% / 24.38% at the 15% ceiling). Breadth (17/13
filled; eff-HHI 14.6/10.5) and invested days (1,153/1,086) clear at every
level. **Add to §8 the standing disclosure that this is a
capacity-feasibility choice, not a risk-sizing choice, and must be revisited
once real V4 returns exist.**

### B4. §8 — replace the sizing capacity cap with the dynamic rule

| row | current | proposed |
|---|---|---|
| capacity cap | contracts ≤ min(vega-implied, **10% of entry open interest**, **20% of trailing-5-day volume**) | contracts ≤ min(vega-implied, **1% of entry open interest**, **5% of trailing-20-day AVERAGE contract volume**) |

C6 (OI ≥ 100) and C7 (volume floors) are **retained unchanged** as
eligibility screens — they are not redundant with the dynamic rule, because
the rule's OI leg can be satisfied by stale open interest with no recent
trading at all.

**⚠ FLAGGED.** This cuts median invested-day utilization at $100k from
22.76%/24.38% to **9.41%/7.07%**, and makes the 20% floor unreachable at
every NAV. **The combined effect of B2 (10% ceiling) and B4 (dynamic cap)
together was NOT measured** — only each separately. I recommend measuring
that combination before applying both.

### B5. §8.1 — replace the capacity findings block

Replace with the five-NAV table from memo §2, including the corrected
non-linearity finding: whole-contract rounding at $100k forces **0.00%** of
positions upward and median filled positions is identical (17/13) at every
NAV, so the anticipated small-NAV rounding divergence **does not occur**;
the actual non-linearity is at the large end, where a 20× NAV increase buys
only ~4.1× the contracts because participation caps bind on 2.0% → 48.4%
of candidates.

### B6. §14 — status and open items

Update to record: Group A applied (forced by the fixed-exit change), Group B
per your decisions, the three flagged tensions from memo §7, and the
still-open §13 data-pull authorization. **Do not mark LOCKED** until you
confirm.

---

## Unchanged, explicitly confirmed

§11.1 calls-primary/puts-secondary; §7.2 daily contemporaneous midpoint-IV
delta; §7.6 borrow 10% primary with 3%/25% sensitivities; §5.3's no-stop/
no-target; §9's ask-to-bid as primary; §11 item 6's compression-minus-
benchmark gate at NW t(Diff) ≥ 2.0; §11's six-item bar; §4.1's Model B.
