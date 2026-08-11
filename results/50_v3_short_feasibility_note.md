# V3-Short Data Feasibility Audit — Findings and Recommendation

**Status: coverage measurement only. Does NOT authorize building V3-short.
Does NOT change V4 (`results/prereg_V4.md`, DRAFT) in any way. No return,
realized-variance value, option return, P&L, or regression coefficient was
computed anywhere in this audit.** `results/gate_log.md` was not touched.

**Script:** `src/50_v3_short_data_feasibility.py`, run 2026-08-03. Full
886,525,545-row scan of `opprcd.csv` (80.55 GB). Machine-readable detail:
`results/50_v3_short_data_feasibility.json`.

---

## 1. The question

`prereg_V3.md` §3.3-F found that the standardized volatility surface's
10-day node has only **4.30%** both-tenor coverage against the V1/V2
universe (1,733,857 DEV stock-days, decile 6–8) — far too sparse to
support a matched-horizon test, which is why V3's primary moved to 30
calendar days. A 20-calendar-day construction interpolated between the
surface's 10-day and 30-day nodes would inherit that same 4.30% sparsity,
since the 10-day node is the binding constraint on any interpolation that
needs it.

This audit asks a different, narrower question: **do actual traded-contract
quotes in `opprcd.csv`, at contracts genuinely close to 20 days to
expiration, support materially better coverage than the standardized
surface's 10-day node** — enough to make a separately pre-registered
20-day test worth designing?

## 2. Method, in one paragraph

For every stock-date in the same 1,733,857-stock-day base population V3
used, and separately for calls and puts, the actual listed contract nearest
to 20 calendar days to expiration was selected via a fixed, deterministic
tie-break (distance from 20 DTE → distance from 0.50 |delta|, computed from
`opprcd`'s own `impl_volatility` since no vendor delta field exists in this
file → relative spread → open interest → recent volume → contract ID).
Two bands were evaluated: a **primary** band (18–22 DTE) and a broader
**diagnostic** band (15–25 DTE, reported for context only — it does not
and cannot become a future primary spec merely because it has more rows).
A contract counts as **usable** only if it clears the full stated funnel:
finite, positive, crossed bid/ask → satisfies no-static-arbitrage bounds →
a midpoint IV inversion (vectorized Newton–Raphson, bounds
[0.0001, 5.0]) actually converges. No future price, return, or realized
variance was used anywhere in this construction — the one date-based check
(whether a complete future trading calendar exists through the horizon)
verified row *existence* only, never a return or price *value*.

## 3. Headline coverage, apples-to-apples against V3's own cited figures

All percentages below are of the **same 1,733,857-stock-day base
population** V3's 91.45%/4.30% figures were measured against, so they are
directly comparable.

| band | side | eligible (contract exists) | **usable IV** (full funnel) |
|---|---|---|---|
| primary 18–22 DTE | calls | 14.18% | **12.84%** |
| primary 18–22 DTE | puts | 14.18% | **13.04%** |
| diagnostic 15–25 DTE | calls | 40.08% | **35.95%** |
| diagnostic 15–25 DTE | puts | 40.08% | **36.48%** |

For reference: V3's 30-day standardized node = 91.45%; V3's 10-day
standardized node (what a 20-day interpolation would have inherited) =
**4.30%**.

**Real contract-level coverage near 20 DTE is roughly 3× V3's sparse
10-day node at the primary band, and roughly 8× at the diagnostic band.**
The premise that motivated this audit — that actual quotes might clear far
better than the standardized surface's thinnest node — holds.

**Funnel detail, primary band (both sides land in the same range):**
zero-bid rate 7–9%; valid two-sided quote ~91–93%; no-arbitrage failure
rate (of valid quotes) 0.65–0.70%; IV-inversion failure rate (of rows that
reach the step) ~0.05%. The funnel's own internal attrition is small and
well short of where the standardized-node comparison lives — almost all of
the gap between "eligible" and "V3's 91.45%" is simply "no listed contract
existed that close to 20 DTE that day," not a quote-quality failure.

## 4. Coverage is not vacuous or concentrated in one slice

- **By year** (primary band, calls): 28,311 (2015) rising to 42,512 (2021),
  every year represented, no year collapsing toward zero. Coverage grows
  over the sample as options listings on mid-cap names became more common
  — a real trend, not a construction artifact.
- **By size decile** (primary, calls): 78,044 / 82,042 / 85,781 across
  deciles 6/7/8 — reasonably balanced, not concentrated in one size band.
- **By earnings bucket** (primary, calls): 166,533 outside any 20-day
  earnings window, 30,149 / 28,002 / 19,463 across the three within-window
  buckets — a sensible split, not degenerate.
- **Cross-sectional density**: 222,686 usable primary-band call
  observations spread across ~1,763 DEV trading days is a mean
  cross-section of roughly 126 names/day — comfortably above this
  project's `MIN_XSEC = 30` Fama-MacBeth floor on the great majority of
  days, and nearly 3× denser than V3's own 10-trading-day sensitivity
  arm's mean cross-section of 45.2 names/day (`results/49_v3_incremental_variance_test.json`).
  A V3-short built on this population would not face the "424 usable dates,
  9× wider standard error" problem that made V3's own 10-day sensitivity
  arm uninformative (prereg_V4.md §5.1).

## 5. What this audit does NOT resolve — left to a future V3-short's own design

- **Per-side, not both-sides-present.** This audit measured call coverage
  and put coverage independently, matching the task's own "report
  separately for calls and puts" instruction. It did **not** compute the
  intersection (both a usable call AND a usable put on the same
  stock-date), which is what V3's own ATM construction required (mean of
  call-side and put-side IV, both sides present). If a future V3-short
  wants to replicate that convention, the both-sides-present coverage
  would be smaller than either one-sided figure above — a design choice,
  and a further measurement, for that phase's own pre-registration, not
  decided or estimated here.
- **DTE variation inside the band is not horizon-matching.** The primary
  band spans 18–22 actual calendar days; V3's own 30-day primary is exact
  because it reads the surface's native 30-day tenor directly. A future
  V3-short would need its own real-variance horizon construction matched
  to each row's *actual* selected DTE (analogous to V3's `n_t` per-row
  session-count approach), not a single fixed 20-day assumption — this is
  exactly the kind of item the task's own "FUTURE V3-SHORT DESIGN LIMIT"
  section flags as needing to be locked before any outcome is computed.
- **Liquidity beyond IV-validity was not screened.** This audit's funnel
  stops at "does a usable IV exist," not at whether the specific selected
  contract would clear a trading-liquidity bar (V4 draft §6.3's C4–C7
  thresholds — minimum price, maximum spread, minimum OI, minimum recent
  volume). A V3-short that intends to trade, rather than only measure
  information content, would need its own liquidity gate, likely reducing
  these figures further.

## 6. Recommendation

**Coverage at the 18–22 DTE primary band is adequate to justify drafting a
separate, standalone V3-short pre-registration.** It is not as abundant as
V3's own 30-day primary (91.45%), and any V3-short would need to disclose
that difference plainly rather than imply parity with V3. But it is dense
enough — roughly 3× the standardized 10-day node, with a cross-sectional
count per day well above this project's estimation floor, and stable
across years and size deciles — to be a genuinely different, better-founded
construction than the interpolation path V3 rejected.

**This recommendation carries no authority of its own.** Per the task's own
framing, it does not authorize building V3-short, and a V3-short
pre-registration — if the owner wants one drafted — would still need to
lock, before any outcome is computed: the realized-variance horizon
construction matched to each row's actual DTE; the exact IV construction
(one-sided vs. both-sides-present, and how that interacts with V1/V2's
`Compression` and control set); the forecast model and its significance
bars; treatment of calls and puts (separately, per the same volatility-
branch reasoning `prereg_V4.md` §1.3 already established); and complete-
window requirements. None of that is decided here. **V3-short, if it is
ever authorized and locked, remains an independent gate: it cannot revise
V3, cannot rescue V4, and cannot retroactively justify a shorter V4 holding
period** (`prereg_V4.md` §5.2's reasoning against a short trading arm
stands unchanged by anything in this audit).
