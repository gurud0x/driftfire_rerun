# CLAUDE.md — DriftFire Research Pipeline Rules

Rules for any AI agent (Claude Code, Copilot, Claude chat) working in this repo.
These are non-negotiable. If a task conflicts with a rule, stop and flag it —
do not silently work around it.

## Project identity

Systematic equity research pipeline. The deliverable is a *defensible research
process*, not a profitable backtest. Null results are valid outputs and get
written up with the same care as positive ones.

## The two layers (never blur them)

- **SIGNAL LAYER** — produces information: reversal ranks, Kronos volatility
  forecasts, factor exposures, (later) OptionMetrics IV features. Output is
  always a parquet of per-name, per-date forecasts/ranks. All edge claims live
  here and must clear the gate here, in equity space, net of costs.
- **EXPRESSION LAYER** — converts a gated signal into positions: shares vs.
  long call, sizing, entry timing. It is forbidden to run before the signal
  layer passes its gate. The shares-vs-call rule is: long call only when
  Kronos E[RV] exceeds option IV by the pre-registered threshold; otherwise
  shares. Expression logic must never feed back into signal construction.

## Research discipline (hard rules)

1. **Pre-registration before data contact.** No forward-return analysis runs
   until the phase's pre-registration doc is committed with all numeric
   fields filled. If a `[FILL IN]` remains, the task is blocked.
2. **Gate-locked backtest.** `src/backtest/` is not imported, run, or modified
   unless `results/gate_log.md` shows PASS for the current phase.
3. **Holdout is spent once.** Never re-run, retune, or re-split after a
   holdout pass. Dev = 2015–2021, holdout = 2022–2025 (or as the phase doc
   states). Any code path that touches holdout dates outside the single
   promotion pass is a bug.
4. **No post-hoc features.** Features are enumerated in the pre-registration
   doc. Adding one mid-phase requires a new phase, not a code edit.
5. **Lookahead prevention.** Every rolling/derived feature uses `shift(1)`.
   Entries fill at next-day open. Signals computed on day t trade day t+1.
6. **Costs are first-class.** 15 bps per side base case + 30 bps stress case
   on equities; mid-minus-half-spread fills on options. Turnover is reported
   with every result table.
7. **Delisting returns** are incorporated for any universe beyond S&P 500
   members. Zero delistings in a small-cap universe = pipeline bug.

## Data contracts

- All external data enters through `src/ingest/` only. Notebooks never read
  raw sources directly.
- Stores: `data/raw/` (frozen pulls + printed manifest), `data/processed/`
  (keyed on `PERMNO`, `DlyCalDt`), `results/` (tables, gate log, figures).
- CRSP is CIZ format (Flat File 2.0): columns are `DlyCalDt`, `PERMNO`,
  `DlyPrc`, etc. Never use legacy lowercase `date`/`permno` names. Verify
  column names against the actual parquet before writing code — print
  `df.columns`, do not assume.
- Kronos runs in a separate Python 3.10+ env and communicates exclusively via
  `data/processed/kronos_forecasts.parquet`. The core pipeline must skip
  gracefully when that file is absent. Same contract pattern applies to any
  future OptionMetrics features.
- Factor data (Fama-French 5 + momentum + short-term reversal) lives in
  `data/raw/ff_factors/` and is the mandatory benchmark set for alpha claims.

## Verification standard

Every module, when run, prints real numbers: shape, date range, unique
PERMNO count, names-per-month for universes, NaN counts for features.
"Looks good" or a bare `assert` is not verification. Agents must include
these prints in any code they write and show the output before the task is
considered done.

## Environment quirks

- Python 3.9.6 venv on Mac; project lives under a OneDrive path with spaces —
  install packages inside notebooks via
  `subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])`.
- Tracking: GitHub Issues + Project board. Each phase = one milestone.

## Things agents must never do

- Retune anything using holdout results.
- Invent column names, tickers, or data that wasn't printed.
- Wrap a failing signal in options to "fix" it.
- Suppress exceptions or NaN warnings to make a run complete.
- Cite YouTube/social content in any research note. Videos are hypothesis
  generators; citations come from the papers (McLean & Pontiff 2016; Nagel
  2012; Lou, Polk & Skouras 2019; Moreira & Muir 2017; Grinold & Kahn).

## Known issues

- 01_ingest.py output has 716 duplicate PERMNO-day rows from multi-distribution ex-dates (identical on all price/return fields). Defended against in 03_signal.py via halting PASS-gate before dedup. Fix at source deferred — not blocking.
