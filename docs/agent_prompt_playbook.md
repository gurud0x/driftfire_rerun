# Agent Prompt Playbook — DriftFire From-Scratch Build

One entry per pipeline component. Each entry: which agent, the prompt to
paste, and what YOU verify before accepting the output. Work top to bottom;
every prompt assumes `CLAUDE.md` is in the repo root so agents inherit the
rules automatically.

**Division of labor**
- **Claude (chat, this project):** research design, pre-registration drafting,
  reviewing agent output, interpreting results, write-ups.
- **Claude Code:** anything multi-file — modules, tests, refactors. It reads
  CLAUDE.md automatically.
- **Copilot:** inline completions only while you type inside a module Claude
  Code scaffolded. Never let it originate research logic.
- **Figma MCP (generate_diagram / use_figma):** pipeline diagrams for the
  research note and README.

---

## 0 — Pre-registration (BLOCKING — nothing runs before this)

**Agent:** Claude chat. This is a reasoning task; no code agent involved.

**Prompt:**
> Draft `docs/PhaseX_PreRegistration_LongOnly_Reversal.md`. Universe: CRSP
> deciles 6–8 by NYSE breakpoints, price >= $5, share codes 10/11. Signal:
> past 5-day return, cross-sectional rank, shift(1), long bottom decile only.
> Hold 5 days, next-day open entry. Costs 15/30 bps cases. Alpha test:
> intercept vs FF5 + MOM + ST_Rev, gate t >= 2.0 net on dev (2015–2021),
> sign-consistent single holdout pass (2022–2025). Secondary pre-registered
> decomposition: overnight vs intraday legs of the same signal. Expression
> layer (separate section, dormant until gate PASS): long call iff
> Kronos E[RV21] / IV30 >= [FILL IN], else shares. Flag every number I still
> owe you as [FILL IN] and refuse to mark the doc complete until none remain.

**Verify:** git commit timestamp of this doc precedes every analysis commit.
No `[FILL IN]` remains.

## 1 — Ingest & QA

**Agent:** Claude Code.

**Prompt:**
> Read CLAUDE.md. Build `src/ingest/`: (a) `crsp_loader.py` that reads the
> existing crsp parquets, prints shape/date range/unique PERMNO/columns, and
> writes a manifest to `data/raw/manifest_crsp.txt`; (b) `ff_loader.py` that
> parses the Ken French daily FF5, momentum, and short-term reversal CSVs
> from `data/raw/ff_factors/` into one `data/processed/factors_daily.parquet`
> keyed on date, prints head/tail/date range. No forward returns are computed
> anywhere in this module.

**Verify:** run both; the printed CRSP columns include `DlyCalDt`, `PERMNO`,
open price, and delisting fields. Factor parquet date range covers 2015–2025
with no gaps > 5 business days.

## 2 — Universe builder

**Agent:** Claude Code.

**Prompt:**
> Read CLAUDE.md. Build `src/universe_smallcap.py`: month-end market cap from
> CRSP, NYSE breakpoints, assign deciles, keep 6–8, price >= $5, share codes
> 10/11, exchanges NYSE/AMEX/NASDAQ. Decile assignment from month t−1 applies
> through month t (point-in-time). Output
> `data/processed/universe_smallcap_monthly.parquet`. Print names-per-month
> as a series and flag any month deviating > 30% from its trailing 12-month
> mean. Also print delisting counts per year within the universe.

**Verify:** 800–1,500 names/month, stable; delistings per year in the dozens,
never zero.

## 3 — Signal engine (signal layer)

**Agent:** Claude Code to scaffold; Copilot for inline edits after.

**Prompt:**
> Read CLAUDE.md and the pre-registration doc. Build `src/signals_reversal.py`:
> past 5-day return per PERMNO, shift(1), cross-sectional decile rank within
> the universe each day. Merge `kronos_forecasts.parquet` if present (skip
> gracefully if absent). Output `data/processed/signals.parquet` with columns
> [PERMNO, DlyCalDt, rev5_rank, kronos_erv21 (nullable)]. Print NaN share per
> column and confirm via a spot check that rank on date t uses only data
> through t−1: print one PERMNO's raw returns and its rank inputs side by side.

**Verify:** the printed spot check shows no same-day return inside the rank
input. NaN share < 5% outside the first weeks of listing.

## 4 — Kronos forecast layer (separate env)

**Agent:** Claude Code, run inside the Kronos Python 3.10+ env.

**Prompt:**
> In this env only: script `kronos_batch_forecast.py` that loads Kronos-small,
> takes trailing 252-day OHLCV per PERMNO from an input parquet, produces
> 21-day realized-vol forecasts, writes
> `data/processed/kronos_forecasts.parquet` [PERMNO, DlyCalDt, kronos_erv21].
> Include a trailing-vol and a GARCH(1,1) baseline column so the vol-forecast
> horse race is in one file. Print forecast coverage %, and MAE of each
> forecaster vs realized vol on the DEV window only.

**Verify:** coverage > 90% of universe-days; Kronos MAE reported next to
baselines — if it doesn't beat trailing vol on dev, the expression layer
threshold defaults to shares-only and you note it.

## 5 — Alpha isolation

**Agent:** Claude Code.

**Prompt:**
> Read CLAUDE.md. Build `src/alpha_isolation.py`: form the long-only
> bottom-decile portfolio (5-day hold, next-day open, overlapping tranches),
> compute daily returns net of pre-registered costs incl. delisting returns,
> regress on FF5 + MOM + ST_Rev from factors_daily.parquet. DEV window only.
> Output `results/alpha_table_dev.csv` with gross/net raw mean, factor betas,
> intercept, Newey-West t. Also produce the overnight/intraday split of the
> same portfolio. Print annualized turnover.

**Verify:** the ST_Rev beta is reported (it will be large — that's the point);
the claim rides on the intercept t-stat, not raw return. Turnover printed.

## 6 — Gate checker

**Agent:** Claude Code.

**Prompt:**
> Read CLAUDE.md. Build `src/gate_check.py`: parse the numeric gate from the
> pre-registration doc, compare to results/alpha_table_dev.csv, append a
> timestamped PASS/FAIL line with the numbers to `results/gate_log.md`.
> Exit nonzero on FAIL so downstream scripts cannot chain past it.

**Verify:** deliberately feed it a fake failing CSV once and confirm it
blocks. Then run for real.

## 7 — Holdout pass + backtest (only on PASS)

**Agent:** Claude Code — but you type the run command yourself, once.

**Prompt:**
> Gate log shows PASS for this phase (check it; refuse if not). Run the
> identical alpha_isolation config on the holdout window, append results to
> the gate log, and unlock `src/backtest/` for a single equity-curve run with
> the pre-registered costs. No parameter differs from dev. Print a diff of
> the dev and holdout configs proving they match.

**Verify:** the printed config diff is empty except the date range.

## 8 — Expression layer (dormant until step 7 passes)

**Agent:** Claude chat to finalize the threshold design; Claude Code to build.

**Prompt (design, chat):**
> Signal passed gate. Before any OptionMetrics pull, finalize the expression
> pre-registration: E[RV21]/IV30 threshold for long call vs shares, tenor,
> strike rule (ATM), option cost model (mid minus half spread), and the
> comparison metric (net Sharpe of expressed portfolio vs shares-only).
> Challenge my threshold choice before locking it.

**Prompt (build, Claude Code):**
> Read CLAUDE.md. Build `src/expression_layer.py`: for each gated long
> signal, choose shares or 1 ATM call (30–45 DTE) per the locked threshold
> using kronos_erv21 and OptionMetrics IV30 from
> `data/processed/optionmetrics_iv.parquet` (skip gracefully if absent —
> shares-only fallback). Output both expressed and shares-only equity curves
> to results/. Print the fraction of trades expressed as calls.

**Verify:** shares-only fallback runs when the IV parquet is missing; call
fraction is printed and sane (not 0%, not 100%).

## 9 — Report & PDF

**Agent:** Claude chat (drafting + honest framing), then Claude Code for the
figure scripts.

**Prompt:**
> Draft the phase research note from results/: methodology from the
> pre-registration doc verbatim, alpha table, overnight/intraday
> decomposition, turnover/cost sensitivity, gate log excerpt, and — whatever
> the outcome — a limitations section. Frame a null as a finding. Grinold &
> Kahn IC × sqrt(breadth) framing in the discussion. No claims beyond the
> printed tables.

**Verify:** every number in the note traces to a file in results/.

## 10 — Diagrams (README + note)

**Agent:** Figma MCP via Claude chat.

**Prompt:**
> Regenerate the pipeline architecture diagram (generate_diagram, FigJam)
> reflecting the current repo, and export it for the README. If the diagram
> already exists, update it with use_figma instead of recreating.

**Verify:** node names match actual module filenames.

---

**Standing review prompt** (paste to Claude chat after any agent finishes a
component):
> Here is the code/output for step N. Audit it against CLAUDE.md: lookahead,
> holdout contact, contract violations, silent failure handling, and whether
> the printed verification actually proves what the step claims. List
> violations bluntly before anything else.
