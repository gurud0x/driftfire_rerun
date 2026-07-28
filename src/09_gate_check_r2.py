import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Phase R2, Step 3: gate check — DEV WINDOW ONLY, per R2 pre-registration
# Section 8 (identical bar to R1; committed d352510 before data contact).
#
# HOLDOUT ENFORCEMENT: post-2021 rows are dropped on the line after the
# merge; this script contains no holdout branch. The single holdout pass,
# if ever run, is a separate script permitted only after a logged dev PASS.
# ---------------------------------------------------------------------------

DEV_END = '2021-12-31'
GATE_TSTAT = 2.0
FACTORS = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM', 'ST_Rev']

project_root = Path(__file__).parent.parent
port_path = project_root / 'data' / 'processed' / 'portfolio_returns_daily_r2.parquet'
fac_path = project_root / 'data' / 'processed' / 'factors_daily.parquet'
log_path = project_root / 'results' / 'gate_log.md'

print("=" * 80)
print("R2 GATE CHECK — dev window only (2015-01-01 to 2021-12-31)")
print("=" * 80)

port = pd.read_parquet(port_path)
fac = pd.read_parquet(fac_path)
df = port.merge(fac, on='date', how='inner')
df = df[df['date'] <= DEV_END]      # <- holdout discarded before any statistic
print(f"\nDev-window rows: {len(df):,} "
      f"({df['date'].min().date()} to {df['date'].max().date()})")
print("Holdout rows discarded at load; no holdout statistic exists in this script.")


def run_reg(dfin, ret_col):
    d = dfin.dropna(subset=[ret_col] + FACTORS + ['RF'])
    y = d[ret_col] - d['RF']
    X = sm.add_constant(d[FACTORS])
    m = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 10})
    return m, len(d)


def report(m, n, label):
    a = m.params['const']
    t = m.tvalues['const']
    lines = [f"{label}: n={n} days",
             f"  intercept: {a*1e4:+.2f} bps/day  ({a*252*100:+.2f}%/yr)   "
             f"t-stat (NW, 10 lags): {t:+.2f}",
             f"  R-squared: {m.rsquared:.3f}",
             f"  {'factor':>7}  {'beta':>8}  {'t-stat':>7}"]
    for f in FACTORS:
        lines.append(f"  {f:>7}  {m.params[f]:+8.3f}  {m.tvalues[f]:+7.2f}")
    return "\n".join(lines), a, t


print("\n" + "-" * 80)
print("PRIMARY: port_ret_5d_net (excess of RF) ~ FF5 + MOM + ST_Rev")
print("-" * 80)
m5, n5 = run_reg(df, 'port_ret_5d_net')
txt5, alpha5, tstat5 = report(m5, n5, "R2 5-day primary, NET of 15bps/side")
print("\n" + txt5)

gate_pass = (tstat5 >= GATE_TSTAT) and (alpha5 > 0)
decision = "PASS" if gate_pass else "FAIL"

print("\n" + "=" * 80)
print(f"GATE: {decision} (dev)")
print(f"  criterion: intercept t-stat >= {GATE_TSTAT} AND intercept > 0")
print(f"  observed:  t-stat = {tstat5:+.2f}, intercept = {alpha5*1e4:+.2f} bps/day")
print("=" * 80)

# --------------------------------------------------------------------------
# Pre-registered exhibits (Section 9, descriptive, no gate authority):
# gross-alpha regression + 10d/20d horizon decay. Reported on PASS or FAIL.
# --------------------------------------------------------------------------
print("\nPre-registered exhibits (Section 9 — descriptive, no gate authority):")
exhibit_txts = []
for col, lab in [('port_ret_5d_gross', '5-day GROSS (pre-cost)'),
                 ('port_ret_10d_net', '10-day secondary NET'),
                 ('port_ret_20d_net', '20-day secondary NET')]:
    m, n = run_reg(df, col)
    txt, _, _ = report(m, n, lab)
    print("\n" + txt)
    exhibit_txts.append(txt)

# --------------------------------------------------------------------------
# Append to results/gate_log.md (never overwritten)
# --------------------------------------------------------------------------
log_path.parent.mkdir(parents=True, exist_ok=True)
stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
entry = [f"\n---\n\n## Gate evaluation — {stamp}",
         "- Phase: R2 (volume-conditioned long-only 5-day reversal)",
         "- Pre-registration: docs/PhaseR2_PreRegistration_VolumeConditionedReversal.md (commit d352510)",
         "- Script: src/09_gate_check_r2.py",
         f"- Window: DEV 2015-01-01 to {DEV_END} (holdout untouched)",
         "- Costs: 15 bps/side base; realized one-way turnover 50.3x/yr",
         "- Candidates: mean 24.4/day, 4 zero-candidate days (cash rule applied)",
         f"- Regression: daily net excess on {', '.join(FACTORS)}; NW maxlags=10",
         f"\n### Decision: **GATE {decision} (dev)**\n",
         "```\n" + txt5 + "\n```",
         "\n### Pre-registered exhibits (no gate authority)\n"]
for t in exhibit_txts:
    entry.append("```\n" + t + "\n```")
with open(log_path, 'a') as fh:
    fh.write("\n".join(entry) + "\n")
print(f"\n[OK] Appended R2 gate result to {log_path}")

print("\n" + "=" * 80)
print("R2 GATE CHECK COMPLETE — stopping per Section 8.")
if gate_pass:
    print("Dev PASS logged. The single holdout pass may now be authored as a")
    print("separate script. Holdout is spent once; no retuning of any kind.")
else:
    print("Next step: Section 10 null write-up. Backtest and holdout remain locked.")
print("=" * 80)
