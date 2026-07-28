import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Step 6: Gate check — DEV WINDOW ONLY, per pre-registration Section 8.
#
# HOLDOUT ENFORCEMENT: rows after DEV_END are dropped on the line
# immediately following the merge, before any statistic is computed.
# This script contains NO holdout branch and never reads holdout rows.
# The single holdout promotion pass, if ever run, is a separate script
# that may only be written after a dev PASS is logged in
# results/gate_log.md (Section 8: holdout is spent once).
# ---------------------------------------------------------------------------

DEV_END = '2021-12-31'          # Section 8: dev = 2015-01-01 .. 2021-12-31
GATE_TSTAT = 2.0                # Section 8: PASS needs t >= 2.0 AND alpha > 0
FACTORS = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM', 'ST_Rev']

project_root = Path(__file__).parent.parent
port_path = project_root / 'data' / 'processed' / 'portfolio_returns_daily.parquet'
fac_path = project_root / 'data' / 'processed' / 'factors_daily.parquet'
log_path = project_root / 'results' / 'gate_log.md'

print("=" * 80)
print("GATE CHECK — Phase R1, dev window only (2015-01-01 to 2021-12-31)")
print("=" * 80)

port = pd.read_parquet(port_path)
fac = pd.read_parquet(fac_path)
df = port.merge(fac, on='date', how='inner')
df = df[df['date'] <= DEV_END]      # <- holdout discarded here, before anything else
print(f"\nDev-window rows after merge: {len(df):,} "
      f"({df['date'].min().date()} to {df['date'].max().date()})")
print("Holdout rows (post-2021) discarded at load; no holdout statistic is")
print("computed anywhere in this script.")


def run_reg(dfin, ret_col):
    d = dfin.dropna(subset=[ret_col] + FACTORS + ['RF'])
    y = d[ret_col] - d['RF']
    X = sm.add_constant(d[FACTORS])
    m = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 10})
    return m, len(d)


def report(m, n, label):
    a = m.params['const']
    t = m.tvalues['const']
    lines = []
    lines.append(f"{label}: n={n} days")
    lines.append(f"  intercept: {a*1e4:+.2f} bps/day  "
                 f"({a*252*100:+.2f}%/yr annualized)   t-stat (NW, 10 lags): {t:+.2f}")
    lines.append(f"  R-squared: {m.rsquared:.3f}")
    lines.append(f"  {'factor':>7}  {'beta':>8}  {'t-stat':>7}")
    for f in FACTORS:
        lines.append(f"  {f:>7}  {m.params[f]:+8.3f}  {m.tvalues[f]:+7.2f}")
    return "\n".join(lines), a, t


# --------------------------------------------------------------------------
# Primary regression: 5-day net, excess of RF, on FF5 + MOM + ST_Rev
# --------------------------------------------------------------------------
print("\n" + "-" * 80)
print("PRIMARY: port_ret_5d_net (excess of RF) ~ FF5 + MOM + ST_Rev")
print("-" * 80)

m5, n5 = run_reg(df, 'port_ret_5d_net')
txt5, alpha5, tstat5 = report(m5, n5, "5-day primary, NET of 15bps/side")
print("\n" + txt5)

gate_pass = (tstat5 >= GATE_TSTAT) and (alpha5 > 0)
decision = "PASS" if gate_pass else "FAIL"

print("\n" + "=" * 80)
print(f"GATE: {decision} (dev)")
print(f"  criterion: intercept t-stat >= {GATE_TSTAT} AND intercept > 0")
print(f"  observed:  t-stat = {tstat5:+.2f}, intercept = {alpha5*1e4:+.2f} bps/day")
print("=" * 80)

# --------------------------------------------------------------------------
# On FAIL, Section 10 requires the horizon-decay exhibit for the write-up.
# (Secondary horizons carry no gate authority either way.)
# --------------------------------------------------------------------------
decay_txts = []
if not gate_pass:
    print("\nHorizon-decay exhibit (Section 10 write-up requirement on FAIL):")
    for col, lab in [('port_ret_1d_net', '1-day'), ('port_ret_3d_net', '3-day'),
                     ('port_ret_10d_net', '10-day')]:
        m, n = run_reg(df, col)
        txt, _, _ = report(m, n, f"{lab} secondary, NET")
        print("\n" + txt)
        decay_txts.append(txt)

# --------------------------------------------------------------------------
# Append to results/gate_log.md (never overwritten)
# --------------------------------------------------------------------------
log_path.parent.mkdir(parents=True, exist_ok=True)
stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
entry = []
entry.append(f"\n---\n\n## Gate evaluation — {stamp}")
entry.append(f"- Phase: R1 (long-only small-cap 5-day reversal)")
entry.append(f"- Script: src/06_gate_check.py")
entry.append(f"- Window: DEV 2015-01-01 to {DEV_END} (holdout untouched)")
entry.append(f"- Costs: 15 bps/side base case; annualized one-way turnover 50.4x")
entry.append(f"- Regression: daily net excess returns on {', '.join(FACTORS)}; "
             f"Newey-West maxlags=10")
entry.append(f"\n### Decision: **GATE {decision} (dev)**\n")
entry.append("```\n" + txt5 + "\n```")
if decay_txts:
    entry.append("\n### Horizon-decay exhibit (no gate authority)\n")
    for t in decay_txts:
        entry.append("```\n" + t + "\n```")
with open(log_path, 'a') as fh:
    fh.write("\n".join(entry) + "\n")
print(f"\n[OK] Appended gate result to {log_path}")

print("\n" + "=" * 80)
print("GATE CHECK COMPLETE — stopping here per pre-registration Section 8.")
if not gate_pass:
    print("Next step: Section 10 null-result write-up (decay exhibit, cost")
    print("sensitivity, factor loadings, mechanism discussion). The backtest")
    print("module and holdout remain locked.")
print("=" * 80)
