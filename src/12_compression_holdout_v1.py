import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Phase V1 HOLDOUT evaluation — SINGLE PASS, per pre-registration
# (docs/PhaseV1_PreRegistration_VolatilityCompression.md, commit 4474312).
#
# This script loads the HOLDOUT window only (2022-01-01 to 2025-12-31).
# It contains no code path that touches dev-window data. It may only run
# because results/gate_log.md records a Phase V1 dev PASS; it verifies
# that entry exists and halts otherwise. It runs EXACTLY ONCE.
#
# Decision rule: sign consistency only — the mean holdout coefficient must
# be NEGATIVE (same sign as dev). No t-stat threshold applies to holdout.
# ---------------------------------------------------------------------------

HOLDOUT_START = '2022-01-01'
HOLDOUT_END = '2025-12-31'

project_root = Path(__file__).parent.parent
sig_path = project_root / 'data' / 'processed' / 'compression_signal_v1.parquet'
log_path = project_root / 'results' / 'gate_log.md'

print("=" * 80)
print("V1 HOLDOUT EVALUATION - single pre-committed pass, "
      "2022-01-01 to 2025-12-31")
print("=" * 80)

# --------------------------------------------------------------------------
# Precondition: a logged Phase V1 dev PASS must exist
# --------------------------------------------------------------------------
if not log_path.exists():
    print("HALT: results/gate_log.md does not exist. No dev PASS is logged;")
    print("the holdout pass is not authorized. Nothing was evaluated.")
    raise SystemExit(1)
log_text = log_path.read_text()
v1_pos = log_text.find("Phase: V1")
dev_pass_ok = v1_pos >= 0 and "GATE PASS (dev)" in log_text[v1_pos:]
print(f"\nPrecondition - logged Phase V1 dev PASS found: {dev_pass_ok} "
      f"{'[PASS]' if dev_pass_ok else '[HALT]'}")
if not dev_pass_ok:
    print("HALT: no Phase V1 dev PASS in results/gate_log.md. The holdout")
    print("pass is not authorized. Nothing was evaluated.")
    raise SystemExit(1)

# --------------------------------------------------------------------------
# Load holdout window ONLY
# --------------------------------------------------------------------------
panel = pd.read_parquet(sig_path)
panel = panel[(panel['DlyCalDt'] >= HOLDOUT_START) &
              (panel['DlyCalDt'] <= HOLDOUT_END)]
print(f"\nHoldout rows: {len(panel):,}")
print(f"Date range after filter: {panel['DlyCalDt'].min().date()} to "
      f"{panel['DlyCalDt'].max().date()}")
print(f"Dev-window rows present (pre-2022): "
      f"{int((panel['DlyCalDt'] < HOLDOUT_START).sum())} "
      f"[must be 0]")

# --------------------------------------------------------------------------
# Identical Fama-MacBeth procedure to src/11_compression_gate_v1.py
# --------------------------------------------------------------------------
d = panel.dropna(subset=['realized_vol_fwd_10d', 'compression_decile'])
g = d.groupby('DlyCalDt')
x, y = d['compression_decile'], d['realized_vol_fwd_10d']
agg = pd.DataFrame({
    'mx': g['compression_decile'].mean(),
    'my': g['realized_vol_fwd_10d'].mean(),
    'mxy': (x * y).groupby(d['DlyCalDt']).mean(),
    'mxx': (x * x).groupby(d['DlyCalDt']).mean(),
    'n': g.size(),
})
varx = agg['mxx'] - agg['mx'] ** 2
agg = agg[(agg['n'] >= 30) & (varx > 0)]
slopes = (agg['mxy'] - agg['mx'] * agg['my']) / (agg['mxx'] - agg['mx'] ** 2)
m = sm.OLS(slopes.values, np.ones(len(slopes))).fit(
    cov_type='HAC', cov_kwds={'maxlags': 10})
mean_slope, t_nw = m.params[0], m.tvalues[0]

print("\n" + "-" * 80)
print("HOLDOUT: realized_vol_fwd_10d ~ compression_decile, daily "
      "cross-sections")
print("-" * 80)
print(f"\nDaily cross-sectional regressions run: {len(slopes):,}")
print(f"Mean daily slope: {mean_slope:+.6f}")
print(f"Newey-West t-stat (maxlags=10): {t_nw:+.3f}  [reported for the "
      f"record; NOT a holdout criterion]")

print("\n" + "-" * 80)
print("MONOTONICITY TABLE (descriptive, holdout window)")
print("-" * 80)
mono = (d.groupby('compression_decile')['realized_vol_fwd_10d']
        .agg(['mean', 'count']))
print(f"\n{'decile':>7} {'mean rv_fwd_10d':>16} {'count':>10}   "
      f"(1 = most compressed)")
mono_lines = []
for dec, r in mono.iterrows():
    line = f"{dec:>7.0f} {r['mean']:>16.4f} {int(r['count']):>10,}"
    print(line)
    mono_lines.append(line)

# --------------------------------------------------------------------------
# Holdout decision: sign consistency with dev (negative)
# --------------------------------------------------------------------------
holdout_pass = mean_slope < 0
decision = ("HOLDOUT: PASS (sign-consistent)" if holdout_pass
            else "HOLDOUT: FAIL (sign flip)")
print("\n" + "=" * 80)
print(decision)
print(f"  criterion: mean holdout coefficient negative (same sign as dev)")
print(f"  observed:  mean slope = {mean_slope:+.6f}")
print("=" * 80)

# --------------------------------------------------------------------------
# Append below the existing V1 dev entry; verify nothing was lost
# --------------------------------------------------------------------------
stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
entry = [f"\n---\n\n## Holdout evaluation — {stamp}",
         "- Phase: V1 (volatility compression) — HOLDOUT, single "
         "pre-committed pass",
         "- Pre-registration: docs/PhaseV1_PreRegistration_"
         "VolatilityCompression.md (commit 4474312)",
         "- Script: src/12_compression_holdout_v1.py",
         f"- Window: HOLDOUT {HOLDOUT_START} to {HOLDOUT_END}",
         "- Criterion: sign consistency with dev only (dev mean slope was "
         "-0.004265, NW t -6.883)",
         f"\n### Decision: **{decision}**\n",
         "```",
         f"daily regressions: {len(slopes):,}",
         f"mean daily slope: {mean_slope:+.6f}   "
         f"NW t (maxlags=10, for the record): {t_nw:+.3f}",
         "",
         "monotonicity table (descriptive, holdout window):"]
entry += mono_lines
entry.append("```")
entry.append("\nTHIS HOLDOUT IS NOW SPENT. The pre-registration prohibits "
             "re-running,\nre-tuning, or re-splitting it under any "
             "circumstance, regardless of outcome.")
with open(log_path, 'a') as fh:
    fh.write("\n".join(entry) + "\n")
print(f"\n[OK] Appended holdout result to {log_path}")

log_text = log_path.read_text()
checks = [("R1 dev entry", "Phase: R1" in log_text),
          ("R2 dev entry", "Phase: R2" in log_text),
          ("V1 dev PASS entry", "GATE PASS (dev)" in log_text),
          ("V1 holdout entry", "HOLDOUT" in log_text)]
for name, ok in checks:
    print(f"  {name} present: {ok} {'[PASS]' if ok else '[FAIL]'}")

print("\n" + "=" * 80)
print("V1 HOLDOUT EVALUATION COMPLETE.")
print("THIS EVALUATION IS SPENT: the pre-registration prohibits re-running")
print("it under any circumstance, regardless of outcome. Any further work")
print("on this mechanism proceeds from the logged result only.")
print("=" * 80)
