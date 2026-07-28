import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Phase V2 HOLDOUT evaluation — SINGLE PASS, per
# docs/PhaseV2_PreRegistration_SectorRelativeCompression.md (commit 3fdac95).
#
# Loads the HOLDOUT window only (2022-01-01 to 2025-12-31); no dev-window
# code path exists in this script. Authorized only by the logged V2 dev
# PASS (NW t = -7.016), which is verified below before anything runs.
# Decision rule: SIGN CONSISTENCY only (negative mean coefficient), same
# rule as V1's holdout. This script runs EXACTLY ONCE.
# ---------------------------------------------------------------------------

HOLDOUT_START = '2022-01-01'
HOLDOUT_END = '2025-12-31'

project_root = Path(__file__).parent.parent
sig_path = project_root / 'data' / 'processed' / 'sector_compression_signal_v2.parquet'
log_path = project_root / 'results' / 'gate_log.md'

print("=" * 78)
print("V2 HOLDOUT EVALUATION - single pre-committed pass, "
      "2022-01-01 to 2025-12-31")
print("=" * 78)

# --------------------------------------------------------------------------
# Precondition: a logged Phase V2 dev PASS must exist
# --------------------------------------------------------------------------
if not log_path.exists():
    print("HALT: results/gate_log.md missing; holdout not authorized.")
    raise SystemExit(1)
log_text = log_path.read_text()
v2_pos = log_text.find("Phase: V2")
dev_pass_ok = v2_pos >= 0 and "GATE PASS (dev)" in log_text[v2_pos:]
print(f"\nPrecondition - logged Phase V2 dev PASS found: {dev_pass_ok} "
      f"{'[PASS]' if dev_pass_ok else '[HALT]'}")
if not dev_pass_ok:
    print("HALT: no Phase V2 dev PASS in results/gate_log.md. Nothing "
          "was evaluated.")
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
      f"{int((panel['DlyCalDt'] < HOLDOUT_START).sum())} [must be 0]")

# --------------------------------------------------------------------------
# Identical Fama-MacBeth procedure to src/18_sector_compression_gate_v2.py
# --------------------------------------------------------------------------
d = panel.dropna(subset=['realized_vol_fwd_10d', 'sector_rel_decile'])
g = d.groupby('DlyCalDt')
x, y = d['sector_rel_decile'], d['realized_vol_fwd_10d']
agg = pd.DataFrame({
    'mx': g['sector_rel_decile'].mean(),
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

print("\n" + "-" * 78)
print("HOLDOUT: realized_vol_fwd_10d ~ sector_rel_decile, daily "
      "cross-sections")
print("-" * 78)
print(f"\nDaily cross-sectional regressions run: {len(slopes):,}")
print(f"Mean daily slope: {mean_slope:+.6f}")
print(f"Newey-West t-stat (maxlags=10): {t_nw:+.3f}  [for the record; "
      f"NOT a holdout criterion]")

print("\n" + "-" * 78)
print("MONOTONICITY TABLE (descriptive, holdout window)")
print("-" * 78)
mono = d.groupby('sector_rel_decile')['realized_vol_fwd_10d'].agg(
    ['mean', 'count'])
print(f"\n{'decile':>7} {'mean rv_fwd_10d':>16} {'count':>10}   "
      f"(1 = most sector-compressed)")
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
print("\n" + "=" * 78)
print(decision)
print(f"  criterion: mean holdout coefficient negative (same sign as dev)")
print(f"  observed:  mean slope = {mean_slope:+.6f}")
print("=" * 78)

# --------------------------------------------------------------------------
# Append; verify every prior entry is intact
# --------------------------------------------------------------------------
stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
entry = [f"\n---\n\n## Holdout evaluation — {stamp}",
         "- Phase: V2 (sector-relative compression) — HOLDOUT, single "
         "pre-committed pass",
         "- Pre-registration: docs/PhaseV2_PreRegistration_"
         "SectorRelativeCompression.md (commit 3fdac95)",
         "- Script: src/19_sector_compression_holdout_v2.py",
         f"- Window: HOLDOUT {HOLDOUT_START} to {HOLDOUT_END}",
         "- Criterion: sign consistency with dev only (dev mean slope "
         "-0.003816, NW t -7.016)",
         "- Static-GICS limitation applies as pre-registered",
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
print(f"\n[OK] Appended V2 holdout result to {log_path}")

log_text = log_path.read_text()
checks = [
    ("R1 dev", "Phase: R1" in log_text),
    ("R2 dev", "Phase: R2" in log_text),
    ("V1 dev", "Phase: V1 (volatility compression as" in log_text),
    ("V1 holdout", "V1 (volatility compression) — HOLDOUT" in log_text),
    ("V2 dev", "Phase: V2 (sector-relative volatility" in log_text),
    ("V2 holdout (this)", "V2 (sector-relative compression) — HOLDOUT"
     in log_text),
]
for name, ok in checks:
    print(f"  {name} entry present: {ok} {'[PASS]' if ok else '[FAIL]'}")

print("\n" + "=" * 78)
print("V2 HOLDOUT EVALUATION COMPLETE.")
print("THIS EVALUATION IS SPENT: it cannot be re-run under any")
print("circumstance, regardless of outcome, per the pre-registration.")
print("=" * 78)
