import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Phase V1, Step 2: Fama-MacBeth gate check — DEV WINDOW ONLY, per
# docs/PhaseV1_PreRegistration_VolatilityCompression.md (commit 4474312).
#
# HOLDOUT ENFORCEMENT: post-2021 rows are dropped on the line after load;
# no holdout code path exists in this script.
#
# Gate: mean daily slope of (realized_vol_fwd_10d ~ compression_decile)
# is NEGATIVE and |Newey-West t| >= 3.0 (maxlags 10).
# ---------------------------------------------------------------------------

DEV_END = '2021-12-31'
GATE_ABS_T = 3.0

project_root = Path(__file__).parent.parent
sig_path = project_root / 'data' / 'processed' / 'compression_signal_v1.parquet'
log_path = project_root / 'results' / 'gate_log.md'

print("=" * 80)
print("V1 GATE CHECK - Fama-MacBeth, dev window only (2015-01-01 to 2021-12-31)")
print("=" * 80)

panel = pd.read_parquet(sig_path)
panel = panel[panel['DlyCalDt'] <= DEV_END]   # <- holdout discarded at load
print(f"\nDev-window rows: {len(panel):,} "
      f"({panel['DlyCalDt'].min().date()} to {panel['DlyCalDt'].max().date()})")
print("Holdout rows discarded at load; no holdout code path in this script.")


def fama_macbeth(dfin, target):
    d = dfin.dropna(subset=[target, 'compression_decile'])
    g = d.groupby('DlyCalDt')
    x, y = d['compression_decile'], d[target]
    agg = pd.DataFrame({
        'mx': g['compression_decile'].mean(),
        'my': g[target].mean(),
        'mxy': (x * y).groupby(d['DlyCalDt']).mean(),
        'mxx': (x * x).groupby(d['DlyCalDt']).mean(),
        'n': g.size(),
    })
    varx = agg['mxx'] - agg['mx'] ** 2
    agg = agg[(agg['n'] >= 30) & (varx > 0)]
    slopes = (agg['mxy'] - agg['mx'] * agg['my']) / (agg['mxx'] - agg['mx'] ** 2)
    # Newey-West t-test on the daily coefficient series (mean via OLS on const)
    m = sm.OLS(slopes.values, np.ones(len(slopes))).fit(
        cov_type='HAC', cov_kwds={'maxlags': 10})
    return slopes, m.params[0], m.tvalues[0]


print("\n" + "-" * 80)
print("PRIMARY: realized_vol_fwd_10d ~ compression_decile, daily cross-sections")
print("-" * 80)

slopes10, mean10, t10 = fama_macbeth(panel, 'realized_vol_fwd_10d')
print(f"\nDaily cross-sectional regressions run: {len(slopes10):,}")
print(f"Daily slope series: mean={mean10:+.6f} (annualized-vol units per "
      f"decile step)")
print(f"                    std={slopes10.std():.6f}, "
      f"negative days: {(slopes10 < 0).mean()*100:.1f}%")
print(f"Newey-West t-stat (maxlags=10): {t10:+.3f}")

print("\n" + "-" * 80)
print("MONOTONICITY TABLE (required evidence): mean realized_vol_fwd_10d "
      "by compression decile")
print("-" * 80)
mono = (panel.dropna(subset=['realized_vol_fwd_10d', 'compression_decile'])
        .groupby('compression_decile')['realized_vol_fwd_10d']
        .agg(['mean', 'count']))
print(f"\n{'decile':>7} {'mean rv_fwd_10d':>16} {'count':>10}   "
      f"(1 = most compressed)")
mono_lines = []
for dec, r in mono.iterrows():
    line = f"{dec:>7.0f} {r['mean']:>16.4f} {int(r['count']):>10,}"
    print(line)
    mono_lines.append(line)
strictly_decreasing = bool(mono['mean'].is_monotonic_decreasing)
print(f"\nStrictly monotonic decreasing (decile 1 highest -> 10 lowest): "
      f"{strictly_decreasing}")

# --------------------------------------------------------------------------
# Gate decision
# --------------------------------------------------------------------------
gate_pass = (mean10 < 0) and (abs(t10) >= GATE_ABS_T)
decision = "PASS" if gate_pass else "FAIL"
print("\n" + "=" * 80)
print(f"GATE: {decision} (dev)")
print(f"  criterion: mean daily slope < 0 AND |NW t| >= {GATE_ABS_T}")
print(f"  observed:  mean slope = {mean10:+.6f}, NW t = {t10:+.3f}")
print("=" * 80)

# --------------------------------------------------------------------------
# Secondary targets on FAIL (diagnostic record)
# --------------------------------------------------------------------------
sec_lines = []
if not gate_pass:
    print("\nSecondary targets (diagnostic, no gate authority):")
    for tgt in ['realized_vol_fwd_5d', 'realized_vol_fwd_20d']:
        s, mean_s, t_s = fama_macbeth(panel, tgt)
        line = (f"{tgt}: mean slope={mean_s:+.6f}, NW t={t_s:+.3f}, "
                f"days={len(s):,}")
        print("  " + line)
        sec_lines.append(line)

# --------------------------------------------------------------------------
# Append to gate log; confirm R1/R2 entries intact
# --------------------------------------------------------------------------
log_path.parent.mkdir(parents=True, exist_ok=True)
stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
entry = [f"\n---\n\n## Gate evaluation — {stamp}",
         "- Phase: V1 (volatility compression as expansion precursor; "
         "forecasting/IC test, no alpha claim)",
         "- Pre-registration: docs/PhaseV1_PreRegistration_"
         "VolatilityCompression.md (commit 4474312)",
         "- Script: src/11_compression_gate_v1.py",
         f"- Window: DEV 2015-01-01 to {DEV_END} (holdout untouched)",
         "- Method: daily cross-sectional OLS slopes (Fama-MacBeth), "
         "NW t-test maxlags=10 on the slope series",
         f"\n### Decision: **GATE {decision} (dev)**\n",
         "```",
         f"primary target: realized_vol_fwd_10d ~ compression_decile",
         f"daily regressions: {len(slopes10):,}",
         f"mean daily slope: {mean10:+.6f}   NW t (maxlags=10): {t10:+.3f}",
         f"negative-slope days: {(slopes10 < 0).mean()*100:.1f}%",
         "",
         "monotonicity table (mean rv_fwd_10d by compression decile, "
         "1 = most compressed):"]
entry += mono_lines
entry.append(f"strictly monotonic decreasing: {strictly_decreasing}")
entry.append("```")
if sec_lines:
    entry.append("\n### Secondary targets (diagnostic)\n")
    entry.append("```")
    entry += sec_lines
    entry.append("```")
with open(log_path, 'a') as fh:
    fh.write("\n".join(entry) + "\n")
print(f"\n[OK] Appended V1 gate result to {log_path}")

log_text = log_path.read_text()
r1_ok = "Phase: R1" in log_text
r2_ok = "Phase: R2" in log_text
print(f"Prior entries intact after append: R1 {'[PASS]' if r1_ok else '[FAIL]'}, "
      f"R2 {'[PASS]' if r2_ok else '[FAIL]'}")

print("\n" + "=" * 80)
print("V1 GATE CHECK COMPLETE - stopping per pre-registration.")
if gate_pass:
    print("Dev PASS logged. The single holdout pass may now be authored as a")
    print("separate script. After holdout: Kronos horse race vs trailing-vol")
    print("and GARCH(1,1) baselines (pre-registration, dormant section).")
else:
    print("Next step: null write-up (decile table, coefficient summary,")
    print("stated reason). Holdout remains locked.")
print("=" * 80)
