import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Phase V2, Step 2: Fama-MacBeth gate check — DEV WINDOW ONLY, per
# docs/PhaseV2_PreRegistration_SectorRelativeCompression.md (commit 3fdac95).
#
# HOLDOUT ENFORCEMENT: post-2021 rows dropped on the line after load;
# no holdout code path exists in this script.
#
# Gate: mean daily slope of (realized_vol_fwd_10d ~ sector_rel_decile)
# NEGATIVE and |NW t| >= 3.0 (maxlags 10). Required evidence: monotonicity
# table AND side-by-side comparison with V1's logged dev t-stat (-6.883).
# ---------------------------------------------------------------------------

DEV_END = '2021-12-31'
GATE_ABS_T = 3.0
V1_DEV_MEAN = -0.004265        # logged in results/gate_log.md
V1_DEV_T = -6.883

project_root = Path(__file__).parent.parent
sig_path = project_root / 'data' / 'processed' / 'sector_compression_signal_v2.parquet'
log_path = project_root / 'results' / 'gate_log.md'

print("=" * 78)
print("V2 GATE CHECK - Fama-MacBeth, dev window only "
      "(2015-01-01 to 2021-12-31)")
print("=" * 78)

panel = pd.read_parquet(sig_path)
panel = panel[panel['DlyCalDt'] <= DEV_END]   # <- holdout discarded at load
print(f"\nDev-window rows: {len(panel):,} "
      f"({panel['DlyCalDt'].min().date()} to "
      f"{panel['DlyCalDt'].max().date()})")
print("Holdout rows discarded at load; no holdout code path in this script.")


def fama_macbeth(dfin, target, xcol):
    d = dfin.dropna(subset=[target, xcol])
    g = d.groupby('DlyCalDt')
    x, y = d[xcol], d[target]
    agg = pd.DataFrame({
        'mx': g[xcol].mean(),
        'my': g[target].mean(),
        'mxy': (x * y).groupby(d['DlyCalDt']).mean(),
        'mxx': (x * x).groupby(d['DlyCalDt']).mean(),
        'n': g.size(),
    })
    varx = agg['mxx'] - agg['mx'] ** 2
    agg = agg[(agg['n'] >= 30) & (varx > 0)]
    slopes = (agg['mxy'] - agg['mx'] * agg['my']) / (agg['mxx'] - agg['mx'] ** 2)
    m = sm.OLS(slopes.values, np.ones(len(slopes))).fit(
        cov_type='HAC', cov_kwds={'maxlags': 10})
    return slopes, m.params[0], m.tvalues[0]


print("\n" + "-" * 78)
print("PRIMARY: realized_vol_fwd_10d ~ sector_rel_decile, daily "
      "cross-sections")
print("-" * 78)
slopes, mean_s, t_s = fama_macbeth(panel, 'realized_vol_fwd_10d',
                                   'sector_rel_decile')
print(f"\nDaily cross-sectional regressions run: {len(slopes):,}")
print(f"Mean daily slope: {mean_s:+.6f}")
print(f"Negative-slope days: {(slopes < 0).mean()*100:.1f}%")
print(f"Newey-West t-stat (maxlags=10): {t_s:+.3f}")

print("\n" + "-" * 78)
print("REQUIRED COMPARISON vs V1 (logged dev result)")
print("-" * 78)
print(f"\n  {'':>22} {'mean slope':>12} {'NW t':>8}")
print(f"  {'V1 (own-history)':>22} {V1_DEV_MEAN:>+12.6f} {V1_DEV_T:>+8.3f}")
print(f"  {'V2 (sector-relative)':>22} {mean_s:>+12.6f} {t_s:>+8.3f}")

print("\n" + "-" * 78)
print("MONOTONICITY TABLE (required evidence): mean realized_vol_fwd_10d "
      "by decile")
print("-" * 78)
d = panel.dropna(subset=['realized_vol_fwd_10d', 'sector_rel_decile'])
mono = d.groupby('sector_rel_decile')['realized_vol_fwd_10d'].agg(
    ['mean', 'count'])
print(f"\n{'decile':>7} {'mean rv_fwd_10d':>16} {'count':>10}   "
      f"(1 = most sector-compressed)")
mono_lines = []
for dec, r in mono.iterrows():
    line = f"{dec:>7.0f} {r['mean']:>16.4f} {int(r['count']):>10,}"
    print(line)
    mono_lines.append(line)
strictly = bool(mono['mean'].is_monotonic_decreasing)
print(f"\nStrictly monotonic decreasing: {strictly}")

# --------------------------------------------------------------------------
# Gate decision
# --------------------------------------------------------------------------
gate_pass = (mean_s < 0) and (abs(t_s) >= GATE_ABS_T)
decision = "PASS" if gate_pass else "FAIL"
print("\n" + "=" * 78)
print(f"GATE: {decision} (dev)")
print(f"  criterion: mean daily slope < 0 AND |NW t| >= {GATE_ABS_T}")
print(f"  observed:  mean slope = {mean_s:+.6f}, NW t = {t_s:+.3f}")
print("=" * 78)

# --------------------------------------------------------------------------
# Secondary targets on FAIL (diagnostic record, per outcome commitments)
# --------------------------------------------------------------------------
sec_lines = []
if not gate_pass:
    print("\nSecondary targets (diagnostic, no gate authority):")
    for tgt in ['realized_vol_fwd_5d', 'realized_vol_fwd_20d']:
        s2, m2, t2 = fama_macbeth(panel, tgt, 'sector_rel_decile')
        line = f"{tgt}: mean slope={m2:+.6f}, NW t={t2:+.3f}, days={len(s2):,}"
        print("  " + line)
        sec_lines.append(line)

# --------------------------------------------------------------------------
# Append to gate log; confirm all prior entries intact
# --------------------------------------------------------------------------
log_path.parent.mkdir(parents=True, exist_ok=True)
stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
entry = [f"\n---\n\n## Gate evaluation — {stamp}",
         "- Phase: V2 (sector-relative volatility compression; "
         "forecasting/IC test, no alpha claim)",
         "- Pre-registration: docs/PhaseV2_PreRegistration_"
         "SectorRelativeCompression.md (commit 3fdac95)",
         "- Script: src/18_sector_compression_gate_v2.py",
         f"- Window: DEV 2015-01-01 to {DEV_END} (holdout untouched)",
         "- Static-GICS limitation applies as pre-registered (backfilled "
         "current classification)",
         "- Method: daily cross-sectional OLS slopes (Fama-MacBeth), "
         "NW t-test maxlags=10",
         f"\n### Decision: **GATE {decision} (dev)**\n",
         "```",
         f"primary: realized_vol_fwd_10d ~ sector_rel_decile",
         f"daily regressions: {len(slopes):,}",
         f"mean daily slope: {mean_s:+.6f}   NW t (maxlags=10): {t_s:+.3f}",
         f"negative-slope days: {(slopes < 0).mean()*100:.1f}%",
         "",
         f"required V1 comparison: V1 mean {V1_DEV_MEAN:+.6f} "
         f"(t {V1_DEV_T:+.3f}) vs V2 mean {mean_s:+.6f} (t {t_s:+.3f})",
         "",
         "monotonicity table (1 = most sector-compressed):"]
entry += mono_lines
entry.append(f"strictly monotonic decreasing: {strictly}")
entry.append("```")
if sec_lines:
    entry.append("\n### Secondary targets (diagnostic)\n")
    entry.append("```")
    entry += sec_lines
    entry.append("```")
with open(log_path, 'a') as fh:
    fh.write("\n".join(entry) + "\n")
print(f"\n[OK] Appended V2 gate result to {log_path}")

log_text = log_path.read_text()
checks = [("R1", "Phase: R1" in log_text), ("R2", "Phase: R2" in log_text),
          ("V1 dev", "Phase: V1 (volatility compression as" in log_text),
          ("V1 holdout", "V1 (volatility compression) — HOLDOUT" in log_text
           or "HOLDOUT: PASS" in log_text),
          ("V2 (this)", "Phase: V2" in log_text)]
for name, ok in checks:
    print(f"  {name} entry present: {ok} {'[PASS]' if ok else '[FAIL]'}")

print("\n" + "=" * 78)
print("V2 GATE CHECK COMPLETE - stopping per pre-registration.")
if gate_pass:
    print("Dev PASS logged. The single V2 holdout pass may now be authored")
    print("as a separate script, at the user's explicit direction only.")
else:
    print("Next step: null write-up (decile table, coefficient summary,")
    print("V1 comparison, static-GICS limitation restated). Holdout locked.")
print("=" * 78)
