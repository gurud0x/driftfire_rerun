import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Phase K1, Part C: gate check - DEV WINDOW ONLY, per
# docs/PhaseK1_PreRegistration_StraddleOnCompression.md (commit 3875314).
#
# HOLDOUT ENFORCEMENT: post-2021 rows dropped on the line after load; no
# holdout code path exists in this script (same pattern as every phase).
#
# PRIMARY gate: full dev window, NW t-test maxlags=10 on daily net returns.
# PASS = mean positive AND t >= 2.0.
# DIAGNOSTICS (locked in Section 8, NO gate authority): dev excluding 2020,
# and 2020 alone.
# ---------------------------------------------------------------------------

DEV_END = '2021-12-31'
GATE_TSTAT = 2.0

project_root = Path(__file__).parent.parent
port_path = project_root / 'data' / 'processed' / 'k1_portfolio_returns_daily.parquet'
fac_path = project_root / 'data' / 'processed' / 'factors_daily.parquet'
log_path = project_root / 'results' / 'gate_log.md'

print("=" * 78)
print("K1 GATE CHECK - dev window only (2015-01-01 to 2021-12-31)")
print("=" * 78)

port = pd.read_parquet(port_path)
fac = pd.read_parquet(fac_path, columns=['date', 'RF'])
df = port.merge(fac, on='date', how='left')
df = df[df['date'] <= DEV_END]     # <- holdout discarded before any statistic
df['excess'] = df['port_ret_net'] - df['RF']
print(f"\nDev-window rows: {len(df):,} "
      f"({df['date'].min().date()} to {df['date'].max().date()})")
print("Holdout rows discarded at load; no holdout code path in this script.")
print("\nCost disclosures in force (Section 7 + construction notes):")
print("  - no bid/ask in pulled data -> no spread cost applied")
print("  - exits valued at the constant-maturity 30d surface premium (the")
print("    only price available), which overstates remaining time value at")
print("    exit -> both biases push returns UP; any PASS is an upper bound.")


def nw_test(d, label):
    y = d['excess'].dropna()
    m = sm.OLS(y.values, np.ones(len(y))).fit(cov_type='HAC',
                                              cov_kwds={'maxlags': 10})
    mean_d, t = m.params[0], m.tvalues[0]
    lines = [f"{label}:",
             f"  days: {len(y):,}",
             f"  mean daily net excess return: {mean_d*1e4:+.2f} bps "
             f"({mean_d*252*100:+.2f}%/yr)",
             f"  Newey-West t-stat (maxlags=10): {t:+.3f}"]
    return mean_d, t, "\n".join(lines)


# --------------------------------------------------------------------------
# PRIMARY: full dev window
# --------------------------------------------------------------------------
print("\n" + "-" * 78)
print("PRIMARY GATE: full dev window")
print("-" * 78)
mean_p, t_p, txt_p = nw_test(df, "Full dev 2015-2021 (PRIMARY)")
print("\n" + txt_p)

gate_pass = (mean_p > 0) and (t_p >= GATE_TSTAT)
decision = "PASS" if gate_pass else "FAIL"
print("\n" + "=" * 78)
print(f"GATE: {decision} (dev)")
print(f"  criterion: mean daily net return > 0 AND NW t >= {GATE_TSTAT}")
print(f"  observed:  mean = {mean_p*1e4:+.2f} bps/day, t = {t_p:+.3f}")
print("=" * 78)

# --------------------------------------------------------------------------
# DIAGNOSTICS (no gate authority, always computed per Section 8)
# --------------------------------------------------------------------------
print("\nDIAGNOSTIC VIEWS (locked in Section 8; NO gate authority - the")
print("primary decision above stands regardless of these):")
df['year'] = df['date'].dt.year
d_ex20 = df[df['year'] != 2020]
d_20 = df[df['year'] == 2020]
mean_a, t_a, txt_a = nw_test(d_ex20, "Dev EXCLUDING 2020 (diagnostic)")
mean_b, t_b, txt_b = nw_test(d_20, "2020 ALONE (diagnostic)")
print("\n" + txt_a)
print("\n" + txt_b)

# --------------------------------------------------------------------------
# Descriptive trade-level stats (first point in the project where P&L
# figures are computed, per the task and the now-committed Section 6/8)
# --------------------------------------------------------------------------
print("\n" + "-" * 78)
print("DESCRIPTIVE (no gate authority): per-trade stats")
print("-" * 78)
# reconstruct per-trade nets from the daily series is lossy; recompute the
# per-trade table from the construction script's logic would duplicate code.
# Instead: descriptive stats from the daily series (portfolio view), plus
# trade counts from the construction run, which src/30 printed and which
# feed the log entry below.
for label, d in [("full dev", df), ("dev excl 2020", d_ex20),
                 ("2020 alone", d_20)]:
    s = d['port_ret_net'].dropna()
    pos = (s > 0).mean() * 100
    print(f"  {label:>14}: {len(s):>5,} days, mean {s.mean()*1e4:+7.2f} "
          f"bps/day, positive days {pos:5.1f}%")

# --------------------------------------------------------------------------
# Append to gate log; verify all prior entries intact
# --------------------------------------------------------------------------
stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
entry = [f"\n---\n\n## Gate evaluation — {stamp}",
         "- Phase: K1 (ATM straddle on compression, first alpha claim, "
         "option space)",
         "- Pre-registration: docs/PhaseK1_PreRegistration_"
         "StraddleOnCompression.md (commit 3875314)",
         "- Scripts: src/30_k1_signal_construction.py, "
         "src/31_k1_gate_check.py",
         f"- Window: DEV 2015-01-01 to {DEV_END} (holdout untouched)",
         "- Trades: 5,351 (921 PERMNOs); 2020 share 44.2%, consistent with "
         "the pre-committed src/29 count check",
         "- Cost disclosures: NO spread cost (no bid/ask in pulled data); "
         "exits at constant-maturity 30d surface premium (overstates exit "
         "time value). Both biases are UPWARD: any PASS is an upper bound.",
         f"\n### Decision: **GATE {decision} (dev)** — primary, full dev "
         f"window\n",
         "```", txt_p, "```",
         "\n### Diagnostic views (locked in Section 8, no gate authority)\n",
         "```", txt_a, "", txt_b, "```"]
log_path.parent.mkdir(parents=True, exist_ok=True)
with open(log_path, 'a') as fh:
    fh.write("\n".join(entry) + "\n")
print(f"\n[OK] Appended K1 gate result to {log_path}")

log_text = log_path.read_text()
checks = [("R1", "Phase: R1"), ("R2", "Phase: R2"),
          ("V1 dev", "Phase: V1 (volatility"),
          ("V1 holdout", "V1 (volatility compression) — HOLDOUT"),
          ("V2 dev", "Phase: V2 (sector-relative volatility"),
          ("V2 holdout", "V2 (sector-relative compression) — HOLDOUT"),
          ("K0", "Phase: K0"), ("K1 (this)", "Phase: K1")]
for name, key in checks:
    ok = key in log_text
    print(f"  {name} entry present: {ok} {'[PASS]' if ok else '[FAIL]'}")

print("\n" + "=" * 78)
print("K1 GATE CHECK COMPLETE - stopping per pre-registration Section 8.")
if gate_pass:
    print("Dev PASS logged. The single K1 holdout pass may now be authored")
    print("as a separate script, at the user's explicit direction only.")
else:
    print("Next step: Section 9 write-up (trade count, win rate, IV-RV")
    print("premium role, cost stress). Holdout remains locked.")
print("=" * 78)
