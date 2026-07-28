import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Phase K1 CORRECTED gate check (theta-decay exit fix) - DEV WINDOW ONLY.
# Identical structure to src/31; reads the corrected returns from src/32.
# The original flawed entry in results/gate_log.md is NOT modified or
# deleted - it stays as the honest record. This appends a new entry
# labeled "K1 CORRECTED (theta-decay exit fix)".
# ---------------------------------------------------------------------------

DEV_END = '2021-12-31'
GATE_TSTAT = 2.0

project_root = Path(__file__).parent.parent
port_path = (project_root / 'data' / 'processed' /
             'k1_portfolio_returns_daily_corrected.parquet')
fac_path = project_root / 'data' / 'processed' / 'factors_daily.parquet'
log_path = project_root / 'results' / 'gate_log.md'

print("=" * 78)
print("K1 CORRECTED GATE CHECK (theta-decay exit fix) - dev window only")
print("=" * 78)

port = pd.read_parquet(port_path)
fac = pd.read_parquet(fac_path, columns=['date', 'RF'])
df = port.merge(fac, on='date', how='left')
df = df[df['date'] <= DEV_END]     # <- holdout discarded before any statistic
df['excess'] = df['port_ret_net'] - df['RF']
print(f"\nDev-window rows: {len(df):,} "
      f"({df['date'].min().date()} to {df['date'].max().date()})")
print("Holdout rows discarded at load; no holdout code path in this script.")
print("\nDisclosures in force: no spread cost (no bid/ask in pulled data);")
print("exit = held option repriced at original strike, exit spot/rate,")
print("actual remaining tenor (mean 15.4 cdays), sigma from 10d/30d")
print("interpolation (5.9% of trades) or sqrt-time proxy (94.1%).")


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


print("\n" + "-" * 78)
print("PRIMARY GATE: full dev window")
print("-" * 78)
mean_p, t_p, txt_p = nw_test(df, "Full dev 2015-2021 (PRIMARY, corrected)")
print("\n" + txt_p)

gate_pass = (mean_p > 0) and (t_p >= GATE_TSTAT)
decision = "PASS" if gate_pass else "FAIL"
print("\n" + "=" * 78)
print(f"GATE: {decision} (dev) - K1 CORRECTED")
print(f"  criterion: mean daily net return > 0 AND NW t >= {GATE_TSTAT}")
print(f"  observed:  mean = {mean_p*1e4:+.2f} bps/day, t = {t_p:+.3f}")
print("=" * 78)

print("\nDIAGNOSTIC VIEWS (Section 8; NO gate authority):")
df['year'] = df['date'].dt.year
d_ex20 = df[df['year'] != 2020]
d_20 = df[df['year'] == 2020]
mean_a, t_a, txt_a = nw_test(d_ex20, "Dev EXCLUDING 2020 (diagnostic)")
mean_b, t_b, txt_b = nw_test(d_20, "2020 ALONE (diagnostic)")
print("\n" + txt_a)
print("\n" + txt_b)

print("\n" + "-" * 78)
print("DESCRIPTIVE (no gate authority)")
print("-" * 78)
for label, d in [("full dev", df), ("dev excl 2020", d_ex20),
                 ("2020 alone", d_20)]:
    s = d['port_ret_net'].dropna()
    pos = (s > 0).mean() * 100
    print(f"  {label:>14}: {len(s):>5,} days, mean {s.mean()*1e4:+7.2f} "
          f"bps/day, positive days {pos:5.1f}%")

# --------------------------------------------------------------------------
# Append (never overwrite); verify every prior entry incl. the flawed K1
# --------------------------------------------------------------------------
stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
entry = [f"\n---\n\n## Gate evaluation — {stamp}",
         "- Phase: K1 CORRECTED (theta-decay exit fix)",
         "- Pre-registration: docs/PhaseK1_PreRegistration_"
         "StraddleOnCompression.md (commit 3875314) - unchanged; this "
         "corrects the IMPLEMENTATION to match Section 5 as written "
         "(exit the held option, not a fresh 30-day one)",
         "- Scripts: src/32_k1_signal_construction_corrected.py, "
         "src/33_k1_gate_check_corrected.py",
         "- The original flawed K1 entry above stays as the honest record; "
         "its +185%/yr was an exit-valuation artifact (omitted theta), "
         "confirmed by the before/after: +9.96% -> -11.74% mean per-trade",
         f"- Window: DEV 2015-01-01 to {DEV_END} (holdout untouched)",
         "- Trades: 5,277 (913 PERMNOs); no spread cost (none in data): "
         "results remain an UPPER bound",
         f"\n### Decision: **GATE {decision} (dev)** — K1 CORRECTED, "
         f"primary, full dev window\n",
         "```", txt_p, "```",
         "\n### Diagnostic views (Section 8, no gate authority)\n",
         "```", txt_a, "", txt_b, "```"]
log_path.parent.mkdir(parents=True, exist_ok=True)
with open(log_path, 'a') as fh:
    fh.write("\n".join(entry) + "\n")
print(f"\n[OK] Appended K1 CORRECTED entry to {log_path}")

log_text = log_path.read_text()
checks = [("R1", "Phase: R1"), ("R2", "Phase: R2"),
          ("V1 dev", "Phase: V1 (volatility"),
          ("V1 holdout", "V1 (volatility compression) — HOLDOUT"),
          ("V2 dev", "Phase: V2 (sector-relative volatility"),
          ("V2 holdout", "V2 (sector-relative compression) — HOLDOUT"),
          ("K0", "Phase: K0"),
          ("K1 original (flawed, retained)", "Phase: K1 (ATM straddle"),
          ("K1 corrected (this)", "Phase: K1 CORRECTED")]
for name, key in checks:
    ok = key in log_text
    print(f"  {name} entry present: {ok} {'[PASS]' if ok else '[FAIL]'}")

print("\n" + "=" * 78)
print("K1 CORRECTED GATE CHECK COMPLETE.")
if gate_pass:
    print("Dev PASS logged. The single K1 holdout pass may now be authored")
    print("as a separate script, at the user's explicit direction only.")
else:
    print("Next step: Section 9 write-up (trade count, win rate, IV-RV")
    print("premium role, cost stress). Holdout remains locked and unspent.")
print("=" * 78)
