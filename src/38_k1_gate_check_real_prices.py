import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# K1 rebuild, Step 3: gate check on REAL contract prices - DEV WINDOW ONLY.
# Same structure as src/31 and src/33; no holdout code path exists here.
#
# PRIMARY  = full dev window, MID-based returns, NW t maxlags=10,
#            PASS = mean positive AND t >= 2.0.
# DIAGNOSTICS (no gate authority): ex-2020, 2020-only, BID-based
#            (worst realistic fill), and liquidity-filtered.
#
# Prior gate_log entries - including both flawed K1 reconstructions - are
# preserved; this appends a new "K1 REAL PRICES (opprcd)" entry.
# ---------------------------------------------------------------------------

DEV_END = '2021-12-31'
GATE_TSTAT = 2.0

project_root = Path(__file__).parent.parent
port_path = (project_root / 'data' / 'processed' /
             'k1_portfolio_returns_real_prices.parquet')
fac_path = project_root / 'data' / 'processed' / 'factors_daily.parquet'
log_path = project_root / 'results' / 'gate_log.md'

print("=" * 78)
print("K1 GATE CHECK ON REAL CONTRACT PRICES (opprcd) - dev window only")
print("=" * 78)

port = pd.read_parquet(port_path)
fac = pd.read_parquet(fac_path, columns=['date', 'RF'])
df = port.merge(fac, on='date', how='left')
df = df[df['date'] <= DEV_END]     # <- holdout discarded before any statistic
print(f"\nDev-window rows: {len(df):,} "
      f"({df['date'].min().date()} to {df['date'].max().date()})")
print("Holdout rows discarded at load; no holdout code path in this script.")
print("\nPricing basis: real quoted bid/ask of the actual held contracts")
print("(identity-verified by optionid) on both entry and exit dates. No")
print("theta model, no surface interpolation, no assumed spread.")

df['year'] = df['date'].dt.year


def nw(d, col, label):
    y = (d[col] - d['RF']).dropna()
    if len(y) < 30:
        return np.nan, np.nan, f"{label}: insufficient days ({len(y)})"
    m = sm.OLS(y.values, np.ones(len(y))).fit(cov_type='HAC',
                                              cov_kwds={'maxlags': 10})
    mean_d, t = m.params[0], m.tvalues[0]
    txt = "\n".join([
        f"{label}:",
        f"  days: {len(y):,}",
        f"  mean daily net excess return: {mean_d*1e4:+.2f} bps "
        f"({mean_d*252*100:+.2f}%/yr)",
        f"  Newey-West t-stat (maxlags=10): {t:+.3f}"])
    return mean_d, t, txt


print("\n" + "-" * 78)
print("PRIMARY GATE: full dev window, MID-based")
print("-" * 78)
mean_p, t_p, txt_p = nw(df, 'port_ret_mid',
                        "Full dev 2015-2021, MID (PRIMARY)")
print("\n" + txt_p)

gate_pass = bool((mean_p > 0) and (t_p >= GATE_TSTAT))
decision = "PASS" if gate_pass else "FAIL"
print("\n" + "=" * 78)
print(f"GATE: {decision} (dev) - K1 REAL PRICES")
print(f"  criterion: mean daily net return > 0 AND NW t >= {GATE_TSTAT}")
print(f"  observed:  mean = {mean_p*1e4:+.2f} bps/day, t = {t_p:+.3f}")
print("=" * 78)

print("\nDIAGNOSTIC VIEWS (Section 8 + cost sensitivity; NO gate authority):")
diags = []
_, _, d_a = nw(df[df['year'] != 2020], 'port_ret_mid',
               "Dev EXCLUDING 2020, MID (diagnostic)")
_, _, d_b = nw(df[df['year'] == 2020], 'port_ret_mid',
               "2020 ALONE, MID (diagnostic)")
_, _, d_c = nw(df, 'port_ret_bid',
               "Full dev, BID-based worst fill (diagnostic)")
_, _, d_d = nw(df.dropna(subset=['port_ret_mid_liq']), 'port_ret_mid_liq',
               "Full dev, MID, liquidity-filtered (diagnostic)")
for txt in (d_a, d_b, d_c, d_d):
    print("\n" + txt)
    diags.append(txt)

print("\n" + "-" * 78)
print("DESCRIPTIVE (no gate authority)")
print("-" * 78)
for label, col, d in [("full dev MID", 'port_ret_mid', df),
                      ("dev excl 2020 MID", 'port_ret_mid',
                       df[df['year'] != 2020]),
                      ("2020 alone MID", 'port_ret_mid',
                       df[df['year'] == 2020]),
                      ("full dev BID", 'port_ret_bid', df),
                      ("full dev MID liq", 'port_ret_mid_liq', df)]:
    s = d[col].dropna()
    if len(s):
        print(f"  {label:>20}: {len(s):>5,} days, mean "
              f"{s.mean()*1e4:+8.2f} bps/day, positive days "
              f"{(s > 0).mean()*100:5.1f}%")

# --------------------------------------------------------------------------
# Append; verify every prior entry intact
# --------------------------------------------------------------------------
stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
entry = [f"\n---\n\n## Gate evaluation — {stamp}",
         "- Phase: K1 REAL PRICES (opprcd)",
         "- Pre-registration: docs/PhaseK1_PreRegistration_"
         "StraddleOnCompression.md (commit 3875314) - unchanged. This "
         "replaces the surface-based RECONSTRUCTION of option prices with "
         "the actual quoted bid/ask of the held contracts.",
         "- Scripts: src/36_validate_opprcd.py, "
         "src/37_k1_signal_construction_real_prices.py, "
         "src/38_k1_gate_check_real_prices.py",
         "- Supersedes, but does not delete, the two prior K1 entries: the "
         "original (+185%/yr, exit priced as a FRESH 30-day straddle, no "
         "theta) and the corrected (-490%/yr, sqrt-time IV haircut that "
         "double-counted theta). Both were model reconstructions of a price "
         "never observed; this entry uses observed quotes.",
         f"- Window: DEV 2015-01-01 to {DEV_END} (holdout untouched)",
         f"\n### Decision: **GATE {decision} (dev)** — K1 REAL PRICES, "
         f"primary = full dev, MID-based\n",
         "```", txt_p, "```",
         "\n### Diagnostic views (no gate authority)\n",
         "```"] + diags + ["```"]
log_path.parent.mkdir(parents=True, exist_ok=True)
with open(log_path, 'a') as fh:
    fh.write("\n".join(entry) + "\n")
print(f"\n[OK] Appended 'K1 REAL PRICES (opprcd)' entry to {log_path}")

log_text = log_path.read_text()
checks = [("R1", "Phase: R1"), ("R2", "Phase: R2"),
          ("V1 dev", "Phase: V1 (volatility"),
          ("V1 holdout", "V1 (volatility compression) — HOLDOUT"),
          ("V2 dev", "Phase: V2 (sector-relative volatility"),
          ("V2 holdout", "V2 (sector-relative compression) — HOLDOUT"),
          ("K0", "Phase: K0"),
          ("K1 original (flawed, retained)", "Phase: K1 (ATM straddle"),
          ("K1 corrected (retained)", "Phase: K1 CORRECTED"),
          ("K1 real prices (this)", "Phase: K1 REAL PRICES")]
for name, key in checks:
    ok = key in log_text
    print(f"  {name} entry present: {ok} {'[PASS]' if ok else '[FAIL]'}")

print("\n" + "=" * 78)
print("K1 REAL-PRICE GATE CHECK COMPLETE.")
if gate_pass:
    print("Dev PASS logged. The single K1 holdout pass may now be authored")
    print("as a separate script, at the user's explicit direction only.")
else:
    print("Next step: Section 9 write-up. Holdout remains locked and unspent.")
print("=" * 78)
