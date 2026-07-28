import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from arch import arch_model
import warnings

# ---------------------------------------------------------------------------
# Phase K0: volatility forecaster horse race, per
# docs/PhaseK0_PreRegistration_ForecasterHorseRace.md (commit 798d54a).
# Dev window only. Winner becomes the sanctioned E[RV] forecaster and
# unlocks K1 per V1's locked sequencing.
# ---------------------------------------------------------------------------

DEV_START = '2015-01-01'
DEV_END = '2021-12-31'
MIN_GARCH_OBS = 250

project_root = Path(__file__).parent.parent
v1_path = project_root / 'data' / 'processed' / 'compression_signal_v1.parquet'
crsp_path = project_root / 'data' / 'processed' / 'crsp_combined.parquet'
log_path = project_root / 'results' / 'gate_log.md'

print("=" * 78)
print("K0 HORSE RACE: TRAIL20 vs GARCH11 vs COMPDEC, dev window "
      "2015-2021, MAE on realized_vol_fwd_10d")
print("=" * 78)

panel = pd.read_parquet(v1_path)
panel = panel[(panel['DlyCalDt'] >= DEV_START) & (panel['DlyCalDt'] <= DEV_END)]
print(f"\n[OK] V1 panel, dev window: {len(panel):,} stock-days "
      f"({panel['DlyCalDt'].min().date()} to {panel['DlyCalDt'].max().date()})")

px = pd.read_parquet(crsp_path, columns=['PERMNO', 'DlyCalDt', 'DlyRet'])
px = px[px['PERMNO'].isin(panel['PERMNO'].unique())]
px = px[(px['DlyCalDt'] >= '2014-01-01') & (px['DlyCalDt'] <= DEV_END)]
px = (px.drop_duplicates(['PERMNO', 'DlyCalDt'], keep='first')
        .sort_values(['PERMNO', 'DlyCalDt']).reset_index(drop=True))
print(f"[OK] Daily returns for candidate PERMNOs (2014 warmup + dev): "
      f"{len(px):,} rows, {px['PERMNO'].nunique():,} PERMNOs")

# --------------------------------------------------------------------------
# Candidate 1: TRAIL20 — trailing 20d annualized std ending at day t
# --------------------------------------------------------------------------
print("\nBuilding TRAIL20 (rolling 20d std ending at t, min_periods=15)...")
grp = px.groupby('PERMNO', sort=False)
px['trail20'] = grp['DlyRet'].transform(
    lambda s: s.rolling(20, min_periods=15).std()) * np.sqrt(252)

# --------------------------------------------------------------------------
# Candidate 2: GARCH(1,1) per stock, fit once on dev-window returns
# --------------------------------------------------------------------------
print("Building GARCH11 (per-stock constant-mean GARCH(1,1), fit once on")
print(f"dev-window returns; stocks with < {MIN_GARCH_OBS} dev obs skipped)...")
warnings.filterwarnings('ignore')

garch_frames = []
n_fit, n_skip, n_fail = 0, 0, 0
n_stocks = px['PERMNO'].nunique()
for i, (p, h) in enumerate(px.groupby('PERMNO', sort=False)):
    if i % 500 == 0:
        print(f"  ... {i}/{n_stocks} stocks "
              f"(fit {n_fit}, skipped {n_skip}, failed {n_fail})")
    dev = h[h['DlyCalDt'] >= DEV_START].dropna(subset=['DlyRet'])
    if len(dev) < MIN_GARCH_OBS:
        n_skip += 1
        continue
    r = dev['DlyRet'].values * 100.0          # arch convention: percent
    try:
        res = arch_model(r, mean='Constant', vol='GARCH', p=1, q=1,
                         rescale=False).fit(disp='off',
                                            options={'maxiter': 200})
        omega, alpha, beta = (res.params['omega'], res.params['alpha[1]'],
                              res.params['beta[1]'])
        ab = alpha + beta
        if not (0 < ab < 1):
            n_fail += 1
            continue
        sig2_u = omega / (1.0 - ab)
        h_t = res.conditional_volatility ** 2          # in-sample h_t
        eps2 = (r - res.params['mu']) ** 2
        h_next = omega + alpha * eps2 + beta * h_t     # h_{t+1} known at t
        # mean of h over t+1..t+10:  sig2_u + (h_{t+1}-sig2_u)*mean(ab^(k-1))
        decay = np.mean([ab ** (k - 1) for k in range(1, 11)])
        mean_h10 = sig2_u + (h_next - sig2_u) * decay
        # percent^2 daily variance -> annualized decimal vol
        fc = np.sqrt(mean_h10 * 252.0) / 100.0
        garch_frames.append(pd.DataFrame({
            'PERMNO': p, 'DlyCalDt': dev['DlyCalDt'].values, 'garch11': fc}))
        n_fit += 1
    except Exception:
        n_fail += 1

garch = pd.concat(garch_frames, ignore_index=True)
print(f"  GARCH fits: {n_fit:,} ok, {n_skip:,} skipped (< {MIN_GARCH_OBS} "
      f"obs), {n_fail:,} failed/nonstationary")

# --------------------------------------------------------------------------
# Candidate 3: COMPDEC — decile's dev-window mean realized_vol_fwd_10d
# --------------------------------------------------------------------------
print("\nBuilding COMPDEC (decile -> dev-window mean rv_fwd_10d)...")
dec_map = (panel.dropna(subset=['compression_decile', 'realized_vol_fwd_10d'])
           .groupby('compression_decile')['realized_vol_fwd_10d'].mean())
print("  Decile forecast values (annualized vol):")
for dec, v in dec_map.items():
    print(f"    decile {dec:>4.0f}: {v:.4f}")

# --------------------------------------------------------------------------
# Assemble the common evaluation sample
# --------------------------------------------------------------------------
ev = panel[['PERMNO', 'DlyCalDt', 'compression_decile',
            'realized_vol_fwd_10d']].copy()
ev = ev.merge(px[['PERMNO', 'DlyCalDt', 'trail20']],
              on=['PERMNO', 'DlyCalDt'], how='left')
ev = ev.merge(garch, on=['PERMNO', 'DlyCalDt'], how='left')
ev['compdec'] = ev['compression_decile'].map(dec_map)

common = ev.dropna(subset=['realized_vol_fwd_10d', 'trail20', 'garch11',
                           'compdec'])
print(f"\nCommon evaluation sample (target + all three forecasts non-null): "
      f"{len(common):,} stock-days of {len(ev):,} dev stock-days "
      f"({len(common)/len(ev)*100:.1f}%)")

# --------------------------------------------------------------------------
# MAE table and decision
# --------------------------------------------------------------------------
mae = {
    'TRAIL20 (trailing 20d vol)': (common['trail20'] -
                                   common['realized_vol_fwd_10d']).abs().mean(),
    'GARCH11 (per-stock GARCH(1,1))': (common['garch11'] -
                                       common['realized_vol_fwd_10d']).abs().mean(),
    'COMPDEC (compression decile mean)': (common['compdec'] -
                                          common['realized_vol_fwd_10d']).abs().mean(),
}
simplicity = {'TRAIL20 (trailing 20d vol)': 0,
              'COMPDEC (compression decile mean)': 1,
              'GARCH11 (per-stock GARCH(1,1))': 2}
ranked = sorted(mae.items(), key=lambda kv: (round(kv[1], 4),
                                             simplicity[kv[0]]))

print("\n" + "=" * 78)
print("MAE RESULTS (annualized vol units, identical stock-days)")
print("=" * 78)
print(f"\n{'rank':>5}  {'forecaster':<36} {'MAE':>9}")
table_lines = []
for i, (name, v) in enumerate(ranked, 1):
    line = f"{i:>5}  {name:<36} {v:>9.4f}"
    print(line)
    table_lines.append(line)
winner = ranked[0][0]
print(f"\nWINNER: {winner}")
print("Decision rule: lowest MAE; ties to 4dp broken toward simplicity")
print("(TRAIL20 > COMPDEC > GARCH11).")

# --------------------------------------------------------------------------
# Append to gate log
# --------------------------------------------------------------------------
stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
entry = [f"\n---\n\n## K0 horse race decision — {stamp}",
         "- Phase: K0 (volatility forecaster horse race, per V1's locked "
         "post-PASS sequencing)",
         "- Pre-registration: docs/PhaseK0_PreRegistration_"
         "ForecasterHorseRace.md (commit 798d54a)",
         "- Script: src/21_forecaster_horse_race.py",
         f"- Window: DEV {DEV_START} to {DEV_END}; common sample "
         f"{len(common):,} stock-days",
         "- Kronos: not built; a future Kronos competes 1-on-1 against "
         "this winner via the parquet contract",
         f"\n### Decision: **sanctioned E[RV] forecaster = {winner}**\n",
         "```"]
entry += table_lines
entry.append("")
entry.append(f"GARCH fit coverage: {n_fit:,} stocks fit, {n_skip:,} skipped, "
             f"{n_fail:,} failed")
entry.append("reasoning: lowest MAE on the pre-registered common dev sample;")
entry.append("tie rule (simplicity) " +
             ("was needed." if round(ranked[0][1], 4) ==
              round(ranked[1][1], 4) else "was not needed."))
entry.append("```")
entry.append("\nThis result satisfies V1's 'horse race before OptionMetrics "
             "contact'\nrequirement: K1 (options) is now unlocked, using "
             "this forecaster.")
with open(log_path, 'a') as fh:
    fh.write("\n".join(entry) + "\n")
print(f"\n[OK] Appended K0 decision to {log_path}")

log_text = log_path.read_text()
for name, key in [("R1", "Phase: R1"), ("R2", "Phase: R2"),
                  ("V1 dev", "Phase: V1 (volatility"),
                  ("V1 holdout", "V1 (volatility compression) — HOLDOUT"),
                  ("V2 dev", "Phase: V2 (sector-relative volatility"),
                  ("V2 holdout", "V2 (sector-relative compression) — HOLDOUT"),
                  ("K0 (this)", "Phase: K0")]:
    ok = key in log_text
    print(f"  {name} entry present: {ok} {'[PASS]' if ok else '[FAIL]'}")

print("\n" + "=" * 78)
print("K0 HORSE RACE COMPLETE.")
print("=" * 78)
