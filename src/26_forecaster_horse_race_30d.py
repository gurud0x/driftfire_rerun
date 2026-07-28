import pandas as pd
import numpy as np
from pathlib import Path
from arch import arch_model
import warnings

# ---------------------------------------------------------------------------
# EXPLORATORY: K0 horse race re-run at the 30-DAY horizon, informing K1's
# tenor decision. NOT a gated phase: nothing is appended to
# results/gate_log.md. Dev window only; returns loaded only through
# 2021-12-31 (fwd_30d null near the boundary, same treatment as src/25).
# Machinery mirrors src/21_forecaster_horse_race.py exactly except the
# target and the GARCH forecast horizon.
# ---------------------------------------------------------------------------

DEV_START = '2015-01-01'
DEV_END = '2021-12-31'
MIN_GARCH_OBS = 250
H = 30

project_root = Path(__file__).parent.parent
v1_path = project_root / 'data' / 'processed' / 'compression_signal_v1.parquet'
crsp_path = project_root / 'data' / 'processed' / 'crsp_combined.parquet'

print("=" * 78)
print("EXPLORATORY K0 RE-RUN AT 30 DAYS: TRAIL20 vs GARCH11 vs COMPDEC,")
print("dev window, MAE on realized_vol_fwd_30d (not gated, not logged)")
print("=" * 78)

panel = pd.read_parquet(v1_path,
                        columns=['PERMNO', 'DlyCalDt', 'compression_decile'])
panel = panel[(panel['DlyCalDt'] >= DEV_START) & (panel['DlyCalDt'] <= DEV_END)]
print(f"\n[OK] V1 panel, dev window: {len(panel):,} stock-days")

px = pd.read_parquet(crsp_path, columns=['PERMNO', 'DlyCalDt', 'DlyRet'])
px = px[px['PERMNO'].isin(panel['PERMNO'].unique())]
px = px[(px['DlyCalDt'] >= '2014-01-01') & (px['DlyCalDt'] <= DEV_END)]
px = (px.drop_duplicates(['PERMNO', 'DlyCalDt'], keep='first')
        .sort_values(['PERMNO', 'DlyCalDt']).reset_index(drop=True))
print(f"[OK] Daily returns (2014 warmup + dev, nothing past {DEV_END}): "
      f"{len(px):,} rows, {px['PERMNO'].nunique():,} PERMNOs")

grp = px.groupby('PERMNO', sort=False)

# target: fwd_30d, same construction as src/25 (null near dev boundary)
px['rv_fwd_30d'] = grp['DlyRet'].transform(
    lambda s: s.rolling(H, min_periods=H).std().shift(-H)) * np.sqrt(252)

# TRAIL20: trailing 20d vol ending at t (identical to K0)
px['trail20'] = grp['DlyRet'].transform(
    lambda s: s.rolling(20, min_periods=15).std()) * np.sqrt(252)

# --------------------------------------------------------------------------
# GARCH(1,1) per stock, K0 fitting approach, 30-day-ahead forecast
# --------------------------------------------------------------------------
print(f"\nFitting per-stock GARCH(1,1), 30-day-ahead forecasts "
      f"(min {MIN_GARCH_OBS} dev obs)...")
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
    r = dev['DlyRet'].values * 100.0
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
        h_t = res.conditional_volatility ** 2
        eps2 = (r - res.params['mu']) ** 2
        h_next = omega + alpha * eps2 + beta * h_t
        decay = np.mean([ab ** (k - 1) for k in range(1, H + 1)])
        mean_hH = sig2_u + (h_next - sig2_u) * decay
        fc = np.sqrt(mean_hH * 252.0) / 100.0
        garch_frames.append(pd.DataFrame({
            'PERMNO': p, 'DlyCalDt': dev['DlyCalDt'].values, 'garch11': fc}))
        n_fit += 1
    except Exception:
        n_fail += 1

garch = pd.concat(garch_frames, ignore_index=True)
print(f"  GARCH fits: {n_fit:,} ok, {n_skip:,} skipped, {n_fail:,} "
      f"failed/nonstationary")

# --------------------------------------------------------------------------
# COMPDEC: decile -> dev-window mean rv_fwd_30d
# --------------------------------------------------------------------------
ev = panel.merge(px[['PERMNO', 'DlyCalDt', 'rv_fwd_30d', 'trail20']],
                 on=['PERMNO', 'DlyCalDt'], how='left')
dec_map = (ev.dropna(subset=['compression_decile', 'rv_fwd_30d'])
           .groupby('compression_decile')['rv_fwd_30d'].mean())
print("\nCOMPDEC forecast values (decile -> dev mean rv_fwd_30d):")
for dec, v in dec_map.items():
    print(f"  decile {dec:>4.0f}: {v:.4f}")
ev['compdec'] = ev['compression_decile'].map(dec_map)
ev = ev.merge(garch, on=['PERMNO', 'DlyCalDt'], how='left')

common = ev.dropna(subset=['rv_fwd_30d', 'trail20', 'garch11', 'compdec'])
print(f"\nCommon evaluation sample: {len(common):,} stock-days of "
      f"{len(ev):,} dev stock-days ({len(common)/len(ev)*100:.1f}%)")

# --------------------------------------------------------------------------
# MAE table, K0-comparable format
# --------------------------------------------------------------------------
mae = {
    'TRAIL20 (trailing 20d vol)': (common['trail20'] -
                                   common['rv_fwd_30d']).abs().mean(),
    'GARCH11 (per-stock GARCH(1,1))': (common['garch11'] -
                                       common['rv_fwd_30d']).abs().mean(),
    'COMPDEC (compression decile mean)': (common['compdec'] -
                                          common['rv_fwd_30d']).abs().mean(),
}
simplicity = {'TRAIL20 (trailing 20d vol)': 0,
              'COMPDEC (compression decile mean)': 1,
              'GARCH11 (per-stock GARCH(1,1))': 2}
ranked = sorted(mae.items(), key=lambda kv: (round(kv[1], 4),
                                             simplicity[kv[0]]))

print("\n" + "=" * 78)
print("MAE RESULTS AT 30 DAYS (annualized vol units, identical stock-days)")
print("=" * 78)
print(f"\n{'rank':>5}  {'forecaster':<36} {'MAE':>9}")
for i, (name, v) in enumerate(ranked, 1):
    print(f"{i:>5}  {name:<36} {v:>9.4f}")

winner = ranked[0][0]
K0_RANKING = ['TRAIL20 (trailing 20d vol)',
              'GARCH11 (per-stock GARCH(1,1))',
              'COMPDEC (compression decile mean)']
K0_MAE = {'TRAIL20 (trailing 20d vol)': 0.1751,
          'GARCH11 (per-stock GARCH(1,1))': 0.1810,
          'COMPDEC (compression decile mean)': 0.1980}
same_order = [n for n, _ in ranked] == K0_RANKING

print(f"\nWINNER AT 30 DAYS: {winner}")
print(f"\nComparison with the logged K0 result at 10 days "
      f"(MAE: TRAIL20 0.1751, GARCH11 0.1810, COMPDEC 0.1980):")
if same_order:
    print("  Ranking UNCHANGED from K0 — TRAIL20 still best; the K0")
    print("  decision carries to the 30-day horizon without amendment.")
else:
    print("  RANKING CHANGED from K0:")
    for i, (name, v) in enumerate(ranked, 1):
        print(f"    {i}. {name} (30d MAE {v:.4f}, 10d MAE {K0_MAE[name]:.4f})")
    print("  This is an exploratory finding. The K0 pre-registration's")
    print("  sanctioned forecaster remains its logged 10-day winner; any")
    print("  change of sanctioned forecaster for K1's 30-day design must be")
    print("  made explicitly in the K1 pre-registration, not silently here.")

print("\nExploratory only: nothing appended to results/gate_log.md.")
print("=" * 78)
