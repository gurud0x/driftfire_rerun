import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

# ---------------------------------------------------------------------------
# EXPLORATORY decay check (dev window only): does the V1 compression
# signal's forward-vol predictability survive at a 30-day horizon?
# Feeds the K1 tenor decision (10d vs 30d options). NOT a gated phase:
# nothing is appended to results/gate_log.md and no processed file is
# written. Holdout (2022+) is never loaded — returns are loaded only
# through 2021-12-31, so fwd_30d is null for signal days within 30
# trading days of the dev boundary (those days drop out of the 30d test).
# ---------------------------------------------------------------------------

DEV_END = '2021-12-31'

project_root = Path(__file__).parent.parent
crsp_path = project_root / 'data' / 'processed' / 'crsp_combined.parquet'
v1_path = project_root / 'data' / 'processed' / 'compression_signal_v1.parquet'

print("=" * 78)
print("EXPLORATORY: compression-signal forecast decay, 5/10/20/30d, "
      "dev window only")
print("(not gated, not logged; informs the K1 tenor decision)")
print("=" * 78)

panel = pd.read_parquet(v1_path)
panel = panel[panel['DlyCalDt'] <= DEV_END]
print(f"\n[OK] V1 panel, dev window: {len(panel):,} stock-days "
      f"({panel['DlyCalDt'].min().date()} to {panel['DlyCalDt'].max().date()})")

# --------------------------------------------------------------------------
# realized_vol_fwd_30d — identical construction to src/10, but with
# returns loaded ONLY through DEV_END (no 2022+ data in this script)
# --------------------------------------------------------------------------
px = pd.read_parquet(crsp_path, columns=['PERMNO', 'DlyCalDt', 'DlyRet'])
px = px[px['PERMNO'].isin(panel['PERMNO'].unique())]
px = px[px['DlyCalDt'] <= DEV_END]
px = (px.drop_duplicates(['PERMNO', 'DlyCalDt'], keep='first')
        .sort_values(['PERMNO', 'DlyCalDt']).reset_index(drop=True))
print(f"[OK] Daily returns loaded through {DEV_END} only: {len(px):,} rows")

px['realized_vol_fwd_30d'] = px.groupby('PERMNO', sort=False)['DlyRet'].transform(
    lambda s: s.rolling(30, min_periods=30).std().shift(-30)) * np.sqrt(252)
print("fwd_30d: rolling(30).std().shift(-30) * sqrt(252) — window is exactly")
print("t+1..t+30; null within 30 trading days of the dev boundary (no 2022")
print("returns are loaded to complete those windows).")

panel = panel.merge(px[['PERMNO', 'DlyCalDt', 'realized_vol_fwd_30d']],
                    on=['PERMNO', 'DlyCalDt'], how='left')
print(f"fwd_30d non-null in dev panel: "
      f"{int(panel['realized_vol_fwd_30d'].notna().sum()):,} of {len(panel):,}")


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
    m = sm.OLS(slopes.values, np.ones(len(slopes))).fit(
        cov_type='HAC', cov_kwds={'maxlags': 10})
    return len(slopes), m.params[0], m.tvalues[0]


# --------------------------------------------------------------------------
# Decay table
# --------------------------------------------------------------------------
targets = [('5d', 'realized_vol_fwd_5d'), ('10d', 'realized_vol_fwd_10d'),
           ('20d', 'realized_vol_fwd_20d'), ('30d', 'realized_vol_fwd_30d')]
print("\n" + "=" * 78)
print("DECAY TABLE: Fama-MacBeth slope of realized_vol_fwd ~ "
      "compression_decile")
print("=" * 78)
print(f"\n{'horizon':>8} {'days':>6} {'mean slope':>12} {'NW t':>8} "
      f"{'|t|>=3.0':>9}")
results = {}
for lab, col in targets:
    n, mean_s, t_s = fama_macbeth(panel, col)
    clears = abs(t_s) >= 3.0 and mean_s < 0
    results[lab] = (n, mean_s, t_s, clears)
    print(f"{lab:>8} {n:>6} {mean_s:>+12.6f} {t_s:>+8.3f} "
          f"{'YES' if clears else 'NO':>9}")

# --------------------------------------------------------------------------
# Monotonicity at 30d
# --------------------------------------------------------------------------
print("\n" + "-" * 78)
print("MONOTONICITY AT 30d: mean realized_vol_fwd_30d by compression decile")
print("-" * 78)
d30 = panel.dropna(subset=['realized_vol_fwd_30d', 'compression_decile'])
mono = d30.groupby('compression_decile')['realized_vol_fwd_30d'].agg(
    ['mean', 'count'])
print(f"\n{'decile':>7} {'mean rv_fwd_30d':>16} {'count':>10}   "
      f"(1 = most compressed)")
for dec, r in mono.iterrows():
    print(f"{dec:>7.0f} {r['mean']:>16.4f} {int(r['count']):>10,}")
u_left = mono['mean'].iloc[0] - mono['mean'].min()
u_right = mono['mean'].iloc[-1] - mono['mean'].min()
print(f"\nU-shape check: decile-1 elevation over the minimum = {u_left:.4f}; "
      f"decile-10 elevation = {u_right:.4f}")

# --------------------------------------------------------------------------
# Plain summary
# --------------------------------------------------------------------------
n10, m10, t10, _ = results['10d']
n30, m30, t30, c30 = results['30d']
print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
if c30:
    print(f"At 30 days the compression signal STILL clears the "
          f"gate-equivalent bar:")
    print(f"mean slope {m30:+.6f}, NW t {t30:+.2f} (vs the logged 10d dev "
          f"result {m10:+.6f}, t {t10:+.2f}).")
    print(f"Slope magnitude at 30d is {abs(m30)/abs(m10)*100:.0f}% of the "
          f"10d slope — decayed but alive.")
else:
    print(f"At 30 days the signal does NOT clear the gate-equivalent bar")
    print(f"(mean slope {m30:+.6f}, NW t {t30:+.2f}, vs 10d: {m10:+.6f}, "
          f"t {t10:+.2f}) — meaningful decay.")
print("\nExploratory only: nothing appended to results/gate_log.md, no")
print("processed output written. This informs the K1 tenor decision in its")
print("pre-registration; it is not itself a pre-registered claim.")
print("=" * 78)
