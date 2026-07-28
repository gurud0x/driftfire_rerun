import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Phase R2, Step 2: overlapping-tranche portfolio on the volume-conditioned
# candidates. Engine cloned from 05_portfolio.py. Horizons: 5d primary,
# 10d/20d secondary (pre-registration Section 5). Zero-candidate days sit
# in cash (Section 4, locked).
# ---------------------------------------------------------------------------

project_root = Path(__file__).parent.parent
panel_path = project_root / 'data' / 'processed' / 'signal_panel_r2.parquet'
crsp_path = project_root / 'data' / 'processed' / 'crsp_combined.parquet'
output_path = project_root / 'data' / 'processed' / 'portfolio_returns_daily_r2.parquet'

COST_SIDE = 0.0015
HORIZONS = [5, 10, 20]

print("=" * 80)
print("R2 PORTFOLIO: overlapping tranches, horizons 5 (primary) / 10 / 20")
print("=" * 80)

panel = pd.read_parquet(panel_path,
                        columns=['PERMNO', 'DlyCalDt', 'is_long_candidate_r2'])
cands = panel[panel['is_long_candidate_r2']]
print(f"\n[OK] R2 candidates: {len(cands):,} stock-days, "
      f"{cands['PERMNO'].nunique():,} PERMNOs")

px = pd.read_parquet(crsp_path,
                     columns=['PERMNO', 'DlyCalDt', 'DlyRet', 'DlyOpen',
                              'DlyClose'])
px = px[px['PERMNO'].isin(cands['PERMNO'].unique())]
px = (px.drop_duplicates(['PERMNO', 'DlyCalDt'], keep='first')
        .sort_values(['PERMNO', 'DlyCalDt']))
print(f"[OK] Daily price rows: {len(px):,}")

R = px.pivot(index='DlyCalDt', columns='PERMNO', values='DlyRet')
calendar = R.index
OC = (px.pivot(index='DlyCalDt', columns='PERMNO', values='DlyClose') /
      px.pivot(index='DlyCalDt', columns='PERMNO', values='DlyOpen') - 1.0)
OC = OC.reindex(index=calendar, columns=R.columns)
C = (cands.assign(v=1.0)
     .pivot(index='DlyCalDt', columns='PERMNO', values='v')
     .reindex(index=calendar, columns=R.columns).fillna(0.0))

Rv = R.fillna(0.0).to_numpy()
OCv = OC.fillna(0.0).to_numpy()
Cv = C.to_numpy()
n_stocks = Cv.sum(axis=1)
D = len(calendar)
sig_days = n_stocks > 0
first_sig = np.argmax(sig_days)
print(f"\nCalendar days: {D}; matrix {Rv.shape}")
print(f"Non-empty signal days: {int(sig_days.sum())} of "
      f"{D - first_sig} eligible (empty -> cash tranche, no cost)")

results = {}
for H in HORIZONS:
    gross = np.zeros(D)
    net = np.zeros(D)
    live = np.zeros(D)              # tranche slots elapsed (incl. empty ones)
    for k in range(1, H + 1):
        X = OCv if k == 1 else Rv
        with np.errstate(invalid='ignore', divide='ignore'):
            contrib = (Cv[:D - k] * X[k:]).sum(axis=1) / n_stocks[:D - k]
        has_tranche = (n_stocks[:D - k] > 0).astype(float)
        contrib = np.where(has_tranche > 0, contrib, 0.0)   # empty slot = cash
        cost = (COST_SIDE if k == 1 else 0.0) + (COST_SIDE if k == H else 0.0)
        gross[k:] += contrib
        net[k:] += contrib - cost * has_tranche
        # a slot exists once the formation day is inside the sample, even if
        # that day had zero candidates (the slot holds cash)
        live[k:] += (np.arange(D - k) >= first_sig).astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        gross = np.where(live == H, gross / H, np.nan)
        net = np.where(live == H, net / H, np.nan)
    results[H] = {'gross': gross, 'net': net}
    print(f"H={H:>2}: valid days {int((~np.isnan(net)).sum())}, "
          f"gross mean {np.nanmean(gross)*1e4:+.2f} bps/day, "
          f"net mean {np.nanmean(net)*1e4:+.2f} bps/day")

# --------------------------------------------------------------------------
# Realized turnover, 5-day primary: 1/5 of capital enters on each non-empty
# formation day and exits 5 days later; empty slots trade nothing.
# --------------------------------------------------------------------------
print("\n" + "-" * 80)
print("TURNOVER (5-day primary, realized)")
print("-" * 80)
frac_nonempty = sig_days[first_sig:].mean()
daily_oneway = (1.0 / 5.0) * frac_nonempty
ann_oneway = daily_oneway * 252
print(f"Non-empty formation days: {frac_nonempty*100:.2f}%")
print(f"Realized annualized one-way turnover: {ann_oneway:.1f}x "
      f"(mechanical max {252/5:.1f}x)")
print(f"Implied annual cost drag: 2 x {ann_oneway:.1f} x 15bps = "
      f"{2 * ann_oneway * COST_SIDE * 100:.1f}% per year")

# --------------------------------------------------------------------------
# Save + summary
# --------------------------------------------------------------------------
out = pd.DataFrame({
    'date': calendar,
    'port_ret_5d_net': results[5]['net'],
    'port_ret_5d_gross': results[5]['gross'],
    'port_ret_10d_net': results[10]['net'],
    'port_ret_20d_net': results[20]['net'],
    'turnover_5d': np.where(~np.isnan(results[5]['net']),
                            2.0 * daily_oneway, np.nan),
})
out = out.dropna(subset=['port_ret_5d_net', 'port_ret_10d_net',
                         'port_ret_20d_net'], how='all')
out.to_parquet(output_path, index=False)

print("\n" + "-" * 80)
print("VALIDATION SUMMARY")
print("-" * 80)
print(f"Saved rows: {len(out):,}, "
      f"{out['date'].min().date()} to {out['date'].max().date()}")
for c in ['port_ret_5d_gross', 'port_ret_5d_net', 'port_ret_10d_net',
          'port_ret_20d_net']:
    s = out[c].dropna()
    ann_ret = s.mean() * 252
    ann_vol = s.std() * np.sqrt(252)
    print(f"  {c:>18}: n={len(s):>5}  mean/day={s.mean()*1e4:+7.2f} bps  "
          f"ann={ann_ret*100:+6.2f}%  vol={ann_vol*100:5.2f}%  "
          f"Sharpe(desc)={ann_ret/ann_vol:+.2f}")
print("\n[NOTE] Descriptive only — the gate claim is the factor-model")
print("intercept (Section 7), decided by src/09_gate_check_r2.py.")
print(f"[OK] Saved to {output_path}")

print("\n" + "=" * 80)
print("R2 PORTFOLIO CONSTRUCTION COMPLETE")
print("=" * 80)
