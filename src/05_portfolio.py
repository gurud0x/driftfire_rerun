import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Step 5: Portfolio construction — overlapping daily tranches per
# pre-registration Section 5. Primary hold 5 days; secondary 1/3/10 days
# (diagnostic only). Costs per Section 6: 15 bps per side, charged on each
# tranche's entry day and exit day.
#
# Tranche mechanics (horizon H):
#   Signal day t -> tranche of that day's long candidates, equal-weighted,
#   entered at t+1 open, exited at t+H close.
#   Tranche daily return: day t+1 = DlyClose/DlyOpen - 1 (open fill);
#   days t+2..t+H = DlyRet (close-to-close total return).
#   H tranches live at once, each 1/H of capital; portfolio daily return
#   = mean of live tranches' daily returns.
# ---------------------------------------------------------------------------

project_root = Path(__file__).parent.parent
panel_path = project_root / 'data' / 'processed' / 'signal_panel.parquet'
crsp_path = project_root / 'data' / 'processed' / 'crsp_combined.parquet'
output_path = project_root / 'data' / 'processed' / 'portfolio_returns_daily.parquet'

COST_SIDE = 0.0015          # 15 bps per side, Section 6 base case
HORIZONS = [1, 3, 5, 10]

print("=" * 80)
print("PORTFOLIO CONSTRUCTION: overlapping tranches, horizons 1/3/5/10 days")
print("=" * 80)

panel = pd.read_parquet(panel_path,
                        columns=['PERMNO', 'DlyCalDt', 'is_long_candidate'])
cands = panel[panel['is_long_candidate']]
print(f"\n[OK] Signal panel: {len(panel):,} rows; long-candidate stock-days: "
      f"{len(cands):,}")

# signal_panel stores period fwd returns, not daily path returns — the daily
# rolling portfolio needs the per-stock DAILY series, so reload DlyRet /
# DlyOpen / DlyClose from crsp_combined (deduped as in Step 3).
print("[NOTE] signal_panel has period fwd_ret only; loading daily DlyRet/"
      "DlyOpen/DlyClose from crsp_combined for the daily tranche paths.")
px = pd.read_parquet(crsp_path,
                     columns=['PERMNO', 'DlyCalDt', 'DlyRet', 'DlyOpen',
                              'DlyClose'])
px = px[px['PERMNO'].isin(cands['PERMNO'].unique())]
px = (px.drop_duplicates(['PERMNO', 'DlyCalDt'], keep='first')
        .sort_values(['PERMNO', 'DlyCalDt']))
print(f"[OK] Daily price rows for candidate PERMNOs: {len(px):,} "
      f"({px['PERMNO'].nunique():,} PERMNOs)")

# --------------------------------------------------------------------------
# Matrices on the market calendar: R (close-close ret), OC (open-close ret),
# C (candidate indicator). Missing stock-day return -> 0 (position sits in
# cash after a delisting; the delisting return itself is in the last DlyRet).
# --------------------------------------------------------------------------
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
n_stocks = Cv.sum(axis=1)                      # candidates per signal day
D = len(calendar)
print(f"\nCalendar days: {D}, matrix shape: {Rv.shape}")
print(f"Signal days with candidates: {(n_stocks > 0).sum()} "
      f"(first {calendar[n_stocks > 0][0].date()}, "
      f"last {calendar[n_stocks > 0][-1].date()})")

# --------------------------------------------------------------------------
# Build tranche contribution series and aggregate per horizon
# --------------------------------------------------------------------------
results = {}
for H in HORIZONS:
    gross = np.zeros(D)
    net = np.zeros(D)
    live = np.zeros(D)                        # live tranche count per day
    for k in range(1, H + 1):
        X = OCv if k == 1 else Rv
        # tranche formed on day t (0..D-k-1) contributes on day t+k
        with np.errstate(invalid='ignore', divide='ignore'):
            contrib = (Cv[:D - k] * X[k:]).sum(axis=1) / n_stocks[:D - k]
        contrib = np.where(n_stocks[:D - k] > 0, contrib, 0.0)
        cost = (COST_SIDE if k == 1 else 0.0) + (COST_SIDE if k == H else 0.0)
        has_tranche = (n_stocks[:D - k] > 0).astype(float)
        gross[k:] += contrib
        net[k:] += contrib - cost * has_tranche
        live[k:] += has_tranche
    with np.errstate(invalid='ignore', divide='ignore'):
        gross = np.where(live == H, gross / H, np.nan)   # NaN until fully ramped
        net = np.where(live == H, net / H, np.nan)
    results[H] = {'gross': gross, 'net': net}
    valid = ~np.isnan(net)
    print(f"H={H:>2}: valid days {valid.sum()} "
          f"(ramp-up days excluded: {D - valid.sum()}), "
          f"gross mean {np.nanmean(gross)*1e4:+.2f} bps/day, "
          f"net mean {np.nanmean(net)*1e4:+.2f} bps/day")

# --------------------------------------------------------------------------
# Turnover, 5-day primary: each day 1/5 of capital enters and 1/5 exits.
# --------------------------------------------------------------------------
print("\n" + "-" * 80)
print("TURNOVER (5-day primary)")
print("-" * 80)
daily_oneway = 1.0 / 5.0
ann_oneway = daily_oneway * 252
print(f"Daily one-way turnover: {daily_oneway:.3f} of capital "
      f"(1/5 enters, 1/5 exits each day)")
print(f"Annualized one-way turnover: {ann_oneway:.1f}x "
      f"{'[PASS - within expected 40-60x]' if 40 <= ann_oneway <= 60 else '[WARNING - outside expected 40-60x, investigate]'}")
print(f"Implied annual cost drag: 2 x {ann_oneway:.1f} x 15bps = "
      f"{2 * ann_oneway * COST_SIDE * 100:.1f}% per year")

# --------------------------------------------------------------------------
# Save
# --------------------------------------------------------------------------
out = pd.DataFrame({
    'date': calendar,
    'port_ret_5d_net': results[5]['net'],
    'port_ret_5d_gross': results[5]['gross'],
    'port_ret_1d_net': results[1]['net'],
    'port_ret_3d_net': results[3]['net'],
    'port_ret_10d_net': results[10]['net'],
    'turnover_5d': np.where(~np.isnan(results[5]['net']),
                            2.0 * daily_oneway, np.nan),
})
out = out.dropna(subset=['port_ret_5d_net', 'port_ret_1d_net',
                         'port_ret_3d_net', 'port_ret_10d_net'], how='all')
out.to_parquet(output_path, index=False)

print("\n" + "-" * 80)
print("VALIDATION SUMMARY")
print("-" * 80)
print(f"Saved rows: {len(out):,}, "
      f"{out['date'].min().date()} to {out['date'].max().date()}")
for c in ['port_ret_5d_gross', 'port_ret_5d_net', 'port_ret_1d_net',
          'port_ret_3d_net', 'port_ret_10d_net']:
    s = out[c].dropna()
    ann_ret = s.mean() * 252
    ann_vol = s.std() * np.sqrt(252)
    print(f"  {c:>18}: n={len(s):>5}  mean/day={s.mean()*1e4:+7.2f} bps  "
          f"ann={ann_ret*100:+6.2f}%  vol={ann_vol*100:5.2f}%  "
          f"Sharpe(desc)={ann_ret/ann_vol:+.2f}")
print("\n[NOTE] Sharpe figures are descriptive only — the gate claim is the")
print("factor-model intercept (pre-registration Section 7), decided in Step 6.")
print(f"[OK] Saved to {output_path}")

print("\n" + "=" * 80)
print("PORTFOLIO CONSTRUCTION COMPLETE")
print("=" * 80)
