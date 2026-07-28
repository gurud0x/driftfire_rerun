import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Phase R2, Step 1: volume-conditioned signal, per
# docs/PhaseR2_PreRegistration_VolumeConditionedReversal.md Section 4
# (committed d352510 before this script first ran).
#
# is_long_candidate_r2 = R1 bottom-decile rank AND:
#   w = worst-DlyRet day in [t-5, t-1];
#   (1) DlyRet_w < 0
#   (2) DlyPrcVol_w / mean(DlyPrcVol, 20 tdays ending w-1) >= 2.0
#   (3) >= 15 of the 20 baseline days non-null (rolling min_periods=15)
# ---------------------------------------------------------------------------

project_root = Path(__file__).parent.parent
crsp_path = project_root / 'data' / 'processed' / 'crsp_combined.parquet'
panel_path = project_root / 'data' / 'processed' / 'signal_panel.parquet'
output_path = project_root / 'data' / 'processed' / 'signal_panel_r2.parquet'

VOL_THRESH = 2.0        # Section 4, locked
BASE_WIN = 20           # trailing baseline window
BASE_MINP = 15          # data-sufficiency minimum

print("=" * 80)
print("R2 SIGNAL: R1 bottom-decile losers AND worst-day dollar-volume spike")
print("=" * 80)

r1 = pd.read_parquet(panel_path)
print(f"\n[OK] R1 signal panel: {r1.shape[0]:,} rows "
      f"({r1['DlyCalDt'].min().date()} to {r1['DlyCalDt'].max().date()})")
print(f"  R1 long candidates: {r1['is_long_candidate'].sum():,}")

df = pd.read_parquet(crsp_path,
                     columns=['PERMNO', 'DlyCalDt', 'DlyRet', 'DlyPrcVol',
                              'DlyOpen', 'DlyClose'])
df = df[df['PERMNO'].isin(r1['PERMNO'].unique())]

# same multi-distribution dedupe as 03_signal.py, same halting audit
dup_mask = df.duplicated(['PERMNO', 'DlyCalDt'], keep=False)
if dup_mask.any():
    conflicting = (df[dup_mask].groupby(['PERMNO', 'DlyCalDt'])
                   .nunique().gt(1).any(axis=1).sum())
    print(f"\nMulti-distribution duplicate PERMNO-days: "
          f"{df[dup_mask][['PERMNO','DlyCalDt']].drop_duplicates().shape[0]:,}; "
          f"groups differing on loaded fields: {conflicting} "
          f"{'[PASS - safe to dedupe]' if conflicting == 0 else '[FAIL]'}")
    if conflicting != 0:
        raise SystemExit(1)
    df = df.drop_duplicates(['PERMNO', 'DlyCalDt'], keep='first')

df = df.sort_values(['PERMNO', 'DlyCalDt']).reset_index(drop=True)
sorted_ok = df.groupby('PERMNO')['DlyCalDt'].is_monotonic_increasing.all()
print(f"\nDaily rows: {len(df):,}; sorted within PERMNO: {sorted_ok} "
      f"{'[PASS]' if sorted_ok else '[FAIL]'}")
if not sorted_ok:
    raise SystemExit(1)
print(f"DlyPrcVol null rate: {df['DlyPrcVol'].isna().mean()*100:.2f}%")

grp = df.groupby('PERMNO', sort=False)

# --------------------------------------------------------------------------
# Dollar-volume spike ratio: DlyPrcVol_d vs trailing 20d mean ending d-1
# --------------------------------------------------------------------------
base = grp['DlyPrcVol'].transform(
    lambda s: s.rolling(BASE_WIN, min_periods=BASE_MINP).mean().shift(1))
df['vol_ratio'] = np.where(base > 0, df['DlyPrcVol'] / base, np.nan)
print(f"\nVolume ratio non-null: {df['vol_ratio'].notna().mean()*100:.1f}% of "
      f"daily rows (needs >= {BASE_MINP} of {BASE_WIN} baseline days, "
      f"ending the day BEFORE the spike day — no same-day contamination)")

# --------------------------------------------------------------------------
# Worst day w in [t-5, t-1] and its volume ratio
# --------------------------------------------------------------------------
ret_lags = np.column_stack([grp['DlyRet'].shift(k).to_numpy()
                            for k in range(1, 6)])
ratio_lags = np.column_stack([grp['vol_ratio'].shift(k).to_numpy()
                              for k in range(1, 6)])
full_window = ~np.isnan(ret_lags).any(axis=1)      # same requirement as SIG_5d
ret_filled = np.where(np.isnan(ret_lags), np.inf, ret_lags)
wi = ret_filled.argmin(axis=1)
rows = np.arange(len(df))
df['worst_day_ret'] = np.where(full_window, ret_lags[rows, wi], np.nan)
df['vol_ratio_worst'] = np.where(full_window, ratio_lags[rows, wi], np.nan)
df['worst_day_offset'] = np.where(full_window, wi + 1, np.nan)  # t-offset

vol_cond = (full_window &
            (df['worst_day_ret'] < 0) &
            (df['vol_ratio_worst'] >= VOL_THRESH))
df['vol_condition'] = vol_cond
print(f"Stock-days passing volume condition (all daily rows): "
      f"{vol_cond.sum():,} ({vol_cond.mean()*100:.1f}%)")

# --------------------------------------------------------------------------
# fwd_ret_20d, same open-fill construction as 03_signal.py
# --------------------------------------------------------------------------
df['_lg'] = np.log1p(df['DlyRet'])
df['_oc_next'] = grp.apply(
    lambda g: (g['DlyClose'] / g['DlyOpen'] - 1.0).shift(-1),
    include_groups=False).reset_index(level=0, drop=True).sort_index()
tail20 = grp['_lg'].transform(lambda s: s.rolling(19).sum().shift(-20))
df['fwd_ret_20d'] = (1.0 + df['_oc_next']) * np.exp(tail20) - 1.0
df = df.drop(columns=['_lg', '_oc_next'])

# --------------------------------------------------------------------------
# Join onto the R1 panel; apply the AND
# --------------------------------------------------------------------------
panel = r1.merge(
    df[['PERMNO', 'DlyCalDt', 'worst_day_ret', 'vol_ratio_worst',
        'worst_day_offset', 'vol_condition', 'fwd_ret_20d']],
    on=['PERMNO', 'DlyCalDt'], how='left')
panel['vol_condition'] = panel['vol_condition'].fillna(False)
panel['is_long_candidate_r2'] = (panel['is_long_candidate'] &
                                 panel['vol_condition'])

# --------------------------------------------------------------------------
# VALIDATION
# --------------------------------------------------------------------------
print("\n" + "-" * 80)
print("VALIDATION")
print("-" * 80)

dups = panel.duplicated(['PERMNO', 'DlyCalDt']).sum()
print(f"\nDuplicate PERMNO-day rows: {dups} "
      f"{'[PASS]' if dups == 0 else '[FAIL]'}")
print(f"Panel rows: {len(panel):,}; date range "
      f"{panel['DlyCalDt'].min().date()} to {panel['DlyCalDt'].max().date()}")

r2c = panel[panel['is_long_candidate_r2']]
print(f"\nR2 long candidates: {len(r2c):,} stock-days "
      f"(R1: {panel['is_long_candidate'].sum():,}; kept "
      f"{len(r2c)/panel['is_long_candidate'].sum()*100:.1f}%)")

daily = r2c.groupby('DlyCalDt').size()
all_days = panel['DlyCalDt'].drop_duplicates().sort_values()
daily = daily.reindex(all_days, fill_value=0)
zero_days = int((daily == 0).sum())
print(f"Daily R2 candidate count: min={daily.min()}, max={daily.max()}, "
      f"mean={daily.mean():.1f} over {len(daily)} days")
in_range = 15 <= daily.mean() <= 30
print(f"Pre-registered expectation ~15-30/day: "
      f"{'[PASS]' if in_range else f'[WARNING - mean {daily.mean():.1f} outside 15-30, Section 4 audit trigger]'}")
print(f"Zero-candidate days: {zero_days} "
      f"(tranches on those days sit in cash, per Section 4 — locked rule)")

print("\nfwd_ret_20d: count={:,}, mean={:+.5f}, std={:.5f}".format(
    int(panel['fwd_ret_20d'].notna().sum()),
    panel['fwd_ret_20d'].mean(), panel['fwd_ret_20d'].std()))

# --------------------------------------------------------------------------
# Spot check: 3 random R2 candidates (fixed seed 42) — ratio math by eye
# --------------------------------------------------------------------------
print("\n" + "-" * 80)
print("SPOT CHECK: 3 random R2 candidates — verify volume-ratio math by eye")
print("-" * 80)

rng = np.random.default_rng(42)
picks = r2c.iloc[rng.choice(len(r2c), 3, replace=False)]
for _, row in picks.iterrows():
    p, d = row['PERMNO'], row['DlyCalDt']
    hist = df[df['PERMNO'] == p].reset_index(drop=True)
    i = hist.index[hist['DlyCalDt'] == d][0]
    print(f"\nPERMNO {p}, signal day t = {d.date()}  "
          f"(SIG_rank={row['SIG_rank']:.0f})")
    for off in range(5, 0, -1):
        r = hist.iloc[i - off]
        mark = " <- WORST DAY (w)" if off == row['worst_day_offset'] else ""
        print(f"  t-{off}  {r['DlyCalDt'].date()}  DlyRet={r['DlyRet']:+.6f}  "
              f"DlyPrcVol={r['DlyPrcVol']/1e6:8.2f}M{mark}")
    w = i - int(row['worst_day_offset'])
    basevals = hist['DlyPrcVol'].iloc[max(0, w - BASE_WIN):w]
    print(f"  baseline: {basevals.notna().sum()} non-null of {len(basevals)} "
          f"days ending {hist['DlyCalDt'].iloc[w-1].date()}, "
          f"mean={basevals.mean()/1e6:.2f}M")
    manual_ratio = hist['DlyPrcVol'].iloc[w] / basevals.mean()
    print(f"  manual ratio = {hist['DlyPrcVol'].iloc[w]/1e6:.2f}M / "
          f"{basevals.mean()/1e6:.2f}M = {manual_ratio:.3f}   "
          f"script = {row['vol_ratio_worst']:.3f}   "
          f"match: {np.isclose(manual_ratio, row['vol_ratio_worst'])}   "
          f"(>= {VOL_THRESH}: {manual_ratio >= VOL_THRESH})")
    print(f"  worst-day ret = {row['worst_day_ret']:+.6f} (< 0: "
          f"{row['worst_day_ret'] < 0})")

print("\nLook-ahead statement: the ratio at worst day w uses DlyPrcVol on w")
print("and a baseline ending w-1; w <= t-1, so all inputs precede day t.")

# --------------------------------------------------------------------------
# Save
# --------------------------------------------------------------------------
out_cols = ['PERMNO', 'DlyCalDt', 'decile', 'SIG_5d', 'SIG_rank',
            'is_long_candidate', 'is_long_candidate_r2', 'worst_day_ret',
            'vol_ratio_worst', 'fwd_ret_5d', 'fwd_ret_10d', 'fwd_ret_20d']
panel[out_cols].to_parquet(output_path, index=False)
print(f"\n[OK] Saved to {output_path}  shape {panel[out_cols].shape}")

print("\n" + "=" * 80)
print("R2 SIGNAL CONSTRUCTION COMPLETE")
print("=" * 80)
