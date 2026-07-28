import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Phase V1, Step 1: volume-compression score and forward realized vol,
# per docs/PhaseV1_PreRegistration_VolatilityCompression.md
# (locked at commit 4474312 before this script first ran).
#
# compression_ratio = vol20/vol60 of DlyPrcVol, both ending t-1
# (min_periods 15/45 — no substitution, no shorter fallback window).
# Targets: annualized std of DlyRet over t+1..t+N, N in {5, 10, 20}.
# ---------------------------------------------------------------------------

project_root = Path(__file__).parent.parent
crsp_path = project_root / 'data' / 'processed' / 'crsp_combined.parquet'
univ_path = project_root / 'data' / 'processed' / 'universe_membership.parquet'
output_path = project_root / 'data' / 'processed' / 'compression_signal_v1.parquet'

print("=" * 80)
print("V1 SIGNAL: volume compression (vol20/vol60) vs forward realized vol")
print("=" * 80)

univ = pd.read_parquet(univ_path)
univ = univ[univ['in_universe']][['PERMNO', 'year_month']]
print(f"\n[OK] Universe membership: {len(univ):,} in-universe stock-months, "
      f"{univ['PERMNO'].nunique():,} PERMNOs")

df = pd.read_parquet(crsp_path,
                     columns=['PERMNO', 'DlyCalDt', 'DlyRet', 'DlyPrcVol'])
df = df[df['PERMNO'].isin(univ['PERMNO'].unique())]

# same multi-distribution dedupe + halting audit as 03_signal.py
dup_mask = df.duplicated(['PERMNO', 'DlyCalDt'], keep=False)
if dup_mask.any():
    conflicting = (df[dup_mask].groupby(['PERMNO', 'DlyCalDt'])
                   .nunique().gt(1).any(axis=1).sum())
    print(f"\nMulti-distribution duplicate PERMNO-days: "
          f"{df[dup_mask][['PERMNO','DlyCalDt']].drop_duplicates().shape[0]:,}; "
          f"groups differing on loaded fields: {conflicting} "
          f"{'[PASS - safe to dedupe]' if conflicting == 0 else '[FAIL - halting]'}")
    if conflicting != 0:
        raise SystemExit(1)
    df = df.drop_duplicates(['PERMNO', 'DlyCalDt'], keep='first')

df = df.sort_values(['PERMNO', 'DlyCalDt']).reset_index(drop=True)
sorted_ok = df.groupby('PERMNO')['DlyCalDt'].is_monotonic_increasing.all()
print(f"\nDaily rows: {len(df):,}; sorted within PERMNO: {sorted_ok} "
      f"{'[PASS]' if sorted_ok else '[FAIL - halting]'}")
if not sorted_ok:
    raise SystemExit(1)

grp = df.groupby('PERMNO', sort=False)

# --------------------------------------------------------------------------
# Compression score: rolling means of DlyPrcVol ending t-1
# --------------------------------------------------------------------------
vol20 = grp['DlyPrcVol'].transform(
    lambda s: s.rolling(20, min_periods=15).mean().shift(1))
vol60 = grp['DlyPrcVol'].transform(
    lambda s: s.rolling(60, min_periods=45).mean().shift(1))
df['compression_ratio'] = np.where(vol60 > 0, vol20 / vol60, np.nan)
print(f"\ncompression_ratio non-null (full daily series): "
      f"{df['compression_ratio'].notna().sum():,} of {len(df):,} "
      f"({df['compression_ratio'].notna().mean()*100:.1f}%)")
print("Windows end at t-1 via shift(1); min_periods 15/20 and 45/60 per")
print("pre-registration (score is null otherwise; no fallback window).")

# --------------------------------------------------------------------------
# Forward realized vol: std(DlyRet) over t+1..t+N, annualized sqrt(252)
# --------------------------------------------------------------------------
for n in [5, 10, 20]:
    df[f'realized_vol_fwd_{n}d'] = grp['DlyRet'].transform(
        lambda s, n=n: s.rolling(n, min_periods=n).std().shift(-n)
    ) * np.sqrt(252)
print("\nForward realized vol: rolling(N).std() evaluated at t+N, moved to t")
print("via shift(-N) -> the window is exactly days t+1..t+N, strictly after")
print("day t. A window containing any null DlyRet yields null (min_periods=N).")
print("Sample std (ddof=1), annualized with sqrt(252).")

# --------------------------------------------------------------------------
# Filter to in-universe stock-days (point-in-time join), then rank
# --------------------------------------------------------------------------
df['year_month'] = df['DlyCalDt'].dt.to_period('M').astype(str)
panel = df.merge(univ, on=['PERMNO', 'year_month'], how='inner')
del df
print(f"\nIn-universe stock-day panel: {len(panel):,} rows")

pct = panel.groupby('DlyCalDt')['compression_ratio'].rank(method='first',
                                                          pct=True)
panel['compression_decile'] = np.ceil(pct * 10).clip(1, 10)
print("Decile 1 = lowest ratio = most compressed (ascending rank).")

# --------------------------------------------------------------------------
# VALIDATION
# --------------------------------------------------------------------------
print("\n" + "-" * 80)
print("VALIDATION")
print("-" * 80)

dups = panel.duplicated(['PERMNO', 'DlyCalDt']).sum()
print(f"\nDuplicate PERMNO-day rows: {dups} "
      f"{'[PASS]' if dups == 0 else '[FAIL - halting]'}")
if dups != 0:
    raise SystemExit(1)

print(f"Panel rows: {len(panel):,}; date range "
      f"{panel['DlyCalDt'].min().date()} to {panel['DlyCalDt'].max().date()}")
n_null = int(panel['compression_ratio'].isna().sum())
print(f"Null compression_ratio in panel (data-sufficiency rule): {n_null:,} "
      f"({n_null/len(panel)*100:.2f}%)")

scored = panel[panel['compression_ratio'].notna()].groupby('DlyCalDt').size()
print(f"Daily scored stocks: min={scored.min()}, max={scored.max()}, "
      f"mean={scored.mean():.1f} over {len(scored)} days")

for n in [5, 10, 20]:
    c = f'realized_vol_fwd_{n}d'
    s = panel[c]
    print(f"{c}: non-null {s.notna().sum():,}, mean={s.mean():.4f}, "
          f"std={s.std():.4f} (annualized decimals)")

# --------------------------------------------------------------------------
# Spot checks: 3 random scored stock-days (fixed seed 42)
# --------------------------------------------------------------------------
print("\n" + "-" * 80)
print("SPOT CHECK: 3 random scored stock-days — verify math by eye")
print("-" * 80)

full = pd.read_parquet(crsp_path,
                       columns=['PERMNO', 'DlyCalDt', 'DlyRet', 'DlyPrcVol'])
full = (full.drop_duplicates(['PERMNO', 'DlyCalDt'], keep='first')
            .sort_values(['PERMNO', 'DlyCalDt']))

rng = np.random.default_rng(42)
ok = panel[panel['compression_ratio'].notna() &
           panel['realized_vol_fwd_10d'].notna()]
picks = ok.iloc[rng.choice(len(ok), 3, replace=False)]

for _, row in picks.iterrows():
    p, d = row['PERMNO'], row['DlyCalDt']
    hist = full[full['PERMNO'] == p].reset_index(drop=True)
    i = hist.index[hist['DlyCalDt'] == d][0]
    w20 = hist['DlyPrcVol'].iloc[max(0, i - 20):i]
    w60 = hist['DlyPrcVol'].iloc[max(0, i - 60):i]
    print(f"\nPERMNO {p}, day t = {d.date()}  "
          f"(decile {row['compression_decile']:.0f})")
    v20 = [f"{x/1e6:.1f}" for x in w20.tail(20)]
    print(f"  20d DlyPrcVol window (M$, ending t-1): {' '.join(v20)}")
    print(f"  20d mean = {w20.mean()/1e6:.3f}M ({w20.notna().sum()} non-null "
          f"of {len(w20)})")
    print(f"  60d mean = {w60.mean()/1e6:.3f}M ({w60.notna().sum()} non-null "
          f"of {len(w60)})")
    manual_ratio = w20.mean() / w60.mean()
    print(f"  manual ratio = {w20.mean()/1e6:.3f}/{w60.mean()/1e6:.3f} = "
          f"{manual_ratio:.4f}   script = {row['compression_ratio']:.4f}   "
          f"match: {np.isclose(manual_ratio, row['compression_ratio'])}")
    fwd = hist['DlyRet'].iloc[i + 1:i + 11]
    print(f"  next-10 DlyRet: " +
          " ".join(f"{x:+.4f}" for x in fwd))
    manual_rv = fwd.std(ddof=1) * np.sqrt(252)
    print(f"  manual rv_fwd_10d = std*sqrt(252) = {manual_rv:.4f}   "
          f"script = {row['realized_vol_fwd_10d']:.4f}   "
          f"match: {np.isclose(manual_rv, row['realized_vol_fwd_10d'])}")

print("\nLook-ahead confirmation: compression inputs end at t-1 (shift(1));")
print("realized-vol windows start at t+1 (shift(-N) of a backward window).")
print("No input to either quantity includes day t or crosses it.")

# --------------------------------------------------------------------------
# Save
# --------------------------------------------------------------------------
out_cols = ['PERMNO', 'DlyCalDt', 'compression_ratio', 'compression_decile',
            'realized_vol_fwd_5d', 'realized_vol_fwd_10d',
            'realized_vol_fwd_20d']
panel[out_cols].to_parquet(output_path, index=False)
print(f"\n[OK] Saved to {output_path}  shape {panel[out_cols].shape}")

print("\n" + "=" * 80)
print("STOP — V1 SIGNAL STEP COMPLETE. Review the validation above before")
print("running src/11_compression_gate_v1.py.")
print("=" * 80)
