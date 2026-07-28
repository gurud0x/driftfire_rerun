import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Step 3: Signal construction — 5-day reversal signal, per pre-registration
# Sections 4 & 5. SIG_t = return over [t-5, t-1] via shift(1); forward
# returns fill at next-day open for horizons 1/3/5/10.
# ---------------------------------------------------------------------------

project_root = Path(__file__).parent.parent
crsp_path = project_root / 'data' / 'processed' / 'crsp_combined.parquet'
univ_path = project_root / 'data' / 'processed' / 'universe_membership.parquet'
output_path = project_root / 'data' / 'processed' / 'signal_panel.parquet'

print("=" * 80)
print("SIGNAL CONSTRUCTION: 5-day reversal, ranks, forward returns (1/3/5/10d)")
print("=" * 80)

import pyarrow.parquet as pq
available = pq.ParquetFile(crsp_path).schema_arrow.names
if 'DlyOpen' not in available:
    print("STOP: DlyOpen not found in crsp_combined.parquet.")
    print("Available open/close-related columns:",
          [c for c in available if 'Op' in c or 'Cl' in c or 'Prc' in c])
    raise SystemExit(1)

univ = pd.read_parquet(univ_path)
univ = univ[univ['in_universe']][['PERMNO', 'year_month', 'decile']]
print(f"\n[OK] Loaded universe membership: {univ.shape[0]:,} in-universe "
      f"stock-months, {univ['PERMNO'].nunique():,} unique PERMNOs")

# Load full daily history for every PERMNO that is EVER in universe.
# SIG and forward returns are computed on the full series first and the
# panel is filtered to in-universe days afterwards — filtering first would
# leave gaps at membership boundaries and corrupt shift/rolling windows.
df = pd.read_parquet(crsp_path,
                     columns=['PERMNO', 'DlyCalDt', 'DlyRet', 'DlyOpen',
                              'DlyClose'])
df = df[df['PERMNO'].isin(univ['PERMNO'].unique())]
print(f"[OK] Loaded daily rows for ever-in-universe PERMNOs: {df.shape[0]:,}")
print(f"  Date range: {df['DlyCalDt'].min().date()} to "
      f"{df['DlyCalDt'].max().date()}")
print(f"  Null rates: DlyRet {df['DlyRet'].isna().mean()*100:.2f}%, "
      f"DlyOpen {df['DlyOpen'].isna().mean()*100:.2f}%")

# --------------------------------------------------------------------------
# Deduplicate multi-distribution days.
# The raw WRDS daily file emits one row per distribution record, so a day
# with e.g. an ordinary + a special dividend appears twice. The rows are
# identical on every price/return field (they differ only in Dis* columns,
# verified by inspection); DlyRet already reflects all distributions.
# --------------------------------------------------------------------------
dup_mask = df.duplicated(['PERMNO', 'DlyCalDt'], keep=False)
n_dup_rows = int(dup_mask.sum())
if n_dup_rows:
    conflicting = (df[dup_mask]
                   .groupby(['PERMNO', 'DlyCalDt'])
                   .nunique().gt(1).any(axis=1).sum())
    print(f"\nMulti-distribution duplicate PERMNO-days found: "
          f"{df[dup_mask][['PERMNO','DlyCalDt']].drop_duplicates().shape[0]:,} "
          f"({n_dup_rows:,} rows)")
    print(f"  Duplicate groups DIFFERING on loaded price/return fields: "
          f"{conflicting} "
          f"{'[PASS - safe to dedupe]' if conflicting == 0 else '[FAIL - conflicting data, stopping]'}")
    if conflicting != 0:
        raise SystemExit(1)
    before = len(df)
    df = df.drop_duplicates(['PERMNO', 'DlyCalDt'], keep='first')
    print(f"  Dropped {before - len(df):,} duplicate rows "
          f"({len(df):,} remain)")

# --------------------------------------------------------------------------
# Sort and verify sort order before any shift/rolling
# --------------------------------------------------------------------------
df = df.sort_values(['PERMNO', 'DlyCalDt']).reset_index(drop=True)
sorted_ok = df.groupby('PERMNO')['DlyCalDt'].is_monotonic_increasing.all()
dup_raw = df.duplicated(['PERMNO', 'DlyCalDt']).sum()
print(f"\nSort check: dates strictly sorted within every PERMNO: {sorted_ok} "
      f"{'[PASS]' if sorted_ok else '[FAIL]'}")
print(f"Duplicate PERMNO-days after dedupe: {dup_raw} "
      f"{'[PASS]' if dup_raw == 0 else '[FAIL]'}")
if not sorted_ok or dup_raw != 0:
    raise SystemExit(1)

grp = df.groupby('PERMNO', sort=False)

# --------------------------------------------------------------------------
# SIG_5d: cumulative return over trading days [t-5, t-1], shift(1) applied
# --------------------------------------------------------------------------
print("\n" + "-" * 80)
print("SIGNAL: SIG_5d = cumprod(1+DlyRet) - 1 over [t-5, t-1]")
print("-" * 80)

lg = np.log1p(df['DlyRet'])
df['_lg'] = lg
# rolling(5) ending at t-1, aligned to day t via shift(1); a NaN return
# anywhere in the window yields NaN (min_periods defaults to window size)
df['SIG_5d'] = np.expm1(
    grp['_lg'].transform(lambda s: s.rolling(5).sum().shift(1))
)
print("Constructed as: rolling(5).sum() of log(1+DlyRet), then shift(1).")
print("Day t's own return is excluded by the shift(1) — window is [t-5, t-1].")
print(f"SIG_5d non-null rows: {df['SIG_5d'].notna().sum():,} "
      f"({df['SIG_5d'].notna().mean()*100:.1f}%)")

# --------------------------------------------------------------------------
# Forward returns from next-day open, horizons 1/3/5/10
#   fwd_ret_Nd(t) = (1 + oc_{t+1}) * prod_{s=t+2..t+N} (1 + DlyRet_s) - 1
#   where oc = DlyClose/DlyOpen - 1 (day t+1 intraday leg from the open
#   fill), and subsequent days use close-to-close total returns DlyRet.
# --------------------------------------------------------------------------
print("\n" + "-" * 80)
print("FORWARD RETURNS: entry at t+1 open")
print("-" * 80)
print("fwd_ret_Nd(t) = (DlyClose/DlyOpen at t+1) * prod(1+DlyRet, t+2..t+N) - 1")
print("Uses only days t+1 .. t+N — strictly after day t. Verified in spot")
print("check below by printing the raw inputs next to each computed value.")

df['_oc'] = df['DlyClose'] / df['DlyOpen'] - 1.0
df['_oc_next'] = grp['_oc'].shift(-1)          # day t+1 intraday leg

for n in [1, 3, 5, 10]:
    if n == 1:
        tail = 0.0
    else:
        # sum of log-returns over t+2 .. t+n  =  rolling(n-1) sum of _lg
        # evaluated at t+n, moved back to t via shift(-n)
        tail = grp['_lg'].transform(
            lambda s, n=n: s.rolling(n - 1).sum().shift(-n))
    df[f'fwd_ret_{n}d'] = (1.0 + df['_oc_next']) * np.exp(tail) - 1.0

df = df.drop(columns=['_lg', '_oc', '_oc_next'])

# --------------------------------------------------------------------------
# Filter to in-universe stock-days (point-in-time join from Step 2)
# --------------------------------------------------------------------------
print("\n" + "-" * 80)
print("APPLYING UNIVERSE FILTER (point-in-time month t-1 -> month t)")
print("-" * 80)

df['year_month'] = df['DlyCalDt'].dt.to_period('M').astype(str)
panel = df.merge(univ, on=['PERMNO', 'year_month'], how='inner')
del df
print(f"In-universe stock-day panel: {panel.shape[0]:,} rows")

# --------------------------------------------------------------------------
# Cross-sectional decile rank of SIG_5d each day (rank 1 = most negative)
# --------------------------------------------------------------------------
print("\n" + "-" * 80)
print("CROSS-SECTIONAL RANKS (rank 1 = biggest 5-day losers = long candidates)")
print("-" * 80)

pct = panel.groupby('DlyCalDt')['SIG_5d'].rank(method='first', pct=True)
panel['SIG_rank'] = np.ceil(pct * 10).clip(1, 10)
panel['is_long_candidate'] = panel['SIG_rank'] == 1
print(f"Rows with a rank (valid SIG_5d): {panel['SIG_rank'].notna().sum():,} "
      f"of {len(panel):,}")
print(f"Long candidates flagged: {panel['is_long_candidate'].sum():,}")

# --------------------------------------------------------------------------
# VALIDATION
# --------------------------------------------------------------------------
print("\n" + "-" * 80)
print("VALIDATION")
print("-" * 80)

dups = panel.duplicated(['PERMNO', 'DlyCalDt']).sum()
print(f"\nDuplicate PERMNO-day rows: {dups} "
      f"{'[PASS]' if dups == 0 else '[FAIL]'}")

print(f"Panel rows: {len(panel):,}")
print(f"Panel date range: {panel['DlyCalDt'].min().date()} to "
      f"{panel['DlyCalDt'].max().date()}")
print(f"Unique PERMNOs: {panel['PERMNO'].nunique():,}")

lc = panel[panel['is_long_candidate']].groupby('DlyCalDt').size()
all_days = panel['DlyCalDt'].drop_duplicates()
zero_days = len(all_days) - len(lc)
print(f"\nDaily long-candidate count: min={lc.min()}, max={lc.max()}, "
      f"mean={lc.mean():.1f} over {len(lc)} days")
print(f"Days with 0 long candidates: {zero_days} "
      f"{'[PASS]' if zero_days == 0 else '[WARNING - investigate]'}")

print("\nSummary statistics:")
stats_cols = ['SIG_5d', 'fwd_ret_1d', 'fwd_ret_3d', 'fwd_ret_5d', 'fwd_ret_10d']
stats = panel[stats_cols].agg(['count', 'mean', 'std']).T
stats['count'] = stats['count'].astype(int)
for col, row in stats.iterrows():
    print(f"  {col:>12}: count={row['count']:>10,}  mean={row['mean']:+.5f}  "
          f"std={row['std']:.5f}")

# --------------------------------------------------------------------------
# Spot check: 3 random stock-days, raw inputs printed next to computed values
# --------------------------------------------------------------------------
print("\n" + "-" * 80)
print("SPOT CHECK: 3 random stock-days (fixed seed 42) — verify by eye")
print("-" * 80)

full = pd.read_parquet(crsp_path,
                       columns=['PERMNO', 'DlyCalDt', 'DlyRet', 'DlyOpen',
                                'DlyClose'])
full = (full.drop_duplicates(['PERMNO', 'DlyCalDt'], keep='first')
            .sort_values(['PERMNO', 'DlyCalDt']))

rng = np.random.default_rng(42)
ok = panel[panel['SIG_5d'].notna() & panel['fwd_ret_5d'].notna()]
picks = ok.iloc[rng.choice(len(ok), 3, replace=False)]

for _, row in picks.iterrows():
    p, d = row['PERMNO'], row['DlyCalDt']
    hist = full[full['PERMNO'] == p].reset_index(drop=True)
    i = hist.index[hist['DlyCalDt'] == d][0]
    win = hist.iloc[max(0, i - 5): i + 6]
    print(f"\nPERMNO {p}, signal day t = {d.date()}")
    for j, r in win.iterrows():
        off = j - i
        tag = ('  <- t (excluded from SIG)' if off == 0 else
               ' <- SIG window' if -5 <= off <= -1 else
               ' <- fwd window' if 1 <= off <= 5 else '')
        print(f"  t{off:+d}  {r['DlyCalDt'].date()}  DlyRet={r['DlyRet']:+.6f}  "
              f"Open={r['DlyOpen']:.2f}  Close={r['DlyClose']:.2f}{tag}")
    sig_manual = np.prod(1 + hist['DlyRet'].iloc[i - 5:i].values) - 1
    oc = hist['DlyClose'].iloc[i + 1] / hist['DlyOpen'].iloc[i + 1] - 1
    fwd5_manual = (1 + oc) * np.prod(1 + hist['DlyRet'].iloc[i + 2:i + 6].values) - 1
    print(f"  manual SIG_5d      = {sig_manual:+.6f}   script = {row['SIG_5d']:+.6f}   "
          f"match: {np.isclose(sig_manual, row['SIG_5d'])}")
    print(f"  manual fwd_ret_5d  = {fwd5_manual:+.6f}   script = {row['fwd_ret_5d']:+.6f}   "
          f"match: {np.isclose(fwd5_manual, row['fwd_ret_5d'])}")

# --------------------------------------------------------------------------
# Save
# --------------------------------------------------------------------------
print("\n" + "-" * 80)
print("SAVING")
print("-" * 80)

out_cols = ['PERMNO', 'DlyCalDt', 'decile', 'SIG_5d', 'SIG_rank',
            'is_long_candidate', 'fwd_ret_1d', 'fwd_ret_3d', 'fwd_ret_5d',
            'fwd_ret_10d']
panel = panel[out_cols]
panel.to_parquet(output_path, index=False)
print(f"[OK] Saved to {output_path}")
print(f"  Shape: {panel.shape}")

print("\n" + "=" * 80)
print("SIGNAL CONSTRUCTION COMPLETE")
print("=" * 80)
