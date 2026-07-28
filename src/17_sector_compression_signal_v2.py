import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Phase V2, Step 1: sector-relative compression signal, per
# docs/PhaseV2_PreRegistration_SectorRelativeCompression.md (commit 3fdac95).
#
# sector_relative_compression = own compression_ratio (reused from V1,
# NOT recomputed) minus the same-day LEAVE-ONE-OUT median across other
# in-universe stocks in the same gsector. Sector-days with fewer than 20
# valid stocks are excluded. Static-GICS limitation accepted and disclosed
# in the pre-registration.
# ---------------------------------------------------------------------------

project_root = Path(__file__).parent.parent
v1_path = project_root / 'data' / 'processed' / 'compression_signal_v1.parquet'
link_path = project_root / 'data' / 'raw' / 'compustat' / 'ccm_link_gics.csv'
names_path = project_root / 'data' / 'raw' / 'compustat' / 'compustat_gics_names.csv'
output_path = project_root / 'data' / 'processed' / 'sector_compression_signal_v2.parquet'

MIN_SECTOR_DAY = 20

print("=" * 78)
print("V2 SIGNAL: sector-relative volume compression (GICS via CCM link)")
print("=" * 78)

panel = pd.read_parquet(v1_path)
print(f"\n[OK] V1 signal panel (reused, not recomputed): {len(panel):,} rows, "
      f"{panel['DlyCalDt'].min().date()} to {panel['DlyCalDt'].max().date()}")

# --------------------------------------------------------------------------
# CCM link file: confirm casing and linkenddt representation before parsing
# --------------------------------------------------------------------------
link = pd.read_csv(link_path)
print(f"\n[OK] CCM link file: {len(link):,} rows")
print(f"  Columns as pulled: {list(link.columns)}")
link.columns = [c.lower() for c in link.columns]
print("  (normalized to lowercase)")

print(f"\n  5 raw linkenddt example values: "
      f"{link['linkenddt'].head(5).tolist()}")
nondate = link.loc[pd.to_datetime(link['linkenddt'], errors='coerce').isna(),
                   'linkenddt']
print(f"  Non-date linkenddt values and counts: "
      f"{nondate.value_counts(dropna=False).to_dict()}")

# --------------------------------------------------------------------------
# Filter to research-quality links with a usable sector
# --------------------------------------------------------------------------
n0 = len(link)
link = link[link['linktype'].isin(['LC', 'LU'])]
n1 = len(link)
link = link[link['linkprim'].isin(['P', 'C'])]
n2 = len(link)
n_null_gics = int(link['gsector'].isna().sum())
link = link[link['gsector'].notna()]
print(f"\n  Link filter trail: {n0:,} rows -> linktype LC/LU {n1:,} -> "
      f"linkprim P/C {n2:,} -> non-null gsector {len(link):,} "
      f"(dropped {n_null_gics:,} null-GICS link rows)")

link['linkdt'] = pd.to_datetime(link['linkdt'])
# 'E' marks a still-active link: no upper bound (NOT null-and-drop)
link['linkend'] = pd.to_datetime(
    link['linkenddt'].replace('E', pd.NaT), errors='coerce')
link['linkend'] = link['linkend'].fillna(pd.Timestamp('2262-01-01'))
link['gsector'] = link['gsector'].astype(int)

# --------------------------------------------------------------------------
# Join: PERMNO = lpermno, DlyCalDt within [linkdt, linkend]
# --------------------------------------------------------------------------
link['lpermno'] = link['lpermno'].astype(int)
lk = link[['lpermno', 'gvkey', 'gsector', 'linkdt', 'linkend', 'linkprim']]
m = panel.merge(lk, left_on='PERMNO', right_on='lpermno', how='left')
in_window = (m['DlyCalDt'] >= m['linkdt']) & (m['DlyCalDt'] <= m['linkend'])
m = m[in_window | m['lpermno'].isna()]
m = m.drop(columns=['lpermno', 'linkdt', 'linkend'])

# tiebreak: prefer linkprim P over C where a PERMNO-day matched both
m['_prim_rank'] = np.where(m['linkprim'] == 'P', 0, 1)
m = m.sort_values(['PERMNO', 'DlyCalDt', '_prim_rank', 'gvkey'])
dup_key = m.duplicated(['PERMNO', 'DlyCalDt'], keep=False)
n_multi = int(dup_key.sum() - m.duplicated(['PERMNO', 'DlyCalDt']).sum())
m = m.drop_duplicates(['PERMNO', 'DlyCalDt'], keep='first')
print(f"\n  PERMNO-days needing the P-over-C tiebreak (or gvkey tiebreak): "
      f"{n_multi:,}")

matched = m['gsector'].notna()
print(f"\n### Effective coverage (computed, per pre-registration):")
print(f"  V1 panel stock-days: {len(m):,}")
print(f"  with a valid in-window sector match: {int(matched.sum()):,} "
      f"({matched.mean()*100:.2f}%)")
print(f"  excluded (no valid sector; dropped entirely, no synthetic "
      f"sector): {int((~matched).sum()):,} ({(~matched).mean()*100:.2f}%)")
m = m[matched].copy()
m['gsector'] = m['gsector'].astype(int)
m['gvkey'] = m['gvkey'].astype(int)

# --------------------------------------------------------------------------
# Cross-check gsector vs the names pull (link file is authoritative)
# --------------------------------------------------------------------------
names = pd.read_csv(names_path, usecols=['gvkey', 'gsector']).dropna()
names = names.drop_duplicates('gvkey').set_index('gvkey')['gsector'].astype(int)
used_gvkeys = m[['gvkey', 'gsector']].drop_duplicates('gvkey')
used_gvkeys = used_gvkeys[used_gvkeys['gvkey'].isin(names.index)]
dis = used_gvkeys[used_gvkeys['gsector'] !=
                  names.loc[used_gvkeys['gvkey']].values]
print(f"\n### Cross-check vs compustat_gics_names.csv "
      f"({len(used_gvkeys):,} shared gvkeys in use):")
print(f"  Disagreements (link file wins, logged not resolved): {len(dis):,}")
if len(dis):
    print(f"  {'gvkey':>8} {'link gsector':>13} {'names gsector':>14}")
    for _, r in dis.iterrows():
        print(f"  {r['gvkey']:>8} {r['gsector']:>13} "
              f"{names.loc[r['gvkey']]:>14}")

# --------------------------------------------------------------------------
# Sector-relative compression: own ratio minus LEAVE-ONE-OUT sector median
# --------------------------------------------------------------------------
valid = m['compression_ratio'].notna()
print(f"\nStock-days with valid compression_ratio and sector: "
      f"{int(valid.sum()):,} of {len(m):,}")

grp_sizes = (m[valid].groupby(['DlyCalDt', 'gsector'])['compression_ratio']
             .transform('size'))
big_enough = pd.Series(False, index=m.index)
big_enough.loc[grp_sizes.index] = grp_sizes >= MIN_SECTOR_DAY

all_groups = m[valid].groupby(['DlyCalDt', 'gsector']).size()
small_groups = all_groups[all_groups < MIN_SECTOR_DAY]
print(f"Sector-days total: {len(all_groups):,}; excluded for "
      f"< {MIN_SECTOR_DAY} valid stocks: {len(small_groups):,} "
      f"({int(small_groups.sum()):,} stock-days dropped by the rule)")
if len(small_groups):
    per_sector = small_groups.groupby('gsector').size()
    print(f"  Small sector-days by sector: {per_sector.to_dict()}")


def loo_median(vals):
    # leave-one-out median: for each element, the median of the others
    x = np.sort(vals)
    n = len(x)
    order = np.argsort(vals, kind='mergesort')
    r = np.empty(n, dtype=int)
    r[order] = np.arange(n)
    mrem = n - 1
    if mrem % 2 == 1:
        k = (mrem - 1) // 2
        return np.where(k < r, x[k], x[k + 1])
    k1, k2 = mrem // 2 - 1, mrem // 2
    v1 = np.where(k1 < r, x[k1], x[k1 + 1])
    v2 = np.where(k2 < r, x[k2], x[k2 + 1])
    return (v1 + v2) / 2.0


use = valid & big_enough
loo = (m.loc[use].groupby(['DlyCalDt', 'gsector'])['compression_ratio']
       .transform(lambda s: loo_median(s.to_numpy())))
m['sector_relative_compression'] = np.nan
m.loc[use, 'sector_relative_compression'] = (
    m.loc[use, 'compression_ratio'] - loo)

# --------------------------------------------------------------------------
# Cross-sectional decile rank within the FULL universe each day
# --------------------------------------------------------------------------
pct = (m.groupby('DlyCalDt')['sector_relative_compression']
       .rank(method='first', pct=True))
m['sector_rel_decile'] = np.ceil(pct * 10).clip(1, 10)
print(f"\nDecile 1 = most compressed relative to own sector "
      f"(lowest sector_relative_compression).")

# --------------------------------------------------------------------------
# VALIDATION
# --------------------------------------------------------------------------
print("\n" + "-" * 78)
print("VALIDATION")
print("-" * 78)

dups = m.duplicated(['PERMNO', 'DlyCalDt']).sum()
print(f"\nDuplicate PERMNO-day rows: {int(dups)} "
      f"{'[PASS]' if dups == 0 else '[FAIL - halting]'}")
if dups:
    raise SystemExit(1)

scored = m[m['sector_relative_compression'].notna()]
print(f"Rows: {len(m):,}; scored (valid sector-relative value): "
      f"{len(scored):,}")
print(f"Date range: {m['DlyCalDt'].min().date()} to "
      f"{m['DlyCalDt'].max().date()}")
daily = scored.groupby('DlyCalDt').size()
print(f"Daily scored stocks: min={daily.min()}, max={daily.max()}, "
      f"mean={daily.mean():.1f} over {len(daily)} days")
print(f"\nStock-days per sector (scored):")
for sec, cnt in scored['gsector'].value_counts().sort_index().items():
    print(f"  gsector {sec:>2}: {cnt:>9,}")
s = scored['sector_relative_compression']
print(f"\nsector_relative_compression: mean={s.mean():+.4f}, "
      f"std={s.std():.4f}, min={s.min():+.3f}, max={s.max():+.3f}")

# --------------------------------------------------------------------------
# Spot checks: verify the leave-one-out median arithmetic by hand
# --------------------------------------------------------------------------
print("\n" + "-" * 78)
print("SPOT CHECK: 3 random scored stock-days (fixed seed 42)")
print("-" * 78)
rng = np.random.default_rng(42)
picks = scored.iloc[rng.choice(len(scored), 3, replace=False)]
for _, row in picks.iterrows():
    p, d, sec = row['PERMNO'], row['DlyCalDt'], row['gsector']
    peers = m[(m['DlyCalDt'] == d) & (m['gsector'] == sec) &
              m['compression_ratio'].notna()]
    others = peers[peers['PERMNO'] != p]['compression_ratio']
    manual = row['compression_ratio'] - np.median(others)
    print(f"\nPERMNO {p}, {d.date()}, gsector {sec} "
          f"({len(peers)} valid stocks in sector-day)")
    print(f"  own ratio = {row['compression_ratio']:.4f}, "
          f"median of {len(others)} others = {np.median(others):.4f}")
    print(f"  manual sector_relative = {manual:+.4f}   "
          f"script = {row['sector_relative_compression']:+.4f}   "
          f"match: {np.isclose(manual, row['sector_relative_compression'])}")

print("\nLook-ahead statement: compression_ratio inputs end at t-1 (V1,")
print("unchanged); the sector median uses only same-day values of that")
print("already-lagged ratio. Forward targets are V1's, windows t+1..t+N.")
print("Static-GICS limitation applies as disclosed in the pre-registration.")

# --------------------------------------------------------------------------
# Save
# --------------------------------------------------------------------------
out_cols = ['PERMNO', 'DlyCalDt', 'gvkey', 'gsector', 'compression_ratio',
            'sector_relative_compression', 'sector_rel_decile',
            'realized_vol_fwd_5d', 'realized_vol_fwd_10d',
            'realized_vol_fwd_20d']
m[out_cols].to_parquet(output_path, index=False)
print(f"\n[OK] Saved to {output_path}  shape {m[out_cols].shape}")

print("\n" + "=" * 78)
print("STOP - V2 SIGNAL STEP COMPLETE. Review validation above before the")
print("gate script runs.")
print("=" * 78)
