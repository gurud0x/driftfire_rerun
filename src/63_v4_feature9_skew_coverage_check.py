import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# V4 Gate 6a addendum, item 6 (feature 9 flag resolution) - INSPECTION ONLY,
# authorized explicitly. Checks what fraction of base-universe stock-days
# have a usable 25-delta PUT quote at the 30-day tenor in the ALREADY-
# PULLED vol_surface.csv (no new pull needed - this file is local, not
# staging). Nearest-delta selection, same methodology as V3 3.3's ATM
# selection, retargeted from delta=-50 to delta=-25.
# ---------------------------------------------------------------------------

project_root = Path(__file__).parent.parent
surf_path = project_root / 'data' / 'raw' / 'optionmetrics' / 'vol_surface.csv'
om_names_path = project_root / 'data' / 'raw' / 'optionmetrics' / 'om_security_names.csv'
crsp_names_path = project_root / 'data' / 'raw' / 'crsp' / 'crsp_names.parquet'
v1_path = project_root / 'data' / 'processed' / 'compression_signal_v1.parquet'
univ_path = project_root / 'data' / 'processed' / 'universe_membership.parquet'
out_path = project_root / 'results' / '63_v4_feature9_skew_coverage.json'

DEV_START = pd.Timestamp('2015-01-01')
DEV_END = pd.Timestamp('2021-12-31')
CHUNKSIZE = 5_000_000
TARGET_DELTA = -25.0  # OM's delta*100 scale, matching V3's own dpen convention
DELTA_TOLERANCE_REPORT = [1.0, 2.5, 5.0]  # report coverage at a few tolerance bands

print('=' * 96)
print('FEATURE 9 FLAG RESOLUTION - 25-delta put coverage at 30-day tenor, vol_surface.csv')
print('=' * 96)
print(f"\nsource: {surf_path} ({surf_path.stat().st_size / 1e9:.2f} GB)")


def as_cusip8(s):
    return (s.astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            .str.upper().str[:8].str.zfill(8))


univ = pd.read_parquet(univ_path)
univ_in = univ[univ['in_universe']][['PERMNO', 'year_month', 'decile']].copy()
ever = set(univ_in['PERMNO'].unique())
base_v1 = pd.read_parquet(v1_path, columns=['PERMNO', 'DlyCalDt', 'compression_ratio'])
base_v1 = base_v1[(base_v1['DlyCalDt'] >= DEV_START) & (base_v1['DlyCalDt'] <= DEV_END)]
base_v1 = base_v1[base_v1['compression_ratio'].notna()][['PERMNO', 'DlyCalDt']]
base_v1['year_month'] = base_v1['DlyCalDt'].dt.to_period('M').astype(str)
base_v1 = base_v1.merge(univ_in, on=['PERMNO', 'year_month'], how='inner')
base_v1 = base_v1[['PERMNO', 'DlyCalDt']].drop_duplicates()
print(f"shared base universe: {len(base_v1):,} stock-days [reference: 1,733,857]")
assert len(base_v1) == 1_733_857

om = pd.read_csv(om_names_path)
om.columns = [c.lower() for c in om.columns]
om = om.dropna(subset=['secid', 'cusip']).copy()
om['c8'] = as_cusip8(om['cusip'])
om = om[om['c8'].str.len() == 8][['secid', 'c8']].drop_duplicates()
crsp_names = pd.read_parquet(crsp_names_path, columns=['PERMNO', 'CUSIP'])
crsp_names = crsp_names[crsp_names['PERMNO'].isin(ever)].dropna(subset=['CUSIP']).copy()
crsp_names['c8'] = as_cusip8(crsp_names['CUSIP'])
crsp_names = crsp_names[crsp_names['c8'].str.len() == 8][['PERMNO', 'c8']].drop_duplicates()
bridge = crsp_names.merge(om, on='c8', how='inner')[['PERMNO', 'secid']].drop_duplicates()
bridge['secid'] = bridge['secid'].astype('int64')
print(f"bridge: {len(bridge):,} pairs")

usecols = ['secid', 'date', 'days', 'delta', 'cp_flag', 'impl_volatility']
dtypes = {'secid': 'int32', 'date': 'str', 'days': 'int16', 'delta': 'float32',
         'cp_flag': 'category', 'impl_volatility': 'float64'}

t0 = time.time()
put25_rows = []
rows_scanned = 0
for i, ch in enumerate(pd.read_csv(surf_path, usecols=usecols, dtype=dtypes,
                                   chunksize=CHUNKSIZE), 1):
    rows_scanned += len(ch)
    ch = ch[(ch['days'] == 30) & (ch['cp_flag'] == 'P')]
    if len(ch) == 0:
        continue
    ch = ch[ch['delta'].notna() & ch['impl_volatility'].notna() & (ch['impl_volatility'] > 0)].copy()
    if len(ch) == 0:
        continue
    ch['ddist'] = (ch['delta'] - TARGET_DELTA).abs()
    ch = (ch.sort_values(['secid', 'date', 'ddist'])
          .drop_duplicates(['secid', 'date'], keep='first'))
    put25_rows.append(ch[['secid', 'date', 'delta', 'ddist', 'impl_volatility']])
    if i == 1 or i % 5 == 0:
        print(f"  chunk {i:>3}: scanned {rows_scanned:>12,}  "
              f"kept so far {sum(len(p) for p in put25_rows):,}  "
              f"({(time.time()-t0)/60:.1f} min)")

put25 = pd.concat(put25_rows, ignore_index=True) if put25_rows else pd.DataFrame()
del put25_rows
print(f"\nScan complete: {rows_scanned:,} rows, {(time.time()-t0)/60:.2f} min")
print(f"Nearest-to-delta=-25 put candidates (30-day tenor, 1 per secid-date): {len(put25):,}")
print(f"delta distance from target: min {put25['ddist'].min():.3f}, "
      f"median {put25['ddist'].median():.3f}, p90 {put25['ddist'].quantile(0.9):.3f}, "
      f"max {put25['ddist'].max():.3f}")

put25 = put25.merge(bridge, on='secid', how='inner')
put25['date_d'] = pd.to_datetime(put25['date'])
base = base_v1.rename(columns={'DlyCalDt': 'date_d'})

results = {}
for tol in DELTA_TOLERANCE_REPORT:
    hit = put25[put25['ddist'] <= tol][['PERMNO', 'date_d']].drop_duplicates()
    merged = base.merge(hit.assign(_hit=True), on=['PERMNO', 'date_d'], how='left')
    pct = merged['_hit'].notna().mean() * 100
    results[tol] = pct
    print(f"  coverage within +/-{tol} delta points of -25: {pct:.2f}% of base universe")

# also: nearest-available regardless of tolerance (upper bound on coverage)
hit_any = put25[['PERMNO', 'date_d']].drop_duplicates()
merged_any = base.merge(hit_any.assign(_hit=True), on=['PERMNO', 'date_d'], how='left')
pct_any = merged_any['_hit'].notna().mean() * 100
print(f"  coverage with ANY put quote present at 30-day tenor (no delta tolerance): {pct_any:.2f}%")

import json
out = {
    'generated_by': 'src/63_v4_feature9_skew_coverage_check.py',
    'target_delta': TARGET_DELTA,
    'rows_scanned': int(rows_scanned),
    'candidates_found': int(len(put25)),
    'coverage_pct_by_tolerance': {str(k): v for k, v in results.items()},
    'coverage_pct_any_put_30d': pct_any,
}
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f"\n[OK] wrote {out_path}")
print('=' * 96)
