import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# CORRECTION to src/62's coverage check. src/62 measured "row with a valid
# delta exists" and got 92.05% for days in {10,30,60} identically - but a
# spot check (secid 8170, 2015-01-02) found rows with valid delta at
# days=10 whose impl_volatility was NaN: vol_surface_full_grid.csv appears
# to carry a COMPLETE synthetic grid skeleton (a row for every standard
# tenor/delta combination) independent of whether OM's surface model
# actually fit a value there. src/62's coverage figure therefore measured
# grid-skeleton existence, not usable IV. This script adds the missing
# impl_volatility.notna() filter and re-measures.
# ---------------------------------------------------------------------------

project_root = Path(__file__).parent.parent
staging_om = (Path.home() / 'Downloads' / 'quantdata' / 'driftfire' / 'raw' / 'optionmetrics')
grid_path = staging_om / 'vol_surface_full_grid.csv'
v1_path = project_root / 'data' / 'processed' / 'compression_signal_v1.parquet'
univ_path = project_root / 'data' / 'processed' / 'universe_membership.parquet'
om_names_path = project_root / 'data' / 'raw' / 'optionmetrics' / 'om_security_names.csv'
crsp_names_path = project_root / 'data' / 'raw' / 'crsp' / 'crsp_names.parquet'
out_path = project_root / 'results' / '62b_v4_feature8_iv_coverage_corrected.json'

DEV_START = pd.Timestamp('2015-01-01')
DEV_END = pd.Timestamp('2021-12-31')
CHUNKSIZE = 10_000_000


def as_cusip8(s):
    return (s.astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            .str.upper().str[:8].str.zfill(8))


print('=' * 96)
print('CORRECTED feature-8 coverage check: requires impl_volatility.notna(), not just delta')
print('=' * 96)

univ = pd.read_parquet(univ_path)
univ_in = univ[univ['in_universe']][['PERMNO', 'year_month', 'decile']].copy()
ever = set(univ_in['PERMNO'].unique())
base_v1 = pd.read_parquet(v1_path, columns=['PERMNO', 'DlyCalDt', 'compression_ratio'])
base_v1 = base_v1[(base_v1['DlyCalDt'] >= DEV_START) & (base_v1['DlyCalDt'] <= DEV_END)]
base_v1 = base_v1[base_v1['compression_ratio'].notna()][['PERMNO', 'DlyCalDt']]
base_v1['year_month'] = base_v1['DlyCalDt'].dt.to_period('M').astype(str)
base_v1 = base_v1.merge(univ_in, on=['PERMNO', 'year_month'], how='inner')
base_v1 = base_v1[['PERMNO', 'DlyCalDt']].drop_duplicates()
assert len(base_v1) == 1_733_857
base = base_v1.rename(columns={'DlyCalDt': 'date_d'})

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
secid_whitelist = set(bridge['secid'].tolist())
secid_to_permno = bridge.set_index('secid')['PERMNO'].to_dict()

DEV_START_S, DEV_END_S = DEV_START.strftime('%Y-%m-%d'), DEV_END.strftime('%Y-%m-%d')

t0 = time.time()
atm_rows = []
rows_scanned = 0
for i, ch in enumerate(pd.read_csv(
        grid_path, usecols=['secid', 'date', 'days', 'delta', 'cp_flag', 'impl_volatility'],
        dtype={'secid': 'int32', 'date': 'str', 'days': 'int16', 'delta': 'float32',
              'cp_flag': 'category', 'impl_volatility': 'float64'},
        chunksize=CHUNKSIZE), 1):
    rows_scanned += len(ch)
    ch = ch[ch['days'].isin([10, 30, 60])]
    if len(ch) == 0:
        continue
    ch = ch[ch['secid'].astype('int64').isin(secid_whitelist)]
    if len(ch) == 0:
        continue
    ch = ch[(ch['date'] >= DEV_START_S) & (ch['date'] <= DEV_END_S)]
    if len(ch) == 0:
        continue
    # THE FIX: require a genuinely usable IV value, not just a delta label
    ch = ch[ch['delta'].notna() & ch['cp_flag'].isin(['C', 'P']) &
            ch['impl_volatility'].notna() & (ch['impl_volatility'] > 0)].copy()
    if len(ch) == 0:
        continue
    ch['dpen'] = (ch['delta'].abs() / 100.0 - 0.50).abs()
    ch = (ch.sort_values(['secid', 'date', 'days', 'cp_flag', 'dpen'])
          .drop_duplicates(['secid', 'date', 'days', 'cp_flag'], keep='first'))
    atm_rows.append(ch[['secid', 'date', 'days', 'cp_flag']])
    if i == 1 or i % 30 == 0:
        el = time.time() - t0
        print(f"  chunk {i:>4}: scanned {rows_scanned:>15,}  "
              f"kept so far {sum(len(p) for p in atm_rows):,}  ({el / 60:.1f} min)")

atm = pd.concat(atm_rows, ignore_index=True) if atm_rows else pd.DataFrame()
del atm_rows
print(f"\nScan complete: {rows_scanned:,} rows, {(time.time() - t0) / 60:.2f} min")
print(f"ATM candidate rows with USABLE IV (days in 10/30/60): {len(atm):,}")

piv = atm.pivot_table(index=['secid', 'date', 'days'], columns='cp_flag',
                      values='cp_flag', aggfunc='size').reset_index()
both = piv[piv.get('C', 0).fillna(0).astype(bool) & piv.get('P', 0).fillna(0).astype(bool)]
both['PERMNO'] = both['secid'].map(secid_to_permno)
both['date_d'] = pd.to_datetime(both['date'])
both_key = both[['PERMNO', 'date_d', 'days']].drop_duplicates()

results = {}
for d in [10, 30, 60]:
    hit = both_key[both_key['days'] == d][['PERMNO', 'date_d']].drop_duplicates()
    merged = base.merge(hit.assign(_hit=True), on=['PERMNO', 'date_d'], how='left')
    pct = merged['_hit'].notna().mean() * 100
    results[d] = pct
    print(f"  days={d}: both-sides USABLE-IV ATM coverage = {pct:.2f}% of base universe "
          f"[V3 vol_surface.csv reference: 10d=4.30%, 30d=91.45%]")

import json
with open(out_path, 'w') as f:
    json.dump({'corrected_coverage_pct': results, 'rows_scanned': int(rows_scanned)}, f, indent=2)
print(f"\n[OK] wrote {out_path}")
print('=' * 96)
