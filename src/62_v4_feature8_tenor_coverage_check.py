import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# V4 Gate 6a addendum, item 2 (feature 8 flag resolution) - INSPECTION ONLY,
# not hypothesis-relevant data contact, authorized explicitly. Checks
# whether vol_surface_full_grid.csv (109 GB, staging, never scanned by this
# project) carries a genuine 60-calendar-day tenor point, and if so, what
# its row count / coverage looks like against the same base universe V3
# measured its own 10d/30d coverage against (91.45% / 4.30%).
#
# Columns read: secid, date, days ONLY - the minimum needed to answer "does
# a 60-day tenor exist and how often." No delta, IV, strike, or premium
# touched - this is a tenor-existence/coverage count, not a hypothesis test.
# ---------------------------------------------------------------------------

project_root = Path(__file__).parent.parent
staging_om = (Path.home() / 'Downloads' / 'quantdata' / 'driftfire' / 'raw' / 'optionmetrics')
grid_path = staging_om / 'vol_surface_full_grid.csv'
v1_path = project_root / 'data' / 'processed' / 'compression_signal_v1.parquet'
univ_path = project_root / 'data' / 'processed' / 'universe_membership.parquet'
out_path = project_root / 'results' / '62_v4_feature8_tenor_coverage.json'

DEV_START = pd.Timestamp('2015-01-01')
DEV_END = pd.Timestamp('2021-12-31')
CHUNKSIZE = 10_000_000

print('=' * 96)
print('FEATURE 8 FLAG RESOLUTION - tenor existence/coverage check, vol_surface_full_grid.csv')
print('Inspection only: secid, date, days columns. No delta/IV/strike/premium read.')
print('=' * 96)
print(f"\nsource: {grid_path} ({grid_path.stat().st_size / 1e9:.2f} GB)")

# shared base universe, identical to every prior coverage measurement this
# project has made (V3 6(b), V3-short, this session's other audits)
univ = pd.read_parquet(univ_path)
univ_in = univ[univ['in_universe']][['PERMNO', 'year_month', 'decile']].copy()
base_v1 = pd.read_parquet(v1_path, columns=['PERMNO', 'DlyCalDt', 'compression_ratio'])
base_v1 = base_v1[(base_v1['DlyCalDt'] >= DEV_START) & (base_v1['DlyCalDt'] <= DEV_END)]
base_v1 = base_v1[base_v1['compression_ratio'].notna()][['PERMNO', 'DlyCalDt']]
base_v1['year_month'] = base_v1['DlyCalDt'].dt.to_period('M').astype(str)
base_v1 = base_v1.merge(univ_in, on=['PERMNO', 'year_month'], how='inner')
base_v1 = base_v1[['PERMNO', 'DlyCalDt']].drop_duplicates()
print(f"shared base universe: {len(base_v1):,} stock-days [reference: 1,733,857]")
assert len(base_v1) == 1_733_857

t0 = time.time()
rows_scanned = 0
days_value_counts = {}
for i, ch in enumerate(pd.read_csv(grid_path, usecols=['secid', 'date', 'days'],
                                   dtype={'secid': 'int32', 'date': 'str', 'days': 'int16'},
                                   chunksize=CHUNKSIZE), 1):
    rows_scanned += len(ch)
    vc = ch['days'].value_counts()
    for k, v in vc.items():
        days_value_counts[k] = days_value_counts.get(k, 0) + int(v)
    if i == 1 or i % 30 == 0:
        el = time.time() - t0
        print(f"  chunk {i:>4}: scanned {rows_scanned:>15,}  "
              f"({rows_scanned / max(el, 1e-9) / 1e6:.2f}M rows/sec, {el / 60:.1f} min)")

el = time.time() - t0
print(f"\nFull scan complete: {rows_scanned:,} rows, {el / 60:.2f} min")
print("\nAll distinct 'days' (tenor) values found in the file, with row counts:")
for k in sorted(days_value_counts):
    print(f"  days={k:>4}: {days_value_counts[k]:>14,} rows")

has_60 = 60 in days_value_counts
print(f"\n60-day tenor present in file: {has_60}")
if not has_60:
    print("NO 60-DAY TENOR EXISTS ANYWHERE IN vol_surface_full_grid.csv.")
    print("Feature 8 (30d-60d term structure) is not constructible from any file")
    print("this project has access to. Recommend: drop feature 8, per the")
    print("pre-authorized decision rule.")

import json
out = {
    'generated_by': 'src/62_v4_feature8_tenor_coverage_check.py',
    'scope': 'inspection only - secid/date/days columns; no delta/IV/strike/premium read',
    'rows_scanned': int(rows_scanned),
    'days_value_counts': {int(k): int(v) for k, v in sorted(days_value_counts.items())},
    'has_60_day_tenor': bool(has_60),
}

if has_60:
    # measure both-sides ATM coverage at days=60 against the shared base
    # universe, same methodology as V3 3.3-F's own 10d/30d coverage table -
    # a second, targeted pass restricted to days in (10,30,60) to keep this
    # fast and re-derive the 10d/30d baseline for direct comparison.
    print("\n60-day tenor exists - running targeted ATM-coverage pass (days in 10,30,60)...")

    def as_cusip8(s):
        return (s.astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
                .str.upper().str[:8].str.zfill(8))

    om = pd.read_csv(project_root / 'data' / 'raw' / 'optionmetrics' / 'om_security_names.csv')
    om.columns = [c.lower() for c in om.columns]
    om = om.dropna(subset=['secid', 'cusip']).copy()
    om['c8'] = as_cusip8(om['cusip'])
    om = om[om['c8'].str.len() == 8][['secid', 'c8']].drop_duplicates()
    crsp_names = pd.read_parquet(project_root / 'data' / 'raw' / 'crsp' / 'crsp_names.parquet',
                                 columns=['PERMNO', 'CUSIP'])
    ever = set(univ_in['PERMNO'].unique())
    crsp_names = crsp_names[crsp_names['PERMNO'].isin(ever)].dropna(subset=['CUSIP']).copy()
    crsp_names['c8'] = as_cusip8(crsp_names['CUSIP'])
    crsp_names = crsp_names[crsp_names['c8'].str.len() == 8][['PERMNO', 'c8']].drop_duplicates()
    bridge = crsp_names.merge(om, on='c8', how='inner')[['PERMNO', 'secid']].drop_duplicates()
    bridge['secid'] = bridge['secid'].astype('int64')
    secid_whitelist = set(bridge['secid'].tolist())
    secid_to_permno = bridge.set_index('secid')['PERMNO'].to_dict()

    DEV_START_S, DEV_END_S = DEV_START.strftime('%Y-%m-%d'), DEV_END.strftime('%Y-%m-%d')
    import numpy as np
    atm_rows = []
    t1 = time.time()
    for i, ch in enumerate(pd.read_csv(
            grid_path, usecols=['secid', 'date', 'days', 'delta', 'cp_flag'],
            dtype={'secid': 'int32', 'date': 'str', 'days': 'int16', 'delta': 'float32',
                  'cp_flag': 'category'}, chunksize=CHUNKSIZE), 1):
        ch = ch[ch['days'].isin([10, 30, 60])]
        if len(ch) == 0:
            continue
        ch = ch[ch['secid'].astype('int64').isin(secid_whitelist)]
        if len(ch) == 0:
            continue
        ch = ch[(ch['date'] >= DEV_START_S) & (ch['date'] <= DEV_END_S)]
        if len(ch) == 0:
            continue
        ch = ch[ch['delta'].notna() & ch['cp_flag'].isin(['C', 'P'])].copy()
        ch['dpen'] = (ch['delta'].abs() / 100.0 - 0.50).abs()
        ch = (ch.sort_values(['secid', 'date', 'days', 'cp_flag', 'dpen'])
              .drop_duplicates(['secid', 'date', 'days', 'cp_flag'], keep='first'))
        atm_rows.append(ch[['secid', 'date', 'days', 'cp_flag']])
        if i % 50 == 0:
            print(f"  ATM pass chunk {i:>4}, kept so far {sum(len(p) for p in atm_rows):,}, "
                  f"{(time.time()-t1)/60:.1f} min")
    atm = pd.concat(atm_rows, ignore_index=True) if atm_rows else pd.DataFrame()
    print(f"ATM candidate rows (days in 10/30/60): {len(atm):,}, {(time.time()-t1)/60:.2f} min")

    piv = atm.pivot_table(index=['secid', 'date', 'days'], columns='cp_flag',
                          values='cp_flag', aggfunc='size').reset_index()
    both = piv[piv.get('C', 0).fillna(0).astype(bool) & piv.get('P', 0).fillna(0).astype(bool)]
    both['PERMNO'] = both['secid'].map(secid_to_permno)
    both['date_d'] = pd.to_datetime(both['date'])
    both_key = both[['PERMNO', 'date_d', 'days']].drop_duplicates()

    base = base_v1.rename(columns={'DlyCalDt': 'date_d'})
    cov = {}
    for d in [10, 30, 60]:
        hit = both_key[both_key['days'] == d][['PERMNO', 'date_d']].drop_duplicates()
        merged = base.merge(hit.assign(_hit=True), on=['PERMNO', 'date_d'], how='left')
        pct = merged['_hit'].notna().mean() * 100
        cov[d] = pct
        print(f"  days={d}: both-sides ATM coverage = {pct:.2f}% of base universe "
              f"[V3 3.3-F reference: 10d=4.30%, 30d=91.45%]")
    out['atm_both_sides_coverage_pct'] = cov

with open(out_path, 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f"\n[OK] wrote {out_path}")
print('=' * 96)
