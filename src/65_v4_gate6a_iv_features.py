import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# V4 Gate 6a - Step 2, PART 2: IV-based features (7, 8, 9).
#
#   7. IV_ATM       - 30-day ATM IV, vol_surface.csv (V3 3.3-F construction, reused)
#   8. IV_30_grid, IV_60_grid, IVTermStruct - vol_surface_full_grid.csv,
#      NaN-aware ATM selection (the corrected logic validated in src/62b -
#      row existence alone is NOT sufficient, impl_volatility.notna() is
#      required)
#   9. IV_put35, IVSkew - 35-delta put IV (nearest available to the
#      originally-intended 25-delta - vol_surface.csv's put grid at 30d is
#      {-35..-65}, confirmed by direct inspection, see results/61 addendum).
#      Labeled "35D skew (nearest available to 25D)" everywhere, never
#      relabeled as 25-delta.
#
# Variance conversion (squaring) is done as an explicit, separate step
# below the extraction, not inside any extraction function.
# ---------------------------------------------------------------------------

DEV_START = pd.Timestamp('2015-01-01')
DEV_END = pd.Timestamp('2021-12-31')
CHUNKSIZE_SMALL = 5_000_000
CHUNKSIZE_BIG = 10_000_000
TARGET_PUT_DELTA = -25.0
ACTUAL_PUT_DELTA_USED = -35.0  # nearest available, confirmed via inspection

project_root = Path(__file__).parent.parent
surf_path = project_root / 'data' / 'raw' / 'optionmetrics' / 'vol_surface.csv'
staging_om = (Path.home() / 'Downloads' / 'quantdata' / 'driftfire' / 'raw' / 'optionmetrics')
grid_path = staging_om / 'vol_surface_full_grid.csv'
om_names_path = project_root / 'data' / 'raw' / 'optionmetrics' / 'om_security_names.csv'
crsp_names_path = project_root / 'data' / 'raw' / 'crsp' / 'crsp_names.parquet'
v1_path = project_root / 'data' / 'processed' / 'compression_signal_v1.parquet'
univ_path = project_root / 'data' / 'processed' / 'universe_membership.parquet'
realized_path = project_root / 'data' / 'processed' / 'v4_gate6a_features_realized.parquet'
out_path = project_root / 'data' / 'processed' / 'v4_gate6a_features_full.parquet'

print('=' * 96)
print('V4 GATE 6A - IV-BASED FEATURES (7, 8, 9)')
print('=' * 96)


def as_cusip8(s):
    return (s.astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            .str.upper().str[:8].str.zfill(8))


# ==========================================================================
# 0. Shared setup: base universe, bridge
# ==========================================================================
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
print(f"shared base universe: {len(base_v1):,} stock-days [reference: 1,733,857]")

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
print(f"bridge: {len(bridge):,} pairs")

DEV_START_S, DEV_END_S = DEV_START.strftime('%Y-%m-%d'), DEV_END.strftime('%Y-%m-%d')


# ==========================================================================
# PART A: vol_surface.csv - features 7 (ATM 30d) and 9 (35d put) together
# ==========================================================================
print('\n' + '-' * 96)
print('PART A: vol_surface.csv - feature 7 (ATM 30d) and feature 9 (35d put)')
print('-' * 96)

usecols = ['secid', 'date', 'days', 'delta', 'cp_flag', 'impl_volatility']
dtypes = {'secid': 'int32', 'date': 'str', 'days': 'int16', 'delta': 'float32',
         'cp_flag': 'category', 'impl_volatility': 'float64'}

t0 = time.time()
atm_parts, put35_parts = [], []
rows_scanned = 0
for i, ch in enumerate(pd.read_csv(surf_path, usecols=usecols, dtype=dtypes,
                                   chunksize=CHUNKSIZE_SMALL), 1):
    rows_scanned += len(ch)
    ch = ch[(ch['days'] == 30) & ch['delta'].notna() & ch['cp_flag'].isin(['C', 'P']) &
            ch['impl_volatility'].notna() & (ch['impl_volatility'] > 0)]
    if len(ch) == 0:
        continue
    ch = ch.copy()
    # ATM (feature 7): nearest to delta/100 = 0.50 in abs value, both sides
    ch['dpen_atm'] = (ch['delta'].abs() / 100.0 - 0.50).abs()
    atm = (ch.sort_values(['secid', 'date', 'cp_flag', 'dpen_atm'])
          .drop_duplicates(['secid', 'date', 'cp_flag'], keep='first'))
    atm_parts.append(atm[['secid', 'date', 'cp_flag', 'impl_volatility']])
    # 35-delta put (feature 9): puts only, nearest to delta=-25 (resolves to -35)
    puts = ch[ch['cp_flag'] == 'P'].copy()
    if len(puts):
        puts['ddist'] = (puts['delta'] - TARGET_PUT_DELTA).abs()
        p35 = (puts.sort_values(['secid', 'date', 'ddist'])
              .drop_duplicates(['secid', 'date'], keep='first'))
        put35_parts.append(p35[['secid', 'date', 'delta', 'impl_volatility']])
    if i == 1 or i % 5 == 0:
        print(f"  chunk {i:>3}: scanned {rows_scanned:>12,}  ({(time.time()-t0)/60:.1f} min)")

atm_all = pd.concat(atm_parts, ignore_index=True)
del atm_parts
put35_all = pd.concat(put35_parts, ignore_index=True)
del put35_parts
print(f"\nPart A scan complete: {rows_scanned:,} rows, {(time.time()-t0)/60:.2f} min")

piv7 = atm_all.pivot_table(index=['secid', 'date'], columns='cp_flag',
                           values='impl_volatility', aggfunc='first').reset_index()
piv7.columns.name = None
for side in ('C', 'P'):
    if side not in piv7.columns:
        piv7[side] = np.nan
piv7 = piv7.dropna(subset=['C', 'P'])
piv7['IV_ATM'] = (piv7['C'] + piv7['P']) / 2.0
print(f"Feature 7 (ATM, both sides present): {len(piv7):,} secid-dates")

put35_all = put35_all.rename(columns={'impl_volatility': 'IV_put35', 'delta': 'put35_delta_actual'})
print(f"Feature 9 (35d put, before ATM merge): {len(put35_all):,} secid-dates, "
      f"mean actual delta selected: {put35_all['put35_delta_actual'].mean():.2f} "
      f"[target was {TARGET_PUT_DELTA}, nearest available is {ACTUAL_PUT_DELTA_USED}]")


# ==========================================================================
# PART B: vol_surface_full_grid.csv - feature 8 (30d, 60d ATM, NaN-aware)
# ==========================================================================
print('\n' + '-' * 96)
print('PART B: vol_surface_full_grid.csv - feature 8 (30d/60d ATM, NaN-aware selection)')
print('-' * 96)

usecols_big = ['secid', 'date', 'days', 'delta', 'cp_flag', 'impl_volatility']
dtypes_big = {'secid': 'int32', 'date': 'str', 'days': 'int16', 'delta': 'float32',
             'cp_flag': 'category', 'impl_volatility': 'float64'}

t1 = time.time()
grid_parts = []
rows_scanned_big = 0
for i, ch in enumerate(pd.read_csv(grid_path, usecols=usecols_big, dtype=dtypes_big,
                                   chunksize=CHUNKSIZE_BIG), 1):
    rows_scanned_big += len(ch)
    ch = ch[ch['days'].isin([30, 60])]
    if len(ch) == 0:
        continue
    ch = ch[ch['secid'].astype('int64').isin(secid_whitelist)]
    if len(ch) == 0:
        continue
    ch = ch[(ch['date'] >= DEV_START_S) & (ch['date'] <= DEV_END_S)]
    if len(ch) == 0:
        continue
    # THE VALIDATED FIX: require usable IV, not just a delta label
    ch = ch[ch['delta'].notna() & ch['cp_flag'].isin(['C', 'P']) &
            ch['impl_volatility'].notna() & (ch['impl_volatility'] > 0)].copy()
    if len(ch) == 0:
        continue
    ch['dpen'] = (ch['delta'].abs() / 100.0 - 0.50).abs()
    ch = (ch.sort_values(['secid', 'date', 'days', 'cp_flag', 'dpen'])
          .drop_duplicates(['secid', 'date', 'days', 'cp_flag'], keep='first'))
    grid_parts.append(ch[['secid', 'date', 'days', 'cp_flag', 'impl_volatility']])
    if i == 1 or i % 30 == 0:
        el = time.time() - t1
        print(f"  chunk {i:>4}: scanned {rows_scanned_big:>15,}  "
              f"kept so far {sum(len(p) for p in grid_parts):,}  ({el/60:.1f} min)")

grid_atm = pd.concat(grid_parts, ignore_index=True) if grid_parts else pd.DataFrame()
del grid_parts
print(f"\nPart B scan complete: {rows_scanned_big:,} rows, {(time.time()-t1)/60:.2f} min")
print(f"Feature 8 candidate rows (both tenors, usable IV): {len(grid_atm):,}")

piv8 = grid_atm.pivot_table(index=['secid', 'date', 'days'], columns='cp_flag',
                            values='impl_volatility', aggfunc='first').reset_index()
piv8.columns.name = None
for side in ('C', 'P'):
    if side not in piv8.columns:
        piv8[side] = np.nan
piv8 = piv8.dropna(subset=['C', 'P'])
piv8['iv_atm'] = (piv8['C'] + piv8['P']) / 2.0
wide8 = piv8.pivot_table(index=['secid', 'date'], columns='days', values='iv_atm',
                         aggfunc='first').reset_index()
wide8.columns.name = None
for d_ in (30, 60):
    if d_ not in wide8.columns:
        wide8[d_] = np.nan
wide8 = wide8.rename(columns={30: 'IV_30_grid', 60: 'IV_60_grid'})
print(f"Feature 8 (30d and/or 60d present): {len(wide8):,} secid-dates; "
      f"both present: {wide8[['IV_30_grid','IV_60_grid']].notna().all(axis=1).sum():,}")


# ==========================================================================
# MERGE, VALIDATE COVERAGE AGAINST CONFIRMED REFERENCE NUMBERS
# ==========================================================================
print('\n' + '=' * 96)
print('MERGE + COVERAGE VALIDATION (must match src/62b/src/63 confirmed numbers)')
print('=' * 96)

for df_ in (piv7, put35_all, wide8):
    df_['secid'] = df_['secid'].astype('int64')

piv7_b = piv7.merge(bridge, on='secid', how='inner')
piv7_b['date_d'] = pd.to_datetime(piv7_b['date'])
put35_b = put35_all.merge(bridge, on='secid', how='inner')
put35_b['date_d'] = pd.to_datetime(put35_b['date'])
wide8_b = wide8.merge(bridge, on='secid', how='inner')
wide8_b['date_d'] = pd.to_datetime(wide8_b['date'])

panel = base.copy()
panel = panel.merge(piv7_b[['PERMNO', 'date_d', 'IV_ATM']].drop_duplicates(['PERMNO', 'date_d']),
                    on=['PERMNO', 'date_d'], how='left')
panel = panel.merge(put35_b[['PERMNO', 'date_d', 'IV_put35']].drop_duplicates(['PERMNO', 'date_d']),
                    on=['PERMNO', 'date_d'], how='left')
panel = panel.merge(wide8_b[['PERMNO', 'date_d', 'IV_30_grid', 'IV_60_grid']].drop_duplicates(
    ['PERMNO', 'date_d']), on=['PERMNO', 'date_d'], how='left')

n = len(panel)
cov7 = panel['IV_ATM'].notna().mean() * 100
cov9 = panel['IV_put35'].notna().mean() * 100
cov8_30 = panel['IV_30_grid'].notna().mean() * 100
cov8_60 = panel['IV_60_grid'].notna().mean() * 100
cov8_both = panel[['IV_30_grid', 'IV_60_grid']].notna().all(axis=1).mean() * 100

REF7, REF9, REF8 = 91.45, 91.45, 91.10
TOL = 1.0  # percentage points

print(f"\nFeature 7 (IV_ATM, vol_surface.csv 30d):  coverage {cov7:.2f}%  "
      f"[reference 91.45%]  {'OK' if abs(cov7-REF7)<=TOL else '*** MISMATCH ***'}")
print(f"Feature 9 (IV_put35, vol_surface.csv):     coverage {cov9:.2f}%  "
      f"[reference 91.45%]  {'OK' if abs(cov9-REF9)<=TOL else '*** MISMATCH ***'}")
print(f"Feature 8 (IV_30_grid, full_grid):         coverage {cov8_30:.2f}%  "
      f"[reference 91.10%]  {'OK' if abs(cov8_30-REF8)<=TOL else '*** MISMATCH ***'}")
print(f"Feature 8 (IV_60_grid, full_grid):         coverage {cov8_60:.2f}%  "
      f"[reference 91.10%]  {'OK' if abs(cov8_60-REF8)<=TOL else '*** MISMATCH ***'}")
print(f"Feature 8 (both 30d AND 60d present):      coverage {cov8_both:.2f}%")

mismatches = []
if abs(cov7 - REF7) > TOL:
    mismatches.append(f"feature 7: {cov7:.2f}% vs reference {REF7}%")
if abs(cov9 - REF9) > TOL:
    mismatches.append(f"feature 9: {cov9:.2f}% vs reference {REF9}%")
if abs(cov8_30 - REF8) > TOL or abs(cov8_60 - REF8) > TOL:
    mismatches.append(f"feature 8: 30d={cov8_30:.2f}%, 60d={cov8_60:.2f}% vs reference {REF8}%")

if mismatches:
    print("\n*** COVERAGE MISMATCH DETECTED - NOT SILENTLY ACCEPTED ***")
    for m in mismatches:
        print(f"    {m}")
    print("Halting before proceeding to squaring/regression - investigate before continuing.")
    raise SystemExit(2)
print("\n[OK] All three IV feature coverage figures match previously-confirmed reference "
      "numbers within 1.0pp tolerance.")


# ==========================================================================
# EXPLICIT VARIANCE CONVERSION STEP - separate and visible, not embedded
# in any extraction function above, per the addendum's explicit instruction.
# ==========================================================================
print('\n' + '-' * 96)
print('EXPLICIT VARIANCE CONVERSION (separate step, per addendum)')
print('-' * 96)

# Feature 7: plain square (a level, no sign to preserve)
panel['IV_ATM_var'] = panel['IV_ATM'] ** 2

# Feature 8: term structure = 30d - 60d (LEVEL difference), then SIGNED
# SQUARE (x^2 * sign(x)) to preserve direction while expressing in
# variance-scaled units - a plain square would make an upward-sloping and
# an inverted term structure indistinguishable.
term_level = panel['IV_30_grid'] - panel['IV_60_grid']
panel['IVTermStruct_level'] = term_level
panel['IVTermStruct_var'] = (term_level ** 2) * np.sign(term_level)

# Feature 9: skew = 35d-put - ATM (LEVEL difference), same signed-square treatment
skew_level = panel['IV_put35'] - panel['IV_ATM']
panel['IVSkew_level'] = skew_level
panel['IVSkew_var'] = (skew_level ** 2) * np.sign(skew_level)

print("Conversion formulas applied:")
print("  IV_ATM_var        = IV_ATM^2                              (plain square)")
print("  IVTermStruct_var   = (IV_30_grid - IV_60_grid)^2 * sign(.)  (signed square)")
print("  IVSkew_var         = (IV_put35 - IV_ATM)^2 * sign(.)        (signed square)")


# ==========================================================================
# MERGE WITH REALIZED FEATURES + COMPRESSION, VALIDATE, WRITE
# ==========================================================================
print('\n' + '-' * 96)
print('MERGE WITH REALIZED FEATURES (1,2,3,4,5,6,10) + compression_decile')
print('-' * 96)

realized = pd.read_parquet(realized_path)
realized = realized.rename(columns={'DlyCalDt': 'date_d'})
full = realized.merge(
    panel[['PERMNO', 'date_d', 'IV_ATM_var', 'IVTermStruct_var', 'IVSkew_var',
          'IVTermStruct_level', 'IVSkew_level', 'IV_ATM', 'IV_30_grid', 'IV_60_grid', 'IV_put35']],
    on=['PERMNO', 'date_d'], how='left')

iv_feat_cols = ['IV_ATM_var', 'IVTermStruct_var', 'IVSkew_var']
print("\nRequired console output - IV feature summary stats over the DEV panel:")
for c in iv_feat_cols:
    s = full[c]
    n_miss = s.isna().sum()
    print(f"  {c:<16} mean={s.mean():+.6e}  std={s.std():.6e}  "
          f"min={s.min():+.6e}  max={s.max():+.6e}  "
          f"missing={n_miss:,} ({n_miss / len(full) * 100:.2f}%)")

print(f"\nLabel check (must appear verbatim wherever feature 9 is referenced): "
      f"'35D skew (nearest available to 25D)'")

full.to_parquet(out_path, index=False)
print(f"\n[OK] wrote {out_path}  ({len(full):,} rows, {len(full.columns)} columns)")
print('=' * 96)
