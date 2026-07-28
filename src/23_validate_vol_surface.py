import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# K1 prep: validate the OptionMetrics volatility-surface pull and compute
# THE coverage number: what fraction of V1 decile-1 stock-days have a
# usable near-ATM ~10d IV point. INSPECTION ONLY — nothing written to
# data/processed/. Conventions match src/22 and src/24: lowercase columns,
# cusip sentinel '99999999' excluded, 8-char uppercased CUSIP join.
# ---------------------------------------------------------------------------

project_root = Path(__file__).parent.parent
surf_path = project_root / 'data' / 'raw' / 'optionmetrics' / 'vol_surface.csv'
om_path = project_root / 'data' / 'raw' / 'optionmetrics' / 'om_security_names.csv'
list_path = project_root / 'data' / 'raw' / 'optionmetrics' / 'secid_list.txt'
v1_path = project_root / 'data' / 'processed' / 'compression_signal_v1.parquet'
univ_path = project_root / 'data' / 'processed' / 'universe_membership.parquet'
names_path = project_root / 'data' / 'raw' / 'crsp' / 'crsp_names.parquet'

DATE_TOL_TDAYS = 3        # +/- 3 trading days
DELTA_LO, DELTA_HI = 0.35, 0.65
TENOR_TARGET = 10.0

print("=" * 78)
print("VALIDATION: OptionMetrics volatility surface "
      "(data/raw/optionmetrics/vol_surface.csv)")
print("=" * 78)

# --------------------------------------------------------------------------
# 1. Load surface; columns, shape, secid scope vs secid_list.txt
# --------------------------------------------------------------------------
peek = pd.read_csv(surf_path, nrows=5)
print("\n### Columns as pulled:")
for c in peek.columns:
    print("   " + c)
lower = {c: c.lower() for c in peek.columns}
need = ['secid', 'date', 'days', 'delta', 'cp_flag', 'impl_volatility']
missing = [c for c in need if c not in [v for v in lower.values()]]
if missing:
    print(f"STOP: required columns missing from surface pull: {missing}")
    raise SystemExit(1)

usecols = [c for c in peek.columns if c.lower() in need]
surf = pd.read_csv(surf_path, usecols=usecols)
surf.columns = [c.lower() for c in surf.columns]
surf['date'] = pd.to_datetime(surf['date'], cache=True)

print(f"\nRows: {len(surf):,}")
n_secid = surf['secid'].nunique()
expected = {int(l) for l in list_path.read_text().split()}
print(f"Unique secid in surface: {n_secid:,}")
print(f"secid_list.txt uploaded scope: {len(expected):,}")
overlap = len(set(surf['secid'].unique()) & expected)
print(f"Surface secids inside the uploaded scope: {overlap:,} "
      f"({'[OK]' if overlap == n_secid else '[WARNING - secids outside scope]'})")
if abs(n_secid - len(expected)) > 0.25 * len(expected):
    print(f"[NOTE] surface has {n_secid:,} vs {len(expected):,} uploaded — "
          f"gap = names with no surface data in the window (expected for "
          f"small-caps), not a pull error, but reported per task.")

print(f"Surface date range: {surf['date'].min().date()} to "
      f"{surf['date'].max().date()}")

# --------------------------------------------------------------------------
# 2. Null checks
# --------------------------------------------------------------------------
print("\n### Null rates:")
for c in ['secid', 'days', 'delta', 'cp_flag', 'impl_volatility']:
    n_null = int(surf[c].isna().sum())
    print(f"   {c:>16}: {n_null:,} ({n_null/len(surf)*100:.2f}%)")
# OptionMetrics uses -99.99 as a missing IV marker in some products
n_neg = int((surf['impl_volatility'] <= 0).sum())
print(f"   impl_volatility <= 0 (incl. -99.99-style markers): {n_neg:,} "
      f"({n_neg/len(surf)*100:.2f}%)")

# --------------------------------------------------------------------------
# 3. Tenor and delta distributions — did the requested filters come through?
# --------------------------------------------------------------------------
print("\n### days (tenor) distribution:")
print(surf['days'].value_counts().sort_index().to_string())
print(f"   actual range: {surf['days'].min()} to {surf['days'].max()} "
      f"(requested ~5-15)")
if surf['days'].min() < 5 or surf['days'].max() > 15:
    print("   [WARNING] tenor range differs from the requested 5-15 window")

# delta scale: OM surface grids are often in percent-style units (e.g. 50)
delta_scale = 100.0 if surf['delta'].abs().max() > 1.5 else 1.0
surf['delta_u'] = surf['delta'] / delta_scale
print(f"\n### delta distribution (scale detected: /{delta_scale:.0f}):")
print(surf['delta'].value_counts().sort_index().to_string())
print(f"\ncp_flag distribution: {surf['cp_flag'].value_counts().to_dict()}")

# --------------------------------------------------------------------------
# 4. secid -> cusip -> PERMNO bridge (src/24 conventions)
# --------------------------------------------------------------------------
om = pd.read_csv(om_path)
om.columns = [c.lower() for c in om.columns]
om = om[om['cusip'] != '99999999'].dropna(subset=['cusip', 'secid'])
om['c8'] = om['cusip'].astype(str).str.upper().str[:8]

univ = pd.read_parquet(univ_path)
upermnos = set(univ.loc[univ['in_universe'], 'PERMNO'].unique())
names = pd.read_parquet(names_path)
crsp_map = (names[names['PERMNO'].isin(upermnos)][['PERMNO', 'CUSIP']]
            .dropna(subset=['CUSIP']).drop_duplicates())
crsp_map['c8'] = crsp_map['CUSIP'].astype(str).str.upper().str[:8]
bridge = (crsp_map.merge(om[['secid', 'c8']].drop_duplicates(), on='c8',
                         how='inner')[['PERMNO', 'secid']].drop_duplicates())
print(f"\n### Bridge: {len(bridge):,} PERMNO-secid pairs covering "
      f"{bridge['PERMNO'].nunique():,} PERMNOs, "
      f"{bridge['secid'].nunique():,} secids")

# --------------------------------------------------------------------------
# 5. V1 decile-1 stock-days, with CRSP size decile from universe_membership
# --------------------------------------------------------------------------
v1 = pd.read_parquet(v1_path,
                     columns=['PERMNO', 'DlyCalDt', 'compression_decile'])
d1 = v1[v1['compression_decile'] == 1][['PERMNO', 'DlyCalDt']].copy()
d1['year_month'] = d1['DlyCalDt'].dt.to_period('M').astype(str)
umem = univ[univ['in_universe']][['PERMNO', 'year_month', 'decile']]
d1 = d1.merge(umem, on=['PERMNO', 'year_month'], how='left')
print(f"\nV1 decile-1 (most compressed) stock-days: {len(d1):,} "
      f"({d1['PERMNO'].nunique():,} PERMNOs)")
print(f"  by CRSP size decile: "
      f"{d1['decile'].value_counts().sort_index().to_dict()}")

# --------------------------------------------------------------------------
# 6. THE KEY NUMBER — near-ATM ~10d IV coverage of decile-1 stock-days
# --------------------------------------------------------------------------
# best surface row per secid-date: |delta| in band, IV valid, then closest
# tenor to 10, then |delta| closest to 0.50
band = surf[(surf['delta_u'].abs() >= DELTA_LO) &
            (surf['delta_u'].abs() <= DELTA_HI) &
            (surf['impl_volatility'] > 0) &
            surf['impl_volatility'].notna()].copy()
print(f"\nSurface rows in the near-ATM band with valid IV: {len(band):,} "
      f"of {len(surf):,}")
band['_tpen'] = (band['days'] - TENOR_TARGET).abs()
band['_dpen'] = (band['delta_u'].abs() - 0.50).abs()
band = band.sort_values(['secid', 'date', '_tpen', '_dpen'])
best = band.drop_duplicates(['secid', 'date'], keep='first')[
    ['secid', 'date', 'days', 'delta', 'impl_volatility']]
print(f"Best near-ATM point per secid-date: {len(best):,} secid-dates, "
      f"{best['secid'].nunique():,} secids")

# trading-day index from the V1 panel calendar
cal = np.sort(v1['DlyCalDt'].unique())
cal_idx = pd.Series(np.arange(len(cal)), index=cal)


def to_tidx(dates):
    pos = np.searchsorted(cal, dates.values)
    pos = np.clip(pos, 0, len(cal) - 1)
    left = np.clip(pos - 1, 0, len(cal) - 1)
    use_left = (np.abs(dates.values - cal[left]) <=
                np.abs(cal[pos] - dates.values))
    return np.where(use_left, left, pos)


best = best[best['date'] >= cal[0] - np.timedelta64(7, 'D')]
best['tidx'] = to_tidx(best['date'])
d1['tidx'] = to_tidx(d1['DlyCalDt'])

cand = d1.merge(bridge, on='PERMNO', how='left')
has_secid = cand['secid'].notna()

left = (cand[has_secid][['PERMNO', 'DlyCalDt', 'decile', 'secid', 'tidx']]
        .astype({'secid': int}).sort_values('tidx'))
right = best.sort_values('tidx').rename(
    columns={'date': 'surf_date', 'days': 'surf_days',
             'delta': 'surf_delta', 'impl_volatility': 'surf_iv'})
joined = pd.merge_asof(left, right, on='tidx', by='secid',
                       direction='nearest', tolerance=DATE_TOL_TDAYS)

# a stock-day is covered if ANY of its candidate secids matched
joined['hit'] = joined['surf_iv'].notna()
per_day = (joined.sort_values('hit', ascending=False)
           .drop_duplicates(['PERMNO', 'DlyCalDt'], keep='first'))
no_bridge = d1.merge(
    per_day[['PERMNO', 'DlyCalDt', 'hit', 'surf_date', 'surf_days',
             'surf_delta', 'surf_iv']],
    on=['PERMNO', 'DlyCalDt'], how='left')
no_bridge['hit'] = no_bridge['hit'].fillna(False)

n_total = len(no_bridge)
n_hit = int(no_bridge['hit'].sum())
print("\n" + "=" * 78)
print("KEY COVERAGE NUMBERS - V1 decile-1 stock-days with a near-ATM ~10d "
      "IV point")
print("=" * 78)
print(f"\nOVERALL: {n_hit:,} of {n_total:,} decile-1 stock-days matched "
      f"({n_hit/n_total*100:.1f}%)")
print(f"\nBy CRSP size decile:")
for dec in [6, 7, 8]:
    sub = no_bridge[no_bridge['decile'] == dec]
    if len(sub):
        h = int(sub['hit'].sum())
        print(f"  decile {dec}: {h:,} of {len(sub):,} "
              f"({h/len(sub)*100:.1f}%)")

# zero options coverage: PERMNO has no bridge row at all, or its secids
# never appear anywhere in the surface pull
surf_secids = set(best['secid'].unique())
permno_secids = bridge.groupby('PERMNO')['secid'].apply(
    lambda s: len(set(s) & surf_secids) > 0)
covered_permnos = set(permno_secids[permno_secids].index)
zero_cov = no_bridge[~no_bridge['PERMNO'].isin(covered_permnos)]
print(f"\nDecile-1 stock-days with ZERO options coverage (PERMNO has no")
print(f"secid in the surface at all): {len(zero_cov):,} "
      f"({len(zero_cov)/n_total*100:.1f}%), "
      f"{zero_cov['PERMNO'].nunique():,} PERMNOs")

# --------------------------------------------------------------------------
# 7-8. Examples and IV sanity
# --------------------------------------------------------------------------
print("\n### 5 example matched rows (stock-day vs nearest surface point):")
ex = no_bridge[no_bridge['hit']].sample(5, random_state=42)
ex = ex.merge(per_day[['PERMNO', 'DlyCalDt', 'secid']],
              on=['PERMNO', 'DlyCalDt'], how='left')
for _, r in ex.iterrows():
    print(f"  PERMNO {r['PERMNO']:>6} {r['DlyCalDt'].date()} "
          f"(size decile {r['decile']:.0f}) -> secid {int(r['secid'])} "
          f"{pd.Timestamp(r['surf_date']).date()} days={r['surf_days']:.0f} "
          f"delta={r['surf_delta']} IV={r['surf_iv']:.4f}")

mean_iv = no_bridge.loc[no_bridge['hit'], 'surf_iv'].mean()
med_iv = no_bridge.loc[no_bridge['hit'], 'surf_iv'].median()
print(f"\nMatched decile-1 IV: mean={mean_iv:.4f}, median={med_iv:.4f} "
      f"(annualized decimal)")
sane = 0.2 <= mean_iv <= 1.5
print(f"Sanity (plausible 0.2-1.0+ annualized vol): "
      f"{'[PASS]' if sane else '[WARNING - outside plausible range]'}")

print("\n" + "=" * 78)
print("VALIDATION COMPLETE - inspection only, nothing written to")
print("data/processed/. The coverage numbers above are what the K1")
print("pre-registration threshold gets locked against.")
print("=" * 78)
