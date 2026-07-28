import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# K1 prep: validate the CORRECTED, filtered volatility surface (output of
# src/27) and compute THE coverage number over the full 2015-2025 range:
# what fraction of V1 decile-1 stock-days have a usable near-ATM 30-day IV
# point. INSPECTION ONLY - nothing written to data/processed/.
#
# Conventions follow src/23 (and src/22, src/24): lowercase columns, cusip
# sentinel '99999999' excluded, 8-char uppercased CUSIP join, trading-day
# merge_asof with +/- 3 day tolerance.
#
# TWO DELIBERATE DIFFERENCES FROM src/23:
#  1. Tenor is 30 days, not 10 - per the decay check (src/25: signal is as
#     strong at 30d, t -8.18) and the 30d horse race (src/26).
#  2. The surface is read in CHUNKS and reduced to one best near-ATM point
#     per secid-date-side as it goes. src/23 loaded a 1.86 GB file whole;
#     this file is several GB and would risk a MemoryError loaded flat.
#     pandas-only per this task's requirements (src/23 used numpy).
# ---------------------------------------------------------------------------

project_root = Path(__file__).parent.parent
surf_path = project_root / 'data' / 'raw' / 'optionmetrics' / 'vol_surface.csv'
om_path = project_root / 'data' / 'raw' / 'optionmetrics' / 'om_security_names.csv'
v1_path = project_root / 'data' / 'processed' / 'compression_signal_v1.parquet'
univ_path = project_root / 'data' / 'processed' / 'universe_membership.parquet'
names_path = project_root / 'data' / 'raw' / 'crsp' / 'crsp_names.parquet'

DATE_TOL_TDAYS = 3
DELTA_LO, DELTA_HI = 0.35, 0.65
TENOR = 30
CHUNKSIZE = 5_000_000

print("=" * 78)
print("VALIDATION: corrected filtered volatility surface, full 2015-2025")
print("=" * 78)

peek = pd.read_csv(surf_path, nrows=5)
print("\n### Columns:")
for c in peek.columns:
    print("   " + c)
need = ['secid', 'date', 'days', 'delta', 'cp_flag', 'impl_volatility']
have = [c.lower() for c in peek.columns]
missing = [c for c in need if c not in have]
if missing:
    print(f"STOP: required columns missing: {missing}")
    raise SystemExit(1)
usecols = [c for c in peek.columns if c.lower() in need]

DT = {'secid': 'int32', 'date': 'str', 'days': 'int16', 'delta': 'float32',
      'cp_flag': 'category', 'impl_volatility': 'float64'}

print(f"\nReading in chunks of {CHUNKSIZE:,} rows, reducing to the best")
print(f"near-ATM point per secid-date-side at the {TENOR}-day tenor as we go.")

rows = 0
secids = set()
dmin = dmax = None
days_ct = {}
delta_ct = {}
cp_ct = {}
iv_null = 0
iv_le0 = 0
parts = []

reader = pd.read_csv(surf_path, usecols=usecols, dtype=DT,
                     chunksize=CHUNKSIZE)
for i, ch in enumerate(reader, 1):
    ch.columns = [c.lower() for c in ch.columns]
    rows += len(ch)
    secids.update(ch['secid'].unique().tolist())
    cmin, cmax = ch['date'].min(), ch['date'].max()
    dmin = cmin if dmin is None else min(dmin, cmin)
    dmax = cmax if dmax is None else max(dmax, cmax)
    for k, v in ch['days'].value_counts().items():
        days_ct[int(k)] = days_ct.get(int(k), 0) + int(v)
    for k, v in ch['delta'].value_counts().items():
        delta_ct[int(k)] = delta_ct.get(int(k), 0) + int(v)
    for k, v in ch['cp_flag'].value_counts().items():
        cp_ct[str(k)] = cp_ct.get(str(k), 0) + int(v)
    iv_null += int(ch['impl_volatility'].isna().sum())
    iv_le0 += int((ch['impl_volatility'] <= 0).sum())

    # delta scale detection (raw grid units vs true decimal delta)
    scale = 100.0 if ch['delta'].abs().max() > 1.5 else 1.0
    du = ch['delta'].abs() / scale
    sel = ch[(ch['days'] == TENOR) & (du >= DELTA_LO) & (du <= DELTA_HI) &
             ch['impl_volatility'].notna() & (ch['impl_volatility'] > 0)].copy()
    if len(sel):
        sel['dpen'] = (sel['delta'].abs() / scale - 0.50).abs()
        sel = (sel.sort_values(['secid', 'date', 'cp_flag', 'dpen'])
               .drop_duplicates(['secid', 'date', 'cp_flag'], keep='first'))
        parts.append(sel[['secid', 'date', 'cp_flag', 'delta',
                          'impl_volatility', 'dpen']])
    if i % 5 == 0 or i == 1:
        print(f"  chunk {i:>3}: {rows:>14,} rows scanned")

print(f"\nRows: {rows:,}")
print(f"Unique secid: {len(secids):,}")
print(f"Date range: {dmin} to {dmax}")

print(f"\n### days distribution (expect only {{10, 30}}):")
for k in sorted(days_ct):
    print(f"   days {k:>3}: {days_ct[k]:>14,} "
          f"({days_ct[k]/rows*100:5.1f}%)")
ok_days = set(days_ct) <= {10, 30}
print(f"   restricted to {{10, 30}}: {ok_days} "
      f"{'[PASS]' if ok_days else '[FAIL]'}")

print(f"\n### delta distribution (raw grid units; /100 = true delta):")
for k in sorted(delta_ct):
    print(f"   delta {k:>4} ({k/100:+.2f}): {delta_ct[k]:>14,}")
bad = [k for k in delta_ct if not (DELTA_LO <= abs(k) / 100 <= DELTA_HI)]
print(f"   all within near-ATM band {DELTA_LO}-{DELTA_HI}: {not bad} "
      f"{'[PASS]' if not bad else '[FAIL: ' + str(bad) + ']'}")
print(f"\ncp_flag distribution: {cp_ct}")

print(f"\n### impl_volatility data sufficiency:")
print(f"   null: {iv_null:,} ({iv_null/rows*100:.2f}%)")
print(f"   <= 0: {iv_le0:,} ({iv_le0/rows*100:.2f}%)")

# --------------------------------------------------------------------------
# Global reduce: best near-ATM point per secid-date-side, then per secid-date
# --------------------------------------------------------------------------
side = pd.concat(parts, ignore_index=True)
del parts
side = (side.sort_values(['secid', 'date', 'cp_flag', 'dpen'])
        .drop_duplicates(['secid', 'date', 'cp_flag'], keep='first'))
best = (side.sort_values(['secid', 'date', 'dpen'])
        .drop_duplicates(['secid', 'date'], keep='first')
        [['secid', 'date', 'delta', 'impl_volatility']])
best['date'] = pd.to_datetime(best['date'], cache=True)
print(f"\nBest near-ATM {TENOR}d point per secid-date: {len(best):,} "
      f"secid-dates, {best['secid'].nunique():,} secids")

# straddle diagnostic: both legs quoted at the same secid-date
legs = side.groupby(['secid', 'date'], observed=True)['cp_flag'].nunique()
both = int((legs >= 2).sum())
print(f"secid-dates with BOTH call and put near-ATM points (straddle-ready): "
      f"{both:,} of {len(legs):,} ({both/len(legs)*100:.1f}%)")

# --------------------------------------------------------------------------
# Bridge: secid -> cusip -> PERMNO (src/23 conventions)
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
# V1 decile-1 stock-days, FULL 2015-2025 range
# --------------------------------------------------------------------------
v1 = pd.read_parquet(v1_path,
                     columns=['PERMNO', 'DlyCalDt', 'compression_decile'])
d1 = v1[v1['compression_decile'] == 1][['PERMNO', 'DlyCalDt']].copy()
d1['year_month'] = d1['DlyCalDt'].dt.to_period('M').astype(str)
umem = univ[univ['in_universe']][['PERMNO', 'year_month', 'decile']]
d1 = d1.merge(umem, on=['PERMNO', 'year_month'], how='left')
print(f"\nV1 decile-1 stock-days (full range): {len(d1):,} "
      f"({d1['PERMNO'].nunique():,} PERMNOs), "
      f"{d1['DlyCalDt'].min().date()} to {d1['DlyCalDt'].max().date()}")
print(f"  by CRSP size decile: "
      f"{d1['decile'].value_counts().sort_index().to_dict()}")

# --------------------------------------------------------------------------
# THE KEY NUMBER: near-ATM 30d IV coverage, matched within +/- 3 trading days
# --------------------------------------------------------------------------
cal = pd.Index(pd.Series(v1['DlyCalDt'].unique()).sort_values())
best = best[best['date'] >= cal[0] - pd.Timedelta(days=7)].copy()
best['tidx'] = cal.get_indexer(best['date'], method='nearest')
d1['tidx'] = cal.get_indexer(d1['DlyCalDt'], method='nearest')

cand = d1.merge(bridge, on='PERMNO', how='left')
left = (cand[cand['secid'].notna()]
        [['PERMNO', 'DlyCalDt', 'decile', 'secid', 'tidx']]
        .astype({'secid': 'int64'}).sort_values('tidx'))
right = (best.astype({'secid': 'int64'}).sort_values('tidx')
         .rename(columns={'date': 'surf_date', 'delta': 'surf_delta',
                          'impl_volatility': 'surf_iv'}))
joined = pd.merge_asof(left, right, on='tidx', by='secid',
                       direction='nearest', tolerance=DATE_TOL_TDAYS)
joined['hit'] = joined['surf_iv'].notna()
per_day = (joined.sort_values('hit', ascending=False)
           .drop_duplicates(['PERMNO', 'DlyCalDt'], keep='first'))
res = d1.merge(per_day[['PERMNO', 'DlyCalDt', 'hit', 'surf_date',
                        'surf_delta', 'surf_iv', 'secid']],
               on=['PERMNO', 'DlyCalDt'], how='left')
res['hit'] = res['hit'].fillna(False)

n_total, n_hit = len(res), int(res['hit'].sum())
print("\n" + "=" * 78)
print(f"KEY COVERAGE - V1 decile-1 stock-days with a near-ATM {TENOR}d IV "
      f"point")
print("=" * 78)
print(f"\nOVERALL (2015-2025): {n_hit:,} of {n_total:,} matched "
      f"({n_hit/n_total*100:.1f}%)")

print(f"\nBy year:")
res['year'] = res['DlyCalDt'].dt.year
print(f"  {'year':>6} {'matched':>10} {'total':>10} {'coverage':>10}")
for y, sub in res.groupby('year'):
    h = int(sub['hit'].sum())
    print(f"  {y:>6} {h:>10,} {len(sub):>10,} {h/len(sub)*100:>9.1f}%")

print(f"\nBy CRSP size decile:")
for dec in [6, 7, 8]:
    sub = res[res['decile'] == dec]
    if len(sub):
        h = int(sub['hit'].sum())
        print(f"  decile {dec}: {h:,} of {len(sub):,} "
              f"({h/len(sub)*100:.1f}%)")

surf_secids = set(best['secid'].unique())
pmap = bridge.groupby('PERMNO')['secid'].apply(
    lambda s: len(set(s) & surf_secids) > 0)
covered = set(pmap[pmap].index)
zero_cov = res[~res['PERMNO'].isin(covered)]
print(f"\nDecile-1 stock-days with ZERO options coverage (PERMNO has no")
print(f"secid anywhere in the surface): {len(zero_cov):,} "
      f"({len(zero_cov)/n_total*100:.1f}%), "
      f"{zero_cov['PERMNO'].nunique():,} PERMNOs")

# --------------------------------------------------------------------------
# Examples and IV sanity
# --------------------------------------------------------------------------
print(f"\n### 5 example matched rows:")
ex = res[res['hit']].sample(5, random_state=42)
for _, r in ex.iterrows():
    print(f"  PERMNO {r['PERMNO']:>6} {r['DlyCalDt'].date()} "
          f"(size decile {r['decile']:.0f}) -> secid {int(r['secid'])} "
          f"{pd.Timestamp(r['surf_date']).date()} days={TENOR} "
          f"delta={r['surf_delta']:.0f} IV={r['surf_iv']:.4f}")

mean_iv = res.loc[res['hit'], 'surf_iv'].mean()
med_iv = res.loc[res['hit'], 'surf_iv'].median()
print(f"\nMatched decile-1 IV: mean={mean_iv:.4f}, median={med_iv:.4f} "
      f"(annualized decimal)")
print(f"Sanity (plausible 0.2-1.0+): "
      f"{'[PASS]' if 0.2 <= mean_iv <= 1.5 else '[WARNING - outside range]'}")

print("\n" + "=" * 78)
print("VALIDATION COMPLETE - inspection only, nothing written to")
print("data/processed/. These are the numbers the K1 pre-registration")
print("threshold gets locked against.")
print("=" * 78)
