import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Step 2: Universe construction — NYSE-breakpoint size deciles 6-8,
# point-in-time monthly refresh, per docs/PhaseR1_PreRegistration Section 3.
# ---------------------------------------------------------------------------

project_root = Path(__file__).parent.parent
input_path = project_root / 'data' / 'processed' / 'crsp_combined.parquet'
output_path = project_root / 'data' / 'processed' / 'universe_membership.parquet'

print("=" * 80)
print("UNIVERSE CONSTRUCTION: NYSE-breakpoint deciles 6-8, monthly point-in-time")
print("=" * 80)

need_cols = ['PERMNO', 'DlyCalDt', 'DlyClose', 'ShrOut', 'PrimaryExch',
             'SecurityType', 'SecuritySubType', 'ShareType', 'IssuerType',
             'ShrAdrFlg']

import pyarrow.parquet as pq
available = pq.ParquetFile(input_path).schema_arrow.names
missing = [c for c in need_cols if c not in available]
if missing:
    print("STOP: required columns missing from crsp_combined.parquet:")
    print("  missing:", missing)
    print("  available:", available)
    raise SystemExit(1)

df = pd.read_parquet(input_path, columns=need_cols)
print(f"\n[OK] Loaded {input_path.name}: {df.shape}")
print(f"  Date range: {df['DlyCalDt'].min()} to {df['DlyCalDt'].max()}")

# CIZ format has no legacy SHRCD/EXCHCD. Verified mapping (printed from data):
#   SHRCD 10/11 (US ordinary common) -> SecurityType='EQTY' & SecuritySubType='COM'
#       & ShareType='NS' & IssuerType='CORP'
#   (IssuerType='REIT' excludes REITs; ShareType='AD' / ShrAdrFlg='Y' excludes
#   ADRs — both flags exist in this extract, so both exclusions are applied.)
#   EXCHCD 1/2/3 (NYSE/AMEX/NASDAQ) -> PrimaryExch in ('N','A','Q')
print("\nShares outstanding field used: ShrOut (thousands of shares)")
print("Price field used: DlyClose (abs value per pre-registration)")
print("Ordinary-common filter: SecurityType='EQTY' & SecuritySubType='COM' "
      "& ShareType='NS' & IssuerType='CORP' & ShrAdrFlg='N'")
print("Exchange filter: PrimaryExch in ('N','A','Q')  [NYSE, AMEX, NASDAQ]")
print("NOTE: REIT flag (IssuerType='REIT') and ADR flag (ShrAdrFlg) both exist "
      "in this extract and are excluded, per pre-registration 'ordinary common "
      "shares only'.")

# --------------------------------------------------------------------------
# Month-end snapshot per PERMNO
# --------------------------------------------------------------------------
print("\n" + "-" * 80)
print("MONTH-END SNAPSHOTS")
print("-" * 80)

df['month'] = df['DlyCalDt'].dt.to_period('M')
df = df.sort_values(['PERMNO', 'DlyCalDt'])
snap = df.groupby(['PERMNO', 'month'], as_index=False).tail(1).copy()
del df
print(f"Month-end snapshot rows: {snap.shape[0]:,} "
      f"({snap['month'].nunique()} months x avg "
      f"{snap.groupby('month').size().mean():.0f} names)")

snap['market_cap'] = snap['DlyClose'].abs() * snap['ShrOut']

is_common = (
    (snap['SecurityType'] == 'EQTY') &
    (snap['SecuritySubType'] == 'COM') &
    (snap['ShareType'] == 'NS') &
    (snap['IssuerType'] == 'CORP') &
    (snap['ShrAdrFlg'] == 'N')
)
is_exch = snap['PrimaryExch'].isin(['N', 'A', 'Q'])
has_cap = snap['market_cap'].notna() & (snap['market_cap'] > 0)
price_ok = snap['DlyClose'].abs() >= 5.0

print(f"\nFilter counts on month-end snapshots (n={len(snap):,}):")
print(f"  ordinary common (EQTY/COM/NS/CORP, non-ADR): {is_common.sum():,}")
print(f"  exchange N/A/Q:                              {is_exch.sum():,}")
print(f"  valid market cap (>0, non-null):             {has_cap.sum():,}")
print(f"  price >= $5:                                 {price_ok.sum():,}")
print(f"  all four combined:                           {(is_common & is_exch & has_cap & price_ok).sum():,}")

# Decile assignment pool: every NYSE/AMEX/NASDAQ stock with a valid cap.
# Breakpoints: NYSE ordinary common only (Fama-French convention).
snap = snap[is_exch & has_cap].copy()
snap['eligible'] = (is_common & price_ok).loc[snap.index]
snap['is_nyse_common'] = (is_common.loc[snap.index]) & (snap['PrimaryExch'] == 'N')

# --------------------------------------------------------------------------
# NYSE breakpoints per month-end, decile assignment (1 = largest)
# --------------------------------------------------------------------------
print("\n" + "-" * 80)
print("NYSE BREAKPOINTS AND DECILE ASSIGNMENT (decile 1 = largest)")
print("-" * 80)

records = []
lookahead_log = []
months = sorted(snap['month'].unique())
for m in months:
    ms = snap[snap['month'] == m]
    nyse_caps = ms.loc[ms['is_nyse_common'], 'market_cap'].values
    if len(nyse_caps) < 100:
        print(f"  WARNING: month {m} has only {len(nyse_caps)} NYSE commons — skipped")
        continue
    bps = np.quantile(nyse_caps, np.arange(0.1, 0.91, 0.1))
    # searchsorted gives 0..9 ascending by size; convert to 1=largest
    dec_asc = np.searchsorted(bps, ms['market_cap'].values, side='right')  # 0..9
    decile = 10 - dec_asc  # 10=smallest ... 1=largest
    out = pd.DataFrame({
        'PERMNO': ms['PERMNO'].values,
        'year_month': str(m + 1),                      # applies to NEXT month
        'decile': decile.astype(int),
        'in_universe': (np.isin(decile, [6, 7, 8]) & ms['eligible'].values),
    })
    records.append(out)
    lookahead_log.append((str(m + 1), str(ms['DlyCalDt'].max().date()),
                          len(nyse_caps)))

membership = pd.concat(records, ignore_index=True)

# Drop assignment months with no trading days in the data (e.g. 2026-01)
last_data_month = str(months[-1])
membership = membership[membership['year_month'] <= last_data_month]

print("\nLook-ahead check — breakpoint data date vs. month the decile applies to:")
print(f"{'applies to':>12} | {'computed from month-end':>24} | {'# NYSE commons':>14}")
for row in lookahead_log[:3]:
    print(f"{row[0]:>12} | {row[1]:>24} | {row[2]:>14}")
print(f"{'...':>12} |")
for row in lookahead_log[-3:]:
    print(f"{row[0]:>12} | {row[1]:>24} | {row[2]:>14}")
bad = [r for r in lookahead_log if r[1] >= r[0] + '-01']
print(f"Months where breakpoint date >= start of application month: {len(bad)} "
      f"{'[PASS - no look-ahead]' if len(bad) == 0 else '[FAIL - LOOK-AHEAD DETECTED]'}")

# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
print("\n" + "-" * 80)
print("VALIDATION: MONTHLY UNIVERSE SIZE (in_universe == True)")
print("-" * 80)

counts = (membership[membership['in_universe']]
          .groupby('year_month').size().sort_index())

print(f"\n{'month':>8} {'count':>7}  flags")
trailing = counts.shift(1).rolling(12).mean()
n_bound_flags = 0
n_drift_flags = 0
for ym, n in counts.items():
    flags = []
    if not (800 <= n <= 1500):
        flags.append("OUT-OF-BOUND (800-1500)")
        n_bound_flags += 1
    tm = trailing.loc[ym]
    if pd.notna(tm) and abs(n / tm - 1) > 0.30:
        flags.append(f"WARNING >30% vs trailing 12m mean ({tm:.0f}) — AUDIT REQUIRED")
        n_drift_flags += 1
    print(f"{ym:>8} {n:>7}  {'; '.join(flags)}")

print(f"\nUniverse size summary: min={counts.min()}, max={counts.max()}, "
      f"mean={counts.mean():.1f}, months={len(counts)}")
print(f"Months outside 800-1500 sanity bound: {n_bound_flags} "
      f"{'[PASS]' if n_bound_flags == 0 else '[WARNING - see rows above]'}")
print(f"Months deviating >30% from trailing 12m mean: {n_drift_flags} "
      f"{'[PASS]' if n_drift_flags == 0 else '[WARNING - pre-registration Section 3 audit trigger]'}")

print("\n" + "-" * 80)
print("DECILE DISTRIBUTION SPOT-CHECK (deciles 6/7/8, in-universe rows)")
print("-" * 80)

sample_months = [counts.index[0], counts.index[len(counts) // 2], counts.index[-1]]
for ym in sample_months:
    sub = membership[(membership['year_month'] == ym) & membership['in_universe']]
    dist = sub['decile'].value_counts().sort_index()
    print(f"  {ym}: " + ", ".join(f"decile {d}: {c}" for d, c in dist.items())
          + f"  (total {len(sub)})")

full_dist = membership['decile'].value_counts().sort_index()
print("\nAll-months decile distribution (all N/A/Q stocks assigned):")
for d, c in full_dist.items():
    print(f"  decile {d:>2}: {c:>9,}")

# --------------------------------------------------------------------------
# Save
# --------------------------------------------------------------------------
print("\n" + "-" * 80)
print("SAVING")
print("-" * 80)

membership = membership[['PERMNO', 'year_month', 'decile', 'in_universe']]
membership.to_parquet(output_path, index=False)
print(f"[OK] Saved to {output_path}")
print(f"  Shape: {membership.shape}")
print(f"  year_month range: {membership['year_month'].min()} to "
      f"{membership['year_month'].max()}")
print(f"  in_universe rows: {membership['in_universe'].sum():,}")

print("\n" + "=" * 80)
print("UNIVERSE CONSTRUCTION COMPLETE")
print("=" * 80)
