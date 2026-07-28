import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
data_raw = project_root / 'data' / 'raw' / 'crsp'
data_processed = project_root / 'data' / 'processed'

data_processed.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("CRSP INGEST: Loading and merging daily returns, names, and S&P 500 membership")
print("=" * 80)

crsp_daily_path = data_raw / 'crsp_daily.parquet'
crsp_names_path = data_raw / 'crsp_names.parquet'
crsp_sp500_path = data_raw / 'crsp_sp500_members.parquet'

print(f"\nLoading from: {data_raw}")
print(f"Saving to: {data_processed}")

df_daily = pd.read_parquet(crsp_daily_path)
print(f"\n[OK] Loaded crsp_daily.parquet: {df_daily.shape}")
print(f"  Columns: {list(df_daily.columns)}")

df_names = pd.read_parquet(crsp_names_path)
print(f"\n[OK] Loaded crsp_names.parquet: {df_names.shape}")
print(f"  Columns: {list(df_names.columns)}")

df_sp500 = pd.read_parquet(crsp_sp500_path)
print(f"\n[OK] Loaded crsp_sp500_members.parquet: {df_sp500.shape}")
print(f"  Columns: {list(df_sp500.columns)}")

print("\n" + "-" * 80)
print("MERGING DATA")
print("-" * 80)

df = df_daily.copy()

df_sp500_members = df_sp500[['PERMNO', 'DlyCalDt']].drop_duplicates().assign(in_sp500=True)
df = df.merge(df_sp500_members, on=['PERMNO', 'DlyCalDt'], how='left')
df['in_sp500'] = df['in_sp500'].fillna(False).astype(bool)
print(f"After merge with S&P 500 membership: {df.shape}")

df_names_latest = df_names.sort_values('SecInfoStartDt').drop_duplicates(subset=['PERMNO'], keep='last')[['PERMNO', 'IssuerNm']]
df = df.merge(df_names_latest, on='PERMNO', how='left')
print(f"After merge with names: {df.shape}")

print("\n" + "-" * 80)
print("VALIDATION")
print("-" * 80)

valid_cols = ['DlyCalDt', 'PERMNO', 'DlyClose', 'DlyVol', 'DlyRet']
missing_cols = [c for c in valid_cols if c not in df.columns]
if missing_cols:
    print(f"WARNING: Missing columns {missing_cols}")

date_min = df['DlyCalDt'].min()
date_max = df['DlyCalDt'].max()
num_permnos = df['PERMNO'].nunique()
sp500_pct = (df['in_sp500'].sum() / len(df) * 100) if len(df) > 0 else 0

null_counts = {}
for col in valid_cols:
    if col in df.columns:
        null_pct = (df[col].isna().sum() / len(df) * 100)
        null_counts[col] = null_pct

print(f"\nDate range: {date_min} to {date_max}")
print(f"Expected: 2015-01-02 to 2025-12-31")

date_valid = (
    pd.Timestamp('2015-01-02') <= date_min and
    date_max <= pd.Timestamp('2025-12-31')
)
print(f"  [PASS]" if date_valid else "  [FAIL]")

print(f"\nUnique PERMNOs: {num_permnos}")
print(f"Expected: ~15,000–20,000 (includes historical/delisted)")
permno_valid = 10000 <= num_permnos <= 25000
print(f"  [PASS]" if permno_valid else f"  [WARNING] (got {num_permnos})")

print(f"\nS&P 500 coverage: {sp500_pct:.1f}%")
print(f"Expected: ~5–10% (500 members out of ~16k total)")
sp500_valid = 3 <= sp500_pct <= 15
print(f"  [PASS]" if sp500_valid else f"  [WARNING] (got {sp500_pct:.1f}%)")

print(f"\nNull counts (< 5% threshold):")
all_valid = True
for col in valid_cols:
    if col in df.columns:
        pct = null_counts[col]
        status = "[PASS]" if pct < 5 else "[FAIL]"
        print(f"  {col}: {pct:.2f}% {status}")
        if pct >= 5:
            all_valid = False
    else:
        print(f"  {col}: [MISSING COLUMN]")
        all_valid = False

print(f"\nDataFrame shape: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

print("\n" + "-" * 80)
print("SAVING")
print("-" * 80)

output_path = data_processed / 'crsp_combined.parquet'
df.to_parquet(output_path, index=False)
print(f"[OK] Saved to {output_path}")
print(f"  Final shape: {df.shape}")
print(f"  File size: {output_path.stat().st_size / 1e9:.2f} GB")

print("\n" + "=" * 80)
print("INGEST COMPLETE")
print("=" * 80)
