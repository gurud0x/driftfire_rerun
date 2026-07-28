import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Step 4: Factor ingest — FF5 + Momentum + Short-Term Reversal, DAILY.
# Note: the original pull contained the MONTHLY FF5 (inside a zip renamed
# .csv) and MONTHLY ST_Rev; those are quarantined as *.WRONG_* in
# data/raw/factors/. The daily files below were re-pulled from the Ken
# French library on 2026-07-11.
# ---------------------------------------------------------------------------

project_root = Path(__file__).parent.parent
factors_dir = project_root / 'data' / 'raw' / 'factors'
output_path = project_root / 'data' / 'processed' / 'factors_daily.parquet'

FILES = [
    ('F-F_Research_Data_5_Factors_2x3_daily.csv',
     ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']),
    ('F-F_Momentum_Factor_daily.csv', ['MOM']),
    ('F-F_ST_Reversal_Factor_daily.csv', ['ST_Rev']),
]

print("=" * 80)
print("FACTOR INGEST: FF5 + MOM + ST_Rev, daily, percent -> decimal")
print("=" * 80)

frames = []
for fname, want_cols in FILES:
    path = factors_dir / fname
    raw = path.read_text().splitlines()
    print(f"\n### {fname} ({len(raw)} lines) — first 10 raw lines:")
    for line in raw[:10]:
        print("  |" + line)

    # find the header row: starts with a comma or 'Date', names the columns
    hdr_i = None
    for i, line in enumerate(raw):
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 2 and parts[0] in ('', 'Date') and parts[1] != '':
            hdr_i = i
            break
    if hdr_i is None:
        print(f"STOP: no header row found in {fname}")
        raise SystemExit(1)
    cols = [p.strip() for p in raw[hdr_i].split(',')][1:]

    # data rows: 8-digit YYYYMMDD in the first field; stop at first non-match
    # (blank line / annual summary section / copyright footer)
    rows = []
    for line in raw[hdr_i + 1:]:
        first = line.split(',')[0].strip()
        if len(first) == 8 and first.isdigit():
            rows.append(line.split(','))
        elif rows:
            break
    parsed = pd.DataFrame(rows).iloc[:, :len(cols) + 1]
    parsed.columns = ['date'] + cols
    parsed['date'] = pd.to_datetime(parsed['date'].str.strip(),
                                    format='%Y%m%d')
    for c in cols:
        parsed[c] = pd.to_numeric(parsed[c], errors='coerce')
        # Ken French missing-data markers
        parsed.loc[parsed[c].isin([-99.99, -999.0]), c] = np.nan
        parsed[c] = parsed[c] / 100.0          # percent -> decimal

    parsed = parsed.rename(columns={'Mom': 'MOM'})
    missing = [c for c in want_cols if c not in parsed.columns]
    if missing:
        print(f"STOP: expected columns {missing} not found in {fname}; "
              f"got {list(parsed.columns)}")
        raise SystemExit(1)
    parsed = parsed[['date'] + want_cols]
    print(f"  parsed: {parsed.shape[0]:,} daily rows, "
          f"{parsed['date'].min().date()} to {parsed['date'].max().date()}, "
          f"cols {want_cols}")
    frames.append(parsed)

# --------------------------------------------------------------------------
# Merge and clip to project sample
# --------------------------------------------------------------------------
print("\n" + "-" * 80)
print("MERGE")
print("-" * 80)

fac = frames[0]
for f in frames[1:]:
    fac = fac.merge(f, on='date', how='inner')

fac = fac[(fac['date'] >= '2015-01-01') & (fac['date'] <= '2025-12-31')]
fac = fac.sort_values('date').reset_index(drop=True)

print(f"Merged daily factor table: {fac.shape[0]:,} rows x "
      f"{fac.shape[1] - 1} factors")
print(f"Date range: {fac['date'].min().date()} to {fac['date'].max().date()}")
n = len(fac)
print(f"Row count check (expect ~2700-2800 for 11y of trading days): {n} "
      f"{'[PASS]' if 2700 <= n <= 2800 else '[WARNING - investigate]'}")
print("\nNulls per column:")
for c in fac.columns[1:]:
    print(f"  {c:>7}: {int(fac[c].isna().sum())}")
print("\nSummary (daily, decimal):")
print(fac.drop(columns='date').describe().loc[['mean', 'std', 'min', 'max']]
      .T.to_string(float_format=lambda x: f"{x:+.5f}"))

fac.to_parquet(output_path, index=False)
print(f"\n[OK] Saved to {output_path}")

print("\n" + "=" * 80)
print("FACTOR INGEST COMPLETE")
print("=" * 80)
