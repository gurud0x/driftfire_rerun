import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# K1 prep: build secid_list.txt for WRDS OptionMetrics volatility-surface
# query upload.  RECONNAISSANCE OUTPUT ONLY — produces a plain text list
# of integer secids for manual WRDS web query, not a processed data artifact.
# ---------------------------------------------------------------------------

project_root = Path(__file__).parent.parent
om_path   = project_root / 'data' / 'raw' / 'optionmetrics' / 'om_security_names.csv'
univ_path = project_root / 'data' / 'processed' / 'universe_membership.parquet'
names_path = project_root / 'data' / 'raw' / 'crsp' / 'crsp_names.parquet'
out_path  = project_root / 'data' / 'raw' / 'optionmetrics' / 'secid_list.txt'

# --------------------------------------------------------------------------
# 1. Load OptionMetrics security names; exclude sentinel cusip
# --------------------------------------------------------------------------
om = pd.read_csv(om_path)
om.columns = [c.lower() for c in om.columns]

before = len(om)
om = om[om['cusip'] != '99999999']
print(f"om_security_names: {before:,} rows loaded, "
      f"{before - len(om):,} sentinel cusip=='99999999' excluded, "
      f"{len(om):,} remaining")

# --------------------------------------------------------------------------
# 2. Universe PERMNOs
# --------------------------------------------------------------------------
univ = pd.read_parquet(univ_path)
upermnos = set(univ.loc[univ['in_universe'], 'PERMNO'].unique())
print(f"Universe PERMNOs: {len(upermnos):,}")

# --------------------------------------------------------------------------
# 3. PERMNO -> 8-char CUSIP from crsp_names
# --------------------------------------------------------------------------
names = pd.read_parquet(names_path)
if 'CUSIP' not in names.columns:
    raise SystemExit("STOP: no CUSIP field in crsp_names — cannot proceed.")

crsp_map = (
    names[names['PERMNO'].isin(upermnos)][['PERMNO', 'CUSIP']]
    .dropna(subset=['CUSIP'])
    .drop_duplicates()
    .copy()
)
crsp_map['c8'] = crsp_map['CUSIP'].astype(str).str.upper().str[:8]
print(f"crsp_names PERMNO-CUSIP pairs for universe: {len(crsp_map):,}")

# --------------------------------------------------------------------------
# 4. Join on 8-char CUSIP (same method as src/22 validation)
# --------------------------------------------------------------------------
om_clean = om.dropna(subset=['cusip', 'secid']).copy()
om_clean['c8'] = om_clean['cusip'].astype(str).str.upper().str[:8]

merged = crsp_map.merge(om_clean[['secid', 'c8']], on='c8', how='inner')

# --------------------------------------------------------------------------
# 5. Unique secids from the join
# --------------------------------------------------------------------------
secids = sorted(merged['secid'].unique().tolist())

# --------------------------------------------------------------------------
# 6. Print count
# --------------------------------------------------------------------------
print(f"\nUnique secids matched: {len(secids):,}  (expect ~3,706)")

# --------------------------------------------------------------------------
# 7. Write one secid per line — plain integers, no header, no index
# --------------------------------------------------------------------------
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w') as f:
    for sid in secids:
        f.write(f"{int(sid)}\n")

print(f"Written: {out_path}")
