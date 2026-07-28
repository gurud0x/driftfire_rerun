import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Phase V2 prep: validate the Compustat GICS pull. INSPECTION ONLY —
# no processed output is written; the real ingest happens after the CCM
# gvkey-to-PERMNO linking table is also pulled and validated.
# ---------------------------------------------------------------------------

project_root = Path(__file__).parent.parent
csv_path = project_root / 'data' / 'raw' / 'compustat' / 'compustat_gics_names.csv'

# GICS sector names per the published standard, keyed by 2-digit code.
# Used only to label codes that actually appear in the data.
GICS_SECTOR_NAMES = {
    10: 'Energy', 15: 'Materials', 20: 'Industrials',
    25: 'Consumer Discretionary', 30: 'Consumer Staples',
    35: 'Health Care', 40: 'Financials', 45: 'Information Technology',
    50: 'Communication Services', 55: 'Utilities', 60: 'Real Estate',
}

print("=" * 78)
print("VALIDATION: Compustat GICS pull (data/raw/compustat/"
      "compustat_gics_names.csv)")
print("=" * 78)

if not csv_path.exists():
    print(f"STOP: {csv_path} not found.")
    raise SystemExit(1)

df = pd.read_csv(csv_path)

# --------------------------------------------------------------------------
# 1-3. Columns, shape, date range
# --------------------------------------------------------------------------
print("\n### Full column list as pulled:")
for c in df.columns:
    print("   " + c)

print(f"\nRows: {len(df):,}")
if 'gvkey' in df.columns:
    print(f"Unique gvkey: {df['gvkey'].nunique():,}")
else:
    print("STOP: no gvkey column — wrong file?")
    raise SystemExit(1)
if 'datadate' in df.columns:
    dd = pd.to_datetime(df['datadate'], errors='coerce')
    print(f"datadate range: {dd.min().date()} to {dd.max().date()} "
          f"({dd.isna().sum()} unparseable)")
else:
    print("NOTE: no datadate column in this pull.")

# --------------------------------------------------------------------------
# 4. Null rates on the four GICS fields the V2 signal depends on
# --------------------------------------------------------------------------
print("\n### Null rates on GICS fields (signal-critical):")
for c in ['gsector', 'ggroup', 'gind', 'gsubind']:
    if c in df.columns:
        n_null = int(df[c].isna().sum())
        print(f"   {c:>8}: {n_null:,} null of {len(df):,} "
              f"({n_null/len(df)*100:.2f}%)")
    else:
        print(f"   {c:>8}: COLUMN MISSING FROM PULL")

# --------------------------------------------------------------------------
# 5. gsector distribution with standard names where identifiable
# --------------------------------------------------------------------------
if 'gsector' in df.columns:
    print("\n### gsector distribution:")
    counts = df['gsector'].value_counts(dropna=False).sort_index()
    print(f"   Unique non-null sector codes: {df['gsector'].nunique()}")
    for code, cnt in counts.items():
        if pd.isna(code):
            print(f"   {'NaN':>6}  {cnt:>9,}  (unclassified rows)")
        else:
            name = GICS_SECTOR_NAMES.get(int(code), 'UNRECOGNIZED CODE')
            print(f"   {int(code):>6}  {cnt:>9,}  {name}")
    codes = set(int(c) for c in counts.index if pd.notna(c))
    expected = set(GICS_SECTOR_NAMES)
    print(f"   Matches the 11 published GICS sectors: "
          f"{codes == expected} (present: {sorted(codes)})")
    if codes - expected:
        print(f"   [WARNING] unexpected codes: {sorted(codes - expected)}")
    if expected - codes:
        print(f"   [NOTE] published sectors absent from pull: "
              f"{sorted(expected - codes)}")

# --------------------------------------------------------------------------
# 6. gvkey-datadate uniqueness (report only, no silent drop)
# --------------------------------------------------------------------------
print("\n### gvkey-datadate identifier check:")
if 'datadate' in df.columns:
    n_dup = int(df.duplicated(['gvkey', 'datadate']).sum())
    print(f"   Duplicate gvkey-datadate pairs: {n_dup:,} "
          f"{'[PASS - clean identifier]' if n_dup == 0 else '[WARNING - not a clean key; reported, NOT dropped]'}")
    if n_dup:
        dup_keys = df[df.duplicated(['gvkey', 'datadate'], keep=False)]
        print(f"   Rows involved: {len(dup_keys):,} across "
              f"{dup_keys[['gvkey','datadate']].drop_duplicates().shape[0]:,} keys; "
              f"first examples:")
        print(dup_keys.head(4).to_string(index=False))
else:
    print("   (no datadate column; uniqueness check on gvkey alone:)")
    print(f"   Duplicate gvkey rows: {int(df.duplicated(['gvkey']).sum()):,}")

# --------------------------------------------------------------------------
# 7. Static vs time-varying gsector per gvkey
# --------------------------------------------------------------------------
print("\n### Static vs time-varying (same question asked of SICCD):")
if 'gsector' in df.columns:
    nu = df.dropna(subset=['gsector']).groupby('gvkey')['gsector'].nunique()
    movers = int((nu > 1).sum())
    print(f"   gvkeys with a non-null gsector: {len(nu):,}")
    print(f"   gvkeys with >1 distinct gsector across rows: {movers:,} "
          f"({movers/len(nu)*100:.1f}%)")
    if movers > 0:
        print("   => TIME-VARYING: the V2 signal needs point-in-time GICS")
        print("      handling (as-of-date), same as SICCD.")
        ex = nu[nu > 1].index[:2]
        for g in ex:
            h = (df[df['gvkey'] == g].dropna(subset=['gsector'])
                 .sort_values('datadate' if 'datadate' in df.columns else 'gvkey'))
            ch = h['gsector'] != h['gsector'].shift()
            idx = ch[ch].index[1:2]
            if len(idx):
                i = h.index.get_loc(idx[0])
                win = h.iloc[max(0, i - 1):i + 1]
                parts = [f"{r['datadate']}: {int(r['gsector'])}"
                         for _, r in win.iterrows()] if 'datadate' in df.columns else []
                print(f"      gvkey {g}: " + "  ->  ".join(parts))
    else:
        print("   => STATIC in this pull: every gvkey carries one gsector.")
        print("      (Note: a names-file pull often reflects only the CURRENT")
        print("      classification; true GICS history may require the")
        print("      Compustat co_hgic history table. Flagged for the V2")
        print("      pre-registration to state explicitly.)")

# --------------------------------------------------------------------------
# 8. Example rows
# --------------------------------------------------------------------------
print("\n### 5 example rows:")
show_cols = [c for c in ['gvkey', 'tic', 'datadate', 'gsector', 'ggroup',
                         'gind', 'gsubind'] if c in df.columns]
ex = df.dropna(subset=['gsector']).sample(5, random_state=42)[show_cols]
print(ex.to_string(index=False))

print("\n" + "=" * 78)
print("VALIDATION COMPLETE - inspection only, nothing written to")
print("data/processed/. Ingest waits for the CCM linking table.")
print("=" * 78)
