import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Phase V2 prep: validate the CCM link table (gvkey <-> PERMNO) with GICS.
# INSPECTION ONLY — nothing written; ingest happens after both Compustat
# pulls are validated.
# ---------------------------------------------------------------------------

project_root = Path(__file__).parent.parent
link_path = project_root / 'data' / 'raw' / 'compustat' / 'ccm_link_gics.csv'
names_path = project_root / 'data' / 'raw' / 'compustat' / 'compustat_gics_names.csv'
univ_path = project_root / 'data' / 'processed' / 'universe_membership.parquet'

print("=" * 78)
print("VALIDATION: CCM link table (data/raw/compustat/ccm_link_gics.csv)")
print("=" * 78)

link = pd.read_csv(link_path)

# --------------------------------------------------------------------------
# 1. Columns, shape, identifier counts
# --------------------------------------------------------------------------
print("\n### Full column list as pulled:")
for c in link.columns:
    print("   " + c)
# WRDS delivered link fields uppercase (LPERMNO, LINKDT, ...) and gvkey/tic
# lowercase; normalize to lowercase for the checks below. Case only — no
# renaming beyond that.
link.columns = [c.lower() for c in link.columns]
print("   (normalized to lowercase for the checks below)")
print(f"\nRows: {len(link):,}")
print(f"Unique gvkey: {link['gvkey'].nunique():,}")
if 'lpermno' in link.columns:
    print(f"Unique lpermno: {link['lpermno'].nunique():,}")
else:
    print("STOP: no lpermno column — wrong file?")
    raise SystemExit(1)

# --------------------------------------------------------------------------
# 2. Link quality
# --------------------------------------------------------------------------
print("\n### Link quality:")
for c in ['linktype', 'linkprim']:
    if c in link.columns:
        print(f"   {c} distribution: "
              f"{link[c].value_counts(dropna=False).to_dict()}")
    else:
        print(f"   {c}: COLUMN MISSING")
n_null_permno = int(link['lpermno'].isna().sum())
print(f"   Null lpermno rows: {n_null_permno:,} "
      f"({n_null_permno/len(link)*100:.2f}%) — gvkeys with no CRSP match")
if 'linkdt' in link.columns:
    ld = pd.to_datetime(link['linkdt'], errors='coerce')
    print(f"   linkdt range: {ld.min().date()} to {ld.max().date()}")
if 'linkenddt' in link.columns:
    le = pd.to_datetime(link['linkenddt'], errors='coerce')
    n_open = int(le.isna().sum())
    print(f"   linkenddt range: {le.min().date()} to {le.max().date()}; "
          f"null (still-active links): {n_open:,} "
          f"({n_open/len(link)*100:.1f}%)")
key_cols = [c for c in ['gvkey', 'lpermno', 'linkdt'] if c in link.columns]
n_dup = int(link.duplicated(key_cols).sum())
print(f"   Duplicate {'-'.join(key_cols)} rows: {n_dup:,} "
      f"{'[PASS]' if n_dup == 0 else '[WARNING - reported, not dropped]'}")

# --------------------------------------------------------------------------
# 3. Coverage vs the deciles 6-8 universe
# --------------------------------------------------------------------------
print("\n### Coverage vs universe_membership.parquet (deciles 6-8):")
univ = pd.read_parquet(univ_path)
upermnos = set(univ.loc[univ['in_universe'], 'PERMNO'].unique())
lpermnos = set(link['lpermno'].dropna().astype(int).unique())
matched = upermnos & lpermnos
print(f"   Universe PERMNOs (ever in-universe): {len(upermnos):,}")
print(f"   Matched by a CCM link row: {len(matched):,} "
      f"({len(matched)/len(upermnos)*100:.1f}%)")
print(f"   Unmatched: {len(upermnos)-len(matched):,} "
      f"({(len(upermnos)-len(matched))/len(upermnos)*100:.1f}%)")

# --------------------------------------------------------------------------
# 4. GICS fields on the link file
# --------------------------------------------------------------------------
print("\n### GICS fields in this file:")
for c in ['gsector', 'ggroup', 'gind', 'gsubind']:
    if c in link.columns:
        n_null = int(link[c].isna().sum())
        print(f"   {c:>8}: {n_null:,} null ({n_null/len(link)*100:.2f}%)")
    else:
        print(f"   {c:>8}: COLUMN MISSING")
if 'gsector' in link.columns:
    counts = link['gsector'].value_counts(dropna=True).sort_index()
    print(f"   Unique gsector codes: {len(counts)} "
          f"(expected 11): {sorted(int(c) for c in counts.index)}")
    nu = link.dropna(subset=['gsector']).groupby('gvkey')['gsector'].nunique()
    movers = int((nu > 1).sum())
    print(f"   gvkeys with >1 distinct gsector: {movers:,} of {len(nu):,} "
          f"({movers/len(nu)*100:.1f}%) — "
          f"{'TIME-VARYING' if movers else 'STATIC in this pull'}")
print("\n   5 example rows:")
show = [c for c in ['gvkey', 'lpermno', 'linkdt', 'linkenddt', 'linktype',
                    'linkprim', 'gsector', 'gsubind'] if c in link.columns]
print(link.dropna(subset=['lpermno']).sample(5, random_state=42)[show]
      .to_string(index=False))

# --------------------------------------------------------------------------
# 5. Cross-check gsector against the earlier names pull
# --------------------------------------------------------------------------
print("\n### Cross-check vs compustat_gics_names.csv (earlier pull):")
names = pd.read_csv(names_path, usecols=['gvkey', 'gsector'])
a = (link.dropna(subset=['gsector'])[['gvkey', 'gsector']]
     .drop_duplicates('gvkey').set_index('gvkey')['gsector'])
b = (names.dropna(subset=['gsector'])
     .drop_duplicates('gvkey').set_index('gvkey')['gsector'])
shared = a.index.intersection(b.index)
agree = int((a.loc[shared] == b.loc[shared]).sum())
print(f"   Shared gvkeys with gsector in both files: {len(shared):,}")
print(f"   gsector agreement: {agree:,} ({agree/len(shared)*100:.2f}%) "
      f"{'[PASS]' if agree == len(shared) else '[WARNING - disagreements below]'}")
if agree != len(shared):
    dis = shared[a.loc[shared] != b.loc[shared]]
    print(f"   Disagreeing gvkeys: {len(dis):,}; first 5:")
    for g in dis[:5]:
        print(f"      gvkey {g}: link file={int(a.loc[g])}, "
              f"names file={int(b.loc[g])}")

print("\n" + "=" * 78)
print("VALIDATION COMPLETE - inspection only, nothing written to")
print("data/processed/.")
print("=" * 78)
