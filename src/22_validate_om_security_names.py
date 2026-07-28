import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# K1 prep: validate the OptionMetrics security-names pull. INSPECTION ONLY.
# OptionMetrics contact is sanctioned by the K0 horse race decision logged
# in results/gate_log.md (winner TRAIL20), per V1's locked sequencing.
# No processed output is written; the real coverage number needs Pull 2
# (the volatility surface) — this file only establishes the name link.
# ---------------------------------------------------------------------------

project_root = Path(__file__).parent.parent
om_path = project_root / 'data' / 'raw' / 'optionmetrics' / 'om_security_names.csv'
univ_path = project_root / 'data' / 'processed' / 'universe_membership.parquet'
names_path = project_root / 'data' / 'raw' / 'crsp' / 'crsp_names.parquet'

print("=" * 78)
print("VALIDATION: OptionMetrics security names "
      "(data/raw/optionmetrics/om_security_names.csv)")
print("=" * 78)

om = pd.read_csv(om_path)

# --------------------------------------------------------------------------
# 1-3. Columns, shape, join-key null rates
# --------------------------------------------------------------------------
print("\n### Full column list as pulled:")
for c in om.columns:
    print("   " + c)
om.columns = [c.lower() for c in om.columns]

print(f"\nRows: {len(om):,}")
print(f"Unique secid: {om['secid'].nunique():,}")
print(f"Unique cusip: {om['cusip'].nunique():,}")

print("\n### Join-key null rates (a null here breaks the CRSP link):")
for c in ['secid', 'cusip']:
    n_null = int(om[c].isna().sum())
    print(f"   {c:>6}: {n_null:,} null ({n_null/len(om)*100:.2f}%)")

# --------------------------------------------------------------------------
# 4. secid stability: one row per secid, or history rows?
# --------------------------------------------------------------------------
print("\n### secid row multiplicity:")
per_secid = om.groupby('secid').size()
multi = per_secid[per_secid > 1]
print(f"   secids with >1 row: {len(multi):,} of {len(per_secid):,} "
      f"({len(multi)/len(per_secid)*100:.1f}%)")
print(f"   max rows for a single secid: {per_secid.max()}")
if len(multi):
    print("   => secid identifying info changes over time (effect_date")
    print("      versioning) — the eventual link must be as-of-date, the")
    print("      same treatment SICCD and gsector required.")
    ex_id = multi.index[0]
    ex = om[om['secid'] == ex_id]
    show = [c for c in ['secid', 'cusip', 'ticker', 'effect_date', 'issuer']
            if c in om.columns]
    print(f"   example (secid {ex_id}):")
    print(ex[show].head(4).to_string(index=False))

# --------------------------------------------------------------------------
# 5. cusip -> secid conflicts
# --------------------------------------------------------------------------
print("\n### cusip -> secid mapping check:")
cus = om.dropna(subset=['cusip'])
sec_per_cusip = cus.groupby('cusip')['secid'].nunique()
conflicts = sec_per_cusip[sec_per_cusip > 1]
print(f"   cusips mapping to >1 secid: {len(conflicts):,} of "
      f"{len(sec_per_cusip):,} ({len(conflicts)/len(sec_per_cusip)*100:.2f}%)")
if len(conflicts):
    print("   [WARNING] these need a disambiguation rule (e.g. effect_date")
    print("   windows or issue-level matching) before the K1 join; first 3:")
    for c in conflicts.index[:3]:
        rows = om[om['cusip'] == c]
        print(f"     cusip {c}: secids "
              f"{sorted(rows['secid'].unique().tolist())}")

# --------------------------------------------------------------------------
# 6-7. effect_date range, example rows
# --------------------------------------------------------------------------
if 'effect_date' in om.columns:
    ed = pd.to_datetime(om['effect_date'], errors='coerce')
    print(f"\n### effect_date range: {ed.min().date()} to {ed.max().date()} "
          f"({int(ed.isna().sum())} unparseable)")

print("\n### 5 example rows:")
show = [c for c in ['secid', 'cusip', 'ticker', 'effect_date', 'issue',
                    'issuer'] if c in om.columns]
print(om.sample(5, random_state=42)[show].to_string(index=False))

# --------------------------------------------------------------------------
# 8. Preliminary universe coverage via CUSIP (name match only, NOT IV)
# --------------------------------------------------------------------------
print("\n" + "-" * 78)
print("PRELIMINARY COVERAGE: universe PERMNOs with ANY OptionMetrics name "
      "match")
print("-" * 78)

names = pd.read_parquet(names_path)
print(f"\ncrsp_names columns (confirming CUSIP field, not assuming): "
      f"{list(names.columns)}")
if 'CUSIP' not in names.columns:
    print("STOP: no CUSIP field in crsp_names — preliminary join not possible.")
    raise SystemExit(1)

univ = pd.read_parquet(univ_path)
upermnos = set(univ.loc[univ['in_universe'], 'PERMNO'].unique())
crsp_cusips = names[names['PERMNO'].isin(upermnos)][['PERMNO', 'CUSIP']]
crsp_cusips = crsp_cusips.dropna(subset=['CUSIP']).drop_duplicates()
print(f"Universe PERMNOs: {len(upermnos):,}; PERMNO-CUSIP pairs from "
      f"crsp_names: {len(crsp_cusips):,}")

# CRSP CUSIPs are 8-character; OptionMetrics cusip may be 8 or 9 — compare
# on the first 8 characters of each, uppercased.
om_cusip8 = set(cus['cusip'].astype(str).str.upper().str[:8])
crsp_cusips['c8'] = crsp_cusips['CUSIP'].astype(str).str.upper().str[:8]
matched_permnos = set(crsp_cusips.loc[crsp_cusips['c8'].isin(om_cusip8),
                                      'PERMNO'])
n_match = len(matched_permnos & upermnos)
print(f"\nUniverse PERMNOs with >=1 CUSIP match in OptionMetrics: "
      f"{n_match:,} of {len(upermnos):,} "
      f"({n_match/len(upermnos)*100:.1f}%)")
print(f"Unmatched: {len(upermnos)-n_match:,} "
      f"({(len(upermnos)-n_match)/len(upermnos)*100:.1f}%)")
print("\n[NOTE] This is a NAME-LINK ceiling only. A matched name does not")
print("guarantee usable IV data (many small-caps have no listed options or")
print("no surface coverage). The real K1 coverage number - decile-1")
print("stock-days with usable IV - requires Pull 2 (volatility surface)")
print("and is what the K1 threshold gets locked against.")

print("\n" + "=" * 78)
print("VALIDATION COMPLETE - inspection only, nothing written to")
print("data/processed/.")
print("=" * 78)
