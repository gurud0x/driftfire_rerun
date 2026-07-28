import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# K1 prep: filter the full-grid OptionMetrics volatility surface down to the
# near-ATM band at the two tenors K1 needs. Chunked throughout — the source
# is ~102 GB and is never loaded whole.
#
# NOTE ON SOURCE LOCATION: the full grid stays in the download staging
# folder; only the filtered output is written into the repo's data/raw/.
# Copying 102 GB into the OneDrive-synced project tree would trigger a
# 102 GB sync for no benefit.
#
# NOTE ON WRITING: filtered chunks are appended to disk rather than
# concatenated into one in-memory DataFrame. At ~10^8 surviving rows the
# concatenated frame (object-dtype date and cp_flag columns) would need
# tens of GB of RAM and would lose an hour of work on a MemoryError. The
# deliverable — one canonical CSV — is identical either way; summary
# statistics are accumulated across chunks for the validation prints.
# Output goes to a .tmp file and is promoted only on success, so the
# existing vol_surface.csv survives a mid-run failure.
# ---------------------------------------------------------------------------

CHUNKSIZE = 3_000_000
KEEP_DAYS = (10, 30)
DELTA_LO, DELTA_HI = 35, 65      # near-ATM band on |delta|, raw (scaled) units

project_root = Path(__file__).parent.parent
repo_om = project_root / 'data' / 'raw' / 'optionmetrics'
staging_om = Path.home() / 'Downloads' / 'quantdata' / 'driftfire' / 'raw' / 'optionmetrics'

candidates = [repo_om / 'vol_surface_full_grid.csv',
              staging_om / 'vol_surface_full_grid.csv']
src = None
for c in candidates:
    if c.exists():
        src = c
        break
if src is None:
    print("STOP: vol_surface_full_grid.csv not found in either location:")
    for c in candidates:
        print("   " + str(c))
    raise SystemExit(1)

out_path = repo_om / 'vol_surface.csv'
tmp_path = repo_om / 'vol_surface.csv.tmp'

print("=" * 78)
print("FILTER: full-grid volatility surface -> near-ATM, days 10/30")
print("=" * 78)
print(f"\nSource: {src}")
print(f"  size: {src.stat().st_size / 1e9:.1f} GB")
print(f"Output: {out_path}")
if out_path.exists():
    print(f"  [NOTE] output path already exists "
          f"({out_path.stat().st_size / 1e9:.2f} GB) and will be REPLACED.")
    print(f"  The prior file is the one-year 2024-2025 pull; a copy remains")
    print(f"  in the staging folder ({staging_om}).")
print(f"  Writing to {tmp_path.name} first; promoted only on success.")

DTYPES = {
    'secid': 'int32',
    'date': 'str',
    'days': 'int16',
    'delta': 'float32',
    'impl_volatility': 'float64',
    'impl_strike': 'float64',
    'impl_premium': 'float64',
    'cp_flag': 'str',
    'index_flag': 'int8',
}

print(f"\nChunked read: chunksize={CHUNKSIZE:,}, explicit dtypes, dates kept")
print("as strings (no per-chunk datetime parsing).")
print("\n" + "-" * 78)
print("SCANNING")
print("-" * 78)

if tmp_path.exists():
    tmp_path.unlink()

rows_in = 0
rows_out = 0
first_write = True
delta_min = None
delta_max = None
iv_null = 0
iv_zero = 0
days_seen = {}
out_days = {}
out_delta = {}
out_secids = set()
out_date_min = None
out_date_max = None
prem_zero_out = 0
recon_printed = False

reader = pd.read_csv(src, chunksize=CHUNKSIZE, dtype=DTYPES)
for i, chunk in enumerate(reader, 1):
    rows_in += len(chunk)

    # ---- pre-filter diagnostics (accumulated over the whole file) ----
    cmin, cmax = chunk['delta'].min(), chunk['delta'].max()
    delta_min = cmin if delta_min is None else min(delta_min, cmin)
    delta_max = cmax if delta_max is None else max(delta_max, cmax)
    iv_null += int(chunk['impl_volatility'].isna().sum())
    iv_zero += int((chunk['impl_volatility'] == 0).sum())
    for d, n in chunk['days'].value_counts().items():
        days_seen[int(d)] = days_seen.get(int(d), 0) + int(n)

    if not recon_printed:
        print("\n### First-chunk reconnaissance (before any filtering):")
        print(f"   columns: {list(chunk.columns)}")
        print(f"   delta observed min/max in chunk 1: {cmin:.1f} / {cmax:.1f}")
        print(f"   => delta is stored as an INTEGER SCALED BY 100 "
              f"(-90 means -0.90 delta).")
        print(f"      The near-ATM band {DELTA_LO}-{DELTA_HI} in these raw "
              f"units = 0.35-0.65 in true delta.")
        print(f"   distinct delta values in chunk 1: "
              f"{sorted(int(v) for v in chunk['delta'].dropna().unique())}")
        n_null = int(chunk['impl_volatility'].isna().sum())
        n_zero = int((chunk['impl_volatility'] == 0).sum())
        print(f"\n   impl_volatility in chunk 1: {n_null:,} null "
              f"({n_null/len(chunk)*100:.1f}%), {n_zero:,} exactly zero "
              f"({n_zero/len(chunk)*100:.1f}%)")
        print(f"   impl_strike zero: "
              f"{int((chunk['impl_strike'] == 0).sum()):,}; "
              f"impl_premium zero: "
              f"{int((chunk['impl_premium'] == 0).sum()):,}")
        print("   => missing IV is represented as an EMPTY FIELD (true null),")
        print("      not as 0. The 0s appear in impl_strike/impl_premium on")
        print("      those same no-data rows. Both null and zero IV are")
        print("      excluded by the filter regardless, per spec.")
        recon_printed = True

    # ---- the filter ----
    f = chunk[chunk['days'].isin(KEEP_DAYS)]
    ad = f['delta'].abs()
    f = f[(ad >= DELTA_LO) & (ad <= DELTA_HI)]
    f = f[f['impl_volatility'].notna() & (f['impl_volatility'] != 0)]

    if len(f):
        for d, n in f['days'].value_counts().items():
            out_days[int(d)] = out_days.get(int(d), 0) + int(n)
        for d, n in f['delta'].value_counts().items():
            out_delta[int(d)] = out_delta.get(int(d), 0) + int(n)
        out_secids.update(f['secid'].unique().tolist())
        cdmin, cdmax = f['date'].min(), f['date'].max()
        out_date_min = cdmin if out_date_min is None else min(out_date_min, cdmin)
        out_date_max = cdmax if out_date_max is None else max(out_date_max, cdmax)
        prem_zero_out += int((f['impl_premium'] == 0).sum())

        f.to_csv(tmp_path, mode='w' if first_write else 'a',
                 header=first_write, index=False)
        first_write = False
        rows_out += len(f)

    if i % 20 == 0 or i == 1:
        print(f"  chunk {i:>5}: read {rows_in:>15,} rows -> kept "
              f"{rows_out:>13,} ({rows_out/rows_in*100:5.2f}%)")

print(f"\n  FINAL: read {rows_in:,} rows -> kept {rows_out:,} "
      f"({rows_out/rows_in*100:.2f}%)")

# --------------------------------------------------------------------------
# Promote the temp file
# --------------------------------------------------------------------------
if rows_out == 0:
    print("\nSTOP: filter kept zero rows; leaving the existing output file "
          "untouched.")
    if tmp_path.exists():
        tmp_path.unlink()
    raise SystemExit(1)
tmp_path.replace(out_path)

# --------------------------------------------------------------------------
# VALIDATION
# --------------------------------------------------------------------------
print("\n" + "=" * 78)
print("VALIDATION OF THE FILTERED FILE")
print("=" * 78)

print(f"\nRows: {rows_out:,}")
print(f"Unique secid: {len(out_secids):,}")
print(f"Date range: {out_date_min} to {out_date_max}")

print(f"\ndays distribution (requested: {KEEP_DAYS}):")
for d in sorted(out_days):
    print(f"   days {d:>3}: {out_days[d]:>14,} "
          f"({out_days[d]/rows_out*100:5.1f}%)")
ok_days = set(out_days) == set(KEEP_DAYS)
print(f"   matches request: {ok_days} {'[PASS]' if ok_days else '[FAIL]'}")

print(f"\ndelta distribution (requested |delta| in "
      f"[{DELTA_LO}, {DELTA_HI}], both signs):")
for d in sorted(out_delta):
    print(f"   delta {d:>4}: {out_delta[d]:>14,}")
bad = [d for d in out_delta if not (DELTA_LO <= abs(d) <= DELTA_HI)]
print(f"   out-of-band values present: {bad if bad else 'none'} "
      f"{'[PASS]' if not bad else '[FAIL]'}")

print(f"\nPre-filter diagnostics over the whole source file:")
print(f"   delta min/max observed: {delta_min:.1f} / {delta_max:.1f} "
      f"(confirms x100 integer scaling)")
print(f"   impl_volatility null: {iv_null:,} of {rows_in:,} "
      f"({iv_null/rows_in*100:.2f}%)")
print(f"   impl_volatility exactly zero: {iv_zero:,} "
      f"({iv_zero/rows_in*100:.2f}%)")
print(f"   tenors present in source: {sorted(days_seen)}")
print(f"\n   impl_premium == 0 among KEPT rows: {prem_zero_out:,} "
      f"({prem_zero_out/rows_out*100:.2f}%)  [relevant to K1 straddle "
      f"pricing]")

size_gb = out_path.stat().st_size / 1e9
print(f"\nOutput file: {out_path}")
print(f"   size: {size_gb:.2f} GB (source was "
      f"{src.stat().st_size / 1e9:.1f} GB; "
      f"{size_gb / (src.stat().st_size / 1e9) * 100:.1f}% of source)")
print(f"   source file left untouched at: {src}")

print("\n" + "=" * 78)
print("FILTER COMPLETE - vol_surface.csv is now the canonical near-ATM file.")
print("=" * 78)
