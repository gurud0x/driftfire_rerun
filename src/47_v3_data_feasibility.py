from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEV_START = pd.Timestamp("2015-01-01")
DEV_END = pd.Timestamp("2021-12-31")
YEARS = list(range(2015, 2022))
SIZE_DECILES = [6, 7, 8]
SURFACE_CHUNKSIZE = 5_000_000

# STEP 3 (2026-08-02): full targeted liquidity pull, EXPLICITLY AUTHORIZED by
# the project owner after the STEP 2b cost estimate (~9.4 min projected).
# Columns are held to the authorized minimum - secid, date, volume,
# open_interest - and nothing else. bid/ask were NOT authorized and are NOT
# read; see prereg_V3 section 6(b) on the consequence for the spread criterion.
OPPRCD_CHUNKSIZE = 5_000_000
OPPRCD_LIQ_COLS = ["secid", "date", "volume", "open_interest"]
CONSOLIDATE_EVERY = 25
# Durable artifact: once written, no future run re-scans 75 GB. Same
# regenerate-without-rescan discipline src/37 established for its trade list.
LIQ_CACHE_NAME = "opprcd_liquidity_daily.parquet"


def human_size(num_bytes: int) -> str:
    gb = num_bytes / (1024 ** 3)
    return f"{gb:.2f} GB"


def as_cusip8(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.replace(r"\\.0$", "", regex=True).str.strip().str.upper()
    out = out.str[:8].str.zfill(8)
    return out


def print_inventory(project_root: Path) -> dict[str, Path | None]:
    repo_om = project_root / "data" / "raw" / "optionmetrics"
    staging_om = Path.home() / "Downloads" / "quantdata" / "driftfire" / "raw" / "optionmetrics"
    processed = project_root / "data" / "processed"

    tracked = {
        "raw/optionmetrics/vol_surface.csv": repo_om / "vol_surface.csv",
        "raw/optionmetrics/om_security_names.csv": repo_om / "om_security_names.csv",
        "raw/optionmetrics/secid_list.txt": repo_om / "secid_list.txt",
        "raw/optionmetrics/opprcd.csv": repo_om / "opprcd.csv",
        "staging/opprcd.csv": staging_om / "opprcd.csv",
        "processed/k1_portfolio_returns_daily.parquet": processed / "k1_portfolio_returns_daily.parquet",
        "processed/k1_portfolio_returns_daily_corrected.parquet": processed / "k1_portfolio_returns_daily_corrected.parquet",
        "processed/k1_portfolio_returns_real_prices.parquet": processed / "k1_portfolio_returns_real_prices.parquet",
        "processed/k1_trades_real_prices.parquet": processed / "k1_trades_real_prices.parquet",
        "processed/k1_trade_summary.json": processed / "k1_trade_summary.json",
    }

    print("=" * 88)
    print("V3 FEASIBILITY - LOCAL OPTIONMETRICS INVENTORY (READ-ONLY)")
    print("=" * 88)
    print("\nChecked assets tied to src/30, src/32, src/36, src/37:")
    for label, path in tracked.items():
        exists = path.exists()
        if exists:
            st = path.stat()
            print(f"  [YES] {label:<55}  {human_size(st.st_size):>10}  mtime {pd.Timestamp(st.st_mtime, unit='s')}")
        else:
            print(f"  [ NO] {label:<55}")

    print("\nInterpretation:")
    print("  - vol_surface.csv is present locally (standardized/surface IV source).")
    print("  - opprcd.csv is NOT in repo raw/, but an 80+ GB copy exists in the staging folder.")
    print("  - This script does not pull or re-scan opprcd.")

    return {
        "surf": tracked["raw/optionmetrics/vol_surface.csv"],
        "om_names": tracked["raw/optionmetrics/om_security_names.csv"],
        "opprcd_repo": tracked["raw/optionmetrics/opprcd.csv"],
        "opprcd_staging": tracked["staging/opprcd.csv"],
    }


def build_v1_dev_size_universe(project_root: Path) -> pd.DataFrame:
    v1_path = project_root / "data" / "processed" / "compression_signal_v1.parquet"
    univ_path = project_root / "data" / "processed" / "universe_membership.parquet"

    v1 = pd.read_parquet(v1_path, columns=["PERMNO", "DlyCalDt", "compression_ratio"])
    v1 = v1[(v1["DlyCalDt"] >= DEV_START) & (v1["DlyCalDt"] <= DEV_END)].copy()
    v1 = v1[v1["compression_ratio"].notna()].copy()
    v1["year_month"] = v1["DlyCalDt"].dt.to_period("M").astype(str)

    univ = pd.read_parquet(univ_path)
    univ = univ[univ["in_universe"] & univ["decile"].isin(SIZE_DECILES)][["PERMNO", "year_month", "decile"]]

    base = v1.merge(univ, on=["PERMNO", "year_month"], how="inner")
    base = base[["PERMNO", "DlyCalDt", "decile"]].drop_duplicates()
    base["year"] = base["DlyCalDt"].dt.year
    base = base[base["year"].isin(YEARS)].copy()
    base["date_str"] = base["DlyCalDt"].dt.strftime("%Y-%m-%d")

    print("\n" + "=" * 88)
    print("DENOMINATOR UNIVERSE")
    print("=" * 88)
    print("Definition: V1 compression_ratio is defined (non-null), DEV window, in_universe, size decile 6/7/8.")
    print(f"Rows (PERMNO-date): {len(base):,}")
    print(f"Unique PERMNO: {base['PERMNO'].nunique():,}")
    print(f"Date range: {base['DlyCalDt'].min().date()} to {base['DlyCalDt'].max().date()}")

    return base


def build_permno_secid_bridge(
    project_root: Path, om_names_path: Path, permnos: Iterable[int]
) -> pd.DataFrame:
    names_path = project_root / "data" / "raw" / "crsp" / "crsp_names.parquet"

    om = pd.read_csv(om_names_path)
    om.columns = [c.lower() for c in om.columns]
    om = om.dropna(subset=["secid", "cusip"]).copy()
    om["c8"] = as_cusip8(om["cusip"])
    om = om[om["c8"].str.len() == 8]
    om = om[["secid", "c8"]].drop_duplicates()

    permnos = set(int(p) for p in permnos)
    crsp = pd.read_parquet(names_path, columns=["PERMNO", "CUSIP"])
    crsp = crsp[crsp["PERMNO"].isin(permnos)].copy()
    crsp = crsp.dropna(subset=["PERMNO", "CUSIP"]).copy()
    crsp["c8"] = as_cusip8(crsp["CUSIP"])
    crsp = crsp[crsp["c8"].str.len() == 8][["PERMNO", "c8"]].drop_duplicates()

    bridge = crsp.merge(om, on="c8", how="inner")[["PERMNO", "secid"]].drop_duplicates()
    bridge["secid"] = bridge["secid"].astype("int64")
    return bridge


def scan_surface_atm(surface_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[int], int | None, pd.DataFrame]:
    usecols = ["secid", "date", "days", "delta", "impl_volatility", "cp_flag"]
    dtypes = {
        "secid": "int32",
        "date": "str",
        "days": "int16",
        "delta": "float32",
        "impl_volatility": "float64",
        "cp_flag": "category",
    }

    rows_scanned = 0
    # Unique keys at secid-date-days-side granularity after nearest-ATM selection.
    side_keys: set[tuple[int, str, int, str]] = set()

    print("\n" + "=" * 88)
    print("SCANNING SURFACE IV (READ-ONLY)")
    print("=" * 88)
    for i, ch in enumerate(pd.read_csv(surface_path, usecols=usecols, dtype=dtypes, chunksize=SURFACE_CHUNKSIZE), 1):
        rows_scanned += len(ch)
        ch = ch[ch["impl_volatility"].notna() & (ch["impl_volatility"] > 0)].copy()
        ch = ch[ch["delta"].notna() & ch["cp_flag"].isin(["C", "P"])].copy()
        if len(ch) == 0:
            continue

        ch["dpen"] = (ch["delta"].abs() / 100.0 - 0.50).abs()
        ch = (
            ch.sort_values(["secid", "date", "days", "cp_flag", "dpen"])
            .drop_duplicates(["secid", "date", "days", "cp_flag"], keep="first")
        )

        key_rows = ch[["secid", "date", "days", "cp_flag"]].drop_duplicates()
        side_keys.update(key_rows.itertuples(index=False, name=None))

        if i == 1 or i % 5 == 0:
            print(f"  chunk {i:>4}: scanned {rows_scanned:>14,} rows")

    if not side_keys:
        raise RuntimeError("No valid surface IV rows found.")

    side = pd.DataFrame(list(side_keys), columns=["secid", "date", "days", "cp_flag"])
    tenors = sorted(int(x) for x in side["days"].dropna().unique())
    fixed_tenor = 30 if 30 in tenors else (min(tenors) if tenors else None)

    tenor_cp = (
        side.groupby(["secid", "date", "days"], observed=True)["cp_flag"]
        .nunique()
        .reset_index(name="cp_sides")
    )
    valid = tenor_cp[tenor_cp["cp_sides"] == 2].copy()

    any_iv = valid[["secid", "date"]].drop_duplicates()
    any_iv["has_any_atm_iv"] = True

    if fixed_tenor is None:
        fixed_iv = pd.DataFrame(columns=["secid", "date"])
    else:
        fixed_iv = valid[valid["days"] == fixed_tenor][["secid", "date"]].drop_duplicates()
    fixed_iv["has_fixed_tenor_iv"] = True

    print(f"\nSurface rows scanned: {rows_scanned:,}")
    print(f"Surface tenors found (days): {tenors}")
    if fixed_tenor is not None:
        print(f"Fixed short tenor used for coverage metric: {fixed_tenor}d")

    return any_iv, fixed_iv, tenors, fixed_tenor, valid[["secid", "date", "days"]].copy()


def summarize_coverage(
    base: pd.DataFrame,
    bridge: pd.DataFrame,
    any_iv: pd.DataFrame,
    fixed_iv: pd.DataFrame,
    fixed_tenor: int | None,
    liquidity_available: bool,
) -> pd.DataFrame:
    cand = base.merge(bridge, on="PERMNO", how="left")
    cand = cand.merge(any_iv, left_on=["secid", "date_str"], right_on=["secid", "date"], how="left")
    cand = cand.merge(fixed_iv, left_on=["secid", "date_str"], right_on=["secid", "date"], how="left")

    for c in ["has_any_atm_iv", "has_fixed_tenor_iv"]:
        if c in cand.columns:
            cand[c] = cand[c].fillna(False).astype(bool)

    per_stock_day = (
        cand.groupby(["PERMNO", "DlyCalDt", "year", "decile"], observed=True)
        .agg(
            has_any_atm_iv=("has_any_atm_iv", "max"),
            has_fixed_tenor_iv=("has_fixed_tenor_iv", "max"),
            bridged_to_any_secid=("secid", lambda s: s.notna().any()),
        )
        .reset_index()
    )

    stats = (
        per_stock_day.groupby(["year", "decile"], observed=True)
        .agg(
            n_obs=("PERMNO", "size"),
            n_any_iv=("has_any_atm_iv", "sum"),
            n_fixed_tenor_iv=("has_fixed_tenor_iv", "sum"),
            n_bridged=("bridged_to_any_secid", "sum"),
        )
        .reset_index()
    )

    if liquidity_available:
        # Placeholder path if a pre-aggregated chain-liquidity dataset is added later.
        stats["n_any_active_chain"] = np.nan
    else:
        stats["n_any_active_chain"] = np.nan

    grid = pd.MultiIndex.from_product([YEARS, SIZE_DECILES], names=["year", "decile"]).to_frame(index=False)
    out = grid.merge(stats, on=["year", "decile"], how="left")
    out[["n_obs", "n_any_iv", "n_fixed_tenor_iv", "n_bridged"]] = out[["n_obs", "n_any_iv", "n_fixed_tenor_iv", "n_bridged"]].fillna(0)

    out["pct_any_iv"] = np.where(out["n_obs"] > 0, out["n_any_iv"] / out["n_obs"] * 100.0, np.nan)
    out["pct_fixed_tenor_iv"] = np.where(out["n_obs"] > 0, out["n_fixed_tenor_iv"] / out["n_obs"] * 100.0, np.nan)
    out["pct_bridged"] = np.where(out["n_obs"] > 0, out["n_bridged"] / out["n_obs"] * 100.0, np.nan)
    out["pct_any_active_chain"] = np.nan

    print("\n" + "=" * 88)
    print("COVERAGE SUMMARY (2015-2021, SIZE DECILES 6/7/8)")
    print("=" * 88)
    print("Columns:")
    print("  - pct_any_iv: same-date ATM IV exists at any standardized tenor (requires both call+put sides).")
    if fixed_tenor is not None:
        print(f"  - pct_fixed_tenor_iv: same-date ATM IV exists specifically at {fixed_tenor}d tenor.")
    else:
        print("  - pct_fixed_tenor_iv: unavailable (no tenor values found in surface file).")
    print("  - pct_any_active_chain: NA in this run (no pre-aggregated chain-liquidity file; opprcd not re-scanned).")

    display_cols = [
        "year",
        "decile",
        "n_obs",
        "pct_bridged",
        "pct_any_iv",
        "pct_fixed_tenor_iv",
        "pct_any_active_chain",
    ]
    shown = out[display_cols].copy()
    for c in ["pct_bridged", "pct_any_iv", "pct_fixed_tenor_iv"]:
        shown[c] = shown[c].map(lambda v: f"{v:6.2f}" if pd.notna(v) else "   NA")
    shown["pct_any_active_chain"] = "NA"
    print(shown.to_string(index=False))

    return out


def compute_and_save_tenor_breakdown(
    project_root: Path, base: pd.DataFrame, bridge: pd.DataFrame,
    valid_tenor_days: pd.DataFrame,
) -> dict:
    """STEP 4 (2026-08-02, prereg_V3 section 3.3-F): persist the 10d/30d/both
    ATM-IV-availability breakdown by year x decile as a durable JSON artifact
    - the number this project's convention requires before any figure or
    prose can cite it (matching fig6/src/41/src/42's read-from-JSON, never
    transcribed, discipline). No return/RV/regression data touched - this is
    purely a surface-tenor coverage count, the same category already used to
    write section 3.3-F."""
    print("\n" + "=" * 88)
    print("STEP 4 - 10d/30d/BOTH ATM IV COVERAGE BY YEAR x DECILE (prereg_V3 3.3-F)")
    print("=" * 88)

    piv = (valid_tenor_days.assign(v=True)
           .pivot_table(index=["secid", "date"], columns="days", values="v",
                        aggfunc="max", fill_value=False))
    piv.columns = [f"iv{int(c)}" for c in piv.columns]
    piv = piv.reset_index()
    for c in ["iv10", "iv30"]:
        if c not in piv.columns:
            piv[c] = False
    piv["iv_both"] = piv["iv10"] & piv["iv30"]

    cand = base.merge(bridge, on="PERMNO", how="left").merge(
        piv, left_on=["secid", "date_str"], right_on=["secid", "date"], how="left")
    for c in ["iv10", "iv30", "iv_both"]:
        cand[c] = cand[c].fillna(False).astype(bool)
    per = (cand.groupby(["PERMNO", "DlyCalDt", "year", "decile"], observed=True)
           .agg(iv10=("iv10", "max"), iv30=("iv30", "max"),
                iv_both=("iv_both", "max")).reset_index())

    n = len(per)
    overall = {"pct_10": float(per["iv10"].mean() * 100),
              "pct_30": float(per["iv30"].mean() * 100),
              "pct_both": float(per["iv_both"].mean() * 100)}
    print(f"Universe stock-days: {n:,}")
    print(f"  10d tenor available : {overall['pct_10']:6.2f}%")
    print(f"  30d tenor available : {overall['pct_30']:6.2f}%")
    print(f"  BOTH (10d and 30d)  : {overall['pct_both']:6.2f}%")

    t = (per.groupby(["year", "decile"], observed=True)
         .agg(n=("iv10", "size"), n10=("iv10", "sum"), n30=("iv30", "sum"),
              nboth=("iv_both", "sum")).reset_index())
    t["pct_10"] = t["n10"] / t["n"] * 100
    t["pct_30"] = t["n30"] / t["n"] * 100
    t["pct_both"] = t["nboth"] / t["n"] * 100
    print("\nBy year x decile:")
    print(t[["year", "decile", "n", "pct_10", "pct_30", "pct_both"]].to_string(
        index=False, float_format=lambda v: f"{v:6.2f}"))

    result = {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "vol_surface.csv full scan (secid, date, days, delta, "
                  "impl_volatility, cp_flag), same pass as the STEP 1 "
                  "pct_fixed_tenor_iv table - no second scan",
        "purpose": "prereg_V3.md section 3.3-F feasibility finding and its "
                  "figure - descriptive of IV data availability only, no "
                  "return/RV/regression data",
        "denominator": "V1/V2 universe stock-days, decile 6/7/8, DEV, "
                       "compression_ratio defined",
        "n_universe_stock_days": int(n),
        "overall": overall,
        "by_year_decile": t.to_dict(orient="records"),
        "note_both_equals_10d": bool(
            (t["pct_both"] == t["pct_10"]).all()),
    }
    out_json = project_root / "results" / "47_v3_iv_tenor_coverage.json"
    out_json.write_text(json.dumps(result, indent=2, default=str),
                        encoding="utf-8")
    print(f"\n[OK] Saved {out_json}")
    if result["note_both_equals_10d"]:
        print("NOTE: pct_both == pct_10 in every year x decile cell - 10d "
              "availability is a strict subset of 30d, confirmed structurally "
              "(not an approximation).")
    return result


def scan_for_preaggregated_liquidity(project_root: Path) -> list[Path]:
    """STEP 2a (2026-08-02 extension): search data/processed/ and
    data/raw/optionmetrics/ for anything already carrying option liquidity
    (volume/open_interest/spread/bid/ask) at a general daily stock-level
    panel, as distinct from K1's own narrow trade list. Discovery is by
    column content, not filename, matching src/43's RDQ-source-discovery
    precedent. Read-only; nothing written."""
    print("\n" + "=" * 88)
    print("STEP 2a - SEARCH FOR AN EXISTING PRE-AGGREGATED LIQUIDITY FILE")
    print("=" * 88)

    liquidity_terms = {"volume", "open_interest", "oi", "bid", "ask",
                       "spread", "best_bid", "best_offer", "liq_ok"}
    date_terms = {"date", "dlycaldt", "entry_date", "exit_date"}
    search_dirs = [project_root / "data" / "processed",
                   project_root / "data" / "raw" / "optionmetrics"]

    hits: list[Path] = []
    for d in search_dirs:
        if not d.exists():
            continue
        for f in sorted(list(d.glob("*.parquet")) + list(d.glob("*.csv"))):
            try:
                if f.suffix == ".parquet":
                    import pyarrow.parquet as pq
                    pf = pq.ParquetFile(f)
                    cols = list(pf.schema_arrow.names)
                    nrows = pf.metadata.num_rows
                else:
                    cols = list(pd.read_csv(f, nrows=0).columns)
                    nrows = None
            except Exception as e:
                print(f"  [unreadable] {f.relative_to(project_root)}: {e}")
                continue
            low = {c.lower() for c in cols}
            has_liq = bool(low & liquidity_terms)
            has_date = bool(low & date_terms)
            marker = ("  <-- liquidity-like AND date-like columns present"
                      if (has_liq and has_date) else "")
            nrow_s = f"{nrows:,}" if nrows is not None else "n/a (csv, not counted)"
            print(f"  {str(f.relative_to(project_root)):<52} rows={nrow_s:>18}{marker}")
            print(f"      columns: {cols}")
            if has_liq and has_date:
                hits.append(f)

    print("\nInterpretation:")
    if not hits:
        print("  No file under data/processed/ or data/raw/optionmetrics/ carries "
              "both a liquidity-like and a date-like column.")
    for f in hits:
        if f.name == "k1_trades_real_prices.parquet":
            import pyarrow.parquet as pq
            nrows = pq.ParquetFile(f).metadata.num_rows
            print(f"  {f.name}: {nrows:,} rows -- this IS K1's own trade list. "
                  f"liq_ok/bid/ask columns exist, but only for the specific "
                  f"stock-days K1's T=25% straddle-selection rule picked (one "
                  f"secid per PERMNO on a small set of entry dates), not a "
                  f"general daily panel across the full V1/V2 universe "
                  f"(~1.7M PERMNO-days). NOT usable as a general liquidity "
                  f"coverage source.")
        else:
            print(f"  {f.name}: flagged, needs manual review (unexpected hit).")

    usable = [f for f in hits if f.name != "k1_trades_real_prices.parquet"]
    print(f"\nCONCLUSION: {'no' if not usable else str(len(usable))} usable "
          f"pre-aggregated daily-level option-liquidity panel exists in this "
          f"repo. {'A targeted pull is required.' if not usable else ''}")
    return usable


def estimate_targeted_pull_cost(project_root: Path, opprcd_path: Path) -> None:
    """STEP 2b (2026-08-02 extension): estimate the cost of a liquidity-only
    pull from opprcd without running a full scan. Runs a small, EXPLICITLY
    CAPPED calibration read (a fixed, tiny number of chunks) to measure real
    throughput on this exact file and machine, then extrapolates. This is
    NOT a full scan and does not compute or claim any real coverage number -
    that is deferred pending explicit confirmation of a full run."""
    print("\n" + "=" * 88)
    print("STEP 2b - TARGETED LIQUIDITY-ONLY PULL: COST ESTIMATE (no full scan)")
    print("=" * 88)
    print(f"opprcd source: {opprcd_path}  ({human_size(opprcd_path.stat().st_size)})")

    summary_path = project_root / "data" / "processed" / "k1_trade_summary.json"
    full_rows = None
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        full_rows = s.get("opprcd_whole_file_liquidity_pct", {}).get("rows_scanned")
    if full_rows:
        print(f"\nFull-file row count, read from {summary_path.name} (the field "
              f"src/37 wrote after its own completed full scan of this exact "
              f"file - not re-derived or retyped here): {full_rows:,}")
    else:
        print("\nFull-file row count: unknown (k1_trade_summary.json missing or "
              "missing that field). Cost estimate below will use throughput "
              "and byte-size only, not a row-count extrapolation.")

    print("\nCSV IS ROW-ORIENTED. Selecting fewer columns (usecols) shrinks the "
          "in-memory DataFrame and per-column dtype-cast work, but does NOT let "
          "pandas skip reading bytes from disk - every row must still be read "
          "and split before unwanted columns are discarded. There is no "
          "column-store shortcut for a flat CSV. So a liquidity-only pull's "
          "wall-clock cost is dominated by the SAME full sequential read src/37 "
          "already performed once for this exact file - narrowing to just "
          "volume/open_interest reduces memory risk, not scan time.")

    MINIMAL_COLS = ["secid", "date", "volume", "open_interest"]
    DTYPES = {"secid": "int32", "date": "str", "volume": "float64",
             "open_interest": "float64"}
    CAL_CHUNKS = 3
    CHUNKSIZE = 5_000_000
    cap_rows = CAL_CHUNKS * CHUNKSIZE
    cap_pct = f"{cap_rows / full_rows * 100:.3f}%" if full_rows else "unknown %"
    print(f"\nRunning a CAPPED calibration read: {CAL_CHUNKS} chunks x "
          f"{CHUNKSIZE:,} rows = {cap_rows:,} rows ({cap_pct} of the full "
          f"file), columns={MINIMAL_COLS}. This is NOT a full scan.")

    t0 = time.perf_counter()
    rows_read = 0
    date_ranges = []
    for i, ch in enumerate(pd.read_csv(opprcd_path, usecols=MINIMAL_COLS,
                                       dtype=DTYPES, chunksize=CHUNKSIZE), 1):
        rows_read += len(ch)
        date_ranges.append((ch["date"].min(), ch["date"].max()))
        if i >= CAL_CHUNKS:
            break
    elapsed = time.perf_counter() - t0

    rows_per_sec = rows_read / elapsed if elapsed > 0 else float("nan")
    print(f"\nCalibration result: {rows_read:,} rows read in {elapsed:.1f}s "
          f"({rows_per_sec:,.0f} rows/sec)")
    print(f"Per-chunk date ranges observed: {date_ranges}")
    ascending = all(date_ranges[i][1] <= date_ranges[i + 1][0]
                    for i in range(len(date_ranges) - 1))
    print(f"Dates non-decreasing across these {CAL_CHUNKS} chunks: {ascending} "
          f"(checked on this small slice only, NOT confirmed for the whole "
          f"file - if true throughout, an early-stop-after-2021-12-31 "
          f"optimization might be possible for a DEV-only pull; that would "
          f"need a full-file check to confirm safely, which was not run here)")

    if full_rows and rows_per_sec > 0:
        est_seconds = full_rows / rows_per_sec
        print(f"\nEXTRAPOLATED full-file time estimate (linear projection from "
              f"this {rows_read / full_rows * 100:.3f}% slice):")
        print(f"  {est_seconds / 60:.1f} minutes ({est_seconds / 3600:.2f} hours)")
    print("\nCAVEAT: this is a projection from a small, EARLY-FILE slice, not a "
          "measured full-file run. Disk cache state, OS readahead, and whether "
          "row density is uniform across the file can all make the true "
          "full-file time differ from a linear extrapolation. Treat this as an "
          "order-of-magnitude estimate, not a guarantee.")
    print("\nMemory footprint note: this 4-column read is far smaller than "
          "src/36/src/37's original 10-11 column NEED set (no best_bid/"
          "best_offer/optionid/strike_price/exdate/cp_flag) - this reduces "
          "peak RAM and MemoryError risk, not wall-clock scan time (row-"
          "oriented note above).")
    print("\nNO FULL SCAN WAS RUN. A full pull requires explicit confirmation "
          "before it is executed.")


def full_liquidity_pull(project_root: Path, opprcd_path: Path,
                        keep_secids: set[int]) -> pd.DataFrame:
    """STEP 3 (2026-08-02, authorized): single sequential pass over opprcd
    reading ONLY secid/date/volume/open_interest. Aggregates to one row per
    (secid, date) carrying two flags: did ANY contract on that chain have
    nonzero open interest / nonzero volume that day.

    Rows are filtered to the bridged secids and the DEV date window before
    aggregation - this is a scope restriction to the stock-days V3 actually
    evaluates, not a liquidity screen, and it does not touch which contracts
    count as active.

    Result is cached to parquet; a re-run reuses it and never re-scans."""
    cache = project_root / "data" / "processed" / LIQ_CACHE_NAME
    if cache.exists():
        out = pd.read_parquet(cache)
        print(f"\n[CACHE HIT] {cache.name} exists ({len(out):,} secid-days) - "
              f"reusing, no re-scan of opprcd.")
        return out

    print("\n" + "=" * 88)
    print("STEP 3 - FULL TARGETED LIQUIDITY PULL (AUTHORIZED 2026-08-02)")
    print("=" * 88)
    print(f"source: {opprcd_path}  ({human_size(opprcd_path.stat().st_size)})")
    print(f"columns read: {OPPRCD_LIQ_COLS}  (bid/ask NOT read - not authorized)")
    print(f"scope filter: secid in {len(keep_secids):,} bridged secids, "
          f"date in [{DEV_START.date()}, {DEV_END.date()}]")

    dtypes = {"secid": "int32", "date": "str",
              "volume": "float64", "open_interest": "float64"}
    lo, hi = str(DEV_START.date()), str(DEV_END.date())

    parts: list[pd.DataFrame] = []
    rows_scanned = 0
    rows_kept = 0
    t0 = time.perf_counter()

    for i, ch in enumerate(pd.read_csv(opprcd_path, usecols=OPPRCD_LIQ_COLS,
                                       dtype=dtypes,
                                       chunksize=OPPRCD_CHUNKSIZE), 1):
        rows_scanned += len(ch)
        ch = ch[ch["secid"].isin(keep_secids)]
        if len(ch):
            # ISO-format strings compare lexicographically as dates.
            ch = ch[(ch["date"] >= lo) & (ch["date"] <= hi)]
        if len(ch):
            rows_kept += len(ch)
            g = (ch.assign(_oi=(ch["open_interest"].fillna(0) > 0),
                           _vol=(ch["volume"].fillna(0) > 0))
                 .groupby(["secid", "date"], observed=True, sort=False)
                 .agg(has_oi=("_oi", "max"), has_vol=("_vol", "max"))
                 .reset_index())
            parts.append(g)

        if i % CONSOLIDATE_EVERY == 0 and len(parts) > 1:
            # Fold accumulated partials so a (secid, date) split across chunk
            # boundaries is combined, and memory stays bounded.
            parts = [pd.concat(parts, ignore_index=True)
                     .groupby(["secid", "date"], observed=True, sort=False)
                     .agg(has_oi=("has_oi", "max"), has_vol=("has_vol", "max"))
                     .reset_index()]
        if i % 20 == 0 or i == 1:
            el = time.perf_counter() - t0
            print(f"  chunk {i:>4}: scanned {rows_scanned:>14,}  "
                  f"kept {rows_kept:>13,}  elapsed {el/60:5.1f} min")

    if not parts:
        raise RuntimeError("No opprcd rows survived the secid/date scope filter.")

    out = (pd.concat(parts, ignore_index=True)
           .groupby(["secid", "date"], observed=True, sort=False)
           .agg(has_oi=("has_oi", "max"), has_vol=("has_vol", "max"))
           .reset_index())
    elapsed = time.perf_counter() - t0

    print(f"\nScan complete in {elapsed/60:.1f} min")
    print(f"  rows scanned (whole file): {rows_scanned:,}")
    print(f"  rows kept (scope filter):  {rows_kept:,} "
          f"({rows_kept/rows_scanned*100:.2f}%)")
    print(f"  unique secid-days:         {len(out):,}")
    print(f"  secid-days with any nonzero OI:     {int(out['has_oi'].sum()):,} "
          f"({out['has_oi'].mean()*100:.2f}% of observed chains)")
    print(f"  secid-days with any nonzero volume: {int(out['has_vol'].sum()):,} "
          f"({out['has_vol'].mean()*100:.2f}% of observed chains)")
    print("  NOTE: those two percentages are shares of secid-days that APPEAR "
          "in opprcd at all. They are not universe coverage - a universe "
          "stock-day with no chain whatsoever is absent here and is counted as "
          "not-liquid in the universe table below, which is the correct "
          "denominator for section 6(b).")

    out.to_parquet(cache, index=False)
    print(f"[OK] Cached to {cache} - future runs reuse this, no re-scan.")
    return out


def summarize_liquidity_coverage(project_root: Path, base: pd.DataFrame,
                                 bridge: pd.DataFrame, liq: pd.DataFrame,
                                 fixed_iv: pd.DataFrame) -> dict:
    """Year x decile coverage for nonzero-OI and nonzero-volume, on the same
    denominator and breakdown structure as STEP 1's IV-tenor table, plus a
    threshold menu used only to confirm non-vacuity."""
    print("\n" + "=" * 88)
    print("STEP 3 - LIQUIDITY COVERAGE (2015-2021, SIZE DECILES 6/7/8)")
    print("=" * 88)

    cand = base.merge(bridge, on="PERMNO", how="left")
    cand = cand.merge(liq, left_on=["secid", "date_str"],
                      right_on=["secid", "date"], how="left")
    cand = cand.merge(fixed_iv, left_on=["secid", "date_str"],
                      right_on=["secid", "date"], how="left")
    for c in ["has_oi", "has_vol", "has_fixed_tenor_iv"]:
        cand[c] = cand[c].fillna(False).astype(bool)

    per = (cand.groupby(["PERMNO", "DlyCalDt", "year", "decile"], observed=True)
           .agg(has_oi=("has_oi", "max"), has_vol=("has_vol", "max"),
                has_iv=("has_fixed_tenor_iv", "max"))
           .reset_index())

    stats = (per.groupby(["year", "decile"], observed=True)
             .agg(n_obs=("PERMNO", "size"), n_oi=("has_oi", "sum"),
                  n_vol=("has_vol", "sum"), n_iv=("has_iv", "sum"))
             .reset_index())
    stats["pct_any_nonzero_oi"] = stats["n_oi"] / stats["n_obs"] * 100
    stats["pct_any_nonzero_volume"] = stats["n_vol"] / stats["n_obs"] * 100
    stats["pct_fixed_tenor_iv"] = stats["n_iv"] / stats["n_obs"] * 100

    print("Denominator = every V1/V2 universe stock-day (decile 6/7/8, DEV, "
          "compression defined). A stock-day with no option chain at all "
          "counts as NOT liquid, not as missing.\n")
    shown = stats[["year", "decile", "n_obs", "pct_any_nonzero_oi",
                   "pct_any_nonzero_volume", "pct_fixed_tenor_iv"]].copy()
    for c in ["pct_any_nonzero_oi", "pct_any_nonzero_volume",
              "pct_fixed_tenor_iv"]:
        shown[c] = shown[c].map(lambda v: f"{v:6.2f}")
    print(shown.to_string(index=False))

    # ---- underlying dollar volume, from CRSP (no new pull needed) ----
    crsp = pd.read_parquet(project_root / "data" / "processed" /
                           "crsp_combined.parquet",
                           columns=["PERMNO", "DlyCalDt", "DlyPrcVol"])
    crsp = crsp.drop_duplicates(["PERMNO", "DlyCalDt"], keep="first")
    per = per.merge(crsp, on=["PERMNO", "DlyCalDt"], how="left")
    # Point-in-time cross-sectional rank within the universe that day.
    per["dvol_pct"] = per.groupby("DlyCalDt")["DlyPrcVol"].rank(pct=True)

    print("\n" + "-" * 88)
    print("THRESHOLD MENU - survival share of the universe under candidate "
          "criteria")
    print("-" * 88)
    print("Used ONLY to confirm a threshold is not vacuous (near-0% or "
          "near-100%). No V3 outcome exists yet, so nothing here can be tuned "
          "to a result.\n")

    n = len(per)
    menu = {
        "any nonzero open interest": per["has_oi"],
        "any nonzero volume": per["has_vol"],
        "nonzero OI AND 30d ATM IV available": per["has_oi"] & per["has_iv"],
        "nonzero VOLUME AND 30d ATM IV available": per["has_vol"] & per["has_iv"],
        "nonzero VOLUME AND IV AND dvol>=median": (
            per["has_vol"] & per["has_iv"] & (per["dvol_pct"] >= 0.50)),
        "nonzero VOLUME AND IV AND dvol>=60th pct": (
            per["has_vol"] & per["has_iv"] & (per["dvol_pct"] >= 0.60)),
        "nonzero VOLUME AND IV AND dvol>=75th pct": (
            per["has_vol"] & per["has_iv"] & (per["dvol_pct"] >= 0.75)),
    }
    menu_out = {}
    print(f"  {'criterion':<48} {'n kept':>12} {'% of universe':>14}")
    for label, mask in menu.items():
        k = int(mask.sum())
        menu_out[label] = {"n": k, "pct": k / n * 100}
        print(f"  {label:<48} {k:>12,} {k/n*100:>13.2f}%")

    print("\n  By-year stability of the leading candidate "
          "(nonzero VOLUME AND IV AND dvol>=median):")
    lead = per["has_vol"] & per["has_iv"] & (per["dvol_pct"] >= 0.50)
    by_year = per.assign(_k=lead).groupby("year")["_k"].agg(["sum", "size"])
    by_year["pct"] = by_year["sum"] / by_year["size"] * 100
    for y, r in by_year.iterrows():
        print(f"    {int(y)}: {int(r['sum']):>8,} / {int(r['size']):>8,}  "
              f"{r['pct']:6.2f}%")
    by_dec = per.assign(_k=lead).groupby("decile")["_k"].agg(["sum", "size"])
    by_dec["pct"] = by_dec["sum"] / by_dec["size"] * 100
    print("\n  By-decile:")
    for d, r in by_dec.iterrows():
        print(f"    decile {int(d)}: {int(r['sum']):>8,} / {int(r['size']):>8,}"
              f"  {r['pct']:6.2f}%")

    result = {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "opprcd.csv full targeted pull (secid, date, volume, "
                  "open_interest), authorized 2026-08-02",
        "denominator": "V1/V2 universe stock-days, decile 6/7/8, DEV, "
                       "compression_ratio defined",
        "n_universe_stock_days": int(n),
        "by_year_decile": stats.to_dict(orient="records"),
        "threshold_menu": menu_out,
        "leading_candidate_by_year": {
            int(y): float(r["pct"]) for y, r in by_year.iterrows()},
        "leading_candidate_by_decile": {
            int(d): float(r["pct"]) for d, r in by_dec.iterrows()},
    }
    out_json = project_root / "results" / "47_v3_liquidity_coverage.json"
    out_json.write_text(json.dumps(result, indent=2, default=str),
                        encoding="utf-8")
    print(f"\n[OK] Saved {out_json}")
    return result


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    paths = print_inventory(project_root)

    if not paths["surf"] or not paths["surf"].exists():
        raise SystemExit("STOP: surface IV file missing; cannot run feasibility checks.")
    if not paths["om_names"] or not paths["om_names"].exists():
        raise SystemExit("STOP: om_security_names.csv missing; cannot build PERMNO->secid bridge.")

    base = build_v1_dev_size_universe(project_root)
    bridge = build_permno_secid_bridge(
        project_root, paths["om_names"], base["PERMNO"].unique().tolist()
    )
    print(f"\nPERMNO->secid bridge pairs: {len(bridge):,} (unique PERMNO: {bridge['PERMNO'].nunique():,})")

    any_iv, fixed_iv, tenors, fixed_tenor, valid_tenor_days = scan_surface_atm(paths["surf"])

    # Honor the request to avoid opprcd re-scan in this feasibility pass.
    liquidity_available = False
    _ = summarize_coverage(base, bridge, any_iv, fixed_iv, fixed_tenor, liquidity_available)

    print("\n" + "=" * 88)
    print("NOTES (STEP 1 baseline)")
    print("=" * 88)
    print("1) Standardized/surface IV file present: YES (vol_surface.csv).")
    print(f"2) Surface tenors observed: {tenors}")
    print("3) opprcd chain-liquidity coverage by year/decile is NOT computed in this run")
    print("   because the required daily chain activity was not pre-aggregated and opprcd")
    print("   re-scan was explicitly excluded by request.")
    print("4) No regressions, coefficients, or correlations were computed.")

    # ------------------------------------------------------------------
    # STEP 2 (2026-08-02 extension): close the liquidity gap without a
    # full 80GB opprcd re-scan. Additive only - nothing above this point
    # was modified.
    # ------------------------------------------------------------------
    usable_liquidity_files = scan_for_preaggregated_liquidity(project_root)

    if usable_liquidity_files:
        print("\n[usable pre-aggregated liquidity file(s) found - STOP: "
              "coverage computation from these files is a follow-up step, "
              "not yet implemented, since none were expected to exist.]")
    else:
        opprcd_path = paths["opprcd_repo"] if (paths["opprcd_repo"] and
                                               paths["opprcd_repo"].exists()) \
            else paths["opprcd_staging"]
        if opprcd_path and opprcd_path.exists():
            estimate_targeted_pull_cost(project_root, opprcd_path)
        else:
            print("\nSTOP: opprcd.csv not found in repo or staging location; "
                  "cannot estimate a targeted-pull cost.")

    print("\n" + "=" * 88)
    print("STEP 2 SUMMARY")
    print("=" * 88)
    print("No pre-aggregated daily-level option-liquidity panel exists in this "
          "repo (checked by column content, not filename). A targeted pull from "
          "opprcd is required to compute real liquidity coverage.")

    # ------------------------------------------------------------------
    # STEP 3 (2026-08-02): full targeted pull, AUTHORIZED. Additive only.
    # ------------------------------------------------------------------
    opprcd_path = paths["opprcd_repo"] if (paths["opprcd_repo"] and
                                           paths["opprcd_repo"].exists()) \
        else paths["opprcd_staging"]
    if not (opprcd_path and opprcd_path.exists()):
        print("\nSTOP: opprcd.csv not found in repo or staging; cannot run the "
              "authorized liquidity pull.")
        return

    keep_secids = set(int(s) for s in bridge["secid"].unique())
    liq = full_liquidity_pull(project_root, opprcd_path, keep_secids)
    summarize_liquidity_coverage(project_root, base, bridge, liq, fixed_iv)

    print("\n" + "=" * 88)
    print("STEP 3 COMPLETE - real liquidity coverage computed.")
    print("=" * 88)
    print("results/prereg_V3.md section 6(b) can now be locked on these "
          "numbers. No V3 regression was run and no coefficient was computed "
          "anywhere in this script.")

    # ------------------------------------------------------------------
    # STEP 4 (2026-08-02): persist the 10d/30d/both tenor breakdown that
    # motivated the section 3.3-F horizon decision, as a durable JSON, so
    # the prereg's figure and prose read from a file rather than a
    # transcribed number. No return/RV/regression data touched.
    # ------------------------------------------------------------------
    compute_and_save_tenor_breakdown(project_root, base, bridge, valid_tenor_days)


if __name__ == "__main__":
    main()
