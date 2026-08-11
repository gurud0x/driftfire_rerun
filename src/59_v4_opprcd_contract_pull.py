import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# V4 prereg_V4.md Section 13 - THE AUTHORIZED opprcd CONTRACT-LEVEL PULL.
#
# This is a RAW EXTRACT ONLY, matching Section 13's stated scope exactly:
# "this pull measures the cost of the thresholds locked in Section 6.3. It
# does not authorize revising them." No underlying-close join, no Greeks,
# no C1-C9 funnel, no tie-break selection, no earnings exclusion - none of
# that is "the pull," it is "the build," and the build is explicitly NOT
# started here (owner instruction: report pull completion and wait for
# go-ahead before writing the V4 trading/backtest script).
#
# Scope, exactly as locked in Section 13:
#   columns:   secid, date, exdate, cp_flag, strike_price, best_bid,
#              best_offer, volume, open_interest, impl_volatility,
#              optionid, index_flag, exercise_style  (13 of 14; issuer
#              excluded)
#   filters:   index_flag == 0; secid in the V1/V2 universe bridge
#              whitelist; date in DEV (2015-01-01..2021-12-31);
#              DTE in [8, 90] calendar days at quote date
#
# Output: a durable cached parquet, scanned once, never re-scanned by any
# future phase touching this DTE/liquidity scope - same discipline
# src/47 applied to opprcd_liquidity_daily.parquet.
# ---------------------------------------------------------------------------

DEV_START = pd.Timestamp('2015-01-01')
DEV_END = pd.Timestamp('2021-12-31')
DTE_LO, DTE_HI = 8, 90
OPPRCD_CHUNKSIZE = 5_000_000

project_root = Path(__file__).parent.parent
univ_path = project_root / 'data' / 'processed' / 'universe_membership.parquet'
om_names_path = project_root / 'data' / 'raw' / 'optionmetrics' / 'om_security_names.csv'
crsp_names_path = project_root / 'data' / 'raw' / 'crsp' / 'crsp_names.parquet'
out_path = project_root / 'data' / 'processed' / 'opprcd_v4_contracts_dte8_90.parquet'

repo_om = project_root / 'data' / 'raw' / 'optionmetrics'
staging_om = (Path.home() / 'Downloads' / 'quantdata' / 'driftfire' / 'raw' / 'optionmetrics')
opp_path = next((c for c in [repo_om / 'opprcd.csv', staging_om / 'opprcd.csv'] if c.exists()), None)
if opp_path is None:
    raise SystemExit('STOP: opprcd.csv not found.')

print('=' * 96)
print('prereg_V4.md SECTION 13 - AUTHORIZED opprcd CONTRACT-LEVEL PULL (raw extract only)')
print(f'DTE window [{DTE_LO},{DTE_HI}] calendar days, DEV dates, index_flag==0, secid whitelist')
print('=' * 96)
print(f"\nopprcd source: {opp_path} ({opp_path.stat().st_size / 1e9:.2f} GB)")


def as_cusip8(s):
    return (s.astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            .str.upper().str[:8].str.zfill(8))


# ---- secid <-> PERMNO bridge whitelist (identical construction used throughout) ----
univ = pd.read_parquet(univ_path)
univ_in = univ[univ['in_universe']][['PERMNO', 'year_month', 'decile']].copy()
ever = set(univ_in['PERMNO'].unique())

om = pd.read_csv(om_names_path)
om.columns = [c.lower() for c in om.columns]
om = om.dropna(subset=['secid', 'cusip']).copy()
om['c8'] = as_cusip8(om['cusip'])
om = om[om['c8'].str.len() == 8][['secid', 'c8']].drop_duplicates()

crsp_names = pd.read_parquet(crsp_names_path, columns=['PERMNO', 'CUSIP'])
crsp_names = crsp_names[crsp_names['PERMNO'].isin(ever)].dropna(subset=['CUSIP']).copy()
crsp_names['c8'] = as_cusip8(crsp_names['CUSIP'])
crsp_names = crsp_names[crsp_names['c8'].str.len() == 8][['PERMNO', 'c8']].drop_duplicates()
bridge = crsp_names.merge(om, on='c8', how='inner')[['PERMNO', 'secid']].drop_duplicates()
bridge['secid'] = bridge['secid'].astype('int64')
secid_whitelist = set(bridge['secid'].unique().tolist())
print(f"secid whitelist: {len(secid_whitelist):,} unique secid "
      f"({bridge['PERMNO'].nunique():,} unique PERMNO)")

NEED = ['secid', 'date', 'exdate', 'cp_flag', 'strike_price', 'best_bid', 'best_offer',
        'volume', 'open_interest', 'impl_volatility', 'optionid', 'index_flag',
        'exercise_style']
# cp_flag/exercise_style read as plain str, NOT 'category' - a chunked read
# with dtype='category' builds an independent per-chunk dictionary, and
# pyarrow.ParquetWriter requires a consistent schema across write_table
# calls, so a dictionary mismatch between chunks would crash the run
# partway through. Plain str sidesteps dictionary encoding entirely.
DT = {'secid': 'int32', 'date': 'str', 'exdate': 'str', 'cp_flag': 'str',
      'strike_price': 'float64', 'best_bid': 'float64', 'best_offer': 'float64',
      'volume': 'float64', 'open_interest': 'float64', 'impl_volatility': 'float64',
      'optionid': 'int64', 'index_flag': 'int8', 'exercise_style': 'str'}

DEV_START_S, DEV_END_S = DEV_START.strftime('%Y-%m-%d'), DEV_END.strftime('%Y-%m-%d')
OUT_COLS = ['secid', 'date', 'exdate', 'dte', 'cp_flag', 'strike_price', 'best_bid',
           'best_offer', 'volume', 'open_interest', 'impl_volatility', 'optionid',
           'exercise_style']

# This DTE band [8,90] is far wider than any prior scan in this project (retains
# ~50%+ of rows, vs ~15-30% for the narrower bands used in src/50/53/55/56) - a
# single accumulate-then-concat-then-write blew RAM on the first attempt
# (11.4 GiB allocation failure at 15.8GB total system RAM). Fixed by writing
# incrementally via pyarrow.parquet.ParquetWriter (one row group per surviving
# chunk, never holding more than one chunk's worth of retained rows in memory)
# and downcasting the float64 price/volume columns to float32 (ample precision
# for prices/IV/strikes/volumes, halves the memory and disk footprint).
import pyarrow as pa
import pyarrow.parquet as pq

FLOAT32_COLS = ['strike_price', 'best_bid', 'best_offer', 'volume',
                'open_interest', 'impl_volatility']

t0 = time.time()
rows_scanned = 0
rows_kept = 0
writer = None
n_secid_seen = set()
n_optionid_seen = set()
date_min = date_max = None
dte_min = dte_max = None
exercise_styles_seen = set()

for i, ch in enumerate(pd.read_csv(opp_path, usecols=NEED, dtype=DT,
                                   chunksize=OPPRCD_CHUNKSIZE), 1):
    rows_scanned += len(ch)
    ch = ch[ch['index_flag'] == 0]
    ch = ch[ch['secid'].astype('int64').isin(secid_whitelist)]
    if len(ch) == 0:
        if i == 1 or i % 25 == 0:
            elapsed = time.time() - t0
            print(f"  chunk {i:>4}: scanned {rows_scanned:>13,}  kept {rows_kept:>12,}  "
                  f"({rows_scanned / max(elapsed, 1e-9) / 1e6:.2f}M rows/sec)")
        continue
    ch = ch[(ch['date'] >= DEV_START_S) & (ch['date'] <= DEV_END_S)]
    if len(ch) == 0:
        continue
    ch = ch.copy()
    ch['date'] = pd.to_datetime(ch['date'])
    ch['exdate'] = pd.to_datetime(ch['exdate'])
    ch['dte'] = (ch['exdate'] - ch['date']).dt.days.astype('int16')
    ch = ch[(ch['dte'] >= DTE_LO) & (ch['dte'] <= DTE_HI)]
    if len(ch) == 0:
        continue
    ch['secid'] = ch['secid'].astype('int64')
    for c in FLOAT32_COLS:
        ch[c] = ch[c].astype('float32')
    ch = ch[OUT_COLS]

    table = pa.Table.from_pandas(ch, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(out_path, table.schema, compression='snappy')
    writer.write_table(table)

    rows_kept += len(ch)
    n_secid_seen.update(ch['secid'].unique().tolist())
    # optionid set can grow large but stays far smaller than the row count
    n_optionid_seen.update(ch['optionid'].unique().tolist())
    cmin, cmax = ch['date'].min(), ch['date'].max()
    date_min = cmin if date_min is None else min(date_min, cmin)
    date_max = cmax if date_max is None else max(date_max, cmax)
    dmin, dmax = int(ch['dte'].min()), int(ch['dte'].max())
    dte_min = dmin if dte_min is None else min(dte_min, dmin)
    dte_max = dmax if dte_max is None else max(dte_max, dmax)
    exercise_styles_seen.update(ch['exercise_style'].dropna().unique().tolist())

    if i == 1 or i % 25 == 0:
        elapsed = time.time() - t0
        print(f"  chunk {i:>4}: scanned {rows_scanned:>13,}  kept {rows_kept:>12,}  "
              f"({rows_scanned / max(elapsed, 1e-9) / 1e6:.2f}M rows/sec, "
              f"{elapsed / 60:.1f} min elapsed)")

if writer is not None:
    writer.close()
scan_elapsed = time.time() - t0

print(f"\nScan complete: {rows_scanned:,} rows scanned in {scan_elapsed / 60:.2f} min "
      f"({rows_scanned / scan_elapsed / 1e6:.2f}M rows/sec)")
print(f"Retained (index_flag==0, secid whitelist, DEV dates, DTE in "
      f"[{DTE_LO},{DTE_HI}]): {rows_kept:,} rows")
print(f"unique secid: {len(n_secid_seen):,}   unique optionid: {len(n_optionid_seen):,}")
print(f"date range: {date_min.date() if date_min is not None else None} to "
      f"{date_max.date() if date_max is not None else None}")
print(f"dte range: {dte_min} to {dte_max}")
print(f"exercise_style values: {sorted(exercise_styles_seen)}")

total_elapsed = time.time() - t0
print(f"\n[OK] wrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
print(f"Total wall time (scan + incremental write): {total_elapsed / 60:.2f} min")
print("\nThis is a RAW EXTRACT ONLY - no underlying-close join, no Greeks, no C1-C9")
print("funnel, no tie-break, no earnings exclusion. That is the build, not the pull,")
print("and the build has not been started.")
print("gate_log.md not touched.")
