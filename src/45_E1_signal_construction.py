import pandas as pd
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Phase E1: signal construction only.
#
# This script computes consolidation/ribbon features using CRSP daily data,
# applies strict no-lookahead alignment via shift(1), detects non-overlapping
# episode starts, and joins the existing pre-lagged compression score as-is.
#
# No hypothesis tests, no forward returns, no regressions, and no writes to
# results/ or gate logs are performed here.
# ---------------------------------------------------------------------------

EMA_FAST = 8
EMA_SLOW = 21
ATR_WIN = 14
RESIDENCY_THRESH = 0.25
RESIDENCY_THRESH_GRID = {
    'thr0': 0.0,
    'thr025': 0.25,
    'thr05': 0.5,
}
RES_WINDOWS = [5, 10, 15, 20, 25]
PRIMARY_WIN = 10
PRIMARY_MIN_HITS = 6
DEV_START = pd.Timestamp('2015-01-01')
DEV_END = pd.Timestamp('2021-12-31')

project_root = Path(__file__).parent.parent
crsp_path = project_root / 'data' / 'processed' / 'crsp_combined.parquet'
univ_path = project_root / 'data' / 'processed' / 'universe_membership.parquet'
comp_path = project_root / 'data' / 'processed' / 'compression_signal_v1.parquet'
sector_v2_path = project_root / 'data' / 'processed' / 'sector_compression_signal_v2.parquet'
out_path = project_root / 'data' / 'processed' / 'e1_signal_panel.parquet'

print('=' * 80)
print('E1 SIGNAL CONSTRUCTION: ribbon/residency episodes with sensitivity grid')
print('=' * 80)


def pick_col(cols, candidates, label):
    for c in candidates:
        if c in cols:
            return c
    raise SystemExit(
        f"STOP: none of the expected {label} columns were found. "
        f"Expected one of {candidates}. Available columns include: {list(cols)}"
    )


# --------------------------------------------------------------------------
# Load universe and CRSP rows for ever-in-universe PERMNOs
# --------------------------------------------------------------------------
univ = pd.read_parquet(univ_path)
univ = univ[univ['in_universe']][['PERMNO', 'year_month']]
print(f"\n[OK] Universe membership rows: {len(univ):,}; "
      f"PERMNOs: {univ['PERMNO'].nunique():,}")

cols = pq.ParquetFile(crsp_path).schema_arrow.names

col_open = pick_col(cols, ['DlyOpen', 'DlyOpenBidAsk'], 'open')
col_high = pick_col(cols, ['DlyHigh', 'DlyHighBidAsk'], 'high')
col_low = pick_col(cols, ['DlyLow', 'DlyLowBidAsk'], 'low')
col_close = pick_col(cols, ['DlyClose', 'DlyCloseBidAsk'], 'close')
col_vol = pick_col(cols, ['DlyVol', 'DlyPrcVol'], 'volume')
col_cap = pick_col(cols, ['DlyCap'], 'market cap')

need = ['PERMNO', 'DlyCalDt', col_open, col_high, col_low, col_close, col_vol, col_cap]
missing = [c for c in need if c not in cols]
if missing:
    raise SystemExit(f"STOP: required columns missing: {missing}")

# Reload with only needed columns to reduce memory.
df = pd.read_parquet(crsp_path, columns=need)
df = df[df['PERMNO'].isin(univ['PERMNO'].unique())].copy()
print(f"[OK] Loaded CRSP daily rows for ever-in-universe names: {len(df):,}")
print(f"  Price fields used: open={col_open}, high={col_high}, "
      f"low={col_low}, close={col_close}")
print(f"  Volume field used: {col_vol}")
print(f"  Market-cap field used: {col_cap}")

# Same duplicate-day handling style as existing signal scripts.
dup_mask = df.duplicated(['PERMNO', 'DlyCalDt'], keep=False)
if dup_mask.any():
    check_cols = [col_open, col_high, col_low, col_close]
    conflicting = (df[dup_mask].groupby(['PERMNO', 'DlyCalDt'])[check_cols]
                   .nunique().gt(1).any(axis=1).sum())
    print(f"\nDuplicate PERMNO-days found: "
          f"{df[dup_mask][['PERMNO', 'DlyCalDt']].drop_duplicates().shape[0]:,}; "
          f"conflicting price groups: {int(conflicting)} "
          f"{'[PASS - safe to dedupe]' if conflicting == 0 else '[FAIL - halting]'}")
    if conflicting != 0:
        raise SystemExit(1)
    before = len(df)
    df = df.drop_duplicates(['PERMNO', 'DlyCalDt'], keep='first')
    print(f"  Dropped {before - len(df):,} duplicate rows")

df = df.sort_values(['PERMNO', 'DlyCalDt']).reset_index(drop=True)
sorted_ok = df.groupby('PERMNO')['DlyCalDt'].is_monotonic_increasing.all()
print(f"\nSorted within PERMNO: {sorted_ok} "
      f"{'[PASS]' if sorted_ok else '[FAIL - halting]'}")
if not sorted_ok:
    raise SystemExit(1)

grp = df.groupby('PERMNO', sort=False)

# --------------------------------------------------------------------------
# Lag-aligned feature primitives (shift(1) discipline)
# --------------------------------------------------------------------------
# EMAs based on close history ending t-1, with explicit warmup enforcement.
# min_periods avoids unconverged values in early rows.
df['ema8'] = grp[col_close].transform(
    lambda s: s.ewm(span=EMA_FAST, adjust=False, min_periods=EMA_FAST).mean().shift(1)
)
df['ema21'] = grp[col_close].transform(
    lambda s: s.ewm(span=EMA_SLOW, adjust=False, min_periods=EMA_SLOW).mean().shift(1)
)
df['ema8_lag1'] = grp['ema8'].shift(1)
df['ema8_slope'] = df['ema8'] - df['ema8_lag1']

# ATR(14) from true range, then shifted so day t uses up-to-t-1 info only.
prev_close = grp[col_close].shift(1)
tr1 = (df[col_high] - df[col_low]).abs()
tr2 = (df[col_high] - prev_close).abs()
tr3 = (df[col_low] - prev_close).abs()
tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
df['_tr'] = tr
df['atr14'] = grp['_tr'].transform(
    lambda s: s.rolling(ATR_WIN, min_periods=ATR_WIN).mean().shift(1)
)
df = df.drop(columns=['_tr'])

# Use lagged OHLC for body/close geometry to keep full shift(1) alignment.
open_lag1 = grp[col_open].shift(1)
close_lag1 = grp[col_close].shift(1)

# --------------------------------------------------------------------------
# Ribbon, body interval, distances, and normalized quantities
# --------------------------------------------------------------------------
df['ribbon_low'] = np.minimum(df['ema8'], df['ema21'])
df['ribbon_high'] = np.maximum(df['ema8'], df['ema21'])

df['body_low'] = np.minimum(open_lag1, close_lag1)
df['body_high'] = np.maximum(open_lag1, close_lag1)

body_overlaps = (
    (df['body_low'] <= df['ribbon_high']) &
    (df['body_high'] >= df['ribbon_low'])
)
body_gap_below = df['ribbon_low'] - df['body_high']
body_gap_above = df['body_low'] - df['ribbon_high']
df['body_to_ribbon_distance'] = np.where(
    body_overlaps,
    0.0,
    np.where(df['body_high'] < df['ribbon_low'], body_gap_below, body_gap_above)
)

df['normalized_body_distance'] = np.where(
    df['atr14'] > 0,
    df['body_to_ribbon_distance'] / df['atr14'],
    np.nan
)

close_overlaps = (close_lag1 >= df['ribbon_low']) & (close_lag1 <= df['ribbon_high'])
close_gap_below = df['ribbon_low'] - close_lag1
close_gap_above = close_lag1 - df['ribbon_high']
df['close_to_ribbon_distance'] = np.where(
    close_overlaps,
    0.0,
    np.where(close_lag1 < df['ribbon_low'], close_gap_below, close_gap_above)
)
df['close_to_ribbon_distance_norm'] = np.where(
    df['atr14'] > 0,
    df['close_to_ribbon_distance'] / df['atr14'],
    np.nan
)

df['ribbon_width'] = np.where(
    df['atr14'] > 0,
    (df['ema8'] - df['ema21']).abs() / df['atr14'],
    np.nan
)
df['daily_range'] = (df[col_high] - df[col_low]).abs()
df['episode_high'] = df[col_high]

# --------------------------------------------------------------------------
# Residency scores and primary condition (>= 6 of last 10 sessions)
# --------------------------------------------------------------------------
for tag, thr in RESIDENCY_THRESH_GRID.items():
    resident_col = f'is_resident_{tag}'
    df[resident_col] = np.where(
        df['normalized_body_distance'].notna(),
        (df['normalized_body_distance'] <= thr).astype(float),
        np.nan
    )

    for w in RES_WINDOWS:
        hits_col = f'residency_{w}_{tag}_hits'
        score_col = f'residency_{w}_{tag}'
        df[hits_col] = df.groupby('PERMNO', sort=False)[resident_col].transform(
            lambda s, w=w: s.rolling(w, min_periods=w).sum()
        )
        df[score_col] = df[hits_col] / w

    primary_hits_col = f'primary_hits_{tag}'
    primary_cond_col = f'primary_residency_condition_{tag}'
    episode_start_col = f'episode_start_{tag}'
    episode_seq_col = f'episode_seq_{tag}'

    df[primary_hits_col] = df[f'residency_{PRIMARY_WIN}_{tag}_hits']
    df[primary_cond_col] = df[primary_hits_col] >= PRIMARY_MIN_HITS

    prev_cond = df.groupby('PERMNO', sort=False)[primary_cond_col].shift(1).fillna(False)
    prev_cond = prev_cond.astype(bool)
    df[episode_start_col] = df[primary_cond_col] & (~prev_cond)
    df[episode_seq_col] = df.groupby('PERMNO', sort=False)[episode_start_col].cumsum().astype(int)

# Preserve unsuffixed columns for the default threshold (0.25).
df['is_resident'] = df['is_resident_thr025']
for w in RES_WINDOWS:
    df[f'residency_{w}'] = df[f'residency_{w}_thr025']
df['residency_10_hits'] = df['residency_10_thr025_hits']
df['primary_residency_condition'] = df['primary_residency_condition_thr025']
df['episode_start'] = df['episode_start_thr025']
df['episode_seq'] = df['episode_seq_thr025']

episode_start_cols = [f'episode_start_{tag}' for tag in RESIDENCY_THRESH_GRID]
df['episode_start_any'] = df[episode_start_cols].fillna(False).any(axis=1)

# --------------------------------------------------------------------------
# Filter to point-in-time universe and join existing compression score as-is
# --------------------------------------------------------------------------
df['year_month'] = df['DlyCalDt'].dt.to_period('M').astype(str)
panel = df.merge(univ, on=['PERMNO', 'year_month'], how='inner')

comp = pd.read_parquet(comp_path, columns=['PERMNO', 'DlyCalDt', 'compression_ratio'])
panel = panel.merge(comp, on=['PERMNO', 'DlyCalDt'], how='left')

sector = pd.read_parquet(sector_v2_path, columns=['PERMNO', 'DlyCalDt', 'gsector'])
panel = panel.merge(sector, on=['PERMNO', 'DlyCalDt'], how='left')

# Restrict to the DEV window; warmup remains reflected in early-feature nulls.
panel = panel[(panel['DlyCalDt'] >= DEV_START) & (panel['DlyCalDt'] <= DEV_END)].copy()

# Emit episode starts across the configured threshold grid.
out = panel[panel['episode_start_any']].copy()

out_cols = [
    'PERMNO',
    'DlyCalDt',
    col_open,
    col_high,
    col_low,
    col_close,
    col_vol,
    col_cap,
    'gsector',
    'daily_range',
    'episode_high',
    'ema8',
    'ema8_lag1',
    'ema8_slope',
    'ema21',
    'atr14',
    'ribbon_low',
    'ribbon_high',
    'body_low',
    'body_high',
    'body_to_ribbon_distance',
    'normalized_body_distance',
    'close_to_ribbon_distance',
    'close_to_ribbon_distance_norm',
    'ribbon_width',
    'residency_5',
    'residency_10',
    'residency_15',
    'residency_20',
    'residency_25',
    'residency_10_hits',
    'primary_residency_condition',
    'episode_start',
    'episode_seq',
    'episode_start_any',
    'is_resident_thr0',
    'is_resident_thr025',
    'is_resident_thr05',
    'residency_10_thr0',
    'residency_10_thr025',
    'residency_10_thr05',
    'primary_hits_thr0',
    'primary_hits_thr025',
    'primary_hits_thr05',
    'primary_residency_condition_thr0',
    'primary_residency_condition_thr025',
    'primary_residency_condition_thr05',
    'episode_start_thr0',
    'episode_start_thr025',
    'episode_start_thr05',
    'episode_seq_thr0',
    'episode_seq_thr025',
    'episode_seq_thr05',
    'compression_ratio',
]
out = out[out_cols]

# --------------------------------------------------------------------------
# Save + required data-availability report
# --------------------------------------------------------------------------
out.to_parquet(out_path, index=False)

print('\n' + '-' * 80)
print('DATA AVAILABILITY REPORT')
print('-' * 80)
print(f"Rows: {len(out):,}")
print(f"PERMNO count: {out['PERMNO'].nunique():,}")
if len(out) > 0:
    print(f"Date range: {out['DlyCalDt'].min().date()} to {out['DlyCalDt'].max().date()}")
else:
    print('Date range: <empty output>')

new_cols = [
    col_open, col_high, col_low, col_close, col_vol, col_cap,
    'gsector', 'daily_range', 'episode_high',
    'ema8', 'ema8_lag1', 'ema8_slope', 'ema21', 'atr14', 'ribbon_low', 'ribbon_high',
    'body_low', 'body_high', 'body_to_ribbon_distance',
    'normalized_body_distance', 'close_to_ribbon_distance',
    'close_to_ribbon_distance_norm', 'ribbon_width',
    'residency_5', 'residency_10', 'residency_15', 'residency_20', 'residency_25',
    'residency_10_hits', 'primary_residency_condition',
    'episode_start', 'episode_seq', 'episode_start_any',
    'is_resident_thr0', 'is_resident_thr025', 'is_resident_thr05',
    'residency_10_thr0', 'residency_10_thr025', 'residency_10_thr05',
    'primary_hits_thr0', 'primary_hits_thr025', 'primary_hits_thr05',
    'primary_residency_condition_thr0',
    'primary_residency_condition_thr025',
    'primary_residency_condition_thr05',
    'episode_start_thr0', 'episode_start_thr025', 'episode_start_thr05',
    'episode_seq_thr0', 'episode_seq_thr025', 'episode_seq_thr05',
    'compression_ratio'
]

print('\nNull rate by new column:')
for c in new_cols:
    null_rate = out[c].isna().mean() * 100 if len(out) else np.nan
    if pd.isna(null_rate):
        print(f"  {c}: n/a (empty output)")
    else:
        print(f"  {c}: {null_rate:.2f}%")

print('\nEpisode counts by threshold:')
print(f"  threshold 0.00: {int(out['episode_start_thr0'].sum()):,}")
print(f"  threshold 0.25: {int(out['episode_start_thr025'].sum()):,}")
print(f"  threshold 0.50: {int(out['episode_start_thr05'].sum()):,}")
print(f"  union (episode_start_any): {int(out['episode_start_any'].sum()):,}")

print('\nEMA slope availability choice: wrote ema8 and ema8_lag1 explicitly;')
print('downstream slope can be computed as ema8 - ema8_lag1 without')
print('reloading full daily history.')

print(f"\n[OK] Saved {out_path}  shape={out.shape}")

print('\n' + '=' * 80)
print('E1 SIGNAL CONSTRUCTION COMPLETE')
print('=' * 80)
