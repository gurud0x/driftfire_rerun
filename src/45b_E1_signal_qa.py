import pandas as pd
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Phase E1 QA: validation-only checks for e1_signal_panel.parquet.
#
# Read-only script:
# - No modifications to e1_signal_panel.parquet
# - No writes to results/ or gate_log.md
# - Prints a full QA report to stdout
# ---------------------------------------------------------------------------

DEV_START = pd.Timestamp('2015-01-01')
DEV_END = pd.Timestamp('2021-12-31')
EMA_FAST = 8
EMA_SLOW = 21
ATR_WIN = 14
PRIMARY_WIN = 10
PRIMARY_MIN_HITS = 6
RESIDENCY_THRESH = 0.25
RANDOM_SEED = 42
N_SPOTCHECK = 5

project_root = Path(__file__).parent.parent
crsp_path = project_root / 'data' / 'processed' / 'crsp_combined.parquet'
e1_path = project_root / 'data' / 'processed' / 'e1_signal_panel.parquet'
univ_path = project_root / 'data' / 'processed' / 'universe_membership.parquet'
ccm_path = project_root / 'data' / 'raw' / 'compustat' / 'ccm_link_gics.csv'


def pick_col(cols, candidates, label):
    for c in candidates:
        if c in cols:
            return c
    raise SystemExit(
        f"STOP: none of the expected {label} columns found. "
        f"Expected one of {candidates}."
    )


def attach_decile(panel, univ):
    x = panel.copy()
    x['year_month'] = x['DlyCalDt'].dt.to_period('M').astype(str)
    u = univ[['PERMNO', 'year_month', 'decile']].drop_duplicates()
    x = x.merge(u, on=['PERMNO', 'year_month'], how='left')
    return x.drop(columns=['year_month'])


def attach_gsector(panel, ccm):
    # Same dated-link logic used in existing sector scripts.
    lk = ccm.copy()
    lk.columns = [c.lower() for c in lk.columns]
    lk = lk[lk['linktype'].isin(['LC', 'LU'])]
    lk = lk[lk['linkprim'].isin(['P', 'C'])]
    lk = lk[lk['gsector'].notna()]

    lk['linkdt'] = pd.to_datetime(lk['linkdt'], errors='coerce')
    lk['linkend'] = pd.to_datetime(lk['linkenddt'].replace('E', pd.NaT), errors='coerce')
    lk['linkend'] = lk['linkend'].fillna(pd.Timestamp('2262-01-01'))
    lk['lpermno'] = lk['lpermno'].astype(int)
    lk['gsector'] = lk['gsector'].astype(int)

    m = panel.merge(
        lk[['lpermno', 'gsector', 'linkdt', 'linkend', 'linkprim', 'gvkey']],
        left_on='PERMNO',
        right_on='lpermno',
        how='left'
    )
    in_window = (m['DlyCalDt'] >= m['linkdt']) & (m['DlyCalDt'] <= m['linkend'])
    m = m[in_window | m['lpermno'].isna()].copy()

    m['_prim_rank'] = np.where(m['linkprim'] == 'P', 0, 1)
    m = m.sort_values(['PERMNO', 'DlyCalDt', '_prim_rank', 'gvkey'])
    m = m.drop_duplicates(['PERMNO', 'DlyCalDt'], keep='first')

    return m.drop(columns=['lpermno', 'linkdt', 'linkend', '_prim_rank'])


def compute_normalized_body_distance_for_permno(dfp, col_open, col_high, col_low, col_close):
    # Recompute exactly the E1-construction primitives on a single PERMNO series.
    s = dfp.sort_values('DlyCalDt').copy()

    s['ema8'] = s[col_close].ewm(
        span=EMA_FAST,
        adjust=False,
        min_periods=EMA_FAST,
    ).mean().shift(1)
    s['ema21'] = s[col_close].ewm(
        span=EMA_SLOW,
        adjust=False,
        min_periods=EMA_SLOW,
    ).mean().shift(1)

    prev_close = s[col_close].shift(1)
    tr1 = (s[col_high] - s[col_low]).abs()
    tr2 = (s[col_high] - prev_close).abs()
    tr3 = (s[col_low] - prev_close).abs()
    s['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    s['atr14'] = s['tr'].rolling(ATR_WIN, min_periods=ATR_WIN).mean().shift(1)

    open_lag1 = s[col_open].shift(1)
    close_lag1 = s[col_close].shift(1)

    s['ribbon_low'] = np.minimum(s['ema8'], s['ema21'])
    s['ribbon_high'] = np.maximum(s['ema8'], s['ema21'])

    body_low = np.minimum(open_lag1, close_lag1)
    body_high = np.maximum(open_lag1, close_lag1)

    overlaps = (body_low <= s['ribbon_high']) & (body_high >= s['ribbon_low'])
    gap_below = s['ribbon_low'] - body_high
    gap_above = body_low - s['ribbon_high']
    s['body_to_ribbon_distance'] = np.where(
        overlaps,
        0.0,
        np.where(body_high < s['ribbon_low'], gap_below, gap_above)
    )

    s['normalized_body_distance'] = np.where(
        s['atr14'] > 0,
        s['body_to_ribbon_distance'] / s['atr14'],
        np.nan
    )

    s['is_resident'] = np.where(
        s['normalized_body_distance'].notna(),
        (s['normalized_body_distance'] <= RESIDENCY_THRESH).astype(float),
        np.nan
    )
    s['residency_10_hits'] = s['is_resident'].rolling(PRIMARY_WIN, min_periods=PRIMARY_WIN).sum()
    s['primary_cond'] = s['residency_10_hits'] >= PRIMARY_MIN_HITS

    return s


def describe_series(x):
    return {
        'min': x.min(),
        'q25': x.quantile(0.25),
        'median': x.quantile(0.50),
        'q75': x.quantile(0.75),
        'max': x.max(),
    }


print('=' * 88)
print('E1 SIGNAL QA REPORT (READ-ONLY)')
print('=' * 88)

if not e1_path.exists():
    raise SystemExit(
        f"STOP: missing input file {e1_path}. Run the E1 signal construction "
        f"script first, then rerun this QA script."
    )

panel = pd.read_parquet(e1_path)
if panel.empty:
    raise SystemExit('STOP: e1_signal_panel.parquet is empty; no QA possible.')

panel['DlyCalDt'] = pd.to_datetime(panel['DlyCalDt'])
print(f"\nLoaded E1 panel: {len(panel):,} rows; "
      f"{panel['PERMNO'].nunique():,} PERMNOs; "
      f"{panel['DlyCalDt'].min().date()} to {panel['DlyCalDt'].max().date()}")

# --------------------------------------------------------------------------
# 4) Duplicate check
# --------------------------------------------------------------------------
print('\n' + '-' * 88)
print('4) DUPLICATE CHECK (PERMNO, DlyCalDt)')
print('-' * 88)
ndup = int(panel.duplicated(['PERMNO', 'DlyCalDt']).sum())
print(f"Duplicate PERMNO-date rows: {ndup} {'[PASS]' if ndup == 0 else '[BUG]'}")

# --------------------------------------------------------------------------
# Attach decile + sector for concentration sanity checks
# --------------------------------------------------------------------------
univ = pd.read_parquet(univ_path)
univ = univ[univ['in_universe']][['PERMNO', 'year_month', 'decile']]
panel = attach_decile(panel, univ)

if 'gsector' not in panel.columns:
    ccm = pd.read_csv(ccm_path)
    panel = attach_gsector(panel, ccm)

# --------------------------------------------------------------------------
# 1) Episode concentration by year/sector/decile
# --------------------------------------------------------------------------
print('\n' + '-' * 88)
print('1) EPISODE COUNT CONCENTRATION CHECKS')
print('-' * 88)

if 'episode_start' in panel.columns:
    starts = panel[panel['episode_start'] == True].copy()
else:
    starts = panel.copy()

starts['year'] = starts['DlyCalDt'].dt.year

by_year = starts.groupby('year').size().sort_index()
print('\nEpisode count by year:')
for y, n in by_year.items():
    print(f"  {int(y)}: {int(n):,}")  # type: ignore[reportArgumentType]

print('\nEpisode count by gsector (NaN shown as UNMAPPED):')
sec = starts.copy()
sec['gsector_label'] = sec['gsector'].astype('Int64').astype(str).replace('<NA>', 'UNMAPPED')
by_sec = sec.groupby('gsector_label').size().sort_values(ascending=False)
for g, n in by_sec.items():
    print(f"  {g:>8}: {int(n):,}")

print('\nEpisode count by size decile:')
by_dec = starts.groupby('decile').size().sort_index()
for d, n in by_dec.items():
    dlab = 'UNMAPPED' if pd.isna(d) else f"{int(d)}"  # type: ignore[reportArgumentType]
    print(f"  decile {dlab:>2}: {int(n):,}")

if len(starts):
    year_share = by_year.max() / len(starts)
    sec_share = by_sec.max() / len(starts)
    dec_share = by_dec.max() / len(starts)
    print('\nConcentration heuristics (max bucket share):')
    print(f"  Year max share:   {year_share:.2%}")
    print(f"  Sector max share: {sec_share:.2%}")
    print(f"  Decile max share: {dec_share:.2%}")
    if sec_share > 0.50 or dec_share > 0.50:
        print('  FLAG: heavy concentration in a single sector/decile bucket.')

# --------------------------------------------------------------------------
# 2) Episode duration determinability
# --------------------------------------------------------------------------
print('\n' + '-' * 88)
print('2) EPISODE DURATION')
print('-' * 88)
print('Duration is NOT determinable from episode-start rows alone.')
print('Reason: end-of-episode timestamps are absent in e1_signal_panel.parquet.')
print('To compute durations, a full daily panel containing the ongoing')
print('primary residency condition (or daily in-episode flag) is required.')

# --------------------------------------------------------------------------
# 5) Date-range check vs DEV + warmup buffer
# --------------------------------------------------------------------------
print('\n' + '-' * 88)
print('5) DATE-RANGE CHECK (DEV + warmup buffer)')
print('-' * 88)

crsp_dates = pd.read_parquet(crsp_path, columns=['DlyCalDt'])['DlyCalDt'].drop_duplicates().sort_values().reset_index(drop=True)
crsp_dates = pd.to_datetime(crsp_dates)

# Warmup sessions implied by feature definitions:
# ATR14 with shift(1) and residency-10 over aligned rows requires at most
# roughly ATR_WIN + PRIMARY_WIN + 1 prior sessions.
warmup_sessions = ATR_WIN + PRIMARY_WIN + 1
before_dev = crsp_dates[crsp_dates < DEV_START]  # type: ignore[reportIndexIssue]
if len(before_dev) >= warmup_sessions:
    warmup_start = before_dev.iloc[-warmup_sessions]
elif len(before_dev) > 0:
    warmup_start = before_dev.iloc[0]
else:
    warmup_start = DEV_START

allowed_start = warmup_start
allowed_end = DEV_END

outside = starts[(starts['DlyCalDt'] < allowed_start) | (starts['DlyCalDt'] > allowed_end)]
print(f"Allowed range: {allowed_start.date()} to {allowed_end.date()}")
print(f"Rows outside allowed range: {len(outside):,} {'[PASS]' if len(outside) == 0 else '[BUG]'}")
if len(outside):
    print('First 10 offending rows:')
    show = outside[['PERMNO', 'DlyCalDt']].sort_values(['DlyCalDt', 'PERMNO']).head(10)
    for r in show.itertuples(index=False):
        off_date = pd.Timestamp(r.DlyCalDt)  # type: ignore[reportArgumentType]
        print(f"  PERMNO {int(r.PERMNO)}  date {off_date.date()}")  # type: ignore[reportArgumentType]

# --------------------------------------------------------------------------
# 6) Distribution checks for ribbon_width and normalized_body_distance
# --------------------------------------------------------------------------
print('\n' + '-' * 88)
print('6) DISTRIBUTION CHECKS (episode-start rows)')
print('-' * 88)

for c in ['ribbon_width', 'normalized_body_distance']:
    z = starts[c].dropna()
    if z.empty:
        print(f"{c}: all null")
        continue
    s = describe_series(z)
    print(f"\n{c}:")
    print(f"  min={s['min']:.6f}  q25={s['q25']:.6f}  median={s['median']:.6f}  "
          f"q75={s['q75']:.6f}  max={s['max']:.6f}")

# Basic unit-error heuristics
rw = starts['ribbon_width'].dropna()
if not rw.empty:
    if rw.max() > 1000 or rw.quantile(0.99) > 100:
        print('  FLAG: ribbon_width has extreme scale, possible ATR normalization bug.')
    elif rw.max() > 50:
        print('  WARNING: ribbon_width upper tail is very large; verify ATR units.')

nbd = starts['normalized_body_distance'].dropna()
if not nbd.empty:
    if nbd.max() > 1000 or nbd.quantile(0.99) > 100:
        print('  FLAG: normalized_body_distance has extreme scale, possible unit bug.')
    elif nbd.max() > 10:
        print('  WARNING: normalized_body_distance upper tail is very large; inspect scaling.')

# --------------------------------------------------------------------------
# 3) Spot-check 5 random episode starts against CRSP lookback math
# --------------------------------------------------------------------------
print('\n' + '-' * 88)
print('3) RANDOM SPOT-CHECKS (5 rows): row-level value + prior-10-session count')
print('-' * 88)

# Detect OHLC names using a tiny read and then load needed columns.
schema_cols = pq.ParquetFile(crsp_path).schema_arrow.names
col_open = pick_col(schema_cols, ['DlyOpen', 'DlyOpenBidAsk'], 'open')
col_high = pick_col(schema_cols, ['DlyHigh', 'DlyHighBidAsk'], 'high')
col_low = pick_col(schema_cols, ['DlyLow', 'DlyLowBidAsk'], 'low')
col_close = pick_col(schema_cols, ['DlyClose', 'DlyCloseBidAsk'], 'close')

need = ['PERMNO', 'DlyCalDt', col_open, col_high, col_low, col_close]
crsp = pd.read_parquet(crsp_path, columns=need)
crsp = crsp.drop_duplicates(['PERMNO', 'DlyCalDt'], keep='first')
crsp = crsp.sort_values(['PERMNO', 'DlyCalDt'])

if len(starts) < N_SPOTCHECK:
    picks = starts.copy()
else:
    picks = starts.sample(n=N_SPOTCHECK, random_state=RANDOM_SEED)

for i, row in enumerate(picks.itertuples(index=False), start=1):
    p = int(row.PERMNO)  # type: ignore[reportArgumentType]
    d = pd.Timestamp(row.DlyCalDt)  # type: ignore[reportArgumentType]

    hist = crsp[crsp['PERMNO'] == p].copy().reset_index(drop=True)
    rec = compute_normalized_body_distance_for_permno(hist, col_open, col_high, col_low, col_close)

    pos = np.flatnonzero(rec['DlyCalDt'].to_numpy() == np.datetime64(d))
    if len(pos) == 0:
        print(f"\n[{i}] PERMNO {p}  date {d.date()}  -> NOT FOUND in CRSP subset")
        continue

    j = int(pos[0])
    lo = max(0, j - (PRIMARY_WIN - 1))
    win = rec.iloc[lo:j + 1].copy()

    vals = win['normalized_body_distance'].to_numpy()
    quals = (vals <= RESIDENCY_THRESH) & ~pd.isna(vals)
    count_qual = int(np.nansum(quals))

    row_nbd = float(row.normalized_body_distance) if pd.notna(row.normalized_body_distance) else np.nan  # type: ignore[reportArgumentType]
    calc_nbd = rec.loc[j, 'normalized_body_distance']
    panel_hits = (float(row.residency_10_hits)  # type: ignore[reportArgumentType]
                  if hasattr(row, 'residency_10_hits') and pd.notna(row.residency_10_hits)
                  else np.nan)
    panel_episode_start = (bool(row.episode_start)
                           if hasattr(row, 'episode_start') and pd.notna(row.episode_start)
                           else None)
    today_raw = rec.at[j, 'primary_cond']
    prev_raw = rec.at[j - 1, 'primary_cond'] if j > 0 else np.nan
    rec_primary_today = bool(today_raw) if pd.notna(today_raw) else False  # type: ignore[reportGeneralTypeIssues]
    rec_primary_prev = bool(prev_raw) if pd.notna(prev_raw) else False
    rec_episode_start = rec_primary_today and (not rec_primary_prev)

    print(f"\n[{i}] PERMNO {p}  episode_start date {d.date()}")
    print(f"  panel normalized_body_distance: {row_nbd:.6f}")
    print(f"  recomputed normalized_body_distance: {calc_nbd:.6f}")
    print(f"  row check (<= 0.25): {bool(calc_nbd <= RESIDENCY_THRESH)}")
    print(f"  panel residency_10_hits: {panel_hits:.1f}")
    print(f"  prior-10-session qualifying count: {count_qual} "
          f"(threshold >= {PRIMARY_MIN_HITS})")
    print(f"  panel episode_start: {panel_episode_start}")
    print(f"  recomputed episode_start: {rec_episode_start}")
    print(f"  consistency: {panel_episode_start == rec_episode_start}")
    print('  window values (oldest -> newest):')
    print('   ' + ' '.join('nan' if pd.isna(v) else f"{v:.4f}" for v in vals))  # type: ignore[reportGeneralTypeIssues]

print('\n' + '=' * 88)
print('END OF E1 QA REPORT')
print('=' * 88)
