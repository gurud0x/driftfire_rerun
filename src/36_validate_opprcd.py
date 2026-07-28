import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# K1 rebuild, Step 1: validate the OptionMetrics contract-level price file
# (opprcd). INSPECTION ONLY - nothing written to data/processed/.
#
# This file is what K1 actually needed all along: real quoted bid/ask per
# listed contract, so entry and exit can be priced from observed quotes
# rather than reconstructed from a constant-maturity surface (the flaw that
# made src/30's +185%/yr and src/32's -490%/yr both uninterpretable).
#
# SOURCE LOCATION: like the full-grid surface, this 80.5 GB file stays in
# the download staging folder; copying it into the OneDrive-synced repo
# would trigger an 80 GB cloud sync for no research benefit. Repo path is
# checked first, staging second, and the resolved path is printed.
#
# Conventions follow src/27/28: chunked read, explicit dtypes, ASCII only.
# ---------------------------------------------------------------------------

CHUNKSIZE = 5_000_000

project_root = Path(__file__).parent.parent
repo_om = project_root / 'data' / 'raw' / 'optionmetrics'
staging_om = (Path.home() / 'Downloads' / 'quantdata' / 'driftfire' / 'raw' /
              'optionmetrics')
list_path = repo_om / 'secid_list.txt'

src = None
for c in [repo_om / 'opprcd.csv', staging_om / 'opprcd.csv']:
    if c.exists():
        src = c
        break
if src is None:
    print("STOP: opprcd.csv not found in either location:")
    print(f"   {repo_om / 'opprcd.csv'}")
    print(f"   {staging_om / 'opprcd.csv'}")
    raise SystemExit(1)

print("=" * 78)
print("VALIDATION: OptionMetrics contract prices (opprcd.csv)")
print("=" * 78)
print(f"\nSource: {src}")
print(f"  size: {src.stat().st_size / 1e9:.1f} GB")

peek = pd.read_csv(src, nrows=5)
print("\n### Columns as pulled:")
for c in peek.columns:
    print("   " + c)

NEED = ['secid', 'date', 'exdate', 'cp_flag', 'strike_price', 'best_bid',
        'best_offer', 'volume', 'open_interest', 'optionid', 'index_flag']
missing = [c for c in NEED if c not in peek.columns]
if missing:
    print(f"STOP: required columns missing: {missing}")
    raise SystemExit(1)

DT = {'secid': 'int32', 'date': 'str', 'exdate': 'str', 'cp_flag': 'category',
      'strike_price': 'float64', 'best_bid': 'float64',
      'best_offer': 'float64', 'volume': 'float64',
      'open_interest': 'float64', 'optionid': 'int64', 'index_flag': 'int8'}

print(f"\nChunked read: chunksize={CHUNKSIZE:,}, "
      f"{len(NEED)} of {len(peek.columns)} columns loaded.")
print("NOTE: strike_price is in OptionMetrics' x1000 convention "
      "(17500 = $17.50).")

rows = 0
secids = set()
dmin = dmax = None
nulls = {c: 0 for c in NEED}
n_zero_liq = 0          # volume == 0 AND open_interest == 0
n_vol0 = 0
n_oi0 = 0
n_bid_gt_offer = 0
n_bid_zero = 0
n_offer_zero = 0
dte_ct = {}
idx_ct = {}
cp_ct = {}
examples = None

for i, ch in enumerate(pd.read_csv(src, usecols=NEED, dtype=DT,
                                   chunksize=CHUNKSIZE), 1):
    rows += len(ch)
    secids.update(ch['secid'].unique().tolist())
    cmin, cmax = ch['date'].min(), ch['date'].max()
    dmin = cmin if dmin is None else min(dmin, cmin)
    dmax = cmax if dmax is None else max(dmax, cmax)
    for c in NEED:
        nulls[c] += int(ch[c].isna().sum())

    v0 = ch['volume'].fillna(0) == 0
    o0 = ch['open_interest'].fillna(0) == 0
    n_vol0 += int(v0.sum())
    n_oi0 += int(o0.sum())
    n_zero_liq += int((v0 & o0).sum())

    both = ch['best_bid'].notna() & ch['best_offer'].notna()
    n_bid_gt_offer += int((both & (ch['best_bid'] > ch['best_offer'])).sum())
    n_bid_zero += int((ch['best_bid'].fillna(-1) == 0).sum())
    n_offer_zero += int((ch['best_offer'].fillna(-1) == 0).sum())

    # days to expiration (cache=True: few unique dates per chunk, cheap)
    d = pd.to_datetime(ch['date'], cache=True)
    e = pd.to_datetime(ch['exdate'], cache=True)
    dte = (e - d).dt.days
    for k, v in dte.value_counts().items():
        dte_ct[int(k)] = dte_ct.get(int(k), 0) + int(v)
    for k, v in ch['index_flag'].value_counts().items():
        idx_ct[int(k)] = idx_ct.get(int(k), 0) + int(v)
    for k, v in ch['cp_flag'].value_counts().items():
        cp_ct[str(k)] = cp_ct.get(str(k), 0) + int(v)

    if examples is None:
        examples = ch.head(5).copy()
    if i % 10 == 0 or i == 1:
        print(f"  chunk {i:>4}: {rows:>15,} rows scanned")

print(f"\n### Shape and scope")
print(f"Rows: {rows:,}")
print(f"Unique secid: {len(secids):,}")
print(f"Date range: {dmin} to {dmax}")

if list_path.exists():
    expected = {int(x) for x in list_path.read_text().split()}
    inside = len(secids & expected)
    print(f"secid_list.txt scope: {len(expected):,}; "
          f"opprcd secids inside that scope: {inside:,} of {len(secids):,} "
          f"{'[PASS]' if inside == len(secids) else '[WARNING - secids outside the requested list]'}")

print(f"\n### Null counts")
for c in NEED:
    print(f"   {c:>15}: {nulls[c]:>15,} ({nulls[c]/rows*100:6.2f}%)")

print(f"\n### Liquidity")
print(f"   volume == 0:                    {n_vol0:>15,} "
      f"({n_vol0/rows*100:6.2f}%)")
print(f"   open_interest == 0:             {n_oi0:>15,} "
      f"({n_oi0/rows*100:6.2f}%)")
print(f"   BOTH zero (stale/illiquid):     {n_zero_liq:>15,} "
      f"({n_zero_liq/rows*100:6.2f}%)")
print("   Disclosed, not dropped here. src/37 reports a liquidity-filtered")
print("   variant (volume>0 or open_interest>0) alongside the unrestricted.")

print(f"\n### Quote validity")
print(f"   best_bid > best_offer (invalid): {n_bid_gt_offer:>14,} "
      f"({n_bid_gt_offer/rows*100:6.2f}%) "
      f"{'[PASS]' if n_bid_gt_offer == 0 else '[WARNING]'}")
print(f"   best_bid == 0 (no bid side):     {n_bid_zero:>14,} "
      f"({n_bid_zero/rows*100:6.2f}%)")
print(f"   best_offer == 0:                 {n_offer_zero:>14,} "
      f"({n_offer_zero/rows*100:6.2f}%)")
print("   A zero bid matters for the BID-based exit scenario in src/37:")
print("   selling into a zero bid realizes a total loss on that leg.")

print(f"\n### Days to expiration (filter cross-check)")
tot = sum(dte_ct.values())
lo, hi = min(dte_ct), max(dte_ct)
print(f"   observed DTE range: {lo} to {hi}")
common = sorted(dte_ct.items(), key=lambda kv: -kv[1])[:12]
print(f"   most common DTE values (count, share):")
for k, v in common:
    print(f"     {k:>4} days: {v:>14,} ({v/tot*100:5.2f}%)")
in_band = sum(v for k, v in dte_ct.items() if 20 <= k <= 45)
print(f"   DTE in [20, 45] (K1's 30-day tenor neighbourhood): "
      f"{in_band:,} ({in_band/tot*100:.2f}%)")

print(f"\n### Instrument type")
print(f"   index_flag counts: {idx_ct}  (0 = equity option)")
eq = idx_ct.get(0, 0)
print(f"   equity share: {eq/rows*100:.2f}% "
      f"{'[PASS - equity only]' if eq == rows else '[NOTE - non-equity rows present]'}")
print(f"   cp_flag counts: {cp_ct}")

print(f"\n### 5 example rows")
ex = examples.copy()
ex['strike_$'] = ex['strike_price'] / 1000.0
print(ex[['secid', 'date', 'exdate', 'cp_flag', 'strike_price', 'strike_$',
          'best_bid', 'best_offer', 'volume', 'open_interest',
          'optionid']].to_string(index=False))

print("\n" + "=" * 78)
print("VALIDATION COMPLETE - inspection only, nothing written to")
print("data/processed/.")
print("=" * 78)
