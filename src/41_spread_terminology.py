import pandas as pd
import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Establish the EXACT spread definition used in this project, unambiguously,
# so the paper's prose/Figure 6 label can be hand-fixed to match the code.
#
# This script computes nothing new: it (1) shows the literal source line(s)
# from src/37 that produced the 18.75%/29.80% figures, (2) restates that
# formula in unambiguous notation, (3) recomputes the same quantity directly
# from the persisted per-trade parquet as an independent check, and
# (4) asserts it matches the persisted JSON. No filter, threshold, or trade
# selection is touched.
# ---------------------------------------------------------------------------

project_root = Path(__file__).parent.parent
src37_path = project_root / 'src' / '37_k1_signal_construction_real_prices.py'
trades_path = project_root / 'data' / 'processed' / 'k1_trades_real_prices.parquet'
summary_path = project_root / 'data' / 'processed' / 'k1_trade_summary.json'
out_path = project_root / 'results' / '41_spread_terminology.json'
out_path.parent.mkdir(parents=True, exist_ok=True)

print("=" * 78)
print("SPREAD TERMINOLOGY - establishing the authoritative definition")
print("=" * 78)

# --------------------------------------------------------------------------
# 1. The exact source line(s), read from src/37, not retyped from memory
# --------------------------------------------------------------------------
print(f"\n### 1. Source lines from {src37_path.name} (verbatim)")
lines = src37_path.read_text(encoding='utf-8').splitlines()

# locate by content, not a hardcoded line number, so this stays correct if
# src/37 is ever edited above this point
def find_line(pattern):
    for i, l in enumerate(lines, 1):
        if re.search(pattern, l):
            return i, l
    return None, None

i_mid, l_mid = find_line(r"t\['entry_mid'\]\s*=")
i_ask, l_ask = find_line(r"t\['entry_ask'\]\s*=")
i_sp, l_sp = find_line(r"^sp\s*=")
i_print, l_print = find_line(r"observed entry half-spread")

assert i_mid is not None and l_mid is not None, (
    "Pattern not found in src/37: t['entry_mid'] = ...")
assert i_mid < len(lines), (
    "Pattern found for t['entry_mid'], but continuation line is missing.")
assert i_ask is not None and l_ask is not None, (
    "Pattern not found in src/37: t['entry_ask'] = ...")
assert i_sp is not None and l_sp is not None, (
    "Pattern not found in src/37: ^sp = ...")
assert i_print is not None and l_print is not None, (
    "Pattern not found in src/37: observed entry half-spread")

# entry_mid is a two-line continuation; grab both physical lines
quote_block = {
    'entry_mid (lines %d-%d)' % (i_mid, i_mid + 1): lines[i_mid - 1].strip() +
        ' ' + lines[i_mid].strip(),
    f'entry_ask (line {i_ask})': l_ask.strip(),
    f'sp - the spread itself (line {i_sp})': l_sp.strip(),
    f'console label (line {i_print})': l_print.strip(),
}
for label, code in quote_block.items():
    print(f"  {label}:")
    print(f"    {code}")

# --------------------------------------------------------------------------
# 2. Unambiguous statement of what the formula computes
# --------------------------------------------------------------------------
print("\n### 2. Unambiguous definition")
print("""
  entry_mid = (bid_call + offer_call)/2 + (bid_put + offer_put)/2
            = sum of each LEG's own mid  (equivalently, the straddle's
              bid-ask midpoint: (straddle_bid + straddle_offer)/2, since
              straddle_bid = bid_call+bid_put and straddle_offer =
              offer_call+offer_put)

  entry_ask = offer_call + offer_put
            = the straddle's own posted OFFER (cost to buy both legs
              at their quoted ask)

  sp = (entry_ask - entry_mid) / entry_mid * 100

  Algebraically, since entry_mid = (entry_bid + entry_ask)/2:
    entry_ask - entry_mid = (entry_ask - entry_bid) / 2

  So sp = (entry_ask - entry_bid) / (2 * entry_mid) * 100

  IN WORDS: sp is the HALF of the quoted bid-ask spread, expressed as a
  percentage of the mid price - i.e. (ask - mid)/mid, which is exactly
  HALF of the more familiar (ask - bid)/mid "full spread as % of mid".
  It is computed at the STRADDLE level (call leg + put leg combined),
  not averaged per-leg.

  It is NOT (ask - bid)/mid (that would be the FULL spread, ~2x this
  number) and NOT (ask - mid)/bid or any bid-denominated ratio.
""")
formula_ambiguous_check = "(ask - mid) / mid,  equivalently  (ask - bid) / (2*mid)"
print(f"  ANSWER to 'which formula': {formula_ambiguous_check}")

# --------------------------------------------------------------------------
# 3. Recompute independently from the persisted per-trade parquet
# --------------------------------------------------------------------------
print(f"\n### 3. Independent recomputation from {trades_path.name}")
if not trades_path.exists():
    print(f"STOP: {trades_path} not found.")
    raise SystemExit(1)
t = pd.read_parquet(trades_path, columns=['entry_mid', 'entry_ask'])
sp = (t['entry_ask'] - t['entry_mid']) / t['entry_mid'] * 100

n = len(sp)
median_recomputed = round(float(sp.median()), 4)
mean_recomputed = round(float(sp.mean()), 4)
print(f"  rows: {n:,}")
print(f"  median: {median_recomputed:.4f}%")
print(f"  mean:   {mean_recomputed:.4f}%")

# --------------------------------------------------------------------------
# Cross-check against the persisted summary JSON
# --------------------------------------------------------------------------
print(f"\n### Cross-check against {summary_path.name}")
if not summary_path.exists():
    print(f"STOP: {summary_path} not found.")
    raise SystemExit(1)
k1s = json.loads(summary_path.read_text())
median_logged = k1s['entry_half_spread_pct_of_mid']['median']
mean_logged = k1s['entry_half_spread_pct_of_mid']['mean']
n_logged = k1s['trades']['matched_entry_and_exit']

print(f"  JSON median: {median_logged}%   recomputed: {median_recomputed}%")
print(f"  JSON mean:   {mean_logged}%   recomputed: {mean_recomputed}%")
print(f"  JSON matched_entry_and_exit: {n_logged:,}   parquet rows: {n:,}")

assert n == n_logged, (
    f"Row count mismatch: parquet has {n}, JSON says {n_logged}")
assert abs(median_recomputed - median_logged) < 1e-4, (
    f"Median mismatch: {median_recomputed} vs {median_logged}")
assert abs(mean_recomputed - mean_logged) < 1e-4, (
    f"Mean mismatch: {mean_recomputed} vs {mean_logged}")
print("  [PASS] recomputed values match the persisted summary exactly, "
      "and n matches matched_entry_and_exit.")

AUTHORITATIVE_MEDIAN = median_recomputed   # 18.75
AUTHORITATIVE_MEAN = mean_recomputed       # 29.805 (prints 29.8 at 1dp)

# --------------------------------------------------------------------------
# 4. Plain-English sentence for the paper
# --------------------------------------------------------------------------
paper_sentence = (
    f"At entry, the quoted offer price for the straddle exceeds its "
    f"bid-ask midpoint by a median of {AUTHORITATIVE_MEDIAN:.2f}% of that "
    f"midpoint (mean {AUTHORITATIVE_MEAN:.2f}%) -- that is, the half-"
    f"spread, not the full bid-ask spread, expressed as a percentage of "
    f"the option's mid price."
)
print("\n### 4. Sentence for the paper")
print(f'  "{paper_sentence}"')
print("\n  CAUTION: 'bid-ask spread' in the paper's prose, if used")
print("  unqualified, should refer to this HALF-spread quantity only if")
print("  the paper means (ask-mid)/mid. If the paper's prose means the")
print("  FULL spread (ask-bid)/mid, the correct figures are twice these:")
print(f"    median {AUTHORITATIVE_MEDIAN*2:.2f}%, mean "
      f"{AUTHORITATIVE_MEAN*2:.2f}%")
print("  Figure 6's own label says 'Entry half-spread' - so 18.75%/29.80%")
print("  is consistent with that label. The risk is only in prose that")
print("  separately says 'the bid-ask spread is 18.75%' without the word")
print("  'half', which would misstate the full spread by 2x.")

# --------------------------------------------------------------------------
# Save
# --------------------------------------------------------------------------
result = {
    'generated_by': 'src/41_spread_terminology.py',
    'source_script': 'src/37_k1_signal_construction_real_prices.py',
    'source_lines': {k: v for k, v in quote_block.items()},
    'formula': {
        'entry_mid': '(bid_call + offer_call)/2 + (bid_put + offer_put)/2',
        'entry_ask': 'offer_call + offer_put',
        'sp_as_coded': '(entry_ask - entry_mid) / entry_mid * 100',
        'equivalent_form': '(entry_ask - entry_bid) / (2 * entry_mid) * 100',
        'classification': 'HALF bid-ask spread as a percent of mid, at the '
                          'combined straddle (call+put) level',
        'is_ask_minus_bid_over_mid': False,
        'is_ask_minus_mid_over_mid': True,
    },
    'recomputed_from_parquet': {
        'file': 'data/processed/k1_trades_real_prices.parquet',
        'n_trades': n,
        'median_pct': median_recomputed,
        'mean_pct': mean_recomputed,
    },
    'cross_check_vs_summary_json': {
        'file': 'data/processed/k1_trade_summary.json',
        'median_pct_logged': median_logged,
        'mean_pct_logged': mean_logged,
        'n_matched_entry_and_exit_logged': n_logged,
        'match_status': 'PASS - exact match',
    },
    'authoritative_values_pct_of_mid': {
        'half_spread_median': AUTHORITATIVE_MEDIAN,
        'half_spread_mean': round(AUTHORITATIVE_MEAN, 2),
        'full_spread_median_if_paper_means_full': round(
            AUTHORITATIVE_MEDIAN * 2, 2),
        'full_spread_mean_if_paper_means_full': round(
            AUTHORITATIVE_MEAN * 2, 2),
    },
    'paper_sentence': paper_sentence,
}
out_path.write_text(json.dumps(result, indent=2))
print(f"\n[OK] Saved {out_path}")

print("\n" + "=" * 78)
print("COMPLETE - no computation changed, no gate result touched.")
print("=" * 78)
