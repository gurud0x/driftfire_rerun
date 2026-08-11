import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# V3 pre-registration figure: ATM implied-vol coverage by tenor (10d, 30d,
# both), by year and by decile. This is the data-availability evidence
# behind prereg_V3.md section 3.3-F's horizon decision (10 trading days ->
# 30 calendar days). DESCRIPTIVE OF DATA AVAILABILITY ONLY - no return, RV,
# or regression quantity appears anywhere in this script or its output.
#
# VISUALIZATION ONLY, same discipline as src/40: every number is READ from
# results/47_v3_iv_tenor_coverage.json (written by src/47's STEP 4), never
# retyped. This script computes no statistic and fits no model.
#
# Kept separate from src/40_generate_paper_figures.py: that script's own
# docstring scopes it to "ALREADY-GATED DriftFire results" (fig1-fig6). This
# figure supports a pre-registration decision, not a gated result, so it
# does not belong in that sequence or under gate_log.md provenance.
# ---------------------------------------------------------------------------

project_root = Path(__file__).parent.parent
json_path = project_root / 'results' / '47_v3_iv_tenor_coverage.json'
out_dir = project_root / 'results' / 'figures'
out_dir.mkdir(parents=True, exist_ok=True)

# --- validated palette + ink tokens (identical to src/40 - same project,
# same validated categorical slots 1-3; not re-derived here) -------------
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS = '#e1e0d9', '#c3c2b7'
SURFACE = '#ffffff'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'DejaVu Sans', 'Arial'],
    'text.color': INK, 'axes.labelcolor': INK2, 'axes.edgecolor': AXIS,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.axisbelow': True, 'font.size': 9,
    'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.titlelocation':
    'left', 'axes.titlepad': 30,
    'legend.frameon': False, 'legend.fontsize': 8.5,
})

print("=" * 78)
print("V3 PRE-REGISTRATION FIGURE: IV coverage by tenor, year, decile")
print("=" * 78)

if not json_path.exists():
    print(f"STOP: {json_path} not found. Run src/47_v3_data_feasibility.py "
          f"first (its STEP 4 writes this file).")
    raise SystemExit(1)

data = json.loads(json_path.read_text(encoding='utf-8'))
print(f"\n[OK] Loaded {json_path.name}")
print(f"  source: {data['source']}")
print(f"  n_universe_stock_days: {data['n_universe_stock_days']:,}")
print(f"  overall: 10d={data['overall']['pct_10']:.2f}%  "
      f"30d={data['overall']['pct_30']:.2f}%  "
      f"both={data['overall']['pct_both']:.2f}%")
print(f"  pct_both == pct_10 in every year x decile cell: "
      f"{data['note_both_equals_10d']}")
if not data['note_both_equals_10d']:
    raise SystemExit("STOP: the 10d-subset-of-30d structural fact this "
                      "figure's caption asserts no longer holds in the "
                      "data - caption text would be wrong. Halting rather "
                      "than plotting a claim the data doesn't support.")

rows = data['by_year_decile']
years = sorted({int(r['year']) for r in rows})
deciles = sorted({int(r['decile']) for r in rows})
by_key = {(int(r['year']), int(r['decile'])): r for r in rows}

fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.3), sharey=True)

SERIES = [
    ('pct_30', '30-day tenor (native)', BLUE, 'o', '-'),
    ('pct_10', '10-day tenor (native)', ORANGE, 's', '-'),
    ('pct_both', 'BOTH 10d and 30d', AQUA, '^', '--'),
]

for ax, dec in zip(axes, deciles):
    for col, label, color, marker, ls in SERIES:
        vals = [by_key[(y, dec)][col] for y in years]
        ax.plot(years, vals, color=color, marker=marker, markersize=6,
                linewidth=2.0 if ls == '-' else 2.4, linestyle=ls,
                markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=4,
                label=label)
    ax.set_title(f'decile {dec}', fontsize=10, fontweight='bold',
                 color=INK, loc='left', pad=8)
    ax.set_xticks(years)
    ax.set_xticklabels([str(y)[-2:] for y in years], fontsize=8)
    ax.set_ylim(-3, 103)
    ax.set_xlim(years[0] - 0.4, years[-1] + 0.4)

axes[0].set_ylabel('ATM IV available, both call+put sides (%)')

# Explicit margins (not tight_layout) so the header block sits flush above
# the panels instead of floating with a gap - tight_layout's rect and
# figure-fraction text above y=1.0 don't reconcile their reserved space.
# top=0.60 leaves enough room below the legend (0.80) that the panel titles
# (which render ~0.03-0.07 fraction above the axes top, from their own pad)
# land well clear of it, rather than the two bands colliding.
fig.subplots_adjust(top=0.60, bottom=0.16, left=0.065, right=0.99, wspace=0.05)

# Shared legend, once, above the panels - avoids repeating it 3x
handles = [mlines.Line2D([], [], color=c, marker=m, markersize=6,
                         linestyle=ls, linewidth=2.0 if ls == '-' else 2.4,
                         markeredgecolor=SURFACE, markeredgewidth=0.8,
                         label=lbl)
          for _, lbl, c, m, ls in SERIES]
fig.legend(handles=handles, loc='upper center', ncol=3,
          bbox_to_anchor=(0.5, 0.80), fontsize=9, frameon=False)

fig.suptitle(
    'The 10-day surface tenor barely exists; the 30-day tenor is nearly '
    'universal',
    x=0.01, y=0.975, ha='left', fontsize=11, fontweight='bold', color=INK)
fig.text(0.01, 0.90,
         'V1/V2 universe, DEV 2015-2021, size deciles 6/7/8. "Both" traces '
         'exactly on top of "10-day" (dashed over solid, same value in every '
         'year x decile cell) because 10-day availability is a strict subset '
         'of 30-day, confirmed structurally - not an approximation.',
         fontsize=8, color=INK2, transform=fig.transFigure)

fig.text(0.01, 0.03,
         f"Source: results/47_v3_iv_tenor_coverage.json, written by "
         f"src/47_v3_data_feasibility.py STEP 4 ({data['generated']}). "
         f"Overall: 30d={data['overall']['pct_30']:.2f}%, "
         f"10d=both={data['overall']['pct_10']:.2f}%. Descriptive of IV data "
         f"availability only - no return, RV, or regression quantity is "
         f"shown. This figure motivates prereg_V3.md section 3.3-F's "
         f"decision to move V3's primary horizon from 10 trading days to "
         f"30 calendar days.",
         fontsize=7, color=MUTED, ha='left', va='top',
         transform=fig.transFigure, wrap=True)

for ext in ('png', 'pdf'):
    p = out_dir / f'v3_iv_tenor_coverage.{ext}'
    fig.savefig(p, dpi=300, bbox_inches='tight')
plt.close(fig)
png = out_dir / 'v3_iv_tenor_coverage.png'
print(f"\n[OK] Saved {png}  ({png.stat().st_size/1024:.0f} KB)")
print(f"[OK] Saved {out_dir / 'v3_iv_tenor_coverage.pdf'}")

print("\n" + "=" * 78)
print("FIGURE COMPLETE - descriptive of data availability only.")
print("=" * 78)
