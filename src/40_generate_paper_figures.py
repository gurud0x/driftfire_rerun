import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Publication figures from ALREADY-GATED DriftFire results.
#
# VISUALIZATION ONLY. This script computes no statistic, fits no model, and
# reads no data/ parquet. Every number below is transcribed from a logged
# result and is re-verified against results/gate_log.md at runtime where the
# log contains it (see verify_against_log()). No gate decision can change.
#
# PROVENANCE (stated per figure, because it is NOT uniform):
#   fig1  results/gate_log.md  - R1 and R2 dev entries (logged, gated)
#   fig2  results/gate_log.md  - V1 dev/holdout, V2 dev/holdout (logged, gated)
#   fig3  src/25_compression_decay_check.py console output - EXPLORATORY,
#         deliberately never written to gate_log.md (no gate authority)
#   fig4  10d MAE from results/gate_log.md (K0 decision, logged);
#         30d MAE from src/26_forecaster_horse_race_30d.py - EXPLORATORY,
#         deliberately never logged
#   fig5  +9.96% / -11.74% from gate_log.md (K1 CORRECTED entry, logged);
#         +3.85% from src/37_k1_signal_construction_real_prices.py console
#         output (the K1 REAL PRICES log entry records daily-series stats,
#         not the per-trade mean)
#   fig6  READ from data/processed/k1_trade_summary.json (written by
#         src/37) and verified against the K1 REAL PRICES per-trade block in
#         gate_log.md. No transcription at all - strictly stronger than the
#         hardcode-plus-check used above.
#
# DESIGN NOTES (dataviz skill):
#   Palette = validated categorical slots 1-3 (blue/orange/aqua) checked with
#   scripts/validate_palette.js against the WHITE paper surface: all hard
#   gates PASS (worst adjacent CVD dE 9.2, normal-vision dE 27.6). Aqua warns
#   at 2.82:1 contrast, so the relief rule applies - every aqua mark carries a
#   visible direct value label.
#   Static print figures: no hover layer and no dark mode (a paper figure is
#   printed on white); those parts of the method do not apply here.
#   Identity is never colour-alone: bar charts use direct value labels, the
#   4-series line chart adds dash pattern + distinct markers on top of hue.
#   Text uses ink tokens, never the series colour.
# ---------------------------------------------------------------------------

project_root = Path(__file__).parent.parent
log_path = project_root / 'results' / 'gate_log.md'
out_dir = project_root / 'results' / 'figures'
out_dir.mkdir(parents=True, exist_ok=True)

# --- validated palette + ink tokens ---------------------------------------
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


def save(fig, name):
    for ext in ('png', 'pdf'):
        p = out_dir / f'{name}.{ext}'
        fig.savefig(p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    png = out_dir / f'{name}.png'
    print(f"  [OK] {name}.png / .pdf  ({png.stat().st_size/1024:.0f} KB png)")


def subtitle(ax, text):
    # sits between the axes top and the title (titlepad reserves the room)
    ax.text(0, 1.025, text, transform=ax.transAxes, fontsize=8.5,
            color=INK2, va='bottom', ha='left')


def source(fig, text, y=-0.03):
    # figure coords just below the canvas; bbox_inches='tight' expands to
    # include it, so it never collides with the x-label. Pass a lower y on
    # figures that carry a long explanatory note, so the source stays last.
    fig.text(0.0, y, text, fontsize=7, color=MUTED, ha='left', va='top')


# ---------------------------------------------------------------------------
# Runtime provenance check: every value that the log DOES contain must be
# found there verbatim. Guards against transcription error.
# ---------------------------------------------------------------------------
def verify_against_log(values):
    txt = log_path.read_text(encoding='utf-8', errors='replace')
    missing = [v for v in values if v not in txt]
    print(f"\nProvenance check against {log_path.name}: "
          f"{len(values)-len(missing)}/{len(values)} logged values found "
          f"verbatim {'[PASS]' if not missing else '[FAIL]'}")
    if missing:
        print(f"  NOT FOUND (transcription error?): {missing}")
        raise SystemExit(1)


# (note_unverifiable() was removed once src/37 began persisting
# k1_trade_summary.json: every figure value is now file-checkable.)


print("=" * 78)
print("GENERATING PAPER FIGURES (visualization only - no analysis)")
print("=" * 78)

verify_against_log([
    '+0.969', '+1.094', '-0.252', '-0.568', '+0.054', '-0.157', '+0.562',
    '+0.810', '+1.111', '-0.298', '-0.382', '-0.072', '-0.129', '+0.326',
    '0.4656', '0.4318', '0.5150', '0.4903', '0.4700', '0.4386', '0.5295',
    '0.5121', '0.1751', '0.1810', '0.1980', '+9.96% -> -11.74%',
])

FACTORS = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM', 'ST_Rev']

# ===========================================================================
# FIG 1 - R1 vs R2 factor loadings (gate_log.md, both dev entries)
# ===========================================================================
print("\nfig1: R1 vs R2 factor loadings")
r1 = [0.969, 1.094, -0.252, -0.568, 0.054, -0.157, 0.562]
r2 = [0.810, 1.111, -0.298, -0.382, -0.072, -0.129, 0.326]

fig, ax = plt.subplots(figsize=(7.2, 4.0))
x = np.arange(len(FACTORS))
w = 0.38
b1 = ax.bar(x - w/2 - 0.01, r1, w, label='R1  unconditional reversal',
            color=BLUE, zorder=3)
b2 = ax.bar(x + w/2 + 0.01, r2, w, label='R2  volume-conditioned',
            color=ORANGE, zorder=3)
ax.axhline(0, color=AXIS, lw=1.0, zorder=2)
for bars, vals in ((b1, r1), (b2, r2)):
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                v + (0.045 if v >= 0 else -0.045), f'{v:+.2f}',
                ha='center', va='bottom' if v >= 0 else 'top',
                fontsize=7, color=INK2)
ax.set_xticks(x, FACTORS)
ax.set_ylabel('factor loading (beta)')
ax.set_ylim(-0.85, 1.45)
ax.set_title('Both reversal portfolios are dominated by passive factor exposure')
subtitle(ax, 'Dev window 2015-2021, daily net excess returns, Newey-West '
              '(10 lags). Both phases FAILED their gate.')
ax.legend(loc='upper right', ncol=1)
source(fig, 'Source: results/gate_log.md - R1 dev entry (2026-07-11 23:24) '
            'and R2 dev entry (2026-07-11 23:45).')
save(fig, 'fig1_r1_r2_factor_loadings')

# ===========================================================================
# FIG 2 - V1/V2 monotonicity, dev + holdout (gate_log.md, four entries)
# ===========================================================================
print("fig2: V1/V2 monotonicity, all four windows")
dec = np.arange(1, 11)
v1_dev = [.4656, .4300, .4130, .4021, .3964, .3899, .3892, .3888, .3976, .4318]
v1_hold = [.5150, .4771, .4592, .4473, .4377, .4309, .4287, .4367, .4459, .4903]
v2_dev = [.4700, .4315, .4149, .4048, .3993, .3943, .3942, .3945, .4026, .4386]
v2_hold = [.5295, .4809, .4620, .4516, .4411, .4340, .4398, .4412, .4573, .5121]

fig, ax = plt.subplots(figsize=(7.2, 4.3))
series = [
    (v1_dev,  BLUE,   '-',  'o', 'V1 dev (2015-2021)'),
    (v1_hold, BLUE,   '--', 's', 'V1 holdout (2022-2025)'),
    (v2_dev,  ORANGE, '-',  '^', 'V2 dev (2015-2021)'),
    (v2_hold, ORANGE, '--', 'D', 'V2 holdout (2022-2025)'),
]
for vals, c, ls, mk, lab in series:
    ax.plot(dec, vals, color=c, linestyle=ls, marker=mk, markersize=4.5,
            linewidth=2, label=lab, zorder=3, markeredgecolor=SURFACE,
            markeredgewidth=0.8)
ax.set_xticks(dec)
ax.set_xlabel('compression decile   (1 = most compressed, 10 = most expanded)')
ax.set_ylabel('mean forward 10-day realized vol (annualized)')
ax.set_title('The U-shape replicates in all four windows, in and out of sample')
subtitle(ax, 'Both tails predict higher forward volatility; the compressed '
              'tail (decile 1) is always the maximum.')
ax.legend(loc='upper center', ncol=2)
ax.set_ylim(0.37, 0.585)
source(fig, 'Source: results/gate_log.md - V1 dev/holdout and V2 dev/holdout '
            'entries. Identity carried by hue + dash + marker, not hue alone.')
save(fig, 'fig2_v1_v2_monotonicity')

# ===========================================================================
# FIG 3 - decay curve (EXPLORATORY, src/25 - never logged)
# ===========================================================================
print("fig3: V1 decay curve across horizons")
hz = [5, 10, 20, 30]
tst = [-5.425, -6.883, -8.182, -8.176]

fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.axhspan(-3.0, 0.2, color=GRID, alpha=0.55, zorder=1)
ax.axhline(-3.0, color=MUTED, lw=1.0, ls=':', zorder=2)
ax.text(31.6, -3.16, "|t| = 3.0 bar", va='top', ha='right',
        fontsize=7.5, color=MUTED)
ax.text(5.2, -1.4, 'would FAIL the pre-registered bar', fontsize=7.5,
        color=MUTED, va='center')
ax.plot(hz, tst, color=BLUE, marker='o', markersize=6, linewidth=2, zorder=4,
        markeredgecolor=SURFACE, markeredgewidth=1.0)
for h, t in zip(hz, tst):
    ax.annotate(f'{t:.2f}', (h, t), textcoords='offset points',
                xytext=(0, -14), ha='center', fontsize=8, color=INK2)
ax.set_xticks(hz, [f'{h}d' for h in hz])
ax.set_xlabel('forecast horizon (trading days)')
ax.set_ylabel('Newey-West t-stat of the Fama-MacBeth slope')
ax.set_ylim(-9.6, 0.2)
ax.set_xlim(4, 32)
ax.set_title('The compression signal strengthens with horizon; it does not decay')
subtitle(ax, 'More negative = stronger (compression predicts higher forward '
              'vol). Dev window only.')
source(fig, 'Source: src/25_compression_decay_check.py console output - '
            'EXPLORATORY, deliberately never written to gate_log.md '
            '(no gate authority).')
save(fig, 'fig3_v1_decay_curve')

# ===========================================================================
# FIG 4 - K0 horse race, 10d (logged) + 30d (exploratory)
# ===========================================================================
print("fig4: K0 forecaster horse race at both horizons")
mae10 = [0.1751, 0.1810, 0.1980]     # gate_log.md, K0 decision
mae30 = [0.1644, 0.1621, 0.1847]     # src/26, exploratory
labels = ['TRAIL20\ntrailing 20d vol', 'GARCH11\nper-stock GARCH(1,1)',
          'COMPDEC\ncompression-decile mean']
cols = [BLUE, ORANGE, AQUA]

fig, ax = plt.subplots(figsize=(7.2, 4.2))
x = np.arange(2)
w = 0.26
for i, (lab, c) in enumerate(zip(labels, cols)):
    vals = [mae10[i], mae30[i]]
    off = (i - 1) * (w + 0.02)
    bars = ax.bar(x + off, vals, w, label=lab.replace('\n', '  '), color=c,
                  zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.004, f'{v:.4f}',
                ha='center', va='bottom', fontsize=7.5, color=INK2)
ax.set_xticks(x, ['10-day horizon\n(K0 gate, logged)',
                  '30-day horizon\n(exploratory)'])
ax.set_ylabel('MAE vs realized vol   (lower is better)')
ax.set_ylim(0, 0.278)     # headroom so the legend clears the value labels
ax.set_title('TRAIL20 wins at 10 days; GARCH edges it at 30 by less than its '
             'in-sample advantage')
subtitle(ax, 'Identical dev-window stock-days per horizon. GARCH parameters '
              'are fit in-sample, so its 30d edge is not treated as a win.')
ax.legend(loc='upper left', ncol=1)
source(fig, 'Sources: 10-day MAE from results/gate_log.md (K0 decision, '
            'logged). 30-day MAE from src/26_forecaster_horse_race_30d.py '
            '- EXPLORATORY, never logged. Every bar carries a direct value '
            'label (relief rule for the aqua slot).')
save(fig, 'fig4_k0_horse_race')

# ===========================================================================
# FIG 5 - the three K1 valuations
# ===========================================================================
print("fig5: K1 three valuations of the same trades")
k1_labels = ['Reconstruction A\nfresh-30d exit\n(no theta at all)',
             'Reconstruction B\nsqrt-time IV haircut\n(double-counts theta)',
             'REAL QUOTES\nsame contract, both dates\n(observed)']
k1_vals = [9.96, -11.74, 3.85]
k1_cols = [MUTED, MUTED, BLUE]

fig, ax = plt.subplots(figsize=(7.2, 4.4))
bars = ax.bar(np.arange(3), k1_vals, 0.52, color=k1_cols, zorder=3)
ax.axhline(0, color=AXIS, lw=1.0, zorder=2)
for i, (bar, v) in enumerate(zip(bars, k1_vals)):
    ax.text(bar.get_x() + bar.get_width()/2, v + (0.9 if v >= 0 else -0.9),
            f'{v:+.2f}%', ha='center', va='bottom' if v >= 0 else 'top',
            fontsize=10, fontweight='bold', color=INK)
    tag = 'ARTIFACT' if i < 2 else 'TRUSTWORTHY'
    ax.text(bar.get_x() + bar.get_width()/2, -15.4, tag, ha='center',
            va='top', fontsize=7.5, fontweight='bold',
            color=MUTED if i < 2 else BLUE)
ax.set_xticks(np.arange(3), k1_labels)
ax.tick_params(axis='x', labelsize=8)
ax.set_ylabel('mean per-trade return (%)')
ax.set_ylim(-17.5, 13.5)
ax.set_title('The same 5,000+ trades, priced three ways: only one was observed')
subtitle(ax, 'Reconstructions A and B model a price that was never quoted; '
              'the truth lies between them.')
ax.text(0.5, -0.21,
        'A and B reconstructed the exit from a constant-maturity surface: A omitted time decay '
        'entirely, B over-corrected it.\nOnly the third prices the actual held contract from its '
        'real quoted bid/ask on both dates (98.9% match rate).\nEven so, the observed entry '
        'half-spread is 18.8% of mid (median) - so the +3.85% mid-to-mid edge is not capturable.',
        transform=ax.transAxes, fontsize=7.5, color=INK2, ha='center',
        va='top', linespacing=1.6)
source(fig, 'Sources: +9.96% and -11.74% from results/gate_log.md (K1 '
            'CORRECTED entry). +3.85% from '
            'src/37_k1_signal_construction_real_prices.py console output.',
       y=-0.17)
save(fig, 'fig5_k1_three_valuations')

# ===========================================================================
# FIG 6 - the spread that eats the edge (all values from src/37 console)
# ===========================================================================
print("fig6: K1 edge vs observed spread")

# fig6's values are now READ from the persisted summary artifact rather than
# transcribed, then verified to appear in gate_log.md - strictly stronger
# than the hardcode-plus-check used by figs 1-5. Reading a summary artifact
# is not recomputation: no statistic is derived here.
import json
summary_path = project_root / 'data' / 'processed' / 'k1_trade_summary.json'
if not summary_path.exists():
    print(f"STOP: {summary_path.name} not found. Run "
          f"src/37_k1_signal_construction_real_prices.py to generate it "
          f"(it persists the per-trade summary).")
    raise SystemExit(1)
k1s = json.loads(summary_path.read_text())
EDGE = k1s['mean_per_trade_return_pct']['mid_to_mid']
SPREAD_MED = k1s['entry_half_spread_pct_of_mid']['median']
SPREAD_MEAN = k1s['entry_half_spread_pct_of_mid']['mean']
print(f"  values read from {summary_path.name}: edge {EDGE:+.2f}%, "
      f"half-spread median {SPREAD_MED:.2f}%, mean {SPREAD_MEAN:.2f}%")
verify_against_log([f'{EDGE:+.2f}%', f'{SPREAD_MED:.2f}% of mid',
                    f'{SPREAD_MEAN:.2f}% of mid'])
vals6 = [EDGE, SPREAD_MED, SPREAD_MEAN]
labs6 = ['Mid-to-mid edge\nmean per trade',
         'Entry half-spread\nmedian',
         'Entry half-spread\nmean']
cols6 = [BLUE, ORANGE, ORANGE]

fig, ax = plt.subplots(figsize=(7.2, 4.4))
bars = ax.bar(np.arange(3), vals6, 0.5, color=cols6, zorder=3)
ax.axhline(EDGE, color=BLUE, lw=1.0, ls=':', zorder=2)
ax.text(2.46, EDGE + 0.5, 'the edge', ha='right', va='bottom', fontsize=7.5,
        color=BLUE)
for bar, v in zip(bars, vals6):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.7, f'{v:.2f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold',
            color=INK)
ax.set_xticks(np.arange(3), labs6)
ax.tick_params(axis='x', labelsize=8.5)
ax.set_ylabel('percent of option mid price')
ax.set_ylim(0, 36)
ax.set_title('The spread on these contracts dwarfs the edge they would '
             'have to capture')
subtitle(ax, 'Real quoted bid/ask on the matched K1 contracts, dev window. '
              'One axis: all three are percentages of mid.')
# round-trip hurdle derived from the artifact values, never hardcoded
rt_med, rt_mean = 2 * SPREAD_MED, 2 * SPREAD_MEAN
ax.text(0.5, -0.235,
        f'A round trip pays the half-spread twice - on entry and again on exit - so the cost hurdle is '
        f'{rt_med:.1f}% (median) to {rt_mean:.1f}% (mean) of mid.\nAgainst a {EDGE:+.2f}% mid-to-mid '
        f'edge, the spread is roughly {rt_med/EDGE:.0f}x to {rt_mean/EDGE:.0f}x larger. This is why the '
        f'mid-based result is not capturable, and it is measured,\nnot assumed: the surface pull '
        f'contained no bid/ask at all, so no earlier K1 attempt could quantify it.',
        transform=ax.transAxes, fontsize=7.5, color=INK2, ha='center',
        va='top', linespacing=1.6)
source(fig, 'Source: data/processed/k1_trade_summary.json, written by '
            'src/37_k1_signal_construction_real_prices.py; the same values '
            'are logged in the K1 REAL PRICES per-trade block of '
            'results/gate_log.md and verified against it at build time. '
            'Bar identity carried by the category axis and direct value '
            'labels, not colour.', y=-0.19)
save(fig, 'fig6_k1_spread_vs_edge')

print("\n" + "=" * 78)
print(f"COMPLETE - 6 figures x 2 formats written to {out_dir}")
print("Visualization only: no statistic recomputed, no gate decision touched.")
print("=" * 78)
