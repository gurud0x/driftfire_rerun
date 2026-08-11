import pandas as pd
import numpy as np
import json
import shutil
from datetime import datetime
from pathlib import Path
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# EARNINGS-TIMING CONFOUND TEST for the V1/V2 volume-compression signal.
#
# WHY THIS MATTERS. With R1, R2, E1 and K1 all closed, V1/V2 are the only
# positive result the paper claims. V1/V2 report that compression predicts
# forward realized volatility. A forward window contains a quarterly earnings
# announcement some fraction of the time by construction. If compression
# deciles differ in how often their forward window contains an announcement,
# that alone could generate the entire V1/V2 result - "stocks are volatile
# around earnings" is a calendar fact, not a research finding. This script
# tests exactly that.
#
# ===========================================================================
# PRE-COMMITMENT - LOCKED BEFORE ANY RETURN OR VOLATILITY DATA WAS TOUCHED
# ===========================================================================
# These three rules are fixed here, are printed before anything runs, and are
# echoed into results/43_earnings_confound.json. NO FAVORABLE READING IS
# PERMITTED regardless of which way this project's other results have gone.
#
#   RULE 1 - NOT AN EARNINGS-TIMING ARTIFACT.
#     Requires BOTH:
#       (1a) the earnings-in-window rate is within 3.0 percentage points
#            across ALL compression deciles (max rate - min rate <= 3.0 pp),
#            AND
#       (1b) compression retains Newey-West |t| >= 3.0 in specification (ii),
#            i.e. with the earnings dummy controlled.
#
#   RULE 2 - SUBSTANTIALLY AN EARNINGS-TIMING PROXY.
#     Triggers if the compression coefficient loses MORE THAN HALF its
#     magnitude when the dummy is added:
#            |beta_compression(ii)| < 0.5 * |beta_compression(i)|.
#
#   RULE 3 - INCONCLUSIVE. Anything else, disclosed as such.
#
#   CONFLICT RESOLUTION, pre-committed here rather than decided after the
#   fact: RULE 1 and RULE 2 are not mutually exclusive as written (a
#   coefficient can more than halve and still carry |t| >= 3 in a large
#   panel). If BOTH fire, the outcome is RULE 3 - INCONCLUSIVE, and the
#   conflict is reported explicitly. The more favorable of the two is NOT
#   selected.
#
#   A sign flip in the compression coefficient between (i) and (ii) is
#   reported as its own flag and, if present, forces RULE 3 regardless of
#   magnitudes: a coefficient that changes sign has not merely "retained" or
#   "lost" magnitude.
#
# ===========================================================================
# TWO DISCREPANCIES BETWEEN THE TASK AS SPECIFIED AND THIS PROJECT'S FACTS.
# Both are stated here, in the printed output, and in the JSON. Neither is
# resolved silently.
# ===========================================================================
#
# DISCREPANCY 1 - THE FORECAST HORIZON IS 10 DAYS, NOT 30.
#   The task describes V1/V2 as predicting "forward 30-day realized
#   volatility" and asks for a [t+1, t+30] earnings window. The LOCKED V1
#   pre-registration (docs/PhaseV1_PreRegistration_VolatilityCompression.md,
#   commit 4474312) defines realized_vol_fwd_10d - 10 TRADING days starting
#   at t+1 - as the PRIMARY target, with 5d and 20d as secondary/diagnostic.
#   No 30-day target exists anywhere in the project; the V1/V2 panels carry
#   only the 5d/10d/20d columns. V2's locked prereg uses the same primary.
#
#   Consequence for the motivation: a 10-trading-day window covers roughly
#   16% of a ~63-trading-day quarter, not "roughly half". The confound is
#   therefore a priori WEAKER than the task's framing assumes - which is a
#   reason to measure it, not to skip it.
#
#   RESOLUTION, applied here: the PRIMARY test uses V1/V2's actual locked
#   primary target (realized_vol_fwd_10d) with a HORIZON-MATCHED earnings
#   window (an RDQ in the calendar span from session t+1 through session
#   t+10). Testing a 30-day regression that does not exist in this project
#   would answer a question the paper does not ask. The requested
#   [t+1, t+30] CALENDAR-day window is ALSO computed and reported in full -
#   in the decile table and as an alternative control - so the specified
#   quantity is delivered, not dropped. The 5d and 20d targets are reported
#   as secondaries with horizon-matched windows.
#
# DISCREPANCY 2 - RDQ IS NOT IN THIS REPOSITORY.
#   The task specifies Compustat RDQ (report date of quarterly earnings)
#   joined via the existing CCM link. The Compustat pull in this repo is
#   ONLY:
#     data/raw/compustat/ccm_link_gics.csv       - the CCM link itself
#     data/raw/compustat/compustat_gics_names.csv - ANNUAL GICS/company file
#                                                   (~8 rows per gvkey; its
#                                                   datadate is a FISCAL
#                                                   PERIOD END, not a report
#                                                   date)
#   Neither carries RDQ. RDQ lives in Compustat Fundamentals Quarterly
#   (comp.fundq) and requires a WRDS pull that has not been made.
#
#   THIS SCRIPT WILL NOT SUBSTITUTE datadate FOR RDQ. A fiscal period end is
#   not an announcement date: firms report weeks after the period closes,
#   with a firm-specific and time-varying lag. Using datadate would place
#   the "announcement" systematically early by a variable amount, producing
#   an earnings dummy that is wrong in a way correlated with firm identity -
#   which is precisely the kind of error this test exists to detect. A test
#   built on it could not distinguish "no confound" from "the dummy is
#   mismeasured", and a null would be uninterpretable.
#
#   BEHAVIOUR: everything not dependent on RDQ runs NOW and is reported -
#   the pre-commitment (locked above), the CCM link build and its join
#   diagnostics, and specification (i), the baseline, which is cross-checked
#   against the V1/V2 gate results already in results/gate_log.md. The script
#   then HALTS before specifications (ii) and (iii) and writes NO gate_log
#   entry, matching the precedent set by src/46 when Test 1a proved not
#   estimable: a test that did not run does not get a gate_log block.
#   results/43_earnings_confound.json is still written, recording the
#   pre-commitment, the diagnostics, the baseline, and the blocker.
#
#   TO UNBLOCK, pull from WRDS and drop the file in data/raw/compustat/
#   (any filename; discovery is by column, not by name):
#
#     SELECT gvkey, datadate, rdq, fyearq, fqtr
#     FROM comp.fundq
#     WHERE indfmt = 'INDL' AND datafmt = 'STD'
#       AND popsrc = 'D'    AND consol  = 'C'
#       AND datadate BETWEEN '2014-06-01' AND '2026-06-30'
#
#   The date range is padded either side of the V1/V2 panel so that windows
#   at both edges are fully covered. On the next run the script detects the
#   RDQ column, executes (ii) and (iii), evaluates the pre-committed rules
#   and writes the gate_log block. Nothing else needs to change.
#
# ===========================================================================
# METHOD (unchanged from V1/V2 where it overlaps)
# ===========================================================================
#  - Fama-MacBeth: a cross-sectional OLS each trading day, then a Newey-West
#    t-test (maxlags=10) on the daily coefficient series. This is V1/V2's
#    locked methodology, extended from univariate to multivariate for
#    specifications (ii) and (iii).
#  - A date enters the average only with >= MIN_XSEC usable observations and
#    a full-column-rank design matrix. Dates failing either are counted.
#  - shift(1) alignment: the compression inputs are already lagged to end at
#    t-1 by src/10 and are REUSED here, not recomputed. The forward targets
#    are src/10's, spanning t+1..t+N. The earnings dummy is built from the
#    forward window only. No input to any term includes day t or crosses it.
#  - DEV WINDOW ONLY. The 2022-2025 holdout is already SPENT for V1 and V2
#    (both logged as single pre-committed passes). This script makes no
#    holdout claim and has no holdout code path.
#
# gate_log.md receives numbers only, no interpretation.
# ---------------------------------------------------------------------------

# --- Pre-committed thresholds (locked; see PRE-COMMITMENT above) -----------
RULE1_MAX_DECILE_SPREAD_PP = 3.0
RULE1_MIN_ABS_T = 3.0
RULE2_MAGNITUDE_RETENTION = 0.5

# --- Method constants (inherited from V1/V2's locked methodology) ----------
DEV_END = pd.Timestamp('2021-12-31')
NW_MAXLAGS = 10
MIN_XSEC = 30                 # src/11 and src/18 use n >= 30 per date

PRIMARY_TARGET = 'realized_vol_fwd_10d'
PRIMARY_HORIZON_TDAYS = 10
SECONDARY_TARGETS = {'realized_vol_fwd_5d': 5, 'realized_vol_fwd_20d': 20}
REQUESTED_CALENDAR_WINDOW_DAYS = 30      # the task's [t+1, t+30], reported

SIGNALS = [('V1', 'compression_decile',
            'data/processed/compression_signal_v1.parquet'),
           ('V2', 'sector_rel_decile',
            'data/processed/sector_compression_signal_v2.parquet')]

project_root = Path(__file__).parent.parent
v1_path = project_root / 'data' / 'processed' / 'compression_signal_v1.parquet'
v2_path = (project_root / 'data' / 'processed' /
           'sector_compression_signal_v2.parquet')
link_path = project_root / 'data' / 'raw' / 'compustat' / 'ccm_link_gics.csv'
out_json = project_root / 'results' / '43_earnings_confound.json'
log_path = project_root / 'results' / 'gate_log.md'

print('=' * 88)
print('EARNINGS-TIMING CONFOUND TEST for V1/V2 volume compression')
print('=' * 88)

# ==========================================================================
# 0. PRINT THE PRE-COMMITMENT BEFORE ANYTHING ELSE RUNS
# ==========================================================================
print('\n' + '#' * 88)
print('# PRE-COMMITMENT - fixed before any return or volatility data is read')
print('#' * 88)
print(f"""
  RULE 1  NOT an earnings-timing artifact. Requires BOTH:
     1a  earnings-in-window rate within {RULE1_MAX_DECILE_SPREAD_PP:.1f} pp
         across ALL compression deciles (max - min <= {RULE1_MAX_DECILE_SPREAD_PP:.1f} pp)
     1b  compression retains NW |t| >= {RULE1_MIN_ABS_T:.1f} with the earnings
         dummy controlled (specification ii)

  RULE 2  SUBSTANTIALLY an earnings-timing proxy. Triggers if the
          compression coefficient loses more than half its magnitude when
          the dummy is added:
             |beta_comp(ii)| < {RULE2_MAGNITUDE_RETENTION:.2f} * |beta_comp(i)|

  RULE 3  INCONCLUSIVE. Anything else, disclosed as such.

  CONFLICT: if RULE 1 and RULE 2 both fire, the outcome is RULE 3. The more
  favorable of the two is NOT selected. A sign flip in the compression
  coefficient between (i) and (ii) forces RULE 3 regardless of magnitudes.

  No favorable reading is permitted regardless of which way this project's
  other results have gone.
""")
print('#' * 88)

precommit = {
    'locked_before_any_outcome_observed': True,
    'rule_1_not_an_artifact': {
        '1a_max_decile_spread_pp': RULE1_MAX_DECILE_SPREAD_PP,
        '1b_min_abs_nw_t_with_dummy': RULE1_MIN_ABS_T,
        'requires': 'BOTH 1a and 1b',
    },
    'rule_2_substantially_a_proxy': {
        'condition': ('|beta_compression(ii)| < '
                      f'{RULE2_MAGNITUDE_RETENTION} * '
                      '|beta_compression(i)|'),
    },
    'rule_3_inconclusive': 'anything else, disclosed as such',
    'conflict_resolution': ('if RULE 1 and RULE 2 both fire the outcome is '
                           'RULE 3; the more favorable is not selected'),
    'sign_flip_rule': ('a sign flip in the compression coefficient between '
                       '(i) and (ii) forces RULE 3 regardless of magnitudes'),
    'no_favorable_reading_permitted': True,
}

# ==========================================================================
# 1. DISCREPANCIES - stated before any computation
# ==========================================================================
print('\n' + '!' * 88)
print('! DISCREPANCIES BETWEEN THE TASK AS SPECIFIED AND THIS PROJECT')
print('!' * 88)
print(f"""
  1. HORIZON. The task describes a forward-30-day target and a [t+1, t+30]
     window. V1's LOCKED pre-registration (commit 4474312) makes
     {PRIMARY_TARGET} - {PRIMARY_HORIZON_TDAYS} TRADING days from t+1 - the
     PRIMARY target; 5d and 20d are secondary. No 30-day target exists in
     this project. A {PRIMARY_HORIZON_TDAYS}-trading-day window covers ~16%
     of a ~63-day quarter, not "roughly half", so the confound is a priori
     WEAKER than the framing assumes.
     -> PRIMARY test uses the actual locked target with a horizon-matched
        earnings window. The requested {REQUESTED_CALENDAR_WINDOW_DAYS}-calendar-day
        window is ALSO computed and reported in full.

  2. RDQ IS NOT IN THIS REPOSITORY. The Compustat pull here is the CCM link
     plus an ANNUAL GICS/company file whose datadate is a FISCAL PERIOD END,
     not a report date. RDQ requires a comp.fundq pull that has not been
     made. This script will NOT substitute datadate for RDQ - a fiscal
     period end precedes the announcement by a firm-specific, time-varying
     lag, so the resulting dummy would be mismeasured in a way correlated
     with firm identity, and a null would be uninterpretable.
     -> Everything not dependent on RDQ runs now. The script halts before
        specifications (ii) and (iii) and writes no gate_log entry.
""")
print('!' * 88)

# ==========================================================================
# 2. RDQ DISCOVERY - by column, not by filename
# ==========================================================================
print('\n' + '-' * 88)
print('2. RDQ SOURCE DISCOVERY')
print('-' * 88)

search_dirs = [
    project_root / 'data' / 'raw' / 'compustat',
    Path.home() / 'Downloads' / 'quantdata' / 'driftfire' / 'raw' / 'compustat',
]


def find_rdq_source():
    """Return (path, columns) for the first file carrying gvkey AND rdq."""
    seen = []
    for d in search_dirs:
        if not d.exists():
            continue
        for f in sorted(list(d.glob('*.csv')) + list(d.glob('*.parquet'))):
            try:
                if f.suffix == '.csv':
                    cols = list(pd.read_csv(f, nrows=0).columns)
                else:
                    import pyarrow.parquet as pq
                    cols = list(pq.read_schema(f).names)
            except Exception as e:
                seen.append((str(f), f'unreadable: {e}'))
                continue
            low = {c.lower() for c in cols}
            seen.append((str(f.relative_to(project_root)
                             if project_root in f.parents else f), cols))
            if 'rdq' in low and 'gvkey' in low:
                return f, cols, seen
    return None, None, seen


rdq_path, rdq_cols, files_seen = find_rdq_source()
print('  Files inspected for an RDQ column:')
for name, cols in files_seen:
    shown = cols if isinstance(cols, str) else ', '.join(map(str, cols))
    print(f"    {name}")
    print(f"        columns: {shown}")
RDQ_AVAILABLE = rdq_path is not None
print(f"\n  RDQ source found: {RDQ_AVAILABLE}"
      + (f"  -> {rdq_path}" if RDQ_AVAILABLE else ''))

# ==========================================================================
# 3. CCM LINK - built exactly as src/17 does, with join diagnostics
# ==========================================================================
print('\n' + '-' * 88)
print('3. CCM LINK BUILD AND JOIN DIAGNOSTICS '
      '(same approach as src/17: date-windowed LINKDT/LINKENDDT, '
      'LINKTYPE-filtered)')
print('-' * 88)

link = pd.read_csv(link_path)
n_link_raw = len(link)
link.columns = [c.lower() for c in link.columns]
print(f"  CCM link rows as pulled: {n_link_raw:,}")

n0 = len(link)
link = link[link['linktype'].isin(['LC', 'LU'])]
n1 = len(link)
link = link[link['linkprim'].isin(['P', 'C'])]
n2 = len(link)
link = link[link['lpermno'].notna()]
n3 = len(link)
print(f"  filter trail: {n0:,} -> linktype LC/LU {n1:,} -> "
      f"linkprim P/C {n2:,} -> non-null lpermno {n3:,}")

link['linkdt'] = pd.to_datetime(link['linkdt'])
# 'E' marks a still-active link: no upper bound (NOT null-and-drop), same
# handling as src/17.
link['linkend'] = pd.to_datetime(
    link['linkenddt'].replace('E', pd.NaT), errors='coerce')
n_active = int(link['linkend'].isna().sum())
link['linkend'] = link['linkend'].fillna(pd.Timestamp('2262-01-01'))
link['lpermno'] = link['lpermno'].astype(int)
link['gvkey'] = link['gvkey'].astype(int)
print(f"  still-active links ('E' linkenddt, given an open upper bound): "
      f"{n_active:,}")

lk = link[['lpermno', 'gvkey', 'linkdt', 'linkend', 'linkprim']].copy()

# --- Load the V1/V2 panels (DEV only; holdout discarded at load) ----------
panels = {}
for tag, xcol, rel in SIGNALS:
    p = project_root / rel
    df = pd.read_parquet(p)
    n_all = len(df)
    df = df[df['DlyCalDt'] <= DEV_END]      # <- holdout discarded at load
    panels[tag] = df
    print(f"\n  {tag} panel {rel}")
    print(f"    rows all-sample {n_all:,} -> DEV only {len(df):,} "
          f"({df['DlyCalDt'].min().date()} to {df['DlyCalDt'].max().date()})")
    print(f"    PERMNOs {df['PERMNO'].nunique():,}   signal column '{xcol}'")
print("\n  Holdout rows discarded at load; no holdout code path in this "
      "script. The 2022-2025 holdout is already SPENT for both V1 and V2.")


def attach_gvkey(panel, tag):
    """Date-windowed CCM join, P-over-C tiebreak. Returns (panel, diag)."""
    n_before = len(panel)
    # V2's panel already carries a gvkey from src/17, so the merge suffixes
    # both columns. Resolve the link-side name up front rather than assuming
    # a bare 'gvkey' exists.
    m = panel.merge(lk, left_on='PERMNO', right_on='lpermno', how='left')
    gcol = 'gvkey_y' if 'gvkey_y' in m.columns else 'gvkey'
    in_window = (m['DlyCalDt'] >= m['linkdt']) & (m['DlyCalDt'] <= m['linkend'])
    m = m[in_window | m['lpermno'].isna()]
    m['_prim_rank'] = np.where(m['linkprim'] == 'P', 0, 1)
    m = m.sort_values(['PERMNO', 'DlyCalDt', '_prim_rank', gcol])
    n_multi = int(m.duplicated(['PERMNO', 'DlyCalDt'], keep=False).sum()
                  - m.duplicated(['PERMNO', 'DlyCalDt']).sum())
    m = m.drop_duplicates(['PERMNO', 'DlyCalDt'], keep='first')
    m = m.drop(columns=['lpermno', 'linkdt', 'linkend', 'linkprim',
                        '_prim_rank'])
    matched = m[gcol].notna()
    permnos_all = set(panel['PERMNO'].unique())
    permnos_ok = set(m.loc[matched, 'PERMNO'].unique())
    # A PERMNO-day whose PERMNO HAS link rows but none covering that date is
    # dropped outright by the window filter (it never becomes a NaN row).
    # Counted explicitly so the shrinkage is visible rather than implied by
    # the before/after difference.
    n_window_dropped = int(n_before - len(m))
    diag = {
        'rows_before_join': int(n_before),
        'rows_after_join': int(len(m)),
        'rows_dropped_by_link_date_window': n_window_dropped,
        'rows_with_gvkey': int(matched.sum()),
        'rows_with_gvkey_pct': float(matched.mean() * 100),
        'rows_without_gvkey': int((~matched).sum()),
        'permno_days_needing_tiebreak': n_multi,
        'permnos_total': len(permnos_all),
        'permnos_matched': len(permnos_ok),
        'permnos_unmatched': len(permnos_all - permnos_ok),
    }
    print(f"\n  {tag} CCM join diagnostics")
    print(f"    rows before join:              {diag['rows_before_join']:>10,}")
    print(f"    rows after date-windowed join: {diag['rows_after_join']:>10,}")
    print(f"    rows dropped by the link date window (PERMNO has links, "
          f"none cover that date): {n_window_dropped:>10,}")
    print(f"    rows with a gvkey:             {diag['rows_with_gvkey']:>10,} "
          f"({diag['rows_with_gvkey_pct']:.2f}%)")
    print(f"    rows without a gvkey:          "
          f"{diag['rows_without_gvkey']:>10,}")
    print(f"    PERMNO-days needing P-over-C tiebreak: {n_multi:>10,}")
    print(f"    PERMNOs matched / unmatched:   "
          f"{diag['permnos_matched']:,} / {diag['permnos_unmatched']:,} "
          f"(of {diag['permnos_total']:,})")
    if gcol != 'gvkey':
        agree = (m['gvkey_x'] == m['gvkey_y'])
        both = m['gvkey_x'].notna() & m['gvkey_y'].notna()
        diag['gvkey_agrees_with_panel_pct'] = float(
            agree[both].mean() * 100) if int(both.sum()) else float('nan')
        print(f"    gvkey agrees with the panel's own gvkey (src/17): "
              f"{diag['gvkey_agrees_with_panel_pct']:.2f}% of "
              f"{int(both.sum()):,} comparable rows")
        m = m.drop(columns=['gvkey_x']).rename(columns={'gvkey_y': 'gvkey'})
    return m, diag


link_diag = {}
# The pre-join panel is kept so specification (i) can be validated against
# the logged V1/V2 gate numbers on the SAME sample those gates used. The CCM
# join necessarily shrinks the panel (see rows_dropped_by_link_date_window),
# so running the validation baseline on the joined panel would not reproduce
# them - and a near-miss would be indistinguishable from a harness bug.
panels_prejoin = {tag: panels[tag].copy() for tag, _, _ in SIGNALS}
for tag, xcol, rel in SIGNALS:
    panels[tag], link_diag[tag] = attach_gvkey(panels[tag], tag)

# ==========================================================================
# 4. FAMA-MACBETH MACHINERY (V1/V2's locked methodology, generalised)
# ==========================================================================


def fama_macbeth(df, ycol, xcols, min_xsec=MIN_XSEC):
    """Daily cross-sectional OLS of ycol on [const] + xcols; NW t-test
    (maxlags=NW_MAXLAGS) on each daily coefficient series. Returns a dict
    keyed by regressor name plus bookkeeping."""
    use = df.dropna(subset=[ycol] + xcols)
    k = len(xcols) + 1
    coefs, dates, ns = [], [], []
    n_dropped = 0
    for dt, gg in use.groupby('DlyCalDt', sort=True):
        if len(gg) < min_xsec:
            n_dropped += 1
            continue
        X = np.column_stack([np.ones(len(gg))] +
                            [gg[c].to_numpy(float) for c in xcols])
        y = gg[ycol].to_numpy(float)
        if not np.isfinite(X).all() or np.linalg.matrix_rank(X) < k:
            n_dropped += 1
            continue
        coefs.append(np.linalg.lstsq(X, y, rcond=None)[0])
        dates.append(dt)
        ns.append(len(gg))
    if not coefs:
        return None
    C = np.asarray(coefs)
    out = {'n_dates': len(dates), 'n_dates_dropped': n_dropped,
           'mean_xsec_n': float(np.mean(ns)), 'coefs': {}}
    for j, name in enumerate(['const'] + xcols):
        s = C[:, j]
        m = sm.OLS(s, np.ones(len(s))).fit(cov_type='HAC',
                                           cov_kwds={'maxlags': NW_MAXLAGS})
        out['coefs'][name] = {
            'mean_coef': float(m.params[0]),
            'nw_t': float(m.tvalues[0]),
            'std_daily': float(s.std()),
            'pct_days_negative': float((s < 0).mean() * 100),
        }
    return out


# ==========================================================================
# 5. SPECIFICATION (i) - BASELINE, runs without RDQ
# ==========================================================================
print('\n' + '=' * 88)
print('5. SPECIFICATION (i) BASELINE - compression -> forward realized vol')
print('   (no earnings term; reproduces V1/V2 to validate this harness)')
print('=' * 88)

baseline = {}
for tag, xcol, rel in SIGNALS:
    df = panels_prejoin[tag]        # PRE-join: the sample V1/V2 gated on
    res = fama_macbeth(df, PRIMARY_TARGET, [xcol])
    baseline[tag] = res
    c = res['coefs'][xcol]
    print(f"\n  {tag}: {PRIMARY_TARGET} ~ {xcol}")
    print(f"    daily cross-sections: {res['n_dates']:,} "
          f"(dropped {res['n_dates_dropped']:,}; mean n per date "
          f"{res['mean_xsec_n']:.0f})")
    print(f"    mean daily slope: {c['mean_coef']:+.6f}   "
          f"NW t (maxlags={NW_MAXLAGS}): {c['nw_t']:+.3f}")
    print(f"    negative-slope days: {c['pct_days_negative']:.1f}%")

print(f"\n  Estimated on the PRE-CCM-join panel - the same sample src/11 and")
print(f"  src/18 gated on - so this is a like-for-like reproduction.")
print(f"  CROSS-CHECK against results/gate_log.md (V1 dev and V2 dev gate")
print(f"  entries). This harness adds a fitted intercept where src/11 and")
print(f"  src/18 used an algebraically identical closed-form slope, so these")
print(f"  should reproduce the logged numbers. Nothing here is transcribed.")

# Secondary horizons, baseline only
print(f"\n  Secondary targets (no gate authority):")
baseline_secondary = {}
for tag, xcol, rel in SIGNALS:
    baseline_secondary[tag] = {}
    for tgt in SECONDARY_TARGETS:
        r = fama_macbeth(panels_prejoin[tag], tgt, [xcol])
        if r is None:
            continue
        c = r['coefs'][xcol]
        baseline_secondary[tag][tgt] = {'mean_coef': c['mean_coef'],
                                        'nw_t': c['nw_t'],
                                        'n_dates': r['n_dates']}
        print(f"    {tag} {tgt}: mean slope {c['mean_coef']:+.6f}   "
              f"NW t {c['nw_t']:+.3f}   ({r['n_dates']:,} dates)")

# ==========================================================================
# 6. THE RDQ-DEPENDENT PATH
# ==========================================================================
result = {
    'phase': 'Earnings-timing confound test for V1/V2 volume compression',
    'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'script': 'src/43_earnings_confound.py',
    'window': f'DEV to {DEV_END.date()} - holdout already spent for V1 and V2',
    'pre_commitment': precommit,
    'discrepancies': {
        'horizon': {
            'task_stated': 'forward 30-day realized vol, [t+1, t+30] window',
            'project_actual': (
                f'{PRIMARY_TARGET} = {PRIMARY_HORIZON_TDAYS} TRADING days '
                f'from t+1, PRIMARY per the locked V1 pre-registration '
                f'(commit 4474312); 5d and 20d secondary; no 30-day target '
                f'exists in the project'),
            'consequence': (
                f'a {PRIMARY_HORIZON_TDAYS}-trading-day window covers ~16% of '
                f'a ~63-day quarter, not "roughly half" - the confound is a '
                f'priori weaker than the framing assumes'),
            'resolution': (
                'PRIMARY test uses the actual locked target with a '
                'horizon-matched earnings window; the requested '
                f'{REQUESTED_CALENDAR_WINDOW_DAYS}-calendar-day window is '
                'also computed and reported in full'),
        },
        'rdq_availability': {
            'task_stated': 'join Compustat RDQ via the existing CCM link',
            'project_actual': (
                'RDQ is not in this repository. The Compustat pull is '
                'ccm_link_gics.csv (the link) and compustat_gics_names.csv '
                '(ANNUAL GICS/company file, ~8 rows per gvkey, whose '
                'datadate is a FISCAL PERIOD END, not a report date).'),
            'datadate_substitution_refused': (
                'A fiscal period end precedes the announcement by a '
                'firm-specific, time-varying lag. A dummy built from it '
                'would be mismeasured in a way correlated with firm '
                'identity - exactly the error this test exists to detect - '
                'and a null would be uninterpretable.'),
            'files_inspected': [
                {'file': n, 'columns': (c if isinstance(c, str) else list(c))}
                for n, c in files_seen],
            'unblock_query': (
                "SELECT gvkey, datadate, rdq, fyearq, fqtr FROM comp.fundq "
                "WHERE indfmt='INDL' AND datafmt='STD' AND popsrc='D' AND "
                "consol='C' AND datadate BETWEEN '2014-06-01' AND "
                "'2026-06-30'"),
            'unblock_instruction': (
                'drop the pull in data/raw/compustat/ under any filename; '
                'discovery is by column (gvkey + rdq), not by name'),
        },
    },
    'method': {
        'estimator': ('Fama-MacBeth: daily cross-sectional OLS, Newey-West '
                      f'maxlags={NW_MAXLAGS} on the daily coefficient series '
                      '- V1/V2 locked methodology, extended to multivariate'),
        'min_cross_section': MIN_XSEC,
        'primary_target': PRIMARY_TARGET,
        'primary_horizon_trading_days': PRIMARY_HORIZON_TDAYS,
        'secondary_targets': list(SECONDARY_TARGETS),
        'requested_calendar_window_days': REQUESTED_CALENDAR_WINDOW_DAYS,
        'shift1_alignment': (
            'compression inputs are already lagged to end at t-1 by src/10 '
            'and are REUSED here, not recomputed; forward targets span '
            't+1..t+N; the earnings dummy is built from the forward window '
            'only. No term includes day t or crosses it.'),
    },
    'ccm_link': {
        'file': 'data/raw/compustat/ccm_link_gics.csv',
        'rows_as_pulled': n_link_raw,
        'filter_trail': {'linktype_LC_LU': n1, 'linkprim_P_C': n2,
                         'non_null_lpermno': n3},
        'still_active_links_E': n_active,
        'per_signal': link_diag,
    },
    'specification_i_baseline': {
        tag: {
            'target': PRIMARY_TARGET,
            'regressor': xcol,
            'sample': ('PRE-CCM-join DEV panel - the same sample src/11 and '
                       'src/18 gated on, so this reproduces the logged V1/V2 '
                       'numbers like-for-like'),
            'n_dates': baseline[tag]['n_dates'],
            'n_dates_dropped': baseline[tag]['n_dates_dropped'],
            'mean_xsec_n': baseline[tag]['mean_xsec_n'],
            'mean_coef': baseline[tag]['coefs'][xcol]['mean_coef'],
            'nw_t': baseline[tag]['coefs'][xcol]['nw_t'],
            'pct_days_negative': baseline[tag]['coefs'][xcol]['pct_days_negative'],
            'secondary_targets': baseline_secondary[tag],
        } for tag, xcol, rel in SIGNALS},
}

if not RDQ_AVAILABLE:
    print('\n' + '!' * 88)
    print('! BLOCKED - specifications (ii) and (iii) require RDQ, which is '
          'not present')
    print('!' * 88)
    print("""
  What ran and is reported:
    - the pre-commitment, locked above and written to the JSON
    - the CCM link build and full join diagnostics
    - specification (i), the baseline, for V1 and V2, primary and secondary

  What did NOT run:
    - step 1, the earnings-in-window flag
    - step 2, the earnings rate by compression decile
    - specifications (ii) dummy control and (iii) interaction
    - the pre-committed rule evaluation

  NO RULE IS TRIGGERED. The test did not run, so it has no outcome - this
  is NOT an "inconclusive" result under RULE 3, which describes a test that
  ran and produced ambiguous numbers. Recording it as RULE 3 would overstate
  what is known.

  gate_log.md is NOT written, matching the precedent set by src/46 when
  Test 1a proved not estimable: a test that did not run does not get a
  gate_log block.

  To unblock, pull from WRDS into data/raw/compustat/ (any filename):

     SELECT gvkey, datadate, rdq, fyearq, fqtr
     FROM comp.fundq
     WHERE indfmt='INDL' AND datafmt='STD'
       AND popsrc='D'    AND consol='C'
       AND datadate BETWEEN '2014-06-01' AND '2026-06-30'
""")
    print('!' * 88)
    result['status'] = 'BLOCKED - RDQ unavailable'
    result['rule_triggered'] = None
    result['rule_triggered_note'] = (
        'No rule is triggered: the test did not run. This is NOT RULE 3, '
        'which describes a test that ran and produced ambiguous numbers.')
    result['steps_not_run'] = [
        'earnings-in-window flag', 'earnings rate by compression decile',
        'specification (ii) dummy control',
        'specification (iii) interaction', 'rule evaluation']
    result['gate_log_written'] = False
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, default=str),
                        encoding='utf-8')
    print(f"\n[OK] Saved {out_json}")
    print(f"[OK] gate_log.md deliberately NOT written (test did not run)")
    print('\n' + '=' * 88)
    print('HALTED - pre-commitment locked, link and baseline reported, '
          'confound test pending RDQ.')
    print('=' * 88)
    raise SystemExit(0)

# --------------------------------------------------------------------------
# 6a. Load RDQ and join to the panels
# --------------------------------------------------------------------------
print('\n' + '-' * 88)
print('6. EARNINGS DATES (RDQ) - load, join, diagnostics')
print('-' * 88)

if rdq_path.suffix == '.csv':
    rdq = pd.read_csv(rdq_path)
else:
    rdq = pd.read_parquet(rdq_path)
rdq.columns = [c.lower() for c in rdq.columns]
n_rdq_raw = len(rdq)
print(f"  RDQ file: {rdq_path}")
print(f"  rows as pulled: {n_rdq_raw:,}   columns: {list(rdq.columns)}")

# Standard fundq filters, applied only if the columns are present (a pull
# already filtered server-side will not carry them).
for col, keep in [('indfmt', 'INDL'), ('datafmt', 'STD'),
                  ('popsrc', 'D'), ('consol', 'C')]:
    if col in rdq.columns:
        before = len(rdq)
        rdq = rdq[rdq[col] == keep]
        print(f"    filter {col} == {keep}: {before:,} -> {len(rdq):,}")

n_rdq_null = int(rdq['rdq'].isna().sum())
print(f"  RDQ null rate: {n_rdq_null:,} of {len(rdq):,} "
      f"({n_rdq_null / max(len(rdq), 1) * 100:.2f}%)")
rdq['rdq'] = pd.to_datetime(rdq['rdq'], errors='coerce')
n_unparsed = int(rdq['rdq'].isna().sum()) - n_rdq_null
print(f"  RDQ values that failed to parse as dates: {n_unparsed:,}")
rdq = rdq.dropna(subset=['rdq'])
rdq['gvkey'] = pd.to_numeric(rdq['gvkey'], errors='coerce')
rdq = rdq.dropna(subset=['gvkey'])
rdq['gvkey'] = rdq['gvkey'].astype(int)
rdq = rdq[['gvkey', 'rdq']].drop_duplicates().sort_values(['gvkey', 'rdq'])
print(f"  usable (gvkey, rdq) pairs: {len(rdq):,}   "
      f"gvkeys {rdq['gvkey'].nunique():,}")
print(f"  RDQ date range: {rdq['rdq'].min().date()} to "
      f"{rdq['rdq'].max().date()}")

# Announcements per gvkey per year - a sanity check that this is quarterly
_per_year = (rdq.assign(y=rdq['rdq'].dt.year)
             .groupby(['gvkey', 'y']).size())
print(f"  announcements per gvkey-year: median "
      f"{_per_year.median():.1f} (expect ~4 for quarterly reporting)")

rdq_by_gvkey = {g: s['rdq'].to_numpy() for g, s in rdq.groupby('gvkey')}

# --------------------------------------------------------------------------
# 6b. Earnings-in-window flags
# --------------------------------------------------------------------------
cal = pd.DatetimeIndex(sorted(panels['V1']['DlyCalDt'].unique()))


def earnings_flags(panel):
    """Flag whether an RDQ falls in each forward window.

    Horizon-matched windows use the CALENDAR span from session t+1 through
    session t+h, so the dummy covers exactly the days the forward realized
    vol is measured over. The requested window uses t+1 .. t+30 calendar
    days. Both are half-open on the left (strictly after t) so day t is
    never included - matching the look-ahead convention throughout.
    """
    full_cal = pd.DatetimeIndex(
        sorted(pd.read_parquet(
            project_root / 'data' / 'processed' / 'crsp_combined.parquet',
            columns=['DlyCalDt'])['DlyCalDt'].unique()))
    pos = full_cal.get_indexer(panel['DlyCalDt'])
    d0 = panel['DlyCalDt'].to_numpy()
    horizons = {PRIMARY_TARGET: PRIMARY_HORIZON_TDAYS}
    horizons.update(SECONDARY_TARGETS)
    ends = {}
    for tgt, h in horizons.items():
        idx = np.clip(pos + h, 0, len(full_cal) - 1)
        ends[f'earnings_in_fwd_{h}d'] = full_cal.to_numpy()[idx]
    ends[f'earnings_in_cal_{REQUESTED_CALENDAR_WINDOW_DAYS}d'] = (
        d0 + np.timedelta64(REQUESTED_CALENDAR_WINDOW_DAYS, 'D'))

    gv = panel['gvkey'].to_numpy()
    out = {}
    for name, hi in ends.items():
        flag = np.full(len(panel), np.nan)
        order = np.argsort(gv, kind='mergesort')
        gs = gv[order]
        bounds = np.searchsorted(gs, np.unique(gs[~np.isnan(gs)]),
                                 side='left')
        for g in np.unique(gs[~np.isnan(gs)]):
            arr = rdq_by_gvkey.get(int(g))
            sel = order[gs == g]
            if arr is None or len(arr) == 0:
                flag[sel] = np.nan
                continue
            lo_i = np.searchsorted(arr, d0[sel], side='right')
            hi_i = np.searchsorted(arr, hi[sel], side='right')
            flag[sel] = (hi_i > lo_i).astype(float)
        out[name] = flag
    return out


for tag, xcol, rel in SIGNALS:
    fl = earnings_flags(panels[tag])
    for k, v in fl.items():
        panels[tag][k] = v
    cov = panels[tag][f'earnings_in_fwd_{PRIMARY_HORIZON_TDAYS}d'].notna()
    print(f"\n  {tag}: rows with a usable earnings flag: {int(cov.sum()):,} "
          f"of {len(panels[tag]):,} ({cov.mean() * 100:.2f}%)")
    print(f"      rows dropped for no RDQ history on the linked gvkey: "
          f"{int((~cov).sum()):,}")

PRIMARY_DUMMY = f'earnings_in_fwd_{PRIMARY_HORIZON_TDAYS}d'
REQUESTED_DUMMY = f'earnings_in_cal_{REQUESTED_CALENDAR_WINDOW_DAYS}d'

# ==========================================================================
# 7. STEP 2 - EARNINGS RATE BY DECILE, PRINTED BEFORE ANY REGRESSION
# ==========================================================================
print('\n' + '=' * 88)
print('7. EARNINGS-IN-WINDOW RATE BY COMPRESSION DECILE')
print('   (printed BEFORE any regression, so the raw pattern is visible '
       'on its own)')
print('=' * 88)

decile_tables = {}
for tag, xcol, rel in SIGNALS:
    df = panels[tag]
    decile_tables[tag] = {}
    for dummy, lbl in [(PRIMARY_DUMMY,
                        f'horizon-matched, {PRIMARY_HORIZON_TDAYS} trading '
                        f'days (PRIMARY)'),
                       (REQUESTED_DUMMY,
                        f'{REQUESTED_CALENDAR_WINDOW_DAYS} calendar days '
                        f'(as requested)')]:
        sub = df.dropna(subset=[dummy, xcol])
        tab = (sub.groupby(xcol)[dummy].agg(['mean', 'size']))
        tab['pct'] = tab['mean'] * 100
        spread = float(tab['pct'].max() - tab['pct'].min())
        decile_tables[tag][dummy] = {
            'label': lbl,
            'by_decile_pct': {int(k): float(v) for k, v in tab['pct'].items()},
            'counts': {int(k): int(v) for k, v in tab['size'].items()},
            'max_minus_min_pp': spread,
            'overall_pct': float(sub[dummy].mean() * 100),
        }
        print(f"\n  {tag} - earnings in window, {lbl}")
        print(f"    {'decile':>7} {'rate %':>9} {'n':>12}   "
              f"(1 = most compressed)")
        for k, r in tab.iterrows():
            print(f"    {int(k):>7} {r['pct']:>9.2f} {int(r['size']):>12,}")
        print(f"    overall {sub[dummy].mean() * 100:.2f}%   "
              f"max - min across deciles = {spread:.2f} pp   "
              f"(RULE 1a bar: <= {RULE1_MAX_DECILE_SPREAD_PP:.1f} pp -> "
              f"{'MET' if spread <= RULE1_MAX_DECILE_SPREAD_PP else 'NOT MET'})")

# ==========================================================================
# 8. SPECIFICATIONS (i) (ii) (iii) SIDE BY SIDE
# ==========================================================================
print('\n' + '=' * 88)
print('8. THREE SPECIFICATIONS, SIDE BY SIDE')
print('=' * 88)

specs = {}
for tag, xcol, rel in SIGNALS:
    df = panels[tag]
    # Common estimation sample across all three, so the comparison is not
    # contaminated by a changing sample.
    common = df.dropna(subset=[PRIMARY_TARGET, xcol, PRIMARY_DUMMY]).copy()
    common['_interact'] = common[xcol] * common[PRIMARY_DUMMY]
    print(f"\n  {tag}: common estimation sample {len(common):,} rows "
          f"(all three specifications use it)")

    s_i = fama_macbeth(common, PRIMARY_TARGET, [xcol])
    s_ii = fama_macbeth(common, PRIMARY_TARGET, [xcol, PRIMARY_DUMMY])
    s_iii = fama_macbeth(common, PRIMARY_TARGET,
                         [xcol, PRIMARY_DUMMY, '_interact'])
    specs[tag] = {'i': s_i, 'ii': s_ii, 'iii': s_iii}

    print(f"\n    {'term':<26} {'(i) base':>18} {'(ii) +dummy':>18} "
          f"{'(iii) +interact':>18}")
    for term, disp in [(xcol, 'compression'),
                       (PRIMARY_DUMMY, 'earnings dummy'),
                       ('_interact', 'compression x dummy'),
                       ('const', 'intercept')]:
        cells = []
        for sp in (s_i, s_ii, s_iii):
            c = sp['coefs'].get(term)
            cells.append(f"{c['mean_coef']:+.6f}" if c else '-')
        print(f"    {disp:<26} {cells[0]:>18} {cells[1]:>18} {cells[2]:>18}")
        cells = []
        for sp in (s_i, s_ii, s_iii):
            c = sp['coefs'].get(term)
            cells.append(f"t {c['nw_t']:+.3f}" if c else '')
        print(f"    {'':<26} {cells[0]:>18} {cells[1]:>18} {cells[2]:>18}")
    print(f"    {'daily cross-sections':<26} {s_i['n_dates']:>18,} "
          f"{s_ii['n_dates']:>18,} {s_iii['n_dates']:>18,}")

# ==========================================================================
# 9. PRE-COMMITTED RULE EVALUATION
# ==========================================================================
print('\n' + '=' * 88)
print('9. PRE-COMMITTED RULE EVALUATION')
print('=' * 88)

verdicts = {}
for tag, xcol, rel in SIGNALS:
    b_i = specs[tag]['i']['coefs'][xcol]
    b_ii = specs[tag]['ii']['coefs'][xcol]
    spread = decile_tables[tag][PRIMARY_DUMMY]['max_minus_min_pp']

    r1a = bool(spread <= RULE1_MAX_DECILE_SPREAD_PP)
    r1b = bool(abs(b_ii['nw_t']) >= RULE1_MIN_ABS_T)
    rule1 = bool(r1a and r1b)
    rule2 = bool(abs(b_ii['mean_coef']) <
                 RULE2_MAGNITUDE_RETENTION * abs(b_i['mean_coef']))
    sign_flip = bool(np.sign(b_ii['mean_coef']) != np.sign(b_i['mean_coef']))
    retention = (abs(b_ii['mean_coef']) / abs(b_i['mean_coef'])
                 if b_i['mean_coef'] != 0 else float('nan'))

    if sign_flip:
        rule = 3
        why = ('sign flip between (i) and (ii) - forces RULE 3 by the '
               'pre-committed sign-flip rule')
    elif rule1 and rule2:
        rule = 3
        why = ('RULE 1 and RULE 2 both fire - forces RULE 3 by the '
               'pre-committed conflict rule; the more favorable was not '
               'selected')
    elif rule1:
        rule = 1
        why = 'RULE 1 met: decile spread within bar AND |t| retained'
    elif rule2:
        rule = 2
        why = 'RULE 2 met: coefficient lost more than half its magnitude'
    else:
        rule = 3
        why = 'neither RULE 1 nor RULE 2 conditions met'

    verdicts[tag] = {
        'rule_triggered': rule, 'reason': why,
        'rule1a_decile_spread_pp': spread,
        'rule1a_met': r1a, 'rule1b_abs_t': abs(b_ii['nw_t']),
        'rule1b_met': r1b, 'rule1_met': rule1,
        'rule2_met': rule2,
        'coef_i': b_i['mean_coef'], 'coef_ii': b_ii['mean_coef'],
        'magnitude_retention': retention,
        'sign_flip': sign_flip,
        't_i': b_i['nw_t'], 't_ii': b_ii['nw_t'],
    }
    print(f"\n  {tag}")
    print(f"    1a decile spread {spread:.2f} pp "
          f"(bar <= {RULE1_MAX_DECILE_SPREAD_PP:.1f})      "
          f"{'MET' if r1a else 'NOT MET'}")
    print(f"    1b |t| with dummy {abs(b_ii['nw_t']):.3f} "
          f"(bar >= {RULE1_MIN_ABS_T:.1f})        "
          f"{'MET' if r1b else 'NOT MET'}")
    print(f"    RULE 1 (both)                          "
          f"{'MET' if rule1 else 'NOT MET'}")
    print(f"    coefficient {b_i['mean_coef']:+.6f} -> "
          f"{b_ii['mean_coef']:+.6f}  (retains {retention * 100:.1f}%)")
    print(f"    RULE 2 (< {RULE2_MAGNITUDE_RETENTION * 100:.0f}% retained)  "
          f"                {'MET' if rule2 else 'NOT MET'}")
    print(f"    sign flip: {sign_flip}")
    print(f"    -> RULE {rule} TRIGGERED: {why}")

rules_hit = sorted({v['rule_triggered'] for v in verdicts.values()})
print(f"\n  Rules triggered across signals: "
      + ', '.join(f"{t}=RULE {v['rule_triggered']}"
                  for t, v in verdicts.items()))

result['status'] = 'COMPLETE'
result['rdq'] = {
    'file': str(rdq_path),
    'rows_as_pulled': n_rdq_raw,
    'usable_pairs': int(len(rdq)),
    'gvkeys': int(rdq['gvkey'].nunique()),
    'null_rate_pct': float(n_rdq_null / max(n_rdq_raw, 1) * 100),
    'date_range': [str(rdq['rdq'].min().date()), str(rdq['rdq'].max().date())],
    'median_announcements_per_gvkey_year': float(_per_year.median()),
}
result['earnings_rate_by_decile'] = decile_tables
result['specifications'] = {
    tag: {k: {'n_dates': v['n_dates'],
              'n_dates_dropped': v['n_dates_dropped'],
              'coefs': v['coefs']}
          for k, v in specs[tag].items()} for tag, _, _ in SIGNALS}
result['rule_evaluation'] = verdicts
result['gate_log_written'] = True
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(result, indent=2, default=str),
                    encoding='utf-8')
print(f"\n[OK] Saved {out_json}")

# ==========================================================================
# 10. Append-only, marker-guarded gate_log entry. Numbers only.
# ==========================================================================
MARKER = '## V1/V2 earnings-timing confound test (DEV)'
log_text = log_path.read_text(encoding='utf-8', errors='replace')
if MARKER in log_text:
    print('[SKIP] gate_log.md already carries the earnings-confound block; '
          'not duplicating.')
else:
    bak = log_path.with_suffix('.md.bak_before_earnings_confound_block')
    shutil.copy2(log_path, bak)
    print(f"[safety] gate_log.md backed up to {bak.name}")
    L = [
        f"\n---\n\n{MARKER}",
        "",
        f"Run {result['generated']} - script src/43_earnings_confound.py. "
        f"Tests whether the V1/V2 volume-compression signal is a proxy for "
        f"earnings-announcement timing. Rules pre-committed before any "
        f"return or volatility data was read. DEV only; the 2022-2025 "
        f"holdout is already spent for V1 and V2 and is not touched. "
        f"Machine-readable copy: results/43_earnings_confound.json.",
        "",
        f"Primary target {PRIMARY_TARGET} ({PRIMARY_HORIZON_TDAYS} trading "
        f"days from t+1) per the locked V1 pre-registration; the earnings "
        f"dummy is horizon-matched. The {REQUESTED_CALENDAR_WINDOW_DAYS}"
        f"-calendar-day window is also reported. RDQ source: "
        f"{Path(rdq_path).name}.",
        "",
        "```",
        f"pre-committed bars: 1a decile spread <= "
        f"{RULE1_MAX_DECILE_SPREAD_PP:.1f} pp; 1b |NW t| >= "
        f"{RULE1_MIN_ABS_T:.1f} with dummy;",
        f"                    rule 2 if |beta(ii)| < "
        f"{RULE2_MAGNITUDE_RETENTION:.2f}*|beta(i)|",
        "",
    ]
    for tag, xcol, rel in SIGNALS:
        dt = decile_tables[tag][PRIMARY_DUMMY]
        v = verdicts[tag]
        L += [
            f"{tag}  ({xcol})",
            f"  earnings-in-window rate by decile (%), 1 = most compressed:",
        ]
        L.append("    " + "  ".join(
            f"{d}:{dt['by_decile_pct'][d]:.2f}"
            for d in sorted(dt['by_decile_pct'])))
        L += [
            f"    overall {dt['overall_pct']:.2f}%   max-min "
            f"{dt['max_minus_min_pp']:.2f} pp",
            f"  (i)   compression coef {v['coef_i']:+.6f}   NW t "
            f"{v['t_i']:+.3f}",
            f"  (ii)  compression coef {v['coef_ii']:+.6f}   NW t "
            f"{v['t_ii']:+.3f}   (retains "
            f"{v['magnitude_retention'] * 100:.1f}%)",
        ]
        c3 = specs[tag]['iii']['coefs']
        L += [
            f"  (iii) compression {c3[xcol]['mean_coef']:+.6f} "
            f"(t {c3[xcol]['nw_t']:+.3f})   "
            f"dummy {c3[PRIMARY_DUMMY]['mean_coef']:+.6f} "
            f"(t {c3[PRIMARY_DUMMY]['nw_t']:+.3f})   "
            f"interaction {c3['_interact']['mean_coef']:+.6f} "
            f"(t {c3['_interact']['nw_t']:+.3f})",
            f"  1a {'MET' if v['rule1a_met'] else 'NOT MET'}   "
            f"1b {'MET' if v['rule1b_met'] else 'NOT MET'}   "
            f"RULE 1 {'MET' if v['rule1_met'] else 'NOT MET'}   "
            f"RULE 2 {'MET' if v['rule2_met'] else 'NOT MET'}   "
            f"sign flip {v['sign_flip']}",
            f"  RULE TRIGGERED: {v['rule_triggered']}",
            "",
        ]
    L += ["```", ""]
    with log_path.open('a', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print(f"[OK] Appended earnings-confound block to {log_path} "
          f"(prior entries untouched)")

print('\n' + '=' * 88)
print('EARNINGS-CONFOUND TEST COMPLETE - '
      + ', '.join(f"{t}: RULE {v['rule_triggered']}"
                  for t, v in verdicts.items()))
print('=' * 88)
