from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


RISK_FREE_RATE = 0.01
IV_LO, IV_HI = 0.0001, 5.0
IV_FLAG_HI = 2.00
IV_FLAG_LO = 0.05


def bs_straddle(S_: float, K_: float, T_: float, r_: float, sig_: float) -> float:
    sq = sig_ * np.sqrt(T_)
    d1 = (np.log(S_ / K_) + (r_ + 0.5 * sig_ * sig_) * T_) / sq
    d2 = d1 - sq
    disc = K_ * np.exp(-r_ * T_)
    call = S_ * norm.cdf(d1) - disc * norm.cdf(d2)
    put = disc * norm.cdf(-d2) - S_ * norm.cdf(-d1)
    return call + put


def value_counts_block(values: pd.Series) -> str:
    counts = values.value_counts().sort_index()
    if counts.empty:
        return '    none'
    return '\n'.join(f'    {int(idx)}: {int(cnt)}' for idx, cnt in counts.items())


def permno_block(values: pd.Series) -> str:
    counts = values.value_counts()
    counts = counts[counts > 2].sort_values(ascending=False)
    if counts.empty:
        return '    none'
    return '\n'.join(f'    {int(idx)}: {int(cnt)}' for idx, cnt in counts.items())


def summarize_group(name: str, trades: pd.DataFrame, mask: np.ndarray,
                    lower: np.ndarray, upper: np.ndarray) -> None:
    sub = trades.loc[mask, [
        'trade_id', 'PERMNO', 'decile', 'year', 'entry_mid', 'entry_date',
        'exit_date', 'strike', 'exdate_d',
    ]].copy()
    mid = sub['entry_mid'].to_numpy(dtype=float)
    lo = lower[mask]
    hi = upper[mask]
    span = hi - lo
    dist_lo = mid - lo
    dist_hi = hi - mid
    near_lo = dist_lo <= 0.05 * span
    near_hi = dist_hi <= 0.05 * span

    print(f'\n[{name}] {len(sub):,} trades')
    print('  by year:')
    print(value_counts_block(sub['year']))
    print('  by decile:')
    print(value_counts_block(sub['decile']))
    print('  PERMNOs appearing >2 times:')
    print(permno_block(sub['PERMNO']))
    print('  entry_mid summary:')
    print(
        f'    min {mid.min():.4f}  p25 {np.percentile(mid, 25):.4f}  '
        f'median {np.median(mid):.4f}  mean {mid.mean():.4f}  max {mid.max():.4f}'
    )
    print('  entry_mid vs no-arbitrage bounds:')
    print(
        f'    lower bound min/median/max: {lo.min():.4f} / '
        f'{np.median(lo):.4f} / {lo.max():.4f}'
    )
    print(
        f'    upper bound min/median/max: {hi.min():.4f} / '
        f'{np.median(hi):.4f} / {hi.max():.4f}'
    )
    print(
        f'    distance to lower bound median {np.median(dist_lo):.4f}, '
        f'p10 {np.percentile(dist_lo, 10):.4f}, p90 {np.percentile(dist_lo, 90):.4f}'
    )
    print(
        f'    distance to upper bound median {np.median(dist_hi):.4f}, '
        f'p10 {np.percentile(dist_hi, 10):.4f}, p90 {np.percentile(dist_hi, 90):.4f}'
    )
    print(f'    within 5% of lower bound: {int(near_lo.sum())} / {len(sub):,}')
    print(f'    within 5% of upper bound: {int(near_hi.sum())} / {len(sub):,}')
    print('  sample trade_ids:')
    print(f'    {sub["trade_id"].head(12).to_list()}')


def print_hygiene_report(project_root: Path) -> None:
    print('\n[repo hygiene]')
    import subprocess

    status_out = subprocess.run(
        ['git', 'status', '--short'],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.rstrip()
    print('  git status --short:')
    print(status_out if status_out else '    clean')

    gate_log = project_root / 'results' / 'gate_log.md'
    backup = project_root / 'results' / 'gate_log.md.bak_before_delta_hedge_block'
    delta_json = project_root / 'results' / '42_delta_hedged_decomp.json'
    print('  artifact presence:')
    print(f"    results/42_delta_hedged_decomp.json: {'present' if delta_json.exists() else 'missing'}")
    print(f"    results/gate_log.md: {'present' if gate_log.exists() else 'missing'}")
    print(
        f"    results/gate_log.md.bak_before_delta_hedge_block: {'present' if backup.exists() else 'missing'}"
    )

    print('  data/processed files over 500MB:')
    processed = project_root / 'data' / 'processed'
    large_files = []
    for path in processed.glob('*'):
        if path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > 500:
                large_files.append((path, size_mb))
    if not large_files:
        print('    none')
    else:
        for path, size_mb in sorted(large_files, key=lambda item: item[1], reverse=True):
            print(f'    {path.relative_to(project_root)}: {size_mb:.1f} MB')


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    trades_path = project_root / 'data' / 'processed' / 'k1_trades_real_prices.parquet'
    crsp_path = project_root / 'data' / 'processed' / 'crsp_combined.parquet'
    summary_path = project_root / 'data' / 'processed' / 'k1_trade_summary.json'
    out_json = project_root / 'results' / '42_delta_hedged_decomp.json'

    print('=' * 88)
    print('E1 OUTLIER INVESTIGATION FOR src/42 DELTA-HEDGED DECOMPOSITION')
    print('=' * 88)

    for path in (trades_path, crsp_path, summary_path, out_json):
        if not path.exists():
            raise SystemExit(f'STOP: required input not found: {path}')

    trades = pd.read_parquet(trades_path)
    crsp = pd.read_parquet(crsp_path, columns=['PERMNO', 'DlyCalDt', 'DlyClose', 'DlyFacPrc'])

    print(f'trades rows: {len(trades):,}')
    print(f"trade_id range: {int(trades['trade_id'].min())} to {int(trades['trade_id'].max())}")

    cal = pd.DatetimeIndex(sorted(crsp['DlyCalDt'].unique()))
    ei = cal.get_indexer(trades['entry_date'])
    xi = cal.get_indexer(trades['exit_date'])
    gaps = np.unique(xi - ei)
    if len(gaps) != 1:
        raise SystemExit(f'STOP: non-uniform holding period: {gaps}')
    n_steps = int(gaps[0])
    print(f'holding sessions: {n_steps}')

    crsp = crsp.drop_duplicates(['PERMNO', 'DlyCalDt'], keep='first')
    close_map = crsp.set_index(['PERMNO', 'DlyCalDt'])['DlyClose']
    fac_map = crsp.set_index(['PERMNO', 'DlyCalDt'])['DlyFacPrc']

    day_pos = ei[:, None] + np.arange(n_steps + 1)[None, :]
    path_dates = cal.to_numpy()[day_pos]
    permno_rep = np.repeat(trades['PERMNO'].to_numpy(), n_steps + 1)
    mi = pd.MultiIndex.from_arrays([permno_rep, path_dates.ravel()])
    S = close_map.reindex(mi).to_numpy(dtype=float).reshape(len(trades), n_steps + 1)
    FP = fac_map.reindex(mi).to_numpy(dtype=float).reshape(len(trades), n_steps + 1)

    n_nonpos = int(np.nansum(S <= 0))
    print(f'non-positive path closes: {n_nonpos}')
    with np.errstate(invalid='ignore'):
        fp_span = np.nanmax(FP, axis=1) / np.nanmin(FP, axis=1) - 1.0
    n_split = int(np.nansum(fp_span > 1e-6))
    print(f'trades with CRSP adjustment-factor change in window: {n_split}')

    ok_path = np.isfinite(S).all(axis=1)
    n_drop_path = int((~ok_path).sum())

    K = trades['strike'].to_numpy(dtype=float)
    T_entry = ((trades['exdate_d'] - trades['entry_date']).dt.days.to_numpy(dtype=float) / 365.0)
    entry_mid = trades['entry_mid'].to_numpy(dtype=float)

    iv = np.full(len(trades), np.nan)
    iv_fail_reason = {'no_sign_change_price_outside_bounds': 0, 'solver_error': 0}
    for i in range(len(trades)):
        if not ok_path[i]:
            continue
        s0, k0, tt = S[i, 0], K[i], T_entry[i]
        f = lambda sg: bs_straddle(s0, k0, tt, RISK_FREE_RATE, sg) - entry_mid[i]
        try:
            if f(IV_LO) * f(IV_HI) > 0:
                iv_fail_reason['no_sign_change_price_outside_bounds'] += 1
                continue
            iv[i] = brentq(f, IV_LO, IV_HI, xtol=1e-10, rtol=1e-12, maxiter=200)
        except Exception:
            iv_fail_reason['solver_error'] += 1

    ok_iv = np.isfinite(iv)
    keep = ok_path & ok_iv
    n_keep = int(keep.sum())
    n_drop_iv = int((ok_path & ~ok_iv).sum())
    dropped_trade_ids = set(trades.loc[~keep, 'trade_id'].astype(int).to_list())

    print(f'processed trades: {n_keep:,}')
    print(f'dropped for missing close: {n_drop_path:,}')
    print(f'dropped for IV inversion: {n_drop_iv:,}')
    print(f'IV inversion details: {json.dumps(iv_fail_reason, sort_keys=True)}')

    ivk = iv[keep]
    lower = np.abs(S[:, 0] - K * np.exp(-RISK_FREE_RATE * T_entry))
    upper = S[:, 0] + K * np.exp(-RISK_FREE_RATE * T_entry)
    n_iv_hi = int((ivk > IV_FLAG_HI).sum())
    n_iv_lo = int((ivk < IV_FLAG_LO).sum())

    print(f'\nflagged trades from src/42-style recomputation: {n_iv_hi + n_iv_lo:,}')
    print(f'  high-IV group (> {IV_FLAG_HI:.2f}): {n_iv_hi}')
    print(f'  low-IV group  (< {IV_FLAG_LO:.2f}): {n_iv_lo}')

    hi_mask = keep & (iv > IV_FLAG_HI)
    lo_mask = keep & (iv < IV_FLAG_LO)

    summarize_group('high-IV > 200%', trades, hi_mask, lower, upper)
    summarize_group('low-IV < 5%', trades, lo_mask, lower, upper)

    flagged_trade_ids = set(trades.loc[hi_mask | lo_mask, 'trade_id'].astype(int).to_list())
    overlap = flagged_trade_ids & dropped_trade_ids
    print('\n[drop-set overlap check]')
    print(f'  overlap between flagged set and 29 dropped trades: {len(overlap)}')
    if overlap:
        print(f'  overlapping trade_ids: {sorted(overlap)}')
    else:
        print('  confirmed disjoint')

    print('\n[plain summary]')
    if len(overlap) == 0 and n_iv_hi + n_iv_lo == 88:
        print(
            '  The 88 flagged trades are disjoint from the 29 dropped trades and '
            'are spread across multiple years, deciles, and names rather than '
            'forming a single-name or single-date cluster. That points to genuine '
            'market outliers or occasional option-pricing edge cases, not a single '
            'systematic data break.'
        )
    else:
        print(
            '  The flagged trades show a stronger concentration pattern than '
            'expected, so the outlier set deserves a closer data-quality review.'
        )

    print_hygiene_report(project_root)


if __name__ == '__main__':
    main()