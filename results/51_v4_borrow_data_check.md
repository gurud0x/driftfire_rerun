# V4 Audit Item 6 — Short-Borrow Data Availability

**Return-blind data-inventory check. No return, P&L, or regression computed.**
`gate_log.md` not touched.

---

## 1. Data sources searched

This project has exactly four raw data families, confirmed by directory
listing of both `data/raw/` and the staging mirror
(`~/Downloads/quantdata/driftfire/raw/`, identical contents): **CRSP**,
**Compustat**, **OptionMetrics**, and **Fama-French factors**. Every file
in each family was inventoried:

| family | files | checked for |
|---|---|---|
| CRSP | `crsp_daily.parquet`, `crsp_names.parquet`, `crsp_sp500_members.parquet`, `crsp_combined.parquet` (processed) | borrow fee, HTB flag, shares available, lending utilization |
| Compustat | `ccm_link_gics.csv`, `compustat_gics_names.csv`, `rdq_pull_fundq_2014_2026.parquet` | same |
| OptionMetrics | `om_security_names.csv`, `vol_surface.csv`, `opprcd.csv` | same |
| Fama-French | 5-factor, momentum, short-term reversal daily files | same |

`crsp_combined.parquet`'s full column list was pulled and inspected
directly (not a keyword grep on a binary file, which is unreliable):

```
PERMNO, HdrCUSIP, CUSIP, PrimaryExch, TradingStatusFlg, IssuerType,
SecurityType, SecuritySubType, ShareType, DelActionType, DelReasonType,
Ticker, PERMCO, SICCD, DlyCalDt, DlyDelFlg, DlyPrcFlg, DlyCap, DlyRet,
DlyRetx, DlyRetMissFlg, DlyFacPrc, DlyVol, DlyClose, DlyLow, DlyHigh,
DlyBid, DlyAsk, DlyOpen, DlyNumTrd, DlyPrcVol, ShrOut, ShrAdrFlg, DisExDt,
DisOrdinaryFlg, DisDivAmt, DisFacPr, DisFacShr, vwretd, sprtrn, in_sp500,
IssuerNm
```

No borrow-fee, hard-to-borrow flag, shares-available, or lending-
utilization field exists. The two CSV files were text-grepped for
`borrow|lending|short.interest|htb|hard.to.borrow|utilization`; the only
hit was the literal substring `HTB` inside ticker symbols (`CHTB`,
`HTB.1`, `HTBB` — real company tickers, e.g. HomeTrust Bancshares), not a
data field. Compustat's fundamentals pull (`rdq_pull_fundq`) carries only
`costat, curcdq, datafmt, indfmt, consol, tic, datadate, gvkey, fqtr,
fyearq, rdq` — no short-interest or lending column of any kind.

**Finding: no point-in-time securities-lending source (borrow fee, HTB
flag, availability, or utilization) is available anywhere in this
environment.** This is a genuine data gap, not a naming mismatch — CRSP,
Compustat, and OptionMetrics are not securities-lending vendors; that data
typically comes from a separate feed (e.g. Markit/IHS Securities Finance,
S3 Partners, Orbisa/FIS, or a prime broker's own inventory), none of which
this project has ever pulled. Two of the connectors listed as available
but requiring authorization in this session — **S&P Global** and
**LSEG** — are exactly the kind of vendors that sometimes carry
securities-lending data as part of a broader market-data package, but they
are locked behind an OAuth flow this non-interactive session cannot
complete, and their presence is not a confirmation that either specific
feed includes lending data. This is noted as a possible future avenue,
not a resolution — **the finding above stands as of this audit and is not
assumed to be resolvable via those connectors.**

## 2. Feasibility of a flat-rate borrow-cost model — confirmed feasible

In the absence of point-in-time borrow data, a flat-rate model is the only
option, and it is straightforward to build: the machinery already exists
in `results/prereg_V4.md` §7.6 (previously a single 50 bps/year flat
assumption with a 0/200 bps bracket). Recalibrating to a named
primary/sensitivity/stress triple is a parameter change, not a new
construction:

| tier | annualized rate | status |
|---|---|---|
| **primary** | **10%** | the new primary — see §3 below for why this replaces the earlier 50 bps |
| sensitivity | **3%** | optimistic (near GC) bracket, no verdict authority |
| stress | **25%** | pessimistic (specials-adjacent) bracket, no verdict authority |

All three are computable from data already in hand: `contracts × 100 ×
S_t × rate / 365`, accrued daily on short-stock hedge notional only
(unchanged mechanism from the superseded draft, only the rate calibration
changes). No new pull is required.

**Dividends owed on short shares — feasible, and CRSP already has the
data.** A short-stock position must pay any cash dividend declared during
the holding period to the share lender — a real, distinct cash cost from
the borrow fee itself. `crsp_combined.parquet` already carries `DisExDt`
and `DisDivAmt` (confirmed present in the column dump above). This is
feasible to model precisely: for each hedge session where the position is
short, sum `DisDivAmt × shares_short` over any `DisExDt` falling inside
the holding window, charged as an additional cost on the call book's hedge
leg. This is **not** the same simplification as §7.5's `q = 0` (no
dividend adjustment in the Black-Scholes pricing/delta formula, a
disclosed pricing-convenience assumption) — `q = 0` affects what the
*model believes the option is worth*; dividends-owed-on-short affects a
*real cash flow the hedge actually pays*, and the two are independent.
Feasible with data on hand; not yet implemented since no V4 script exists.

**Financing on the stock hedge — feasible, currently asymmetric by
omission, should be closed.** Shorting stock (the call book's hedge)
generates cash proceeds; §7.6 already states "short rebate: none
credited (conservative)" — i.e. the model already assumes zero interest
income on that cash, a conservative choice. Going long stock (the put
book's hedge) requires financing the purchase; the superseded draft never
stated a symmetric charge for that side. This is an omission, not a
principled asymmetry, and it is feasible to close using the same `r =
0.01` constant already locked in §7.5: charge financing cost on long
stock-hedge notional at the same rate the short side is denied a rebate
at, so both hedge directions are treated with the same conservative
convention (cost, never credit).

## 3. What this model captures, and — stated as its own required
   disclosure, not a footnote — what it categorically cannot

**Captures:** the ongoing dollar cost of paying to borrow shares, at three
named rate levels bracketing plausible GC-to-specials territory for a
mid/small-cap universe; the incremental cash cost of dividends paid to a
lender during a short window; and symmetric financing treatment on both
hedge directions.

**Cannot capture — stated plainly, because a rate assumption invites
exactly this confusion: a borrow-cost model answers "what would it cost,"
never "could it be done at all."** A flat annualized rate says nothing
about whether shares are actually **available to borrow** on the specific
name, on the specific date, in the specific quantity the hedge needs. In
the absence of any availability/utilization/HTB-flag feed, this project
has no way to distinguish:

- a stock that is easily borrowed at 10 bps every day of the sample, from
- a stock that is impossible to borrow at all on some days (general
  collateral one week, hard-to-borrow or fully unavailable the next), from
- a stock where the required share quantity exceeds what the market can
  actually supply, forcing a partial hedge or a buy-in.

**The consequence for V4 specifically:** every call-book delta-hedge
simulation this project could build implicitly assumes the short leg is
always executable at the assumed rate. If a name in the traded universe
is, in reality, occasionally or persistently unborrowable, the simulated
call-book result would be reporting a hedge that could not have been
placed — not merely mispriced, but **impossible**, a different and more
severe failure mode than a cost-model error. This is a structural
limitation of every borrow-cost figure this audit can produce, at any of
the three rate tiers, and must be disclosed alongside any V4 call-book
result exactly as prominently as the rate assumption itself — not folded
into the same sentence as if a rate bracket already covers it.

## 4. Recommendation for `results/prereg_V4.md` §7.6

Replace the superseded 50 bps / 0 bps / 200 bps borrow treatment with the
10% primary / 3% sensitivity / 25% stress bracket above, add the
dividends-owed-on-short and symmetric hedge-financing components, and
carry the availability caveat in §4 forward as a standing, required
disclosure wherever a call-book result is reported. Applied directly to
`results/prereg_V4.md` §7.6 alongside this note.
