"""The metamorphic relations. Each MR states: the transformation, the relation that must hold between the source and
follow-up contracts (see contract.py), the input subdomain where it should hold, the tolerance, and the fault a
violation would expose. An MR is a hypothesis: some are expected to fail on parts of the pipeline, and the harness
exists to find out where and why.

applies(c, info) decides whether a base input lies in the subdomain; info = {"last_month": last month of the snapshot}.
check(src, fol, c, meta) -> list of problems (empty = the relation holds); src/fol are contract tables indexed by month.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import transforms as T
import numpy as np

from .compare import mismatches, not_below
from .contract import zip_table

RTOL = 1e-9
ALL = ["price", "ppsf", "n", "mom_price_pct"]
ZIPS = ["price", "ppsf", "n_own", "n_used", "mom_price_pct"]
OUTLIER_TOL = 0.15             # MR10: a single planted outlier may move a median by less than this (relative)


@dataclass(frozen=True)
class MR:
    id: str
    name: str
    family: str                     # Invariance | Monotonic | Directional
    transform: Callable
    relation: str
    subdomain: str
    tolerance: str
    fault: str
    justification: str
    applies: Callable
    check: Callable


# ---------------------------------------------------------------- checks
def _equal(src, fol, c, meta):
    """Everything identical: the overall table and every per ZIP-month row."""
    return mismatches(src, fol, ALL, rtol=RTOL) + mismatches(zip_table(src), zip_table(fol), ZIPS, rtol=RTOL)


def _dup_zip_scaled(src, fol, c, meta):
    """Each requested ZIP appeared N_zip times: its own count is N_zip x, its medians and $/sqft do not move."""
    sb, fb = zip_table(src), zip_table(fol)
    exp = sb.copy()
    exp["n_own"] = exp.n_own * exp.zip_code.map(meta["n_of"])
    return mismatches(exp, fb, ["price", "ppsf", "n_own", "mom_price_pct"], rtol=RTOL)


def _more_listings(src, fol, c, meta):
    return not_below(src, fol, "n", what="widening the band must not remove listings")


def _fewer_listings(src, fol, c, meta):
    return not_below(fol, src, "n", what="a subset of ZIPs must not have more listings")


def _add_x(src, fol, c, meta):
    out = not_below(src, fol, "price", tol=1e-6, what="adding a positive amount must not lower the median")
    m, x = meta["month"], meta["x"]
    a, b = src["price"].get(m), fol["price"].get(m)
    if a == a and b == b and abs((b - a) - x) > 0.01 * x:                 # "by ~X": within 1% of X
        out.append(f"price {m}: median should rise by ~{x:,.0f}, rose by {b - a:,.0f}")
    return out


def _remove_top(src, fol, c, meta):
    """No ZIP-month median increases (a month emptied by the removal has no median to compare)."""
    sb, fb = zip_table(src), zip_table(fol)
    both = sb.price.notna() & fb.price.reindex(sb.index).notna()
    bad = sb.index[both & (fb.price.reindex(sb.index) > sb.price + 1e-6)]
    return [f"price {k}: removing the dearest listing raised the median ({sb.price[k]:,.0f} -> {fb.price[k]:,.0f})"
            for k in bad][:5]


def _zip_doubled(src, fol, c, meta):
    """The doubled ZIP's price and $/sqft double; its counts and % changes, and every other ZIP's row, do not move."""
    sb, fb = zip_table(src), zip_table(fol)
    exp = sb.copy()
    hit = exp.zip_code == meta["zip"]
    exp.loc[hit, ["price", "ppsf"]] = exp.loc[hit, ["price", "ppsf"]] * meta["factor"]
    return mismatches(exp, fb, ["price", "ppsf", "n_own", "mom_price_pct"], rtol=RTOL)


def _outlier_contained(src, fol, c, meta):
    out = []
    z, lid = meta["zip"], meta["id"]
    with np.errstate(invalid="ignore"):
        rel = ((fol["price"] - src["price"]) / src["price"]).abs()
        for m, r in rel[rel > OUTLIER_TOL].items():
            out.append(f"overall price {m}: moved {100 * r:.1f}% (limit {100 * OUTLIER_TOL:.0f}%)")
        sb, fb = zip_table(src), zip_table(fol)
        rz = ((fb.price - sb.price.reindex(fb.index)) / sb.price.reindex(fb.index)).abs()
        for k, r in rz[(fb.zip_code == z) & (rz > OUTLIER_TOL)].items():
            out.append(f"ZIP-month {k}: moved {100 * r:.1f}% (limit {100 * OUTLIER_TOL:.0f}%)")
    judged = [i for i, vz, _ in fol.attrs.get("verdicts", []) if vz == z]
    if judged and lid not in judged:                                        # the Diagnoser looked at this ZIP, yet missed it
        out.append(f"Diagnoser examined {len(judged)} suspect(s) in ZIP {z} but not the planted listing {lid}")
    return out[:5]


def _month_copied(src, fol, c, meta):
    """The replaced month equals its predecessor: same median, same $/sqft, same count, month-over-month change 0%."""
    m, p = meta["month"], meta["prev"]
    cur, prv = fol.loc[[m]], fol.loc[[p]]
    out = [f"{col} {m}: {_f(cur[col].iloc[0])} but {p} has {_f(prv[col].iloc[0])}" for col in ("price", "ppsf", "n")
           if not np.isclose(cur[col].iloc[0], prv[col].iloc[0], rtol=RTOL, atol=1e-9, equal_nan=True)]
    mom = fol.mom_price_pct.get(m)
    if mom == mom and abs(mom) > 1e-6:
        out.append(f"mom_price_pct {m}: expected 0%, got {mom:.4f}%")
    return out


def _f(v):
    return "NaN" if v != v else f"{v:.6g}"


def _scaled(src, fol, c, meta):
    """Prices and $/sqft scale by k, in the overall table and in every ZIP-month row; counts and % changes do not move."""
    k, e = meta["k"], src.copy()
    e["price"], e["ppsf"] = e["price"] * k, e["ppsf"] * k
    eb = zip_table(src).copy()
    eb["price"], eb["ppsf"] = eb["price"] * k, eb["ppsf"] * k
    return mismatches(e, fol, ALL, rtol=RTOL) + mismatches(eb, zip_table(fol), ZIPS, rtol=RTOL)


# ---------------------------------------------------------------- subdomains
def _always(c, info):
    return True


def _has_band(c, info):
    return c.price_min is not None or c.price_max is not None


def _no_band(c, info):
    return not _has_band(c, info)


def _many_zips(c, info):
    return len(c.zips) >= 2


MRS = [
    # ---- Invariance
    MR("MR1", "Reorder listings", "Invariance", T.reorder,
       "Shuffling the rows of the listing file leaves the whole contract (overall and per ZIP-month) unchanged.",
       "Every base input.", "Exact: relative 1e-9 on every contract column.",
       "Order dependence: a first/last-row choice, an unstable sort or an unkeyed groupby in the Scanner or Statistician.",
       "Row order carries no meaning in a listing feed, so any change is a defect, not a modelling choice. Baseline sanity: "
       "if this fails, no other result is interpretable.",
       _always, _equal),
    MR("MR2", "Sort listings by price (adversarial order)", "Invariance", T.sort_by_price,
       "Sorting the rows ascending by price inside every ZIP-month leaves the whole contract unchanged.",
       "Every base input.", "Exact: relative 1e-9 on every contract column.",
       "Position-dependent logic that a random shuffle rarely triggers: head/tail slicing, 'first N' caps or id/position "
       "tie-breaks that matter only when the cheapest or dearest listings come first.",
       "A sorted feed is just another valid order; a random shuffle and an adversarial order exercise different failure modes.",
       _always, _equal),
    MR("MR3", "Add a 10x-priced fake month outside the window", "Invariance", T.add_fake_month,
       "Adding listings in a month far outside the query window (and beyond Pooling's padding) leaves the whole contract "
       "unchanged.",
       "Every base input.", "Exact: relative 1e-9 on every contract column.",
       "Out-of-window data leaking into the answer: the Critic's ZipProfile (log-price median/MAD behind the outlier "
       "z-scores) is computed from the WHOLE snapshot, so out-of-window rows can change which listings count as suspects.",
       "The query asks about 12 months; listings of other months are not part of the question. If the answer depends on them, "
       "the system's scope is wider than its contract.",
       _always, _equal),
    MR("MR4", "Multiply each ZIP's listings by a random N (1-10)", "Invariance", T.duplicate_random_n,
       "With every ZIP's listings present N_zip times (fresh ids, N_zip random per ZIP), each requested ZIP's median price, "
       "$/sqft and month-over-month % are unchanged and its monthly own count is exactly N_zip times larger.",
       "Every base input (violations are expected where buckets cross the coverage threshold: n changes which buckets are thin).",
       "Exact: relative 1e-9 on price, ppsf, mom_price_pct per ZIP-month; n_own must be exactly N_zip x.",
       "Estimators that are not scale-free in sample size, coverage logic that reacts to the count, or id-keyed exclusions "
       "that do not follow a listing's copies.",
       "Replicating a ZIP's sample leaves its empirical distribution unchanged, so any distribution statistic is unchanged. "
       "The coverage threshold depends on n by design, so this MR is partly ill-conceived on thin buckets.",
       _always, _dup_zip_scaled),
    # ---- Monotonic
    MR("MR5", "Add $X to every price in one month", "Monotonic", T.add_x_to_one_month,
       f"Adding ${T.ADD_X:,} to every price in one month raises that month's median by ~${T.ADD_X:,} (within 1% of X) "
       "and no month's median decreases.",
       "Base inputs without a price filter (a price band would move listings in or out).",
       "Shift within 1% of X; no decrease beyond 1e-6.",
       "Statistics that react to the injected shift (a jump triggers the robust recompute, which replaces the median by a "
       "trimmed mean), or month bucketing errors.",
       "The median is translation-equivariant, so a uniform shift of a month's prices shifts its median by the same amount. "
       "The system's own jump logic is expected to interfere.",
       _no_band, _add_x),
    MR("MR6", "Remove the dearest listing of each ZIP", "Monotonic", T.remove_top_per_zip,
       "Deleting the single most expensive in-scope listing of every requested ZIP raises no ZIP-month median.",
       "Every base input.", "No increase beyond 1e-6 in any ZIP-month with a median on both sides.",
       "Buckets pushed across the coverage threshold (the removal makes a bucket thin, Pooling then borrows neighbours "
       "that raise its median), or any estimator that is not monotone in its sample.",
       "Removing the maximum of a sample can never raise its median, so only the pipeline's reaction (pooling, robust "
       "recompute, exclusions) can break it.",
       _always, _remove_top),
    MR("MR7", "Widen the price band", "Monotonic", T.widen_price_band,
       "Lowering price_min by 25% and/or raising price_max by 25% (a superset of listings) never lowers the listing "
       "count of any month.",
       "Base inputs that have a price_min or price_max.", "Exact: follow-up n >= source n for every month.",
       "Filters applied wrongly (inclusive/exclusive bounds), or exclusions that remove more listings after widening.",
       "A superset of listings cannot have fewer members; only an agent that drops listings differently can break it.",
       _has_band, _more_listings),
    MR("MR8", "Restrict to a sub-region", "Monotonic", T.restrict_zips,
       "Keeping a strict subset of the ZIPs never gives any month more listings than the full set.",
       "Base inputs with at least two ZIPs.", "Exact: follow-up n <= source n for every month.",
       "ZIP scoping errors, or pooled listings leaking into the 'own' count.",
       "Own listings of fewer ZIPs are a subset of own listings of more ZIPs. Tests ZIP isolation from the count side "
       "(MR9 tests it from the price side).",
       _many_zips, _fewer_listings),
    # ---- Directional
    MR("MR9", "Double all prices in one ZIP", "Directional", T.double_one_zip,
       "Multiplying every price in ONE ZIP by 2 doubles that ZIP's median price and $/sqft in every month and leaves its "
       "counts and % changes, and every other ZIP's rows, unchanged.",
       "Base inputs without a price filter (a price band is not scaled).",
       "Exact: relative 1e-9 on every ZIP-month row.",
       "Cross-ZIP leakage: Pooling borrows listings from neighbouring ZIPs, so a changed ZIP changes the medians of its "
       "thin neighbours (and the doubled ZIP's own pooled buckets stop scaling cleanly).",
       "A ZIP's statistics describe that ZIP's listings. MR9 tests a local scale change; MR12 a global one: if MR12 holds and "
       "MR9 fails, the cause is cross-ZIP pooling, not a scale-sensitive threshold.",
       _no_band, _zip_doubled),
    MR("MR10", "Inject one 10x outlier", "Directional", T.inject_outlier,
       f"Multiplying the price of one counted listing by {T.OUTLIER_FACTOR} moves no month's overall median, and no month of "
       f"its ZIP, by {100 * OUTLIER_TOL:.0f}% or more; and whenever the Diagnoser examines that ZIP, the planted listing is "
       "among its suspects.",
       "Base inputs without a price filter (the 10x price would leave a band).",
       f"Relative change < {100 * OUTLIER_TOL:.0f}% (overall and ZIP-month); suspect list membership.",
       "Fragile estimator or loop reaction: one listing flips a ZIP to the robust trimmed mean (or triggers exclusions), "
       "moving the whole series; or the outlier screen does not surface a 10x listing the Diagnoser should judge.",
       "A single extreme value is exactly what the median, the Critic's screen and the Diagnoser exist to absorb.",
       _no_band, _outlier_contained),
    MR("MR11", "Replace a month with a copy of the previous month", "Directional", T.replace_month_with_previous,
       "If one month's listings are replaced by a copy of the previous month's, the two months have equal median price, "
       "median $/sqft and count, and the month-over-month change into the replaced month is 0%.",
       "Every base input.", "Exact: relative 1e-9; mom_price_pct within 1e-6 of 0.",
       "Identical months treated differently: thin-bucket pooling or robust recompute that depends on neighbouring months, "
       "id-keyed exclusions that hit one copy but not the other.",
       "Two months with the same listings have the same distribution, so any per-month statistic must agree.",
       _always, _month_copied),
    MR("MR12", "Scale all prices by k > 1", "Directional", T.scale_prices,
       "Multiplying every price by k (k in {1.5, 2, 3}) scales median price and $/sqft by k (overall and per ZIP-month) and "
       "leaves the month-over-month % and the counts unchanged.",
       "Base inputs without a price filter (a price band is not scaled).",
       "Relative 1e-9 on every column after applying k.",
       "Absolute (non-relative) thresholds or rounding in the Critic, Diagnoser evidence or Statistician.",
       "All of the system's tests are meant to be relative (percentage jumps, log-price z-scores), so a currency "
       "rescaling must not change a decision.",
       _no_band, _scaled),
]
MR_BY_ID = {m.id: m for m in MRS}
