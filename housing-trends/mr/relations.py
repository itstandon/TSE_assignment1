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
from .compare import mismatches, not_below

RTOL = 1e-9
ALL = ["price", "ppsf", "n", "mom_price_pct"]


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
    return mismatches(src, fol, ALL, rtol=RTOL)


def _duplicate(src, fol, c, meta):
    expected = src.copy()
    expected["n"] = expected["n"] * 2                                     # only the count may change, and it must double
    return mismatches(expected, fol, ALL, rtol=RTOL)


def _scaled(src, fol, c, meta):
    k, expected = meta["k"], src.copy()
    expected["price"], expected["ppsf"] = expected["price"] * k, expected["ppsf"] * k
    return mismatches(expected, fol, ALL, rtol=RTOL)                      # counts and % changes must not move


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


def _shifted(src, fol, c, meta):
    overlap = list(src.index[1:])                                         # the 11 months both windows contain
    return (mismatches(src, fol, ["price", "ppsf", "n"], rtol=RTOL, months=overlap)
            + mismatches(src, fol, ["mom_price_pct"], rtol=RTOL, months=overlap[1:]))   # first overlap month has no prior in the follow-up


def _not_lower(src, fol, c, meta):
    return not_below(src, fol, "price", tol=1e-6, what="raising prices in one ZIP must not lower the median")


# ---------------------------------------------------------------- subdomains
def _always(c, info):
    return True


def _has_band(c, info):
    return c.price_min is not None or c.price_max is not None


def _no_band(c, info):
    return not _has_band(c, info)


def _many_zips(c, info):
    return len(c.zips) >= 2


def _room_to_shift(c, info):
    return c.end_month < info["last_month"]


MRS = [
    MR("MR1", "Reorder listings", "Invariance", T.reorder,
       "Shuffling the rows of the listing file leaves the whole contract unchanged.",
       "Every base input.", f"Exact: relative 1e-9 on all four contract columns.",
       "Order dependence: a first/last-row choice, an unstable sort or an unkeyed groupby in the Scanner or Statistician.",
       "Row order carries no meaning in a listing feed, so any change is a defect, not a modelling choice.",
       _always, _equal),
    MR("MR2", "Relabel listing ids", "Invariance", T.relabel_ids,
       "Giving the same listings different ids (a permutation of the id set) leaves the contract unchanged.",
       "Every base input.", "Exact: relative 1e-9 on all four contract columns.",
       "Logic that depends on listing_id values: id-based tie-breaking, caps or sorts (the Critic's suspect cap and the "
       "Diagnoser's exclusions are ordered by id), or ids used as positions.",
       "A listing's id is a label; behaviour must not depend on which label it carries.",
       _always, _equal),
    MR("MR3", "Duplicate every listing", "Invariance", T.duplicate_all,
       "With every listing present twice, medians and $/sqft and the month-over-month % are unchanged and the "
       "monthly listing count exactly doubles.",
       "Every base input (violations are expected where thin buckets exist: more listings change which buckets are 'thin').",
       "Exact: relative 1e-9 on price, ppsf and mom_price_pct; n must be exactly 2x.",
       "Estimators that are not scale-free in sample size (floor-based trimming, n-dependent weights), or coverage logic "
       "that reacts to the count.",
       "Duplicating a sample leaves its empirical distribution unchanged, so any distribution statistic is unchanged. "
       "The system's coverage threshold depends on n by design, so this MR may be ill-conceived on thin buckets.",
       _always, _duplicate),
    MR("MR4", "Unit change: USD to USD thousands", "Invariance", T.to_thousands,
       "Expressing prices (and the price filter) in thousands scales price and $/sqft by 1/1000 and leaves counts and "
       "month-over-month % unchanged.",
       "Every base input (price filters are whole thousands by construction).",
       "Relative 1e-9 on all four columns after applying the factor.",
       "Hard-coded dollar magnitudes (rounding, absolute thresholds, formatting) in the agents.",
       "A display unit must not change rankings or percentages.",
       _always, _scaled),
    MR("MR5", "Widen the price band", "Monotonic", T.widen_price_band,
       "Lowering price_min by 25% and/or raising price_max by 25% (a superset of listings) never lowers the "
       "listing count of any month.",
       "Base inputs that have a price_min or price_max.", "Exact: follow-up n >= source n for every month.",
       "Filters applied wrongly (inclusive/exclusive bounds), or exclusions that remove more listings after widening.",
       "A superset of listings cannot have fewer members; only an agent that drops listings differently can break it.",
       _has_band, _more_listings),
    MR("MR6", "Restrict to a sub-region", "Monotonic", T.restrict_zips,
       "Keeping a strict subset of the ZIPs never gives any month more listings than the full set.",
       "Base inputs with at least two ZIPs.", "Exact: follow-up n <= source n for every month.",
       "ZIP scoping errors, or pooled listings leaking into the 'own' count.",
       "Own listings of fewer ZIPs are a subset of own listings of more ZIPs.",
       _many_zips, _fewer_listings),
    MR("MR7", "Add $X to one month", "Monotonic", T.add_x_to_one_month,
       f"Adding ${T.ADD_X:,} to every price in one month raises that month's median by ~${T.ADD_X:,} (within 1% of X) "
       "and no month's median decreases.",
       "Base inputs without a price filter (a price band would move listings in or out).",
       "Shift within 1% of X; no decrease beyond 1e-6.",
       "Statistics that react to the injected shift (outlier handling excluding or trimming the shifted month), or "
       "month bucketing errors.",
       "The median is translation-equivariant, so a uniform shift of a month's prices shifts its median by the same amount.",
       _no_band, _add_x),
    MR("MR8", "Shift the window forward one month", "Directional", T.shift_window,
       "Moving the 12-month window forward by one month leaves the 11 overlapping months unchanged (price, $/sqft, n; "
       "month-over-month % for months with a prior in both runs).",
       "Base inputs whose window can move forward inside the snapshot.",
       "Relative 1e-9 on the overlapping months.",
       "Results that depend on the window position: loop decisions (robust recompute, exclusions) driven by months "
       "outside the overlap, or edge effects in pooling.",
       "A month's statistics describe that month's listings, not which other months were requested.",
       _room_to_shift, _shifted),
    MR("MR9", "Scale all prices by k > 1", "Directional", T.scale_prices,
       "Multiplying every price by k (k in {1.5, 2, 3}) scales median price and $/sqft by k and leaves the "
       "month-over-month % and the counts unchanged.",
       "Base inputs without a price filter (a price band is not scaled).",
       "Relative 1e-9 on all four columns after applying k.",
       "Absolute (non-relative) thresholds or rounding in the Critic, Diagnoser evidence or Statistician.",
       "All of the system's tests are meant to be relative (percentage jumps, log-price z-scores), so a currency "
       "rescaling must not change a decision.",
       _no_band, _scaled),
    MR("MR10", "Raise one ZIP's prices", "Directional", T.raise_one_zip,
       f"Multiplying every price in one ZIP by {T.ZIP_RAISE} never lowers any month's overall median price.",
       "Base inputs without a price filter.", "No decrease beyond 1e-6 in any month.",
       "Loops that react to the injected change (a new 'jump' triggering exclusions that remove listings elsewhere), "
       "or ZIP scoping errors.",
       "Raising some prices can only move a median up or leave it unchanged.",
       _no_band, _not_lower),
]
MR_BY_ID = {m.id: m for m in MRS}
