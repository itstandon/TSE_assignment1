"""Scanner Agent (plain code, no LLM): retrieve matching listings per ZIP-month bucket and report,
for each bucket, whether and how widening the scope could surface more listings.

WIDEN ONLY WHAT IS THIN: a bucket whose own listings (all user filters, no pooling) already reach
`min_listings` is never pooled or relaxed, whatever the scope says. Only thin buckets receive the scope.
Otherwise a rescan meant to fix thin buckets would silently change healthy ones.

Per-bucket report (one row per requested ZIP x month in the criteria window):
  widened              True if this bucket was thin at base and therefore received the scope
  zip_extra_used, time_pad_used   widening depth actually applied to this bucket (0 if not widened)
  n_base               own listings under ALL user filters, no pooling (decides `widened`; never changes)
  n_own, n, n_pooled   listings from the bucket itself / in the current scope / pooled in from neighbours
  gain_time            extra listings if the date window were widened to the MAX allowed padding
  gain_zip             extra listings if the ZIP radius were widened to the MAX allowed neighbours
  gain_filters         extra listings if every still-relaxable (set, non-protected) filter were dropped
  potential            extra listings if ALL moves were applied together (upper bound)
  moves                which of {time, zip, filters} would add >= 1 listing  (Pooling picks from these)
  exhaustive           potential == 0: no allowed widening can surface another listing  -> Pooling must stop
  reachable            n + potential >= min_listings: the Critic's threshold is attainable at all
  thin                 n < min_listings
"Gain" is exact (it equals what the corresponding rescan would add); tests check this by brute force.

Assumptions (document in the report):
  * "Adjacent ZIPs" = the numerically nearest ZIP codes in the snapshot (no coordinates available).
  * A listing with a missing value for an ACTIVE filter's field (e.g. bed) does not match that filter.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..context import month_index, month_range
from ..criteria import Criteria
from ..geo import load_centroids, neighbour_order
from ..state import MAX_TIME_PAD, MAX_ZIP_EXTRA, MIN_LISTINGS, Scope

OPTIONAL_COLS = ("bed", "bath", "house_size", "acre_lot", "property_type")


# ---------------------------------------------------------------- snapshot
class Snapshot:
    def __init__(self, df: pd.DataFrame, months=None, zips=None, centroids=None):
        """`months` / `zips` declare the coverage explicitly; by default it is inferred from the rows.
        `centroids` {zip: (lat, lon)} orders the pooling neighbours geographically; without it (the default, and what
        the unit tests use) the numerically nearest ZIP codes are used. See housing_trends.geo."""
        need = {"listing_id", "zip_code", "month", "price"} - set(df.columns)
        if need:
            raise ValueError(f"snapshot is missing columns: {sorted(need)}")
        df = df.copy()
        df["zip_code"] = df.zip_code.astype(str).str.extract(r"(\d{5})")[0]
        df["month"] = df.month.astype(str).str[:7]
        for c in OPTIONAL_COLS:
            if c not in df.columns:
                df[c] = np.nan
        self.df = df
        self.zips = sorted(set(zips) if zips else df.zip_code.dropna().unique(), key=int)
        self.months = list(months) if months else month_range(df.month.min(), df.month.max())
        self.centroids = dict(centroids or {})
        self._order = neighbour_order(self.zips, self.centroids)

    @classmethod
    def from_csv(cls, path, centroids=None) -> "Snapshot":
        return cls(pd.read_csv(path, dtype={"zip_code": str}), centroids=centroids)

    @classmethod
    def frozen(cls, df, **kw) -> "Snapshot":
        """The snapshot as the application uses it: pooling neighbours ordered by the frozen ZIP centroids."""
        return cls(df, centroids=load_centroids(), **kw)

    def neighbours(self, z: str, k: int) -> list[str]:
        return self._order[z][:k]


# ---------------------------------------------------------------- filters
def active_filters(c: Criteria) -> list[str]:
    on = {"price": c.price_min is not None or c.price_max is not None,
          "beds": c.beds_min is not None or c.beds_max is not None,
          "baths": c.baths_min is not None,
          "sqft": c.sqft_min is not None or c.sqft_max is not None,
          "property_type": bool(c.property_types)}
    return [f for f in ("price", "beds", "baths", "sqft", "property_type") if on[f]]


def relaxable_filters(c: Criteria, scope: Scope) -> list[str]:
    """Set, NON-protected filters that Pooling has not relaxed yet."""
    return [f for f in active_filters(c) if f not in c.protected_filters and f not in scope.relaxed]


def filter_mask(df: pd.DataFrame, c: Criteria, filters: list[str]) -> pd.Series:
    m = pd.Series(True, index=df.index)
    if "price" in filters:
        if c.price_min is not None: m &= df.price >= c.price_min
        if c.price_max is not None: m &= df.price <= c.price_max
    if "beds" in filters:
        if c.beds_min is not None: m &= df.bed >= c.beds_min
        if c.beds_max is not None: m &= df.bed <= c.beds_max
    if "baths" in filters and c.baths_min is not None:
        m &= df.bath >= c.baths_min
    if "sqft" in filters:
        if c.sqft_min is not None: m &= df.house_size >= c.sqft_min
        if c.sqft_max is not None: m &= df.house_size <= c.sqft_max
    if "property_type" in filters and c.property_types:
        m &= df.property_type.astype(str).str.lower().isin(c.property_types)
    return m                                           # NaN comparisons are False -> not a match


# ---------------------------------------------------------------- scan
@dataclass
class ScanResult:
    criteria_key: str
    scope: Scope
    buckets: pd.DataFrame        # one row per (zip_code, month); see module docstring
    members: pd.DataFrame        # (zip_code, month, listing_id, own): listings behind each bucket

    def thin(self) -> pd.DataFrame:
        return self.buckets[self.buckets.thin]

    def summary(self) -> str:
        b = self.buckets
        return (f"{len(b)} buckets, {int(b.widened.sum())} widened, {int(b.thin.sum())} thin (<min), {int(b.exhaustive.sum())} exhaustive, "
                f"{int((b.thin & ~b.reachable).sum())} thin&unreachable | depth used: time_pad max {int(b.time_pad_used.max())}, "
                f"zip_extra max {int(b.zip_extra_used.max())}, relaxed={list(self.scope.relaxed)}")


def _count_matrix(df, zips, months):
    g = df.groupby(["zip_code", "month"]).size().unstack(fill_value=0)
    return g.reindex(index=zips, columns=months, fill_value=0).to_numpy()


def scan(c: Criteria, scope: Scope, snap: Snapshot, min_listings: int = MIN_LISTINGS) -> ScanResult:
    illegal = sorted(set(scope.relaxed) & set(c.protected_filters))
    if illegal:
        raise ValueError(f"scope relaxes protected filter(s) {illegal}; Pooling may only relax non-protected ones")
    missing = [z for z in c.zips if z not in snap.zips]
    if missing:
        raise ValueError(f"criteria ZIPs not in snapshot: {missing}")
    window = month_range(c.start_month, c.end_month)
    if month_index(window[0]) < month_index(snap.months[0]) or month_index(window[-1]) > month_index(snap.months[-1]):
        raise ValueError(f"window {window[0]}..{window[-1]} outside snapshot {snap.months[0]}..{snap.months[-1]}")

    act = active_filters(c)
    cur = [f for f in act if f not in scope.relaxed]                          # filters in force for widened buckets
    all_relaxed = [f for f in cur if f in c.protected_filters]                # what remains if everything relaxable is dropped
    df = snap.df
    m_base, m_cur = filter_mask(df, c, act), filter_mask(df, c, cur)
    A_base = _count_matrix(df[m_base], snap.zips, snap.months)
    A_cur = _count_matrix(df[m_cur], snap.zips, snap.months)
    A_rel = _count_matrix(df[filter_mask(df, c, all_relaxed)], snap.zips, snap.months)
    zidx, midx = {z: i for i, z in enumerate(snap.zips)}, {m: i for i, m in enumerate(snap.months)}
    nm = len(snap.months)

    def pooled(A, z, mi, extra, pad):
        rows = [zidx[z]] + [zidx[o] for o in snap.neighbours(z, extra)]
        return int(A[np.ix_(rows, range(max(0, mi - pad), min(nm - 1, mi + pad) + 1))].sum())

    rows, plan = [], []
    for z in c.zips:
        for m in window:
            mi = midx[m]
            n_base = int(A_base[zidx[z], mi])
            widen = n_base < min_listings                                    # healthy buckets keep the default scope
            ze, tp = scope.depth(z, m) if widen else (0, 0)
            A = A_cur if widen else A_base
            n = pooled(A, z, mi, ze, tp)
            g_t = pooled(A, z, mi, ze, MAX_TIME_PAD) - n
            g_z = pooled(A, z, mi, MAX_ZIP_EXTRA, tp) - n
            g_f = pooled(A_rel, z, mi, ze, tp) - n
            pot = pooled(A_rel, z, mi, MAX_ZIP_EXTRA, MAX_TIME_PAD) - n
            n_own = int(A[zidx[z], mi])
            plan.append((z, m, widen, ze, tp))
            rows.append(dict(zip_code=z, month=m, widened=widen, zip_extra_used=ze, time_pad_used=tp,
                             n_base=n_base, n_own=n_own, n=n, n_pooled=n - n_own,
                             gain_time=g_t, gain_zip=g_z, gain_filters=g_f, potential=pot,
                             moves=",".join(k for k, g in (("time", g_t), ("zip", g_z), ("filters", g_f)) if g > 0),
                             exhaustive=pot == 0, reachable=n + pot >= min_listings, thin=n < min_listings))
    buckets = pd.DataFrame(rows)

    ids_cur = {k: v.to_numpy() for k, v in df[m_cur].groupby(["zip_code", "month"]).listing_id}
    ids_base = {k: v.to_numpy() for k, v in df[m_base].groupby(["zip_code", "month"]).listing_id}
    mem = []
    for z, m, widen, ze, tp in plan:
        ids, mi = (ids_cur if widen else ids_base), midx[m]
        for zz in [z] + snap.neighbours(z, ze):
            for mm in snap.months[max(0, mi - tp): min(nm - 1, mi + tp) + 1]:
                for lid in ids.get((zz, mm), ()):
                    mem.append((z, m, int(lid), zz == z and mm == m))
    members = (pd.DataFrame(mem, columns=["zip_code", "month", "listing_id", "own"])
               .sort_values(["zip_code", "month", "listing_id"]).reset_index(drop=True))
    return ScanResult(c.key(), scope, buckets, members)


def make_scanner(snap: Snapshot, min_listings: int = MIN_LISTINGS):
    """LangGraph node factory: reads state['criteria'] and state['scope'], writes state['scan']."""
    def scanner(state):
        res = scan(state["criteria"], state.get("scope") or Scope(), snap, min_listings)
        return {"scan": res, "trace": [f"scanner: {res.summary()}"]}
    return scanner