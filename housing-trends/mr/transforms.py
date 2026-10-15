"""Input transformations. Each takes (df, criteria, rng) and returns a Follow: the transformed listings, the (possibly
changed) criteria, and `meta` that the relation's check needs (the scale factor, the month that was shifted, ...).
Nothing here touches the frozen snapshot on disk: `df` is copied before any change."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from housing_trends import Criteria
from housing_trends.agents.scanner import active_filters, filter_mask
from housing_trends.context import add_months, month_range

ADD_X = 50_000                 # dollars added to every price of one month (MR7)
SCALE_FACTORS = (1.5, 2.0, 3.0)    # k > 1 for the scaling MR
ZIP_RAISE = 1.5                # factor applied to one ZIP's prices (MR10)
UNIT = 1000                    # USD -> USD thousands (retired MR, kept for the unit tests)
FAKE_FACTOR = 10               # price multiplier of the fake month (MR3)
FAKE_MONTH_OFFSET = 6          # months after the snapshot's last month: beyond any Pooling padding (MAX_TIME_PAD = 3)
MAX_COPIES = 10                # N in 1..MAX_COPIES per ZIP (MR4)
ZIP_DOUBLE = 2.0               # factor applied to one ZIP's prices (MR9)
OUTLIER_FACTOR = 10            # one planted listing is multiplied by this (MR10)


@dataclass
class Follow:
    df: pd.DataFrame
    criteria: Criteria
    meta: dict = field(default_factory=dict)


def reorder(df, c, rng):
    return Follow(df.sample(frac=1, random_state=int(rng.integers(2**31 - 1))).reset_index(drop=True), c)


def relabel_ids(df, c, rng):
    d = df.copy()
    d["listing_id"] = rng.permutation(d.listing_id.to_numpy())      # same id set, different owner of each id
    return Follow(d, c)


def duplicate_all(df, c, rng):
    dup = df.copy()
    dup["listing_id"] = dup.listing_id + int(df.listing_id.max())   # fresh unique ids
    return Follow(pd.concat([df, dup], ignore_index=True), c)


def to_thousands(df, c, rng):
    d = df.copy()
    d["price"] = d.price / UNIT
    scaled = {k: (None if getattr(c, k) is None else int(round(getattr(c, k) / UNIT))) for k in ("price_min", "price_max")}
    return Follow(d, c.model_copy(update=scaled), {"k": 1 / UNIT})


def widen_price_band(df, c, rng):
    upd = {}
    if c.price_min is not None:
        upd["price_min"] = int(round(c.price_min * 0.75))
    if c.price_max is not None:
        upd["price_max"] = int(round(c.price_max * 1.25))
    return Follow(df, c.model_copy(update=upd))


def restrict_zips(df, c, rng):
    keep = int(rng.integers(1, len(c.zips)))                          # 1 .. len-1 ZIPs: a strict subset
    zips = tuple(sorted(rng.choice(list(c.zips), size=keep, replace=False).tolist()))
    return Follow(df, c.model_copy(update={"zips": zips}), {"zips": zips})


def add_x_to_one_month(df, c, rng):
    month = str(rng.choice(month_range(c.start_month, c.end_month)))
    d = df.copy()
    d.loc[d.month == month, "price"] = d.loc[d.month == month, "price"] + ADD_X
    return Follow(d, c, {"month": month, "x": ADD_X})


def shift_window(df, c, rng):
    return Follow(df, c.model_copy(update={"start_month": add_months(c.start_month, 1), "end_month": add_months(c.end_month, 1)}))


def scale_prices(df, c, rng):
    k = float(rng.choice(SCALE_FACTORS))
    d = df.copy()
    d["price"] = d.price * k
    return Follow(d, c, {"k": k})


def raise_one_zip(df, c, rng):
    z = str(rng.choice(list(c.zips)))
    d = df.copy()
    d.loc[d.zip_code == z, "price"] = d.loc[d.zip_code == z, "price"] * ZIP_RAISE
    return Follow(d, c, {"zip": z, "factor": ZIP_RAISE})


# ---------------------------------------------------------------- helpers for the transforms below
def _fresh_ids(df, n, start=None):
    top = int(df.listing_id.max()) if start is None else start
    return np.arange(top + 1, top + 1 + n)


def _in_scope(df, c):
    """Listings the Scanner would count for the criteria: its ZIPs, its window, its filters (no pooling)."""
    in_scope = df.zip_code.isin(c.zips) & (df.month >= c.start_month) & (df.month <= c.end_month)
    return in_scope & filter_mask(df, c, active_filters(c))


def _set_month(d, month):
    d["month"] = month
    if "sold_date" in d.columns:
        d["sold_date"] = month + "-15"
    return d


def sort_by_price(df, c, rng):
    """Adversarial order: ascending price inside every ZIP-month (so 'first row' / 'last row' = cheapest / dearest)."""
    return Follow(df.sort_values(["zip_code", "month", "price"], kind="stable").reset_index(drop=True), c)


def add_fake_month(df, c, rng):
    """A month far outside the query window (and beyond Pooling's padding) with the criteria ZIPs' listings at 10x price."""
    fake = add_months(df.month.max(), FAKE_MONTH_OFFSET)
    src = df[df.zip_code.isin(c.zips) & (df.month == c.start_month)]
    new = _set_month(src.copy(), fake)
    new["price"] = new.price * FAKE_FACTOR
    new["listing_id"] = _fresh_ids(df, len(new))
    return Follow(pd.concat([df, new], ignore_index=True), c, {"fake_month": fake, "rows": len(new)})


def duplicate_random_n(df, c, rng):
    """Every ZIP's listings appear N_zip times (N_zip drawn per ZIP in 1..10), copies carrying fresh ids."""
    zips = sorted(df.zip_code.unique())
    n_of = {z: int(rng.integers(1, MAX_COPIES + 1)) for z in zips}
    top, parts = int(df.listing_id.max()), [df]
    for j in range(1, MAX_COPIES):
        part = df[df.zip_code.map(n_of) > j].copy()
        part["listing_id"] = part.listing_id + j * top              # id in (j*top, (j+1)*top]: unique across copies
        parts.append(part)
    return Follow(pd.concat(parts, ignore_index=True), c, {"n_of": n_of})


def remove_top_per_zip(df, c, rng):
    """Delete the single most expensive in-scope listing of every requested ZIP."""
    d = df[_in_scope(df, c)]
    gone = d.loc[d.groupby("zip_code").price.idxmax()]
    return Follow(df.drop(index=gone.index).reset_index(drop=True), c, {"removed": gone.listing_id.astype(int).tolist()})


def double_one_zip(df, c, rng):
    z = str(rng.choice(list(c.zips)))
    d = df.copy()
    d.loc[d.zip_code == z, "price"] = d.loc[d.zip_code == z, "price"] * ZIP_DOUBLE
    return Follow(d, c, {"zip": z, "factor": ZIP_DOUBLE})


def inject_outlier(df, c, rng):
    """Multiply the price of ONE listing the criteria counts by 10."""
    pick = df.index[_in_scope(df, c)]
    i = int(rng.choice(pick.to_numpy()))
    d = df.copy()
    d.loc[i, "price"] = d.loc[i, "price"] * OUTLIER_FACTOR
    return Follow(d, c, {"id": int(d.loc[i, "listing_id"]), "zip": str(d.loc[i, "zip_code"]), "month": str(d.loc[i, "month"])})


def replace_month_with_previous(df, c, rng):
    """One month of the window (not the first) is thrown away and replaced by a copy of the month before it."""
    window = month_range(c.start_month, c.end_month)
    m = str(rng.choice(window[1:]))
    prev = add_months(m, -1)
    new = _set_month(df[df.month == prev].copy(), m)
    new["listing_id"] = _fresh_ids(df, len(new))
    return Follow(pd.concat([df[df.month != m], new], ignore_index=True), c, {"month": m, "prev": prev})
