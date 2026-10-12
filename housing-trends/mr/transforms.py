"""Input transformations. Each takes (df, criteria, rng) and returns a Follow: the transformed listings, the (possibly
changed) criteria, and `meta` that the relation's check needs (the scale factor, the month that was shifted, ...).
Nothing here touches the frozen snapshot on disk: `df` is copied before any change."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from housing_trends import Criteria
from housing_trends.context import add_months, month_range

ADD_X = 50_000                 # dollars added to every price of one month (MR7)
SCALE_FACTORS = (1.5, 2.0, 3.0)    # k > 1 for the scaling MR
ZIP_RAISE = 1.5                # factor applied to one ZIP's prices (MR10)
UNIT = 1000                    # USD -> USD thousands (MR4)


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
