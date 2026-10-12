"""Reproducible base inputs: a pool of distinct criteria sets (ZIP subsets x price band x beds/baths x window) that all
match enough listings in the frozen snapshot. Same seed + same snapshot => the same pool, so results can be re-run.

Each MR then takes the first N pool members that lie in its subdomain (see relations.MR.applies)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from housing_trends import Criteria
from housing_trends.agents.scanner import active_filters, filter_mask
from housing_trends.context import add_months, month_range

POOL_SEED = 20261016          # same as the snapshot seed: one number to remember
MIN_MATCHES = 50              # a base input must match at least this many listings, or it is not worth testing


@dataclass(frozen=True)
class BaseInput:
    id: int
    criteria: Criteria


def months_of(df: pd.DataFrame) -> list[str]:
    return month_range(df.month.min(), df.month.max())


def matching_rows(df: pd.DataFrame, c: Criteria) -> int:
    """Listings in the criteria's ZIPs and window that pass its filters (the Scanner's own filter helpers)."""
    in_scope = df.zip_code.isin(c.zips) & (df.month >= c.start_month) & (df.month <= c.end_month)
    return int(filter_mask(df, c, active_filters(c))[in_scope].sum())


def generate_pool(df: pd.DataFrame, *, size: int = 240, seed: int = POOL_SEED) -> list[BaseInput]:
    rng = np.random.default_rng(seed)
    all_zips = sorted(df.zip_code.unique())
    months = months_of(df)
    ends = months[11:]                                   # window ends that leave a full 12 months
    pool, attempts = [], 0
    while len(pool) < size and attempts < size * 20:
        attempts += 1
        zips = sorted(rng.choice(all_zips, size=int(rng.integers(4, 31)), replace=False).tolist())
        end = str(rng.choice(ends))
        band = rng.choice(["none", "none", "min", "max", "both"])          # ~40% without a price band
        lo = int(rng.choice(range(100_000, 400_001, 25_000)))
        hi = int(lo + rng.choice(range(100_000, 500_001, 25_000)))
        beds = rng.choice([0, 0, 2, 3, 4])
        baths = rng.choice([0.0, 0.0, 1.0, 2.0])
        c = Criteria(zips=tuple(zips), start_month=add_months(end, -11), end_month=end,
                     price_min=lo if band in ("min", "both") else None,
                     price_max=(hi if band in ("max", "both") else None),
                     beds_min=int(beds) or None, baths_min=float(baths) or None)
        if matching_rows(df, c) >= MIN_MATCHES:
            pool.append(BaseInput(len(pool), c))
    return pool
