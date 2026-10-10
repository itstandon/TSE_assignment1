"""Per-ZIP reference distributions (computed once from the snapshot).

Used for (a) the Critic's listing-level outlier screen: robust z-score of log(price) within the listing's ZIP, and
(b) the evidence the Diagnoser judges from: where a listing's lot size, house size and price sit among its ZIP's listings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class ZipProfile:
    FIELDS = ("price", "house_size", "acre_lot")

    def __init__(self, df: pd.DataFrame):
        self._ref, self._sorted = {}, {}
        for z, g in df.groupby("zip_code"):
            lp = np.log(g.price[g.price > 0].to_numpy(float))
            med = float(np.median(lp))
            mad = float(np.median(np.abs(lp - med)))
            self._ref[z] = (med, max(1.4826 * mad, 0.05))              # floor keeps a tiny MAD from exploding z
            self._sorted[z] = {f: np.sort(g[f].dropna().to_numpy(float)) for f in self.FIELDS}

    def median_price(self, z: str) -> float:
        return float(np.exp(self._ref[z][0]))

    def z_price(self, z: str, price: float) -> float:
        med, sd = self._ref[z]
        return float((np.log(price) - med) / sd) if price and price > 0 else float("nan")

    def pct(self, z: str, field: str, value) -> float | None:
        """Percentile rank (0-100) of `value` among the ZIP's listings; None if unknown."""
        arr = self._sorted[z][field]
        if value is None or (isinstance(value, float) and np.isnan(value)) or arr.size == 0:
            return None
        return round(100.0 * np.searchsorted(arr, value, side="right") / arr.size, 1)