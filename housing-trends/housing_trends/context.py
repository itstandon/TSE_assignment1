"""Facts about the frozen snapshot that the Input Handler validates against.

Everything "relative" (default window, 'affordable' thresholds) is derived from the
snapshot, never from today's date, so results are reproducible.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd


# ---- month helpers ("YYYY-MM" strings) -------------------------------------
def parse_month(v) -> str | None:
    """Accept 'YYYY-MM', 'YYYY-MM-DD', date/datetime. Returns 'YYYY-MM' or None."""
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    if hasattr(v, "year") and hasattr(v, "month"):
        return f"{v.year:04d}-{v.month:02d}"
    m = re.fullmatch(r"(\d{4})-(\d{1,2})(?:-\d{1,2})?", str(v).strip())
    if not m or not 1 <= int(m[2]) <= 12:
        raise ValueError(f"month: cannot parse {v!r} (use YYYY-MM)")
    return f"{int(m[1]):04d}-{int(m[2]):02d}"


def month_index(s: str) -> int:
    y, m = s.split("-")
    return int(y) * 12 + int(m) - 1


def from_index(i: int) -> str:
    return f"{i // 12:04d}-{i % 12 + 1:02d}"


def add_months(s: str, n: int) -> str:
    return from_index(month_index(s) + n)


def month_range(first: str, last: str) -> list[str]:
    return [from_index(i) for i in range(month_index(first), month_index(last) + 1)]


# ---- snapshot context ------------------------------------------------------
@dataclass(frozen=True)
class SnapshotContext:
    valid_zips: frozenset
    zip_to_city: dict
    first_month: str
    last_month: str
    property_types: frozenset = frozenset()       # empty => snapshot has no such column
    price_quantiles: dict = field(default_factory=dict)
    area_gazetteer: dict = field(default_factory=dict)  # e.g. {"north dallas": ["75230", ...]}

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, *, zip_col="zip_code", city_col="city",
                       price_col="price", month_col="month", type_col=None,
                       area_gazetteer=None) -> "SnapshotContext":
        zips = df[zip_col].astype(str).str.extract(r"(\d{5})")[0]
        tmp = pd.DataFrame({"zip": zips, "city": df[city_col].astype(str).str.strip().str.title()}).dropna()
        zip_to_city = tmp.groupby("zip")["city"].agg(lambda s: s.mode().iat[0]).to_dict()
        months = pd.to_datetime(df[month_col]).dt.strftime("%Y-%m")
        q = df[price_col].quantile([0.25, 0.5, 0.75])
        types = frozenset(df[type_col].dropna().astype(str).str.lower().unique()) if type_col else frozenset()
        return cls(
            valid_zips=frozenset(zip_to_city),
            zip_to_city=zip_to_city,
            first_month=months.min(),
            last_month=months.max(),
            property_types=types,
            price_quantiles={"p25": int(q[0.25]), "p50": int(q[0.5]), "p75": int(q[0.75])},
            area_gazetteer={k.lower(): [str(z) for z in v] for k, v in (area_gazetteer or {}).items()},
        )