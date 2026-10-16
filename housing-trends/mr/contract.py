"""Output contract. Every metamorphic relation is a statement about THESE numbers and nothing else.

"The 12-month trend" for one Criteria has two tables, both read from the final Report:

1. OVERALL: one row per month of the 12-month window, over all requested ZIPs (own listings only, so pooled listings
   are never double counted; this is `Report.overall`). Returned by `contract()`:
     price           median list price of the month
     ppsf            median price per square foot of the month (listings without a usable house size are skipped)
     n               number of active listings behind the month (after the Diagnoser's exclusions)
     mom_price_pct   month-over-month % change of `price` (NaN for the first month)

2. PER ZIP-MONTH: one row per requested ZIP x month (`Report.buckets`), indexed "zip@month", attached to the overall
   table as `contract(...).attrs["buckets"]` and read with `zip_table()`:
     price, ppsf     the bucket statistic (pooled neighbours included when Pooling widened a thin bucket)
     n_own           the bucket's own listings (never pooled ones)
     n_used          listings behind the statistic (own + pooled, after exclusions)
     mom_price_pct   month-over-month % change within the ZIP

Diagnoser verdicts (listing id, ZIP, verdict) ride along as `attrs["verdicts"]` so MR8 can ask whether a planted outlier
was ever examined. Notes, loop counters and the decision trace are NOT part of the contract: they are diagnostics.
"""
from __future__ import annotations

import pandas as pd

CONTRACT_COLUMNS = ["price", "ppsf", "n", "mom_price_pct"]
ZIP_COLUMNS = ["price", "ppsf", "n_own", "n_used", "mom_price_pct"]


def contract(report) -> pd.DataFrame:
    """Report -> DataFrame indexed by month ('YYYY-MM') with the contract columns, as floats."""
    o = report.overall.sort_values("month")
    out = o.set_index("month")[CONTRACT_COLUMNS].astype(float)
    b = report.buckets.sort_values(["zip_code", "month"])
    z = b.set_index(b.zip_code + "@" + b.month)[ZIP_COLUMNS].astype(float)
    z["zip_code"], z["month"] = b.zip_code.to_numpy(), b.month.to_numpy()
    out.attrs["buckets"] = z
    out.attrs["verdicts"] = [(int(v["listing_id"]), str(v["zip_code"]), str(v["verdict"])) for v in report.verdicts]
    return out


def zip_table(c: pd.DataFrame) -> pd.DataFrame:
    """The per ZIP-month table of a contract (empty if the table was built by hand without one)."""
    return c.attrs.get("buckets", pd.DataFrame(columns=ZIP_COLUMNS + ["zip_code", "month"]))
