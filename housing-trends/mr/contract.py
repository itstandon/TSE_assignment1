"""Output contract. Every metamorphic relation is a statement about THESE numbers and nothing else.

"The 12-month trend" for one Criteria is one row per month of the 12-month window, over all requested ZIPs
(own listings only, so pooled listings are never double counted; this is `Report.overall`):

  price           median list price of the month
  ppsf            median price per square foot of the month (listings without a usable house size are skipped)
  n               number of active listings behind the month (after the Diagnoser's exclusions)
  mom_price_pct   month-over-month % change of `price` (NaN for the first month)

Per-ZIP tables, notes, loop counters and the decision trace are NOT part of the contract: they are diagnostics.
"""
from __future__ import annotations

import pandas as pd

CONTRACT_COLUMNS = ["price", "ppsf", "n", "mom_price_pct"]


def contract(report) -> pd.DataFrame:
    """Report -> DataFrame indexed by month ('YYYY-MM') with the contract columns, as floats."""
    o = report.overall.sort_values("month")
    return o.set_index("month")[CONTRACT_COLUMNS].astype(float)
