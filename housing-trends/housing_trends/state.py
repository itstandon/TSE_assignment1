"""Shared pipeline state + the scope Pooling is allowed to widen.

Pooling's three moves (design note, section 2.5) are the three axes of Scope:
  zip_extra : widen ZIP radius       -> pool the k nearest other ZIPs into each requested ZIP
  time_pad  : extend the date window -> pool +-k neighbouring months into each requested month
  relaxed   : relax non-protected filters (never anything in Criteria.protected_filters)
The Scanner reports, per bucket, what each move WOULD add (see agents/scanner.py).
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator

MIN_LISTINGS = 20                  # Critic's coverage threshold per ZIP-month bucket
ZIP_STEP, TIME_STEP = 2, 1         # how much one Pooling rescan widens an axis
MAX_ZIP_EXTRA, MAX_TIME_PAD = 6, 3  # hard limits on widening (ZIP neighbours / months each side)
FILTERS = ("price", "beds", "baths", "sqft", "property_type")


class Scope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    zip_extra: int = Field(default=0, ge=0, le=MAX_ZIP_EXTRA)
    time_pad: int = Field(default=0, ge=0, le=MAX_TIME_PAD)
    relaxed: tuple[str, ...] = ()

    @field_validator("relaxed")
    @classmethod
    def _canon(cls, v):
        bad = [f for f in v if f not in FILTERS]
        if bad:
            raise ValueError(f"unknown filter(s) to relax: {bad}")
        return tuple(sorted(set(v)))


class PipelineState(TypedDict, total=False):
    """LangGraph state. Later agents add fields (stats, verdicts, counters)."""
    criteria: Any                       # housing_trends.Criteria
    scope: Scope
    scan: Any                           # ScanResult
    rescans: int                        # Cycle A counter (cap 3), owned by Pooling
    trace: Annotated[list, operator.add]  # append-only execution log