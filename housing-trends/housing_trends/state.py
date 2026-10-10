"""Shared pipeline state, tunable constants, and the scope Pooling is allowed to widen.

Pooling's three moves (design note, section 2.5) are the axes of Scope:
  zip_extra : widen ZIP radius       -> pool the k nearest other ZIPs into a bucket
  time_pad  : extend the date window -> pool +-k neighbouring months into a bucket
  relaxed   : relax non-protected filters (never anything in Criteria.protected_filters)
Depth (zip_extra / time_pad) is tracked PER BUCKET, so a bucket that needs one extra month never receives
the three that a sparser bucket needs. `Scope.zip_extra/time_pad` are the defaults for buckets without an entry.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---- coverage loop (Critic -> Pooling -> Scanner)
MIN_LISTINGS = 20                  # coverage threshold per ZIP-month bucket
ZIP_STEP, TIME_STEP = 2, 1         # how much one Pooling rescan widens a bucket
MAX_ZIP_EXTRA, MAX_TIME_PAD = 6, 3  # hard limits on widening (ZIP neighbours / months each side)
MAX_RESCANS = 3                    # cap on Cycle A
# ---- method loop (Critic -> Statistician)
JUMP_PCT = 30.0                    # month-over-month PRICE change that counts as a jump
TRIM = 0.1                         # trim fraction of the robust recompute
MAX_RECOMPUTES = 1                 # cap on Cycle B
# ---- outlier loop (Critic -> Diagnoser -> Statistician)
SUSPECT_Z = 3.5                    # robust z of log price within its ZIP
MAX_SUSPECTS, MAX_ANOMALIES = 8, 6  # per Diagnoser round (LLM cost bound)
MAX_DIAG_ROUNDS = 2                # cap on Cycle C

FILTERS = ("price", "beds", "baths", "sqft", "property_type")


class BucketScope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    zip_code: str
    month: str
    zip_extra: int = Field(default=0, ge=0, le=MAX_ZIP_EXTRA)
    time_pad: int = Field(default=0, ge=0, le=MAX_TIME_PAD)


class Scope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    zip_extra: int = Field(default=0, ge=0, le=MAX_ZIP_EXTRA)
    time_pad: int = Field(default=0, ge=0, le=MAX_TIME_PAD)
    relaxed: tuple[str, ...] = ()
    per_bucket: tuple[BucketScope, ...] = ()

    @field_validator("relaxed")
    @classmethod
    def _canon(cls, v):
        bad = [f for f in v if f not in FILTERS]
        if bad:
            raise ValueError(f"unknown filter(s) to relax: {bad}")
        return tuple(sorted(set(v)))

    @field_validator("per_bucket")
    @classmethod
    def _unique(cls, v):
        keys = [(b.zip_code, b.month) for b in v]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate per-bucket scopes")
        return tuple(sorted(v, key=lambda b: (b.zip_code, b.month)))

    def depth(self, zip_code: str, month: str) -> tuple[int, int]:
        """(zip_extra, time_pad) in force for one bucket."""
        for b in self.per_bucket:
            if b.zip_code == zip_code and b.month == month:
                return b.zip_extra, b.time_pad
        return self.zip_extra, self.time_pad


class PipelineState(TypedDict, total=False):
    """LangGraph state. Every node returns a partial update; `trace` is append-only."""
    criteria: Any                       # housing_trends.Criteria
    scope: Scope
    scan: Any                           # ScanResult
    stat_instruction: Any               # StatInstruction (Critic / Diagnoser edit it; default = plain median)
    stats: Any                          # StatResult
    route: str                          # Critic's decision: pooling | recompute | diagnoser | render
    rescans: int                        # Cycle A counter (cap MAX_RESCANS)
    recomputes: int                     # Cycle B counter (cap MAX_RECOMPUTES)
    diag_rounds: int                    # Cycle C counter (cap MAX_DIAG_ROUNDS)
    jumps_initial: int                  # price jumps seen before the robust recompute
    persistent: list                    # "zip@month" jumps still present when the Critic last looked
    anomalies: Any                      # list[dict]: persistent jumps handed to the Diagnoser
    suspects: Any                       # list[dict] of outlier-listing evidence for the Diagnoser
    diagnoses: list                     # all verdicts / explanations so far
    unresolved: list                    # human-readable list of problems we could not resolve
    report: Any                         # Report (set by the renderer)
    trace: Annotated[list, operator.add]  # append-only execution log