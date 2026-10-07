"""The single structured-criteria object + the ONE deterministic validator.

Both input paths (form, LLM-parsed free text) must pass through validate_and_structure(),
so downstream agents cannot tell which path produced a Criteria.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, ConfigDict

from .context import (SnapshotContext, add_months, month_index, parse_month)

WINDOW_MONTHS = 12
ALL_FILTERS = ("price", "beds", "baths", "sqft", "property_type")
DEFAULT_PROTECTED = ("price", "beds", "baths")   # Pooling may NOT relax these


class Criteria(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    zips: tuple[str, ...]
    start_month: str
    end_month: str
    price_min: Optional[int] = None
    price_max: Optional[int] = None
    beds_min: Optional[int] = None
    beds_max: Optional[int] = None
    baths_min: Optional[float] = None
    sqft_min: Optional[int] = None
    sqft_max: Optional[int] = None
    property_types: tuple[str, ...] = ()
    protected_filters: tuple[str, ...] = DEFAULT_PROTECTED

    def key(self) -> str:
        """Stable fingerprint: equal criteria => equal key (for caching / MR comparisons)."""
        return hashlib.sha1(self.model_dump_json().encode()).hexdigest()[:12]


@dataclass
class InputResult:
    criteria: Optional[Criteria] = None
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    assumptions: list = field(default_factory=list)   # how vague terms were resolved (LLM path)
    unsupported: list = field(default_factory=list)   # request parts we cannot honour
    mode: str = ""                                    # provenance, for the log only
    attempts: int = 0
    trace: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.criteria is not None and not self.errors


# ---- coercion helpers --------------------------------------------------------
def _num(v, name, errors, *, integer=False, lo=0.0, hi=None):
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    try:
        if isinstance(v, str):
            s, mult = v.strip().lower().replace("$", "").replace(",", ""), 1
            if s.endswith("k"):
                s, mult = s[:-1], 1_000
            elif s.endswith("m"):
                s, mult = s[:-1], 1_000_000
            x = float(s) * mult
        else:
            x = float(v)
    except (TypeError, ValueError):
        errors.append(f"{name}: not a number ({v!r})")
        return None
    if not math.isfinite(x) or x < lo or (hi is not None and x > hi):
        errors.append(f"{name}: {v!r} out of range")
        return None
    if integer:
        if x != int(x):
            errors.append(f"{name}: must be a whole number ({v!r})")
            return None
        return int(x)
    return x


def _as_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        return [p for p in re.split(r"[,;\s]+", v.strip()) if p]
    if isinstance(v, (int, float)):
        return [v]
    return list(v)


def _norm_zips(v, errors):
    out = set()
    for z in _as_list(v):
        s = str(z).strip()
        s = s[:-2] if s.endswith(".0") else s
        if re.fullmatch(r"\d{5}", s):
            out.add(s)
        else:
            errors.append(f"zip: {z!r} is not a 5-digit ZIP")
    return sorted(out)


def _check_pair(lo, hi, name, errors):
    if lo is not None and hi is not None and lo > hi:
        errors.append(f"{name}: min ({lo}) exceeds max ({hi})")


def _resolve_window(raw, ctx, errors, warnings):
    try:
        s, e = parse_month(raw.get("start_month")), parse_month(raw.get("end_month"))
    except ValueError as ex:
        errors.append(str(ex))
        return None, None
    if s is None and e is None:
        e = ctx.last_month
        s = add_months(e, -(WINDOW_MONTHS - 1))
        warnings.append(f"no date range given; defaulted to the last {WINDOW_MONTHS} months of the snapshot ({s}..{e})")
    elif s is None:
        s = add_months(e, -(WINDOW_MONTHS - 1))
    elif e is None:
        e = add_months(s, WINDOW_MONTHS - 1)
    elif month_index(e) - month_index(s) + 1 != WINDOW_MONTHS:
        errors.append(f"date range {s}..{e} spans {month_index(e) - month_index(s) + 1} months; the trend needs exactly {WINDOW_MONTHS}")
        return None, None
    if month_index(s) < month_index(ctx.first_month) or month_index(e) > month_index(ctx.last_month):
        errors.append(f"window {s}..{e} falls outside snapshot coverage {ctx.first_month}..{ctx.last_month}")
        return None, None
    return s, e


# ---- the validator -------------------------------------------------------------
_KNOWN = set(Criteria.model_fields)


def validate_and_structure(raw: dict, ctx: SnapshotContext) -> InputResult:
    """Pure function: same (raw, ctx) -> same result. No LLM, no clock, no randomness."""
    errors, warnings = [], []
    raw = dict(raw or {})
    for k in sorted(set(raw) - _KNOWN):
        warnings.append(f"ignored unknown field {k!r}")

    zips = _norm_zips(raw.get("zips"), errors)
    if not zips and not errors:
        errors.append("zips: select at least one ZIP code")
    unknown = [z for z in zips if z not in ctx.valid_zips]
    if unknown:
        errors.append(f"zips not in snapshot: {', '.join(unknown)}")

    price_min = _num(raw.get("price_min"), "price_min", errors, integer=True)
    price_max = _num(raw.get("price_max"), "price_max", errors, integer=True)
    beds_min = _num(raw.get("beds_min"), "beds_min", errors, integer=True, hi=20)
    beds_max = _num(raw.get("beds_max"), "beds_max", errors, integer=True, hi=20)
    baths_min = _num(raw.get("baths_min"), "baths_min", errors, hi=20)
    sqft_min = _num(raw.get("sqft_min"), "sqft_min", errors, integer=True)
    sqft_max = _num(raw.get("sqft_max"), "sqft_max", errors, integer=True)
    for lo, hi, n in [(price_min, price_max, "price"), (beds_min, beds_max, "beds"), (sqft_min, sqft_max, "sqft")]:
        _check_pair(lo, hi, n, errors)

    types = sorted({str(t).strip().lower() for t in _as_list(raw.get("property_types")) if str(t).strip()})
    if types and not ctx.property_types:
        warnings.append("snapshot has no property-type column; property_types filter dropped")
        types = []
    elif types:
        bad = [t for t in types if t not in ctx.property_types]
        if bad:
            errors.append(f"property_types not in snapshot: {', '.join(bad)} (valid: {', '.join(sorted(ctx.property_types))})")

    prot = raw.get("protected_filters")
    prot = list(DEFAULT_PROTECTED) if prot is None else sorted({str(p).strip().lower() for p in _as_list(prot)})
    bad = [p for p in prot if p not in ALL_FILTERS]
    if bad:
        errors.append(f"protected_filters: unknown {', '.join(bad)} (valid: {', '.join(ALL_FILTERS)})")

    start, end = _resolve_window(raw, ctx, errors, warnings)

    if errors:
        return InputResult(errors=errors, warnings=warnings)
    crit = Criteria(zips=tuple(zips), start_month=start, end_month=end,
                    price_min=price_min, price_max=price_max, beds_min=beds_min, beds_max=beds_max,
                    baths_min=baths_min, sqft_min=sqft_min, sqft_max=sqft_max,
                    property_types=tuple(types), protected_filters=tuple(prot))
    return InputResult(criteria=crit, warnings=warnings)