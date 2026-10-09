"""Statistician Agent (plain code, no LLM): compute the requested statistic per ZIP-month bucket and overall,
plus month-over-month % change. It is told WHAT to compute (method, per-bucket overrides, listings to exclude)
and never inspects or judges its own output; deciding that a number looks wrong is the Critic's job.

Output contract (every MR is a statement about these):
  buckets  one row per requested ZIP x month
           price, ppsf        the statistic of listing price / price per sqft (see `method`)
           n_used             listings behind the statistic (own + pooled, after exclusions)
           n_own, n_pooled    own listings of the bucket / listings borrowed by Pooling
           n_excluded         members dropped because their listing_id was excluded
           n_ppsf             listings with a usable house_size (price per sqft needs it)
           mom_price_pct, mom_ppsf_pct, mom_count_pct   % change vs previous month in the same ZIP
           (count change uses n_own, so pooling cannot hide a real drop in listing volume)
  overall  one row per month over ALL requested ZIPs, own listings only (never double-counts pooled ones)

Methods (the Critic / Pooling choose; the Statistician only executes):
  median             default
  trimmed_mean       mean of the middle (1 - 2*trim) of the DISTRIBUTION. Fractional trimming, so duplicating every
                     listing leaves it unchanged (a floor()-based trim would not: an MR-relevant design choice).
  median_of_medians  median of the per-origin (ZIP, month) medians; equals the plain median for an unpooled bucket,
                     so use trimmed_mean to be robust inside a single bucket.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .scanner import ScanResult, Snapshot

NAN = float("nan")


# ---------------------------------------------------------------- instruction
class Method(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: Literal["median", "trimmed_mean", "median_of_medians"] = "median"
    trim: float = Field(default=0.1, ge=0.0, lt=0.5)       # only used by trimmed_mean

    def label(self) -> str:
        return f"trimmed_mean({self.trim:g})" if self.name == "trimmed_mean" else self.name


class BucketOverride(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    zip_code: str
    month: str
    method: Method


class StatInstruction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    method: Method = Method()
    overrides: tuple[BucketOverride, ...] = ()
    exclude_ids: tuple[int, ...] = ()

    @field_validator("exclude_ids")
    @classmethod
    def _canon_ids(cls, v):
        return tuple(sorted(set(v)))

    @model_validator(mode="after")
    def _unique_overrides(self):
        keys = [(o.zip_code, o.month) for o in self.overrides]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate bucket overrides")
        return self


# ---------------------------------------------------------------- estimators
def trimmed_mean(values: np.ndarray, trim: float) -> float:
    """Mean of the empirical distribution between its `trim` and `1-trim` quantile masses (fractional trimming)."""
    v = np.sort(np.asarray(values, dtype=float))
    n = v.size
    if n == 0:
        return NAN
    lo, hi = trim * n, n - trim * n
    idx = np.arange(n)
    w = np.clip(np.minimum(idx + 1, hi) - np.maximum(idx, lo), 0.0, None)
    return float((w * v).sum() / w.sum())


def _stat(g: pd.DataFrame, col: str, m: Method, origin: list[str]) -> float:
    d = g.dropna(subset=[col])
    if d.empty:
        return NAN
    if m.name == "median":
        return float(np.median(d[col].to_numpy()))
    if m.name == "trimmed_mean":
        return trimmed_mean(d[col].to_numpy(), m.trim)
    return float(np.median(d.groupby(origin)[col].median().to_numpy()))


def _mom(prev: pd.Series, cur: pd.Series) -> pd.Series:
    out = (cur - prev) / prev * 100.0
    return out.where(prev.notna() & cur.notna() & (prev > 0))


# ---------------------------------------------------------------- result
@dataclass
class StatResult:
    instruction: StatInstruction
    buckets: pd.DataFrame
    overall: pd.DataFrame

    def summary(self) -> str:
        b, i = self.buckets, self.instruction
        return (f"method={i.method.label()}, {len(b)} buckets ({int(b.price.isna().sum())} empty), "
                f"{len(i.overrides)} override(s), {len(i.exclude_ids)} excluded id(s)")


def compute(scan: ScanResult, snap: Snapshot, instr: StatInstruction | None = None) -> StatResult:
    instr = instr or StatInstruction()
    lst = (snap.df.set_index("listing_id")[["price", "house_size", "zip_code", "month"]]
           .rename(columns={"zip_code": "o_zip", "month": "o_month"}))
    mem = scan.members.join(lst, on="listing_id")
    mem["ppsf"] = (mem.price / mem.house_size).where(mem.house_size > 0)
    gone = mem.listing_id.isin(instr.exclude_ids)
    kept = mem[~gone]
    n_exc = mem[gone].groupby(["zip_code", "month"]).size().to_dict()
    groups = {k: g for k, g in kept.groupby(["zip_code", "month"])}
    ov = {(o.zip_code, o.month): o.method for o in instr.overrides}
    origin = ["o_zip", "o_month"]

    rows = []
    for b in scan.buckets.itertuples():
        key = (b.zip_code, b.month)
        m = ov.get(key, instr.method)
        g = groups.get(key)
        if g is None:
            rows.append(dict(zip_code=b.zip_code, month=b.month, n_used=0, n_own=0, n_pooled=0,
                             n_excluded=n_exc.get(key, 0), n_ppsf=0, method=m.label(), price=NAN, ppsf=NAN))
            continue
        own = int(g.own.sum())
        rows.append(dict(zip_code=b.zip_code, month=b.month, n_used=len(g), n_own=own, n_pooled=len(g) - own,
                         n_excluded=n_exc.get(key, 0), n_ppsf=int(g.ppsf.notna().sum()), method=m.label(),
                         price=_stat(g, "price", m, origin), ppsf=_stat(g, "ppsf", m, origin)))
    buckets = pd.DataFrame(rows).sort_values(["zip_code", "month"]).reset_index(drop=True)
    grp = buckets.groupby("zip_code")
    for col, name in (("price", "mom_price_pct"), ("ppsf", "mom_ppsf_pct"), ("n_own", "mom_count_pct")):
        buckets[name] = _mom(grp[col].shift(1), buckets[col])

    own_rows = kept[kept.own]
    gone_own = mem[gone & mem.own]
    months = sorted(scan.buckets.month.unique())
    og = {k: g for k, g in own_rows.groupby("month")}
    ex = gone_own.groupby("month").size().to_dict()
    orows = []
    for mo in months:
        g = og.get(mo)
        orows.append(dict(month=mo, n=0 if g is None else len(g), n_excluded=ex.get(mo, 0),
                          n_ppsf=0 if g is None else int(g.ppsf.notna().sum()), method=instr.method.label(),
                          price=NAN if g is None else _stat(g, "price", instr.method, ["o_zip"]),
                          ppsf=NAN if g is None else _stat(g, "ppsf", instr.method, ["o_zip"])))
    overall = pd.DataFrame(orows)
    for col, name in (("price", "mom_price_pct"), ("ppsf", "mom_ppsf_pct"), ("n", "mom_count_pct")):
        overall[name] = _mom(overall[col].shift(1), overall[col])
    return StatResult(instr, buckets, overall)


def make_statistician(snap: Snapshot):
    """LangGraph node factory: reads state['scan'] and state['stat_instruction'], writes state['stats']."""
    def statistician(state):
        res = compute(state["scan"], snap, state.get("stat_instruction"))
        return {"stats": res, "trace": [f"statistician: {res.summary()}"]}
    return statistician