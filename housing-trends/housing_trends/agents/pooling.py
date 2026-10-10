"""Pooling Agent (plain code, no LLM): decide HOW to widen the scope for buckets that are still thin.

Invoked only by the Critic's coverage check (a "Data Issue"). It reads the Scanner's per-bucket flags and
returns a revised Scope for the Scanner to re-run (Cycle A).

Per thin-but-fixable bucket (thin and `reachable`), in this order of preference (least change to the user's question):
  1. time     if one more month of padding can still be enough on its own:  n + gain_time >= min_listings
  2. filters  relax the first still-relaxable (set, NON-protected) filter, if that would add listings   [global]
  3. zip      pool 2 more neighbouring ZIPs, if that would add listings
  4. time     as partial progress, if padding adds listings but cannot suffice alone
This is where the Scanner's flags change what Pooling does (design note, section 5). Buckets that are healthy,
or that no allowed move can improve, are left alone. Returns None when no move helps anywhere (Pooling stops).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..criteria import Criteria
from ..state import (MAX_TIME_PAD, MAX_ZIP_EXTRA, MIN_LISTINGS, TIME_STEP, ZIP_STEP, BucketScope, Scope)
from .scanner import ScanResult, relaxable_filters


@dataclass
class Plan:
    scope: Scope
    counts: dict            # {"time": n_buckets, "zip": n_buckets, "filters": [names]}
    thin: int
    fixable: int

    def describe(self) -> str:
        parts = []
        if self.counts["time"]: parts.append(f"time +{TIME_STEP} month for {self.counts['time']} bucket(s)")
        if self.counts["zip"]: parts.append(f"ZIP +{ZIP_STEP} neighbours for {self.counts['zip']} bucket(s)")
        if self.counts["filters"]: parts.append(f"relax filter {self.counts['filters']}")
        return f"{self.fixable} fixable of {self.thin} thin: " + "; ".join(parts)


def plan_widening(c: Criteria, scope: Scope, scan: ScanResult, min_listings: int = MIN_LISTINGS) -> Plan | None:
    b = scan.buckets
    thin = b[b.thin]
    fix = thin[thin.reachable]
    if fix.empty:
        return None
    rel = relaxable_filters(c, scope)
    new, n_time, n_zip, relax = {}, 0, 0, False
    for r in fix.itertuples():
        ze, tp = scope.depth(r.zip_code, r.month)
        time_ok, zip_ok = tp < MAX_TIME_PAD and r.gain_time > 0, ze < MAX_ZIP_EXTRA and r.gain_zip > 0
        if time_ok and r.n + r.gain_time >= min_listings:
            new[(r.zip_code, r.month)] = (ze, min(MAX_TIME_PAD, tp + TIME_STEP)); n_time += 1
        elif rel and r.gain_filters > 0:
            relax = True
        elif zip_ok:
            new[(r.zip_code, r.month)] = (min(MAX_ZIP_EXTRA, ze + ZIP_STEP), tp); n_zip += 1
        elif time_ok:
            new[(r.zip_code, r.month)] = (ze, min(MAX_TIME_PAD, tp + TIME_STEP)); n_time += 1
    if not new and not relax:
        return None
    keep = {(x.zip_code, x.month): (x.zip_extra, x.time_pad) for x in scope.per_bucket}
    keep.update(new)
    per = tuple(BucketScope(zip_code=z, month=m, zip_extra=e, time_pad=t) for (z, m), (e, t) in keep.items())
    relaxed = scope.relaxed + ((rel[0],) if relax else ())
    return Plan(Scope(zip_extra=scope.zip_extra, time_pad=scope.time_pad, relaxed=relaxed, per_bucket=per),
                {"time": n_time, "zip": n_zip, "filters": [rel[0]] if relax else []}, len(thin), len(fix))


def make_pooling(min_listings: int = MIN_LISTINGS):
    """LangGraph node factory: reads scan + scope, writes a wider scope and bumps the rescan counter."""
    def pooling(state):
        plan = plan_widening(state["criteria"], state.get("scope") or Scope(), state["scan"], min_listings)
        n = state.get("rescans", 0) + 1
        if plan is None:                                    # the Critic only routes here when a plan exists
            return {"rescans": n, "trace": [f"pooling: rescan {n}: no move can help; scope unchanged"]}
        return {"scope": plan.scope, "rescans": n, "trace": [f"pooling: rescan {n}: {plan.describe()}"]}
    return pooling