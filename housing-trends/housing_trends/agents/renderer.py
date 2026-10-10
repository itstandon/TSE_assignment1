"""Rendering Agent (plain code): assemble the final, annotated result. Presentation only: it copies statistics,
never computes or alters them."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Report:
    criteria: object
    scope: object
    buckets: pd.DataFrame
    overall: pd.DataFrame
    explanations: list = field(default_factory=list)
    verdicts: list = field(default_factory=list)
    unresolved: list = field(default_factory=list)
    counters: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)


def _notes(r, persistent):
    n = []
    if r.n_pooled > 0:
        n.append(f"pooled (+-{r.time_pad_used} mo, +{r.zip_extra_used} ZIPs)")
    if r.thin:
        n.append(f"thin: {r.n_used} listings")
    if r.method != "median":
        n.append(r.method)
    if r.n_excluded > 0:
        n.append(f"{r.n_excluded} listing(s) excluded")
    if f"{r.zip_code}@{r.month}" in persistent:
        n.append("price jump unresolved")
    return "; ".join(n)


def make_renderer():
    def render(state):
        scan, stats = state["scan"], state["stats"]
        persistent = set(state.get("persistent", []))
        extra = scan.buckets[["zip_code", "month", "thin", "widened", "zip_extra_used", "time_pad_used"]]
        b = stats.buckets.merge(extra, on=["zip_code", "month"], how="left")
        b["notes"] = [_notes(r, persistent) for r in b.itertuples()]
        diag = state.get("diagnoses", [])
        unresolved = list(state.get("unresolved", []))
        bad = [d for d in diag if d["type"] == "verdict" and d["verdict"] == "unexplained"]
        if bad:
            unresolved.append(f"{len(bad)} outlier listing(s) excluded as unexplained data-quality issue(s): "
                              + ", ".join(str(d["listing_id"]) for d in bad))
        counters = {k: state.get(k, 0) for k in ("rescans", "recomputes", "diag_rounds")}
        rep = Report(state["criteria"], state.get("scope"), b, stats.overall,
                     list({(d["zip_code"], d["month"]): d for d in diag if d["type"] == "explanation"}.values()),
                     [d for d in diag if d["type"] == "verdict"],
                     unresolved, counters, list(state.get("trace", [])))
        return {"report": rep, "trace": [f"render: {len(b)} buckets, {len(unresolved)} unresolved item(s), loops used {counters}"]}
    return render