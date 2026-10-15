"""Critic Agent (plain code, no LLM): the reviewer. It owns every stop / iterate decision.

Each time it runs it takes the FIRST applicable step, so the three loops are mutually ordered and all bounded:

  1. COVERAGE (Cycle A: Critic -> Pooling -> Scanner)
       thin buckets (n < MIN_LISTINGS) exist, Pooling still has a move, and rescans < MAX_RESCANS  -> route "pooling".
       Otherwise coverage is closed; whatever is still thin is reported as unresolved (cap reached / no move helps).
  2. TREND CHECK (price only; listing-count changes are reported but not flagged: see NOTEBOOK)
       A month-over-month price change > JUMP_PCT in some ZIP is a "jump".
       First sighting -> METHOD LOOP (Cycle B: Critic -> Statistician): robust recompute (trimmed mean) over the
       whole ZIP series, once (MAX_RECOMPUTES). Jumps that vanish are labelled estimator artifacts.
  3. OUTLIER LOOP (Cycle C: Critic -> Diagnoser -> Statistician)
       Jumps that survive the robust recompute are "persistent anomalies". The Critic screens the listings behind
       them for outliers (robust z of log price within the ZIP) and hands both to the Diagnoser (LLM), at most
       MAX_DIAG_ROUNDS times. Whatever still persists is reported as unresolved.
  4. Otherwise route "render".
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..context import add_months, month_range
from ..parser_agent import _extract_json as extract_json
from ..state import (JUMP_PCT, MAX_ANOMALIES, MAX_DIAG_ROUNDS, MAX_RECOMPUTES, MAX_RESCANS, MAX_SUSPECTS,
                     MIN_LISTINGS, SUSPECT_Z, TRIM, Scope)
from .pooling import plan_widening
from .profile import ZipProfile
from .scanner import Snapshot
from .statistician import BucketOverride, Method, StatInstruction


def jump_flags(buckets: pd.DataFrame, jump_pct: float = JUMP_PCT) -> pd.DataFrame:
    d = buckets[buckets.mom_price_pct.abs() > jump_pct]
    return d[["zip_code", "month", "mom_price_pct", "n_used", "n_own"]].reset_index(drop=True)


def robust_instruction(instr: StatInstruction, zips, months, trim: float = TRIM) -> StatInstruction:
    """Trimmed-mean override for every month of each given ZIP (keeps the month-over-month series consistent)."""
    ov = {(o.zip_code, o.month): o for o in instr.overrides}
    for z in zips:
        for m in months:
            ov[(z, m)] = BucketOverride(zip_code=z, month=m, method=Method(name="trimmed_mean", trim=trim))
    return StatInstruction(method=instr.method, overrides=tuple(ov.values()), exclude_ids=instr.exclude_ids)


def _py(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return x.item() if hasattr(x, "item") else x


def evidence(flagged: pd.DataFrame, scan, stats, snap: Snapshot, profile: ZipProfile, excluded=()):
    """Anomalies (capped) and outlier-listing suspects (capped) for the Diagnoser. Pure function of its inputs."""
    top = flagged.reindex(flagged.mom_price_pct.abs().sort_values(ascending=False, kind="stable").index).head(MAX_ANOMALIES)
    lst = snap.df.set_index("listing_id")
    b = stats.buckets.set_index(["zip_code", "month"])
    anomalies, cand = [], {}
    for r in top.itertuples():
        prev = add_months(r.month, -1)
        facts = {"zip_code": r.zip_code, "month": r.month, "prev_month": prev, "mom_price_pct": round(float(r.mom_price_pct), 1)}
        for tag, mo in (("now", r.month), ("prev", prev)):
            if (r.zip_code, mo) in b.index:
                row = b.loc[(r.zip_code, mo)]
                mem = scan.members[(scan.members.zip_code == r.zip_code) & (scan.members.month == mo)]
                sizes = lst.loc[mem.listing_id, "house_size"]
                facts[tag] = {"n_used": int(row.n_used), "n_own": int(row.n_own), "price": _py(round(float(row.price), 0)) if row.price == row.price else None,
                              "median_house_size": _py(float(sizes.median())) if sizes.notna().any() else None}
                for lid in mem.listing_id:
                    if lid not in excluded:
                        cand[int(lid)] = r.zip_code
        anomalies.append(facts)
    rows = []
    for lid, z in cand.items():
        x = lst.loc[lid]
        zscore = profile.z_price(z, x.price)
        if abs(zscore) > SUSPECT_Z:
            rows.append(dict(listing_id=lid, zip_code=z, month=x.month, price=_py(float(x.price)), bed=_py(x.bed), bath=_py(x.bath),
                             house_size=_py(x.house_size), acre_lot=_py(x.acre_lot), z_price=round(float(zscore), 1),
                             price_vs_zip_median=round(float(x.price / profile.median_price(z)), 2),
                             acre_lot_pct=profile.pct(z, "acre_lot", x.acre_lot), house_size_pct=profile.pct(z, "house_size", x.house_size),
                             price_pct=profile.pct(z, "price", x.price)))
    rows.sort(key=lambda d: (-abs(d["z_price"]), d["listing_id"]))
    return anomalies, [{k: _py(v) for k, v in d.items()} for d in rows[:MAX_SUSPECTS]]


# ---------------------------------------------------------------- the stop / iterate decision (the LLM's job)
MARK = "CRITIC_JSON:\n"

SYSTEM_PROMPT = """You are the critic of a housing price-trend system: you decide whether it should iterate again or
stop and accept what it has. Iterating costs time and distorts the user's question; stopping too early ships a number
nobody should trust. CRITIC_JSON states the open question, the evidence, and the options you may choose from.

question "coverage": some ZIP-month buckets hold fewer listings than the threshold. Choosing "pool" widens the scope
(borrowing neighbouring months, ZIPs or relaxing a filter) and rescans; choosing "accept" keeps the thin buckets and
reports them as unresolved. Weigh how many buckets are thin, how far below the threshold they are, and how many
rescans have already been spent.

question "trend": a month-over-month price change larger than the threshold is still flagged in some ZIP.
- "recompute": recompute those ZIP series with a robust trimmed mean. Cheap, and it tells you whether the jump was an
  artefact of the plain median. Only offered before any robust recompute has run.
- "diagnoser": send the outlier listings behind the jump to a data-quality analyst that may exclude bad listings.
  Only worth it when there are outlier suspects to judge.
- "accept": the jump is real or unexplainable; report it as unresolved.

Choose exactly one option from "options". Reply with ONE JSON object and nothing else:
{"decision":"<one of options>","reason":"<short>"}"""


def build_route_prompt(question: str, options: list[str], facts: dict) -> str:
    return MARK + json.dumps({"question": question, "options": options, "evidence": facts}, sort_keys=True)


def validate_route(obj, options: list[str]) -> tuple[str | None, list[str]]:
    """-> (decision, errors). The reply must name exactly one of the options offered. Pure function."""
    d = obj.get("decision")
    if not isinstance(d, str):
        return None, ["reply must contain a string 'decision'"]
    if d not in options:
        return None, [f"decision {d!r} is not one of {options}"]
    return d, []


def rule_based_route(user: str) -> str:
    """Deterministic stand-in: the hard-coded ladder this agent followed before it was given a model.
    Coverage always widens while it legally can; the trend check prefers the cheap robust recompute, then the
    Diagnoser, then accepts."""
    ev = json.loads(user.split(MARK, 1)[1].split("\n", 1)[0])
    options = ev["options"]
    order = ("pool", "accept") if ev["question"] == "coverage" else ("recompute", "diagnoser", "accept")
    pick = next(o for o in order if o in options)
    return json.dumps({"decision": pick, "reason": "rule-based preference order"})


def ask_route(llm, question: str, options: list[str], facts: dict, max_attempts: int = 2):
    """LLM proposes, code validates against `options`. -> (decision, trace lines). Falls back to the rule order."""
    fallback = json.loads(rule_based_route(build_route_prompt(question, options, facts)))["decision"]
    if llm is None or len(options) < 2:                      # nothing to decide: do not spend a call
        return fallback, []
    user, feedback, notes = build_route_prompt(question, options, facts), "", []
    for attempt in range(1, max_attempts + 1):
        try:
            obj = extract_json(llm(SYSTEM_PROMPT, user + feedback))
        except Exception as ex:
            notes.append(f"attempt {attempt}: unusable reply ({ex})")
            feedback = f"\n\nYour previous reply was not valid JSON ({ex}). Reply with ONLY the JSON object."
            continue
        decision, errors = validate_route(obj, options)
        if decision:
            why = str(obj.get("reason", ""))[:160]
            tag = "" if decision == fallback else f" (rule-based would have chosen {fallback!r})"
            return decision, [f"critic:   {question}: {getattr(llm, 'name', 'llm')} chose {decision!r}{tag}: {why}"] + notes
        notes.append(f"attempt {attempt}: rejected: {'; '.join(errors)}")
        feedback = f"\n\nYour previous reply was rejected: {'; '.join(errors)}. Fix it and reply with ONLY the JSON object."
    return fallback, [f"critic:   {question}: no usable decision, falling back to {fallback!r}"] + notes


def make_critic(snap: Snapshot, min_listings: int = MIN_LISTINGS, jump_pct: float = JUMP_PCT, llm=None):
    profile = ZipProfile(snap.df)

    def critic(state):
        c, scan, stats = state["criteria"], state["scan"], state["stats"]
        scope, instr = state.get("scope") or Scope(), state.get("stat_instruction") or StatInstruction()
        rescans, recomputes, rounds = state.get("rescans", 0), state.get("recomputes", 0), state.get("diag_rounds", 0)
        b, unresolved, out = scan.buckets, [], {}

        # ---- 1. coverage ------------------------------------------------------------------------------
        thin = b[b.thin]
        if len(thin):
            plan = plan_widening(c, scope, scan, min_listings)          # feasibility only; Pooling picks the moves
            can_pool = plan is not None and rescans < MAX_RESCANS
            short = (min_listings - thin.n).clip(lower=0)
            decision, notes = ask_route(
                llm, "coverage", ["pool", "accept"] if can_pool else ["accept"],
                {"thin_buckets": len(thin), "total_buckets": len(b), "fixable": int(thin.reachable.sum()),
                 "min_listings": min_listings, "median_listings_in_thin_buckets": float(thin.n.median()),
                 "median_shortfall": float(short.median()), "worst_shortfall": int(short.max()),
                 "rescans_used": rescans, "rescans_allowed": MAX_RESCANS,
                 "widening_planned": plan.describe() if plan else None})
            if decision == "pool":
                return {"route": "pooling", "trace": [
                    f"critic: coverage: {len(thin)}/{len(b)} buckets below {min_listings} listings "
                    f"({int(thin.reachable.sum())} fixable) -> Data Issue, route to Pooling (rescan {rescans + 1}/{MAX_RESCANS})"] + notes}
            why = ("accepted by the critic" if can_pool else
                   f"rescan cap {MAX_RESCANS} reached" if plan is not None else "no allowed widening can reach the threshold")
            unresolved.append(f"{len(thin)} bucket(s) remain below {min_listings} listings ({why})")
            trace = [f"critic: coverage closed with {len(thin)} thin bucket(s) left ({why})"] + notes
        else:
            trace = [f"critic: coverage OK (0 thin of {len(b)}; {rescans} rescan(s) used)"]
        if recomputes or rounds:                       # coverage was settled earlier; do not repeat the verdict
            trace = []

        # ---- 2/3. trend, method loop, outlier loop --------------------------------------------------------
        flagged = jump_flags(stats.buckets, jump_pct)
        months = sorted(b.month.unique())
        if flagged.empty:
            if recomputes or rounds:
                trace.append(f"critic: trend: no price jumps > {jump_pct:g}% remain "
                             f"(first seen: {state.get('jumps_initial', 0)}; robust recompute {recomputes}, diagnoser rounds {rounds})")
            else:
                trace.append(f"critic: trend: no price jumps > {jump_pct:g}%")
            return {"route": "render", "unresolved": unresolved, "persistent": [], "trace": trace}

        persistent = [f"{r.zip_code}@{r.month}" for r in flagged.itertuples()]
        judged = {d["listing_id"] for d in state.get("diagnoses", []) if d["type"] == "verdict"}     # never re-send a judged listing
        anomalies, suspects = (evidence(flagged, scan, stats, snap, profile, set(instr.exclude_ids) | judged)
                               if rounds < MAX_DIAG_ROUNDS else ([], []))
        options = (["recompute"] if recomputes < MAX_RECOMPUTES and rounds == 0 else []) \
            + (["diagnoser"] if rounds < MAX_DIAG_ROUNDS and (suspects or rounds == 0) else []) + ["accept"]
        biggest = flagged.reindex(flagged.mom_price_pct.abs().sort_values(ascending=False, kind="stable").index)
        decision, notes = ask_route(
            llm, "trend", options,
            {"jumps": len(flagged), "jump_threshold_pct": jump_pct, "zips_affected": int(flagged.zip_code.nunique()),
             "largest_changes_pct": [round(float(x), 1) for x in biggest.mom_price_pct.head(5)],
             "listings_behind_them": [int(x) for x in biggest.n_used.head(5)],
             "outlier_suspects": len(suspects), "robust_recomputes_used": recomputes, "robust_recomputes_allowed": MAX_RECOMPUTES,
             "diagnoser_rounds_used": rounds, "diagnoser_rounds_allowed": MAX_DIAG_ROUNDS,
             "jumps_before_robust_recompute": state.get("jumps_initial", 0) if recomputes else None})
        trace += notes

        if decision == "recompute":
            zips = sorted(flagged.zip_code.unique())
            trace.append(f"critic: trend: {len(flagged)} price jump(s) > {jump_pct:g}% in {len(zips)} ZIP(s) -> "
                         f"robust recompute (trimmed mean {TRIM:g}) on those ZIP series (recompute {recomputes + 1}/{MAX_RECOMPUTES})")
            return {"route": "recompute", "recomputes": recomputes + 1, "jumps_initial": len(flagged),
                    "stat_instruction": robust_instruction(instr, zips, months), "trace": trace}

        if recomputes and rounds == 0:
            first = state.get("jumps_initial", 0)
            trace.append(f"critic: trend: robust recompute resolved {max(first - len(flagged), 0)} of {first} jump(s); {len(flagged)} persist")
        if decision == "diagnoser":
            trace.append(f"critic: trend: {len(flagged)} persistent anomaly(ies) -> Diagnoser round {rounds + 1}/{MAX_DIAG_ROUNDS} "
                         f"({len(anomalies)} sent, {len(suspects)} outlier suspect(s))")
            return {"route": "diagnoser", "diag_rounds": rounds + 1, "anomalies": anomalies, "suspects": suspects,
                    "persistent": persistent, "unresolved": unresolved, "trace": trace}
        unresolved.append(f"{len(flagged)} price jump(s) > {jump_pct:g}% unresolved after {rounds} Diagnoser round(s): " + ", ".join(persistent))
        why = ("accepted by the critic" if "diagnoser" in options or "recompute" in options else
               "round cap reached" if rounds >= MAX_DIAG_ROUNDS else "no new outlier suspects to investigate")
        trace.append(f"critic: trend: {len(flagged)} jump(s) persist after {rounds} Diagnoser round(s) ({why}) -> accept and flag as unresolved")
        return {"route": "render", "unresolved": unresolved, "persistent": persistent, "trace": trace}
    return critic