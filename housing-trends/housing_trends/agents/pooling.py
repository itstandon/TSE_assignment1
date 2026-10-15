"""Pooling Agent: decide HOW to widen the scope for buckets that are still thin.

Invoked only by the Critic's coverage check (a "Data Issue"). It reads the Scanner's per-bucket flags and
returns a revised Scope for the Scanner to re-run (Cycle A).

The three moves, per thin-but-fixable bucket (thin and `reachable`):
  time     pad +-TIME_STEP months around the bucket
  filters  relax the first still-relaxable (set, NON-protected) filter   [GLOBAL: it applies to the whole scan]
  zip      pool ZIP_STEP more neighbouring ZIPs into the bucket

WHICH move to spend on a bucket is a judgement about which compromise least distorts the user's question, and that
judgement is the LLM's (requirement c: the Scanner's flags change Pooling's reasoning, not merely its arithmetic):

  * code computes, for every fixable bucket, which moves are LEGAL (cap not reached, and the move would actually add
    a listing) and whether padding months alone would already clear the threshold;
  * buckets are grouped into CLASSES by their legal-move signature, so one small prompt covers a 500-bucket scan;
  * the LLM picks one move per class, with a reason. Code validates every pick against that class's legal set and
    silently falls back to the rule-based preference for anything missing, unknown or illegal.
  * `rule_choice` is that preference order ("least change to the user's question" - time if it suffices, else relax a
    filter, else borrow ZIPs, else pad anyway). It is the deterministic stand-in, so `--llm offline` reproduces
    exactly the behaviour this agent had before it was given a model.

Buckets that are healthy, or that no allowed move can improve, are left alone. Returns None when no move helps anywhere.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from ..criteria import Criteria
from ..state import (MAX_TIME_PAD, MAX_ZIP_EXTRA, MIN_LISTINGS, TIME_STEP, ZIP_STEP, BucketScope, Scope)
from .scanner import ScanResult, relaxable_filters

MOVES = ("time", "filters", "zip")
MARK = "POOLING_JSON:\n"

SYSTEM_PROMPT = """You are the pooling strategist of a housing price-trend system.
Some ZIP-month buckets hold too few listings to compute a trustworthy median, so the scope must be widened. Three
moves are available, and each one distorts the user's question differently:
- "time": also count listings from the neighbouring months of the same ZIP. Distorts seasonality and blurs the
  month-over-month trend, but keeps the geography the user asked about.
- "filters": stop applying one of the user's non-essential filters (never price, beds or baths). Distorts WHAT is
  being counted, and it applies to the whole scan, not just one bucket.
- "zip": also count listings from the nearest other ZIP codes. Distorts the geography: a neighbouring ZIP can be a
  very different market.
POOLING_JSON describes classes of buckets. Each class lists the moves that are legal for it ("legal"), how many
buckets are in it, their median listing count, how many more listings each move would add, and whether padding
months alone would already reach the threshold ("time_suffices").
For EVERY class choose exactly one move from its own "legal" list, preferring the move that least distorts the
user's question while actually fixing the bucket.
Reply with ONE JSON object and nothing else:
{"choices":[{"class":"<class id>","move":"time|filters|zip","reason":"<short>"}]}"""


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


# ---------------------------------------------------------------- what the agent may choose from
def bucket_options(c: Criteria, scope: Scope, scan: ScanResult, min_listings: int = MIN_LISTINGS) -> list[dict]:
    """One entry per thin-but-fixable bucket: the legal moves and what each would buy. Pure function of its inputs."""
    thin = scan.buckets[scan.buckets.thin]
    rel = relaxable_filters(c, scope)
    out = []
    for r in thin[thin.reachable].itertuples():
        ze, tp = scope.depth(r.zip_code, r.month)
        legal = []
        if tp < MAX_TIME_PAD and r.gain_time > 0:
            legal.append("time")
        if rel and r.gain_filters > 0:
            legal.append("filters")
        if ze < MAX_ZIP_EXTRA and r.gain_zip > 0:
            legal.append("zip")
        out.append(dict(zip_code=r.zip_code, month=r.month, n=int(r.n), zip_extra=ze, time_pad=tp,
                        gain_time=int(r.gain_time), gain_zip=int(r.gain_zip), gain_filters=int(r.gain_filters),
                        time_suffices=bool("time" in legal and r.n + r.gain_time >= min_listings), legal=legal))
    return out


def class_of(o: dict) -> str:
    """Buckets facing the same decision share a class, so one prompt covers a scan of any size."""
    return "+".join(o["legal"]) + ("|time_suffices" if o["time_suffices"] else "") if o["legal"] else "none"


def rule_choice(o: dict) -> str | None:
    """The deterministic preference order: least change to the user's question that still fixes the bucket."""
    if o["time_suffices"]:
        return "time"
    for m in ("filters", "zip", "time"):
        if m in o["legal"]:
            return m
    return None


# ---------------------------------------------------------------- turning choices into a Scope
def plan_from_choices(c: Criteria, scope: Scope, scan: ScanResult, options: list[dict], choices: dict) -> Plan | None:
    """`choices` maps a bucket's (zip, month) to one of MOVES (or None). Illegal picks are ignored by construction."""
    thin = int(scan.buckets.thin.sum())
    if not options:
        return None
    rel = relaxable_filters(c, scope)
    new, n_time, n_zip, relax = {}, 0, 0, False
    for o in options:
        move = choices.get((o["zip_code"], o["month"]))
        if move == "time":
            new[(o["zip_code"], o["month"])] = (o["zip_extra"], min(MAX_TIME_PAD, o["time_pad"] + TIME_STEP))
            n_time += 1
        elif move == "zip":
            new[(o["zip_code"], o["month"])] = (min(MAX_ZIP_EXTRA, o["zip_extra"] + ZIP_STEP), o["time_pad"])
            n_zip += 1
        elif move == "filters":
            relax = True
    if not new and not relax:
        return None
    keep = {(x.zip_code, x.month): (x.zip_extra, x.time_pad) for x in scope.per_bucket}
    keep.update(new)
    per = tuple(BucketScope(zip_code=z, month=m, zip_extra=e, time_pad=t) for (z, m), (e, t) in keep.items())
    relaxed = scope.relaxed + ((rel[0],) if relax and rel else ())
    return Plan(Scope(zip_extra=scope.zip_extra, time_pad=scope.time_pad, relaxed=relaxed, per_bucket=per),
                {"time": n_time, "zip": n_zip, "filters": [rel[0]] if relax and rel else []}, thin, len(options))


def plan_widening(c: Criteria, scope: Scope, scan: ScanResult, min_listings: int = MIN_LISTINGS) -> Plan | None:
    """Rule-based planning (no model): the preference order applied to every fixable bucket."""
    options = bucket_options(c, scope, scan, min_listings)
    return plan_from_choices(c, scope, scan, options, {(o["zip_code"], o["month"]): rule_choice(o) for o in options})


# ---------------------------------------------------------------- the LLM path
def build_user_prompt(options: list[dict], c: Criteria, min_listings: int) -> str:
    """One line per class of buckets: what they look like and which moves are legal for them."""
    classes = {}
    for o in options:
        classes.setdefault(class_of(o), []).append(o)
    body = []
    for cid, group in sorted(classes.items()):
        ns = sorted(g["n"] for g in group)
        body.append({"class": cid, "buckets": len(group), "median_listings": ns[len(ns) // 2],
                     "needed": min_listings, "legal": group[0]["legal"], "time_suffices": group[0]["time_suffices"],
                     "would_add": {m: sorted(g[f"gain_{m}"] for g in group)[len(group) // 2] for m in MOVES},
                     "example": f"{group[0]['zip_code']}@{group[0]['month']}"})
    return MARK + json.dumps({"min_listings": min_listings, "relaxable_filters": list(relaxable_filters(c, Scope())),
                              "classes": body}, sort_keys=True)


def validate_reply(obj, options: list[dict]) -> tuple[dict, list[str]]:
    """-> ({class id: move}, errors). A pick must name a known class and a move that class is allowed to make."""
    legal_by_class = {class_of(o): o["legal"] for o in options}
    picks, errors = {}, []
    if not isinstance(obj.get("choices"), list):
        return {}, ["reply must contain a list 'choices'"]
    for ch in obj["choices"]:
        try:
            cid, move = str(ch["class"]), str(ch["move"])
        except (KeyError, TypeError):
            errors.append(f"malformed choice entry: {ch!r}")
            continue
        if cid not in legal_by_class:
            errors.append(f"unknown bucket class {cid!r}")
        elif move not in legal_by_class[cid]:
            errors.append(f"move {move!r} is not legal for class {cid!r} (legal: {legal_by_class[cid]})")
        else:
            picks[cid] = move
    missing = [k for k in legal_by_class if k not in picks]
    if missing:
        errors.append(f"no move chosen for class(es) {missing}")
    return picks, errors


def rule_based_moves(user: str) -> str:
    """Deterministic stand-in with the same prompt -> JSON contract: reproduces the hard-coded preference order."""
    ev = json.loads(user.split(MARK, 1)[1].split("\n", 1)[0])
    out = []
    for cl in ev["classes"]:
        o = {"legal": cl["legal"], "time_suffices": cl["time_suffices"]}
        move = rule_choice(o)
        if move:
            out.append({"class": cl["class"], "move": move, "reason": "least change to the user's question"})
    return json.dumps({"choices": out})


def llm_plan_widening(llm, c: Criteria, scope: Scope, scan: ScanResult, min_listings: int = MIN_LISTINGS,
                      max_attempts: int = 2) -> tuple[Plan | None, list[str]]:
    """LLM proposes one move per bucket class, code validates and applies. -> (Plan, trace lines)."""
    from ..parser_agent import _extract_json as extract_json
    options = bucket_options(c, scope, scan, min_listings)
    if not options:
        return None, []
    rule = {(o["zip_code"], o["month"]): rule_choice(o) for o in options}
    if llm is None:
        return plan_from_choices(c, scope, scan, options, rule), []

    user, feedback, picks, notes = build_user_prompt(options, c, min_listings), "", {}, []
    for attempt in range(1, max_attempts + 1):
        try:
            obj = extract_json(llm(SYSTEM_PROMPT, user + feedback))
        except Exception as ex:
            notes.append(f"attempt {attempt}: unusable reply ({ex})")
            feedback = f"\n\nYour previous reply was not valid JSON ({ex}). Reply with ONLY the JSON object."
            continue
        picks, errors = validate_reply(obj, options)
        if not errors:
            break
        notes.append(f"attempt {attempt}: rejected: {'; '.join(errors[:3])}")
        feedback = f"\n\nYour previous reply was rejected: {'; '.join(errors[:3])}. Fix it and reply with ONLY the JSON object."

    choices, overridden = {}, 0
    for o in options:
        key = (o["zip_code"], o["month"])
        move = picks.get(class_of(o))
        choices[key] = move or rule[key]
        overridden += bool(move and move != rule[key])
    source = getattr(llm, "name", "llm") if picks else "fallback (rule-based)"
    trace = [f"pooling:   strategy by {source}: {len(picks)}/{len(set(class_of(o) for o in options))} class(es) chosen, "
             f"{overridden} bucket(s) differ from the rule-based preference"] + [f"pooling:   {n}" for n in notes]
    return plan_from_choices(c, scope, scan, options, choices), trace


def make_pooling(min_listings: int = MIN_LISTINGS, llm=None):
    """LangGraph node factory: reads scan + scope, writes a wider scope and bumps the rescan counter."""
    def pooling(state):
        plan, extra = llm_plan_widening(llm, state["criteria"], state.get("scope") or Scope(), state["scan"], min_listings)
        n = state.get("rescans", 0) + 1
        if plan is None:                                    # the Critic only routes here when a plan exists
            return {"rescans": n, "trace": [f"pooling: rescan {n}: no move can help; scope unchanged"] + extra}
        return {"scope": plan.scope, "rescans": n, "trace": [f"pooling: rescan {n}: {plan.describe()}"] + extra}
    return pooling
