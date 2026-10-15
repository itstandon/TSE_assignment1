"""Anomaly Diagnoser (the LLM node): judge outlier listings behind persistent price jumps and explain the jumps.

The code gathers the evidence (the Critic's screen + ZIP percentiles); the LLM judges it; code validates the reply.
Its verdicts CHANGE WHAT THE STATISTICIAN DOES (requirement c), not just its data:
  bad_listing  -> the listing is EXCLUDED and the plain median recomputed
  legit_rural  -> the listing is KEPT, and the ZIP series is computed with the robust trimmed mean
  unexplained  -> excluded and flagged as an unresolved data-quality issue (design note: cap, then exclude + flag)
Replies are untrusted: bad JSON, unknown listing ids, unknown verdicts are rejected, the model gets the errors back
once (2 attempts), and if it still fails every suspect becomes `unexplained` (we never guess).
"""
from __future__ import annotations

import json

from .critic import robust_instruction
from ..parser_agent import _extract_json as extract_json
from .statistician import StatInstruction

VERDICTS = ("bad_listing", "legit_rural", "unexplained")

SYSTEM_PROMPT = """You are a data-quality analyst for a housing price-trend system.
A month-over-month jump in a ZIP's price statistic survived a robust recompute. EVIDENCE_JSON contains
"anomalies" (the jumps, with sample sizes and composition) and "suspects" (listings whose price is extreme for their ZIP,
with percentile ranks 0-100 of their lot size, house size and price among that ZIP's listings).
For EVERY suspect give exactly one verdict:
- "bad_listing": the price looks like a data-entry error (for example about 10x or 1/10 of the ZIP's typical price)
  while the lot and house size are ordinary for the ZIP.
- "legit_rural": the low price is plausible because the lot is very large and/or the house very small for the ZIP
  (high acre_lot_pct, low house_size_pct).
- "unexplained": the evidence supports neither.
For EVERY anomaly give a one-sentence explanation using only numbers present in the evidence. Do not invent facts.
Reply with ONE JSON object and nothing else:
{"verdicts":[{"listing_id":<int>,"verdict":"bad_listing|legit_rural|unexplained","reason":"<short>"}],
 "explanations":[{"zip_code":"<zip>","month":"YYYY-MM","text":"<one sentence>"}]}"""

MARK = "EVIDENCE_JSON:\n"


def build_user_prompt(anomalies, suspects) -> str:
    return MARK + json.dumps({"anomalies": anomalies, "suspects": suspects}, sort_keys=True)


def validate_reply(obj, anomalies, suspects):
    """-> (verdicts {id: (verdict, reason)}, explanations {(zip, month): text}, errors [str]). Pure function."""
    errors, verdicts, expl = [], {}, {}
    ids = {s["listing_id"] for s in suspects}
    keys = {(a["zip_code"], a["month"]) for a in anomalies}
    if not isinstance(obj.get("verdicts"), list) or not isinstance(obj.get("explanations"), list):
        return {}, {}, ["reply must contain lists 'verdicts' and 'explanations'"]
    for v in obj["verdicts"]:
        try:
            lid, ver = int(v["listing_id"]), v["verdict"]
        except (KeyError, TypeError, ValueError):
            errors.append(f"malformed verdict entry: {v!r}")
            continue
        if lid not in ids:
            errors.append(f"verdict for unknown listing_id {lid}")
        elif ver not in VERDICTS:
            errors.append(f"unknown verdict {ver!r} for listing {lid}")
        else:
            verdicts[lid] = (ver, str(v.get("reason", ""))[:200])
    for e in obj["explanations"]:
        try:
            k = (str(e["zip_code"]), str(e["month"]))
        except (KeyError, TypeError):
            errors.append(f"malformed explanation entry: {e!r}")
            continue
        if k not in keys:
            errors.append(f"explanation for unknown anomaly {k}")
        else:
            expl[k] = str(e.get("text", ""))[:300]
    return verdicts, expl, errors


def rule_based_llm(system: str, user: str) -> str:
    """Deterministic stand-in for EVERY LLM node (Diagnoser, Critic, Pooling), dispatched on the prompt's marker.

    Each node's stand-in reproduces the hard-coded policy that node had before it was given a model, so `--llm offline`
    is both reproducible without an API key and a true baseline: any difference between an offline run and a real-model
    run is the model's judgement, not a change of plumbing. Results produced with it must be labelled as such.
    """
    from .critic import MARK as CRITIC_MARK, rule_based_route
    from .pooling import MARK as POOLING_MARK, rule_based_moves
    if CRITIC_MARK in user:
        return rule_based_route(user)
    if POOLING_MARK in user:
        return rule_based_moves(user)
    ev = json.loads(user.split(MARK, 1)[1].split("\n", 1)[0])
    out = {"verdicts": [], "explanations": []}
    for s in ev["suspects"]:
        ratio, lot, size = s["price_vs_zip_median"], s["acre_lot_pct"], s["house_size_pct"]
        if ratio >= 5 or ratio <= 0.2:
            v, why = "bad_listing", f"price is {ratio:g}x the ZIP median"
        elif lot is not None and size is not None and lot >= 90 and size <= 35 and ratio < 1:
            v, why = "legit_rural", f"lot at {lot:g}th and house at {size:g}th percentile of the ZIP"
        else:
            v, why = "unexplained", "no rule applies"
        out["verdicts"].append({"listing_id": s["listing_id"], "verdict": v, "reason": why})
    for a in ev["anomalies"]:
        k = sum(1 for s in ev["suspects"] if s["zip_code"] == a["zip_code"])
        out["explanations"].append({"zip_code": a["zip_code"], "month": a["month"], "text":
            f"Median price moved {a['mom_price_pct']:+.0f}% from {a['prev_month']} ({a.get('prev', {}).get('n_used', '?')} listings) "
            f"to {a['month']} ({a.get('now', {}).get('n_used', '?')} listings); {k} outlier listing(s) in this ZIP."})
    return json.dumps(out)


rule_based_llm.name = "rule_based_standin"


def make_diagnoser(llm, max_attempts: int = 2):
    label = getattr(llm, "name", "llm")                                    # which model actually answered (for the log)

    def diagnoser(state):
        anomalies, suspects = state["anomalies"], state["suspects"]
        rnd, instr = state.get("diag_rounds", 1), state.get("stat_instruction") or StatInstruction()
        user, feedback, result, notes = build_user_prompt(anomalies, suspects), "", None, []
        for attempt in range(1, max_attempts + 1):
            try:
                obj = extract_json(llm(SYSTEM_PROMPT, user + feedback))
            except Exception as ex:                                       # bad JSON or transport error
                notes.append(f"attempt {attempt}: unusable reply ({ex})")
                feedback = f"\n\nYour previous reply was not valid JSON ({ex}). Reply with ONLY the JSON object."
                continue
            verdicts, expl, errors = validate_reply(obj, anomalies, suspects)
            if not errors:
                result = (verdicts, expl, attempt)
                break
            notes.append(f"attempt {attempt}: rejected: {'; '.join(errors)}")
            feedback = f"\n\nYour previous reply was rejected: {'; '.join(errors)}. Fix it and reply with ONLY the JSON object."
        source = label
        if result is None:
            verdicts, expl, source = {}, {}, "fallback"
            result = (verdicts, expl, max_attempts)
        verdicts, expl, attempts = result

        exclude, keep_zips, records, trace = set(instr.exclude_ids), set(), [], []
        for s in suspects:
            ver, why = verdicts.get(s["listing_id"], ("unexplained", "no valid verdict from the diagnoser"))
            if ver in ("bad_listing", "unexplained"):
                exclude.add(s["listing_id"])
            else:
                keep_zips.add(s["zip_code"])
            records.append({"round": rnd, "type": "verdict", "listing_id": s["listing_id"], "zip_code": s["zip_code"],
                            "month": s["month"], "price": s["price"], "verdict": ver, "reason": why, "source": source})
            trace.append(f"diagnoser:   listing {s['listing_id']} ({s['zip_code']}, {s['month']}, ${s['price']:,.0f}) -> {ver}: {why}")
        for a in anomalies:
            records.append({"round": rnd, "type": "explanation", "zip_code": a["zip_code"], "month": a["month"], "source": source,
                            "text": expl.get((a["zip_code"], a["month"]), "no explanation provided")})
        # Verdicts decide the computation path per ZIP (design note, requirement c):
        #   any legit_rural listing in the ZIP -> keep it and compute the ZIP series with the robust trimmed mean
        #   only bad/unexplained listings     -> exclude them and go back to the PLAIN MEDIAN for that ZIP
        months = sorted(state["scan"].buckets.month.unique())
        excluded_zips = {s["zip_code"] for s in suspects if s["listing_id"] in exclude} - keep_zips
        kept = tuple(o for o in instr.overrides if o.zip_code not in excluded_zips)
        new = StatInstruction(method=instr.method, overrides=robust_instruction(
            StatInstruction(method=instr.method, overrides=kept), keep_zips, months).overrides, exclude_ids=tuple(exclude))
        counts = {v: sum(1 for r in records if r["type"] == "verdict" and r["verdict"] == v) for v in VERDICTS}
        head = (f"diagnoser: round {rnd}: {len(suspects)} suspect(s) judged by {source} in {attempts} attempt(s): "
                + ", ".join(f"{k}={v}" for k, v in counts.items()) + f" -> statistician excludes {len(new.exclude_ids) - len(instr.exclude_ids)} listing(s)")
        return {"stat_instruction": new, "diagnoses": state.get("diagnoses", []) + records,
                "trace": [head] + [f"diagnoser:   {n}" for n in notes] + trace}
    return diagnoser