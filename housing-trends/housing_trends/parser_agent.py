"""Parser Agent: free text -> raw criteria dict -> the SAME deterministic validator as the form.

The LLM's output is untrusted. Hallucinated ZIPs, inverted ranges, wrong-length windows are all
caught by validate_and_structure(); the agent gets the error list back and one chance to repair.
If it still fails we return errors (and point the user at the form) rather than guess.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import replace

from .context import SnapshotContext, add_months
from .criteria import InputResult, WINDOW_MONTHS, validate_and_structure

SYSTEM_PROMPT = """You convert a home-buyer's free-text request into a JSON search specification.
Reply with ONE JSON object and nothing else. Rules:
- Use ONLY ZIP codes from the provided table. Resolve place names via the city table or area table. Never invent ZIPs.
- Map vague words using the provided numbers only: 'affordable'/'cheap' -> price_max = p50; 'luxury'/'high-end' -> price_min = p75.
- Anything you cannot map to a field (schools, commute, crime, style...) goes in "unsupported". Do NOT guess.
- Record every interpretation you made in "assumptions" (short strings).
- Months are YYYY-MM. 'past year' / 'last 12 months' -> end_month = the snapshot's last month; leave start_month null.
Fields: zips (list of 5-digit strings), price_min, price_max, beds_min, beds_max, baths_min, sqft_min, sqft_max,
property_types (list), start_month, end_month, assumptions (list), unsupported (list). Use null for unspecified."""


def _zip_table(ctx: SnapshotContext) -> str:
    by_city = defaultdict(list)
    for z, c in sorted(ctx.zip_to_city.items()):
        by_city[c].append(z)
    return "\n".join(f"{c}: {', '.join(zs)}" for c, zs in sorted(by_city.items()))


def build_user_prompt(text: str, ctx: SnapshotContext) -> str:
    areas = "\n".join(f"{a}: {', '.join(z)}" for a, z in sorted(ctx.area_gazetteer.items())) or "(none provided)"
    return (f"Snapshot covers {ctx.first_month}..{ctx.last_month}; the window must be exactly {WINDOW_MONTHS} months.\n"
            f"Price quantiles: {json.dumps(ctx.price_quantiles)}\n"
            f"Valid property types: {sorted(ctx.property_types) or 'none (do not output property_types)'}\n\n"
            f"CITY -> ZIPs:\n{_zip_table(ctx)}\n\nNAMED AREAS -> ZIPs:\n{areas}\n\n"
            f"USER REQUEST: {text.strip()}")


def _extract_json(reply: str) -> dict:
    t = re.sub(r"^```(?:json)?|```$", "", reply.strip(), flags=re.M).strip()
    a, b = t.find("{"), t.rfind("}")
    if a == -1 or b <= a:
        raise ValueError("no JSON object found")
    obj = json.loads(t[a:b + 1])
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON is not an object")
    return obj


def _strs(v) -> list:
    return [str(x) for x in v] if isinstance(v, list) else ([str(v)] if v else [])


def parse_free_text(text: str, ctx: SnapshotContext, llm, max_attempts: int = 2) -> InputResult:
    if not text or not text.strip():
        return InputResult(errors=["empty request"], mode="free_text")
    trace, feedback = [], ""
    for attempt in range(1, max_attempts + 1):
        try:
            raw = _extract_json(llm(SYSTEM_PROMPT, build_user_prompt(text, ctx) + feedback))
        except Exception as ex:                                   # bad JSON / transport error
            trace.append(f"parser attempt {attempt}: unusable output ({ex})")
            feedback = f"\n\nYour previous reply was not valid JSON ({ex}). Reply with ONLY the JSON object."
            continue
        assumptions, unsupported = _strs(raw.pop("assumptions", None)), _strs(raw.pop("unsupported", None))
        res = validate_and_structure(raw, ctx)
        if res.ok:
            trace.append(f"parser attempt {attempt}: accepted (criteria {res.criteria.key()})")
            return replace(res, mode="free_text", assumptions=assumptions, unsupported=unsupported,
                           attempts=attempt, trace=trace)
        trace.append(f"parser attempt {attempt}: rejected by validator: {'; '.join(res.errors)}")
        feedback = f"\n\nYour previous JSON was rejected: {'; '.join(res.errors)}. Fix and reply with ONLY the JSON object."
    return InputResult(errors=["could not turn the request into valid criteria; please use the form instead"] ,
                       mode="free_text", attempts=max_attempts, trace=trace,
                       warnings=[t for t in trace if "rejected" in t or "unusable" in t])