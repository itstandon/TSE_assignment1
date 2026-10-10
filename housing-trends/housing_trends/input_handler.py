"""Input Handler entry point (dual mode). Form -> pure code. Free text -> Parser Agent -> same validator."""
from __future__ import annotations

from .context import SnapshotContext
from .criteria import InputResult, validate_and_structure
from .parser_agent import parse_free_text


def handle_input(*, form: dict | None = None, text: str | None = None,
                 ctx: SnapshotContext, llm=None) -> InputResult:
    if (form is None) == (text is None):
        raise ValueError("provide exactly one of `form` or `text`")
    if form is not None:                       # deterministic path: no agent
        res = validate_and_structure(form, ctx)
        res.mode = "form"
        res.trace.append("input handler: form validated by code (no LLM)")
        if res.ok:
            res.trace.append(f"criteria {res.criteria.key()}")
        return res
    if llm is None:
        raise ValueError("free-text mode needs an llm callable, e.g. llm_from_env()")
    return parse_free_text(text, ctx, llm)