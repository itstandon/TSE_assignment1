"""LangGraph wiring. Nodes are the plain functions from housing_trends.agents; this file only connects them.

            START -> scanner -> statistician -> critic --(route)--> render -> END
                       ^            ^   ^                |
   Cycle A (coverage)  +-- pooling <-----|----------------+ "pooling"
   Cycle B (method)                      +----------------+ "recompute"   (critic -> statistician)
   Cycle C (outlier)                     +-- diagnoser <--+ "diagnoser"   (critic -> diagnoser -> statistician)
Loops are bounded by counters in the state (MAX_RESCANS, MAX_RECOMPUTES, MAX_DIAG_ROUNDS).
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .agents.critic import make_critic
from .agents.diagnoser import make_diagnoser
from .agents.pooling import make_pooling
from .agents.renderer import make_renderer
from .agents.scanner import Snapshot, make_scanner
from .agents.statistician import StatInstruction, make_statistician
from .state import JUMP_PCT, MIN_LISTINGS, PipelineState, Scope


def build_graph(snap: Snapshot, llm, min_listings: int = MIN_LISTINGS, jump_pct: float = JUMP_PCT):
    g = StateGraph(PipelineState)
    g.add_node("scanner", make_scanner(snap, min_listings))
    g.add_node("statistician", make_statistician(snap))
    g.add_node("critic", make_critic(snap, min_listings, jump_pct))
    g.add_node("pooling", make_pooling(min_listings))
    g.add_node("diagnoser", make_diagnoser(llm))
    g.add_node("render", make_renderer())
    g.add_edge(START, "scanner")
    g.add_edge("scanner", "statistician")
    g.add_edge("statistician", "critic")
    g.add_conditional_edges("critic", lambda s: s["route"],
                            {"pooling": "pooling", "recompute": "statistician", "diagnoser": "diagnoser", "render": "render"})
    g.add_edge("pooling", "scanner")            # Cycle A
    g.add_edge("diagnoser", "statistician")     # Cycle C
    g.add_edge("render", END)
    return g.compile()


def run(criteria, snap: Snapshot, llm, *, min_listings: int = MIN_LISTINGS, jump_pct: float = JUMP_PCT, app=None):
    """Run the whole pipeline for one Criteria; returns the final state (`report` holds the result)."""
    app = app or build_graph(snap, llm, min_listings, jump_pct)
    init = {"criteria": criteria, "scope": Scope(), "stat_instruction": StatInstruction(), "rescans": 0, "recomputes": 0,
            "diag_rounds": 0, "diagnoses": [], "unresolved": [], "persistent": [],
            "trace": [f"run: criteria {criteria.key()} | {len(criteria.zips)} ZIP(s) | {criteria.start_month}..{criteria.end_month} "
                      f"| thresholds: min_listings={min_listings}, jump={jump_pct:g}%"]}
    return app.invoke(init, config={"recursion_limit": 100})