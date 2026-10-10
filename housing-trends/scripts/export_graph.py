"""Write the agent graph as a Mermaid diagram (docs/architecture.mmd). Paste into https://mermaid.live to render a PNG."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd
from housing_trends.agents.diagnoser import rule_based_llm
from housing_trends.agents.scanner import Snapshot
from housing_trends.graph import build_graph

snap = Snapshot(pd.DataFrame([dict(listing_id=1, zip_code="75201", month="2025-01", price=1)]))
mer = build_graph(snap, rule_based_llm).get_graph().draw_mermaid()
mer += """
%% Cycles (all other edges are forward flow):
%%   A coverage : critic -> pooling -> scanner -> statistician -> critic   (cap 3 rescans)
%%   B method   : critic -> statistician -> critic                          (cap 1 robust recompute)
%%   C outlier  : critic -> diagnoser -> statistician -> critic             (cap 2 rounds)
"""
out = ROOT / "docs/architecture.mmd"
out.parent.mkdir(exist_ok=True)
out.write_text(mer)
print(mer)