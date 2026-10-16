"""Compare two harness runs trial by trial: does the model's judgement change the metamorphic verdicts?

  python scripts/compare_runs.py results/mr results/mr_llm

Both directories must be outputs of scripts/run_mrs.py. Trials are matched on (MR, base input, seed), so only the
MRs and base inputs that both runs covered are compared. Reports, per MR: how many trials changed verdict, and how
many ran a different number of loops (a different number of rescans / robust recomputes / diagnoser rounds is the
visible trace of the Critic or Pooling having decided differently).
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("a", help="baseline run directory (e.g. results/mr)")
ap.add_argument("b", help="comparison run directory (e.g. results/mr_llm)")
args = ap.parse_args()

KEY = ["mr", "base_id", "seed"]
COLS = KEY + ["status", "src_loops", "fol_loops", "detail"]
a = pd.read_csv(pathlib.Path(args.a) / "runs.csv")[COLS].fillna({"src_loops": "-", "fol_loops": "-"})   # skipped trials have no loops: "-" == "-"
b = pd.read_csv(pathlib.Path(args.b) / "runs.csv")[COLS].fillna({"src_loops": "-", "fol_loops": "-"})
m = a.merge(b, on=KEY, suffixes=("_a", "_b"))
if m.empty:
    sys.exit("the two runs share no (MR, base input, seed) trial")

print(f"{args.a} vs {args.b}: {len(m)} trial(s) in common, {m.mr.nunique()} MR(s)\n")
rows = []
for mr, g in m.groupby("mr"):
    verdict = (g.status_a != g.status_b).sum()
    loops = ((g.src_loops_a != g.src_loops_b) | (g.fol_loops_a != g.fol_loops_b)).sum()
    va, vb = (g.status_a == "violated").sum(), (g.status_b == "violated").sum()
    rows.append(dict(mr=mr, trials=len(g), violated_a=va, violated_b=vb,
                     verdict_changed=verdict, loops_changed=loops))
summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
print(f"\ntotals: {summary.verdict_changed.sum()} verdict change(s), {summary.loops_changed.sum()} loop-count change(s)"
      f" out of {len(m)} trial(s)")

changed = m[(m.status_a != m.status_b) | (m.src_loops_a != m.src_loops_b) | (m.fol_loops_a != m.fol_loops_b)]
if not changed.empty:
    print("\nchanged trials (loops are rescans/recomputes/diagnoser-rounds, source -> follow-up):")
    for r in changed.head(25).itertuples():
        print(f"  {r.mr} base {r.base_id} seed {r.seed}: {r.status_a} -> {r.status_b} | "
              f"loops {r.src_loops_a}/{r.fol_loops_a} -> {r.src_loops_b}/{r.fol_loops_b}")
        if r.status_a != r.status_b:
            print(f"      {args.b}: {str(r.detail_b)[:160]}")
else:
    print("\nno trial changed: the model reached the same decision as the rule-based policy everywhere")
