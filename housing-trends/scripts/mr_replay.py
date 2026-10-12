"""Replay one metamorphic trial and show why it passed or failed (root-cause analysis).

  python scripts/mr_replay.py MR3 7 101        # MR id, base input id, seed (as listed in results/mr/runs.csv)

Prints the criteria, the problems the relation found, the source and follow-up contracts side by side, and the loops
that fired in each run. The full decision traces of both runs are saved under results/mr/replays/.
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from mr.harness import DEFAULT_SNAPSHOT, replay
from mr.inputs import POOL_SEED

ap = argparse.ArgumentParser()
ap.add_argument("mr")
ap.add_argument("base_id", type=int)
ap.add_argument("seed", type=int)
ap.add_argument("--llm", choices=["offline", "real"], default="offline")
ap.add_argument("--pool-seed", type=int, default=POOL_SEED)
a = ap.parse_args()

r = replay(a.mr, a.base_id, a.seed, snapshot=DEFAULT_SNAPSHOT, pool_seed=a.pool_seed, llm_mode=a.llm)
c, mr = r["criteria"], r["mr"]
print(f"{mr.id} {mr.name} | base input {a.base_id} | seed {a.seed}\nRelation: {mr.relation}")
print(f"criteria: {len(c.zips)} ZIPs, {c.start_month}..{c.end_month}, price {c.price_min}-{c.price_max}, beds>={c.beds_min}, baths>={c.baths_min}")
if r["follow_criteria"] != c:
    f = r["follow_criteria"]
    print(f"follow-up criteria: {len(f.zips)} ZIPs, {f.start_month}..{f.end_month}, price {f.price_min}-{f.price_max}")
print(f"transform meta: {r['meta']}")
print(f"loops (rescans/recomputes/diagnoser rounds): source {r['source_loops']}, follow-up {r['followup_loops']}")
print("RESULT:", "VIOLATED" if r["problems"] else "holds")
for p in r["problems"]:
    print("  -", p)
both = r["source"].join(r["followup"], lsuffix="_src", rsuffix="_fol", how="outer")
with pd.option_context("display.width", 200, "display.float_format", "{:,.2f}".format):
    print("\n" + both.to_string())

out = ROOT / "results/mr/replays"
out.mkdir(parents=True, exist_ok=True)
for name, rep in (("source", r["source_report"]), ("followup", r["followup_report"])):
    (out / f"{a.mr}_{a.base_id}_{a.seed}_{name}.txt").write_text("\n".join(rep.trace) + "\n")
print(f"\ntraces saved to {out}")
