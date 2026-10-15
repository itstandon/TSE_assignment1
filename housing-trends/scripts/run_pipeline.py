"""Run the full agent graph on the frozen snapshot and save the execution trace.

  python scripts/run_pipeline.py --all --offline                 # whole city, deterministic stand-in for the Diagnoser
  python scripts/run_pipeline.py --zips 75241,75219              # real LLM (needs LLM2_API_KEY in .env.local), replies cached
  python scripts/run_pipeline.py --all --price-max 400000 --beds-min 3 --name filtered

The trace file records which model answered, so a report can say exactly what produced it.
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from housing_trends import SnapshotContext, handle_input
from housing_trends.agents.diagnoser import rule_based_llm
from housing_trends.agents.scanner import Snapshot
from housing_trends.graph import run
from housing_trends.llm import CachedLLM, llm_from_env

ap = argparse.ArgumentParser()
ap.add_argument("--snapshot", default=str(ROOT / "data/snapshot/listings_dfw.csv"))
ap.add_argument("--zips", help="comma-separated ZIPs")
ap.add_argument("--all", action="store_true", help="every ZIP in the snapshot")
ap.add_argument("--end-month", default=None)
ap.add_argument("--price-min", type=int); ap.add_argument("--price-max", type=int)
ap.add_argument("--beds-min", type=int); ap.add_argument("--sqft-min", type=int)
ap.add_argument("--offline", action="store_true", help="use the rule-based stand-in instead of the LLM (label results accordingly)")
ap.add_argument("--name", default="run")
a = ap.parse_args()

df = pd.read_csv(a.snapshot, dtype={"zip_code": str})
ctx = SnapshotContext.from_dataframe(df)
snap = Snapshot.frozen(df)
zips = sorted(ctx.valid_zips) if a.all or not a.zips else a.zips.split(",")
form = dict(zips=zips, end_month=a.end_month, price_min=a.price_min, price_max=a.price_max, beds_min=a.beds_min, sqft_min=a.sqft_min)
inp = handle_input(ctx=ctx, form={k: v for k, v in form.items() if v is not None})
if not inp.ok:
    sys.exit("invalid input: " + "; ".join(inp.errors))

if a.offline:
    llm = rule_based_llm
else:
    base = llm_from_env()
    llm = CachedLLM(base, ROOT / "results/llm_cache", tag=base.name)

st = run(inp.criteria, snap, llm)
rep = st["report"]
manifest = ROOT / "data/snapshot/snapshot_manifest.json"
sha = json.loads(manifest.read_text())["output_sha256"]["listings_dfw.csv"][:12] if manifest.exists() else "unknown"
header = [f"# execution trace: {a.name}", f"# snapshot sha256 {sha} | diagnoser model: {getattr(llm, 'name', 'llm')}",
          f"# loops used: {rep.counters}", ""]
text = "\n".join(header + rep.trace)
out = ROOT / "results/traces" / f"{a.name}.txt"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(text + "\n")
print(text)
print(f"\nsaved to {out}")
print("\nOVERALL TREND (all requested ZIPs, own listings):")
print(rep.overall[["month", "n", "price", "ppsf", "mom_price_pct"]].round(1).to_string(index=False))
if rep.unresolved:
    print("\nUNRESOLVED:"); [print(" -", u) for u in rep.unresolved]