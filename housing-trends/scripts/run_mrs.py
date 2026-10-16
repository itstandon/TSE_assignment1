"""Run the metamorphic harness and write the results table.

  python scripts/run_mrs.py                       # all 10 MRs, 30 base inputs, seeds 101,202,303, offline Diagnoser
  python scripts/run_mrs.py --quick               # smoke run: 5 base inputs, 1 seed
  python scripts/run_mrs.py --only MR1,MR3        # a subset
  python scripts/run_mrs.py --llm real            # real LLM (local Qwen by default, see .env.local; replies are cached)

Writes results/mr/{runs.csv, summary.csv, summary.md, base_inputs.json, run_config.json} and docs/mr_catalog.md.
"""
import argparse
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mr.catalog import catalog_markdown
from mr.harness import DEFAULT_BASES, DEFAULT_SEEDS, DEFAULT_SNAPSHOT, run_all
from mr.inputs import POOL_SEED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)), help="comma-separated follow-up seeds (record them!)")
    ap.add_argument("--bases", type=int, default=DEFAULT_BASES, help="base inputs per MR")
    ap.add_argument("--pool-seed", type=int, default=POOL_SEED)
    ap.add_argument("--llm", choices=["offline", "real"], default="offline")
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    ap.add_argument("--only", help="comma-separated MR ids, e.g. MR1,MR3")
    ap.add_argument("--quick", action="store_true", help="5 base inputs, first seed only (smoke test)")
    ap.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    ap.add_argument("--out", default=str(ROOT / "results/mr"))
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",")]
    bases = a.bases
    if a.quick:
        seeds, bases = seeds[:1], 5
    runs, summary = run_all(snapshot=a.snapshot, seeds=seeds, n_bases=bases, pool_seed=a.pool_seed, llm_mode=a.llm,
                            workers=a.workers, only=a.only.split(",") if a.only else None, out_dir=a.out)

    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "mr_catalog.md").write_text(catalog_markdown())
    print("\n" + (pathlib.Path(a.out) / "summary.md").read_text())
    print(f"saved to {a.out}  (runs.csv has every trial; docs/mr_catalog.md is the MR catalog)")
    if (runs.status == "error").any():
        print(f"WARNING: {(runs.status == 'error').sum()} trial(s) crashed; see the 'detail' column of runs.csv")


if __name__ == "__main__":                      # required on Windows: worker processes re-import this file
    main()
