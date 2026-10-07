"""Run the Parser Agent against the frozen snapshot.
Usage: python scripts/try_parser.py "affordable homes in dallas"
Needs LLM2_API_KEY in .env.local (see README).
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from housing_trends import SnapshotContext, handle_input, llm_from_env

if len(sys.argv) != 2:
    sys.exit('usage: python scripts/try_parser.py "your request in quotes"')

df = pd.read_csv(ROOT / "data/snapshot/listings_dfw.csv", dtype={"zip_code": str})
ctx = SnapshotContext.from_dataframe(df)

res = handle_input(ctx=ctx, text=sys.argv[1], llm=llm_from_env())

print("OK:", res.ok, "| attempts:", res.attempts)
if res.ok:
    print(json.dumps(res.criteria.model_dump(), indent=2))
print("errors:     ", res.errors)
print("warnings:   ", res.warnings)
print("assumptions:", res.assumptions)
print("unsupported:", res.unsupported)
print("trace:      ", *res.trace, sep="\n  ")
