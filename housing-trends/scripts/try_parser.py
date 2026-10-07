"""Try the Parser Agent against a toy snapshot.
Usage: python scripts/try_parser.py "affordable homes in north dallas"
Needs LLM2_API_KEY (and optionally LLM2_MODEL, LLM2_BASE_URL) in the environment.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
from housing_trends import SnapshotContext, handle_input, llm_from_env

if len(sys.argv) != 2:
    sys.exit('usage: python scripts/try_parser.py "your request in quotes"')

# Toy snapshot: replace with your real one once it exists.
rows = []
for i, m in enumerate(str(p) for p in pd.period_range("2021-01", "2022-06", freq="M")):
    for z, c, base in [(75201, "Dallas", 400_000), (75204, "Dallas", 450_000),
                       (75230, "Dallas", 700_000), (75024, "Plano", 600_000)]:
        rows.append(dict(zip_code=float(z), city=c, price=base + i * 1000, month=m))
ctx = SnapshotContext.from_dataframe(
    pd.DataFrame(rows),
    area_gazetteer={"North Dallas": ["75230"]},   # placeholder: verify real ZIPs later
)

res = handle_input(ctx=ctx, text=sys.argv[1], llm=llm_from_env())

print("OK:", res.ok, "| attempts:", res.attempts)
if res.ok:
    print(json.dumps(res.criteria.model_dump(), indent=2))
print("errors:     ", res.errors)
print("warnings:   ", res.warnings)
print("assumptions:", res.assumptions)
print("unsupported:", res.unsupported)
print("trace:      ", *res.trace, sep="\n  ")
