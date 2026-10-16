"""Run: python scripts/diagnose_top3.py [workers]   (offline stand-in; writes results/mr_diagnosis/)
Evidence for the root-cause analysis of the top-3 MRs (offline stand-in).
  ablation    : re-run MR2/MR4/MR7 with Pooling disabled (min_listings=0) and the jump loops disabled (jump_pct=1e9)
  attribution : for every violating ZIP-month of the real run, was it pooled / robust-recomputed / had exclusions?
"""
import json
import pathlib
import re
import sys
from concurrent.futures import ProcessPoolExecutor

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = REPO / "results/mr_diagnosis"
sys.path.insert(0, str(REPO))
import pandas as pd  # noqa: E402

from mr.harness import replay, run_all  # noqa: E402

MRS = ["MR2", "MR4", "MR7"]
KEY = re.compile(r"(\d{5}@\d{4}-\d{2})")


def attribute(t):
    mr, base, seed = t
    r = replay(mr, base, seed)
    keys = sorted({k for p in r["problems"] for k in KEY.findall(p)})
    sb = r["source_report"].buckets.set_index(r["source_report"].buckets.zip_code + "@" + r["source_report"].buckets.month)
    fb = r["followup_report"].buckets.set_index(r["followup_report"].buckets.zip_code + "@" + r["followup_report"].buckets.month)
    rows = []
    for k in keys:
        s, f = sb.loc[k], fb.loc[k]
        rows.append(dict(mr=mr, base_id=base, seed=seed, bucket=k,
                         src_pooled=int(s.n_pooled) > 0, fol_pooled=int(f.n_pooled) > 0,
                         src_n_own=int(s.n_own), fol_n_own=int(f.n_own), src_n_pooled=int(s.n_pooled), fol_n_pooled=int(f.n_pooled),
                         src_method=s.method, fol_method=f.method,
                         src_excl=int(s.n_excluded), fol_excl=int(f.n_excluded),
                         src_price=float(s.price), fol_price=float(f.price)))
    return dict(mr=mr, base_id=base, seed=seed, n_problems=len(r["problems"]), meta=str(r["meta"])[:300],
                src_loops=r["source_loops"], fol_loops=r["followup_loops"],
                src_trace=list(r["source_report"].trace) if hasattr(r["source_report"], "trace") else [],
                fol_trace=list(r["followup_report"].trace) if hasattr(r["followup_report"], "trace") else [],
                buckets=rows)


def main():
    OUT.mkdir(exist_ok=True)
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    print("ablation: Pooling + jump loops OFF", flush=True)
    _, s = run_all(seeds=(101, 202, 303), n_bases=30, only=MRS, min_listings=0, jump_pct=1e9, workers=workers,
                   out_dir=OUT / "ablation_no_loops")
    print(s[["mr", "base_inputs", "base_violations", "base_rate", "run_violations", "runs", "errors"]].to_string(), flush=True)
    print("ablation: Pooling OFF only (jump loops ON)", flush=True)
    _, s = run_all(seeds=(101, 202, 303), n_bases=30, only=MRS, min_listings=0, workers=workers,
                   out_dir=OUT / "ablation_no_pooling")
    print(s[["mr", "base_inputs", "base_violations", "base_rate", "run_violations", "runs", "errors"]].to_string(), flush=True)
    print("ablation: jump loops OFF only (Pooling ON)", flush=True)
    _, s = run_all(seeds=(101, 202, 303), n_bases=30, only=MRS, jump_pct=1e9, workers=workers,
                   out_dir=OUT / "ablation_no_jumploops")
    print(s[["mr", "base_inputs", "base_violations", "base_rate", "run_violations", "runs", "errors"]].to_string(), flush=True)

    runs = pd.read_csv(REPO / "results/mr/runs.csv")
    v = runs[runs.mr.isin(MRS) & (runs.status == "violated")]
    tasks = [tuple(x) for x in v.groupby(["mr", "base_id"]).seed.min().reset_index()[["mr", "base_id", "seed"]].itertuples(index=False)]
    print(f"attribution: replaying {len(tasks)} violating trials (one seed per violating base input)", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(attribute, tasks))
    (OUT / "attribution.json").write_text(json.dumps(res, indent=1, default=str))
    b = pd.DataFrame([row for r in res for row in r["buckets"]])
    b.to_csv(OUT / "attribution_buckets.csv", index=False)
    b["pooled_any"] = b.src_pooled | b.fol_pooled
    b["robust_any"] = (b.src_method != "median") | (b.fol_method != "median")
    b["excl_any"] = (b.src_excl > 0) | (b.fol_excl > 0)
    b["plain"] = ~(b.pooled_any | b.robust_any | b.excl_any)
    print(b.groupby("mr")[["pooled_any", "robust_any", "excl_any", "plain"]].agg(["sum", "count"]).to_string(), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
