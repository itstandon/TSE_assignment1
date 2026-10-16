"""The harness: source run -> transform -> follow-up run -> check, for every (MR, base input, seed).

Statistical protocol (assignment section 2):
  * each MR is run on N distinct base inputs (default 30) taken from a reproducible pool (inputs.generate_pool);
  * each base input is repeated under k fixed seeds (default 3). The seeds drive the random parts of the transforms
    (the shuffle, the ids, which ZIPs, which month); with the real LLM the model seed is fixed and replies are cached;
  * a base input counts as VIOLATED when at least one of its seeds violates the relation. The headline rate is
    violated base inputs / base inputs, with a Wilson 95% interval; the per-run rate is reported as well.
A run that crashes is recorded as an ERROR (a harness or system bug), and a follow-up that matches no listing as SKIPPED;
neither is counted as a violation or a pass.
"""
from __future__ import annotations

import json
import pathlib
import platform
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from housing_trends import Criteria
from housing_trends.agents.diagnoser import rule_based_llm
from housing_trends.agents.scanner import Snapshot
from housing_trends.graph import run
from housing_trends.state import JUMP_PCT, MIN_LISTINGS

from .contract import contract
from .inputs import POOL_SEED, generate_pool, matching_rows, months_of
from .relations import MR_BY_ID, MRS, STREAM
from .stats import wilson

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_SNAPSHOT = ROOT / "data/snapshot/listings_dfw.csv"
DEFAULT_SEEDS = (101, 202, 303)
DEFAULT_BASES = 30


def seed_stream(seed: int, base_id: int, mr_id: str):
    """The random stream of one trial: a function of (seed, base input, relation) only. The relation's stream index is
    fixed in relations.STREAM, not its position in MRS, so adding, withdrawing or renumbering relations never changes the
    transform instances (which month, ZIP, listing, N) of the others."""
    return np.random.default_rng([seed, base_id, STREAM[mr_id]])


class CountingLLM:
    """Counts the calls an agent makes and the ones that FAIL. Every LLM node catches exceptions and falls back to its
    rule-based policy, so without this a dead API key or an unreachable server would silently turn a 'real model' run
    into an offline one. A trial in which any call failed is recorded as an error, never as a pass or a violation."""

    def __init__(self, llm):
        self.llm, self.calls, self.failures = llm, 0, 0
        self.name = getattr(llm, "name", "llm")

    def __call__(self, system, user):
        self.calls += 1
        try:
            return self.llm(system, user)
        except Exception:
            self.failures += 1
            raise


def make_llm(mode: str, seed: int | None = None):
    """offline: the deterministic stand-in. real: the hosted/local model with ITS seed set to `seed` (the trial's recorded
    seed, so source and follow-up runs of a trial share it), replies cached on disk for replay."""
    if mode == "offline":
        return rule_based_llm
    from housing_trends.llm import CachedLLM, llm_from_env
    base = llm_from_env(seed)
    return CountingLLM(CachedLLM(base, ROOT / "results/llm_cache", tag=base.name))


def preflight(llm_mode, seed):
    """Fail BEFORE a long run if the real model cannot answer (bad key, wrong provider, server down)."""
    if llm_mode == "offline":
        return
    llm = make_llm(llm_mode, seed)
    try:
        llm("Reply with ONE JSON object and nothing else.", 'Return {"ok": true}')
    except Exception as ex:
        raise RuntimeError(f"preflight failed: the real model is not answering ({ex}). Every LLM node would silently fall "
                           "back to its rule-based policy, so this run would not measure the model. Set LLM_PROVIDER "
                           "(groq | qwen) and its key/server, or use --llm offline.") from ex


def run_contract(df, c, llm, min_listings=MIN_LISTINGS, jump_pct=JUMP_PCT):
    """One pipeline run -> (contract table, loops used 'rescans/recomputes/diagnoser rounds', Report)."""
    state = run(c, Snapshot.frozen(df), llm, min_listings=min_listings, jump_pct=jump_pct)
    loops = "/".join(str(state.get(k, 0)) for k in ("rescans", "recomputes", "diag_rounds"))
    return contract(state["report"]), loops, state["report"]


def evaluate(mr, df, c, base_id, seed, *, llm=rule_based_llm, min_listings=MIN_LISTINGS, jump_pct=JUMP_PCT, src_cache=None):
    """One (MR, base input, seed) trial. Returns a flat dict (one row of runs.csv)."""
    src_cache = {} if src_cache is None else src_cache
    row = dict(mr=mr.id, base_id=base_id, seed=seed, criteria=c.key(), status="", detail="", src_loops="", fol_loops="")
    counted = hasattr(llm, "failures")
    calls0, fails0 = (llm.calls, llm.failures) if counted else (0, 0)
    try:
        rng = seed_stream(seed, base_id, mr.id)
        follow = mr.transform(df, c, rng)
        if matching_rows(follow.df, follow.criteria) == 0:
            return {**row, "status": "skipped", "detail": "follow-up criteria match no listing"}
        key = (c.key(), seed if counted else None)               # a seeded model answers per seed; the stand-in does not
        if key not in src_cache:
            src_cache[key] = run_contract(df, c, llm, min_listings, jump_pct)[:2]
        src, row["src_loops"] = src_cache[key]
        fol, row["fol_loops"], _ = run_contract(follow.df, follow.criteria, llm, min_listings, jump_pct)
        problems = mr.check(src, fol, c, follow.meta)
        row.update(status="violated" if problems else "holds", detail=" | ".join(problems))
    except Exception as ex:                                                  # a crash is a finding of its own
        row.update(status="error", detail=f"{type(ex).__name__}: {ex} @ {traceback.format_exc().splitlines()[-3].strip()}")
    if counted:
        row["llm_calls"], row["llm_failures"] = llm.calls - calls0, llm.failures - fails0
        if row["llm_failures"]:                                              # degraded to the fallback: not a valid measurement
            row.update(status="error", detail=f"{row['llm_failures']} LLM call(s) failed; agents used their fallback policy")
    return row


# ---------------------------------------------------------------- parallel workers
_G: dict = {}


def _init(snapshot_path, llm_mode, min_listings, jump_pct):
    df = pd.read_csv(snapshot_path, dtype={"zip_code": str})
    _G.update(df=df, mode=llm_mode, llms={}, min_listings=min_listings, jump_pct=jump_pct, cache={})


def _llm_for(seed):
    """One model client per trial seed (offline: the single stand-in)."""
    if _G["mode"] == "offline":
        return rule_based_llm
    if seed not in _G["llms"]:
        _G["llms"][seed] = make_llm(_G["mode"], seed)
    return _G["llms"][seed]


def _task(t):
    mr_id, base_id, criteria_json, seed = t
    return evaluate(MR_BY_ID[mr_id], _G["df"], Criteria.model_validate_json(criteria_json), base_id, seed, llm=_llm_for(seed),
                    min_listings=_G["min_listings"], jump_pct=_G["jump_pct"], src_cache=_G["cache"])


def build_tasks(df, pool, mr_ids, n_bases, seeds):
    info = {"last_month": months_of(df)[-1]}
    tasks, chosen = [], {}
    for mid in mr_ids:
        mr = MR_BY_ID[mid]
        bases = [b for b in pool if mr.applies(b.criteria, info)][:n_bases]
        chosen[mid] = [b.id for b in bases]
        tasks += [(mid, b.id, b.criteria.model_dump_json(), s) for b in bases for s in seeds]
    return tasks, chosen


# ---------------------------------------------------------------- summary
def summarize(runs: pd.DataFrame, seeds) -> pd.DataFrame:
    rows = []
    for mr in MRS:
        r = runs[runs.mr == mr.id]
        valid = r[r.status.isin(["holds", "violated"])]
        by_base = valid.groupby("base_id").status.apply(lambda s: (s == "violated").any()).astype(bool)
        rv, rn, bv, bn = int((valid.status == "violated").sum()), len(valid), int(by_base.sum()), len(by_base)
        (rl, rh), (bl, bh) = wilson(rv, rn), wilson(bv, bn)
        rows.append(dict(mr=mr.id, name=mr.name, family=mr.family, base_inputs=bn, seeds=" ".join(map(str, seeds)),
                         runs=rn, run_violations=rv, run_rate=rv / rn if rn else float("nan"), run_ci_low=rl, run_ci_high=rh,
                         base_violations=bv, base_rate=bv / bn if bn else float("nan"), base_ci_low=bl, base_ci_high=bh,
                         skipped=int((r.status == "skipped").sum()), errors=int((r.status == "error").sum())))
    return pd.DataFrame(rows)


def summary_markdown(s: pd.DataFrame) -> str:
    head = ("| MR | Family | Relation | Base inputs | Seeds | Violating base inputs | Rate (95% Wilson CI) | "
            "Violating runs / runs | Skipped | Errors |\n|---|---|---|---|---|---|---|---|---|---|\n")
    pct = lambda p, lo, hi: "n/a" if p != p else f"{100 * p:.1f}% ({100 * lo:.1f}-{100 * hi:.1f})"
    body = "".join(f"| {r.mr} | {r.family} | {r['name']} | {r.base_inputs} | {r.seeds} | {r.base_violations} | "
                   f"{pct(r.base_rate, r.base_ci_low, r.base_ci_high)} | {r.run_violations} / {r.runs} | {r.skipped} | {r.errors} |\n"
                   for _, r in s.iterrows())
    return head + body


def snapshot_sha(path):
    try:
        m = json.loads((pathlib.Path(path).parent / "snapshot_manifest.json").read_text())
        return m["output_sha256"][pathlib.Path(path).name]
    except Exception:
        return "unknown"


def run_all(*, snapshot=DEFAULT_SNAPSHOT, seeds=DEFAULT_SEEDS, n_bases=DEFAULT_BASES, pool_seed=POOL_SEED, llm_mode="offline",
            workers=1, only=None, min_listings=MIN_LISTINGS, jump_pct=JUMP_PCT, out_dir=ROOT / "results/mr", log=print):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(snapshot, dtype={"zip_code": str})
    pool = generate_pool(df, seed=pool_seed)
    mr_ids = list(only) if only else [m.id for m in MRS]
    preflight(llm_mode, seeds[0])
    tasks, chosen = build_tasks(df, pool, mr_ids, n_bases, seeds)
    log(f"{len(pool)} base inputs in the pool; {len(tasks)} trials for {len(mr_ids)} MR(s); {workers} worker(s); LLM: {llm_mode}")

    results = []
    if workers <= 1:
        _init(snapshot, llm_mode, min_listings, jump_pct)
        for i, t in enumerate(tasks, 1):
            results.append(_task(t))
            if i % max(1, len(tasks) // 10) == 0:
                log(f"  {i}/{len(tasks)} trials")
    else:
        with ProcessPoolExecutor(workers, initializer=_init, initargs=(str(snapshot), llm_mode, min_listings, jump_pct)) as ex:
            for i, r in enumerate(ex.map(_task, tasks, chunksize=2), 1):
                results.append(r)
                if i % max(1, len(tasks) // 10) == 0:
                    log(f"  {i}/{len(tasks)} trials")

    runs = pd.DataFrame(results)
    summary = summarize(runs, seeds)
    runs.to_csv(out / "runs.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(summary_markdown(summary))
    (out / "base_inputs.json").write_text(json.dumps(
        {"pool_seed": pool_seed, "chosen_per_mr": chosen,
         "criteria": {b.id: json.loads(b.criteria.model_dump_json()) for b in pool}}, indent=1))
    llm = make_llm(llm_mode, seeds[0])
    real = llm_mode != "offline"
    extra = ({"llm_seed_policy": "the model's seed = the trial seed (source and follow-up runs share it)",
              "llm_calls": int(runs.llm_calls.sum()), "llm_failures": int(runs.llm_failures.sum())} if real else {})
    (out / "run_config.json").write_text(json.dumps(
        {**extra, "seeds": list(seeds), "bases_per_mr": n_bases, "pool_seed": pool_seed,
         "llm": getattr(getattr(llm, "llm", llm), "name", llm_mode),
         "min_listings": min_listings, "jump_pct": jump_pct, "snapshot": str(pathlib.Path(snapshot).name),
         "snapshot_sha256": snapshot_sha(snapshot), "python": platform.python_version(), "pandas": pd.__version__,
         "numpy": np.__version__, "mrs": mr_ids}, indent=1))
    return runs, summary


# ---------------------------------------------------------------- replay one trial (for root-cause analysis)
def replay(mr_id, base_id, seed, *, snapshot=DEFAULT_SNAPSHOT, pool_seed=POOL_SEED, llm_mode="offline",
           min_listings=MIN_LISTINGS, jump_pct=JUMP_PCT):
    """Re-run one trial and return everything needed to explain it: criteria, problems, both contracts, loops, traces."""
    df = pd.read_csv(snapshot, dtype={"zip_code": str})
    base = {b.id: b for b in generate_pool(df, seed=pool_seed)}[base_id]
    mr, llm = MR_BY_ID[mr_id], make_llm(llm_mode, seed)
    rng = seed_stream(seed, base_id, mr.id)
    follow = mr.transform(df, base.criteria, rng)
    src, src_loops, src_rep = run_contract(df, base.criteria, llm, min_listings, jump_pct)
    fol, fol_loops, fol_rep = run_contract(follow.df, follow.criteria, llm, min_listings, jump_pct)
    return dict(mr=mr, criteria=base.criteria, follow_criteria=follow.criteria, meta=follow.meta,
                problems=mr.check(src, fol, base.criteria, follow.meta), source=src, followup=fol,
                source_loops=src_loops, followup_loops=fol_loops, source_report=src_rep, followup_report=fol_rep)
