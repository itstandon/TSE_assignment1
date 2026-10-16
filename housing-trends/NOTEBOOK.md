## 2026-10-07: Data snapshot
- Built with `python scripts/build_snapshot.py` (defaults): seed 20261016, window 2025-09..2026-08,
  calibration window 2021-11..2022-04, typo rate 0.4%, rural rate 0.3%, shrink k = 20.
- Inputs: Kaggle realtor-data.zip.csv (178,858,272 bytes), Zillow City ZHVI (middle tier, smoothed, SA).
- Output: 54,004 rows, 248 ZIPs, 111 cities; hashes recorded in data/snapshot/snapshot_manifest.json.
- Assumption: Kaggle `prev_sold_date` on sold rows treated as the sale date (inferred, ~3% effect if wrong).
- Known limits: city-level ZHVI trend shared across ZIPs; calibration uses 2021-22 ratios applied to 2026.

> SUPERSEDED on 2026-10-08: first draft (all-DFW, 12-month). The frozen snapshot is described in the next entry.

## 2026-10-08: Data snapshot and first two agents

### Data
- Source 1: Kaggle USA Real Estate (realtor.com scrape), `realtor-data.zip.csv`, 178,858,272 bytes, ~2.2M rows.
- Source 2: Zillow ZHVI, City geography, mid tier (0.33-0.67), smoothed, seasonally adjusted, through 2026-08.
- Finding: Texas AND Dallas rows exist (Dallas city + Texas = 7,438 rows, 48 ZIPs). `state` holds the full name "Texas".
- Finding: Kaggle has no listing date. `prev_sold_date` on `sold` rows falls entirely in 2021-10-18..2022-05-02
  (all 812,009 sold rows have a date); on `for_sale` rows it is an old previous sale (back to 1901, ~49% blank).
  Bad dates exist (one row dated 3019, placeholder 1901-01-01).
- Consequence: a real 12-month trend is impossible from Kaggle alone (~6 usable months, Nov 2021-Apr 2022).

### Decision: synthetic snapshot grounded in both sources (option B)
- Built with `python scripts/build_snapshot.py --cities Dallas --window 24`, defaults otherwise.
- Seed 20261016. Window 2024-09..2026-08 (24 months). Dallas only: 42 ZIPs, 1 city.
- Price = f_zip x ZHVI(city, month) x size adjustment x noise. ZHVI supplies month-to-month movement;
  Kaggle supplies ZIP price level, size elasticity, noise, property mix and volume.
- Calibration (real rows, Nov 2021-Apr 2022): size elasticity 0.8177, noise sigma 0.1695,
  f_metro 0.0352, 26,838 real rows, 260 ZIPs calibrated (calibration uses all DFW; output is Dallas only).
- Injected faults (labels in `listings_dfw_truth.csv`, agents never read it): typos 0.4%, rural low-value 0.3%.
- Output: 19,976 rows, 42 ZIPs. Faults: 29 typo_x10, 42 typo_div10, 60 rural_lowvalue.
- Reproducibility: rerun gives identical SHA-256 hashes (see `snapshot_manifest.json`). Another machine produced the same data.

### Assumptions
- `prev_sold_date` on sold rows is treated as the sale date (inferred from its tight clustering, not documented).
  If wrong, calibration shifts ~3%.
- ZHVI is a city-level series, so every ZIP in the city shares one underlying trend.
- Calibration applies 2021-22 price ratios and property mixes to 2025-26.
- The window 2025-09..2026-08 is a label I chose; no row is a real 2026 transaction.

### Input Handler (built, 12 tests)
- Form path is pure code; free-text path is the Parser Agent (LLM) -> same validator.
- Parser model: openai/gpt-oss-120b via an OpenAI-compatible endpoint, temperature 0, seed 42.
  Wording of its `assumptions` varies between runs even at temperature 0; the validated Criteria stay identical.
- Rules: exactly 12-month window inside snapshot coverage; protected filters default to price, beds, baths.
- Decision: no neighbourhood table ("North Dallas" etc.); neighbourhood phrases are unsupported.

### Scanner (built, 55 tests including brute-force oracle)
- Plain code. Per ZIP-month bucket: counts, what each widening would add, and flags
  (`thin`, `moves`, `exhaustive`, `reachable`, `widened`).
- Thin means n < 20, where n = listings matching the user's filters in the current scope. **Threshold 20 kept for now.**
- Widening applies only to buckets thin at base; healthy buckets are never altered (0 of 200 altered; 98% before the fix).
- Limits: max 6 neighbour ZIPs, max 3 months each side, 3 rescans. "Adjacent ZIP" = numerically nearest ZIP (a proxy).
- A missing value on an active filter counts as a non-match.

### Measurements to cite
- At threshold 20, 304 of 504 buckets are thin (60%); at threshold 10 about 30%.
- Median noise: about +-13% with 5 listings, +-9% with 10, +-6.5% with 20, +-4.8% with 40 (ZIP 75241).
- Pooling 2 neighbour ZIPs shifts a median ~25% (thin buckets); pooling +-1 month ~7%.
- Time pooling alone fixes 278 of 304 thin buckets (229 at +-1, 44 at +-2, 5 at +-3); 26 need ZIPs too.
- The literal `exhaustive` flag never fired on this data; `reachable` is the practical stop signal.

### Open decisions
- Coverage check: count threshold + bias check (A), or precision-based using a CI on the median (B).
- Per-ZIP Scanner workers (assignment says "several agents each scan") vs one Scanner.
- Ask TA whether deterministic nodes count as agents.
- Optional LLM narrative node in the Rendering Agent.

### Next
Statistician, Critic, Pooling, graph wiring, MR harness.


## 2026-10-09: Statistician, and what the data says about the Critic

### Built
- Statistician (plain code, no LLM). Input: ScanResult + StatInstruction (method, per-bucket overrides,
  listing ids to exclude). Output per ZIP-month: price, ppsf, n_used / n_own / n_pooled / n_excluded,
  month-over-month % for price, ppsf and count; plus an overall series over all requested ZIPs.
- Methods: median (default), trimmed_mean, median_of_medians. Total tests now: 77 passing.
- Why no LLM: every decision is made upstream (Critic / Pooling / Diagnoser); it only does arithmetic, and
  the MRs need exact properties (scaling prices by k scales statistics by exactly k; duplicating all listings
  leaves medians unchanged), verified to 1e-9 in unit tests.
- Design choices: (1) trimmed mean uses FRACTIONAL trimming, so duplicating every listing leaves it unchanged
  (a floor-based trim does not: it would produce false MR violations); (2) median_of_medians equals the plain
  median for an unpooled bucket; (3) count change uses own listings so pooling cannot hide a real drop;
  (4) overall series uses own listings once, never pooled ones; (5) it never judges its own output.

### Measurements (Dallas snapshot, 42 ZIPs, window 2025-09..2026-08, default scope)
- Overall median price 397,400 -> 391,000 (-1.6%); ZHVI index -1.5%. Month-to-month sd of the overall median
  5.2%, versus 2.8% for a composition-free index: about half the swing is which ZIPs happen to list.
- A ZIP's own listing count changes by a median 57% month to month; 74% of healthy ZIP-months change by >30%.
  Cause: my generator's volumes are more volatile than a real market (gamma-mixed Poisson). Threat to validity.
- Price jumps >30% among ZIP-months: 53 of 451 (14% of thin buckets, 8% of healthy).
  Trimmed mean (0.1): 45. Removing all 71 typo listings: 52. Pooling +-3 months: 8. Then trimmed mean: 5.
  Pooling +-3 months and +2 ZIPs: 38 (ZIP pooling adds composition noise).
- Typos barely move medians: 33 of 504 bucket medians change when typos are removed; median shift 0.6%
  for buckets with n_own >= 20, 4.5% for thinner ones.

### Decisions
- Snapshot kept as frozen; volume volatility recorded as a threat to validity.
- Critic trend check flags PRICE jumps only; count changes are reported, not flagged.
- Order inside the Critic: coverage check first, then trend check. Pooling tries time before ZIPs.
- Because medians hide typos, the Critic also needs a listing-level outlier screen (price or ppsf far outside
  its ZIP's distribution), otherwise the Diagnoser (LLM) is almost never reached.
- Coverage check, variant A: count threshold 20 plus a post-widening bias check vs the unpooled value.

### Open
- Per-ZIP Scanner workers (assignment says "several agents each scan"): do only if time remains.
- Ask TA whether deterministic nodes count as agents. Narrative node in Rendering: optional.

### Next
Critic, Pooling, Diagnoser (LLM), LangGraph wiring and a captured loop trace, MR harness, UI, report.



## 2026-10-12: Streamlit UI and first MR harness

### Built
- Rendered the full agent pipeline in a Streamlit app (`app_input.py`, dashboard in `agents/renderer.py`; run with
  `make app`).
- Built a basic metamorphic harness around the app (`mr/`, `scripts/run_mrs.py`).

## 2026-10-16: Full metamorphic run, Qwen run, root-cause analysis

### State of the catalog
- 10 MRs (MR1..MR10): 2 Invariance, 4 Monotonic, 4 Directional. Two of the original 12 were withdrawn: "reorder
  listings" and "add a 10x fake month outside the window". Both were removed because they were redundant. The random stream
  of each MR is keyed by `relations.STREAM`, so renumbering did not change the transform instances of the others.
- The old `results/mr/summary.md` was stale: it still had the 12-MR numbering (its MR4 is today's MR2, and so on).
  It was regenerated today, so the old numbers must not be quoted.

### Offline run (headline table, `make mr`)
- 10 MRs x 30 distinct base inputs x seeds 101, 202, 303 = 900 trials, rule-based stand-in for every LLM node.
  0 errors, 1 skipped (MR6, the follow-up criteria matched no listing). Took about 13 min on 8 workers (18:05-18:18).
  Each trial is about 13-17 s, because the full pipeline runs twice on about 20k listings.
- Violating base inputs (Wilson 95% CI): MR2 30/30 (88.6-100), MR4 16/30 (36.1-69.8), MR7 13/30 (27.4-60.8),
  MR3 5/30, MR9 3/30, MR8 2/30. MR1, MR5, MR6 and MR10 are 0/30 (CI 0-11.4).
- The stand-in is deterministic, so the 3 seeds vary only the transform's random draws (shuffle, ids, which ZIP or
  month, N), not the agents. This is a threat to validity for the "non-deterministic agents" requirement, and it is
  why the Qwen run below exists.

### Qwen run (top 3 MRs only)
- Qwen 2.5 7B Instruct Q4_K_M served by **Ollama** 0.30.10 (`qwen2.5:7b`) on a GTX 1080 Ti, not by llama-server as
  the README describes. Command:
  `LLM_PROVIDER=qwen LLM2_BASE_URL=http://127.0.0.1:11434/v1 LLM2_MODEL=qwen2.5:7b python scripts/run_mrs.py --llm real --only MR2,MR4,MR7 --out results/mr_llm`
  Both overrides are needed: `.env.local` points `LLM2_*` at Groq. The model seed is the trial seed (101/202/303),
  temperature 0, replies cached in `results/llm_cache/`. `OLLAMA_NUM_PARALLEL=4`, 5 harness workers.
- Why only 3 MRs: about 25-60 LLM calls per trial with prompts of about 3k tokens, 6-7 tokens/s per request on this
  GPU, so about 4.5 h for 270 trials. All 10 MRs (900 trials) would not finish before the deadline. 30 bases x 3
  seeds per MR is kept, so the statistical protocol still holds for the MRs that were run.
- First attempt (8 workers) was killed: the SLURM job has a 10 GB memory cap. 8 harness workers + 2 diagnosis
  workers + the model server exceeded it, and the kernel OOM-killed llama-server at 18:33 (`memory.events: oom_kill 1`;
  Ollama returned HTTP 500 for the in-flight calls). 0 trials had finished. Restarted with 5 workers, and the cached
  replies were replayed. A trial with any failed LLM call is recorded as an error, never silently run on the fallback.
  Log of the failed attempt: `results/mr_llm_run_oom_attempt.log`.
- Final (`results/mr_llm/summary.md`): 270 trials, 9,108 LLM calls, 0 failed calls, 0 errors, 0 skipped. Took about
  4 h 21 min after the restart.

  | MR | Qwen bases (95% CI) | Qwen runs | Offline bases (95% CI) | Offline runs |
  |---|---|---|---|---|
  | MR2 | 30/30, 100.0% (88.6-100.0) | 90/90 | 30/30, 100.0% (88.6-100.0) | 90/90 |
  | MR4 | 25/30, 83.3% (66.4-92.7) | 75/90 | 16/30, 53.3% (36.1-69.8) | 48/90 |
  | MR7 | 21/30, 70.0% (52.1-83.3) | 61/90 | 13/30, 43.3% (27.4-60.8) | 24/90 |

- Qwen breaks MR4 and MR7 more often than the stand-in. In the cache, 957 of Qwen's 2,617 Pooling class choices (36.6%)
  borrow ZIPs, while the rule pads months whenever that suffices (278 of 304 thin buckets). That is consistent with the
  cross-ZIP leakage diagnosis. 12 invalid Qwen moves were rejected by the validator.
- Seeds barely matter: in 89 of 90 (MR, base) pairs all 3 seeds gave the same outcome (one mixed MR7 base).

### Root-cause analysis of the top 3 (`docs/root_cause_top3.md`, `scripts/diagnose_top3.py`, `results/mr_diagnosis/`)
- Ablation on the same 30 bases x 3 seeds (offline): `min_listings=0` disables Pooling, `jump_pct=1e9` disables
  the robust recompute and the Diagnoser.

  | MR | full | both loops off | Pooling off | jump loops off |
  |---|---|---|---|---|
  | MR2 | 90/90 runs | 0/90 | 18/90 | 90/90 |
  | MR4 | 48/90 | 0/90 | 36/90 | 48/90 |
  | MR7 | 24/90 | 0/90 | 0/90 | 24/90 |

  With the loops off every relation holds on 270/270 trials. That rules out a harness bug and a data artifact: same
  transforms, same checks, same listings.
- Bucket attribution (59 replays, one violating seed per violating base input): 0 of 261 violating ZIP-months is a
  plain own-listing median. Pooled: MR2 133/150, MR4 46/46, MR7 65/65.
- Verdicts:
  - MR4: genuine aggregation fault. Removing the maximum pushes buckets across the threshold, and Pooling then
    borrows dearer neighbours. Example: `75216@2025-06` (0 own listings) goes from $415,550 to $804,150 after
    removing a listing.
  - MR7: genuine fault, cross-ZIP leakage through Pooling. 0/90 with Pooling off. MR10 (global scale) holds.
  - MR2: mostly an ill-conceived MR. The coverage threshold is on counts, and copying changes counts, so different
    buckets get pooled. There is also a genuine Diagnoser fault. The suspect list is capped at 8 and chosen by
    listing id: 8 copies of one $3.8M bad listing fill every slot, so the `legit_rural` listing in the same ZIP is
    never judged, and the ZIP reverts from trimmed mean to plain median (`diagnoser.py:153-157`). Replay: MR2 base 0,
    seed 101, `min_listings=0`, ZIP 75232, January $276,242 -> $267,700.
- Not yet confirmed: that the 36 MR4 runs that still fail with Pooling off have the same Diagnoser cause.

### Decisions
- Headline table = offline run (re-runnable by anyone in about 15 min on 8 cores, with no GPU). The Qwen table is reported
  separately for MR2/MR4/MR7, and it is reproducible from the committed `results/llm_cache/`.
- Commit `results/llm_cache/`: without it a re-run must query Qwen again, and parallel batching or a different
  Ollama version can change a reply.

### Next
Finish the Qwen run, update README reproduction steps (Ollama), write the report (results, root cause, threats to
validity, reflection), first git commit.
