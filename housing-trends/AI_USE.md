# AI_USE.md

Team: Arushi Tandon, Bavya Sri Kilari. Period covered: 7-16 October 2026.

## 1. Tools used

| Tool | Model / version | Used for |
|---|---|---|
| Claude (claude.ai chat) | Claude Sonnet 5.5 or Claude Opus 5 (the model per conversation was not recorded) | Design discussion, writing and testing code, data analysis, drafting notes |
| Claude Code (CLI agent on the ada cluster) | Claude Opus 5.5 (`claude-opus-5-5`) | 2026-10-16: ran the MR harness (offline and Qwen), set up the Ollama-served Qwen run, wrote `scripts/diagnose_top3.py`, drafted `docs/root_cause_top3.md` and the 2026-10-16 NOTEBOOK entry |
| Qwen 2.5 7B Instruct, local | `qwen2.5:7b` (Q4_K_M) served by Ollama 0.30.10, temperature 0, seed = trial seed (101/202/303) | LLM nodes INSIDE the application for the Qwen MR run (`results/mr_llm/`) |
| gpt-oss-120b via an OpenAI-compatible hosted endpoint | `openai/gpt-oss-120b`, temperature 0, seed 42 | LLM nodes INSIDE the application (earlier runs): Parser Agent, Scanner neighbour-ZIP ranking, Critic and Pooling decisions, Anomaly Diagnoser |

## 2. What the AI did and what we did

| Component | AI role | Our role |
|---|---|---|
| Design note and architecture | Reviewed our draft; pointed out inconsistencies (agent counts, ZHVI not wired in) | Wrote the design note entirely ourselves (both team members): architecture, loops, data source, MR sketches; made all final decisions |
| Data snapshot (`build_snapshot.py`) | Wrote the generator and analysed the Kaggle and Zillow files | Downloaded data, ran it, chose option B, Dallas-only, 24 months |
| Input Handler, Parser Agent | Wrote the code and 12 tests | Ran and tested against the hosted model |
| Scanner, Statistician | Wrote the code and tests | Ran them on the snapshot, reviewed the output |
| Critic, Pooling, Diagnoser, graph, harness, UI | Wrote the code and tests | Designed the loops, tested the UI |
| MR runs (2026-10-16) | Claude Code ran `scripts/run_mrs.py` offline (10 MRs, 900 trials) and with Qwen (MR2, MR4, MR7, 270 trials), and diagnosed and recovered an OOM kill of the model server | Decided, given the deadline, to run all 10 MRs on the offline stand-in and only the top 3 on Qwen, and to keep the local Qwen instead of switching to a hosted API |
| Root-cause analysis of the top 3 MRs | Claude Code designed and ran the ablation (loops switched off) and the bucket attribution, replayed MR2 base 0 seed 101 to find the Diagnoser suspect-cap mechanism, and drafted `docs/root_cause_top3.md` | Asked for the diagnosis to be backed by a replay of a concrete trial; will edit the draft into the report |

## 3. How we checked AI output

- Unit tests (77 passing at the time of writing). The Scanner's counts and flags are compared with an independent brute-force oracle. The Statistician is tested against hand-computed values and properties (scaling, duplication, row-order invariance).
- Reproducibility: the snapshot builder is seeded and records SHA-256 hashes. A rerun on our laptop and a run in the AI's environment (different numpy/pandas versions) gave byte-identical files.
- Measurement on real data before accepting design claims (see section 5).
- Ablation as a check on the diagnosis: every violation of MR2/MR4/MR7 disappears when Pooling and the jump loops are switched off (270/270 trials hold). So the transform and check code is not the cause, and the AI's causal claims are tested by switching each loop on and off, not just argued.
- Each causal claim in `docs/root_cause_top3.md` cites a replayable trial (`scripts/mr_replay.py <MR> <base> <seed>`) or a file in `results/mr_diagnosis/`. One claim that was not replayed (the Diagnoser cause of the 36 MR4 runs that fail with Pooling off) is marked as not confirmed.

## 4. LLMs inside the application

- Parser Agent: free text goes to the model, which returns a JSON spec. The output is treated as untrusted. The same deterministic validator as the form path checks it. On failure the agent gets the errors back once, then gives up and asks the user to use the form.
- Vague terms ("affordable" = snapshot median, "luxury" = 75th percentile) are rules in the prompt. They are our own definitions, not market facts.
- The wording of the model's `assumptions` varies between runs even at temperature 0 with a fixed seed. The validated criteria stayed identical.
- Anomaly Diagnoser: code gathers the evidence (the Critic's outlier screen plus ZIP percentiles); the LLM judges each suspect listing as `bad_listing` (excluded, plain median), `legit_rural` (kept, ZIP series computed with the trimmed mean) or `unexplained` (excluded and flagged). Replies are untrusted: bad JSON, unknown ids or verdicts are rejected, the model gets the errors back once, and if it still fails every suspect becomes `unexplained`.
- Every LLM reply is cached on disk (`results/llm_cache/`, keyed by model tag + seed + prompt), so the Qwen MR run replays exactly without a GPU. Without the cache, a re-run is NOT guaranteed identical: 4 parallel Ollama slots batch requests, and GPU arithmetic or a different Ollama version can change a reply even at temperature 0.
- The harness counts failed LLM calls. A trial in which any call failed is recorded as an error, not as a pass or violation, because each node would otherwise fall back silently to its rule-based policy.

## 5. Cases where the AI was wrong, and how we caught it

| # | What went wrong | How it was caught | Fix |
|---|---|---|---|
| 1 | AI diagnosed a "missing" API key by assuming the shell and Python share environment variables | Our `grep dotenv` showed `llm.py` loaded `.env.local` itself | Corrected diagnosis; anchored the path to the repo root |
| 2 | Builder script hard-coded a Zillow filename taken from an upload that had been renamed (dots became underscores) | `FileNotFoundError` on our machine | Renamed file; paths are now CLI flags |
| 3 | AI generated an all-DFW snapshot and used "Plano" as a test example; we wanted Dallas only | We asked why Plano was there | Added a `--cities` flag; rebuilt and re-froze |
| 4 | AI told us to edit `scripts/try_parser.py`, a file we had never created | We said so | AI supplied the full file |
| 5 | AI presented a price-vs-ZHVI ratio as corroborating our reading of `prev_sold_date`; the test could not distinguish the alternatives (ratio 1.03 vs 1.00) | Re-examined when we asked whether the dates were sale dates | Downgraded to a stated assumption (~3% effect if wrong) |
| 6 | AI gave inconsistent figures for blank previous-sale dates (61% vs 49%) and the earliest sold date | Re-ran the check when we questioned the date column | Reconciled: 61% was DFW only, 49% nationwide; earliest sold date is 2021-10-18 |
| 7 | The Scanner applied one scope to every bucket, so a rescan meant to fix thin buckets also changed healthy ones | Measured while answering whether the coverage loop made sense: 98% of healthy buckets were altered | Thin-only widening; tests added; 0 of 200 altered afterwards |
| 8 | AI justified the coverage loop as necessary because Pooling "picks the gentlest move" | Our question: is the gentlest move not always time? Measurement: time suffices for 278 of 304 thin buckets | Conceded the loop is partly decorative; justification rewritten around per-bucket depth and verification of the widened result |
| 9 | Generator made listing volumes too volatile (listing counts change by >30% in 74% of healthy ZIP-months) | Statistician output on the real snapshot | Documented; Critic will flag price jumps only |
| 10 | AI-written tests were wrong twice (a neighbour-order expectation; a threshold that prevented pooling) | Tests failed, and the code was right | Tests fixed, which shows tests were run and not trusted |
| 11 | Claude Code launched the first Qwen run with 8 workers while a 2-worker diagnosis job ran alongside. Together with the model server this exceeded the SLURM job's 10 GB memory cap, and the kernel OOM-killed the model server | Zero trials finished after 14 min; the Ollama log showed HTTP 500s and `llama-server ... signal: killed`, and the job cgroup's `memory.events` showed `oom_kill 1` | Restarted with 5 workers. Cached replies were replayed, and failed calls are recorded as errors, so no result was affected. Logged as a threat to validity |
| 12 | Claude Code estimated the offline run at "a few minutes"; it took about 13 min, and its first Qwen estimate was too optimistic (about 4.5 h for 270 trials) | Measured per-trial time (13-17 s offline, 25-60 LLM calls per Qwen trial) | Qwen run limited to the top 3 MRs (still 30 bases x 3 seeds each); the decision is recorded in NOTEBOOK |
| 13 | Claude Code's first explanation of MR2's non-pooling violations ("the Diagnoser excludes by id, so copies escape exclusion") was only partly right | The replay of MR2 base 0 seed 101 showed the main effect is the suspect cap (8 copies crowd out the `legit_rural` listing, so the ZIP reverts to the plain median); one copy escaping exclusion is secondary | Root-cause draft rewritten from the replay trace |
| 14 | While drafting NOTEBOOK.md, Claude Code wrote a reason for withdrawing two MRs and a run time (30 min) that it had not verified | Checked against `mr/relations.py` (no reason given) and file timestamps (18:05-18:18) | Reason left as a placeholder for us; time corrected to about 13 min |

## 6. Claims we have not yet independently verified

- "Pooling neighbouring ZIPs distorts a median more than thinness does" was measured as the change from the unpooled median. For thin buckets part of that change is noise correction, not bias. Not yet re-tested against the generator's ground truth (`data/snapshot/listings_dfw_truth.csv`), so we do not cite it as a finding.
- Sampling-noise figures (about ±13% for 5 listings, ±6.5% for 20) come from one ZIP (75241).
- Reading `prev_sold_date` on sold rows as the sale date is an inference.