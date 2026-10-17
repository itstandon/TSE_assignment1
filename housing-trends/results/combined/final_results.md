# Final MR results: offline (rule-based stand-in) vs Qwen 2.5 7B

Setup (identical in both): frozen Dallas snapshot `listings_dfw.csv` (sha256 2aa4f1fd...), 30 base inputs per MR, seeds 101/202/303 (90 runs per MR), min_listings=20, jump=30%. "Bases" = base inputs with at least one violating seed; the rate and 95% Wilson CI are over base inputs. "Runs" = violating runs / runs.
Offline = the agents' LLM decisions replaced by a deterministic rule-based stand-in (`results/mr/`). Qwen = the real model, Qwen 2.5 7B via Ollama, replayed from `results/llm_cache/` (`results/mr_llm/` for MR2/MR4/MR7, `results/mr_llm_rest/` for the other seven).

## Main table (all 10 MRs)

| MR | Relation | Family | Offline bases | Offline rate (95% CI) | Offline runs | Qwen bases | Qwen rate (95% CI) | Qwen runs |
|---|---|---|---|---|---|---|---|---|
| MR1 | Sort listings by price (adversarial order) | Invariance | 0/30 | 0.0% (0.0-11.4) | 0/90 | 0/30 | 0.0% (0.0-11.4) | 0/90 |
| MR2 | Multiply each ZIP's listings by a random N (1-10) | Invariance | 30/30 | 100.0% (88.6-100.0) | 90/90 | 30/30 | 100.0% (88.6-100.0) | 90/90 |
| MR3 | Add $X to every price in one month | Monotonic | 5/30 | 16.7% (7.3-33.6) | 7/90 | 19/30 | 63.3% (45.5-78.1) | 40/90 |
| MR4 | Remove the dearest listing of each ZIP | Monotonic | 16/30 | 53.3% (36.1-69.8) | 48/90 | 25/30 | 83.3% (66.4-92.7) | 75/90 |
| MR5 | Widen the price band | Monotonic | 0/30 | 0.0% (0.0-11.4) | 0/90 | 0/30 | 0.0% (0.0-11.4) | 0/90 |
| MR6 | Restrict to a sub-region | Monotonic | 0/30 | 0.0% (0.0-11.4) | 0/89 | 1/30 | 3.3% (0.6-16.7) | 1/89 |
| MR7 | Double all prices in one ZIP | Directional | 13/30 | 43.3% (27.4-60.8) | 24/90 | 21/30 | 70.0% (52.1-83.3) | 61/90 |
| MR8 | Inject one 10x outlier | Directional | 2/30 | 6.7% (1.8-21.3) | 3/90 | 9/30 | 30.0% (16.7-47.9) | 13/90 |
| MR9 | Replace a month with a copy of the previous month | Directional | 3/30 | 10.0% (3.5-25.6) | 6/90 | 4/30 | 13.3% (5.3-29.7) | 6/90 |
| MR10 | Scale all prices by k > 1 | Directional | 0/30 | 0.0% (0.0-11.4) | 0/90 | 22/30 | 73.3% (55.6-85.8) | 64/90 |

Reading it: MR2, MR4 and MR7 fail offline too, so they are structural faults in the pipeline, not LLM effects. MR1 and MR5 hold in both settings. MR3, MR8 and MR10 are much worse with the LLM. MR10 is the clearest case: 0/30 offline, 22/30 with Qwen.

## What each results folder is

| Folder | What it is |
|---|---|
| `mr` | Offline run, all 10 MRs, 900 trials. Source of the offline column. |
| `mr_llm` | Qwen run for MR2, MR4, MR7 only (270 trials, 8,901 LLM calls, about 4.3 h). Source of those three Qwen rows. |
| `mr_llm_rest` | Qwen run for the other seven MRs (630 trials). The first attempt was interrupted: MR6 had 70/90 and MR10 0/90 done. The MR6 and MR10 rows were then filled in from `mr_llm_rest2` (backup of the earlier file: `runs.backup.csv`). So `mr_llm_rest` is already complete for MR1, 3, 5, 6, 8, 9, 10. |
| `mr_llm_rest2` | The re-run of MR6 + MR10 (180 trials) that was merged into `mr_llm_rest`. It is the same data, so it is not a separate result. The 70 MR6 rows it had in common with the first attempt matched exactly. |
| `mr_llm_mr10_noloops`, `mr_llm_mr10_nopool`, `mr_llm_ctrl_k1e-6` | Extra diagnostic runs for the MR10 Qwen finding (below). |
| `mr_diagnosis/ablation_*` | Offline ablations for the top-3 MRs (below). |
| `mr_llm_run_oom_attempt.log` | Failed first start of the Qwen run (8 workers, ran out of memory). It was restarted with 5 workers. |

Do not add up the rows of `mr_llm_rest` and `mr_llm_rest2`: that counts MR6 and MR10 twice. `all_runs.csv` and `all_summary.csv` in this folder have that double count, so use this file and `final_table.csv` for results.

## Additional runs and their results

### A. Offline ablations of MR2, MR4, MR7 (`results/mr_diagnosis/`)
Same 30 bases x 3 seeds, offline. `min_listings=0` switches Pooling off. `jump_pct=1e9` switches off the robust-recompute and Diagnoser loops. Violating runs out of 90:

| MR | Full pipeline | Both loops off | Pooling off | Jump loops off |
|---|---|---|---|---|
| MR2 | 90 | 0 | 18 | 90 |
| MR4 | 48 | 0 | 36 | 48 |
| MR7 | 24 | 0 | 0 | 24 |

- With both loops off, all 270/270 trials hold. So the violations come from the pipeline's feedback loops, not from the MR transforms or the checks.
- MR7 is entirely Pooling: leakage across ZIPs. It is 0/90 with Pooling off.
- MR4 is a genuine aggregation fault. Removing the top listing pushes buckets below the threshold and Pooling then borrows dearer neighbours. Example: `75216@2025-06` goes from $415,550 to $804,150.
- 36 of the MR4 runs still fail with Pooling off. The cause is not yet confirmed. Suspected: the Diagnoser.
- MR2 is mostly a badly designed MR, because duplicating listings changes the counts that the coverage threshold uses, so different buckets get pooled. There is also a real Diagnoser fault: its suspect list is capped at 8 and sorted by id, so 8 copies of one bad listing fill the list and a legitimate listing in the same ZIP is never judged.
- Bucket attribution (59 replays): 0 of 261 violating ZIP-months are a plain own-listing median. Pooled values explain MR2 133/150, MR4 46/46, MR7 65/65.

### B. MR10 under Qwen: why does a global price scaling break?
MR10 holds 90/90 offline but fails 64/90 runs (22/30 bases) with Qwen. Three extra runs isolate the cause:

| Run | Change | Violating runs | LLM calls |
|---|---|---|---|
| `mr_llm_rest` / `rest2` | none (normal Qwen run) | 64/90 | - |
| `mr_llm_mr10_noloops` | jump loops off (`jump_pct=1e9`) | 59/90 | 2,683 |
| `mr_llm_mr10_nopool` | jump loops and Pooling off (`min_listings=0`) | 0/90 | 0 |
| `mr_llm_ctrl_k1e-6` | k = 1.000001 control, 3 bases, seed 101 | 1/3 | 140 |

- Turning off the jump loops barely helps (64 to 59), so the Diagnoser and robust recompute are not the cause. My first hypothesis was wrong.
- Turning off Pooling too removes all violations (0/90, and 0 LLM calls, so the LLM is never consulted).
- The cause is that Qwen ranks neighbouring ZIPs differently for the source and the follow-up in the Scanner/Pooling step.
- The control confirms it. Scaling prices by 1.000001 is effectively a no-op, yet one of three bases still violates. Small changes in the prompt text alone are enough to change Qwen's choice. So this is LLM nondeterminism and prompt sensitivity, not a pipeline fault. It is a separate finding from the MR2/MR4/MR7 root causes.
- The control is only 3 bases with 1 seed, so it is suggestive and not a rate.

### C. Other observations
- Qwen breaks MR4 and MR7 more often than the stand-in. 957 of Qwen's 2,617 Pooling choices (36.6%) borrow ZIPs, while the rule pads months whenever that suffices (278 of 304 thin buckets). This fits the cross-ZIP leakage diagnosis.
- Seeds matter very little: in 89 of 90 (MR, base) pairs all three seeds gave the same outcome. One MR7 base was mixed.
- MR6 has one skipped run in both settings (89 runs rather than 90).
- Zero LLM failures across all Qwen runs. 12 invalid Qwen moves were rejected by the validator.
- Caveat: `mr_llm_rest2` was run with Python 3.10.12, pandas 2.3.3 and numpy 2.2.6, while `mr` and `mr_llm` used Python 3.12.9, pandas 2.2.3 and numpy 1.26.4. The overlapping MR6 rows matched exactly, so it did not change results.
