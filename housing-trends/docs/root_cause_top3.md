# Root-cause analysis of the three most-violated MRs

Source: offline run `results/mr/summary.md` (10 MRs x 30 distinct base inputs x seeds 101, 202, 303; rule-based stand-in for
every LLM node). The three most-violated relations are:

| MR | Relation | Violating base inputs | Rate (95% Wilson CI) | Violating runs |
|---|---|---|---|---|
| MR2 | Multiply each ZIP's listings by a random N (1-10) | 30 / 30 | 100.0% (88.6-100.0) | 90 / 90 |
| MR4 | Remove the dearest listing of each ZIP | 16 / 30 | 53.3% (36.1-69.8) | 48 / 90 |
| MR7 | Double all prices in one ZIP | 13 / 30 | 43.3% (27.4-60.8) | 24 / 90 |

Evidence is reproducible with `python scripts/diagnose_top3.py` (outputs in `results/mr_diagnosis/`); single trials with
`python scripts/mr_replay.py <MR> <base_id> <seed>`.

## Method: two independent lines of evidence

**1. Ablation (switch the adaptive loops off, same bases and seeds).** `min_listings=0` disables the coverage loop
(Cycle A, Pooling); `jump_pct=1e9` disables the method and outlier loops (Cycles B and C: robust recompute, Diagnoser).

| MR | Full pipeline | Both loops off | Pooling off only | Jump loops off only |
|---|---|---|---|---|
| MR2 | 30/30 bases, 90/90 runs | **0/30, 0/90** | 10/30, 18/90 | 30/30, 90/90 |
| MR4 | 16/30, 48/90 | **0/30, 0/90** | 12/30, 36/90 | 16/30, 48/90 |
| MR7 | 13/30, 24/90 | **0/30, 0/90** | 0/30, 0/90 | 13/30, 24/90 |

**2. Bucket attribution.** One violating seed per violating base input (59 trials) was replayed. For every ZIP-month that
the relation flagged, we recorded whether it was pooled, recomputed with the trimmed mean, or had excluded listings, on
either side (`results/mr_diagnosis/attribution_buckets.csv`).

| MR | Violating ZIP-months | Pooled (src or fol) | Robust trimmed mean | Exclusions | Plain own-listing median |
|---|---|---|---|---|---|
| MR2 | 150 | 133 | 58 | 7 | **0** |
| MR4 | 46 | 46 | 17 | 5 | **0** |
| MR7 | 65 | 65 | 25 | 16 | **0** |

**What this rules out, for all three MRs.**
- *Harness bug:* with the loops off, the same transforms, checks and tolerances hold exactly on 270 / 270 trials. The
  transform, comparison and tolerance code is therefore correct on these inputs.
- *Data artifact:* the listings are identical in the full and ablated runs, and the plain median satisfies every relation
  on them. Not one violating bucket is a plain own-listing median.

What remains is the pipeline's adaptive control flow. The verdicts below separate a genuine fault from an ill-conceived MR.

## MR4: Remove the dearest listing. Verdict: genuine aggregation fault (coverage loop)

*Property.* Removing the maximum of a sample can never raise its median. The relation is mathematically sound for the
contract statistic.

*Evidence.* All 46 violating buckets are pooled. Pooling off and jump loops on leaves 36 violating runs (see the MR2 Diagnoser
mechanism); jump loops off and Pooling on reproduces all 48. Worst case: base 3, seed 101, `75216@2025-06`. The source
bucket has 0 own listings and 12 pooled ones (median $415,550). After the removal the follow-up borrows 28 neighbour listings
and reports a median of **$804,150 (+93.5%)**. A second case, base 1, `75203@2025-10`, goes from $369,706 to $714,674
(+93.3%).

*Mechanism.* Deleting a listing lowers some bucket's count below `MIN_LISTINGS = 20` or changes which buckets are thin.
Pooling then widens those buckets by different amounts (more months, or more and dearer neighbouring ZIPs). The reported
"ZIP-month median" becomes the median of a *different population*, and that population can be far more expensive.
Several violating buckets have 0 or 2 own listings, so the published number describes the neighbours, not the ZIP.

*Why it is a fault and not an ill-conceived MR.* The contract states that the bucket price is the ZIP-month's statistic.
An estimator that rises when the sample's maximum is removed is not monotone, and Pooling makes the population depend
discontinuously on a single listing. Possible fixes: keep an own-listing-only bucket statistic in the contract and report
the pooled estimate separately, or mark buckets with fewer than k own listings as low-confidence instead of publishing
a borrowed median.

## MR7: Double all prices in one ZIP. Verdict: genuine aggregation fault (cross-ZIP leakage via Pooling)

*Property.* A ZIP's statistics describe that ZIP's listings. Scaling one ZIP must scale that ZIP's prices and leave
every other ZIP unchanged.

*Evidence.* The cleanest ablation of the three: **Pooling off gives 0 / 90 violated runs**, and jump loops off with Pooling
on gives exactly 24 / 90, as in the full pipeline. All 65 violating buckets are pooled on both sides. Examples:
`75210@2025-08` (base 64, seed 202) has 2 own and 20 pooled listings, $299,050 becomes $620,407 (+107%). `75219@2025-03`
(base 43, seed 101) has 3 own and 18 pooled listings, $807,900 becomes $1,665,214 (+106%). In 25 of the 65 buckets the
follow-up also switched from median to trimmed mean. The doubled neighbour created a >30% jump that fired the robust
recompute, so the leakage also changes *which estimator* a neighbouring ZIP gets.

*Mechanism.* A thin bucket of ZIP A borrows listings from its nearest ZIPs. If neighbour B is doubled, A's median moves,
although nothing in A changed. Conversely, the doubled ZIP's own pooled buckets contain undoubled neighbour listings, so they
do not scale cleanly by 2.

*Contrast with MR10 (scale all prices by k).* Offline, MR10 holds on 0 / 90 runs: a global scale keeps pooled populations
proportional. MR7 is local, and MR7 fails only through Pooling. This pins the cause on cross-ZIP borrowing, not on a
scale-sensitive threshold (the catalog predicted this contrast). With Qwen, MR10 also fails (see the MR10 section
below), so the contrast holds only for the offline stand-in.

## MR2: Multiply each ZIP's listings by N. Verdict: mostly an ill-conceived MR, plus one genuine Diagnoser fault

*Property.* Replicating a sample leaves its empirical distribution, and so its median and trimmed mean, unchanged.
That holds for the **estimators**, but **not for the pipeline**: the coverage threshold is defined on counts.

**Part 1 (ill-conceived on thin buckets, about 89% of violating buckets).** 133 / 150 violating buckets are pooled. A ZIP copied
N times has N times the listings, so buckets that were thin in the source are healthy in the follow-up and are no longer pooled.
41 buckets are pooled only in the source, 88 on both sides with different depths. Example: base 26, seed 101,
`75201@2025-07` has 5 own and 17 pooled listings in the source (median $400,050) and 20 own and 0 pooled in the follow-up
(median $948,500). The relation should be restricted to buckets that are non-thin in the source **and** the follow-up,
or the threshold should be stated per unique listing. The catalog already flagged MR2 as partly ill-conceived for this
reason, so this result confirms the prediction.

**Part 2 (genuine fault: Diagnoser evidence selection is not duplicate-aware).** With Pooling off, MR2 still fails on
18 / 90 runs (10 / 30 bases). Every one of them used at least one Diagnoser round. Replay of base 0, seed 101
(`min_listings=0`, ZIP 75232 copied N = 9 times):

- *Source:* the Critic's suspect list (cap `MAX_SUSPECTS = 8`) for 75232 holds the $3,818,800 listing 4795 (13.6x the ZIP
  median, judged `bad_listing`) **and** listing 14998 ($95,800, judged `legit_rural`). Because 75232 contains a
  `legit_rural` listing, it keeps the robust trimmed mean (`diagnoser.py:153`). Its January value is **$276,242**.
- *Follow-up:* the 9 copies of the $3.8M listing all have the same, largest |z|, so **8 copies fill all 8 suspect slots**.
  The `legit_rural` listing is crowded out and never judged. 75232 now has only `bad_listing` verdicts, so the Diagnoser
  reverts it to the **plain median** (`diagnoser.py:154-157`; overrides drop from 168 to 156 to 144). January becomes
  **$267,700**, and every month of 75232 changes. The 9th copy of the bad listing is never excluded (`n_excluded = 8`).

The relation is correct here: the data distribution is identical, and the output changes only because the agent's
evidence is chosen by listing id under a fixed cap. This is a **genuine agent fault**. Possible fixes: de-duplicate
suspects by (ZIP, month, price, size) before applying the cap, or decide the ZIP's estimator from the share of
listings judged legitimate rather than from which listings happened to be sent. The 36 MR4 runs that fail with Pooling
off also go through the jump loops. Removing a ZIP's maximum can change which listings are suspects, and so whether a ZIP
keeps the trimmed mean. This is the likely cause, but it has **not yet been confirmed by a replay**.

## MR10 with Qwen only: unstable neighbour ranking in Pooling. Verdict: LLM robustness fault, not a code logic fault

Offline MR10 holds 0/90; with Qwen it is violated on 22/30 base inputs (64/90 runs), with errors of about 1-2%.

| Qwen run, MR10, 30 bases x 3 seeds | Violating runs | LLM calls |
|---|---|---|
| Full pipeline | 64/90 | - |
| Jump loops off (`jump_pct=1e9`) | 59/90 | 2,683 |
| Jump loops and Pooling off (`min_listings=0`) | **0/90** | 0 |

The Diagnoser is not the cause (loops off barely changes the rate). Pooling is necessary. Replay of base 1, seed 101,
loops off: the Scanner's Qwen neighbour ranking differs between source and x3 follow-up (75211, 75214, 75220 ordered
differently, 75223 pooled only in the follow-up), so different listings enter the medians.

Control: k = 1.000001 (prompt text changes, meaning does not), first 3 bases, seed 101: bases 0 and 5 hold, base 1 still
violates (n 139 vs 140). So prompt-text noise alone is sufficient, and real scale changes make it worse. Only 3 trials: it
does not estimate the noise share. Results: `results/mr_llm_mr10_noloops/`, `results/mr_llm_mr10_nopool/`,
`results/mr_llm_ctrl_k1e-6/`.

## Summary

| MR | Cause category | Loop responsible | Key evidence |
|---|---|---|---|
| MR4 | Genuine aggregation fault | Coverage (Pooling) | 46/46 violating buckets pooled; 0/90 with loops off; median +93.5% after removing the maximum |
| MR7 | Genuine aggregation fault (cross-ZIP leakage) | Coverage (Pooling) | 0/90 with Pooling off; 65/65 buckets pooled; MR10 (global scale) holds 0/90 |
| MR10 (Qwen only) | LLM instability in Pooling's neighbour ranking | Pooling | 0/90 offline; 64/90 Qwen; 59/90 loops off; 0/90 Pooling off; k=1.000001 control still violates 1 of 3 |
| MR2 | Ill-conceived MR on thin buckets **plus** genuine Diagnoser fault | Pooling (main), Diagnoser suspect cap | 133/150 pooled; with Pooling off 18/90 remain, all after a Diagnoser round; replay shows copies crowding out the `legit_rural` listing |

## Threats to validity of this diagnosis

- **Qwen results (final, `results/mr_llm/`):** MR2 30/30 bases (90/90 runs), MR4 25/30 (75/90), MR7 21/30 (61/90),
  0 errors. MR4 and MR7 are higher than offline. 36% of Qwen's Pooling choices borrow ZIPs, the move that causes the
  cross-ZIP leakage above.
- The ablations and replays use the deterministic offline stand-in. For the Qwen run (`results/mr_llm/`), the LLM
  chooses Pooling moves and Critic routes, so the *rates* can differ. The mechanisms are structural (count threshold,
  cross-ZIP borrowing, id-keyed suspect cap), and the first Qwen trials violate MR2 in the same ZIP-months (e.g.
  75202's January median at $300,500 in both).
- Attribution replays one violating seed per base input. With the stand-in all seeds of a base input share the same
  pipeline decisions, so other seeds differ only through the transform's random draws.
- "Pooled" is recorded per bucket on either side. It shows that pooling was involved, and the ablation shows it is
  necessary. Together they support a causal reading for MR7 and MR4. For MR2, both causes are present, and the share
  given to each comes from the pooling-off ablation.
