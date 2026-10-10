# AI_USE.md

Team: Arushi Tandon, Bavya Sri Kilari. Period covered: [Oct 2026 dates].

## 1. Tools used

| Tool | Model / version | Used for |
|---|---|---|
| Claude (claude.ai chat) | [exact model name shown in the model selector] | Design discussion, writing and testing code, data analysis, drafting notes |
| gpt-oss-120b via an OpenAI-compatible hosted endpoint | `openai/gpt-oss-120b`, temperature 0, seed 42 | LLM nodes INSIDE the application: Parser Agent [and Anomaly Diagnoser, once built] |

Full chat transcripts are kept at: [link or folder].

## 2. What the AI did and what we did

| Component | AI role | Our role |
|---|---|---|
| Design note and architecture | Reviewed our draft; pointed out inconsistencies (agent counts, ZHVI not wired in) | [Wrote the original design, chose the loops, made final decisions] |
| Data snapshot (`build_snapshot.py`) | Wrote the generator and analysed the Kaggle and Zillow files | [Downloaded data, ran it, chose option B, Dallas-only, 24 months] |
| Input Handler, Parser Agent | Wrote the code and 12 tests | [Ran and tested against the hosted model] |
| Scanner, Statistician | Wrote the code and tests | [Ran them on the snapshot, reviewed the output] |
| [Critic, Pooling, Diagnoser, graph, harness, UI, report] | [fill in] | [fill in] |

## 3. How we checked AI output

- Unit tests (77 passing at the time of writing). The Scanner's counts and flags are compared with an independent brute-force oracle. The Statistician is tested against hand-computed values and properties (scaling, duplication, row-order invariance).
- Reproducibility: the snapshot builder is seeded and records SHA-256 hashes. A rerun on our laptop and a run in the AI's environment (different numpy/pandas versions) gave byte-identical files.
- Measurement on real data before accepting design claims (see section 5).
- [Anything else you did yourselves.]

## 4. LLMs inside the application

- Parser Agent: free text goes to the model, which returns a JSON spec. The output is treated as untrusted. The same deterministic validator as the form path checks it. On failure the agent gets the errors back once, then gives up and asks the user to use the form.
- Vague terms ("affordable" = snapshot median, "luxury" = 75th percentile) are rules in the prompt. They are our own definitions, not market facts.
- The wording of the model's `assumptions` varies between runs even at temperature 0 with a fixed seed. The validated criteria stayed identical.
- [Describe the Diagnoser and any caching of LLM replies when built.]

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

## 6. Claims we have not yet independently verified

- "Pooling neighbouring ZIPs distorts a median more than thinness does" was measured as the change from the unpooled median. For thin buckets part of that change is noise correction, not bias. [Re-test against the known generator ground truth before citing.]
- Sampling-noise figures (about ±13% for 5 listings, ±6.5% for 20) come from one ZIP (75241).
- Reading `prev_sold_date` on sold rows as the sale date is an inference.