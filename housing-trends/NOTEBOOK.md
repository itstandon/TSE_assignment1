## 2025-10-07: Data snapshot
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

