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