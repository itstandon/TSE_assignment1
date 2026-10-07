## 2025-10-07: Data snapshot
- Built with `python scripts/build_snapshot.py` (defaults): seed 20261016, window 2025-09..2026-08,
  calibration window 2021-11..2022-04, typo rate 0.4%, rural rate 0.3%, shrink k = 20.
- Inputs: Kaggle realtor-data.zip.csv (178,858,272 bytes), Zillow City ZHVI (middle tier, smoothed, SA).
- Output: 54,004 rows, 248 ZIPs, 111 cities; hashes recorded in data/snapshot/snapshot_manifest.json.
- Assumption: Kaggle `prev_sold_date` on sold rows treated as the sale date (inferred, ~3% effect if wrong).
- Known limits: city-level ZHVI trend shared across ZIPs; calibration uses 2021-22 ratios applied to 2026.