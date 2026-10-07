"""Try the Scanner on the frozen snapshot and watch the coverage numbers change as the scope widens.
Usage: python scripts/try_scanner.py            # all ZIPs, default window (latest 12 months)
       python scripts/try_scanner.py 75241      # one ZIP, with a per-month table
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from housing_trends import Criteria
from housing_trends.agents.scanner import Snapshot, scan
from housing_trends.state import MIN_LISTINGS, Scope

snap = Snapshot.from_csv(ROOT / "data/snapshot/listings_dfw.csv")
zips = tuple(sys.argv[1:]) or tuple(snap.zips)
end = snap.months[-1]
start = str(pd.Period(end, "M") - 11)
c = Criteria(zips=zips, start_month=start, end_month=end)
print(f"snapshot {snap.months[0]}..{snap.months[-1]} | {len(snap.zips)} ZIPs | window {start}..{end} | threshold {MIN_LISTINGS}\n")

for scope in [Scope(), Scope(time_pad=1), Scope(time_pad=2), Scope(time_pad=3), Scope(time_pad=3, zip_extra=2)]:
    r = scan(c, scope, snap)
    print(r.summary())

if len(zips) == 1:
    print(f"\nZIP {zips[0]}, default scope, per month:")
    cols = ["month", "n_own", "thin", "gain_time", "gain_zip", "potential", "moves", "exhaustive", "reachable"]
    print(scan(c, Scope(), snap).buckets[cols].to_string(index=False))