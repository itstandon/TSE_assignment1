"""Shared builders for agent tests: tiny snapshots with hand-chosen prices, so every scenario is explicit."""
import pandas as pd

from housing_trends import Criteria
from housing_trends.agents.scanner import Snapshot, scan
from housing_trends.agents.statistician import StatInstruction, compute
from housing_trends.context import month_range
from housing_trends.state import Scope

NORMAL = dict(size=1500, acre=0.2, bed=3, bath=2.0)
RURAL = dict(size=800, acre=20.0, bed=2, bath=1.0)


def build(spec, months, zips):
    """spec: {(zip, month): [price | (price, attrs), ...]} -> Snapshot with declared coverage."""
    rows, lid = [], 1
    for (z, m), items in spec.items():
        for it in items:
            price, a = (it, NORMAL) if not isinstance(it, tuple) else it
            rows.append(dict(listing_id=lid, zip_code=z, month=m, price=price, house_size=a["size"], acre_lot=a["acre"],
                             bed=a["bed"], bath=a["bath"]))
            lid += 1
    df = pd.DataFrame(rows, columns=["listing_id", "zip_code", "month", "price", "house_size", "acre_lot", "bed", "bath"])
    return Snapshot(df, months=months, zips=zips)


def flat(n, base=300_000):
    """n listings at nearly the same price (+-3%), deterministic; the spread scales with `base`."""
    return [round(base + base / 300 * (i % 10)) for i in range(n)]


def spread(n, base):
    """n listings spread evenly over +-30% around `base` (a realistic within-ZIP price spread)."""
    return [round(base * (0.7 + 0.6 * i / (n - 1))) for i in range(n)]


def crit(zips, start, end, **kw):
    return Criteria(zips=tuple(zips), start_month=start, end_month=end, **kw)


def state_for(snap, c, *, min_listings=5, scope=None, instr=None, **counters):
    """A pipeline state after one Scanner + Statistician pass (what the Critic sees)."""
    scope, instr = scope or Scope(), instr or StatInstruction()
    s = scan(c, scope, snap, min_listings)
    return {"criteria": c, "scope": scope, "scan": s, "stat_instruction": instr, "stats": compute(s, snap, instr),
            "rescans": 0, "recomputes": 0, "diag_rounds": 0, "diagnoses": [], "unresolved": [], "persistent": [], "trace": [], **counters}


MONTHS3 = month_range("2025-01", "2025-03")