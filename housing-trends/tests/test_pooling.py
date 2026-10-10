import pandas as pd
import pytest

from housing_trends import Criteria
from housing_trends.agents.pooling import make_pooling, plan_widening
from housing_trends.agents.scanner import Snapshot, scan
from housing_trends.context import month_range
from housing_trends.state import MAX_TIME_PAD, MAX_ZIP_EXTRA, Scope

MIN = 20
ZIPS = ["75201", "75202", "75203", "75204", "75205", "75206", "75207", "75208"]
MONTHS = month_range("2025-01", "2025-12")


def snap_from(counts):
    """counts: {(zip, month): n} -> Snapshot with declared coverage (empty ZIP-months allowed)."""
    rows, lid = [], 1
    for (z, m), k in counts.items():
        for i in range(k):
            rows.append(dict(listing_id=lid, zip_code=z, month=m, price=300_000 + lid, bed=3, bath=2.0,
                             house_size=1500 if i % 2 == 0 else 1000))
            lid += 1
    df = pd.DataFrame(rows, columns=["listing_id", "zip_code", "month", "price", "bed", "bath", "house_size"])
    return Snapshot(df, months=MONTHS, zips=ZIPS)


def crit(zips, start="2025-04", end="2025-06", **kw):
    return Criteria(zips=tuple(zips), start_month=start, end_month=end, **kw)


def uniform(n_by_zip_month):
    return {(z, m): n_by_zip_month(z, m) for z in ZIPS for m in MONTHS}


def test_no_plan_when_nothing_is_thin():
    snap = snap_from(uniform(lambda z, m: 25))
    c = crit(["75201"])
    assert plan_widening(c, Scope(), scan(c, Scope(), snap, MIN), MIN) is None

def test_time_is_chosen_when_it_can_suffice_alone():
    snap = snap_from(uniform(lambda z, m: 8))                    # 8 per month: +-1 month gives 24 >= 20
    c = crit(["75201"])
    p = plan_widening(c, Scope(), scan(c, Scope(), snap, MIN), MIN)
    assert p.counts == {"time": 3, "zip": 0, "filters": []}
    assert all(b.time_pad == 1 and b.zip_extra == 0 for b in p.scope.per_bucket)

def test_zip_is_chosen_when_time_cannot_suffice():
    snap = snap_from({**uniform(lambda z, m: 0), **{(z, m): 4 for z in ZIPS for m in MONTHS if z != "75201"}})
    # 75201 has no listings of its own, but neighbours do; padding months of an empty ZIP adds nothing
    c = crit(["75201"])
    p = plan_widening(c, Scope(), scan(c, Scope(), snap, MIN), MIN)
    assert p.counts["time"] == 0 and p.counts["zip"] == 3
    assert all(b.zip_extra == 2 for b in p.scope.per_bucket)

def test_each_bucket_gets_only_the_depth_it_needs():
    # 75201: 8 per month, so one extra month suffices. Other ZIPs: 2 per month, so even +-3 months gives only 14 < 20
    counts = uniform(lambda z, m: 8 if z == "75201" else 2)
    snap = snap_from(counts)
    c = crit(["75201", "75202"])
    p = plan_widening(c, Scope(), scan(c, Scope(), snap, MIN), MIN)
    got = {(b.zip_code, b.month): (b.zip_extra, b.time_pad) for b in p.scope.per_bucket}
    assert all(got[("75201", m)] == (0, 1) for m in ("2025-04", "2025-05", "2025-06"))        # time only
    assert all(got[("75202", m)] == (2, 0) for m in ("2025-04", "2025-05", "2025-06"))        # time cannot suffice -> ZIPs

def test_time_alone_is_preferred_when_the_maximum_padding_reaches_the_threshold():
    snap = snap_from(uniform(lambda z, m: 3))                    # 3 per month: +-3 months = 21 >= 20
    c = crit(["75201"])
    p = plan_widening(c, Scope(), scan(c, Scope(), snap, MIN), MIN)
    assert p.counts["zip"] == 0 and p.counts["time"] == 3

def test_relaxable_filter_is_relaxed_before_zip_but_protected_one_never():
    snap = snap_from(uniform(lambda z, m: 2))                    # 1 of 2 passes sqft>=1500: even +-3 months gives 7 < 20
    c = crit(["75201"], sqft_min=1500)
    s0 = scan(c, Scope(), snap, MIN)
    p = plan_widening(c, Scope(), s0, MIN)
    assert p.counts["filters"] == ["sqft"] and "sqft" in p.scope.relaxed
    c2 = crit(["75201"], price_max=10**9)                        # price is protected by default
    p2 = plan_widening(c2, Scope(), scan(c2, Scope(), snap, MIN), MIN)
    assert p2 is None or "price" not in p2.scope.relaxed

def test_unreachable_buckets_are_not_planned():
    snap = snap_from(uniform(lambda z, m: 0))
    c = crit(["75201"])
    s0 = scan(c, Scope(), snap, MIN)
    assert s0.buckets.thin.all() and not s0.buckets.reachable.any()
    assert plan_widening(c, Scope(), s0, MIN) is None

def test_plan_never_exceeds_limits_and_stops_when_exhausted():
    snap = snap_from(uniform(lambda z, m: 1))
    c = crit(["75201"])
    scope = Scope()
    for _ in range(12):                                          # far more rescans than the cap
        p = plan_widening(c, scope, scan(c, scope, snap, MIN), MIN)
        if p is None:
            break
        scope = p.scope
        assert all(b.time_pad <= MAX_TIME_PAD and b.zip_extra <= MAX_ZIP_EXTRA for b in scope.per_bucket)
    else:
        pytest.fail("Pooling never ran out of moves")

def test_repeated_planning_is_monotone_and_reduces_thin_buckets():
    snap = snap_from(uniform(lambda z, m: 5))
    c = crit(["75201", "75202"])
    scope, thin_counts = Scope(), []
    for _ in range(3):
        s = scan(c, scope, snap, MIN)
        thin_counts.append(int(s.buckets.thin.sum()))
        p = plan_widening(c, scope, s, MIN)
        if p is None:
            break
        scope = p.scope
    assert thin_counts == sorted(thin_counts, reverse=True)

def test_node_bumps_counter_and_writes_scope():
    snap = snap_from(uniform(lambda z, m: 8))
    c = crit(["75201"])
    out = make_pooling(MIN)({"criteria": c, "scan": scan(c, Scope(), snap, MIN), "rescans": 1})
    assert out["rescans"] == 2 and out["scope"].per_bucket and out["trace"][0].startswith("pooling: rescan 2")