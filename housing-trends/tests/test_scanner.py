import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from housing_trends import Criteria
from housing_trends.agents.scanner import (Snapshot, active_filters, filter_mask, make_scanner,
                                           relaxable_filters, scan)
from housing_trends.context import month_range
from housing_trends.state import MAX_TIME_PAD, MAX_ZIP_EXTRA, Scope

ZIPS = ["75201", "75202", "75203", "75204", "75210", "75215", "75220", "75230", "75240"]
MONTHS = month_range("2025-01", "2025-08")
MIN = 6          # small threshold so the tiny fixture has BOTH thin and healthy buckets


@pytest.fixture(scope="module")
def snap():
    rows, lid = [], 1
    for i, z in enumerate(ZIPS):
        for j, m in enumerate(MONTHS):
            for k in range(((i + 1) * (j % 3 + 1) % 7) * 2):               # 0..12 per bucket
                rows.append(dict(listing_id=lid, zip_code=z, month=m, price=200_000 + 1000 * lid,
                                 bed=2 + k % 3, bath=2.0, house_size=2000 if (i + k) % 2 == 0 else 1000))
                lid += 1
    df = pd.DataFrame(rows)
    df.loc[df.sample(15, random_state=1).index, "bed"] = np.nan           # missing values
    return Snapshot(df)


def crit(**kw):
    base = dict(zips=("75201", "75202", "75215"), start_month="2025-02", end_month="2025-06")
    return Criteria(**{**base, **kw})


# ---- brute-force oracle: plain pandas filtering, no matrices -----------------------------------
def oracle(snap, c, z, m, extra, pad, dropped):
    zs = [z] + snap.neighbours(z, extra)
    mi = snap.months.index(m)
    ms = snap.months[max(0, mi - pad): mi + pad + 1]
    d = snap.df[snap.df.zip_code.isin(zs) & snap.df.month.isin(ms)]
    return int(filter_mask(d, c, [f for f in active_filters(c) if f not in dropped]).sum())


def effective(snap, c, z, m, scope):
    """The rule under test: only buckets thin at base (no pooling, all filters) receive the scope."""
    return scope if oracle(snap, c, z, m, 0, 0, ()) < MIN else Scope()


SCOPES = [Scope(), Scope(time_pad=1), Scope(zip_extra=2), Scope(zip_extra=4, time_pad=2),
          Scope(time_pad=3, relaxed=("sqft",)), Scope(zip_extra=6, time_pad=3, relaxed=("sqft",))]
CRITERIA = [crit(), crit(sqft_min=1500, beds_min=3)]


@pytest.mark.parametrize("c", CRITERIA)
@pytest.mark.parametrize("scope", SCOPES)
def test_counts_match_oracle(snap, c, scope):
    for r in scan(c, scope, snap, MIN).buckets.itertuples():
        e = effective(snap, c, r.zip_code, r.month, scope)
        assert r.n == oracle(snap, c, r.zip_code, r.month, e.zip_extra, e.time_pad, e.relaxed)


@pytest.mark.parametrize("c", CRITERIA)
@pytest.mark.parametrize("scope", SCOPES)
def test_flags_equal_what_each_pooling_move_really_adds(snap, c, scope):
    """The criteria: every gain must equal what the corresponding rescan would add."""
    relax_all = [f for f in active_filters(c) if f not in c.protected_filters]
    for r in scan(c, scope, snap, MIN).buckets.itertuples():
        e = effective(snap, c, r.zip_code, r.month, scope)
        n = oracle(snap, c, r.zip_code, r.month, e.zip_extra, e.time_pad, e.relaxed)
        assert r.gain_time == oracle(snap, c, r.zip_code, r.month, e.zip_extra, MAX_TIME_PAD, e.relaxed) - n
        assert r.gain_zip == oracle(snap, c, r.zip_code, r.month, MAX_ZIP_EXTRA, e.time_pad, e.relaxed) - n
        assert r.gain_filters == oracle(snap, c, r.zip_code, r.month, e.zip_extra, e.time_pad, relax_all) - n
        assert r.potential == oracle(snap, c, r.zip_code, r.month, MAX_ZIP_EXTRA, MAX_TIME_PAD, relax_all) - n
        assert r.potential >= max(r.gain_time, r.gain_zip, r.gain_filters)
        assert r.exhaustive == (r.potential == 0)
        assert set(filter(None, r.moves.split(","))) == {k for k, g in (("time", r.gain_time), ("zip", r.gain_zip),
                                                                      ("filters", r.gain_filters)) if g > 0}


# ---- widen only what is thin ----------------------------------------------------------------------
def test_healthy_buckets_are_never_touched(snap):
    c = crit()
    b = scan(c, Scope(zip_extra=6, time_pad=3, relaxed=("sqft",)), snap, MIN).buckets
    healthy, thin = b[~b.widened], b[b.widened]
    assert len(healthy) > 0 and len(thin) > 0                      # the test is not vacuous
    assert (healthy.n_base >= MIN).all() and (thin.n_base < MIN).all()
    assert (healthy.n == healthy.n_own).all() and (healthy.n_pooled == 0).all()
    assert (healthy.n == healthy.n_base).all()

def test_healthy_bucket_members_are_only_its_own_listings(snap):
    res = scan(crit(), Scope(zip_extra=6, time_pad=3), snap, MIN)
    healthy = res.buckets[~res.buckets.widened][["zip_code", "month"]]
    mem = res.members.merge(healthy, on=["zip_code", "month"])
    assert len(mem) > 0 and mem.own.all()

def test_widening_never_changes_a_buckets_widened_status(snap):
    c = crit()
    flags = [scan(c, s, snap, MIN).buckets.widened.to_numpy() for s in SCOPES]
    assert all((f == flags[0]).all() for f in flags)

def test_scope_widening_is_monotone(snap):
    c = crit()
    ns = [scan(c, s, snap, MIN).buckets.n.to_numpy() for s in
          (Scope(), Scope(time_pad=1), Scope(time_pad=2), Scope(time_pad=2, zip_extra=2), Scope(time_pad=3, zip_extra=6))]
    assert all((b >= a).all() for a, b in zip(ns, ns[1:]))

def test_fully_widened_scope_is_exhaustive_for_thin_buckets(snap):
    c = crit(sqft_min=1500)
    b = scan(c, Scope(zip_extra=MAX_ZIP_EXTRA, time_pad=MAX_TIME_PAD, relaxed=("sqft",)), snap, MIN).buckets
    t = b[b.widened]
    assert len(t) > 0 and t.exhaustive.all() and (t.potential == 0).all() and (t.moves == "").all()


# ---- Pooling's rules: protected filters, nothing relaxable -----------------------------------------
def test_protected_filters_cannot_be_relaxed(snap):
    with pytest.raises(ValueError, match="protected"):
        scan(crit(price_max=300_000), Scope(relaxed=("price",)), snap, MIN)

def test_user_can_unprotect_a_filter_making_it_relaxable(snap):
    c = crit(price_max=300_000, protected_filters=())
    assert relaxable_filters(c, Scope()) == ["price"]
    assert scan(c, Scope(relaxed=("price",)), snap, 10_000).buckets.n.sum() > scan(c, Scope(), snap, 10_000).buckets.n.sum()

def test_no_relaxable_filter_means_zero_filter_gain(snap):
    b = scan(crit(price_max=500_000, beds_min=2), Scope(), snap, MIN).buckets      # both protected by default
    assert (b.gain_filters == 0).all()

def test_relaxed_filter_is_no_longer_offered(snap):
    c = crit(sqft_min=1500)
    assert relaxable_filters(c, Scope()) == ["sqft"] and relaxable_filters(c, Scope(relaxed=("sqft",))) == []
    b = scan(c, Scope(relaxed=("sqft",)), snap, MIN).buckets
    assert (b[b.widened].gain_filters == 0).all()          # already relaxed where the scope was applied
    assert (b[~b.widened].gain_filters > 0).any()          # healthy buckets keep their filter, so it is still 'available' there

def test_scope_limits_are_enforced():
    for bad in (dict(time_pad=MAX_TIME_PAD + 1), dict(zip_extra=MAX_ZIP_EXTRA + 1), dict(relaxed=("colour",)), dict(zip_extra=-1)):
        with pytest.raises(ValidationError):
            Scope(**bad)


# ---- time / ZIP pooling mechanics --------------------------------------------------------------------
def test_time_padding_is_clipped_at_snapshot_edges(snap):
    c = crit(start_month="2025-01", end_month="2025-03")
    b = scan(c, Scope(time_pad=2), snap, MIN).buckets.set_index(["zip_code", "month"])
    z = "75201"
    assert b.loc[(z, "2025-01"), "widened"]
    assert b.loc[(z, "2025-01"), "n"] == int(((snap.df.zip_code == z) & snap.df.month.isin(MONTHS[:3])).sum())  # months 01..03

def test_zip_neighbours_are_numerically_nearest(snap):
    assert snap.neighbours("75201", 3) == ["75202", "75203", "75204"]
    assert snap.neighbours("75210", 2) == ["75215", "75204"]       # distances 5 and 6

def test_own_vs_pooled_split(snap):
    c = crit()
    b = scan(c, Scope(zip_extra=2, time_pad=1), snap, MIN).buckets
    assert (b.n == b.n_own + b.n_pooled).all() and (b.n_pooled >= 0).all()
    assert (b.n_own.to_numpy() == scan(c, Scope(), snap, MIN).buckets.n.to_numpy()).all()   # own count ignores pooling

def test_members_agree_with_counts_and_own_flag(snap):
    res = scan(crit(sqft_min=1500), Scope(zip_extra=2, time_pad=1), snap, MIN)
    got = res.buckets.set_index(["zip_code", "month"])
    per = res.members.groupby(["zip_code", "month"]).size()
    assert (per.reindex(got.index, fill_value=0) == got.n).all()
    own = res.members[res.members.own].groupby(["zip_code", "month"]).size()
    assert (own.reindex(got.index, fill_value=0) == got.n_own).all()

def test_missing_values_do_not_match_an_active_filter(snap):
    c = crit(beds_min=0)
    d = snap.df[snap.df.zip_code.isin(c.zips) & snap.df.month.isin(month_range(c.start_month, c.end_month))]
    assert scan(c, Scope(), snap, MIN).buckets.n.sum() == int(d.bed.notna().sum())


# ---- reachability / thresholds -----------------------------------------------------------------------
def test_reachable_reflects_the_threshold(snap):
    c = crit()
    assert not scan(c, Scope(), snap, min_listings=10_000).buckets.reachable.any()
    lo = scan(c, Scope(), snap, min_listings=1).buckets
    assert (lo.reachable == ((lo.n + lo.potential) >= 1)).all() and (lo.thin == (lo.n < 1)).all()


# ---- determinism / metamorphic sanity ------------------------------------------------------------------
def test_row_order_does_not_matter(snap):
    c = crit(sqft_min=1500)
    shuffled = Snapshot(snap.df.sample(frac=1, random_state=7).reset_index(drop=True))
    a, b = scan(c, SCOPES[3], snap, MIN), scan(c, SCOPES[3], shuffled, MIN)
    pd.testing.assert_frame_equal(a.buckets, b.buckets)
    pd.testing.assert_frame_equal(a.members, b.members)

def test_node_returns_scan_and_trace(snap):
    out = make_scanner(snap, MIN)({"criteria": crit()})
    assert out["scan"].scope == Scope() and out["trace"][0].startswith("scanner:")

def test_rejects_zip_or_window_outside_snapshot(snap):
    with pytest.raises(ValueError, match="not in snapshot"):
        scan(crit(zips=("99999",)), Scope(), snap, MIN)
    with pytest.raises(ValueError, match="outside snapshot"):
        scan(crit(end_month="2026-12"), Scope(), snap, MIN)


# ---- per-bucket depth -----------------------------------------------------------------------------------
def test_per_bucket_depth_applies_to_that_bucket_only_and_matches_oracle(snap):
    from housing_trends.state import BucketScope
    c = crit()
    thin = scan(c, Scope(), snap, MIN).buckets
    t = thin[thin.widened].iloc[0]
    sc = Scope(per_bucket=(BucketScope(zip_code=t.zip_code, month=t.month, time_pad=2, zip_extra=2),))
    b = scan(c, sc, snap, MIN).buckets.set_index(["zip_code", "month"])
    row = b.loc[(t.zip_code, t.month)]
    assert (row.time_pad_used, row.zip_extra_used) == (2, 2)
    assert row.n == oracle(snap, c, t.zip_code, t.month, 2, 2, ())
    others = b.drop(index=(t.zip_code, t.month))
    assert (others.time_pad_used == 0).all() and (others.zip_extra_used == 0).all()        # default depth is 0

def test_per_bucket_entry_is_ignored_for_healthy_buckets(snap):
    from housing_trends.state import BucketScope
    c = crit()
    base = scan(c, Scope(), snap, MIN).buckets
    h = base[~base.widened].iloc[0]
    sc = Scope(per_bucket=(BucketScope(zip_code=h.zip_code, month=h.month, time_pad=3, zip_extra=6),))
    row = scan(c, sc, snap, MIN).buckets.set_index(["zip_code", "month"]).loc[(h.zip_code, h.month)]
    assert row.n == h.n_base and row.n_pooled == 0 and row.time_pad_used == 0

def test_scope_depth_lookup_and_duplicate_rejection():
    from housing_trends.state import BucketScope
    sc = Scope(zip_extra=1, per_bucket=(BucketScope(zip_code="75201", month="2025-02", time_pad=3),))
    assert sc.depth("75201", "2025-02") == (0, 3) and sc.depth("75202", "2025-02") == (1, 0)
    with pytest.raises(ValidationError):
        Scope(per_bucket=(BucketScope(zip_code="75201", month="2025-02"), BucketScope(zip_code="75201", month="2025-02", time_pad=1)))