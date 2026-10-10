import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from housing_trends import Criteria
from housing_trends.agents.scanner import Snapshot, scan
from housing_trends.agents.statistician import (BucketOverride, Method, StatInstruction, compute,
                                                make_statistician, trimmed_mean)
from housing_trends.state import Scope

TRIM20 = Method(name="trimmed_mean", trim=0.2)


def snapshot(spec):
    """spec: {(zip, month): [(price, house_size), ...]} -> Snapshot with sequential listing ids."""
    rows, lid = [], 1
    for (z, m), items in spec.items():
        for price, size in items:
            rows.append(dict(listing_id=lid, zip_code=z, month=m, price=price, house_size=size, bed=3, bath=2.0))
            lid += 1
    return Snapshot(pd.DataFrame(rows))


def crit(zips=("75201",), start="2025-01", end="2025-03", **kw):
    return Criteria(zips=zips, start_month=start, end_month=end, **kw)


def stats(spec, c=None, scope=Scope(), instr=None, min_listings=1):
    snap = snapshot(spec)
    res = compute(scan(c or crit(), scope, snap, min_listings), snap, instr)
    return snap, res


def bucket(res, z, m):
    return res.buckets.set_index(["zip_code", "month"]).loc[(z, m)]


BASE = {("75201", "2025-01"): [(100, 10), (200, 20), (300, 30), (400, 40), (1000, 50)],   # ppsf 10,10,10,10,20
        ("75201", "2025-02"): [(110, 10)] * 5,
        ("75201", "2025-03"): [(99, 10)] * 5}


# ---- exact arithmetic ---------------------------------------------------------------------------
def test_median_price_and_ppsf_by_hand():
    _, r = stats(BASE)
    b = bucket(r, "75201", "2025-01")
    assert b.price == 300 and b.ppsf == 10 and b.n_used == 5 and b.n_own == 5 and b.n_pooled == 0

def test_trimmed_mean_by_hand():
    assert trimmed_mean(np.array([100, 200, 300, 400, 1000]), 0.2) == pytest.approx(300)   # drops 100 and 1000
    assert trimmed_mean(np.array([100, 200, 300, 400, 1000]), 0.0) == pytest.approx(400)   # plain mean
    assert trimmed_mean(np.array([100, 200, 300, 400, 1000]), 0.1) == pytest.approx(
        (0.5 * 100 + 200 + 300 + 400 + 0.5 * 1000) / 4)                                       # fractional edge weights

def test_trimmed_mean_instruction_is_applied():
    _, r = stats(BASE, instr=StatInstruction(method=TRIM20))
    b = bucket(r, "75201", "2025-01")
    assert b.price == pytest.approx(300) and b.method == "trimmed_mean(0.2)"

def test_median_of_medians_over_origin_months():
    spec = {("75201", "2025-01"): [(1, 1), (2, 1), (3, 1)], ("75201", "2025-02"): [(10, 1), (20, 1), (30, 1)],
            ("75201", "2025-03"): [(100, 1)]}
    c = crit(start="2025-02", end="2025-02")
    _, plain = stats(spec, c, Scope(time_pad=1), min_listings=100)         # bucket is thin, so it receives the scope
    _, mom = stats(spec, c, Scope(time_pad=1), StatInstruction(method=Method(name="median_of_medians")), min_listings=100)
    assert bucket(plain, "75201", "2025-02").price == 10          # median of 7 pooled values
    assert bucket(mom, "75201", "2025-02").price == 20            # median of origin medians 2, 20, 100
    assert bucket(plain, "75201", "2025-02").n_pooled == 4


# ---- instructions: exclusion and per-bucket overrides ----------------------------------------------
def test_excluding_a_listing_changes_only_buckets_that_contain_it():
    snap, base = stats(BASE)
    outlier = int(snap.df[snap.df.price == 1000].listing_id.iloc[0])
    _, r = stats(BASE, instr=StatInstruction(exclude_ids=(outlier,)))
    b = bucket(r, "75201", "2025-01")
    assert b.n_used == 4 and b.n_excluded == 1 and b.price == pytest.approx(250)        # median of 100..400
    assert bucket(r, "75201", "2025-02").n_excluded == 0
    assert bucket(r, "75201", "2025-02").price == bucket(base, "75201", "2025-02").price

def test_override_applies_to_one_bucket_only():
    ov = BucketOverride(zip_code="75201", month="2025-01", method=Method(name="trimmed_mean", trim=0.0))
    _, r = stats(BASE, instr=StatInstruction(overrides=(ov,)))
    assert bucket(r, "75201", "2025-01").price == pytest.approx(400) and bucket(r, "75201", "2025-01").method == "trimmed_mean(0)"
    assert bucket(r, "75201", "2025-02").method == "median"

def test_instruction_validation():
    with pytest.raises(ValidationError):
        Method(name="mode")
    with pytest.raises(ValidationError):
        Method(name="trimmed_mean", trim=0.5)
    o = BucketOverride(zip_code="75201", month="2025-01", method=Method())
    with pytest.raises(ValidationError):
        StatInstruction(overrides=(o, o))
    assert StatInstruction(exclude_ids=(3, 1, 3)).exclude_ids == (1, 3)


# ---- month-over-month ---------------------------------------------------------------------------------
def test_month_over_month_by_hand():
    _, r = stats(BASE)
    b = r.buckets.set_index("month")
    assert np.isnan(b.loc["2025-01", "mom_price_pct"])
    assert b.loc["2025-02", "mom_price_pct"] == pytest.approx((110 - 300) / 300 * 100)
    assert b.loc["2025-03", "mom_price_pct"] == pytest.approx(-10.0)
    assert b.loc["2025-03", "mom_count_pct"] == pytest.approx(0.0)

def test_empty_bucket_gives_nan_and_breaks_the_change_chain():
    spec = {("75201", "2025-01"): [(100, 10)], ("75201", "2025-03"): [(120, 10)]}        # no February listings
    _, r = stats(spec)
    b = r.buckets.set_index("month")
    assert b.loc["2025-02", "n_used"] == 0 and np.isnan(b.loc["2025-02", "price"])
    assert np.isnan(b.loc["2025-02", "mom_price_pct"]) and np.isnan(b.loc["2025-03", "mom_price_pct"])

def test_count_change_uses_own_listings_not_pooled():
    spec = {("75201", "2025-01"): [(100, 10)] * 10, ("75201", "2025-02"): [(100, 10)] * 5, ("75201", "2025-03"): [(100, 10)] * 5}
    _, r = stats(spec, scope=Scope(time_pad=1), min_listings=100)
    b = r.buckets.set_index("month")
    assert b.loc["2025-02", "mom_count_pct"] == pytest.approx(-50.0)                     # real drop is not smoothed away
    assert b.loc["2025-02", "n_pooled"] > 0


# ---- missing sizes, overall series ---------------------------------------------------------------------
def test_missing_house_size_only_affects_ppsf():
    spec = {("75201", "2025-01"): [(100, 10), (200, np.nan), (300, 0), (400, 20)]}
    _, r = stats(spec, crit(end="2025-01"))
    b = bucket(r, "75201", "2025-01")
    assert b.n_used == 4 and b.n_ppsf == 2 and b.price == 250 and b.ppsf == pytest.approx(15)   # ppsf 10 and 20

def test_overall_uses_own_listings_once_even_when_pooled():
    spec = {("75201", "2025-01"): [(100, 10)] * 3, ("75201", "2025-02"): [(100, 10)] * 3,
            ("75202", "2025-01"): [(200, 10)] * 3, ("75202", "2025-02"): [(200, 10)] * 3}
    c = crit(zips=("75201", "75202"), end="2025-02")
    _, narrow = stats(spec, c, Scope(), min_listings=100)
    _, wide = stats(spec, c, Scope(zip_extra=1, time_pad=1), min_listings=100)
    pd.testing.assert_frame_equal(narrow.overall, wide.overall)                          # pooling never leaks into overall
    assert list(narrow.overall.n) == [6, 6] and list(narrow.overall.price) == [150.0, 150.0]


# ---- properties the MRs will rely on --------------------------------------------------------------------
def spec_with(f):
    return {k: [f(p, s) for p, s in v] for k, v in BASE.items()}

@pytest.mark.parametrize("method", [Method(), TRIM20, Method(name="median_of_medians")])
def test_scaling_prices_scales_statistics_and_preserves_percent_changes(method):
    k = 3.5
    _, a = stats(BASE, instr=StatInstruction(method=method))
    _, b = stats(spec_with(lambda p, s: (p * k, s)), instr=StatInstruction(method=method))
    for col in ("price", "ppsf"):
        np.testing.assert_allclose(b.buckets[col], a.buckets[col] * k, rtol=1e-9)
        np.testing.assert_allclose(b.overall[col], a.overall[col] * k, rtol=1e-9)
    for col in ("mom_price_pct", "mom_ppsf_pct", "mom_count_pct"):
        np.testing.assert_allclose(b.buckets[col], a.buckets[col], rtol=1e-9, equal_nan=True)

@pytest.mark.parametrize("method", [Method(), Method(name="trimmed_mean", trim=0.1), TRIM20, Method(name="median_of_medians")])
def test_duplicating_every_listing_leaves_statistics_unchanged_and_doubles_counts(method):
    _, a = stats(BASE, instr=StatInstruction(method=method))
    _, b = stats({k: v * 2 for k, v in BASE.items()}, instr=StatInstruction(method=method))
    for col in ("price", "ppsf"):
        np.testing.assert_allclose(b.buckets[col], a.buckets[col], rtol=1e-9)
    assert (b.buckets.n_own == 2 * a.buckets.n_own).all()
    np.testing.assert_allclose(b.buckets.mom_price_pct, a.buckets.mom_price_pct, rtol=1e-9, equal_nan=True)

def test_row_order_does_not_matter():
    snap, a = stats(BASE, instr=StatInstruction(method=TRIM20))
    shuffled = Snapshot(snap.df.sample(frac=1, random_state=3).reset_index(drop=True))
    b = compute(scan(crit(), Scope(), shuffled, 1), shuffled, StatInstruction(method=TRIM20))
    pd.testing.assert_frame_equal(a.buckets, b.buckets)
    pd.testing.assert_frame_equal(a.overall, b.overall)

def test_compute_is_pure():
    snap = snapshot(BASE)
    s = scan(crit(), Scope(), snap, 1)
    before = (snap.df.copy(), s.members.copy(), s.buckets.copy())
    r1, r2 = compute(s, snap), compute(s, snap)
    pd.testing.assert_frame_equal(r1.buckets, r2.buckets)
    for x, y in zip(before, (snap.df, s.members, s.buckets)):
        pd.testing.assert_frame_equal(x, y)


# ---- graph node ---------------------------------------------------------------------------------------------
def test_node_uses_default_then_instruction():
    snap = snapshot(BASE)
    node = make_statistician(snap)
    s = scan(crit(), Scope(), snap, 1)
    out = node({"scan": s})
    assert out["stats"].instruction == StatInstruction() and out["trace"][0].startswith("statistician: method=median")
    out2 = node({"scan": s, "stat_instruction": StatInstruction(method=TRIM20, exclude_ids=(5,))})
    assert "trimmed_mean(0.2)" in out2["trace"][0] and "1 excluded" in out2["trace"][0]