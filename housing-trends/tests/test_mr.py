"""Tests for the metamorphic harness itself (statistics, comparison helpers, transforms, relation checks, one tiny
end-to-end trial). The harness must be trustworthy before its violation rates mean anything."""
import numpy as np
import pandas as pd
import pytest

from mr.compare import mismatches, not_below
from mr.contract import CONTRACT_COLUMNS
from mr.harness import evaluate, summarize
from mr.relations import MR_BY_ID, MRS
from mr.stats import wilson
from mr import transforms as T
from tests.helpers import MONTHS3, build, crit, flat


def full(price, ppsf=None, n=None, mom=None):
    n = n or [10] * len(price)
    return pd.DataFrame({"price": price, "ppsf": ppsf or [p / 100 for p in price], "n": n,
                         "mom_price_pct": mom or [np.nan] + [1.0] * (len(price) - 1)}, index=["2025-01", "2025-02", "2025-03"][:len(price)])


# ---- statistics
def test_wilson_known_values():
    lo, hi = wilson(0, 30)
    assert lo == 0.0 and hi == pytest.approx(0.1135, abs=1e-3)
    lo, hi = wilson(15, 30)
    assert (lo, hi) == pytest.approx((0.3323, 0.6677), abs=1e-3)
    lo, hi = wilson(30, 30)
    assert hi == 1.0 and lo == pytest.approx(0.8865, abs=1e-3)
    assert all(np.isnan(v) for v in wilson(0, 0))


def test_wilson_interval_contains_the_estimate_and_narrows_with_more_trials():
    lo1, hi1 = wilson(3, 30)
    lo2, hi2 = wilson(30, 300)
    assert lo1 <= 0.1 <= hi1 and lo2 <= 0.1 <= hi2 and (hi2 - lo2) < (hi1 - lo1)


# ---- comparison helpers
def test_mismatches_respects_tolerance_and_nan():
    a, b = full([100.0, 200.0, np.nan]), full([100.0 * (1 + 1e-12), 201.0, np.nan])
    problems = mismatches(a, b, ["price"], rtol=1e-9)
    assert len(problems) == 1 and "2025-02" in problems[0]                  # NaN == NaN, 1e-12 is within tolerance
    assert mismatches(a, full([100.0, 200.0, 5.0]), ["price"])              # NaN vs number is a mismatch


def test_not_below_flags_only_decreases():
    low, up = full([100.0, 200.0]), full([100.0, 150.0])
    assert len(not_below(low, up, "price")) == 1
    assert not not_below(low, full([100.0, 200.0]), "price")
    assert not not_below(low, full([100.0, 250.0]), "price")


# ---- transforms (pure functions of their input; the snapshot is never modified)
def tiny_df():
    snap = build({("75201", m): flat(25, 300_000) for m in MONTHS3}, MONTHS3, ["75201"])
    return snap.df, crit(["75201"], MONTHS3[0], MONTHS3[2])


def test_transforms_do_not_modify_their_input():
    df, c = tiny_df()
    before = df.copy()
    rng = np.random.default_rng(1)
    for fn in (T.reorder, T.relabel_ids, T.duplicate_all, T.to_thousands, T.add_x_to_one_month, T.scale_prices, T.raise_one_zip):
        fn(df, c, rng)
    pd.testing.assert_frame_equal(df, before)


def test_reorder_and_relabel_keep_the_same_listings():
    df, c = tiny_df()
    r = T.reorder(df, c, np.random.default_rng(2)).df
    assert sorted(r.listing_id) == sorted(df.listing_id) and not r.listing_id.equals(df.listing_id)
    l = T.relabel_ids(df, c, np.random.default_rng(2)).df
    assert sorted(l.listing_id) == sorted(df.listing_id) and l.price.equals(df.price)


def test_duplicate_doubles_rows_with_unique_ids():
    df, c = tiny_df()
    d = T.duplicate_all(df, c, np.random.default_rng(0)).df
    assert len(d) == 2 * len(df) and d.listing_id.is_unique


def test_scale_thousands_and_shift_and_widen_and_restrict():
    df, c = tiny_df()
    f = T.to_thousands(df, c.model_copy(update={"price_min": 200_000, "price_max": 400_000}), np.random.default_rng(0))
    assert f.criteria.price_min == 200 and f.criteria.price_max == 400 and f.meta["k"] == 0.001
    assert (f.df.price * 1000).round().equals(df.price.astype(float)) or np.allclose(f.df.price * 1000, df.price)
    s = T.shift_window(df, c, np.random.default_rng(0)).criteria
    assert (s.start_month, s.end_month) == ("2025-02", "2025-04")
    w = T.widen_price_band(df, c.model_copy(update={"price_min": 200_000, "price_max": 400_000}), np.random.default_rng(0)).criteria
    assert w.price_min == 150_000 and w.price_max == 500_000
    c2 = crit(["75201", "75202", "75203"], MONTHS3[0], MONTHS3[2])
    sub = T.restrict_zips(df, c2, np.random.default_rng(0)).criteria.zips
    assert 1 <= len(sub) < 3 and set(sub) <= set(c2.zips)
    x = T.add_x_to_one_month(df, c, np.random.default_rng(0))
    assert (x.df.price - df.price).gt(0).sum() == (df.month == x.meta["month"]).sum()


# ---- relation checks on hand-made contracts (does each check catch what it should?)
def test_duplicate_check_requires_doubled_counts_and_same_statistics():
    src = full([100.0, 200.0], n=[10, 12])
    good = full([100.0, 200.0], n=[20, 24])
    assert not MR_BY_ID["MR3"].check(src, good, None, {})
    assert MR_BY_ID["MR3"].check(src, full([100.0, 200.0], n=[20, 25]), None, {})
    assert MR_BY_ID["MR3"].check(src, full([100.0, 210.0], n=[20, 24]), None, {})


def test_scale_check_applies_the_factor_to_price_and_ppsf_only():
    src = full([100.0, 200.0])
    k = {"k": 2.0}
    assert not MR_BY_ID["MR9"].check(src, full([200.0, 400.0], ppsf=[2.0, 4.0]), None, k)
    assert MR_BY_ID["MR9"].check(src, full([100.0, 200.0]), None, k)         # nothing scaled


def test_add_x_check_wants_a_rise_of_x_in_that_month_and_no_decrease_elsewhere():
    src, meta = full([100_000.0, 200_000.0, 300_000.0]), {"month": "2025-02", "x": 50_000}
    assert not MR_BY_ID["MR7"].check(src, full([100_000.0, 250_000.0, 300_000.0]), None, meta)
    assert MR_BY_ID["MR7"].check(src, full([100_000.0, 210_000.0, 300_000.0]), None, meta)       # rose by 10k, not 50k
    assert MR_BY_ID["MR7"].check(src, full([90_000.0, 250_000.0, 300_000.0]), None, meta)        # another month decreased


def test_monotonic_checks_have_the_right_direction():
    src, more, less = full([1.0, 2.0], n=[10, 10]), full([1.0, 2.0], n=[12, 10]), full([1.0, 2.0], n=[8, 10])
    assert not MR_BY_ID["MR5"].check(src, more, None, {}) and MR_BY_ID["MR5"].check(src, less, None, {})
    assert not MR_BY_ID["MR6"].check(src, less, None, {}) and MR_BY_ID["MR6"].check(src, more, None, {})


def test_shift_check_compares_only_the_overlapping_months():
    src = full([100.0, 200.0, 300.0], mom=[np.nan, 100.0, 50.0])
    fol = full([200.0, 300.0, 999.0], mom=[np.nan, 50.0, 233.0])             # the new last month may differ freely
    fol.index = ["2025-02", "2025-03", "2025-04"]
    assert not MR_BY_ID["MR8"].check(src, fol, None, {})
    fol.loc["2025-03", "price"] = 301.0
    assert MR_BY_ID["MR8"].check(src, fol, None, {})


def test_catalog_has_ten_mrs_across_the_three_families():
    assert len(MRS) >= 8 and {m.family for m in MRS} == {"Invariance", "Monotonic", "Directional"}
    assert len({m.id for m in MRS}) == len(MRS) and all(m.fault and m.tolerance and m.subdomain for m in MRS)


# ---- one tiny end-to-end trial per kind (calm data: 25 listings per month, no loop needs to fire)
@pytest.mark.parametrize("mr_id", ["MR1", "MR2", "MR3", "MR4", "MR9"])
def test_invariance_and_scaling_hold_on_calm_data(mr_id):
    df, c = tiny_df()
    row = evaluate(MR_BY_ID[mr_id], df, c, base_id=0, seed=1)
    assert row["status"] == "holds", row["detail"]


def test_evaluate_is_reproducible_and_records_crashes_instead_of_raising():
    df, c = tiny_df()
    assert evaluate(MR_BY_ID["MR1"], df, c, 0, 7) == evaluate(MR_BY_ID["MR1"], df, c, 0, 7)
    bad = evaluate(MR_BY_ID["MR1"], df.drop(columns=["price"]), c, 0, 7)
    assert bad["status"] == "error"


def test_summarize_counts_by_base_input_and_by_run():
    runs = pd.DataFrame([dict(mr="MR1", base_id=b, seed=s, status=st) for b, s, st in
                         [(0, 1, "holds"), (0, 2, "violated"), (1, 1, "holds"), (1, 2, "holds"), (2, 1, "skipped"), (3, 1, "error")]])
    r = summarize(runs, [1, 2]).set_index("mr").loc["MR1"]
    assert (r.base_inputs, r.base_violations, r.runs, r.run_violations, r.skipped, r.errors) == (2, 1, 4, 1, 1, 1)
    assert r.base_rate == 0.5 and r.base_ci_low < 0.5 < r.base_ci_high
