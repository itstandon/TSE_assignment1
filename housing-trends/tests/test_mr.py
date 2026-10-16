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
    for fn in (T.reorder, T.sort_by_price, T.relabel_ids, T.duplicate_all, T.duplicate_random_n, T.add_fake_month, T.to_thousands,
               T.add_x_to_one_month, T.scale_prices, T.raise_one_zip, T.double_one_zip, T.remove_top_per_zip, T.inject_outlier,
               T.replace_month_with_previous):
        fn(df, c, rng)
    pd.testing.assert_frame_equal(df, before)


def test_reorder_and_relabel_keep_the_same_listings():
    df, c = tiny_df()
    r = T.reorder(df, c, np.random.default_rng(2)).df
    assert sorted(r.listing_id) == sorted(df.listing_id) and not r.listing_id.equals(df.listing_id)
    l = T.relabel_ids(df, c, np.random.default_rng(2)).df
    assert sorted(l.listing_id) == sorted(df.listing_id) and l.price.equals(df.price)


def test_sort_by_price_is_ascending_inside_each_zip_month_and_keeps_the_rows():
    df, c = tiny_df()
    d = T.sort_by_price(df.sample(frac=1, random_state=3), c, np.random.default_rng(0)).df
    assert sorted(d.listing_id) == sorted(df.listing_id)
    assert all(g.price.is_monotonic_increasing for _, g in d.groupby(["zip_code", "month"]))


def test_duplicate_doubles_rows_with_unique_ids():
    df, c = tiny_df()
    d = T.duplicate_all(df, c, np.random.default_rng(0)).df
    assert len(d) == 2 * len(df) and d.listing_id.is_unique


def test_duplicate_random_n_gives_each_zip_n_copies_with_unique_ids():
    snap = build({(z, m): flat(5, 300_000) for z in ("75201", "75202", "75203") for m in MONTHS3}, MONTHS3, ["75201", "75202", "75203"])
    f = T.duplicate_random_n(snap.df, crit(["75201"], MONTHS3[0], MONTHS3[2]), np.random.default_rng(4))
    n_of, d = f.meta["n_of"], f.df
    assert d.listing_id.is_unique and all(1 <= n <= T.MAX_COPIES for n in n_of.values())
    base = snap.df.groupby("zip_code").size()
    assert all(len(d[d.zip_code == z]) == base[z] * n_of[z] for z in n_of)


def test_fake_month_lies_outside_any_pooling_padding_and_is_10x():
    df, c = tiny_df()
    f = T.add_fake_month(df, c, np.random.default_rng(0))
    fake = f.meta["fake_month"]
    assert fake == "2025-09" and len(f.df) == len(df) + f.meta["rows"] and f.df.listing_id.is_unique
    new = f.df[f.df.month == fake]
    assert new.price.min() >= T.FAKE_FACTOR * df.price.min() * 0.999 and set(new.zip_code) <= set(c.zips)


def test_remove_top_deletes_one_listing_per_zip_and_it_is_the_dearest():
    snap = build({(z, m): flat(5, 300_000) for z in ("75201", "75202") for m in MONTHS3}, MONTHS3, ["75201", "75202"])
    c = crit(["75201", "75202"], MONTHS3[0], MONTHS3[2])
    f = T.remove_top_per_zip(snap.df, c, np.random.default_rng(0))
    assert len(f.df) == len(snap.df) - 2 and len(f.meta["removed"]) == 2
    for z in c.zips:
        assert f.df[f.df.zip_code == z].price.max() <= snap.df[snap.df.zip_code == z].price.max()
        assert snap.df[snap.df.zip_code == z].price.max() in snap.df[snap.df.listing_id.isin(f.meta["removed"])].price.values


def test_double_one_zip_and_inject_outlier_change_exactly_what_they_say():
    snap = build({(z, m): flat(5, 300_000) for z in ("75201", "75202") for m in MONTHS3}, MONTHS3, ["75201", "75202"])
    df, c = snap.df, crit(["75201", "75202"], MONTHS3[0], MONTHS3[2])
    f = T.double_one_zip(df, c, np.random.default_rng(1))
    z = f.meta["zip"]
    assert np.allclose(f.df[f.df.zip_code == z].price, 2 * df[df.zip_code == z].price)
    assert f.df[f.df.zip_code != z].price.equals(df[df.zip_code != z].price)
    o = T.inject_outlier(df, c, np.random.default_rng(1))
    changed = o.df[o.df.price != df.price]
    assert len(changed) == 1 and changed.listing_id.iloc[0] == o.meta["id"]
    assert changed.price.iloc[0] == T.OUTLIER_FACTOR * df.loc[changed.index[0], "price"]


def test_replace_month_copies_the_previous_month_and_drops_the_original():
    df, c = tiny_df()
    f = T.replace_month_with_previous(df, c, np.random.default_rng(0))
    m, p = f.meta["month"], f.meta["prev"]
    assert m != c.start_month and f.df.listing_id.is_unique and len(f.df) == len(df) - (df.month == m).sum() + (df.month == p).sum()
    assert sorted(f.df[f.df.month == m].price) == sorted(df[df.month == p].price)


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
def with_zips(c, rows, verdicts=()):
    """Attach a per ZIP-month table: rows = [(zip, month, price, n_own)]; ppsf = price/100, n_used = n_own, mom = 1."""
    t = pd.DataFrame([dict(price=p, ppsf=p / 100, n_own=n, n_used=n, mom_price_pct=1.0, zip_code=z, month=m) for z, m, p, n in rows],
                     index=[f"{z}@{m}" for z, m, _, _ in rows])
    c.attrs["buckets"], c.attrs["verdicts"] = t, list(verdicts)
    return c


def test_equal_check_also_looks_at_the_per_zip_rows():
    rows = [("75201", "2025-01", 100.0, 10), ("75201", "2025-02", 200.0, 10)]
    src = with_zips(full([100.0, 200.0]), rows)
    assert not MR_BY_ID["MR1"].check(src, with_zips(full([100.0, 200.0]), rows), None, {})
    assert MR_BY_ID["MR1"].check(src, with_zips(full([100.0, 200.0]), [rows[0], ("75201", "2025-02", 201.0, 10)]), None, {})


def test_random_n_check_scales_only_the_own_count():
    rows = [("75201", "2025-01", 100.0, 10), ("75202", "2025-01", 150.0, 10)]
    src, meta = with_zips(full([100.0]), rows), {"n_of": {"75201": 3, "75202": 1}}
    good = [("75201", "2025-01", 100.0, 30), ("75202", "2025-01", 150.0, 10)]
    assert not MR_BY_ID["MR2"].check(src, with_zips(full([100.0]), good), None, meta)
    assert MR_BY_ID["MR2"].check(src, with_zips(full([100.0]), [("75201", "2025-01", 100.0, 10), good[1]]), None, meta)
    assert MR_BY_ID["MR2"].check(src, with_zips(full([100.0]), [("75201", "2025-01", 120.0, 30), good[1]]), None, meta)


def test_remove_top_check_flags_only_increases():
    src = with_zips(full([1.0]), [("75201", "2025-01", 100.0, 10), ("75201", "2025-02", 100.0, 10)])
    ok = with_zips(full([1.0]), [("75201", "2025-01", 90.0, 9), ("75201", "2025-02", 100.0, 9)])
    bad = with_zips(full([1.0]), [("75201", "2025-01", 90.0, 9), ("75201", "2025-02", 130.0, 9)])
    assert not MR_BY_ID["MR4"].check(src, ok, None, {}) and len(MR_BY_ID["MR4"].check(src, bad, None, {})) == 1


def test_zip_doubled_check_scales_one_zip_and_leaves_the_rest():
    rows = [("75201", "2025-01", 100.0, 10), ("75202", "2025-01", 150.0, 10)]
    src, meta = with_zips(full([100.0]), rows), {"zip": "75201", "factor": 2.0}
    assert not MR_BY_ID["MR7"].check(src, with_zips(full([100.0]), [("75201", "2025-01", 200.0, 10), rows[1]]), None, meta)
    assert MR_BY_ID["MR7"].check(src, with_zips(full([100.0]), [("75201", "2025-01", 200.0, 10), ("75202", "2025-01", 160.0, 10)]), None, meta)
    assert MR_BY_ID["MR7"].check(src, with_zips(full([100.0]), rows), None, meta)                    # nothing doubled


def test_outlier_check_limits_the_move_and_demands_the_planted_listing_among_the_suspects():
    rows = [("75201", "2025-01", 100.0, 10)]
    src, meta = with_zips(full([100.0, 200.0]), rows), {"zip": "75201", "id": 7, "month": "2025-01"}
    assert not MR_BY_ID["MR8"].check(src, with_zips(full([105.0, 200.0]), [("75201", "2025-01", 105.0, 10)], [(7, "75201", "bad_listing")]), None, meta)
    assert not MR_BY_ID["MR8"].check(src, with_zips(full([105.0, 200.0]), [("75201", "2025-01", 105.0, 10)]), None, meta)   # Diagnoser never ran: fine
    assert MR_BY_ID["MR8"].check(src, with_zips(full([130.0, 200.0]), [("75201", "2025-01", 105.0, 10)]), None, meta)       # overall moved 30%
    assert MR_BY_ID["MR8"].check(src, with_zips(full([105.0, 200.0]), [("75201", "2025-01", 130.0, 10)]), None, meta)       # ZIP-month moved 30%
    assert MR_BY_ID["MR8"].check(src, with_zips(full([105.0, 200.0]), [("75201", "2025-01", 105.0, 10)], [(9, "75201", "bad_listing")]), None, meta)


def test_month_copied_check_wants_equal_statistics_and_zero_change():
    meta = {"month": "2025-02", "prev": "2025-01"}
    good = full([100.0, 100.0, 300.0], n=[10, 10, 12], mom=[np.nan, 0.0, 200.0])
    assert not MR_BY_ID["MR9"].check(None, good, None, meta)
    assert MR_BY_ID["MR9"].check(None, full([100.0, 101.0, 300.0], n=[10, 10, 12], mom=[np.nan, 1.0, 200.0]), None, meta)
    assert MR_BY_ID["MR9"].check(None, full([100.0, 100.0, 300.0], n=[10, 11, 12], mom=[np.nan, 0.0, 200.0]), None, meta)


def test_scale_check_applies_the_factor_to_price_and_ppsf_only():
    src = full([100.0, 200.0])
    k = {"k": 2.0}
    assert not MR_BY_ID["MR10"].check(src, full([200.0, 400.0], ppsf=[2.0, 4.0]), None, k)
    assert MR_BY_ID["MR10"].check(src, full([100.0, 200.0]), None, k)         # nothing scaled


def test_add_x_check_wants_a_rise_of_x_in_that_month_and_no_decrease_elsewhere():
    src, meta = full([100_000.0, 200_000.0, 300_000.0]), {"month": "2025-02", "x": 50_000}
    assert not MR_BY_ID["MR3"].check(src, full([100_000.0, 250_000.0, 300_000.0]), None, meta)
    assert MR_BY_ID["MR3"].check(src, full([100_000.0, 210_000.0, 300_000.0]), None, meta)       # rose by 10k, not 50k
    assert MR_BY_ID["MR3"].check(src, full([90_000.0, 250_000.0, 300_000.0]), None, meta)        # another month decreased


def test_monotonic_checks_have_the_right_direction():
    src, more, less = full([1.0, 2.0], n=[10, 10]), full([1.0, 2.0], n=[12, 10]), full([1.0, 2.0], n=[8, 10])
    assert not MR_BY_ID["MR5"].check(src, more, None, {}) and MR_BY_ID["MR5"].check(src, less, None, {})
    assert not MR_BY_ID["MR6"].check(src, less, None, {}) and MR_BY_ID["MR6"].check(src, more, None, {})


def test_catalog_has_ten_mrs_across_the_three_families():
    assert len(MRS) == 10 and {m.family for m in MRS} == {"Invariance", "Monotonic", "Directional"}
    assert {f: sum(m.family == f for m in MRS) for f in ("Invariance", "Monotonic", "Directional")} == {"Invariance": 2, "Monotonic": 4, "Directional": 4}
    assert len({m.id for m in MRS}) == len(MRS) and all(m.fault and m.tolerance and m.subdomain for m in MRS)


# ---- one tiny end-to-end trial per kind (calm data: 25 listings per month, no loop needs to fire)
@pytest.mark.parametrize("mr_id", ["MR1", "MR2", "MR4", "MR9", "MR10"])
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


# ---- the harness must never present a degraded real-model run as a real-model result
def test_counting_llm_counts_calls_and_failures_and_still_raises():
    from mr.harness import CountingLLM

    def flaky(system, user):
        if user == "boom":
            raise RuntimeError("401")
        return "{}"
    flaky.name = "flaky"
    llm = CountingLLM(flaky)
    assert llm("s", "ok") == "{}" and llm.name == "flaky"
    with pytest.raises(RuntimeError):
        llm("s", "boom")
    assert (llm.calls, llm.failures) == (2, 1)


def test_a_trial_in_which_the_model_failed_is_an_error_not_a_pass_or_a_violation():
    from mr.harness import CountingLLM
    df, c = tiny_df()

    def dead(system, user):
        raise RuntimeError("Invalid API Key")
    row = evaluate(MR_BY_ID["MR1"], df, c, 0, 1, llm=CountingLLM(dead))
    assert row["status"] == "holds" or row["status"] == "error"          # calm data may make no call at all
    thin = build({("75201", m): flat(8, 300_000) for m in MONTHS3}, MONTHS3, ["75201"]).df     # thin -> the Critic asks the model
    row = evaluate(MR_BY_ID["MR1"], thin, c, 0, 1, llm=CountingLLM(dead), min_listings=20)
    assert row["llm_failures"] >= 1 and row["status"] == "error" and "failed" in row["detail"]


def test_preflight_refuses_to_start_a_real_run_when_the_model_does_not_answer(monkeypatch):
    import mr.harness as H

    def dead(seed=None):
        def call(system, user):
            raise RuntimeError("401 Invalid API Key")
        call.name = "dead"
        return call
    monkeypatch.setattr("housing_trends.llm.llm_from_env", dead)
    with pytest.raises(RuntimeError, match="preflight failed"):
        H.preflight("real", 101)
    H.preflight("offline", 101)                                              # offline never needs a model


def test_the_ten_relations_are_numbered_MR1_to_MR10():
    assert [m.id for m in MRS] == [f"MR{i}" for i in range(1, 11)]


def test_random_draws_are_keyed_on_a_fixed_stream_so_renumbering_changes_no_transform_instance():
    """The streams equal each relation's index in the original 12-MR list (old id number - 1)."""
    from mr.harness import seed_stream
    from mr.relations import STREAM
    old_number = {"MR1": 2, "MR2": 4, "MR3": 5, "MR4": 6, "MR5": 7, "MR6": 8, "MR7": 9, "MR8": 10, "MR9": 11, "MR10": 12}
    for new, old in old_number.items():
        assert STREAM[new] == old - 1
        a = seed_stream(101, 7, new).integers(0, 2**31, 5)
        assert (a == np.random.default_rng([101, 7, old - 1]).integers(0, 2**31, 5)).all()
    assert len(set(STREAM.values())) == len(STREAM)
