import pandas as pd
import pytest

from housing_trends.agents.diagnoser import rule_based_llm
from housing_trends.agents.scanner import Snapshot
from housing_trends.context import month_range
from housing_trends.graph import build_graph, run
from housing_trends.state import MAX_DIAG_ROUNDS, MAX_RECOMPUTES, MAX_RESCANS
from tests.helpers import MONTHS3, RURAL, build, crit, flat

M14 = month_range("2025-01", "2026-02")            # 14 months, so a 12-month window has room to pad
W0, W1 = "2025-02", "2026-01"


def lines(st, prefix):
    return [l for l in st["trace"] if l.startswith(prefix)]


def order(st, *needles):
    """Each needle must appear in the trace AFTER the previous one (sequential search)."""
    pos = -1
    for n in needles:
        hits = [i for i, l in enumerate(st["trace"]) if n in l and i > pos]
        assert hits, f"no trace line containing {n!r} after position {pos}:\n" + "\n".join(st["trace"])
        pos = hits[0]


# ---- scenario builders (k scales every price, for the scaling MR) ------------------------------------------------------
def calm(k=1):
    return build({("75201", m): flat(25, 300_000 * k) for m in MONTHS3}, MONTHS3, ["75201"]), crit(["75201"], MONTHS3[0], MONTHS3[2])

def thin(k=1):
    spec = {(z, m): flat(n, 300_000 * k) for m in M14 for z, n in (("75201", 8), ("75202", 2), ("75203", 30))}
    return build(spec, M14, ["75201", "75202", "75203"]), crit(["75201", "75202"], W0, W1)

def typo(k=1):
    spec = {("75201", MONTHS3[0]): flat(25, 300_000 * k), ("75201", MONTHS3[1]): flat(12, 300_000 * k) + flat(13, 3_000_000 * k),
            ("75201", MONTHS3[2]): flat(25, 300_000 * k)}
    return build(spec, MONTHS3, ["75201"]), crit(["75201"], MONTHS3[0], MONTHS3[2])

def rural(k=1):
    rur = [((90_000 + 100 * i) * k, RURAL) for i in range(13)]
    spec = {("75201", MONTHS3[0]): flat(25, 300_000 * k), ("75201", MONTHS3[1]): flat(12, 300_000 * k) + rur,
            ("75201", MONTHS3[2]): flat(25, 300_000 * k)}
    return build(spec, MONTHS3, ["75201"]), crit(["75201"], MONTHS3[0], MONTHS3[2])

def everything():
    """75201 is thin (coverage loop); 75204 has a month that is half typos (method + outlier loops)."""
    spec = {("75201", m): flat(8) for m in M14} | {("75204", m): flat(30) for m in M14}
    spec[("75204", "2025-05")] = flat(15) + flat(15, 3_000_000)
    return build(spec, M14, ["75201", "75204"]), crit(["75201", "75204"], W0, W1)


# ---- each loop fires, in order, and is bounded -----------------------------------------------------------------------------
def test_calm_data_fires_no_loop():
    snap, c = calm()
    st = run(c, snap, rule_based_llm)
    assert {k: st[k] for k in ("rescans", "recomputes", "diag_rounds")} == {"rescans": 0, "recomputes": 0, "diag_rounds": 0}
    assert not lines(st, "pooling") and not lines(st, "diagnoser") and st["report"].unresolved == []
    assert len(lines(st, "scanner")) == 1 == len(lines(st, "statistician"))

def test_coverage_loop_widens_each_bucket_only_as_far_as_it_needs():
    snap, c = thin()
    st = run(c, snap, rule_based_llm)
    r = st["report"]
    assert 1 <= st["rescans"] <= MAX_RESCANS and not r.buckets.thin.any() and r.unresolved == []
    b = r.buckets.set_index(["zip_code", "month"])
    assert (b.loc["75201"].time_pad_used == 1).all() and (b.loc["75201"].zip_extra_used == 0).all()      # 8/month: +-1 month is enough
    assert (b.loc["75202"].zip_extra_used >= 2).all()                                                    # 2/month: needs neighbouring ZIPs
    assert (b.n_used >= 20).all()
    order(st, "critic: coverage", "pooling: rescan 1", "scanner", "statistician", "critic: coverage OK")

def test_method_and_outlier_loops_fire_and_the_verdict_changes_the_computation():
    snap, c = typo()
    st = run(c, snap, rule_based_llm)
    assert (st["recomputes"], st["diag_rounds"], st["rescans"]) == (1, 1, 0)
    order(st, "critic: trend: 2 price jump", "statistician: method=median, 3 buckets (0 empty), 3 override",
          "persist", "diagnoser: round 1", "statistician: method=median, 3 buckets (0 empty), 0 override(s), 8 excluded",
          "no price jumps")
    v = st["report"].verdicts
    assert len(v) == 8 and {x["verdict"] for x in v} == {"bad_listing"}                                 # MAX_SUSPECTS worst per round
    b = st["report"].buckets.set_index("month")
    assert b.loc[MONTHS3[1], "n_excluded"] == 8 and 290_000 < b.loc[MONTHS3[1], "price"] < 320_000       # jump gone
    assert st["report"].unresolved == []

def test_a_different_verdict_takes_a_different_computation_path():
    snap, c = rural()
    st = run(c, snap, rule_based_llm)
    r = st["report"]
    assert {x["verdict"] for x in r.verdicts} == {"legit_rural"}                                         # kept, not excluded
    assert r.buckets.n_excluded.sum() == 0 and (r.buckets.method == "trimmed_mean(0.1)").all()
    assert r.unresolved and "unresolved" in r.unresolved[0] and r.explanations                          # still flagged, but explained
    t_snap, t_c = typo()
    t = run(t_c, t_snap, rule_based_llm)["report"]
    assert t.buckets.n_excluded.sum() > 0 and r.buckets.n_excluded.sum() == 0                             # excluded vs kept
    assert (t.buckets.method == "median").all() and (r.buckets.method == "trimmed_mean(0.1)").all()       # plain median vs robust

def test_all_three_loops_in_one_run():
    snap, c = everything()
    st = run(c, snap, rule_based_llm)
    assert st["rescans"] == 1 and st["recomputes"] == MAX_RECOMPUTES and st["diag_rounds"] == 1
    order(st, "pooling: rescan 1", "critic: coverage OK", "robust recompute", "diagnoser: round 1", "no price jumps")
    r = st["report"]
    assert r.unresolved == [] and not r.buckets.thin.any()
    assert (r.buckets[r.buckets.zip_code == "75201"].time_pad_used == 1).all()
    assert r.buckets[r.buckets.zip_code == "75204"].time_pad_used.eq(0).all()                           # healthy ZIP never pooled

def test_unreachable_thin_buckets_do_not_loop():
    snap = build({("75201", m): flat(1) for m in MONTHS3}, MONTHS3, ["75201"])
    st = run(crit(["75201"], MONTHS3[0], MONTHS3[2]), snap, rule_based_llm)
    assert st["rescans"] == 0 and any("no allowed widening" in u for u in st["report"].unresolved)

def test_loops_are_bounded_whatever_the_data():
    for snap, c in (calm(), thin(), typo(), rural(), everything()):
        st = run(c, snap, rule_based_llm)
        assert st["rescans"] <= MAX_RESCANS and st["recomputes"] <= MAX_RECOMPUTES and st["diag_rounds"] <= MAX_DIAG_ROUNDS


# ---- the graph itself ------------------------------------------------------------------------------------------------------------
def test_graph_has_the_required_cycles():
    snap, _ = calm()
    g = build_graph(snap, rule_based_llm).get_graph()
    edges = {(e.source, e.target) for e in g.edges}
    assert {("pooling", "scanner"), ("diagnoser", "statistician"), ("critic", "statistician"),
            ("critic", "pooling"), ("critic", "diagnoser"), ("critic", "render"), ("scanner", "statistician")} <= edges
    assert {"scanner", "statistician", "critic", "pooling", "diagnoser", "render"} <= set(g.nodes)


# ---- the report only presents what the Statistician computed -----------------------------------------------------------------------
def test_report_does_not_alter_statistics_and_annotates_buckets():
    snap, c = everything()
    st = run(c, snap, rule_based_llm)
    r = st["report"]
    pd.testing.assert_series_equal(r.buckets.set_index(["zip_code", "month"]).price, st["stats"].buckets.set_index(["zip_code", "month"]).price)
    pd.testing.assert_frame_equal(r.overall, st["stats"].overall)
    assert r.buckets[r.buckets.zip_code == "75201"].notes.str.contains("pooled").all()
    assert r.buckets.notes.str.contains("excluded").any()


# ---- metamorphic smoke tests at pipeline level (the harness will generalise these) ---------------------------------------------------
def summary(st):
    r = st["report"]
    return ({k: st[k] for k in ("rescans", "recomputes", "diag_rounds")}, sorted((v["listing_id"], v["verdict"]) for v in r.verdicts),
            r.buckets.set_index(["zip_code", "month"]))

@pytest.mark.parametrize("scenario", [thin, typo, rural, everything])
def test_row_order_does_not_change_the_pipeline_outcome(scenario):
    snap, c = scenario()
    shuffled = Snapshot(snap.df.sample(frac=1, random_state=11).reset_index(drop=True), months=snap.months, zips=snap.zips)
    a, b = summary(run(c, snap, rule_based_llm)), summary(run(c, shuffled, rule_based_llm))
    assert a[0] == b[0] and a[1] == b[1]
    pd.testing.assert_frame_equal(a[2].drop(columns=["notes"]), b[2].drop(columns=["notes"]))

@pytest.mark.parametrize("scenario", [typo, rural])
def test_scaling_all_prices_scales_results_and_keeps_every_decision(scenario):
    k = 2.0
    (s1, c1), (s2, c2) = scenario(1), scenario(k)
    a, b = summary(run(c1, s1, rule_based_llm)), summary(run(c2, s2, rule_based_llm))
    assert a[0] == b[0] and [v for _, v in a[1]] == [v for _, v in b[1]]                              # same loops, same verdicts
    pd.testing.assert_series_equal((a[2].price * k), b[2].price, rtol=1e-6, check_names=False)
    pd.testing.assert_series_equal(a[2].mom_price_pct, b[2].mom_price_pct, rtol=1e-6, check_names=False)