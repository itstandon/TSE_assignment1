import pytest

from housing_trends.agents.critic import evidence, jump_flags, make_critic, robust_instruction
from housing_trends.agents.profile import ZipProfile
from housing_trends.agents.statistician import StatInstruction
from housing_trends.state import MAX_DIAG_ROUNDS, MAX_RECOMPUTES, MAX_RESCANS
from tests.helpers import MONTHS3, RURAL, build, crit, flat, spread, state_for

Z = "75201"
MIN = 5


def calm():
    return build({(Z, m): flat(10) for m in MONTHS3}, MONTHS3, [Z])

def typo_jump():
    """Month 2 is half normal (300k) and half typo (3M): median jumps ~+450%, trimmed mean cannot fix it."""
    return build({(Z, MONTHS3[0]): flat(10), (Z, MONTHS3[1]): flat(5) + flat(5, 3_000_000), (Z, MONTHS3[2]): flat(10)}, MONTHS3, [Z])

def shift_jump():
    """Genuine composition shift (+50%) inside a realistic +-30% spread: a jump, but no outlier listing to blame."""
    return build({(Z, MONTHS3[0]): spread(10, 300_000), (Z, MONTHS3[1]): spread(10, 450_000), (Z, MONTHS3[2]): spread(10, 450_000)}, MONTHS3, [Z])

C = crit([Z], MONTHS3[0], MONTHS3[2])


def run_critic(snap, **kw):
    st = state_for(snap, C, min_listings=MIN, **kw.pop("state", {}))
    return make_critic(snap, MIN)(st), st


# ---- 1. coverage ---------------------------------------------------------------------------------------
def test_thin_buckets_route_to_pooling():
    snap = build({(Z, m): flat(3) for m in MONTHS3}, MONTHS3, [Z])
    snap2 = build({(Z, m): flat(3) for m in MONTHS3} | {("75202", m): flat(40) for m in MONTHS3}, MONTHS3, [Z, "75202"])
    out, _ = run_critic(snap2)
    assert out["route"] == "pooling" and "Data Issue" in out["trace"][0] and "rescan 1/3" in out["trace"][0]

def test_rescan_cap_closes_coverage_and_reports_it():
    snap = build({(Z, m): flat(3) for m in MONTHS3} | {("75202", m): flat(40) for m in MONTHS3}, MONTHS3, [Z, "75202"])
    out, _ = run_critic(snap, state=dict(rescans=MAX_RESCANS))
    assert out["route"] == "render" and any("rescan cap" in u for u in out["unresolved"])

def test_unfixable_thin_buckets_are_reported_not_looped():
    snap = build({(Z, m): [] for m in MONTHS3} | {(Z, MONTHS3[0]): flat(1)}, MONTHS3, [Z])
    out, _ = run_critic(snap)
    assert out["route"] == "render" and any("no allowed widening" in u or "remain below" in u for u in out["unresolved"])


# ---- 2. trend / method loop ------------------------------------------------------------------------------
def test_calm_data_is_accepted():
    out, _ = run_critic(calm())
    assert out["route"] == "render" and out["unresolved"] == [] and "no price jumps" in out["trace"][-1]

def test_first_jump_triggers_robust_recompute_over_the_whole_zip_series():
    out, st = run_critic(typo_jump())
    assert out["route"] == "recompute" and out["recomputes"] == 1 and out["jumps_initial"] == 2
    ov = out["stat_instruction"].overrides
    assert {(o.zip_code, o.month) for o in ov} == {(Z, m) for m in MONTHS3}              # every month, so the series stays consistent
    assert all(o.method.name == "trimmed_mean" for o in ov)

def test_jumps_resolved_by_the_recompute_are_accepted_as_estimator_artifacts():
    # 8 normal + 2 huge: the median ignores them but the plain-median jump comes from a different cause; build a case
    # where the trimmed mean (computed over the same listings) removes the jump: handled in test_graph. Here: no jumps left.
    snap = calm()
    st = state_for(snap, C, min_listings=MIN, recomputes=1, jumps_initial=3)
    out = make_critic(snap, MIN)(st)
    assert out["route"] == "render" and "first seen: 3" in out["trace"][-1]

def test_persistent_jump_goes_to_the_diagnoser_with_suspects():
    snap = typo_jump()
    inst = state_for(snap, C, min_listings=MIN)["stat_instruction"]
    st = state_for(snap, C, min_listings=MIN, recomputes=1, jumps_initial=2, instr=robust_instruction(inst, [Z], MONTHS3))
    out = make_critic(snap, MIN)(st)
    assert out["route"] == "diagnoser" and out["diag_rounds"] == 1
    assert len(out["suspects"]) == 5 and all(s["price"] >= 3_000_000 for s in out["suspects"])
    assert {a["zip_code"] for a in out["anomalies"]} == {Z} and out["persistent"]

def test_no_suspects_still_gets_one_round_for_an_explanation_but_no_second():
    snap = shift_jump()
    st = state_for(snap, C, min_listings=MIN, recomputes=1, jumps_initial=1)
    out = make_critic(snap, MIN)(st)
    assert out["route"] == "diagnoser" and out["suspects"] == []                           # round 1: explanation only
    st2 = state_for(snap, C, min_listings=MIN, recomputes=1, diag_rounds=1)
    out2 = make_critic(snap, MIN)(st2)
    assert out2["route"] == "render" and "no new outlier suspects" in out2["trace"][-1] and out2["unresolved"]

def test_diagnoser_round_cap_ends_in_unresolved():
    snap = typo_jump()
    st = state_for(snap, C, min_listings=MIN, recomputes=1, diag_rounds=MAX_DIAG_ROUNDS)
    out = make_critic(snap, MIN)(st)
    assert out["route"] == "render" and "round cap" in out["trace"][-1] and out["persistent"]

def test_listings_already_judged_are_not_sent_again():
    snap = typo_jump()
    first = make_critic(snap, MIN)(state_for(snap, C, min_listings=MIN, recomputes=1))
    judged = [{"type": "verdict", "listing_id": s["listing_id"]} for s in first["suspects"]]
    again = make_critic(snap, MIN)(state_for(snap, C, min_listings=MIN, recomputes=1, diag_rounds=1, diagnoses=judged))
    assert again["route"] == "render"

def test_recompute_is_bounded_even_if_jumps_remain():
    snap = typo_jump()
    out = make_critic(snap, MIN)(state_for(snap, C, min_listings=MIN, recomputes=MAX_RECOMPUTES))
    assert out["route"] != "recompute"

def test_critic_is_pure():
    snap = typo_jump()
    st = state_for(snap, C, min_listings=MIN)
    a, b = make_critic(snap, MIN)(st), make_critic(snap, MIN)(st)
    assert a["route"] == b["route"] and a["trace"] == b["trace"]


# ---- helpers ----------------------------------------------------------------------------------------------
def test_jump_flags_threshold_and_sign():
    snap = typo_jump()
    st = state_for(snap, C, min_listings=MIN)
    f = jump_flags(st["stats"].buckets, 30)
    assert list(f.month) == [MONTHS3[1], MONTHS3[2]] and f.mom_price_pct.iloc[0] > 0 > f.mom_price_pct.iloc[1]
    assert jump_flags(st["stats"].buckets, 10_000).empty

def test_evidence_screen_finds_outliers_and_respects_exclusions_and_caps():
    snap = typo_jump()
    st = state_for(snap, C, min_listings=MIN)
    flagged = jump_flags(st["stats"].buckets)
    prof = ZipProfile(snap.df)
    anomalies, suspects = evidence(flagged, st["scan"], st["stats"], snap, prof)
    assert len(suspects) == 5 and all(abs(s["z_price"]) > 3.5 and s["price_vs_zip_median"] > 5 for s in suspects)
    ids = [s["listing_id"] for s in suspects]
    _, fewer = evidence(flagged, st["scan"], st["stats"], snap, prof, excluded=set(ids[:2]))
    assert len(fewer) == 3 and not set(ids[:2]) & {s["listing_id"] for s in fewer}
    assert anomalies[0]["now"]["n_used"] == 10 and anomalies[0]["prev"]["n_used"] == 10

def test_rural_listing_evidence_shows_large_lot_small_house():
    spec = {(Z, MONTHS3[0]): flat(10), (Z, MONTHS3[1]): flat(5) + [(90_000 + 100 * i, RURAL) for i in range(5)], (Z, MONTHS3[2]): flat(10)}
    snap = build(spec, MONTHS3, [Z])
    st = state_for(snap, C, min_listings=MIN)
    _, suspects = evidence(jump_flags(st["stats"].buckets), st["scan"], st["stats"], snap, ZipProfile(snap.df))
    assert suspects and all(s["acre_lot_pct"] >= 90 and s["house_size_pct"] <= 35 and 0.2 < s["price_vs_zip_median"] < 1 for s in suspects)

def test_zip_profile_percentiles_and_z():
    snap = typo_jump()
    p = ZipProfile(snap.df)
    assert p.pct(Z, "price", 10**9) == 100.0 and p.pct(Z, "price", 1) == 0.0 and p.pct(Z, "acre_lot", None) is None
    assert abs(p.z_price(Z, p.median_price(Z))) < 1e-9 and p.z_price(Z, 3_000_000) > 10