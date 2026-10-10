import json

import pytest

from housing_trends.agents.diagnoser import (MARK, SYSTEM_PROMPT, build_user_prompt, make_diagnoser, rule_based_llm,
                                              validate_reply)
from housing_trends.agents.statistician import StatInstruction
from housing_trends.llm import CachedLLM
from tests.helpers import MONTHS3, build, crit, flat, state_for

Z = "75201"
SUSPECTS = [dict(listing_id=7, zip_code=Z, month="2025-02", price=3_000_000.0, price_vs_zip_median=10.0, acre_lot_pct=50.0, house_size_pct=50.0, z_price=40.0),
            dict(listing_id=8, zip_code=Z, month="2025-02", price=90_000.0, price_vs_zip_median=0.3, acre_lot_pct=100.0, house_size_pct=25.0, z_price=-24.0),
            dict(listing_id=9, zip_code=Z, month="2025-02", price=450_000.0, price_vs_zip_median=1.5, acre_lot_pct=50.0, house_size_pct=50.0, z_price=-4.0)]
ANOMALIES = [dict(zip_code=Z, month="2025-02", prev_month="2025-01", mom_price_pct=120.0, now=dict(n_used=10), prev=dict(n_used=10))]


def scripted(*replies):
    calls, it = [], iter(replies)
    def llm(system, user):
        calls.append(user)
        return next(it)
    llm.calls = calls
    return llm


def good(verdicts=None):
    return json.dumps({"verdicts": [{"listing_id": i, "verdict": v, "reason": "r"} for i, v in (verdicts or {7: "bad_listing", 8: "legit_rural", 9: "unexplained"}).items()],
                       "explanations": [{"zip_code": Z, "month": "2025-02", "text": "because"}]})


def run(llm, suspects=SUSPECTS, anomalies=ANOMALIES, instr=None):
    snap = build({(Z, m): flat(10) for m in MONTHS3}, MONTHS3, [Z])
    st = state_for(snap, crit([Z], MONTHS3[0], MONTHS3[2]), min_listings=5, instr=instr)
    st.update(anomalies=anomalies, suspects=suspects, diag_rounds=1)
    return make_diagnoser(llm)(st), st


# ---- validation ----------------------------------------------------------------------------------------------
def test_validate_accepts_a_good_reply():
    v, e, errs = validate_reply(json.loads(good()), ANOMALIES, SUSPECTS)
    assert errs == [] and v[7][0] == "bad_listing" and e[(Z, "2025-02")] == "because"

@pytest.mark.parametrize("obj,needle", [
    ({"verdicts": [{"listing_id": 99, "verdict": "bad_listing"}], "explanations": []}, "unknown listing_id"),
    ({"verdicts": [{"listing_id": 7, "verdict": "maybe"}], "explanations": []}, "unknown verdict"),
    ({"verdicts": [{"verdict": "bad_listing"}], "explanations": []}, "malformed verdict"),
    ({"verdicts": [], "explanations": [{"zip_code": "99999", "month": "2025-02", "text": "x"}]}, "unknown anomaly"),
    ({"verdicts": "nope", "explanations": []}, "must contain lists"),
])
def test_validate_rejects_bad_replies(obj, needle):
    assert needle in "; ".join(validate_reply(obj, ANOMALIES, SUSPECTS)[2])


# ---- verdicts change what the Statistician will do ----------------------------------------------------------------
def test_verdicts_update_the_statistician_instruction():
    out, st = run(scripted(good()))
    ins = out["stat_instruction"]
    assert ins.exclude_ids == (7, 9)                                   # bad_listing + unexplained are excluded
    assert 8 not in ins.exclude_ids                                    # legit_rural is kept ...
    assert {(o.zip_code, o.month) for o in ins.overrides} == {(Z, m) for m in MONTHS3}        # ... and its ZIP series is robust
    assert all(o.method.name == "trimmed_mean" for o in ins.overrides)
    kinds = {r["listing_id"]: r["verdict"] for r in out["diagnoses"] if r["type"] == "verdict"}
    assert kinds == {7: "bad_listing", 8: "legit_rural", 9: "unexplained"}

def test_existing_exclusions_and_history_are_preserved():
    prior = StatInstruction(exclude_ids=(1, 2))
    out, st = run(scripted(good()), instr=prior)
    assert set(out["stat_instruction"].exclude_ids) == {1, 2, 7, 9}
    assert len(out["diagnoses"]) == 3 + 1

def test_only_legit_verdicts_leave_the_instruction_free_of_exclusions():
    out, _ = run(scripted(good({7: "legit_rural", 8: "legit_rural", 9: "legit_rural"})))
    assert out["stat_instruction"].exclude_ids == ()


# ---- untrusted replies: retry once, then never guess --------------------------------------------------------------------
def test_invalid_reply_is_retried_with_the_errors_fed_back():
    bad = json.dumps({"verdicts": [{"listing_id": 99, "verdict": "bad_listing"}], "explanations": []})
    llm = scripted(bad, good())
    out, _ = run(llm)
    assert len(llm.calls) == 2 and "unknown listing_id 99" in llm.calls[1] and "attempt(s)" in out["trace"][0] and "in 2 attempt" in out["trace"][0]
    assert out["stat_instruction"].exclude_ids == (7, 9)

def test_non_json_reply_is_retried():
    llm = scripted("sorry, I cannot", good())
    out, _ = run(llm)
    assert len(llm.calls) == 2 and "not valid JSON" in llm.calls[1] and out["stat_instruction"].exclude_ids == (7, 9)

def test_two_failures_fall_back_to_unexplained_never_a_guess():
    llm = scripted("nope", "still nope")
    out, _ = run(llm)
    assert len(llm.calls) == 2                                          # cap: two attempts
    assert out["stat_instruction"].exclude_ids == (7, 8, 9)             # every suspect excluded and flagged
    assert {r["verdict"] for r in out["diagnoses"] if r["type"] == "verdict"} == {"unexplained"}
    assert {r["source"] for r in out["diagnoses"]} == {"fallback"}

def test_missing_verdict_for_a_suspect_counts_as_unexplained():
    out, _ = run(scripted(good({7: "bad_listing"})))
    v = {r["listing_id"]: r["verdict"] for r in out["diagnoses"] if r["type"] == "verdict"}
    assert v == {7: "bad_listing", 8: "unexplained", 9: "unexplained"}

def test_no_suspects_still_records_explanations():
    llm = scripted(json.dumps({"verdicts": [], "explanations": [{"zip_code": Z, "month": "2025-02", "text": "composition shift"}]}))
    out, _ = run(llm, suspects=[])
    assert out["stat_instruction"].exclude_ids == () and [r["text"] for r in out["diagnoses"] if r["type"] == "explanation"] == ["composition shift"]

def test_prompt_contains_the_evidence_and_the_rules():
    p = build_user_prompt(ANOMALIES, SUSPECTS)
    assert p.startswith(MARK) and '"listing_id": 7' in p and "bad_listing" in SYSTEM_PROMPT and "legit_rural" in SYSTEM_PROMPT


# ---- deterministic stand-in and cache -----------------------------------------------------------------------------------
def test_rule_based_standin_follows_its_rules():
    out = json.loads(rule_based_llm("", build_user_prompt(ANOMALIES, SUSPECTS)))
    assert {v["listing_id"]: v["verdict"] for v in out["verdicts"]} == {7: "bad_listing", 8: "legit_rural", 9: "unexplained"}
    assert out["explanations"][0]["zip_code"] == Z
    assert validate_reply(out, ANOMALIES, SUSPECTS)[2] == []            # honours the same contract as a real model

def test_record_labels_name_the_model_that_answered():
    out, _ = run(rule_based_llm)
    assert {r["source"] for r in out["diagnoses"]} == {"rule_based_standin"} and "rule_based_standin" in out["trace"][0]

def test_cached_llm_replays_identical_prompts(tmp_path):
    inner = scripted("A", "B")
    c = CachedLLM(inner, tmp_path, tag="m@1")
    assert (c("s", "u"), c("s", "u"), c("s", "other")) == ("A", "A", "B")
    assert (c.hits, c.misses) == (1, 2) and len(inner.calls) == 2
    c2 = CachedLLM(scripted(), tmp_path, tag="m@1")                       # a fresh process, no model available
    assert c2("s", "u") == "A"
    assert CachedLLM(scripted("Z"), tmp_path, tag="other-model")("s", "u") == "Z"          # tag is part of the key


def test_bad_only_verdicts_return_the_zip_to_the_plain_median():
    from tests.helpers import MONTHS3 as M
    from housing_trends.agents.critic import robust_instruction
    snap = build({(Z, m): flat(10) for m in M}, M, [Z])
    prior = robust_instruction(StatInstruction(), [Z], M)                      # the Critic's robust recompute is in force
    st = state_for(snap, crit([Z], M[0], M[2]), min_listings=5, instr=prior)
    st.update(anomalies=ANOMALIES, suspects=SUSPECTS[:1], diag_rounds=1)
    out = make_diagnoser(scripted(good({7: "bad_listing"})))(st)
    assert out["stat_instruction"].exclude_ids == (7,) and out["stat_instruction"].overrides == ()      # excluded + plain median

def test_a_legit_verdict_in_the_zip_keeps_the_robust_method():
    from tests.helpers import MONTHS3 as M
    snap = build({(Z, m): flat(10) for m in M}, M, [Z])
    st = state_for(snap, crit([Z], M[0], M[2]), min_listings=5)
    st.update(anomalies=ANOMALIES, suspects=SUSPECTS[:2], diag_rounds=1)
    out = make_diagnoser(scripted(good({7: "bad_listing", 8: "legit_rural"})))(st)
    assert out["stat_instruction"].exclude_ids == (7,)
    assert {o.zip_code for o in out["stat_instruction"].overrides} == {Z}