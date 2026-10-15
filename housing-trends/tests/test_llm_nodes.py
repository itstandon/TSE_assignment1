"""Tests for the three LLM-driven decision points (Critic route, Pooling move choice) and the offline stand-in.

The contract every one of them must honour: THE LLM PROPOSES, CODE VALIDATES. A reply that is malformed, names an
unknown option, or asks for a move the agent is not allowed to make must never reach the pipeline; the agent falls
back to its deterministic rule and says so in the trace. These tests use fake models, so they never touch a network.
"""
import json

import pandas as pd
import pytest

from housing_trends import Criteria
from housing_trends.agents import critic as C
from housing_trends.agents import pooling as P
from housing_trends.agents.diagnoser import rule_based_llm
from housing_trends.agents.scanner import Snapshot, scan
from housing_trends.context import month_range
from housing_trends.state import Scope

MIN = 20
ZIPS = ["75201", "75202", "75203", "75204", "75205", "75206", "75207", "75208"]
MONTHS = month_range("2025-01", "2025-12")


def fake(reply):
    """A model that always answers `reply` (a str, or an object that is JSON-dumped)."""
    def llm(system, user):
        llm.seen.append(user)
        return reply if isinstance(reply, str) else json.dumps(reply)
    llm.seen, llm.name = [], "fake"
    return llm


# ---------------------------------------------------------------- Critic: choosing a route
def test_validate_route_accepts_only_an_offered_option():
    assert C.validate_route({"decision": "pool"}, ["pool", "accept"]) == ("pool", [])
    assert C.validate_route({"decision": "render"}, ["pool", "accept"])[0] is None
    assert C.validate_route({"decision": 7}, ["pool"])[0] is None
    assert C.validate_route({}, ["pool"])[0] is None


def test_rule_based_route_reproduces_the_hard_coded_ladder():
    def pick(q, options):
        return json.loads(C.rule_based_route(C.build_route_prompt(q, options, {})))["decision"]
    assert pick("coverage", ["pool", "accept"]) == "pool"          # always widen while it legally can
    assert pick("coverage", ["accept"]) == "accept"
    assert pick("trend", ["recompute", "diagnoser", "accept"]) == "recompute"   # cheap robust recompute first
    assert pick("trend", ["diagnoser", "accept"]) == "diagnoser"
    assert pick("trend", ["accept"]) == "accept"


def test_ask_route_does_not_call_the_model_when_there_is_nothing_to_decide():
    llm = fake({"decision": "accept"})
    assert C.ask_route(llm, "coverage", ["pool"], {}) == ("pool", [])
    assert llm.seen == []                                          # a forced move must not cost an API call


def test_ask_route_uses_a_valid_decision_and_records_when_it_differs_from_the_rule():
    llm = fake({"decision": "accept", "reason": "only 2 buckets are thin"})
    decision, trace = C.ask_route(llm, "coverage", ["pool", "accept"], {"thin_buckets": 2})
    assert decision == "accept" and len(llm.seen) == 1
    assert "chose 'accept'" in trace[0] and "rule-based would have chosen 'pool'" in trace[0]


def test_ask_route_falls_back_when_the_model_insists_on_an_illegal_option():
    llm = fake({"decision": "teleport"})
    decision, trace = C.ask_route(llm, "trend", ["diagnoser", "accept"], {})
    assert decision == "diagnoser" and len(llm.seen) == 2          # one retry with the error fed back
    assert "not one of" in llm.seen[1] and "falling back to 'diagnoser'" in trace[0]


def test_ask_route_falls_back_on_unparseable_output():
    decision, trace = C.ask_route(fake("I am not JSON"), "trend", ["recompute", "accept"], {})
    assert decision == "recompute" and "falling back" in trace[0]


# ---------------------------------------------------------------- Pooling: choosing a move
def snap_from(counts):
    rows, lid = [], 1
    for (z, m), k in counts.items():
        for i in range(k):
            rows.append(dict(listing_id=lid, zip_code=z, month=m, price=300_000 + lid, bed=3, bath=2.0, house_size=1500))
            lid += 1
    df = pd.DataFrame(rows, columns=["listing_id", "zip_code", "month", "price", "bed", "bath", "house_size"])
    return Snapshot(df, months=MONTHS, zips=ZIPS)


@pytest.fixture
def thin_case():
    """75201 has 8 listings a month: padding one month reaches 24 >= 20, so 'time' suffices; 'zip' is also legal."""
    snap = snap_from({(z, m): 8 for z in ZIPS for m in MONTHS})
    c = Criteria(zips=("75201",), start_month="2025-04", end_month="2025-06")
    return c, snap, scan(c, Scope(), snap, MIN)


def test_bucket_options_list_the_legal_moves_and_whether_time_alone_suffices(thin_case):
    c, snap, sc = thin_case
    opts = P.bucket_options(c, Scope(), sc, MIN)
    assert len(opts) == 3 and all(o["time_suffices"] and "time" in o["legal"] and "zip" in o["legal"] for o in opts)
    assert all("filters" not in o["legal"] for o in opts)          # no relaxable filter is set on this criteria
    assert P.class_of(opts[0]) == "time+zip|time_suffices"


def test_validate_reply_rejects_an_illegal_move_an_unknown_class_and_an_omission(thin_case):
    c, snap, sc = thin_case
    opts = P.bucket_options(c, Scope(), sc, MIN)
    cid = P.class_of(opts[0])
    assert P.validate_reply({"choices": [{"class": cid, "move": "zip"}]}, opts) == ({cid: "zip"}, [])
    assert "not legal" in P.validate_reply({"choices": [{"class": cid, "move": "filters"}]}, opts)[1][0]
    assert "unknown bucket class" in P.validate_reply({"choices": [{"class": "nope", "move": "zip"}]}, opts)[1][0]
    assert "no move chosen" in P.validate_reply({"choices": []}, opts)[1][0]
    assert P.validate_reply({}, opts)[1] == ["reply must contain a list 'choices'"]


def test_the_model_can_overrule_the_preference_order(thin_case):
    c, snap, sc = thin_case
    cid = P.class_of(P.bucket_options(c, Scope(), sc, MIN)[0])
    llm = fake({"choices": [{"class": cid, "move": "zip", "reason": "seasonality matters more than geography here"}]})
    plan, trace = P.llm_plan_widening(llm, c, Scope(), sc, MIN)
    assert plan.counts == {"time": 0, "zip": 3, "filters": []}     # the rule would have padded months instead
    assert all(b.zip_extra == 2 and b.time_pad == 0 for b in plan.scope.per_bucket)
    assert "3 bucket(s) differ from the rule-based preference" in trace[0]


def test_an_unusable_reply_falls_back_to_the_rule_based_plan(thin_case):
    c, snap, sc = thin_case
    rule = P.plan_widening(c, Scope(), sc, MIN)
    plan, trace = P.llm_plan_widening(fake("not json at all"), c, Scope(), sc, MIN)
    assert plan.counts == rule.counts == {"time": 3, "zip": 0, "filters": []}
    assert plan.scope == rule.scope and "fallback (rule-based)" in trace[0]


def test_no_model_means_exactly_the_rule_based_plan(thin_case):
    c, snap, sc = thin_case
    plan, trace = P.llm_plan_widening(None, c, Scope(), sc, MIN)
    assert plan.scope == P.plan_widening(c, Scope(), sc, MIN).scope and trace == []


# ---------------------------------------------------------------- the offline stand-in serves every node
def test_the_standin_dispatches_on_the_prompt_marker(thin_case):
    c, snap, sc = thin_case
    opts = P.bucket_options(c, Scope(), sc, MIN)
    pooling_reply = json.loads(rule_based_llm(P.SYSTEM_PROMPT, P.build_user_prompt(opts, c, MIN)))
    assert P.validate_reply(pooling_reply, opts) == ({P.class_of(opts[0]): "time"}, [])

    critic_reply = json.loads(rule_based_llm(C.SYSTEM_PROMPT, C.build_route_prompt("trend", ["diagnoser", "accept"], {})))
    assert critic_reply["decision"] == "diagnoser"

    diag = json.loads(rule_based_llm("", "EVIDENCE_JSON:\n" + json.dumps(
        {"anomalies": [], "suspects": [{"listing_id": 1, "zip_code": "75201", "month": "2025-01", "price": 9e6,
                                        "price_vs_zip_median": 12.0, "acre_lot_pct": 50, "house_size_pct": 50}]})))
    assert diag["verdicts"][0]["verdict"] == "bad_listing"


def test_the_standin_keeps_the_pipeline_deterministic(thin_case):
    """Same inputs -> same decisions, which is what the offline MR runs rely on."""
    c, snap, sc = thin_case
    a = P.llm_plan_widening(rule_based_llm, c, Scope(), sc, MIN)[0]
    b = P.llm_plan_widening(rule_based_llm, c, Scope(), sc, MIN)[0]
    assert a.scope == b.scope == P.plan_widening(c, Scope(), sc, MIN).scope
