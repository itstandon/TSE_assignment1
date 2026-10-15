"""Tests for the Scanner's one LLM decision: ranking comparable neighbour ZIPs to pool from.

Same contract as every LLM node: THE LLM PROPOSES, CODE VALIDATES. A ranking may only reorder the candidates code
offered; anything else is rejected, retried once, then replaced by the geographic order. Fake models only, no network.
"""
import json

import pandas as pd
import pytest

from housing_trends import Criteria
from housing_trends.agents.diagnoser import rule_based_llm
from housing_trends.agents.scanner import (MARK, N_CANDIDATES, Snapshot, candidates, make_scanner, rank_neighbours,
                                           rule_based_neighbours, scan, thin_zips, validate_rankings)
from housing_trends.context import month_range
from housing_trends.state import MAX_ZIP_EXTRA, Scope

ZIPS = [f"752{i:02d}" for i in range(1, 15)]
MONTHS = month_range("2025-01", "2025-06")
MIN = 10


@pytest.fixture(scope="module")
def snap():
    """75201 is thin (3 per month); every other ZIP is healthy. 75202-75204 are a different (10x) market."""
    rows, lid = [], 1
    for z in ZIPS:
        n = 3 if z == "75201" else 12
        price = 2_000_000 if z in ("75202", "75203", "75204") else 200_000
        for m in MONTHS:
            for k in range(n):
                rows.append(dict(listing_id=lid, zip_code=z, month=m, price=price + 100 * k, house_size=1500, city="Dallas"))
                lid += 1
    return Snapshot(pd.DataFrame(rows))


def crit(zips=("75201", "75205")):
    return Criteria(zips=zips, start_month="2025-02", end_month="2025-05")


def ranking_llm(*replies):
    """Fake model: each reply is a function of the prompt's evidence -> dict; records every call."""
    it, calls = iter(replies), []

    def call(system, user):
        calls.append(user)
        ev = json.loads(user.split(MARK, 1)[1].split("\n", 1)[0])
        r = next(it)
        return r(ev) if callable(r) else r
    call.calls, call.name = calls, "fake"
    return call


def cheap_first(ev):
    """Ranks candidates by |price_ratio - 1|, i.e. what a sensible model should do."""
    return json.dumps({"rankings": [{"zip_code": t["zip_code"], "reason": "similar price",
                                     "comparable": [c["zip_code"] for c in sorted(t["candidates"], key=lambda c: (
                                         abs((c["price_ratio"] or 99) - 1), c["proximity_rank"]))][:MAX_ZIP_EXTRA]}
                                    for t in ev["targets"]]})


# ---- which ZIPs are ranked --------------------------------------------------------------------
def test_only_thin_zips_are_ranked(snap):
    assert thin_zips(crit(), snap, MIN) == ["75201"]


def test_no_thin_zip_means_no_llm_call(snap):
    llm = ranking_llm()
    order, trace = rank_neighbours(llm, crit(("75205", "75206")), snap, MIN)
    assert order == {} and trace == [] and llm.calls == []


def test_no_llm_means_geographic_order(snap):
    assert rank_neighbours(None, crit(), snap, MIN) == ({}, [])


# ---- the stand-in is a true baseline ----------------------------------------------------------
def test_offline_standin_keeps_the_geographic_scan(snap):
    order, _ = rank_neighbours(rule_based_llm, crit(), snap, MIN)
    assert order["75201"][:MAX_ZIP_EXTRA] == snap.neighbours("75201", MAX_ZIP_EXTRA)
    scope = Scope(zip_extra=MAX_ZIP_EXTRA)
    a, b = scan(crit(), scope, snap, MIN), scan(crit(), scope, snap, MIN, order=order)
    pd.testing.assert_frame_equal(a.buckets, b.buckets)
    pd.testing.assert_frame_equal(a.members, b.members)


def test_rule_based_llm_dispatches_scanner_prompts(snap):
    order, trace = rank_neighbours(rule_based_llm, crit(), snap, MIN)
    assert "rule_based_standin" in trace[0] and "0 differ" in trace[0]


# ---- a ranking changes what gets pooled -------------------------------------------------------
def test_ranking_skips_a_different_market(snap):
    llm = ranking_llm(cheap_first)
    order, trace = rank_neighbours(llm, crit(), snap, MIN)
    assert not {"75202", "75203", "75204"} & set(order["75201"][:3])          # the 10x ZIPs are no longer pooled first
    assert sorted(order["75201"]) == sorted(snap.neighbours("75201", len(ZIPS)))   # a permutation, nothing lost
    assert "1 differ" in trace[0] and any("75201 pools" in t for t in trace)
    res = scan(crit(), Scope(zip_extra=2), snap, MIN, order=order)
    pooled = set(res.members[res.members.zip_code == "75201"].listing_id)
    by_zip = snap.df.set_index("listing_id").zip_code
    assert set(by_zip[list(pooled)]) == {"75201", *order["75201"][:2]}


def test_scan_counts_stay_exact_under_a_ranking(snap):
    order, _ = rank_neighbours(ranking_llm(cheap_first), crit(), snap, MIN)
    res = scan(crit(), Scope(zip_extra=2), snap, MIN, order=order)
    for _, b in res.buckets[res.buckets.zip_code == "75201"].iterrows():
        assert b.n == len(res.members[(res.members.zip_code == "75201") & (res.members.month == b.month)])


# ---- validation -------------------------------------------------------------------------------
def test_validate_rejects_non_candidates_duplicates_and_overlong(snap):
    allowed = {"75201": candidates(snap, "75201")}
    ok = allowed["75201"][:2]
    assert validate_rankings({"rankings": [{"zip_code": "75201", "comparable": ok}]}, allowed)[0] == {"75201": ok}
    for comp, msg in [(["99999"], "not candidates"), ([ok[0], ok[0]], "repeats"), ([], "non-empty"),
                      (allowed["75201"][:MAX_ZIP_EXTRA + 1], "at most")]:
        picks, _, errors = validate_rankings({"rankings": [{"zip_code": "75201", "comparable": comp}]}, allowed)
        assert picks == {} and msg in errors[0]
    assert "not one of the targets" in validate_rankings({"rankings": [{"zip_code": "75299", "comparable": ok}]},
                                                         allowed)[2][0]
    assert validate_rankings({"nope": 1}, allowed)[2] == ["reply must contain a list 'rankings'"]
    assert len(candidates(snap, "75201")) == N_CANDIDATES


def test_bad_reply_is_retried_with_feedback(snap):
    bad = json.dumps({"rankings": [{"zip_code": "75201", "comparable": ["99999"]}]})
    llm = ranking_llm(bad, cheap_first)
    order, trace = rank_neighbours(llm, crit(), snap, MIN)
    assert len(llm.calls) == 2 and "rejected" in llm.calls[1] and "75201" in order
    assert any("attempt 1: rejected" in t for t in trace)


def test_two_bad_replies_fall_back_to_geography(snap):
    llm = ranking_llm("not json", json.dumps({"rankings": []}))
    order, trace = rank_neighbours(llm, crit(), snap, MIN)
    assert order == {} and "geographic fallback for ['75201']" in trace[0]
    assert scan(crit(), Scope(zip_extra=2), snap, MIN, order=order).buckets.equals(
        scan(crit(), Scope(zip_extra=2), snap, MIN).buckets)


def test_prompt_carries_the_evidence(snap):
    llm = ranking_llm(cheap_first)
    rank_neighbours(llm, crit(), snap, MIN)
    t = json.loads(llm.calls[0].split(MARK, 1)[1])["targets"][0]
    assert t["zip_code"] == "75201" and t["median_price"] == 200_100
    c = {x["zip_code"]: x for x in t["candidates"]}
    assert c["75202"]["price_ratio"] == pytest.approx(10.0, abs=0.1) and c["75202"]["proximity_rank"] == 1


# ---- node: ranked once, reused by rescans -----------------------------------------------------
def test_node_ranks_once_and_rescans_reuse_it(snap):
    llm = ranking_llm(cheap_first)
    node = make_scanner(snap, MIN, llm=llm)
    first = node({"criteria": crit(), "scope": Scope()})
    assert "75201" in first["neighbour_order"] and len(llm.calls) == 1
    again = node({"criteria": crit(), "scope": Scope(zip_extra=2), "neighbour_order": first["neighbour_order"]})
    assert len(llm.calls) == 1 and again["neighbour_order"] is first["neighbour_order"]
