import json
import pandas as pd
import pytest
from housing_trends import SnapshotContext, handle_input


@pytest.fixture
def ctx():
    rows = []
    for i, m in enumerate(str(p) for p in pd.period_range("2021-01", "2022-06", freq="M")):
        for z, c, base in [(75201, "Dallas", 400_000), (75204, "Dallas", 450_000),
                           (75230, "Dallas", 700_000), (75024, "Plano", 600_000)]:
            rows.append(dict(zip_code=float(z), city=c, price=base + i * 1000, month=m))
    return SnapshotContext.from_dataframe(pd.DataFrame(rows), area_gazetteer={"North Dallas": ["75230"]})


def fake_llm(*replies):
    it, calls = iter(replies), []
    def call(system, user):
        calls.append(user)
        return next(it)
    call.calls = calls
    return call


# ---- form path ---------------------------------------------------------------
def test_form_ok_and_defaults_come_from_snapshot_not_clock(ctx):
    r = handle_input(form={"zips": ["75201"], "price_max": 500000}, ctx=ctx)
    assert r.ok and r.criteria.end_month == "2022-06" and r.criteria.start_month == "2021-07"
    assert any("defaulted" in w for w in r.warnings)

def test_zip_order_and_duplicates_do_not_matter(ctx):
    a = handle_input(form={"zips": ["75204", "75201", "75201"]}, ctx=ctx).criteria
    b = handle_input(form={"zips": [75201.0, "75204"]}, ctx=ctx).criteria
    assert a == b and a.key() == b.key()

def test_validator_collects_all_errors(ctx):
    r = handle_input(form={"zips": ["99999"], "price_min": 900, "price_max": 100,
                           "start_month": "2021-01", "end_month": "2021-05"}, ctx=ctx)
    assert not r.ok and len(r.errors) == 3

def test_window_must_be_12_months_and_inside_snapshot(ctx):
    assert not handle_input(form={"zips": ["75201"], "end_month": "2022-07"}, ctx=ctx).ok
    assert not handle_input(form={"zips": ["75201"], "end_month": "2020-12"}, ctx=ctx).ok

def test_property_type_dropped_with_warning_when_snapshot_lacks_column(ctx):
    r = handle_input(form={"zips": ["75201"], "property_types": ["condo"]}, ctx=ctx)
    assert r.ok and r.criteria.property_types == () and any("property-type" in w for w in r.warnings)

def test_unknown_field_is_warned_not_silently_used(ctx):
    r = handle_input(form={"zips": ["75201"], "school_rating": 9}, ctx=ctx)
    assert r.ok and any("school_rating" in w for w in r.warnings)


# ---- free-text path ------------------------------------------------------------
GOOD = json.dumps({"zips": ["75230"], "price_max": "500k", "end_month": "2022-06-30",
                   "assumptions": ["'affordable' -> price_max = p50"], "unsupported": ["good schools"]})

def test_free_text_converges_to_same_criteria_as_form(ctx):
    t = handle_input(text="affordable homes in north dallas near good schools", ctx=ctx, llm=fake_llm(GOOD))
    f = handle_input(form={"zips": ["75230"], "price_max": 500000, "end_month": "2022-06"}, ctx=ctx)
    assert t.ok and t.criteria == f.criteria
    assert t.unsupported == ["good schools"] and t.assumptions

def test_hallucinated_zip_is_rejected_then_repaired(ctx):
    bad = json.dumps({"zips": ["75999"]})
    llm = fake_llm(bad, GOOD)
    r = handle_input(text="north dallas", ctx=ctx, llm=llm)
    assert r.ok and r.attempts == 2 and "75999" in llm.calls[1]      # error fed back to the agent
    assert "rejected by validator" in r.trace[0]

def test_gives_up_instead_of_guessing(ctx):
    r = handle_input(text="north dallas", ctx=ctx, llm=fake_llm("not json", json.dumps({"zips": ["00000"]})))
    assert not r.ok and r.criteria is None and "use the form" in r.errors[0]

def test_code_fenced_json_is_tolerated(ctx):
    r = handle_input(text="x", ctx=ctx, llm=fake_llm("```json\n" + GOOD + "\n```"))
    assert r.ok

def test_prompt_contains_snapshot_facts_for_grounding(ctx):
    llm = fake_llm(GOOD)
    handle_input(text="x", ctx=ctx, llm=llm)
    p = llm.calls[0]
    assert "75230" in p and "p50" in p and "2022-06" in p and "north dallas" in p

def test_exactly_one_input_mode(ctx):
    with pytest.raises(ValueError):
        handle_input(ctx=ctx)
    with pytest.raises(ValueError):
        handle_input(form={}, text="x", ctx=ctx)