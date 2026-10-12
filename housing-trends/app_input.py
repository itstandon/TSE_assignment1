"""Dallas housing trends: enter criteria, run the agents, see the 12-month dashboard.  Run:  python -m streamlit run app_input.py"""
import json
import pathlib

import pandas as pd
import streamlit as st

from housing_trends import SnapshotContext, handle_input, llm_from_env
from housing_trends.agents.diagnoser import rule_based_llm
from housing_trends.agents.renderer import show_dashboard
from housing_trends.agents.scanner import Snapshot, active_filters, filter_mask
from housing_trends.context import add_months, month_range
from housing_trends.graph import run
from housing_trends.llm import CachedLLM
from housing_trends.state import JUMP_PCT, MAX_DIAG_ROUNDS, MAX_RECOMPUTES, MAX_RESCANS, MIN_LISTINGS

ROOT = pathlib.Path(__file__).resolve().parent
FIELD_NAMES = {"zips": "ZIP codes", "zip": "ZIP codes", "price": "Price range", "price_min": "Minimum price", "price_max": "Maximum price",
               "beds": "Bedrooms", "beds_min": "Minimum bedrooms", "beds_max": "Maximum bedrooms", "baths_min": "Minimum bathrooms",
               "sqft": "Square footage", "sqft_min": "Minimum square footage", "sqft_max": "Maximum square footage",
               "property_types": "Property type", "protected_filters": "Protected filters"}

st.set_page_config(page_title="Dallas housing trends", layout="wide")
st.markdown("""<style>[data-testid="stMetric"] {background: rgba(128,128,128,.08); padding: 14px 18px; border-radius: 12px;}</style>""",
            unsafe_allow_html=True)
st.title("Dallas housing trends")
path = st.sidebar.text_input("Snapshot CSV", "data/snapshot/listings_dfw.csv")


@st.cache_resource
def load_data(p):
    df = pd.read_csv(p, dtype={"zip_code": str})
    ctx = SnapshotContext.from_dataframe(df, type_col="property_type" if "property_type" in df.columns else None)
    return ctx, Snapshot(df)


def snapshot_hash(p):
    try:
        manifest = json.loads((pathlib.Path(p).parent / "snapshot_manifest.json").read_text())
        return manifest["output_sha256"][pathlib.Path(p).name][:12]
    except Exception:
        return ""


def real_llm():
    base = llm_from_env()                                  # raises RuntimeError if LLM2_API_KEY is missing
    return CachedLLM(base, ROOT / "results/llm_cache", tag=base.name)


def friendly(error: str) -> str:
    """'zips: select at least one ZIP code' -> '**ZIP codes:** Select at least one ZIP code'."""
    field, sep, rest = error.partition(": ")
    if sep and field in FIELD_NAMES:
        return f"**{FIELD_NAMES[field]}:** {rest[:1].upper() + rest[1:]}"
    return error[:1].upper() + error[1:]


ctx, snap = load_data(path)


def has_matches(c) -> bool:
    """True if at least one listing in the chosen ZIPs and window passes the filters (read-only use of the Scanner's helpers)."""
    df = snap.df
    in_scope = df.zip_code.isin(c.zips) & (df.month >= c.start_month) & (df.month <= c.end_month)
    return bool(filter_mask(df, c, active_filters(c))[in_scope].any())


request = None                                             # ("form", dict) or ("text", str): set when a button is pressed
form_tab, text_tab = st.tabs(["Form", "Free text"])

with form_tab:
    with st.form("criteria"):
        zips = st.multiselect("ZIP codes", sorted(ctx.valid_zips), format_func=lambda z: f"{z} ({ctx.zip_to_city[z]})")
        c1, c2 = st.columns(2)
        pmin = c1.number_input("Min price (0 = none)", 0, step=25_000)
        pmax = c2.number_input("Max price (0 = none)", 0, step=25_000)
        bmin = c1.number_input("Min beds (0 = none)", 0, 10)
        baths = c2.number_input("Min baths (0 = none)", 0.0, 10.0, step=0.5)
        types = st.multiselect("Property type", sorted(ctx.property_types)) if ctx.property_types else []
        months = month_range(ctx.first_month, ctx.last_month)
        end = st.selectbox("12-month window ends", months, index=len(months) - 1)
        st.caption(f"Window: {add_months(end, -11)} to {end}")
        if st.form_submit_button("Analyze", type="primary"):
            request = ("form", dict(zips=zips, price_min=pmin or None, price_max=pmax or None, beds_min=bmin or None,
                                    baths_min=baths or None, property_types=types, end_month=end))

with text_tab:
    q = st.text_area("Describe what you're looking for",
                     placeholder="affordable family homes in North Dallas, past year")
    if st.button("Analyze", key="analyze_text", type="primary"):
        request = ("text", q)

with st.expander("Analysis options"):
    o1, o2, o3 = st.columns(3)
    llm_choice = o1.radio("Diagnoser LLM", ["Offline stand-in", "Real LLM"],
                          help="The Diagnoser judges outlier listings. The stand-in is rule-based and needs no API key.")
    min_listings = o2.number_input("Coverage threshold (listings per ZIP-month)", 1, 200, MIN_LISTINGS)
    jump_pct = o3.number_input("Price jump threshold (%)", 5.0, 100.0, JUMP_PCT, step=5.0)


def analyse(inp) -> dict:
    """Validated input -> check that something matches -> run the agent graph. Returns what the page should show."""
    out = {"input": inp}
    if not inp.ok:
        return out
    if not has_matches(inp.criteria):
        out["nomatch"] = True
        return out
    llm = real_llm() if llm_choice == "Real LLM" else rule_based_llm
    with st.spinner("Running the agents (scan, statistics, critic, pooling, diagnoser)..."):
        state = run(inp.criteria, snap, llm, min_listings=int(min_listings), jump_pct=float(jump_pct))
    out.update(report=state["report"], llm=getattr(llm, "name", "llm"))
    return out


if request:
    st.session_state.pop("out", None)
    try:
        kind, payload = request
        inp = handle_input(ctx=ctx, form=payload) if kind == "form" else handle_input(ctx=ctx, text=payload, llm=real_llm())
        st.session_state["out"] = analyse(inp)
    except RuntimeError as ex:
        st.session_state["out"] = {"fatal": "An LLM API key is needed for this option. Add LLM2_API_KEY to .env.local, "
                                            "or use the form with the offline stand-in." if "LLM2_API_KEY" in str(ex)
                                   else f"Something went wrong: {ex}"}
    except Exception as ex:
        st.session_state["out"] = {"fatal": f"Something went wrong while running the analysis ({type(ex).__name__}: {ex})."}

# ---------------------------------------------------------------- results (kept in session_state so reruns do not lose them)
out = st.session_state.get("out")
if not out:
    st.info("Choose your criteria above and click **Analyze**.")
elif "fatal" in out:
    st.error(out["fatal"])
else:
    inp = out["input"]
    for w in inp.warnings:
        st.warning(w)
    for a in inp.assumptions:
        st.info(f"Assumption: {a}")
    for u in inp.unsupported:
        st.info(f"Not supported (ignored): {u}")
    if not inp.ok:                                          # invalid criteria: ask the user to fix them, no dashboard
        with st.container(border=True):
            st.markdown("### Please correct your search")
            for e in inp.errors:
                st.markdown(f"- {friendly(e)}")
            st.caption(f"Listings are available for {ctx.first_month} to {ctx.last_month}, and the window must be exactly 12 months.")
    elif out.get("nomatch"):                                # valid criteria, but no listing satisfies them
        with st.container(border=True):
            st.markdown("### No listings match these filters")
            st.write("Try a wider price range, fewer bedroom or bathroom requirements, or more ZIP codes.")
    else:
        show_dashboard(out["report"], zip_to_city=ctx.zip_to_city, llm_name=out["llm"], snapshot_hash=snapshot_hash(path),
                       limits={"rescans": MAX_RESCANS, "recomputes": MAX_RECOMPUTES, "diag_rounds": MAX_DIAG_ROUNDS})
    with st.expander("Input handler log"):
        st.write(inp.trace)
