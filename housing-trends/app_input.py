"""Minimal GUI for the Input Handler.  Run:  streamlit run app_input.py"""
import pandas as pd
import streamlit as st

from housing_trends import SnapshotContext, handle_input, llm_from_env
from housing_trends.context import add_months, month_range

st.title("Dallas housing trends: input")
path = st.sidebar.text_input("Snapshot CSV", "data/snapshot/listings_dfw.csv")


@st.cache_resource
def load_ctx(p):
    df = pd.read_csv(p)
    return SnapshotContext.from_dataframe(df, type_col="property_type" if "property_type" in df.columns else None)


ctx = load_ctx(path)
form_tab, text_tab = st.tabs(["Form", "Free text"])
result = None

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
        if st.form_submit_button("Analyze"):
            result = handle_input(ctx=ctx, form=dict(
                zips=zips, price_min=pmin or None, price_max=pmax or None, beds_min=bmin or None,
                baths_min=baths or None, property_types=types, end_month=end))

with text_tab:
    q = st.text_area("Describe what you're looking for",
                     placeholder="affordable family homes in North Dallas, past year")
    if st.button("Parse request"):
        result = handle_input(ctx=ctx, text=q, llm=llm_from_env())

if result:
    for e in result.errors: st.error(e)
    for w in result.warnings: st.warning(w)
    for a in result.assumptions: st.info(f"Assumption: {a}")
    for u in result.unsupported: st.info(f"Not supported (ignored): {u}")
    if result.ok:
        st.success(f"Criteria accepted ({result.mode}), id {result.criteria.key()}")
        st.json(result.criteria.model_dump())
    with st.expander("Agent decision log"):
        st.write(result.trace)