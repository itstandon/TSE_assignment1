"""Rendering Agent (plain code): assemble the final, annotated result. Presentation only: it copies statistics,
never computes or alters them."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Report:
    criteria: object
    scope: object
    buckets: pd.DataFrame
    overall: pd.DataFrame
    explanations: list = field(default_factory=list)
    verdicts: list = field(default_factory=list)
    unresolved: list = field(default_factory=list)
    counters: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)


def _notes(r, persistent):
    n = []
    if r.n_pooled > 0:
        n.append(f"pooled (+-{r.time_pad_used} mo, +{r.zip_extra_used} ZIPs)")
    if r.thin:
        n.append(f"thin: {r.n_used} listings")
    if r.method != "median":
        n.append(r.method)
    if r.n_excluded > 0:
        n.append(f"{r.n_excluded} listing(s) excluded")
    if f"{r.zip_code}@{r.month}" in persistent:
        n.append("price jump unresolved")
    return "; ".join(n)


def make_renderer():
    def render(state):
        scan, stats = state["scan"], state["stats"]
        persistent = set(state.get("persistent", []))
        extra = scan.buckets[["zip_code", "month", "thin", "widened", "zip_extra_used", "time_pad_used"]]
        b = stats.buckets.merge(extra, on=["zip_code", "month"], how="left")
        b["notes"] = [_notes(r, persistent) for r in b.itertuples()]
        diag = state.get("diagnoses", [])
        unresolved = list(state.get("unresolved", []))
        bad = [d for d in diag if d["type"] == "verdict" and d["verdict"] == "unexplained"]
        if bad:
            unresolved.append(f"{len(bad)} outlier listing(s) excluded as unexplained data-quality issue(s): "
                              + ", ".join(str(d["listing_id"]) for d in bad))
        counters = {k: state.get(k, 0) for k in ("rescans", "recomputes", "diag_rounds")}
        rep = Report(state["criteria"], state.get("scope"), b, stats.overall,
                     list({(d["zip_code"], d["month"]): d for d in diag if d["type"] == "explanation"}.values()),
                     [d for d in diag if d["type"] == "verdict"],
                     unresolved, counters, list(state.get("trace", [])))
        return {"report": rep, "trace": [f"render: {len(b)} buckets, {len(unresolved)} unresolved item(s), loops used {counters}"]}
    return render


# ---------------------------------------------------------------- display (Streamlit)
# The Rendering Agent's second half: show a finished Report. It only reads the Report and never recomputes a statistic.
# streamlit / altair are imported inside the functions so the graph and the tests do not need them.
ACCENT, MUTED, UP, DOWN = "#2563eb", "#94a3b8", "#16a34a", "#dc2626"


def _money(x):
    return "n/a" if x is None or pd.isna(x) else f"${x:,.0f}"


def effective_filters(c, relaxed=()) -> str:
    """One line describing the filters that were actually applied (and any that Pooling relaxed)."""
    parts = []
    if c.price_min or c.price_max:
        parts.append(f"price {_money(c.price_min) if c.price_min else 'any'} to {_money(c.price_max) if c.price_max else 'any'}")
    if c.beds_min:
        parts.append(f"{c.beds_min}+ beds")
    if c.baths_min:
        parts.append(f"{c.baths_min:g}+ baths")
    if c.property_types:
        parts.append(", ".join(c.property_types))
    text = " · ".join(parts) or "no extra filters"
    return text + (f" · relaxed by Pooling: {', '.join(relaxed)}" if relaxed else "")


def _with_labels(df):
    df = df.copy()
    df["label"] = pd.to_datetime(df.month).dt.strftime("%b %y")
    return df


def _line_chart(df, y, fmt, title):
    import altair as alt
    df = _with_labels(df.dropna(subset=[y]))
    base = alt.Chart(df).encode(
        x=alt.X("label:O", title=None, sort=list(df.label), axis=alt.Axis(labelAngle=0, labelOverlap=True)),
        y=alt.Y(f"{y}:Q", title=None, scale=alt.Scale(zero=False), axis=alt.Axis(format=fmt)),
        tooltip=[alt.Tooltip("label:O", title="Month"), alt.Tooltip(f"{y}:Q", title=title, format=fmt)])
    return (base.mark_line(color=ACCENT, strokeWidth=3) + base.mark_circle(color=ACCENT, size=70)).properties(height=280)


def _bar_chart(df, y, fmt, title, signed=False):
    import altair as alt
    df = _with_labels(df.dropna(subset=[y]))
    color = alt.condition(f"datum.{y} >= 0", alt.value(UP), alt.value(DOWN)) if signed else alt.value(ACCENT)
    return alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("label:O", title=None, sort=list(df.label), axis=alt.Axis(labelAngle=0, labelOverlap=True)),
        y=alt.Y(f"{y}:Q", title=None, axis=alt.Axis(format=fmt)), color=color,
        tooltip=[alt.Tooltip("label:O", title="Month"), alt.Tooltip(f"{y}:Q", title=title, format=fmt)]).properties(height=240)


def show_dashboard(report: Report, *, zip_to_city: dict | None = None, llm_name: str = "", snapshot_hash: str = "",
                   limits: dict | None = None) -> None:
    """Display the 12-month trend dashboard for one Report.
    limits: loop caps for the counters, e.g. {"rescans": 3, "recomputes": 1, "diag_rounds": 2}."""
    import altair as alt
    import streamlit as st

    c, overall, buckets = report.criteria, report.overall, report.buckets
    zip_to_city, limits = zip_to_city or {}, limits or {}
    st.markdown(f"**{c.start_month} to {c.end_month}** · {len(c.zips)} ZIP codes · "
                f"{effective_filters(c, getattr(report.scope, 'relaxed', ()))}")
    tab_trend, tab_zip, tab_quality, tab_log = st.tabs(["Trend", "ZIP breakdown", "Coverage & warnings", "Decision log"])

    # ---- the 12-month trend (the same series run_pipeline.py prints as "OVERALL TREND")
    with tab_trend:
        valid = overall.dropna(subset=["price"])
        first, last, ppsf = valid.iloc[0], valid.iloc[-1], valid.dropna(subset=["ppsf"])
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Median list price", _money(last.price), f"{(last.price / first.price - 1) * 100:+.1f}% over the window", delta_color="off")
        if len(ppsf):
            k2.metric("Median $ per sq ft", f"${ppsf.iloc[-1].ppsf:,.0f}",
                      f"{(ppsf.iloc[-1].ppsf / ppsf.iloc[0].ppsf - 1) * 100:+.1f}% over the window", delta_color="off")
        k3.metric("Listings analysed", f"{int(overall.n.sum()):,}")
        k4.metric("ZIP codes", len(c.zips))
        st.write("")
        left, right = st.columns(2)
        with left:
            st.markdown("#### Median list price")
            st.altair_chart(_line_chart(overall, "price", "$,.0f", "Median price"), width="stretch")
        with right:
            st.markdown("#### Median price per sq ft")
            st.altair_chart(_line_chart(overall, "ppsf", "$,.0f", "Median $/sqft"), width="stretch")
        left, right = st.columns(2)
        with left:
            st.markdown("#### Month-over-month price change")
            st.caption("Green = up, red = down")
            st.altair_chart(_bar_chart(overall, "mom_price_pct", ".1f", "Change (%)", signed=True), width="stretch")
        with right:
            st.markdown("#### Active listings per month")
            st.altair_chart(_bar_chart(overall, "n", ",d", "Listings"), width="stretch")
        st.markdown("#### 12-month trend table")
        table = overall[["month", "n", "price", "ppsf", "mom_price_pct"]].round(1)
        table.columns = ["Month", "Listings", "Median price", "$ per sq ft", "Price change vs prior month (%)"]
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "Median price": st.column_config.NumberColumn(format="$%d"), "$ per sq ft": st.column_config.NumberColumn(format="$%.1f"),
            "Price change vs prior month (%)": st.column_config.NumberColumn(format="%+.1f%%")})

    # ---- per-ZIP breakdown
    with tab_zip:
        rows = []
        for z, g in buckets.groupby("zip_code"):
            p = g.sort_values("month").dropna(subset=["price"])
            if p.empty:
                continue
            rows.append({"ZIP": z, "City": zip_to_city.get(z, ""), "First month": p.price.iloc[0], "Latest month": p.price.iloc[-1],
                         "Change %": (p.price.iloc[-1] / p.price.iloc[0] - 1) * 100, "Listings": int(g.n_own.sum()),
                         "Months pooled": int((g.n_pooled > 0).sum())})
        ztab = pd.DataFrame(rows)
        st.markdown("#### All ZIP codes")
        st.caption("Click a column header to sort.")
        st.dataframe(ztab, hide_index=True, width="stretch", height=300, column_config={
            "First month": st.column_config.NumberColumn(format="$%d"), "Latest month": st.column_config.NumberColumn(format="$%d"),
            "Change %": st.column_config.NumberColumn(format="%+.1f%%"),
            "Months pooled": st.column_config.NumberColumn(help="Months where Pooling borrowed listings from neighbouring months or ZIPs")})
        st.write("")
        pick = st.selectbox("Compare a ZIP with the whole market", list(ztab.ZIP) if len(ztab) else [])
        if pick:
            one = buckets[buckets.zip_code == pick].sort_values("month")
            both = pd.concat([one.assign(series=f"ZIP {pick}")[["month", "price", "series"]],
                              overall.assign(series="All ZIPs")[["month", "price", "series"]]])
            both = _with_labels(both.dropna(subset=["price"]))
            order = sorted(both.label.unique(), key=lambda s: pd.to_datetime(s, format="%b %y"))
            chart = alt.Chart(both).mark_line(point=True, strokeWidth=3).encode(
                x=alt.X("label:O", title=None, sort=order, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("price:Q", title=None, scale=alt.Scale(zero=False), axis=alt.Axis(format="$,.0f")),
                color=alt.Color("series:N", title=None, scale=alt.Scale(domain=[f"ZIP {pick}", "All ZIPs"], range=[ACCENT, MUTED])),
                tooltip=["label", "series", alt.Tooltip("price:Q", format="$,.0f")]).properties(height=300)
            st.altair_chart(chart, width="stretch")
            notes = one[one.notes != ""][["month", "n_used", "notes"]].rename(
                columns={"month": "Month", "n_used": "Listings used", "notes": "What the agents did"})
            if len(notes):
                with st.expander(f"Adjustments the agents made in ZIP {pick} ({len(notes)} months)"):
                    st.dataframe(notes, hide_index=True, width="stretch")

    # ---- coverage metrics and warnings
    with tab_quality:
        st.markdown("#### Feedback loops that fired")
        st.caption("How many times each loop ran, out of its cap.")
        cnt = report.counters

        def used(key):
            return f"{cnt.get(key, 0)} of {limits[key]}" if key in limits else str(cnt.get(key, 0))
        l1, l2, l3 = st.columns(3)
        l1.metric("Coverage rescans (Pooling)", used("rescans"))
        l2.metric("Method recomputes (Statistician)", used("recomputes"))
        l3.metric("Diagnoser rounds", used("diag_rounds"))
        st.write("")
        st.markdown("#### Coverage")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("ZIP-months analysed", len(buckets))
        q2.metric("Needed extra data", int((buckets.n_pooled > 0).sum()), help="Pooling borrowed listings from neighbouring months or ZIPs")
        q3.metric("Still below threshold", int(buckets.thin.sum()))
        q4.metric("Listings excluded", int(overall.n_excluded.sum()), help="Removed as bad listings by the Diagnoser")
        st.write("")
        st.markdown("#### Open issues")
        if report.unresolved:
            for u in report.unresolved:
                st.warning(u)
        else:
            st.success("Nothing left unresolved: every check passed.")
        st.write("")
        st.markdown("#### Diagnoser findings")
        if report.verdicts or report.explanations:
            for ex in report.explanations:
                st.markdown(f"**{ex['zip_code']} · {ex['month']}**  \n{ex['text']}")
            if report.verdicts:
                vt = pd.DataFrame(report.verdicts)[["listing_id", "zip_code", "month", "price", "verdict", "reason"]]
                vt.columns = ["Listing", "ZIP", "Month", "Price", "Verdict", "Reason"]
                st.dataframe(vt, hide_index=True, width="stretch", column_config={"Price": st.column_config.NumberColumn(format="$%d")})
        else:
            st.caption("The Diagnoser was not needed: no unexplained price jumps survived the earlier checks.")

    # ---- agent decision log
    with tab_log:
        st.markdown("#### Decision log")
        st.caption("Every step each agent took, in order.")
        st.caption(" · ".join(x for x in (f"Snapshot {snapshot_hash}" if snapshot_hash else "", f"Diagnoser: {llm_name}" if llm_name else "",
                                          f"criteria id {c.key()}") if x))
        log = "\n".join(report.trace)
        st.code(log, language=None)
        st.download_button("Download log", log, file_name=f"trace_{c.key()}.txt")