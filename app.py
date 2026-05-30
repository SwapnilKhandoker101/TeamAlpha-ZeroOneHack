"""Gas-hedging decision agent — dashboard.

Reads the cached real Sybilion forecast and turns it into a hedging decision the
viewer can interrogate, end to end: the Featherless tag-picker that configured
the forecast, the probabilistic price band, the curated drivers (kept vs the
spurious ones thrown out), the per-month hedge ratio the deterministic policy
derives, and a Featherless narrative explaining the decision it did not make.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gas_agent import geo
from gas_agent import sybilion_client as sc
from gas_agent.config import EXPLANATION_MODEL, KEYWORD_MODEL
from gas_agent.decision_backtest import backtest_from_cache
from gas_agent.driver_curation import curate_drivers
from gas_agent.explanation_agent import explain_decision
from gas_agent.hedge_policy import DEFAULT_PARAMS, decide_all, quarter_hedge_ratio
from gas_agent.keyword_agent import DEFAULT_PERSONA, select_filters
from gas_agent.scenario import (
    DEFAULT_SHOCK_PARAMS,
    parse_shock_request,
    run_shock,
    shock_forecast_json,
)

st.set_page_config(page_title="TTF Gas Hedging Agent", layout="wide")

BLUE = "#2563eb"
GREY = "#9ca3af"
GREEN = "#16a34a"
AMBER = "#d97706"
RED = "#dc2626"


@st.cache_data
def load_inputs(job_id: str):
    ttf = sc.load_ttf_series()
    forecast_json = sc.load_artifact(job_id, "forecast.json")
    metrics = sc.load_artifact(job_id, "backtest_metrics.json")
    signals = sc.load_artifact(job_id, "external_signals.json")
    return ttf, forecast_json, metrics, signals


@st.cache_data(show_spinner="Featherless is picking Sybilion filters…")
def cached_selection(persona: str):
    return select_filters(persona)


@st.cache_data(show_spinner="Featherless is explaining the decision…")
def cached_explanation(job_id: str, persona: str, quarter_ratio: float):
    """Rebuild the decision context from the cached job and narrate it. Keyed by
    job + persona + ratio so the (slow) Featherless call runs once per state."""
    ttf = sc.load_ttf_series()
    forecast_json = sc.load_artifact(job_id, "forecast.json")
    signals = sc.load_artifact(job_id, "external_signals.json")
    spot = sc.last_actual_price(ttf)
    quarter = decide_all(sc.parse_forecast_months(forecast_json), spot, DEFAULT_PARAMS)[:3]
    curation = curate_drivers(signals)
    return explain_decision(
        quarter, quarter_ratio, spot, curation.top_drivers(6), curation.rejected, persona
    )


@st.cache_data
def cached_backtest(job_id: str):
    """Replay the hedge policy over the cached backtest windows vs naive baselines."""
    return backtest_from_cache(job_id)


@st.cache_data(show_spinner="Featherless is re-explaining under the shock…")
def cached_shock_explanation(job_id: str, persona: str, magnitude: float, label: str):
    """Narrate the *shocked* decision. Rebuilds the shocked context deterministically
    from the cached job (so the LLM still only explains, never decides), keyed by the
    shock magnitude + label so the slow call runs once per scenario."""
    ttf = sc.load_ttf_series()
    forecast_json = sc.load_artifact(job_id, "forecast.json")
    signals = sc.load_artifact(job_id, "external_signals.json")
    spot = sc.last_actual_price(ttf)
    months = sc.parse_forecast_months(forecast_json)
    outcome = run_shock(months, spot, curate_drivers(signals), magnitude, label)
    quarter = outcome.decisions[:3]
    return explain_decision(
        quarter, quarter_hedge_ratio(quarter), spot,
        outcome.curation.top_drivers(6), outcome.curation.rejected, persona,
    )


@st.cache_data(show_spinner="Featherless is briefing the country…")
def cached_country_brief(region: str, kept_names: tuple[str, ...], credible: bool):
    """Per-country brief. Cached by country + drivers + credibility so the Featherless
    call runs once per country. Pure explanation — never a decision."""
    return geo.country_brief(region, list(kept_names), credible)


def price_band_figure(ttf: dict, forecast_json: dict, history_months: int = 30) -> go.Figure:
    history = ttf["timeseries"]
    hist_dates = list(history)[-history_months:]
    hist_values = [history[d] for d in hist_dates]

    rows = sc.forecast_band_table(forecast_json)
    last_actual_date = list(history)[-1]
    last_actual_value = history[last_actual_date]

    # Prepend the last actual so the forecast lines connect to history.
    fdates = [last_actual_date] + [r["month"] for r in rows]
    median = [last_actual_value] + [r["q50"] for r in rows]
    q10 = [last_actual_value] + [max(0.0, r["q10"]) for r in rows]
    q90 = [last_actual_value] + [r["q90"] for r in rows]
    q05 = [last_actual_value] + [max(0.0, r["q05"]) for r in rows]
    q95 = [last_actual_value] + [r["q95"] for r in rows]

    figure = go.Figure()
    # 90% band (outer, q05-q95)
    figure.add_trace(go.Scatter(x=fdates + fdates[::-1], y=q95 + q05[::-1],
                                fill="toself", fillcolor="rgba(37,99,235,0.10)",
                                line=dict(color="rgba(0,0,0,0)"), name="90% band (q05-q95)",
                                hoverinfo="skip"))
    # 80% band (inner, q10-q90)
    figure.add_trace(go.Scatter(x=fdates + fdates[::-1], y=q90 + q10[::-1],
                                fill="toself", fillcolor="rgba(37,99,235,0.22)",
                                line=dict(color="rgba(0,0,0,0)"), name="80% band (q10-q90)",
                                hoverinfo="skip"))
    # History
    figure.add_trace(go.Scatter(x=hist_dates, y=hist_values, mode="lines",
                                line=dict(color=GREY, width=2), name="TTF actual"))
    # Forecast median
    figure.add_trace(go.Scatter(x=fdates, y=median, mode="lines+markers",
                                line=dict(color=BLUE, width=3, dash="dash"),
                                name="forecast median"))
    figure.add_hline(y=last_actual_value, line=dict(color=AMBER, width=1, dash="dot"),
                     annotation_text=f"today's spot {last_actual_value:.0f}", annotation_position="top left")
    figure.update_layout(height=420, margin=dict(t=30, b=10, l=10, r=10),
                         legend=dict(orientation="h", yanchor="bottom", y=1.0),
                         yaxis_title="EUR/MWh", hovermode="x unified")
    return figure


def hedge_ratio_figure(decisions) -> go.Figure:
    months = [d.month[:7] for d in decisions]
    ratios = [d.hedge_ratio for d in decisions]
    colors = [GREEN if d.direction == "rising" else BLUE for d in decisions]

    figure = go.Figure()
    figure.add_trace(go.Bar(x=months, y=ratios, marker_color=colors,
                            text=[f"{r:.0%}" for r in ratios], textposition="outside",
                            name="hedge ratio"))
    figure.add_hline(y=0.5, line=dict(color=AMBER, width=1, dash="dash"),
                     annotation_text="naive lock 50%", annotation_position="top right")
    figure.update_layout(height=420, margin=dict(t=30, b=10, l=10, r=10),
                         yaxis=dict(title="share locked forward", tickformat=".0%", range=[0, 1]),
                         showlegend=False)
    return figure


def backtest_figure(result) -> go.Figure:
    """Realized cost per strategy; the whiskers are the cost volatility (std) — a
    lower bar is cheaper, a shorter whisker is a steadier bill."""
    names = ["Agent policy", "Always spot (0%)", "Always lock 50%"]
    means = [result.policy.mean_cost, result.always_spot.mean_cost, result.always_half.mean_cost]
    stds = [result.policy.std_cost, result.always_spot.std_cost, result.always_half.std_cost]

    figure = go.Figure()
    figure.add_trace(go.Bar(
        x=names, y=means, marker_color=[GREEN, GREY, AMBER],
        error_y=dict(type="data", array=stds, visible=True, color="#374151", thickness=1.5),
        text=[f"€{m:.1f}" for m in means], textposition="outside",
        hovertemplate="%{x}<br>mean €%{y:.2f}/MWh<extra></extra>",
    ))
    # Zoom the axis so the cost gaps and whiskers are legible (annotated in the caption).
    low = min(m - s for m, s in zip(means, stds))
    high = max(m + s for m, s in zip(means, stds))
    pad = (high - low) * 0.18 or 1.0
    figure.update_layout(height=380, margin=dict(t=30, b=10, l=10, r=10),
                         yaxis=dict(title="realized cost (EUR/MWh)", range=[low - pad, high + pad]),
                         showlegend=False)
    return figure


def curation_figure(curation, top_kept: int = 12) -> go.Figure:
    """Kept (green) vs rejected (red) drivers by importance — curation made visible."""
    kept = curation.kept[:top_kept]
    rejected = curation.rejected

    def short(name: str, width: int = 46) -> str:
        return name if len(name) <= width else name[: width - 1] + "…"

    rows = [(short(d.name), d.importance, GREEN, "kept") for d in kept]
    rows += [(short(d.name), d.importance, RED, "rejected") for d in rejected]
    rows.sort(key=lambda r: r[1])  # ascending so highest ends on top

    # Disambiguate any labels that collide after truncation, so Plotly does not
    # stack two bars onto one categorical row.
    seen: dict[str, int] = {}
    deduped_rows = []
    for label, importance, color, tag in rows:
        seen[label] = seen.get(label, 0) + 1
        unique_label = label if seen[label] == 1 else f"{label} ({seen[label]})"
        deduped_rows.append((unique_label, importance, color, tag))
    rows = deduped_rows

    figure = go.Figure()
    figure.add_trace(go.Bar(
        x=[r[1] for r in rows], y=[r[0] for r in rows], orientation="h",
        marker_color=[r[2] for r in rows],
        text=[r[3] for r in rows], textposition="none",
        hovertemplate="%{y}<br>importance %{x:.0f} (%{text})<extra></extra>",
    ))
    figure.update_layout(height=460, margin=dict(t=30, b=10, l=10, r=10),
                         xaxis_title="Sybilion importance", showlegend=False)
    return figure


# Natural Earth coastlines give the globe a surface; columns/markers ride on top.
_GLOBE_BASEMAP = ("https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/"
                  "ne_50m_admin_0_scale_rank.geojson")


def globe_deck(countries) -> pdk.Deck:
    """A 3D globe: green columns for the credible supplier/hub countries (taller =
    more kept importance) and flat red markers for the spurious-only countries."""
    kept = [c for c in countries if c.has_kept]
    rejected = [c for c in countries if c.rejected_only]
    max_importance = max((c.kept_importance for c in kept), default=1.0)
    elevation_scale = 600_000 / max_importance  # tallest column ≈ 600 km, readable on the sphere

    kept_df = pd.DataFrame([{
        "region": c.region, "lon": c.lon, "lat": c.lat,
        "kept_importance": round(c.kept_importance, 1), "kept_count": c.kept_count,
        "rejected_count": c.rejected_count,
        "tip": f"{c.region}  ·  {c.kept_count} kept driver(s)  ·  importance {c.kept_importance:.0f}",
    } for c in kept])
    rejected_df = pd.DataFrame([{
        "region": c.region, "lon": c.lon, "lat": c.lat,
        "kept_count": 0, "rejected_count": c.rejected_count,
        "tip": f"{c.region}  ·  {c.rejected_count} spurious driver(s) dropped",
    } for c in rejected])

    layers = [
        pdk.Layer("GeoJsonLayer", id="basemap", data=_GLOBE_BASEMAP, stroked=False,
                  filled=True, get_fill_color=[40, 48, 66], get_line_color=[20, 24, 34]),
    ]
    if not kept_df.empty:
        layers.append(pdk.Layer(
            "ColumnLayer", id="kept", data=kept_df, get_position=["lon", "lat"],
            get_elevation="kept_importance", elevation_scale=elevation_scale,
            radius=90_000, get_fill_color=[22, 163, 74, 220], pickable=True, auto_highlight=True))
    if not rejected_df.empty:
        layers.append(pdk.Layer(
            "ScatterplotLayer", id="rejected", data=rejected_df, get_position=["lon", "lat"],
            get_radius=130_000, get_fill_color=[220, 38, 38, 230], pickable=True, auto_highlight=True))

    view = pdk.View(type="_GlobeView", controller=True)
    view_state = pdk.ViewState(latitude=35, longitude=20, zoom=0.35)
    return pdk.Deck(
        views=[view], layers=layers, initial_view_state=view_state,
        map_provider=None, parameters={"cull": True},
        tooltip={"text": "{tip}"},
    )


def _clicked_region(event) -> str | None:
    """Pull the region of a clicked globe object out of the pydeck selection event,
    tolerating the slightly different shapes Streamlit returns across versions."""
    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        selection = event.get("selection")
    objects = (selection or {}).get("objects") if isinstance(selection, dict) else None
    if isinstance(objects, dict):
        for rows in objects.values():
            if rows:
                return rows[0].get("region")
    return None


SHOCK_BUTTON_MESSAGE = "Iran closes the Strait of Hormuz"


def _set_shock(magnitude: float, label: str) -> None:
    st.session_state.shock_magnitude = magnitude
    st.session_state.shock_label = label


def _shock_reply(months, spot, curation, magnitude: float, label: str) -> str:
    """The assistant's chat reply: what the deterministic policy did with the shock.
    The number here is computed by the policy, not written by an LLM."""
    baseline = quarter_hedge_ratio(decide_all(months, spot, DEFAULT_PARAMS)[:3])
    outcome = run_shock(months, spot, curation, magnitude, label)
    shocked = quarter_hedge_ratio(outcome.decisions[:3])
    arrow = "up" if shocked > baseline else ("down" if shocked < baseline else "unchanged")
    return (
        f"**{label}** read as severity {magnitude:.0%}. I re-ran the deterministic policy: "
        f"forward median lifts, the band widens, and a supply-risk premium of "
        f"+{outcome.risk_premium:.0%} raises the lock floor.\n\n"
        f"Next-quarter hedge ratio moves **{baseline:.0%} → {shocked:.0%}** ({arrow}). "
        f"The charts and the driver mix on the left have updated; '{outcome.curation.kept[0].name}' "
        f"now leads the drivers. Say *calm* or hit Reset to return to the base case."
    )


def _freeform_reply(message: str) -> str:
    """Free-form path (toggle): re-run the keyword agent on the typed context so the
    LLM→Sybilion-config loop visibly responds. Selection only — no live forecast poll
    on stage. Featherless picks filters from the catalog; it never sets the ratio."""
    persona = f"{DEFAULT_PERSONA}\n\nAdditional context from the user: {message.strip()}"
    selection = select_filters(persona)
    cats = ", ".join(selection.category_names()) or "—"
    regions = ", ".join(selection.region_names()) or "—"
    return (
        f"Re-selected Sybilion filters from your prompt (source: {selection.source}).\n\n"
        f"**Categories:** {cats}\n\n**Regions:** {regions}\n\n"
        f"**Keywords:** {' · '.join(selection.keywords) or '—'}\n\n"
        "A full live re-forecast (submit → poll) is disabled for the demo, so the cached "
        "forecast stays on screen — but this shows the agent re-configuring Sybilion from "
        "free text. Toggle this off to drive the guided supply-shock instead."
    )


def scenario_sidebar(months, spot, curation) -> None:
    """Live scenario controls + chat. Mutates ``st.session_state`` (shock magnitude
    + label + message log); the main body reads that state and re-renders."""
    with st.sidebar:
        st.header("Live scenario")
        st.caption("Type a supply-shock headline and the agent re-decides on the spot — "
                   "the LLM only reads the severity, the deterministic policy moves the ratio.")

        active = st.session_state.shock_magnitude > 0
        button_cols = st.columns(2)
        if button_cols[0].button("⚡ Strait of Hormuz", use_container_width=True,
                                 help=f"Fires the canonical full-severity shock: '{SHOCK_BUTTON_MESSAGE}'."):
            _set_shock(1.0, "Strait of Hormuz disruption")
            st.session_state.messages.append({"role": "user", "content": SHOCK_BUTTON_MESSAGE})
            st.session_state.messages.append({
                "role": "assistant",
                "content": _shock_reply(months, spot, curation, 1.0, "Strait of Hormuz disruption"),
            })
            st.rerun()
        if button_cols[1].button("Reset to calm", use_container_width=True, disabled=not active):
            _set_shock(0.0, "")
            st.session_state.messages.append({"role": "assistant",
                                              "content": "Back to the calm base case."})
            st.rerun()

        use_llm = st.toggle("Let Featherless classify the message",
                            value=False, key="use_llm_classify",
                            help="On: a small model reads severity + a label (never the ratio). "
                                 "Off: deterministic keyword parse.")
        freeform = st.toggle("Free-form re-forecast (re-pick Sybilion filters)",
                             value=False, key="freeform_reforecast",
                             help="On: a non-shock message re-runs the keyword agent to "
                                  "reconfigure Sybilion. Off (default): guided supply-shock only.")

        st.divider()
        for message in st.session_state.messages:
            st.chat_message(message["role"]).write(message["content"])

        if prompt := st.chat_input("e.g. Iran closes the Strait of Hormuz"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            request = parse_shock_request(prompt, use_llm=use_llm)
            if request.is_shock:
                _set_shock(request.magnitude, request.label)
                reply = _shock_reply(months, spot, curation, request.magnitude, request.label)
            elif freeform:
                reply = _freeform_reply(prompt)
            else:
                _set_shock(0.0, "")
                reply = ("No supply shock detected, so I'm showing the calm base case. "
                         "Try something like *'Iran closes the Strait of Hormuz'* or "
                         "*'new sanctions on Russian gas'* — or flip on free-form re-forecast.")
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.rerun()


def main() -> None:
    st.title("European gas (TTF) hedging agent")
    st.caption("How much of next quarter's gas should an energy-intensive EU buyer "
               "lock in forward now, versus leave to spot? A decision built on the "
               "Sybilion forecast's confidence band — not its point estimate.")

    job_id = sc.get_latest_job()
    if not job_id:
        st.error("No cached forecast found. Run a forecast first.")
        return

    ttf, forecast_json, metrics, signals = load_inputs(job_id)
    months = sc.parse_forecast_months(forecast_json)
    spot = sc.last_actual_price(ttf)
    base_curation = curate_drivers(signals)
    selection = cached_selection(DEFAULT_PERSONA)

    # Chat / scenario state. The sidebar mutates these; the body reads them.
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("shock_magnitude", 0.0)
    st.session_state.setdefault("shock_label", "")
    scenario_sidebar(months, spot, base_curation)

    # Resolve what to display: the calm base case, or — if a shock is active — the
    # same deterministic policy re-run over the shocked inputs.
    magnitude = st.session_state.shock_magnitude
    shock_active = magnitude > 0
    if shock_active:
        label = st.session_state.shock_label
        shock = run_shock(months, spot, base_curation, magnitude, label)
        decisions = shock.decisions
        curation = shock.curation
        display_forecast_json = shock_forecast_json(forecast_json, magnitude)
    else:
        decisions = decide_all(months, spot, DEFAULT_PARAMS)
        curation = base_curation
        display_forecast_json = forecast_json

    quarter_months = decisions[:3]
    quarter_ratio = quarter_hedge_ratio(quarter_months)
    quarter_label = " / ".join(d.month[:7] for d in quarter_months)
    mape = metrics["data"]["12m"]["metrics"]["MAPE"]

    st.caption("**Pipeline:** Featherless picks the Sybilion filters  →  Sybilion returns the "
               "forecast + drivers  →  curation drops the spurious drivers  →  a deterministic "
               "policy sets the hedge ratio  →  Featherless explains it. The LLMs prepare and "
               "narrate; they never decide the ratio.")

    calm_ratio = quarter_hedge_ratio(decide_all(months, spot, DEFAULT_PARAMS)[:3])
    if shock_active:
        st.error(f"⚡ **Scenario active — {st.session_state.shock_label}** (severity {magnitude:.0%}). "
                 f"Forward lifted, band widened, supply-risk premium +{shock.risk_premium:.0%} on the "
                 "lock floor. Every panel below is the *same deterministic policy* re-run on the "
                 "shocked inputs — use the sidebar to reset.")

    top = st.columns([1, 1, 1])
    top[0].metric(f"Lock now — next quarter ({quarter_label})", f"{quarter_ratio:.0%}",
                  delta=f"{quarter_ratio - calm_ratio:+.0%} vs calm" if shock_active else None)
    top[1].metric("vs naive baselines", "0% spot / 50% lock")
    top[2].metric("Forecast point-accuracy (backtest MAPE)", f"{mape:.0f}%",
                  help="The point forecast is weak, which is exactly why the decision "
                       "is built on the confidence band and drivers instead.")

    with st.expander(f"Step 1 — how the agent configured the forecast  ·  "
                     f"Featherless {KEYWORD_MODEL.split('/')[-1]}  ·  source: {selection.source}"):
        if selection.notes:
            st.caption(selection.notes)
        cols = st.columns(2)
        cols[0].markdown("**Categories selected**\n\n" +
                         "\n".join(f"- {name}" for name in selection.category_names()))
        cols[1].markdown("**Regions selected**\n\n" +
                         ", ".join(selection.region_names()))
        st.markdown("**Keywords:** " + " · ".join(f"`{k}`" for k in selection.keywords))
        st.caption(f"recency factor {selection.recency_factor:.2f} — higher leans on recent data.")

    left, right = st.columns(2)
    with left:
        st.subheader("Probabilistic price forecast")
        st.plotly_chart(price_band_figure(ttf, display_forecast_json), use_container_width=True)
    with right:
        st.subheader("The decision — hedge ratio per month")
        st.plotly_chart(hedge_ratio_figure(decisions), use_container_width=True)

    st.subheader("Did the decision beat the naive baselines?")
    backtest = cached_backtest(job_id)
    bt_chart, bt_stats = st.columns([3, 2])
    with bt_chart:
        st.plotly_chart(backtest_figure(backtest), use_container_width=True)
    with bt_stats:
        st.metric("Cheaper than buying spot", f"€{backtest.cost_saving_vs_spot:+.2f}/MWh",
                  help="Mean realized cost of the policy vs always buying on the spot market.")
        st.metric("Steadier than buying spot", f"€{backtest.volatility_drop_vs_spot:+.2f}/MWh",
                  help="Reduction in cost volatility (std) vs always-spot — the point of hedging.")
        st.metric("vs a static 50% lock", f"€{backtest.cost_gap_vs_half:+.2f}/MWh",
                  help="Policy mean cost minus a mechanical 50% lock (negative = the policy is cheaper).")
    st.caption(backtest.verdict)
    st.caption(f"Replayed over {backtest.n_months} backtested months. The lock price is proxied by the "
               "decision-time spot — Sybilion's weak point forecast is used only to *size* the hedge ratio, "
               "never as the price you pay. Hedging trades a little average cost for a steadier bill; here "
               "the policy beats do-nothing spot on both cost and volatility.")

    st.subheader("Why — the agent's explanation")
    if shock_active:
        explanation = cached_shock_explanation(
            job_id, DEFAULT_PERSONA, round(magnitude, 4), st.session_state.shock_label)
    else:
        explanation = cached_explanation(job_id, DEFAULT_PERSONA, round(quarter_ratio, 4))
    st.info(explanation.text)
    if explanation.source == "llm":
        narrated = "the shocked drivers" if shock_active else "the curated drivers"
        st.caption(f"Generated by Featherless {explanation.model.split('/')[-1]} from {narrated} "
                   "and the policy's own numbers. It explains the decision — it does not make it.")
    else:
        st.caption("Featherless unavailable — deterministic fallback narrative built from the same numbers.")

    st.subheader("Driver curation — what the agent trusted vs threw out")
    cur_left, cur_right = st.columns([3, 2])
    with cur_left:
        st.plotly_chart(curation_figure(curation), use_container_width=True)
        st.caption(f"{curation.kept_count} credible drivers kept · "
                   f"{curation.rejected_count} spurious dropped. Sybilion ranks by correlation; "
                   "the agent keeps only those with a plausible causal path to European gas.")
    with cur_right:
        st.markdown("**Dropped as spurious**")
        rejected_table = pd.DataFrame([{
            "driver": d.name,
            "importance": round(d.importance, 0),
            "why dropped": d.reason,
        } for d in curation.rejected])
        st.dataframe(rejected_table, use_container_width=True, hide_index=True, height=240)

    st.subheader("Where the drivers live — the agent's world view")
    st.caption("Green columns are the suppliers and hubs the agent trusts (taller = more "
               "kept importance); red markers are countries that only surfaced through "
               "spurious correlations. Drag to spin the globe and click a country — or pick "
               "one below — for a Featherless brief on why it does (or doesn't) move European gas." +
               (" Under the shock, the risk supplier lights up." if shock_active else ""))
    countries = geo.aggregate_drivers(curation)
    globe_col, detail_col = st.columns([3, 2])
    with globe_col:
        event = st.pydeck_chart(globe_deck(countries), use_container_width=True,
                                on_select="rerun", selection_mode="single-object", key="globe")
        clicked = _clicked_region(event)
    with detail_col:
        names = [c.region for c in countries]
        # Keep the picker valid as the country set changes (e.g. a shock adds Iran).
        if st.session_state.get("country_pick") not in names:
            st.session_state.pop("country_pick", None)
        if clicked in names:
            st.session_state["country_pick"] = clicked  # a globe click drives the picker
        region = st.selectbox("Inspect a country", names, key="country_pick") if names else None
        if region:
            agg = next(c for c in countries if c.region == region)
            if agg.has_kept:
                st.markdown(f"**{region}** — {agg.kept_count} kept driver(s) · "
                            f"importance {agg.kept_importance:.0f}")
                brief, source = cached_country_brief(region, tuple(agg.kept_names), True)
            else:
                st.markdown(f"**{region}** — {agg.rejected_count} spurious driver(s), dropped")
                brief, source = cached_country_brief(region, tuple(agg.rejected_names), False)
            st.info(brief)
            st.caption(f"Featherless {EXPLANATION_MODEL.split('/')[-1]} — explanation only, no decision."
                       if source == "llm" else "Offline brief — Featherless unavailable.")
            if agg.kept_names:
                st.caption("Kept: " + ", ".join(agg.kept_names))
            if agg.rejected_names:
                st.caption("Dropped: " + ", ".join(agg.rejected_names))

    st.subheader("How each month's decision was reached")
    table = pd.DataFrame([{
        "month": d.month[:7],
        "median (EUR/MWh)": round(d.median, 1),
        "band width": f"{d.band_width:.0%}",
        "forward vs spot": f"{d.drift_pct:+.0%}",
        "band component": round(d.ratio_from_band, 2),
        "drift tilt": f"{d.direction_tilt:+.2f}",
        "hedge ratio": f"{d.hedge_ratio:.0%}",
        "why": d.reason,
    } for d in decisions])
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption(f"Forecast job {job_id} — cached real Sybilion output. "
               "The hedge ratio is computed by deterministic code, not an LLM.")


if __name__ == "__main__":
    main()
