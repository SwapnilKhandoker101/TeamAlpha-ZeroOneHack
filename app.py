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

from gas_agent import sybilion_client as sc
from gas_agent.config import EXPLANATION_MODEL, KEYWORD_MODEL
from gas_agent.driver_curation import curate_drivers
from gas_agent.explanation_agent import explain_decision
from gas_agent.hedge_policy import DEFAULT_PARAMS, decide_all, quarter_hedge_ratio
from gas_agent.keyword_agent import DEFAULT_PERSONA, select_filters

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
    decisions = decide_all(months, spot, DEFAULT_PARAMS)
    curation = curate_drivers(signals)
    selection = cached_selection(DEFAULT_PERSONA)

    quarter_months = decisions[:3]
    quarter_ratio = quarter_hedge_ratio(quarter_months)
    quarter_label = " / ".join(d.month[:7] for d in quarter_months)
    mape = metrics["data"]["12m"]["metrics"]["MAPE"]

    st.caption("**Pipeline:** Featherless picks the Sybilion filters  →  Sybilion returns the "
               "forecast + drivers  →  curation drops the spurious drivers  →  a deterministic "
               "policy sets the hedge ratio  →  Featherless explains it. The LLMs prepare and "
               "narrate; they never decide the ratio.")

    top = st.columns([1, 1, 1])
    top[0].metric(f"Lock now — next quarter ({quarter_label})", f"{quarter_ratio:.0%}")
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
        st.plotly_chart(price_band_figure(ttf, forecast_json), use_container_width=True)
    with right:
        st.subheader("The decision — hedge ratio per month")
        st.plotly_chart(hedge_ratio_figure(decisions), use_container_width=True)

    st.subheader("Why — the agent's explanation")
    explanation = cached_explanation(job_id, DEFAULT_PERSONA, round(quarter_ratio, 4))
    st.info(explanation.text)
    if explanation.source == "llm":
        st.caption(f"Generated by Featherless {explanation.model.split('/')[-1]} from the curated "
                   "drivers and the policy's own numbers. It explains the decision — it does not make it.")
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
