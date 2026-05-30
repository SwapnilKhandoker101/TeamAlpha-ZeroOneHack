"""The ceramics optimizer tab — inputs in, an auditable recommendation out.

Rendered as a second tab inside the existing Streamlit app (the gas tab is
untouched). It takes a product / quantity / timeline / cost-factor weights /
competition / target-month, runs the deterministic pipeline
(:func:`ceramics_agent.recommend.build_recommendation`), and lays out the result
the same way the gas tab does: the decision up top, then the evidence — the cost
forecast band, the curated suppliers and channels, the negotiation trace, the
recommendation callout, the Featherless explanation (voiced), the three-strategy
backtest, and a CSV export.

Everything heavy is wrapped in ``@st.cache_data`` keyed by *all* inputs + the
forecast job, so the tab is instant on repeat and identical inputs always render
identical numbers. The live Sybilion 4-factor refresh is opt-in and off by default
(behind ``SYBILION_API_KEY``); with no key the tab runs entirely on the committed
mock and shows an offline banner — the same no-keys guarantee the gas tab makes.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gas_agent import config
from gas_agent import sybilion_client as sc

from ceramics_agent import forecast as cforecast
from ceramics_agent.catalog import list_products
from ceramics_agent.cost_policy import CostWeights
from ceramics_agent.explanation import explain_recommendation
from ceramics_agent.recommend import Recommendation, available_months, build_recommendation

BLUE = "#2563eb"
GREY = "#9ca3af"
GREEN = "#16a34a"
AMBER = "#d97706"
RED = "#dc2626"


# --------------------------------------------------------------------------- #
# Cached compute — keyed by all inputs + job so repeats are instant & identical
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Optimising the production run…")
def _cached_recommendation(
    product_id: str,
    quantity: int,
    timeline_days: int,
    weights_tuple: tuple[float, float, float, float],
    competition: str,
    target_month: str,
    job_id: str | None,
) -> Recommendation:
    weights = CostWeights(*weights_tuple)
    return build_recommendation(
        product_id, quantity, timeline_days, weights, competition,
        target_month=target_month, job_id=job_id,
    )


@st.cache_data(show_spinner="Featherless is explaining the recommendation…")
def _cached_explanation(
    product_id: str,
    quantity: int,
    timeline_days: int,
    weights_tuple: tuple[float, float, float, float],
    competition: str,
    target_month: str,
    job_id: str | None,
):
    """Rebuild the (cheap, deterministic) recommendation and narrate it. Keyed by
    the same inputs so the slow Featherless call runs once per state — the LLM only
    explains, it never changes a number."""
    weights = CostWeights(*weights_tuple)
    rec = build_recommendation(
        product_id, quantity, timeline_days, weights, competition,
        target_month=target_month, job_id=job_id,
    )
    return explain_recommendation(rec)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def cost_band_figure(rec: Recommendation) -> go.Figure:
    """Per-unit physical cost band (q10–q90 EUR/unit) over the forecast horizon.
    The median line is green when the cost outlook falls, red when it rises."""
    band = rec.cost_band
    months = [c.month[:7] for c in band]
    q10 = [c.q10 for c in band]
    q50 = [c.q50 for c in band]
    q90 = [c.q90 for c in band]
    rising = len(q50) >= 2 and q50[-1] > q50[0]
    median_color = RED if rising else GREEN

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=months + months[::-1], y=q90 + q10[::-1], fill="toself",
        fillcolor="rgba(37,99,235,0.15)", line=dict(color="rgba(0,0,0,0)"),
        name="80% band (q10–q90)", hoverinfo="skip",
    ))
    figure.add_trace(go.Scatter(
        x=months, y=q50, mode="lines+markers",
        line=dict(color=median_color, width=3),
        name="median EUR/unit",
        hovertemplate="%{x}<br>€%{y:.2f}/unit<extra></extra>",
    ))
    figure.update_layout(
        height=340, margin=dict(t=30, b=10, l=10, r=10),
        yaxis_title="EUR / unit", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0),
    )
    return figure


def backtest_bars_figure(rec: Recommendation) -> go.Figure:
    """Mean realized margin per strategy (whiskers = month-to-month std). Agent green,
    random grey, cheap heuristic amber — higher is better (margin, not cost)."""
    bt = rec.backtest
    names = ["Agent policy", "Random pick", "Cheap + best margin"]
    means = [bt.agent.mean, bt.random_.mean, bt.cheap.mean]
    stds = [bt.agent.std, bt.random_.std, bt.cheap.std]

    figure = go.Figure()
    figure.add_trace(go.Bar(
        x=names, y=means, marker_color=[GREEN, GREY, AMBER],
        error_y=dict(type="data", array=stds, visible=True, color="#374151", thickness=1.5),
        text=[f"€{m:,.0f}" for m in means], textposition="outside",
        hovertemplate="%{x}<br>mean €%{y:,.0f}/month<extra></extra>",
    ))
    figure.update_layout(
        height=360, margin=dict(t=30, b=10, l=10, r=10),
        yaxis=dict(title="realized margin (EUR/month)"), showlegend=False,
    )
    return figure


# --------------------------------------------------------------------------- #
# Card / table renderers
# --------------------------------------------------------------------------- #
def _supplier_card(col, curated, *, chosen: bool) -> None:
    mark = " ✅" if chosen else ""
    border = GREEN if chosen else "#374151"
    col.markdown(
        f"<div style='border:1px solid {border};border-radius:8px;padding:10px 12px'>"
        f"<b>{curated.name}{mark}</b><br>"
        f"<span style='color:{GREY}'>{curated.supplier.region} · "
        f"lead {curated.supplier.lead_time_days}d · {curated.supplier.reliability_pct:.0f}% on-time</span><br>"
        f"<b>score {curated.total_score:.0f}</b> "
        f"<span style='color:{GREY}'>(cost {curated.cost_score:.0f} · "
        f"rel {curated.reliability_score:.0f} · lead {curated.lead_time_score:.0f})</span><br>"
        f"<span style='color:{GREY};font-size:0.85em'>{curated.reason}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _channel_card(col, curated, *, chosen: bool) -> None:
    mark = " ✅" if chosen else ""
    border = GREEN if chosen else "#374151"
    col.markdown(
        f"<div style='border:1px solid {border};border-radius:8px;padding:10px 12px'>"
        f"<b>{curated.name}{mark}</b><br>"
        f"<span style='color:{GREY}'>min order {curated.channel.min_order} · "
        f"target margin {curated.channel.target_margin:.0%}</span><br>"
        f"<b>score {curated.total_score:.0f}</b> "
        f"<span style='color:{GREY}'>(margin {curated.margin_score:.0f} · "
        f"season {curated.season_score:.0f} · order-fit {curated.order_score:.0f})</span><br>"
        f"<span style='color:{GREY};font-size:0.85em'>{curated.reason}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _negotiation_table(rec: Recommendation) -> pd.DataFrame:
    def fmt(row) -> str:
        return f"€{row.value:.2f}" if row.kind == "price" else f"{row.value:.0%}"

    return pd.DataFrame([{
        "round": row.round,
        "party": row.party,
        "quote": fmt(row),
        "reason": row.reason,
    } for row in rec.negotiation.rows])


def _export_frame(rec: Recommendation) -> pd.DataFrame:
    """A flat one-row summary for CSV export — the whole recommendation, auditable."""
    neg = rec.negotiation
    bt = rec.backtest
    weights = rec.weights.normalized()
    row = {
        "product": rec.product.name,
        "quantity": rec.quantity,
        "timeline_days": rec.timeline_days,
        "target_month": rec.target_month,
        "competition": rec.competition,
        "forecast_source": rec.source,
        "weight_gas": round(weights.gas, 3),
        "weight_clay": round(weights.clay, 3),
        "weight_energy": round(weights.energy, 3),
        "weight_transport": round(weights.transport, 3),
        "lock_ratio": round(rec.lock_ratio, 4),
        "blended_band_width": round(rec.band_width, 4),
        "band_regime": rec.band_regime,
        "unit_cost_eur": round(rec.unit_cost, 2),
        "supplier": rec.chosen_supplier.name if rec.chosen_supplier else "",
        "supplier_score": round(rec.chosen_supplier.total_score, 1) if rec.chosen_supplier else "",
        "channel": rec.chosen_channel.name if rec.chosen_channel else "",
        "channel_score": round(rec.chosen_channel.total_score, 1) if rec.chosen_channel else "",
        "buy_price_eur": round(neg.buy_price, 2),
        "sell_price_eur": round(neg.sell_price, 2),
        "unit_margin_eur": round(neg.unit_margin, 2),
        "total_margin_eur": round(neg.total_margin, 2),
        "backtest_agent_margin_per_month": round(bt.agent.mean, 0),
        "backtest_vs_random_pct": round(bt.agent_vs_random_pct, 1),
        "backtest_vs_cheap_pct": round(bt.agent_vs_cheap_pct, 1),
    }
    return pd.DataFrame([row])


# --------------------------------------------------------------------------- #
# Opt-in live forecast (off by default; mirrors the gas live-refresh discipline)
# --------------------------------------------------------------------------- #
def _resolve_job_id() -> str | None:
    """Return the live ceramics job id if a live refresh is active, else None (mock)."""
    if st.session_state.get("ceramics_live_on") and st.session_state.get("ceramics_live_job_id"):
        return st.session_state["ceramics_live_job_id"]
    return None


def _live_controls() -> None:
    """Compact opt-in live control. OFF (default): committed mock, instant & offline.
    ON + refresh: forecast all four factors live into a session-only job; the mock
    artifact is never overwritten, so toggling off restores the deterministic demo."""
    have_key = config.have_sybilion_key()
    with st.expander("Data source — forecast (live refresh is opt-in)"):
        st.toggle(
            "Live Sybilion 4-factor refresh", value=False, key="ceramics_live_on",
            disabled=not have_key,
            help=("OFF (default): the committed 4-factor mock — instant, offline, identical "
                  "every run. ON: submits one Sybilion job per factor (needs SYBILION_API_KEY, "
                  "~1-4 min); the mock is never overwritten."),
        )
        if not have_key:
            st.caption("Set `SYBILION_API_KEY` to enable the live 4-factor refresh.")
            return
        if st.session_state.get("ceramics_live_on") and st.button(
            "Refresh all factors now", use_container_width=True,
        ):
            try:
                with st.spinner("Sybilion is forecasting gas / clay / power / shipping… (~1-4 min)"):
                    job = cforecast.run_live_ceramics_forecast(
                        sc.SybilionClient(), cforecast.default_factor_history()
                    )
                st.session_state["ceramics_live_job_id"] = job
                st.rerun()
            except Exception as exc:  # noqa: BLE001 — surface and keep the mock
                st.error(f"Live refresh failed: {exc}. Showing the committed mock.")
        live_job = st.session_state.get("ceramics_live_job_id")
        if st.session_state.get("ceramics_live_on") and live_job:
            st.success(f"Live forecast active — job `{live_job}`. Toggle off to restore the mock.")


# --------------------------------------------------------------------------- #
# The tab
# --------------------------------------------------------------------------- #
def render_ceramics_tab(render_voiceover) -> None:
    """Render the whole ceramics optimizer tab. ``render_voiceover`` is the gas
    app's narration helper, passed in so voice is reused with no gas-side edits."""
    st.title("Ceramics production optimizer")
    st.caption(
        "For one production run, the agent decides three things — how much input cost to "
        "**lock** now, **which supplier** to buy from, and **which channel** to sell through — "
        "then negotiates the deal. Same spine as the gas tab: deterministic policy decides, the "
        "LLM only explains."
    )
    st.caption(
        "**Pipeline:** a 4-factor cost forecast (gas / clay / power / shipping)  →  a blended "
        "cost band sets the lock %  →  curation keeps credible suppliers & channels  →  a "
        "two-round negotiation sets the margin  →  a backtest checks it beats random  →  "
        "Featherless explains it. The numbers are all deterministic."
    )

    _live_controls()
    job_id = _resolve_job_id()
    months = available_months(job_id)

    # --- input panel ---------------------------------------------------------
    products = list_products()
    name_by_id = {p.id: p.name for p in products}
    with st.container(border=True):
        row1 = st.columns([2, 1, 1, 1])
        product_id = row1[0].selectbox(
            "Product", options=[p.id for p in products],
            format_func=lambda pid: name_by_id[pid], key="cer_product",
        )
        quantity = row1[1].number_input(
            "Quantity (units)", min_value=100, max_value=50000, value=5000, step=100, key="cer_qty",
        )
        timeline_days = row1[2].slider("Timeline (days)", 5, 60, 14, key="cer_timeline")
        competition = row1[3].selectbox(
            "Channel competition", ["low", "medium", "high"], index=1, key="cer_competition",
            help="How crowded the channel's shelf is — more competition pushes the sell price down.",
        )

        row2 = st.columns([1, 1, 1, 1, 1.4])
        gas_w = row2[0].slider("Weight: gas", 0.0, 1.0, 0.40, 0.05, key="cer_w_gas")
        clay_w = row2[1].slider("Weight: clay", 0.0, 1.0, 0.35, 0.05, key="cer_w_clay")
        energy_w = row2[2].slider("Weight: energy", 0.0, 1.0, 0.15, 0.05, key="cer_w_energy")
        transport_w = row2[3].slider("Weight: transport", 0.0, 1.0, 0.10, 0.05, key="cer_w_transport")
        default_target = months[0] if months else ""
        target_month = row2[4].selectbox(
            "Target month", months, index=0, key="cer_target",
            format_func=lambda m: m[:7],
        ) if months else default_target

    weights = CostWeights(gas_w, clay_w, energy_w, transport_w)
    norm = weights.normalized()
    st.caption(
        f"Normalized weights → gas {norm.gas:.0%} · clay {norm.clay:.0%} · "
        f"energy {norm.energy:.0%} · transport {norm.transport:.0%} "
        "(sliders are auto-normalized to sum to 100%)."
    )

    weights_tuple = (gas_w, clay_w, energy_w, transport_w)
    rec = _cached_recommendation(
        product_id, int(quantity), int(timeline_days), weights_tuple,
        competition, target_month, job_id,
    )

    if rec.source == "mock":
        st.info(
            "📦 **Offline demo** — running on the committed 4-factor mock forecast "
            "(gas = the real cached TTF band; clay / power / shipping = deterministic seasonal "
            "mocks). Enable the live refresh above with a Sybilion key for today's market."
        )

    # --- headline decision ---------------------------------------------------
    neg = rec.negotiation
    head = st.columns([1, 1, 1])
    head[0].metric(
        f"Lock now — next quarter input cost", f"{rec.lock_ratio:.0%}",
        help="Share of next quarter's blended input cost to fix forward now. Set by the "
             "blended cost band (tight → lock more, wide → stay flexible), not the point forecast.",
    )
    head[1].metric(
        "Unit margin (negotiated)", f"€{neg.unit_margin:.2f}",
        help="Sell price minus buy price per unit, after the two-round negotiation.",
    )
    head[2].metric(
        "Total margin", f"€{neg.total_margin:,.0f}",
        help=f"Unit margin × {rec.quantity:,} units.",
    )
    st.caption(
        f"Blended cost band **{rec.band_width:.0%}** ({rec.band_regime}) · {rec.lock_label} · "
        f"physical cost ≈ €{rec.unit_cost:.2f}/unit."
    )

    # --- cost forecast band --------------------------------------------------
    st.subheader("Per-unit cost forecast")
    st.plotly_chart(cost_band_figure(rec), use_container_width=True)
    st.caption(
        "Median physical EUR/unit with its 80% band, from the bill of materials × the four "
        "factor price bands. Green = cost outlook falling, red = rising. The *band width* (not "
        "this level) is what drives the lock %."
    )

    # --- suppliers + channels ------------------------------------------------
    st.subheader("Suppliers — kept, scored, ranked")
    chosen_supplier_name = rec.chosen_supplier.name if rec.chosen_supplier else None
    sup_cols = st.columns(min(3, len(rec.supplier_curation.kept)) or 1)
    for col, curated in zip(sup_cols, rec.supplier_curation.kept[:3]):
        _supplier_card(col, curated, chosen=curated.name == chosen_supplier_name)
    if rec.supplier_curation.rejected:
        st.caption("Rejected as off-domain: " +
                   ", ".join(f"{c.name} ({c.reason})" for c in rec.supplier_curation.rejected))

    st.subheader("Sales channels — kept, scored, ranked")
    chosen_channel_name = rec.chosen_channel.name if rec.chosen_channel else None
    chan_cols = st.columns(min(2, len(rec.channel_curation.kept)) or 1)
    for col, curated in zip(chan_cols, rec.channel_curation.kept[:2]):
        _channel_card(col, curated, chosen=curated.name == chosen_channel_name)
    if rec.channel_curation.rejected:
        st.caption("Rejected as off-domain: " +
                   ", ".join(f"{c.name} ({c.reason})" for c in rec.channel_curation.rejected))

    # --- negotiation ---------------------------------------------------------
    st.subheader("Negotiation — two rounds, fully traced")
    st.dataframe(_negotiation_table(rec), use_container_width=True, hide_index=True)

    # --- recommendation callout ----------------------------------------------
    supplier_name = rec.chosen_supplier.name if rec.chosen_supplier else "—"
    channel_name = rec.chosen_channel.name if rec.chosen_channel else "—"
    st.success(
        f"**Recommendation — {rec.quantity:,} × {rec.product.name}.** "
        f"Lock **{rec.lock_ratio:.0%}** of input cost now. Buy from **{supplier_name}** at "
        f"**€{neg.buy_price:.2f}/unit**, sell through **{channel_name}** at "
        f"**€{neg.sell_price:.2f}/unit** → unit margin **€{neg.unit_margin:.2f}** "
        f"(**€{neg.total_margin:,.0f}** total)."
    )

    # --- explanation (LLM → template) + voice --------------------------------
    st.subheader("Why — the agent's explanation")
    explanation = _cached_explanation(
        product_id, int(quantity), int(timeline_days), weights_tuple,
        competition, target_month, job_id,
    )
    st.info(explanation.text)
    if explanation.source == "llm":
        st.caption(f"Generated by Featherless {explanation.model.split('/')[-1]} from the policy's "
                   "own numbers. It explains the recommendation — it does not make it.")
    else:
        st.caption("Featherless unavailable — deterministic fallback narrative built from the same numbers.")
    render_voiceover(explanation.text, "ceramics")

    # --- backtest ------------------------------------------------------------
    st.subheader("Did the policy beat the baselines?")
    bt_chart, bt_stats = st.columns([3, 2])
    with bt_chart:
        st.plotly_chart(backtest_bars_figure(rec), use_container_width=True)
    with bt_stats:
        st.metric("vs a random pick", f"{rec.backtest.agent_vs_random_pct:+.0f}%",
                  help="Agent mean realized margin vs a seeded random supplier/channel pick.")
        st.metric("vs cheap + best-margin", f"{rec.backtest.agent_vs_cheap_pct:+.0f}%",
                  help="vs always the cheapest supplier + highest-margin channel — a strong static heuristic.")
        st.metric("Agent margin / month", f"€{rec.backtest.agent.mean:,.0f}")
    st.caption(rec.backtest.verdict)
    st.caption(
        f"Replayed over {rec.backtest.n_months} historical months. Realized margin discounts each "
        "supplier's nominal margin by its reliability (a stated assumption); the seeded random "
        "baseline reproduces run-to-run. Ceramics maximizes margin, so higher is better."
    )

    # --- CSV export ----------------------------------------------------------
    csv = _export_frame(rec).to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download recommendation (CSV)", data=csv,
        file_name=f"ceramics_recommendation_{rec.product.id}_{rec.quantity}.csv",
        mime="text/csv", use_container_width=True,
    )
