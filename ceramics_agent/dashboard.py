"""The ceramics optimizer section — inputs in, an auditable recommendation out.

Rendered as the second section of the single-page app, stacked below the gas
decision (the gas decision logic is untouched). Its inputs are seeded from the
page's one shared company description (product / quantity / timeline / cost-factor
weights / competition), so the same description drives both decisions; the user can
still tweak them. It runs the deterministic pipeline
(:func:`ceramics_agent.recommend.build_recommendation`) and lays out the result the
same way the gas section does: the decision up top, then the evidence — the cost
forecast band, the curated suppliers and channels, the negotiation trace, the
recommendation callout, the Featherless explanation (voiced), the three-strategy
backtest, and a CSV export.

Everything heavy is wrapped in ``@st.cache_data`` keyed by *all* inputs + the
forecast job, so the section is instant on repeat and identical inputs always
render identical numbers. The forecast job is resolved by the page's unified
Live⟷Cached control: ``job_id=None`` → the committed mock (offline, identical every
run); a live ceramics job when the user opts in with ``SYBILION_API_KEY``.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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
    shock_magnitude: float = 0.0,
    shock_affected: tuple[str, ...] = (),
    shock_label: str = "",
) -> Recommendation:
    weights = CostWeights(*weights_tuple)
    return build_recommendation(
        product_id, quantity, timeline_days, weights, competition,
        target_month=target_month, job_id=job_id,
        shock_magnitude=shock_magnitude, shock_affected=shock_affected, shock_label=shock_label,
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
    shock_magnitude: float = 0.0,
    shock_affected: tuple[str, ...] = (),
    shock_label: str = "",
):
    """Rebuild the (cheap, deterministic) recommendation and narrate it. Keyed by
    the same inputs — including any active shock — so the slow Featherless call runs
    once per state and the prose matches the shocked numbers. The LLM only explains,
    it never changes a number."""
    weights = CostWeights(*weights_tuple)
    rec = build_recommendation(
        product_id, quantity, timeline_days, weights, competition,
        target_month=target_month, job_id=job_id,
        shock_magnitude=shock_magnitude, shock_affected=shock_affected, shock_label=shock_label,
    )
    return explain_recommendation(rec)


@st.cache_data(show_spinner=False)
def _extended_backtest(weights_tuple: tuple[float, float, float, float], job_id: str | None):
    """The 24-month robustness replay (W12) — the calm strategy over the prior year +
    the recent year. Keyed by the cost weights + forecast job (shock-independent: this
    is a robustness check on the strategy, not the scenario), so it is computed once."""
    from ceramics_agent.backtest import run_ceramics_backtest
    from ceramics_agent.catalog import EXTENDED_HISTORICAL_SALES
    from ceramics_agent.cost_policy import decide_procurement, quarter_lock_ratio
    from ceramics_agent.forecast import load_ceramics_forecast

    factors, _ = load_ceramics_forecast(job_id)
    lock = quarter_lock_ratio(decide_procurement(factors, CostWeights(*weights_tuple)))
    return run_ceramics_backtest(factors, lock, records=EXTENDED_HISTORICAL_SALES)


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
    random grey, cheap heuristic amber, always-top-ranked blue — higher is better
    (margin, not cost). The top-ranked bar (W12) ignores the lock routing."""
    bt = rec.backtest
    names = ["Agent policy", "Random pick", "Cheap + best margin", "Always top-ranked"]
    means = [bt.agent.mean, bt.random_.mean, bt.cheap.mean, bt.top_ranked.mean]
    stds = [bt.agent.std, bt.random_.std, bt.cheap.std, bt.top_ranked.std]

    figure = go.Figure()
    figure.add_trace(go.Bar(
        x=names, y=means, marker_color=[GREEN, GREY, AMBER, BLUE],
        error_y=dict(type="data", array=stds, visible=True, color="#374151", thickness=1.5),
        text=[f"€{m:,.0f}" for m in means], textposition="outside",
        hovertemplate="%{x}<br>mean €%{y:,.0f}/month<extra></extra>",
    ))
    figure.update_layout(
        height=360, margin=dict(t=30, b=10, l=10, r=10),
        yaxis=dict(title="realized margin (EUR/month)"), showlegend=False,
    )
    return figure


def _ceramics_globe_figure(points, *, layer_label: str, color: str) -> go.Figure:
    """A drag-spinnable orthographic globe for one ceramics layer (buy or sell).

    Same engine as the gas driver globe — Plotly's built-in country geometry, no
    external basemap to fail on stage — but markers are sized by the layer's 0..100
    weight (supplier score, or demand potential) rather than driver importance."""
    figure = go.Figure()
    if points:
        top = max(p.kept_importance for p in points) or 1.0
        figure.add_trace(go.Scattergeo(
            lon=[p.lon for p in points], lat=[p.lat for p in points],
            text=[f"{p.region} · {layer_label} {p.kept_importance:.0f} · "
                  f"{', '.join(p.kept_names)}" for p in points],
            customdata=[p.region for p in points],
            mode="markers", name=layer_label, hoverinfo="text",
            marker=dict(
                size=[14 + 34 * (p.kept_importance / top) for p in points],
                color=color, opacity=0.85, line=dict(width=1, color="white")),
        ))
    figure.update_geos(
        projection_type="orthographic", showland=True, landcolor="#1f2937",
        showocean=True, oceancolor="#0b1220", showcountries=True, countrycolor="#374151",
        showcoastlines=False, bgcolor="rgba(0,0,0,0)",
        projection_rotation=dict(lon=15, lat=35),
    )
    figure.update_layout(
        height=520, margin=dict(t=0, b=0, l=0, r=0), showlegend=True,
        legend=dict(orientation="h", y=0), paper_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def _clicked_region(event) -> str | None:
    """Region of a clicked globe marker, tolerating the shapes Streamlit returns
    across versions (mirrors the gas globe's click parser)."""
    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        selection = event.get("selection")
    points = (selection or {}).get("points") if isinstance(selection, dict) else None
    if points:
        custom = points[0].get("customdata")
        if isinstance(custom, (list, tuple)):
            return custom[0] if custom else None
        return custom
    return None


def _render_ceramics_globe(rec: Recommendation) -> None:
    """The two-layer globe: toggle between *where to sell* (demand markets, sized by
    demand potential) and *where to buy* (supplier sourcing, sized by curation score).
    Reuses the gas geo engine's :class:`CountryAggregate` shape; a click (or the picker)
    opens an explanation-only Featherless brief on the country."""
    from ceramics_agent import geo

    st.subheader("Where to sell, and where to buy — the agent's map")
    layer = st.radio(
        "Globe layer", ["Where to sell (demand)", "Where to buy (sourcing)"],
        horizontal=True, key="cer_globe_layer", label_visibility="collapsed",
    )
    selling = layer.startswith("Where to sell")
    if selling:
        points = geo.demand_market_points(rec.target_month)
        color, weight_label, brief_layer = GREEN, "demand", "sell"
        st.caption("Bigger markers = more demand potential (historical channel mix × the "
                   "target quarter's seasonality × margin). Drag to spin; click a market — or "
                   "pick one below — for a brief on why it sells.")
    else:
        chosen = rec.chosen_supplier.name if rec.chosen_supplier else None
        points = geo.supplier_sourcing_points(rec.supplier_curation, chosen_name=chosen)
        color, weight_label, brief_layer = BLUE, "score", "buy"
        st.caption("Bigger markers = higher supplier curation score; the chosen supplier is "
                   "flagged ✓. Drag to spin; click a country — or pick one below — for a brief "
                   "on why it's a credible source.")

    globe_col, detail_col = st.columns([3, 2])
    with globe_col:
        event = st.plotly_chart(
            _ceramics_globe_figure(points, layer_label=weight_label, color=color),
            use_container_width=True, on_select="rerun", key=f"cer_globe_{brief_layer}",
        )
        clicked = _clicked_region(event)
    with detail_col:
        names = [p.region for p in points]
        pick_key = f"cer_country_pick_{brief_layer}"
        if st.session_state.get(pick_key) not in names:
            st.session_state.pop(pick_key, None)
        if clicked in names:
            st.session_state[pick_key] = clicked
        region = st.selectbox("Inspect a market" if selling else "Inspect a source",
                              names, key=pick_key) if names else None
        if region:
            agg = next(p for p in points if p.region == region)
            st.markdown(f"**{region}** — {weight_label} {agg.kept_importance:.0f}")
            brief, source = _cached_ceramics_brief(region, brief_layer, tuple(agg.kept_names))
            st.info(brief)
            st.caption("Featherless — explanation only, no decision."
                       if source == "llm" else "Offline brief — Featherless unavailable.")
            st.caption(("Channels: " if selling else "Suppliers: ") + ", ".join(agg.kept_names))


@st.cache_data(show_spinner=False)
def _cached_ceramics_brief(region: str, layer: str, names: tuple[str, ...]):
    """Cache the (LLM or fallback) country brief so re-picking a country is instant."""
    from ceramics_agent import geo

    return geo.ceramics_country_brief(region, layer, list(names))


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
# The ceramics section (rendered below the gas section on the single page)
# --------------------------------------------------------------------------- #
def render_ceramics_tab(render_voiceover, profile=None, job_id=None) -> None:
    """Render the ceramics optimizer section.

    ``render_voiceover`` is the app's narration helper, passed in so voice is reused.
    ``profile`` is the shared :class:`ceramics_agent.intake.CompanyProfile` from the
    page's one company description — when present it seeds the inputs (product /
    quantity / timeline / competition / cost weights), so the same description drives
    both decisions. ``job_id`` is the resolved forecast job (a live ceramics job when
    the unified Live toggle is on, else ``None`` → the committed mock)."""
    st.header("② How should we run the ceramics line?")
    st.caption(
        "For one production run, the agent decides three things — how much input cost to "
        "**lock** now, **which supplier** to buy from, and **which channel** to sell through — "
        "then negotiates the deal. Same spine as the gas decision: deterministic policy decides, "
        "the LLM only explains."
    )
    st.caption(
        "**Pipeline:** a 4-factor cost forecast (gas / clay / power / shipping)  →  a blended "
        "cost band sets the lock %  →  curation keeps credible suppliers & channels  →  a "
        "two-round negotiation sets the margin  →  a backtest checks it beats random  →  "
        "Featherless explains it. The numbers are all deterministic."
    )

    months = available_months(job_id)

    # --- input panel ---------------------------------------------------------
    # Defaults come from the shared company profile when present (so the one
    # description drives these inputs), else the committed demo defaults. Widget keys
    # carry the profile version so a fresh description resets the inputs to the new
    # profile; within a version the user can still tweak and the tweaks stick.
    products = list_products()
    name_by_id = {p.id: p.name for p in products}
    product_ids = [p.id for p in products]
    if profile is not None:
        norm = profile.weights.normalized()
        d_product = profile.product_id if profile.product_id in product_ids else product_ids[0]
        d_qty = min(50000, max(100, int(profile.quantity)))
        d_timeline = min(60, max(5, int(profile.timeline_days)))
        d_competition = profile.competition if profile.competition in ("low", "medium", "high") else "medium"
        d_w = (round(norm.gas, 2), round(norm.clay, 2), round(norm.energy, 2), round(norm.transport, 2))
    else:
        d_product, d_qty, d_timeline, d_competition = "bowl", 5000, 14, "medium"
        d_w = (0.40, 0.35, 0.15, 0.10)
    version = st.session_state.get("profile_version", 0)
    levels = ["low", "medium", "high"]

    with st.container(border=True):
        row1 = st.columns([2, 1, 1, 1])
        product_id = row1[0].selectbox(
            "Product", options=product_ids, index=product_ids.index(d_product),
            format_func=lambda pid: name_by_id[pid], key=f"cer_product_{version}",
        )
        quantity = row1[1].number_input(
            "Quantity (units)", min_value=100, max_value=50000, value=d_qty, step=100,
            key=f"cer_qty_{version}",
        )
        timeline_days = row1[2].slider("Timeline (days)", 5, 60, d_timeline, key=f"cer_timeline_{version}")
        competition = row1[3].selectbox(
            "Channel competition", levels, index=levels.index(d_competition),
            key=f"cer_competition_{version}",
            help="How crowded the channel's shelf is — more competition pushes the sell price down.",
        )

        row2 = st.columns([1, 1, 1, 1, 1.4])
        gas_w = row2[0].slider("Weight: gas", 0.0, 1.0, d_w[0], 0.05, key=f"cer_w_gas_{version}")
        clay_w = row2[1].slider("Weight: clay", 0.0, 1.0, d_w[1], 0.05, key=f"cer_w_clay_{version}")
        energy_w = row2[2].slider("Weight: energy", 0.0, 1.0, d_w[2], 0.05, key=f"cer_w_energy_{version}")
        transport_w = row2[3].slider("Weight: transport", 0.0, 1.0, d_w[3], 0.05, key=f"cer_w_transport_{version}")
        default_target = months[0] if months else ""
        target_month = row2[4].selectbox(
            "Target month", months, index=0, key=f"cer_target_{version}",
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

    # A chat-driven supply shock (set in the bottom chat panel) re-decides the ceramics
    # line too — the same headline that moves the gas hedge. The factor routing decides
    # which cost band(s) move; defaults (no shock) leave the calm decision byte-identical.
    shock_magnitude = float(st.session_state.get("shock_magnitude", 0.0))
    shock_label = st.session_state.get("shock_label", "")
    shock_affected = tuple(st.session_state.get("shock_affected", ()))

    rec = _cached_recommendation(
        product_id, int(quantity), int(timeline_days), weights_tuple,
        competition, target_month, job_id,
        shock_magnitude, shock_affected, shock_label,
    )

    # Stash the (calm) ceramics decision so the shared bottom chat can ground a spoken
    # "what about ceramics?" answer and compute the lock move for a shock reply. Read-only
    # — the chat never alters these numbers. Cleared each run by main(); set when shown.
    st.session_state["_cer_chat"] = {
        "lock_ratio": rec.calm_lock_ratio,
        "band_regime": rec.band_regime,
        "supplier": rec.chosen_supplier.name if rec.chosen_supplier else "",
        "channel": rec.chosen_channel.name if rec.chosen_channel else "",
        "unit_margin": rec.negotiation.unit_margin,
        "scenario_lock": rec.lock_ratio if rec.shock_active else None,
        "weights_tuple": weights_tuple,
        "job_id": job_id,
    }

    if rec.source == "mock":
        st.info(
            "📦 **Offline demo** — running on the committed 4-factor mock forecast "
            "(gas = the real cached TTF band; clay / power / shipping = deterministic seasonal "
            "mocks). Enable the live refresh above with a Sybilion key for today's market."
        )

    if rec.shock_active:
        st.error(
            f"⚡ **Scenario active — {rec.shock_label}** (severity {rec.shock_magnitude:.0%}). "
            f"The shock hits **{', '.join(rec.shock_affected)}**; the input-cost lock rises "
            f"**{rec.calm_lock_ratio:.0%} → {rec.lock_ratio:.0%}** "
            f"(supply-risk premium +{rec.scenario_premium:.0%} on the floor). Same deterministic "
            "cost policy, re-run on the shocked factor band(s) — reset it in the chat below."
        )

    # --- headline decision ---------------------------------------------------
    neg = rec.negotiation
    head = st.columns([1, 1, 1])
    head[0].metric(
        f"Lock now — next quarter input cost", f"{rec.lock_ratio:.0%}",
        delta=f"{rec.lock_delta:+.0%} vs calm" if rec.shock_active else None,
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

    # --- globe: where to sell / where to buy (W5) ----------------------------
    _render_ceramics_globe(rec)

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
        shock_magnitude, shock_affected, shock_label,
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
        st.metric("vs always top-ranked", f"{rec.backtest.agent_vs_top_ranked_pct:+.0f}%",
                  help="vs always committing to the best-scored supplier, ignoring the lock routing.")
    st.caption(rec.backtest.verdict)
    st.caption(rec.backtest.extended_verdict)
    longer = _extended_backtest(weights_tuple, job_id)
    st.caption(
        f"🔁 **Robustness — {longer.n_months}-month replay** (prior year + the recent year): the agent "
        f"books €{longer.agent.mean:,.0f}/month and holds its edge — {longer.agent_vs_random_pct:+.0f}% vs "
        f"random, {longer.agent_vs_cheap_pct:+.0f}% vs cheap, {longer.agent_vs_top_ranked_pct:+.0f}% vs "
        "top-ranked — so the result is not an artefact of one 12-month window."
    )
    st.caption(
        f"Headline replay over {rec.backtest.n_months} historical months. Realized margin discounts each "
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
