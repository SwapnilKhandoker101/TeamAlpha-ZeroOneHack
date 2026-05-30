"""Gas-hedging decision agent — dashboard.

Reads the cached real Sybilion forecast and turns it into a hedging decision the
viewer can interrogate, end to end: the Featherless tag-picker that configured
the forecast, the probabilistic price band, the curated drivers (kept vs the
spurious ones thrown out), the per-month hedge ratio the deterministic policy
derives, and a Featherless narrative explaining the decision it did not make.
"""

from __future__ import annotations

import hashlib

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gas_agent import config
from gas_agent import geo
from gas_agent import sybilion_client as sc
from gas_agent import transcribe
from gas_agent import voice
from gas_agent import voice_chat
from gas_agent.config import EXPLANATION_MODEL, KEYWORD_MODEL
from gas_agent.decision_backtest import backtest_from_cache
from gas_agent.driver_curation import curate_drivers
from gas_agent.explanation_agent import explain_decision
from gas_agent.hedge_policy import DEFAULT_PARAMS, decide_all, quarter_hedge_ratio
from gas_agent.keyword_agent import DEFAULT_PERSONA, select_filters
from gas_agent.scenario import (
    DEFAULT_SHOCK_PARAMS,
    parse_shock_request,
    risk_importance_share,
    run_shock,
    shock_forecast_json,
    standing_risk_premium,
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
    curation = curate_drivers(signals)
    premium = standing_risk_premium(curation)  # the calm-path standing premium
    quarter = decide_all(
        sc.parse_forecast_months(forecast_json), spot, DEFAULT_PARAMS, risk_premium=premium
    )[:3]
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
    curation = curate_drivers(signals)
    outcome = run_shock(months, spot, curation, magnitude, label,
                        base_risk_premium=standing_risk_premium(curation))
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


@st.cache_data(show_spinner="Synthesising narration…")
def cached_voice(text: str):
    """Voice the explanation once per unique text. Returns a plain dict (so it
    caches cleanly) or None when no TTS provider is reachable. The NVIDIA free-tier
    limiter inside voice.synthesize routes to Featherless when needed."""
    clip = voice.synthesize(text)
    if clip is None:
        return None
    return {"audio": clip.audio_bytes, "mime": clip.mime,
            "provider": clip.provider, "voice": clip.voice}


def _disk_voiceover(name: str):
    """Fall back to a pre-synthesised clip from cache/audio/ (written by
    scripts/build_voiceover.py) so the demo has audio even if live TTS is down."""
    from gas_agent import config as _config
    sidecar = _config.CACHE_DIR / "audio" / f"{name}.json"
    if not sidecar.exists():
        return None
    try:
        import json
        meta = json.loads(sidecar.read_text())
        audio_path = _config.CACHE_DIR / "audio" / meta["file"]
        return {"audio": audio_path.read_bytes(), "mime": meta["mime"],
                "provider": meta.get("provider", "cache"), "voice": meta.get("voice", "")}
    except Exception:
        return None


def render_voiceover(text: str, cache_name: str, autoplay: bool = False) -> None:
    """Play a narration of ``text`` under the explanation: live synthesis first
    (so the audio matches the words and the caption names the real provider),
    then a pre-cached disk clip, then nothing. Voicing only — never deciding."""
    data = cached_voice(text) or _disk_voiceover(cache_name)
    if not data:
        return
    st.audio(data["audio"], format=data["mime"], autoplay=autoplay)
    voice_suffix = f" · {data['voice']}" if data.get("voice") else ""
    st.caption(f"🔊 Narrated by {data['provider']}{voice_suffix} — voicing the explanation "
               "above, not deciding it.")


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


def globe_figure(countries) -> go.Figure:
    """A drag-spinnable orthographic globe: green markers for the credible
    supplier/hub countries (bigger = more kept importance) and red markers for
    the spurious-only countries. Plotly's built-in country geometry draws the
    land, so there is no external basemap fetch to fail on stage."""
    kept = [c for c in countries if c.has_kept]
    rejected = [c for c in countries if c.rejected_only]
    figure = go.Figure()
    if kept:
        max_importance = max(c.kept_importance for c in kept)
        figure.add_trace(go.Scattergeo(
            lon=[c.lon for c in kept], lat=[c.lat for c in kept],
            text=[f"{c.region} · {c.kept_count} kept · importance {c.kept_importance:.0f}"
                  for c in kept],
            customdata=[c.region for c in kept],
            mode="markers", name="kept (credible)", hoverinfo="text",
            marker=dict(
                size=[14 + 34 * (c.kept_importance / max_importance) for c in kept],
                color=GREEN, opacity=0.85, line=dict(width=1, color="white")),
        ))
    if rejected:
        figure.add_trace(go.Scattergeo(
            lon=[c.lon for c in rejected], lat=[c.lat for c in rejected],
            text=[f"{c.region} · {c.rejected_count} spurious dropped" for c in rejected],
            customdata=[c.region for c in rejected],
            mode="markers", name="dropped (spurious)", hoverinfo="text",
            marker=dict(size=12, color=RED, opacity=0.9, line=dict(width=1, color="white")),
        ))
    figure.update_geos(
        projection_type="orthographic", showland=True, landcolor="#1f2937",
        showocean=True, oceancolor="#0b1220", showcountries=True, countrycolor="#374151",
        showcoastlines=False, bgcolor="rgba(0,0,0,0)",
        projection_rotation=dict(lon=20, lat=30),
    )
    figure.update_layout(
        height=520, margin=dict(t=0, b=0, l=0, r=0), showlegend=True,
        legend=dict(orientation="h", y=0), paper_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def _clicked_region(event) -> str | None:
    """Pull the region of a clicked globe marker out of the Plotly selection event,
    tolerating the slightly different shapes Streamlit returns across versions."""
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


SHOCK_BUTTON_MESSAGE = "Iran closes the Strait of Hormuz"


def _set_shock(magnitude: float, label: str) -> None:
    st.session_state.shock_magnitude = magnitude
    st.session_state.shock_label = label


def _shock_reply(months, spot, curation, magnitude: float, label: str) -> str:
    """The assistant's chat reply: what the deterministic policy did with the shock.
    The number here is computed by the policy, not written by an LLM."""
    auto = standing_risk_premium(curation)  # the calm-path standing premium already in force
    baseline = quarter_hedge_ratio(decide_all(months, spot, DEFAULT_PARAMS, risk_premium=auto)[:3])
    outcome = run_shock(months, spot, curation, magnitude, label, base_risk_premium=auto)
    shocked = quarter_hedge_ratio(outcome.decisions[:3])
    applied = outcome.decisions[0].risk_premium if outcome.decisions else outcome.risk_premium
    arrow = "up" if shocked > baseline else ("down" if shocked < baseline else "unchanged")
    return (
        f"**{label}** read as severity {magnitude:.0%}. I re-ran the deterministic policy: "
        f"forward median lifts, the band widens, and the supply-risk premium rises to "
        f"+{applied:.0%} on the lock floor.\n\n"
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


def _chat_state(months, spot, curation) -> voice_chat.ChatState:
    """Build the grounded Q&A context from the CURRENT scenario state, mirroring the
    calm/shock resolution in :func:`main` so a spoken answer matches the dashboard.
    Read-only: it computes decisions for grounding, it never sets the ratio."""
    auto = standing_risk_premium(curation)
    magnitude = st.session_state.get("shock_magnitude", 0.0)
    label = st.session_state.get("shock_label", "")
    if magnitude > 0:
        outcome = run_shock(months, spot, curation, magnitude, label, base_risk_premium=auto)
        decisions = outcome.decisions
        used = outcome.curation
        applied_premium = decisions[0].risk_premium if decisions else outcome.risk_premium
    else:
        decisions = decide_all(months, spot, DEFAULT_PARAMS, risk_premium=auto)
        used = curation
        applied_premium = auto
    return voice_chat.ChatState(
        spot_price=spot,
        decisions=decisions,
        quarter_ratio=quarter_hedge_ratio(decisions[:3]),
        kept_drivers=used.kept,
        rejected_drivers=used.rejected,
        standing_premium=applied_premium,
        scenario_label=label,
        scenario_magnitude=magnitude,
    )


def _route_chat_message(prompt: str, months, spot, curation, use_llm: bool,
                        freeform: bool) -> tuple[str, str]:
    """Route a chat message (typed OR transcribed) through the SAME branches and
    return ``(reply_markdown, spoken_text)``. ``spoken_text`` is empty when there's
    nothing worth reading aloud. The LLM only reads severity / explains — the
    deterministic policy owns the ratio."""
    request = parse_shock_request(prompt, use_llm=use_llm)
    if request.is_shock:
        _set_shock(request.magnitude, request.label)
        reply = _shock_reply(months, spot, curation, request.magnitude, request.label)
        return reply, ""  # the main body already narrates the shocked explanation
    if freeform:
        return _freeform_reply(prompt), "I re-selected the Sybilion filters from your request."
    # Grounded question — answer from the current state without touching the shock.
    answer = voice_chat.answer_question(prompt, _chat_state(months, spot, curation))
    return answer.text, answer.text


def _handle_voice_prompt(text: str, months, spot, curation, use_llm: bool,
                         freeform: bool) -> None:
    """Transcribed question → same routing as a typed message, then speak the reply
    after the rerun (so the new transcript shows in the chat log first)."""
    st.session_state.messages.append({"role": "user", "content": f"🎤 {text}"})
    reply, spoken = _route_chat_message(text, months, spot, curation, use_llm, freeform)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.pending_voice = spoken
    st.rerun()


def live_refresh_controls() -> None:
    """Opt-in live Sybilion refresh — off by default.

    OFF: the dashboard reads the pinned cached job (instant, offline, the same
    numbers every run — the demo guarantee). ON + *Refresh now*: re-forecasts
    against today's market into a **session-only** job; ``latest_job.txt`` stays
    pinned, so toggling off restores the deterministic demo with no re-fetch.
    """
    have_key = config.have_sybilion_key()
    st.toggle(
        "Live Sybilion refresh",
        value=False,
        key="live_sybilion",
        disabled=not have_key,
        help=("OFF (default): the dashboard reads the cached forecast — instant, "
              "offline, and the same numbers every run. ON: calls Sybilion live so "
              "the forecast reflects today's market (needs SYBILION_API_KEY, takes "
              "~10-60s, and the numbers will vary run to run)."),
    )
    if not have_key:
        st.caption("Set `SYBILION_API_KEY` to enable live refresh.")
        return

    if st.session_state.get("live_sybilion"):
        if st.button("Refresh now", use_container_width=True,
                     help="Submit a fresh forecast to Sybilion and re-fetch its artifacts."):
            try:
                with st.spinner("Sybilion is forecasting against today's market… (~10-60s)"):
                    selection = select_filters(DEFAULT_PERSONA)
                    ttf = sc.load_ttf_series()
                    payload = sc.build_forecast_payload(
                        ttf["timeseries"],
                        title=ttf.get("meta", {}).get("title") or sc.DEFAULT_SERIES_TITLE,
                        keywords=selection.keywords,
                        category_ids=selection.category_ids,
                        region_codes=selection.region_codes,
                    )
                    new_job = sc.run_live_forecast(sc.SybilionClient(), payload)
                st.session_state.live_job_id = new_job
                st.session_state.live_job_fetched = pd.Timestamp.now().strftime("%H:%M:%S")
                st.rerun()
            except Exception as exc:  # noqa: BLE001 — surface and keep the cached job
                st.error(f"Live refresh failed: {exc}. Showing the cached forecast.")

        live_job = st.session_state.get("live_job_id")
        if live_job:
            fetched = st.session_state.get("live_job_fetched", "")
            st.success(
                f"Live forecast active — job `{live_job[:8]}…`"
                + (f", fetched {fetched}" if fetched else "")
                + ". Cached demo restored when you toggle off."
            )


def scenario_sidebar(months, spot, curation) -> None:
    """Live scenario controls + chat. Mutates ``st.session_state`` (shock magnitude
    + label + message log); the main body reads that state and re-renders."""
    with st.sidebar:
        st.header("Live scenario")
        st.caption("Type (or speak) a supply-shock headline and the agent re-decides on the "
                   "spot — the LLM only reads the severity, the deterministic policy moves the "
                   "ratio. Or just ask *why* — it explains the decision, never re-makes it.")

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
                                  "re-pick the Sybilion filters (no live forecast call — see "
                                  "Data source for that). Off (default): guided supply-shock only.")

        st.divider()
        st.subheader("Data source")
        live_refresh_controls()

        st.divider()
        for message in st.session_state.messages:
            st.chat_message(message["role"]).write(message["content"])

        # Speak the most recent voice answer (queued on the prior run so the
        # transcript renders first). Popped after playing so it never loops.
        pending = st.session_state.pop("pending_voice", "")
        if pending:
            render_voiceover(pending, "chat", autoplay=True)

        # Push-to-talk: record a question, transcribe it, then route it through the
        # SAME branches a typed message hits. Hidden when no ASR backend is wired up,
        # so the text chat and the no-keys demo are completely unchanged.
        if transcribe.available():
            audio = st.audio_input(
                "🎤 Ask by voice",
                key="voice_clip",
                help="Record a question — 'why this hedge ratio?', 'which supplier "
                     "matters most?'. It is transcribed, answered, and spoken back. "
                     "Explanation only — it never changes the ratio.",
            )
            if audio is not None:
                clip = audio.getvalue()
                signature = hashlib.md5(clip).hexdigest() if clip else ""
                if signature and signature != st.session_state.get("last_voice_sig"):
                    st.session_state.last_voice_sig = signature
                    with st.spinner("Transcribing…"):
                        heard = transcribe.transcribe(clip)
                    if heard:
                        _handle_voice_prompt(heard, months, spot, curation, use_llm, freeform)
                    else:
                        st.warning("Couldn't transcribe that clip — try again or type your question.")
        else:
            st.caption("🎤 Voice input: add `HF_API_KEY` or `NVIDIA_ASR_FUNCTION_ID` to ask by voice.")

        if prompt := st.chat_input("Ask 'why this ratio?' — or type a shock like 'Iran closes Hormuz'"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            reply, _ = _route_chat_message(prompt, months, spot, curation, use_llm, freeform)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.rerun()


def render_gas_tab() -> None:
    st.title("European gas (TTF) hedging agent")
    st.caption("How much of next quarter's gas should an energy-intensive EU buyer "
               "lock in forward now, versus leave to spot? A decision built on the "
               "Sybilion forecast's confidence band — not its point estimate.")

    # Job resolution: the pinned demo job by default. A live Sybilion refresh
    # (sidebar toggle, off by default) swaps in a freshly-fetched, session-only
    # job — latest_job.txt stays pinned, so toggling off instantly restores the
    # deterministic demo numbers with no re-fetch.
    live_on = st.session_state.get("live_sybilion", False)
    live_job = st.session_state.get("live_job_id")
    job_id = live_job if (live_on and live_job) else sc.get_latest_job()
    if not job_id:
        st.error("No cached forecast found. Run a forecast first.")
        return

    ttf, forecast_json, metrics, signals = load_inputs(job_id)
    months = sc.parse_forecast_months(forecast_json)
    spot = sc.last_actual_price(ttf)
    base_curation = curate_drivers(signals)
    # Standing supply-risk premium — deterministic "dynamic weighting" from the live
    # driver mix, fed through the SAME hedge_policy.risk_premium the shock uses (THE
    # RULE holds: it is arithmetic on Sybilion importances, not an LLM). It lifts the
    # calm lock when supply-risk-linked drivers dominate, and moves on its own as the
    # mix shifts (e.g. after a live refresh).
    auto_premium = standing_risk_premium(base_curation)
    risk_share = risk_importance_share(base_curation)
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
        shock = run_shock(months, spot, base_curation, magnitude, label,
                          base_risk_premium=auto_premium)
        decisions = shock.decisions
        curation = shock.curation
        display_forecast_json = shock_forecast_json(forecast_json, magnitude)
    else:
        decisions = decide_all(months, spot, DEFAULT_PARAMS, risk_premium=auto_premium)
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

    calm_ratio = quarter_hedge_ratio(
        decide_all(months, spot, DEFAULT_PARAMS, risk_premium=auto_premium)[:3])
    if shock_active:
        applied_premium = shock.decisions[0].risk_premium if shock.decisions else shock.risk_premium
        standing_clause = (f" (standing +{auto_premium:.0%} plus the shock's +{shock.risk_premium:.0%})"
                           if auto_premium > 0 else "")
        st.error(f"⚡ **Scenario active — {st.session_state.shock_label}** (severity {magnitude:.0%}). "
                 f"Forward lifted, band widened, supply-risk premium **+{applied_premium:.0%}** on the "
                 f"lock floor{standing_clause}. Every panel below is the *same deterministic policy* "
                 "re-run on the shocked inputs — use the sidebar to reset.")

    top = st.columns([1, 1, 1])
    top[0].metric(f"Lock now — next quarter ({quarter_label})", f"{quarter_ratio:.0%}",
                  delta=f"{quarter_ratio - calm_ratio:+.0%} vs calm" if shock_active else None)
    top[1].metric("vs naive baselines", "0% spot / 50% lock")
    top[2].metric("Forecast point-accuracy (backtest MAPE)", f"{mape:.0f}%",
                  help="The point forecast is weak, which is exactly why the decision "
                       "is built on the confidence band and drivers instead.")

    if auto_premium > 0:
        st.caption(
            f"📌 **Standing supply-risk premium +{auto_premium:.0%}** baked into the lock floor — "
            f"supply-risk-linked drivers (Russia / Iran / Qatar / Algeria pipelines & LNG, plus any "
            f"global-risk & volatility signals) make up **{risk_share:.0%}** of kept-driver importance. "
            "Deterministic weighting from the live driver mix — not the point forecast — so it moves on "
            "its own as the mix shifts; every month's row below carries it in the trace.")

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
    render_voiceover(explanation.text, "shock" if shock_active else "golden")

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
    st.caption("Green markers are the suppliers and hubs the agent trusts (bigger = more "
               "kept importance); red markers are countries that only surfaced through "
               "spurious correlations. Drag to spin the globe and click a country — or pick "
               "one below — for a Featherless brief on why it does (or doesn't) move European gas." +
               (" Under the shock, the risk supplier lights up." if shock_active else ""))
    countries = geo.aggregate_drivers(curation)
    globe_col, detail_col = st.columns([3, 2])
    with globe_col:
        event = st.plotly_chart(globe_figure(countries), use_container_width=True,
                                on_select="rerun", key="globe")
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


def main() -> None:
    """Two decision agents, one app. The gas tab is the original dashboard,
    unchanged; the ceramics tab is the second agent. The sidebar (gas shock
    scenario + voice) stays global — it is rendered inside the gas tab body."""
    tab_gas, tab_cer = st.tabs(["Gas hedging", "Ceramics optimizer"])
    with tab_gas:
        render_gas_tab()
    with tab_cer:
        from ceramics_agent.dashboard import render_ceramics_tab

        render_ceramics_tab(render_voiceover)


if __name__ == "__main__":
    main()
