"""The explanation layer — Featherless narrates the recommendation it did not make.

Reuses the gas agent's :class:`gas_agent.explanation_agent.Explanation` and its
exact Featherless→template discipline. Every number in the recommendation — the
lock %, the supplier/channel scores, the negotiated buy/sell/margin, the backtest
deltas — is already fixed by deterministic code (THE RULE). This layer only phrases
*why*, for a ceramics ops lead, citing the real numbers it is handed.

The system prompt forbids the model from changing the lock %, the chosen supplier
or channel, or any figure. If Featherless is unavailable, a deterministic template
built from the same numbers is returned, so the dashboard always has prose and the
demo never depends on a live call.
"""

from __future__ import annotations

from gas_agent import llm
from gas_agent.config import EXPLANATION_MODEL
from gas_agent.explanation_agent import Explanation

from ceramics_agent.recommend import Recommendation

# The persona the narrative is written for — the same German manufacturer the gas
# agent serves, viewed from its ceramics production line.
CERAMICS_PERSONA = (
    "A mid-size German ceramics manufacturer (tableware, tiles) that buys raw "
    "materials and kiln energy forward and sells through several channels. Gas-fired "
    "kilns make energy its largest volatile cost; it wants to lock input cost when "
    "the outlook is tight and stay flexible when it is uncertain, then place each "
    "production run with the supplier and channel that maximise a reliable margin."
)

_SYSTEM_PROMPT = (
    "You are the explanation layer of a ceramics supply-chain optimizer. Deterministic "
    "code has ALREADY decided everything: the cost-lock %, the supplier, the channel, the "
    "negotiated prices and margin, and the backtest result. Your only job is to explain, "
    "for a ceramics operations lead, WHY these choices follow from the numbers.\n\n"
    "Strict rules:\n"
    "- Use ONLY the names and numbers in the brief. Never invent suppliers, channels, or figures.\n"
    "- Every number is final. Do NOT propose a different lock %, supplier, channel, price, or margin.\n"
    "- The lock % comes from the blended cost band: a tighter band justifies locking more of the "
    "input cost, a wider band justifies staying flexible. State this correctly.\n"
    "- Name the chosen supplier and channel and the one-line reason each was kept/scored, and the "
    "negotiated buy price, sell price and unit margin.\n"
    "- Note that the policy beats the random baseline in the backtest.\n"
    "- 120-200 words, plain and direct. No headings, no bullet lists, no preamble."
)


def _supplier_line(rec: Recommendation) -> str:
    chosen = rec.chosen_supplier
    if chosen is None:
        return "  - (no credible supplier survived curation)"
    return (
        f"  - {chosen.name}: total score {chosen.total_score:.0f} "
        f"(cost {chosen.cost_score:.0f}, reliability {chosen.reliability_score:.0f}, "
        f"lead-time fit {chosen.lead_time_score:.0f}); {chosen.reason}"
    )


def _channel_line(rec: Recommendation) -> str:
    chosen = rec.chosen_channel
    if chosen is None:
        return "  - (no credible channel survived curation)"
    return (
        f"  - {chosen.name}: total score {chosen.total_score:.0f} "
        f"(margin {chosen.margin_score:.0f}, season {chosen.season_score:.0f}, "
        f"order-fit {chosen.order_score:.0f}); {chosen.reason}"
    )


def _build_factsheet(rec: Recommendation) -> str:
    neg = rec.negotiation
    bt = rec.backtest
    rejected_suppliers = ", ".join(c.name for c in rec.supplier_curation.rejected) or "none"
    rejected_channels = ", ".join(c.name for c in rec.channel_curation.rejected) or "none"
    return (
        f"Request: {rec.quantity} × {rec.product.name}, delivery timeline "
        f"{rec.timeline_days} days, target month {rec.target_month}, competition "
        f"'{rec.competition}'.\n"
        f"Cost-factor weights: {rec.weights.normalized().as_dict()}.\n\n"
        f"Decision (deterministic, FINAL): lock {rec.lock_ratio:.0%} of next quarter's input "
        f"cost now. Blended cost band is {rec.band_regime} ({rec.band_width:.0%} of level); "
        f"{rec.lock_label}.\n"
        f"Physical cost basis: about EUR {rec.unit_cost:.2f}/unit (median, nearest month).\n\n"
        f"Chosen supplier:\n{_supplier_line(rec)}\n"
        f"  (rejected as off-domain: {rejected_suppliers})\n"
        f"Chosen channel:\n{_channel_line(rec)}\n"
        f"  (rejected as off-domain: {rejected_channels})\n\n"
        f"Negotiated deal (two rounds, FINAL): buy EUR {neg.buy_price:.2f}/unit, "
        f"sell EUR {neg.sell_price:.2f}/unit, unit margin EUR {neg.unit_margin:.2f}, "
        f"total margin EUR {neg.total_margin:,.0f} over {rec.quantity} units.\n\n"
        f"Backtest over {bt.n_months} months: agent realizes EUR {bt.agent.mean:,.0f} margin/month, "
        f"{bt.agent_vs_random_pct:+.0f}% vs a random pick and {bt.agent_vs_cheap_pct:+.0f}% vs a "
        f"cheapest-supplier + highest-margin-channel heuristic."
    )


def _fallback_text(rec: Recommendation) -> str:
    neg = rec.negotiation
    bt = rec.backtest
    supplier_name = rec.chosen_supplier.name if rec.chosen_supplier else "no eligible supplier"
    channel_name = rec.chosen_channel.name if rec.chosen_channel else "no eligible channel"
    band_clause = {
        "tight": "the blended cost band is tight, so the agent locks a high share of input cost now",
        "wide": "the blended cost band is wide, so the agent stays flexible and locks less",
    }.get(rec.band_regime, "the blended cost band is moderate, so the agent takes a balanced lock")
    return (
        f"For {rec.quantity} × {rec.product.name} on a {rec.timeline_days}-day timeline, "
        f"the agent locks {rec.lock_ratio:.0%} of next quarter's input cost: {band_clause} "
        f"({rec.band_width:.0%} band, {rec.band_regime}). At roughly EUR {rec.unit_cost:.2f}/unit "
        f"of physical cost, it buys from {supplier_name} — chosen for the best balance of cost, "
        f"reliability and lead-time fit, not price alone — and sells through {channel_name}. "
        f"After a two-round negotiation the deal settles at EUR {neg.buy_price:.2f}/unit buy and "
        f"EUR {neg.sell_price:.2f}/unit sell, a EUR {neg.unit_margin:.2f}/unit margin "
        f"(EUR {neg.total_margin:,.0f} total). Replayed over {bt.n_months} historical months this "
        f"policy books EUR {bt.agent.mean:,.0f} margin/month — {bt.agent_vs_random_pct:+.0f}% versus "
        f"a random supplier/channel pick. Every figure is computed deterministically; this text only "
        f"explains it."
    )


def explain_recommendation(rec: Recommendation, persona: str = CERAMICS_PERSONA) -> Explanation:
    """Produce a faithful narrative for the (already decided) recommendation."""
    factsheet = _build_factsheet(rec)

    if not llm.featherless_available():
        return Explanation(text=_fallback_text(rec), source="fallback")

    user = f"BUYER:\n{persona}\n\nRECOMMENDATION BRIEF:\n{factsheet}"
    try:
        text = llm.chat_text(EXPLANATION_MODEL, _SYSTEM_PROMPT, user, temperature=0.2, max_tokens=400)
    except llm.LLMUnavailable:
        return Explanation(text=_fallback_text(rec), source="fallback")
    if not text:
        return Explanation(text=_fallback_text(rec), source="fallback")
    return Explanation(text=text, source="llm", model=EXPLANATION_MODEL)
