"""The explanation agent — Featherless narrates the decision it did not make.

The hedge ratio is already fixed by :mod:`gas_agent.hedge_policy` before this
agent runs. Its only job is to phrase *why* that ratio is what it is, in language
a procurement lead can act on, citing the real curated driver names and the real
band/drift numbers it is handed. It is explicitly told the decision is final and
that it must not invent drivers, invent numbers, or propose a different ratio —
that keeps the narrative faithful to the deterministic engine rather than letting
the LLM quietly re-decide.

If Featherless is unavailable the agent returns a deterministic template
narrative built from the same numbers, so the dashboard always has an
explanation and the demo never depends on a live call.
"""

from __future__ import annotations

from dataclasses import dataclass

from gas_agent import llm
from gas_agent.config import EXPLANATION_MODEL
from gas_agent.driver_curation import CuratedDriver
from gas_agent.hedge_policy import MonthDecision
from gas_agent.keyword_agent import DEFAULT_PERSONA


@dataclass(frozen=True)
class Explanation:
    """The narrative for one decision, plus how it was produced."""

    text: str
    source: str  # "llm" | "fallback"
    model: str = ""


def _driver_lines(drivers: list[CuratedDriver]) -> str:
    return "\n".join(
        f"  - {d.name} (importance {d.importance:.0f}, {d.theme}, "
        f"correlation {d.correlation:+.2f})"
        for d in drivers
    )


def _decision_lines(decisions: list[MonthDecision]) -> str:
    return "\n".join(
        f"  - {d.month[:7]}: median EUR {d.median:.0f}/MWh, {d.band_regime} band "
        f"({d.band_width:.0%} of median), forward {d.direction} vs spot "
        f"({d.drift_pct:+.0%}) -> lock {d.hedge_ratio:.0%}"
        for d in decisions
    )


def _build_factsheet(
    decisions: list[MonthDecision],
    quarter_ratio: float,
    spot_price: float,
    kept_drivers: list[CuratedDriver],
    rejected_drivers: list[CuratedDriver],
) -> str:
    quarter_label = " / ".join(d.month[:7] for d in decisions)
    rejected_examples = ", ".join(d.name for d in rejected_drivers[:3]) or "none"
    return (
        f"Today's TTF spot price: EUR {spot_price:.0f}/MWh\n"
        f"Decision (computed by deterministic policy, FINAL): lock {quarter_ratio:.0%} "
        f"of next quarter ({quarter_label}) forward now.\n\n"
        f"Per-month detail:\n{_decision_lines(decisions)}\n\n"
        f"Most credible drivers behind the forecast (kept after curation):\n"
        f"{_driver_lines(kept_drivers)}\n\n"
        f"Curation discarded {len(rejected_drivers)} spurious drivers, e.g.: {rejected_examples}."
    )


_SYSTEM_PROMPT = (
    "You are the explanation layer of a gas-hedging agent. A deterministic policy "
    "has ALREADY decided the hedge ratio from the forecast's confidence band and "
    "its drift versus spot. Your job is to explain, for a procurement lead, WHY the "
    "decision is what it is.\n\n"
    "Strict rules:\n"
    "- Use ONLY the driver names and numbers in the brief. Never invent drivers or figures.\n"
    "- The hedge ratio is final. Do NOT suggest a different number.\n"
    "- The confidence band sets the base size: a tighter band justifies locking more, a "
    "wider band justifies keeping optionality. The forward-vs-spot drift is only a "
    "secondary tilt: a forward ABOVE spot tilts toward locking more, a forward BELOW "
    "spot tilts toward locking less. Never call a forward below spot a reason to lock more.\n"
    "- Name the strongest credible drivers, and note that spurious drivers were filtered out.\n"
    "- 120-200 words, plain and direct. No headings, no bullet lists, no preamble."
)


def _fallback_text(
    decisions: list[MonthDecision],
    quarter_ratio: float,
    spot_price: float,
    kept_drivers: list[CuratedDriver],
    rejected_drivers: list[CuratedDriver],
) -> str:
    quarter_label = " / ".join(d.month[:7] for d in decisions)
    top = kept_drivers[0] if kept_drivers else None
    per_month = "; ".join(
        f"{d.month[:7]} {d.band_regime} band -> {d.hedge_ratio:.0%}" for d in decisions
    )
    driver_phrase = (
        f"The strongest credible signal is {top.name} (importance {top.importance:.0f})"
        if top
        else "Credible energy and FX drivers underpin the forecast"
    )
    rejected_phrase = ""
    if rejected_drivers:
        rejected_phrase = (
            f" Curation discarded {len(rejected_drivers)} spurious drivers such as "
            f"{rejected_drivers[0].name}, which correlate in-sample but cannot move the gas price."
        )
    return (
        f"At today's TTF spot of EUR {spot_price:.0f}/MWh, the agent locks "
        f"{quarter_ratio:.0%} of next quarter ({quarter_label}) forward. The size is "
        f"driven by the forecast's confidence band, not its midpoint: a tighter band "
        f"means more conviction and a higher lock, a wider band keeps optionality, and "
        f"a forward above spot tilts toward locking more. Month by month: {per_month}. "
        f"{driver_phrase}.{rejected_phrase}"
    )


def explain_decision(
    decisions: list[MonthDecision],
    quarter_ratio: float,
    spot_price: float,
    kept_drivers: list[CuratedDriver],
    rejected_drivers: list[CuratedDriver],
    persona: str = DEFAULT_PERSONA,
) -> Explanation:
    """Produce a faithful narrative for the (already decided) hedge ratio."""
    factsheet = _build_factsheet(decisions, quarter_ratio, spot_price, kept_drivers, rejected_drivers)

    if not llm.featherless_available():
        text = _fallback_text(decisions, quarter_ratio, spot_price, kept_drivers, rejected_drivers)
        return Explanation(text=text, source="fallback")

    user = f"BUYER:\n{persona}\n\nDECISION BRIEF:\n{factsheet}"
    try:
        text = llm.chat_text(EXPLANATION_MODEL, _SYSTEM_PROMPT, user, temperature=0.2, max_tokens=400)
    except llm.LLMUnavailable:
        text = _fallback_text(decisions, quarter_ratio, spot_price, kept_drivers, rejected_drivers)
        return Explanation(text=text, source="fallback")
    if not text:
        text = _fallback_text(decisions, quarter_ratio, spot_price, kept_drivers, rejected_drivers)
        return Explanation(text=text, source="fallback")
    return Explanation(text=text, source="llm", model=EXPLANATION_MODEL)
