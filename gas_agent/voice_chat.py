"""The grounded Q&A layer for the push-to-talk voice assistant.

A procurement lead can *speak* a question — "why this hedge ratio?", "which
supplier matters most?", "why this country?" — and get a spoken answer grounded
in the numbers the dashboard is already showing. Like :mod:`explanation_agent`,
this layer only EXPLAINS: a deterministic policy has already fixed the ratio, and
the prompt forbids inventing drivers/numbers or proposing a different ratio
(THE RULE). Featherless writes the prose; with no key a deterministic template
answers from the same brief, so the assistant always responds offline.

The answer is intentionally short and conversational because it is read aloud.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gas_agent import llm
from gas_agent.config import EXPLANATION_MODEL
from gas_agent.driver_curation import CuratedDriver
from gas_agent.explanation_agent import Explanation, _decision_lines
from gas_agent.hedge_policy import MonthDecision
from gas_agent.keyword_agent import DEFAULT_PERSONA


@dataclass(frozen=True)
class ChatState:
    """The already-computed dashboard context a spoken answer is grounded in.

    Built once by the app from the same numbers on screen, so the assistant can
    never drift from what the deterministic policy decided.
    """

    spot_price: float
    decisions: list[MonthDecision]
    quarter_ratio: float
    kept_drivers: list[CuratedDriver] = field(default_factory=list)
    rejected_drivers: list[CuratedDriver] = field(default_factory=list)
    standing_premium: float = 0.0
    scenario_label: str = ""
    scenario_magnitude: float = 0.0


_SYSTEM_PROMPT = (
    "You are the voice assistant of a gas-hedging agent, answering a procurement "
    "lead's spoken question. A deterministic policy has ALREADY decided the hedge "
    "ratio; you only EXPLAIN the existing decision and the data behind it.\n\n"
    "Strict rules:\n"
    "- Answer ONLY from the brief below. Never invent drivers, numbers, suppliers, "
    "regions, or events.\n"
    "- The hedge ratio and every figure are FINAL. Never propose, recompute, or hint "
    "at a different ratio. If asked what the ratio 'should' be, restate the decided one.\n"
    "- For 'which supplier/country' questions, name the highest-importance credible "
    "drivers and their regions from the brief.\n"
    "- If the brief does not contain the answer, say so plainly rather than guessing.\n"
    "- 2-4 sentences, plain and conversational — it is read aloud, so no headings, "
    "bullet lists, or preamble."
)


def _qa_driver_lines(drivers: list[CuratedDriver]) -> str:
    return "\n".join(
        f"  - {d.name} (importance {d.importance:.0f}, region {d.region or 'n/a'}, "
        f"{d.theme}, correlation {d.correlation:+.2f})"
        for d in drivers
    )


def _scenario_clause(state: ChatState) -> str:
    if state.scenario_magnitude > 0 and state.scenario_label:
        return (f"Active supply-shock scenario: {state.scenario_label} "
                f"(severity {state.scenario_magnitude:.0%}). The deterministic policy "
                f"re-ran on the shocked band.")
    return "No shock active — calm base case."


def _premium_clause(state: ChatState) -> str:
    if state.standing_premium > 0:
        return (f"Standing supply-risk premium baked into the lock floor: "
                f"+{state.standing_premium:.0%}.")
    return "No standing supply-risk premium (no supply-risk-region drivers dominate)."


def _build_factsheet(question: str, state: ChatState) -> str:
    quarter_label = " / ".join(d.month[:7] for d in state.decisions)
    rejected_examples = ", ".join(d.name for d in state.rejected_drivers[:3]) or "none"
    return (
        f"QUESTION (spoken by the buyer): {question}\n\n"
        f"Today's TTF spot price: EUR {state.spot_price:.0f}/MWh\n"
        f"Decision (deterministic policy, FINAL): lock {state.quarter_ratio:.0%} of next "
        f"quarter ({quarter_label}) forward now.\n"
        f"{_scenario_clause(state)}\n{_premium_clause(state)}\n\n"
        f"Per-month detail:\n{_decision_lines(state.decisions)}\n\n"
        f"Most credible drivers (kept after curation), with region:\n"
        f"{_qa_driver_lines(state.kept_drivers)}\n\n"
        f"Curation discarded {len(state.rejected_drivers)} spurious drivers, "
        f"e.g.: {rejected_examples}."
    )


def _fallback_answer(question: str, state: ChatState) -> str:
    """Deterministic grounded answer for when Featherless is unavailable. It always
    restates the DECIDED ratio — it never emits a different one."""
    quarter_label = " / ".join(d.month[:7] for d in state.decisions)
    top = state.kept_drivers[0] if state.kept_drivers else None
    if top:
        driver_phrase = f"The strongest credible driver is {top.name}"
        if top.region:
            driver_phrase += f" ({top.region})"
    else:
        driver_phrase = "Credible energy and FX drivers underpin the forecast"
    premium_phrase = (
        f" A standing supply-risk premium of +{state.standing_premium:.0%} is baked "
        f"into the lock floor."
        if state.standing_premium > 0 else ""
    )
    scenario_phrase = (
        f" Under the active {state.scenario_label} scenario "
        f"(severity {state.scenario_magnitude:.0%}), the deterministic policy re-ran on "
        f"the shocked band."
        if state.scenario_magnitude > 0 and state.scenario_label else ""
    )
    return (
        f"The agent locks {state.quarter_ratio:.0%} of next quarter ({quarter_label}) "
        f"forward, at a TTF spot of EUR {state.spot_price:.0f}/MWh. The size comes from "
        f"the forecast's confidence band, not its midpoint — a tighter band means a "
        f"higher lock. {driver_phrase}.{premium_phrase}{scenario_phrase}"
    )


def answer_question(
    question: str,
    state: ChatState,
    persona: str = DEFAULT_PERSONA,
) -> Explanation:
    """Answer a spoken question, grounded strictly in ``state``. Featherless writes
    the prose; with no key (or on failure / empty output) a deterministic template
    answers from the same brief. Never proposes a ratio other than the decided one."""
    question = (question or "").strip()
    factsheet = _build_factsheet(question, state)

    if not question or not llm.featherless_available():
        return Explanation(text=_fallback_answer(question, state), source="fallback")

    user = f"BUYER:\n{persona}\n\nDECISION BRIEF:\n{factsheet}"
    try:
        text = llm.chat_text(EXPLANATION_MODEL, _SYSTEM_PROMPT, user, temperature=0.2, max_tokens=320)
    except llm.LLMUnavailable:
        return Explanation(text=_fallback_answer(question, state), source="fallback")
    if not text:
        return Explanation(text=_fallback_answer(question, state), source="fallback")
    return Explanation(text=text, source="llm", model=EXPLANATION_MODEL)
