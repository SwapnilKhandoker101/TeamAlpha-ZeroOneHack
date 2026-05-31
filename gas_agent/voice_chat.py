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

    # Optional second-decision (ceramics) context — explanation-only, set by the app
    # when the same buyer's ceramics line is on screen. All defaults leave the
    # gas-only brief byte-identical, so existing callers are unaffected. The figures
    # here are decided by the deterministic ceramics policy, never by this assistant.
    ceramics_lock_ratio: float | None = None
    ceramics_band_regime: str = ""
    ceramics_supplier: str = ""
    ceramics_channel: str = ""
    ceramics_unit_margin: float | None = None
    ceramics_scenario_lock: float | None = None  # the shocked lock when a shock is active


_SYSTEM_PROMPT = (
    "You are the voice assistant of a forecasting agent for a manufacturer, answering "
    "a procurement lead's spoken question. The buyer faces two decisions — a gas hedge "
    "ratio and, when shown, a ceramics input-cost lock — and a deterministic policy has "
    "ALREADY decided BOTH; you only EXPLAIN the existing decisions and the data behind "
    "them.\n\n"
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


def _ceramics_clause(state: ChatState) -> str:
    """A compact brief of the SECOND decision (the ceramics input-cost lock) when the
    app supplies it, so the assistant can answer 'why this lock / which supplier' for
    the ceramics line too — explanation-only, never re-deciding it. Empty when absent."""
    if state.ceramics_lock_ratio is None:
        return ""
    bits = [
        f"\nSECOND DECISION — ceramics line (also FINAL, decided by the deterministic "
        f"cost policy): lock {state.ceramics_lock_ratio:.0%} of next quarter's input cost "
        f"now ({state.ceramics_band_regime or 'normal'} blended cost band)."
    ]
    if state.ceramics_scenario_lock is not None:
        bits.append(f" Under the active shock this lock moved to "
                    f"{state.ceramics_scenario_lock:.0%}.")
    if state.ceramics_supplier:
        bits.append(f" Chosen supplier: {state.ceramics_supplier}.")
    if state.ceramics_channel:
        bits.append(f" Chosen sales channel: {state.ceramics_channel}.")
    if state.ceramics_unit_margin is not None:
        bits.append(f" Negotiated unit margin: EUR {state.ceramics_unit_margin:.2f}.")
    return "".join(bits)


def _build_factsheet(question: str, state: ChatState) -> str:
    quarter_label = " / ".join(d.month[:7] for d in state.decisions)
    rejected_examples = ", ".join(d.name for d in state.rejected_drivers[:3]) or "none"
    return (
        f"QUESTION (spoken by the buyer): {question}\n\n"
        f"Today's TTF spot price: EUR {state.spot_price:.0f}/MWh\n"
        f"Decision (deterministic policy, FINAL): lock {state.quarter_ratio:.0%} of next "
        f"quarter ({quarter_label}) forward now.\n"
        f"{_scenario_clause(state)}\n{_premium_clause(state)}\n"
        f"{_ceramics_clause(state)}\n\n"
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
    ceramics_phrase = ""
    if state.ceramics_lock_ratio is not None:
        ceramics_phrase = (
            f" On the ceramics line, the same deterministic approach locks "
            f"{state.ceramics_lock_ratio:.0%} of next quarter's input cost"
        )
        if state.ceramics_scenario_lock is not None:
            ceramics_phrase += f" (up to {state.ceramics_scenario_lock:.0%} under the shock)"
        if state.ceramics_supplier:
            ceramics_phrase += f", buying from {state.ceramics_supplier}"
        ceramics_phrase += "."
    return (
        f"The agent locks {state.quarter_ratio:.0%} of next quarter ({quarter_label}) "
        f"forward, at a TTF spot of EUR {state.spot_price:.0f}/MWh. The size comes from "
        f"the forecast's confidence band, not its midpoint — a tighter band means a "
        f"higher lock. {driver_phrase}.{premium_phrase}{scenario_phrase}{ceramics_phrase}"
    )


# --------------------------------------------------------------------------- #
# "About this app" — the agent can explain itself (W11). Explanation-only: this
# describes the SYSTEM, it never emits or alters a decision number. Grounded on a
# single factual overview sourced from CLAUDE.md so it can't drift into invention.
# --------------------------------------------------------------------------- #
APP_OVERVIEW = (
    "This is a forecasting-AI decision agent for a mid-size German glass & ceramics "
    "manufacturer, built on the Sybilion probabilistic forecasting API. From one company "
    "description it makes TWO decisions:\n"
    "1) GAS HEDGE — what share of next quarter's natural gas (the Dutch TTF benchmark) to "
    "lock in forward now versus buy later on the spot market. It is NOT a price prediction: "
    "Sybilion's point forecast for gas is weak (~28% error), so the decision is sized from "
    "the forecast's CONFIDENCE BAND (a tighter band → lock more) plus a curated mix of "
    "credible supply/demand drivers — never the point estimate.\n"
    "2) CERAMICS LOCK — for one production run, how much blended input cost (gas, clay, "
    "power, freight) to lock now, which supplier to buy from, and which sales channel to "
    "sell through. The lock % reuses the SAME hedge-policy engine on a 4-factor cost band.\n"
    "THE CORE PRINCIPLE: the LLM never computes a decision number. Deterministic policy code "
    "computes every hedge ratio, lock %, supplier/channel score, negotiated quote and backtest "
    "result, so identical inputs always reproduce the identical decision and every number is "
    "auditable. The LLM only PREPARES inputs (it picks Sybilion filters and reads a supply "
    "shock's severity from free text) and EXPLAINS outputs (it narrates the already-decided "
    "numbers). The app runs fully offline with no API keys (a committed cached forecast plus "
    "template narration). A live supply-shock scenario lets you change an assumption mid-run and "
    "watch BOTH decisions adapt instantly, and reproducible backtests show the policy beats "
    "naive baselines. Built with Sybilion (forecasts), Featherless (LLM narration), NVIDIA Riva "
    "(voice), and Streamlit + Plotly (the dashboard and globe)."
)

# Substring cues that a message is asking about the app / methodology itself, rather
# than about the current decision (those stay with answer_question). Deterministic.
_ABOUT_TRIGGERS: tuple[str, ...] = (
    "what is this", "what's this", "what is the app", "what does this app",
    "what does the app", "about this app", "about the app", "what can you do",
    "what do you do", "who is this for", "who's this for", "why was it built",
    "why was this built", "why build", "why this design", "why is it built",
    "why did you build", "how does this app", "how does the app", "how does it work",
    "how do you work", "how does the hedge ratio work", "how is the hedge ratio",
    "how does the lock work", "how is the lock computed", "how do you decide",
    "how does it decide", "what is sybilion", "what's sybilion", "explain the app",
    "explain this app", "tell me about this app", "what is this app",
)

_ABOUT_SYSTEM = (
    "You are the assistant of a forecasting-AI decision agent, answering a question about "
    "WHAT THE APP IS and HOW IT WORKS. Answer ONLY from the overview below. Do NOT invent "
    "features, and do NOT state or imply any hedge ratio or lock percentage — this is a "
    "description of the system, not a decision. 2-5 sentences, plain and conversational "
    "(it may be read aloud), no headings or bullet lists."
)


def is_about_question(message: str) -> bool:
    """True when a chat message is asking about the app / methodology itself (so the
    router answers from :data:`APP_OVERVIEW` instead of the current decision)."""
    lower = (message or "").lower()
    return any(trigger in lower for trigger in _ABOUT_TRIGGERS)


def _about_fallback(question: str) -> str:
    """Deterministic self-description for when Featherless is unavailable — captures the
    essence of :data:`APP_OVERVIEW` and THE RULE, with no decision number."""
    return (
        "This is a forecasting-AI decision agent for a German glass & ceramics maker, built on "
        "the Sybilion probabilistic forecasting API. From one company description it makes two "
        "auditable decisions — how much gas to lock forward, and how to run a ceramics production "
        "run (input-cost lock, supplier, sales channel) — sizing each from a forecast confidence "
        "band rather than a point prediction. Crucially, the LLM never computes a number: "
        "deterministic policy code decides every figure and the model only prepares inputs and "
        "explains the result, so the same inputs always reproduce the same decision. It runs "
        "offline with no keys, and a live supply-shock scenario lets you watch both decisions adapt."
    )


def answer_about(question: str) -> Explanation:
    """Answer an 'about this app' question from :data:`APP_OVERVIEW` (explanation-only).
    Featherless writes the prose; with no key (or on failure) the deterministic
    self-description answers. Never emits a decision number."""
    question = (question or "").strip()
    if not question or not llm.featherless_available():
        return Explanation(text=_about_fallback(question), source="fallback")
    user = f"QUESTION: {question}\n\nAPP OVERVIEW:\n{APP_OVERVIEW}"
    try:
        text = llm.chat_text(EXPLANATION_MODEL, _ABOUT_SYSTEM, user, temperature=0.2, max_tokens=320)
    except llm.LLMUnavailable:
        return Explanation(text=_about_fallback(question), source="fallback")
    if not text:
        return Explanation(text=_about_fallback(question), source="fallback")
    return Explanation(text=text, source="llm", model=EXPLANATION_MODEL)


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
