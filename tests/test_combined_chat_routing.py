"""Offline test of the bottom chat's dual contract: one headline, both agents (W7).

No Streamlit, no session, no cache — this pins the *composition* the app's full-width
chat wires together so it runs in a clean checkout. The chat routes one message through
the SAME gas classifier (``parse_shock_request``) the gas hedge uses and the ceramics
factor router (``affected_factors_for``); a supply-shock headline therefore moves BOTH
decisions, while a "why" question moves neither. THE RULE holds throughout: the LLM only
reads severity, the deterministic policies own every number — proven here with the
keyword classifier (``use_llm=False``) and the Featherless-off fallback.
"""

from gas_agent import voice_chat
from gas_agent.hedge_policy import MonthDecision
from gas_agent.scenario import parse_shock_request

from ceramics_agent.cost_policy import CostWeights, decide_procurement, quarter_lock_ratio
from ceramics_agent.scenario import affected_factors_for, ceramics_shock_from_message
from ceramics_agent.forecast import FACTORS

from gas_agent.hedge_policy import MonthForecast

HORMUZ = "Iran closes the Strait of Hormuz"
MONTHS = ["2026-06-01", "2026-07-01", "2026-08-01"]


def _factor(level: float = 100.0, rel_band: float = 0.40) -> list[MonthForecast]:
    half = level * rel_band / 2.0
    return [MonthForecast(m, median=level, low=level - half, high=level + half) for m in MONTHS]


def _flat_factors() -> dict[str, list[MonthForecast]]:
    return {f: _factor() for f in FACTORS}


# --------------------------------------------------------------------------- #
# One headline → a shock signal for gas AND a routed factor for ceramics
# --------------------------------------------------------------------------- #
def test_canonical_headline_fires_both_agents():
    # The gas classifier sees a shock (the gas hedge will re-decide)…
    req = parse_shock_request(HORMUZ, use_llm=False)
    assert req.is_shock and req.magnitude > 0.0
    # …and the ceramics router sends it to the gas cost factor (the lock will re-decide),
    # so the SAME single message moves both decisions.
    assert affected_factors_for(HORMUZ) == ("gas",)


def test_ceramics_lock_rises_for_the_same_headline():
    factors, weights = _flat_factors(), CostWeights()
    calm = quarter_lock_ratio(decide_procurement(factors, weights))
    outcome = ceramics_shock_from_message(HORMUZ, factors, weights)
    assert outcome is not None
    assert outcome.affected_factors == ("gas",)
    assert outcome.lock_ratio > calm  # the ceramics half moves up, mirroring the gas hedge


# --------------------------------------------------------------------------- #
# A "why" question moves neither agent (read-only) and never re-decides a number
# --------------------------------------------------------------------------- #
def test_a_why_question_is_not_a_shock():
    factors, weights = _flat_factors(), CostWeights()
    question = "why this hedge ratio?"
    # Not a shock → the gas side leaves its decision in place…
    assert parse_shock_request(question, use_llm=False).is_shock is False
    # …and the ceramics side returns None (no re-decide), so the calm lock stands.
    assert ceramics_shock_from_message(question, factors, weights) is None


def test_why_answer_echoes_the_decided_numbers_not_new_ones(monkeypatch):
    # With Featherless off, the grounded answer restates the DECIDED gas ratio and the
    # ceramics lock from the state — it never emits a different number.
    import gas_agent.llm as llm
    monkeypatch.setattr(llm, "featherless_available", lambda: False)

    decision = MonthDecision(
        month="2026-06-01", median=47.0, low=40.0, high=55.0, band_width=0.30, drift_pct=0.02,
        ratio_from_band=0.31, direction_tilt=0.0, hedge_ratio=0.31, band_regime="tight",
        direction="rising", reason="trace", risk_premium=0.0,
    )
    state = voice_chat.ChatState(
        spot_price=47.28, decisions=[decision], quarter_ratio=0.31,
        ceramics_lock_ratio=0.40, ceramics_supplier="Alpine Clay Works",
    )
    answer = voice_chat.answer_question("why these decisions?", state)
    assert answer.source == "fallback"
    assert "31%" in answer.text          # the decided gas hedge
    assert "40%" in answer.text          # the decided ceramics lock
    assert "Alpine Clay Works" in answer.text
