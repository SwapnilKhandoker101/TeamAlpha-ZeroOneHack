"""Offline tests for the grounded voice Q&A layer (Workstream 4).

No network: Featherless is monkeypatched. The contract: ``answer_question``
grounds strictly on the supplied state, and — crucially, since this is the voice
of an explanation-only assistant — it NEVER emits a hedge ratio different from the
one the deterministic policy decided. The fallback (no Featherless) must also
hold that line.
"""

from gas_agent import llm
from gas_agent import voice_chat
from gas_agent.driver_curation import CuratedDriver
from gas_agent.hedge_policy import MonthDecision
from gas_agent.voice_chat import ChatState, answer_question


def _decision(month: str, ratio: float, premium: float = 0.0) -> MonthDecision:
    return MonthDecision(
        month=month, median=47.0, low=40.0, high=55.0, band_width=0.30, drift_pct=0.02,
        ratio_from_band=ratio, direction_tilt=0.0, hedge_ratio=ratio, band_regime="tight",
        direction="rising", reason="trace", risk_premium=premium,
    )


def _driver(name: str, importance: float, region: str, theme: str = "energy trade flow") -> CuratedDriver:
    return CuratedDriver(
        name=name, importance=importance, correlation=0.2, region=region, theme=theme,
        verdict="keep", reason="credible",
    )


def _state(ratio: float = 0.53, **overrides) -> ChatState:
    base = dict(
        spot_price=47.28,
        decisions=[_decision("2026-06-01", ratio)],
        quarter_ratio=ratio,
        kept_drivers=[_driver("Exports of Natural gas in Russian Federation", 100.0, "Russian Federation")],
        rejected_drivers=[_driver("Population - Sri Lanka", 12.0, "Sri Lanka", theme="")],
    )
    base.update(overrides)
    return ChatState(**base)


def test_fallback_echoes_the_decided_ratio(monkeypatch):
    monkeypatch.setattr(llm, "featherless_available", lambda: False)
    answer = answer_question("why this hedge ratio?", _state(0.53))
    assert answer.source == "fallback"
    assert "53%" in answer.text


def test_fallback_never_emits_a_different_ratio(monkeypatch):
    monkeypatch.setattr(llm, "featherless_available", lambda: False)
    # The emitted lock figure must TRACK the decided ratio, not a constant.
    assert "20%" in answer_question("q", _state(0.20)).text
    assert "75%" in answer_question("q", _state(0.75)).text
    # And the answer for a 20% decision must not assert a 75% lock (or vice-versa).
    twenty = answer_question("q", _state(0.20)).text
    assert "75%" not in twenty


def test_fallback_grounds_on_supplied_drivers(monkeypatch):
    monkeypatch.setattr(llm, "featherless_available", lambda: False)
    answer = answer_question("which supplier matters most?", _state())
    # Names the top kept driver and its region from the state — invents nothing.
    assert "Russian Federation" in answer.text


def test_fallback_surfaces_active_scenario(monkeypatch):
    monkeypatch.setattr(llm, "featherless_available", lambda: False)
    state = _state(0.40, scenario_label="Strait of Hormuz disruption", scenario_magnitude=1.0)
    answer = answer_question("what changed?", state)
    assert "Hormuz" in answer.text
    assert "40%" in answer.text


def test_fallback_surfaces_standing_premium(monkeypatch):
    monkeypatch.setattr(llm, "featherless_available", lambda: False)
    answer = answer_question("why so high?", _state(0.53, standing_premium=0.07))
    assert "+7%" in answer.text


def test_fallback_surfaces_the_ceramics_lock_when_supplied(monkeypatch):
    # When the app puts the ceramics line on screen, the spoken fallback names that
    # SECOND decision too — grounded in the figure the deterministic cost policy decided,
    # never a re-computed one.
    monkeypatch.setattr(llm, "featherless_available", lambda: False)
    state = _state(0.53, ceramics_lock_ratio=0.40, ceramics_supplier="Alpine Clay Works")
    answer = answer_question("what about ceramics?", state)
    assert "53%" in answer.text  # the gas hedge stays
    assert "40%" in answer.text and "ceramics" in answer.text.lower()
    assert "Alpine Clay Works" in answer.text


def test_fallback_omits_ceramics_when_absent(monkeypatch):
    # Default (gas-only) state must stay byte-identical — no ceramics sentence leaks in.
    monkeypatch.setattr(llm, "featherless_available", lambda: False)
    assert "ceramics" not in answer_question("q", _state(0.53)).text.lower()


def test_fallback_surfaces_the_shocked_ceramics_lock(monkeypatch):
    monkeypatch.setattr(llm, "featherless_available", lambda: False)
    state = _state(0.53, ceramics_lock_ratio=0.40, ceramics_scenario_lock=0.55)
    answer = answer_question("what did the shock do to ceramics?", state)
    assert "55%" in answer.text  # the shocked lock is surfaced for the second line


def test_ceramics_brief_reaches_the_llm(monkeypatch):
    captured: dict[str, str] = {}

    def fake_chat_text(model, system, user, **kwargs):
        captured["user"] = user
        return "Spoken grounded answer."

    monkeypatch.setattr(llm, "featherless_available", lambda: True)
    monkeypatch.setattr(llm, "chat_text", fake_chat_text)
    answer_question("why this ceramics lock?",
                    _state(0.53, ceramics_lock_ratio=0.40, ceramics_supplier="Alpine Clay Works"))
    assert "SECOND DECISION" in captured["user"]
    assert "40%" in captured["user"] and "Alpine Clay Works" in captured["user"]


def test_llm_path_receives_grounded_brief(monkeypatch):
    captured: dict[str, str] = {}

    def fake_chat_text(model, system, user, **kwargs):
        captured["system"] = system
        captured["user"] = user
        return "Spoken grounded answer."

    monkeypatch.setattr(llm, "featherless_available", lambda: True)
    monkeypatch.setattr(llm, "chat_text", fake_chat_text)

    answer = answer_question("which country?", _state(0.53))
    assert answer.source == "llm"
    assert answer.text == "Spoken grounded answer."
    # The brief handed to the model carries the decided ratio and the real driver.
    assert "53%" in captured["user"]
    assert "Russian Federation" in captured["user"]
    # The system prompt forbids re-deciding the ratio.
    assert "FINAL" in captured["system"]


def test_llm_empty_output_falls_back(monkeypatch):
    monkeypatch.setattr(llm, "featherless_available", lambda: True)
    monkeypatch.setattr(llm, "chat_text", lambda *a, **k: "")
    answer = answer_question("q", _state(0.53))
    assert answer.source == "fallback"
    assert "53%" in answer.text


# --------------------------------------------------------------------------- #
# W11 — the app can explain ITSELF (about-intent), explanation-only
# --------------------------------------------------------------------------- #
def test_about_intent_detects_app_and_methodology_questions():
    for q in ("What is this app?", "why was it built this way?",
              "how does the hedge ratio work?", "what is Sybilion?", "what can you do?"):
        assert voice_chat.is_about_question(q)
    # A question about the CURRENT decision is NOT an about-question (stays grounded).
    for q in ("why this hedge ratio?", "which supplier matters most?", "what changed?"):
        assert not voice_chat.is_about_question(q)


def test_about_fallback_describes_the_app_without_a_decision_number(monkeypatch):
    monkeypatch.setattr(llm, "featherless_available", lambda: False)
    answer = voice_chat.answer_about("what is this app?")
    assert answer.source == "fallback"
    # Describes the system + THE RULE…
    assert "Sybilion" in answer.text
    assert "never computes" in answer.text.lower() or "deterministic" in answer.text.lower()
    # …and emits no hedge ratio / lock percentage (it's a description, not a decision).
    assert "%" not in answer.text


def test_about_uses_llm_grounded_on_the_overview(monkeypatch):
    captured: dict[str, str] = {}

    def fake_chat_text(model, system, user, **kwargs):
        captured["system"] = system
        captured["user"] = user
        return "A grounded description of the app."

    monkeypatch.setattr(llm, "featherless_available", lambda: True)
    monkeypatch.setattr(llm, "chat_text", fake_chat_text)

    answer = voice_chat.answer_about("why this design?")
    assert answer.source == "llm" and answer.text == "A grounded description of the app."
    # The model is handed the APP_OVERVIEW and told not to emit a decision number.
    assert "TWO decisions" in captured["user"] or "two decisions" in captured["user"].lower()
    assert "not a decision" in captured["system"].lower()
