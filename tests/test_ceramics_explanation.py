"""Offline tests for the ceramics explanation layer.

Featherless is monkeypatched — no network. The contract mirrors the gas
explainer: the LLM is handed a brief that already contains every final number
(lock %, supplier, channel, margin) under an explain-only, "do not re-decide"
system prompt, and when Featherless is unavailable or empty the deterministic
template restates the *same* chosen lock % and supplier — never a different one.
"""

import pytest

from gas_agent import llm

from ceramics_agent.cost_policy import default_weights
from ceramics_agent.explanation import explain_recommendation
from ceramics_agent.recommend import build_recommendation


@pytest.fixture(scope="module")
def rec():
    # Deterministic, offline (committed mock). Built once for the module.
    return build_recommendation("tile", 5000, 14, default_weights(), "medium")


# --------------------------------------------------------------------------- #
# Fallback (no Featherless)
# --------------------------------------------------------------------------- #
def test_fallback_restates_the_chosen_lock_and_supplier(monkeypatch, rec):
    monkeypatch.setattr(llm, "featherless_available", lambda: False)
    explanation = explain_recommendation(rec)
    assert explanation.source == "fallback"
    assert f"{rec.lock_ratio:.0%}" in explanation.text
    assert rec.chosen_supplier.name in explanation.text
    assert rec.chosen_channel.name in explanation.text


# --------------------------------------------------------------------------- #
# LLM path — grounded brief, explain-only prompt
# --------------------------------------------------------------------------- #
def test_llm_path_receives_a_grounded_final_brief(monkeypatch, rec):
    captured: dict[str, str] = {}

    def fake_chat_text(model, system, user, **kwargs):
        captured["system"] = system
        captured["user"] = user
        return "Narrated faithfully."

    monkeypatch.setattr(llm, "featherless_available", lambda: True)
    monkeypatch.setattr(llm, "chat_text", fake_chat_text)

    explanation = explain_recommendation(rec)
    assert explanation.source == "llm"
    assert explanation.text == "Narrated faithfully."

    # The brief carries the decided lock %, the chosen names and the negotiated margin.
    assert f"{rec.lock_ratio:.0%}" in captured["user"]
    assert rec.chosen_supplier.name in captured["user"]
    assert rec.chosen_channel.name in captured["user"]
    assert f"{rec.negotiation.unit_margin:.2f}" in captured["user"]
    # The numbers are flagged final in the brief...
    assert "FINAL" in captured["user"]
    # ...and the system prompt forbids re-deciding any of them (THE RULE).
    assert "Do NOT propose a different" in captured["system"]


def test_llm_empty_output_falls_back_to_template(monkeypatch, rec):
    monkeypatch.setattr(llm, "featherless_available", lambda: True)
    monkeypatch.setattr(llm, "chat_text", lambda *a, **k: "")
    explanation = explain_recommendation(rec)
    assert explanation.source == "fallback"
    assert f"{rec.lock_ratio:.0%}" in explanation.text


def test_llm_unavailable_error_falls_back_to_template(monkeypatch, rec):
    def boom(*a, **k):
        raise llm.LLMUnavailable("no key")

    monkeypatch.setattr(llm, "featherless_available", lambda: True)
    monkeypatch.setattr(llm, "chat_text", boom)
    explanation = explain_recommendation(rec)
    assert explanation.source == "fallback"
    assert rec.chosen_supplier.name in explanation.text
