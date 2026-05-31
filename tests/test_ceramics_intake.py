"""Offline tests for the company-description intake (W1).

The intake turns one free-text description into a complete ``CompanyProfile`` that
is the base case for both agents. THE RULE: the LLM only *extracts stated facts*
(monkeypatched here — no live model), it never invents an input or computes a
number; anything unstated is reported as a missing field and back-filled with the
committed defaults so the pipeline always runs. With no key it drops to a
deterministic keyword scan.
"""

from gas_agent import llm

from ceramics_agent.cost_policy import default_weights
from ceramics_agent.intake import (
    CompanyProfile,
    apply_answer,
    follow_up_questions,
    parse_description,
    weights_for_gas_exposure,
)


# --------------------------------------------------------------------------- #
# Deterministic fallback (no key / use_llm=False)
# --------------------------------------------------------------------------- #
def test_fallback_scan_extracts_obvious_facts():
    profile, missing = parse_description(
        "We make 5000 floor tiles in 2 weeks, high competition, gas-intensive firing.",
        use_llm=False,
    )
    assert profile.source == "fallback"
    assert profile.product_id == "tile"
    assert profile.quantity == 5000
    assert profile.timeline_days == 14  # "2 weeks"
    assert profile.competition == "high"
    assert profile.gas_exposure == "high"
    # Everything found above is no longer missing; sell regions were never stated.
    assert "product" not in missing
    assert "quantity" not in missing
    assert "sell_regions" in missing


def test_empty_description_is_all_default_and_all_missing():
    profile, missing = parse_description("")
    assert profile == CompanyProfile()  # the committed demo base case
    assert profile.product_id == "bowl"
    assert profile.quantity == 5000
    assert profile.timeline_days == 14
    assert profile.competition == "medium"
    assert set(missing) >= {"product", "quantity", "timeline_days", "competition"}


def test_no_key_forces_the_deterministic_path(monkeypatch):
    monkeypatch.setattr(llm, "featherless_available", lambda: False)
    profile, _ = parse_description("2000 dinnerware sets over 1 month", use_llm=True)
    assert profile.source == "fallback"
    assert profile.product_id == "dinnerware"
    assert profile.quantity == 2000
    assert profile.timeline_days == 30  # "1 month" → 30 days


# --------------------------------------------------------------------------- #
# LLM extraction path (monkeypatched — never a live call)
# --------------------------------------------------------------------------- #
def test_llm_extraction_fills_every_field(monkeypatch):
    monkeypatch.setattr(llm, "featherless_available", lambda: True)
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: {
        "product": "tile",
        "quantity": 8000,
        "timeline_days": 21,
        "competition": "medium",
        "gas_exposure": "high",
        "sell_regions": ["Germany", "France"],
        "notes": "exports across the EU",
    })
    profile, missing = parse_description("anything — the model is stubbed")
    assert profile.source == "llm"
    assert profile.product_id == "tile"
    assert profile.quantity == 8000
    assert profile.timeline_days == 21
    assert profile.competition == "medium"
    assert profile.gas_exposure == "high"
    assert profile.sell_regions == ("Germany", "France")
    assert profile.notes == "exports across the EU"
    assert missing == []  # nothing left to ask


def test_llm_null_fields_are_backfilled_from_the_text(monkeypatch):
    monkeypatch.setattr(llm, "featherless_available", lambda: True)
    # The model only caught the product; the rest must come from the keyword scan.
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: {
        "product": "bowl", "quantity": None, "timeline_days": None,
        "competition": None, "gas_exposure": None, "sell_regions": None,
    })
    profile, missing = parse_description("handmade bowls, 1200 units in 10 days")
    assert profile.product_id == "bowl"
    assert profile.quantity == 1200  # back-filled deterministically
    assert profile.timeline_days == 10
    assert "quantity" not in missing


def test_llm_invalid_payload_falls_back(monkeypatch):
    monkeypatch.setattr(llm, "featherless_available", lambda: True)
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: ["not", "a", "dict"])
    profile, _ = parse_description("3000 tiles in 3 weeks")
    assert profile.source == "fallback"
    assert profile.product_id == "tile"
    assert profile.quantity == 3000


def test_llm_failure_falls_back(monkeypatch):
    monkeypatch.setattr(llm, "featherless_available", lambda: True)

    def _boom(*a, **k):
        raise llm.LLMUnavailable("network down")

    monkeypatch.setattr(llm, "chat_json", _boom)
    profile, _ = parse_description("900 bowls in 5 days")
    assert profile.source == "fallback"
    assert profile.quantity == 900


def test_off_catalog_product_is_coerced_or_dropped(monkeypatch):
    monkeypatch.setattr(llm, "featherless_available", lambda: True)
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: {"product": "espresso cups"})
    # "cups" maps to bowl via the keyword scan; nothing else stated.
    profile, missing = parse_description("espresso cups")
    assert profile.product_id == "bowl"


# --------------------------------------------------------------------------- #
# gas-exposure → weights (deterministic input prep, never a decision number)
# --------------------------------------------------------------------------- #
def test_gas_exposure_maps_to_weights():
    high = weights_for_gas_exposure("high")
    low = weights_for_gas_exposure("low")
    assert high.gas > default_weights().gas >= low.gas
    assert weights_for_gas_exposure(None) == default_weights()
    assert weights_for_gas_exposure("nonsense") == default_weights()


def test_high_gas_exposure_profile_is_gas_heavy(monkeypatch):
    monkeypatch.setattr(llm, "featherless_available", lambda: False)
    profile, _ = parse_description("energy-intensive ceramics maker, 5000 tiles", use_llm=True)
    assert profile.gas_exposure == "high"
    assert profile.weights.gas > default_weights().gas


# --------------------------------------------------------------------------- #
# Follow-up questions
# --------------------------------------------------------------------------- #
def test_follow_up_questions_capped_and_prioritized():
    missing = ["sell_regions", "gas_exposure", "competition", "timeline_days", "quantity", "product"]
    questions = follow_up_questions(missing)
    assert len(questions) == 3
    # Core fields come first — the first question must be about the product.
    assert "producing" in questions[0].lower() or "bowls" in questions[0].lower()


def test_follow_up_questions_empty_when_nothing_missing():
    assert follow_up_questions([]) == []


# --------------------------------------------------------------------------- #
# Folding follow-up answers back in
# --------------------------------------------------------------------------- #
def test_apply_answer_updates_quantity():
    profile = CompanyProfile()
    updated = apply_answer(profile, "quantity", "we need 2500 units")
    assert updated.quantity == 2500
    assert profile.quantity == 5000  # original is frozen / untouched


def test_apply_answer_gas_exposure_also_moves_weights():
    profile = CompanyProfile()
    updated = apply_answer(profile, "gas_exposure", "low — we fire electric")
    assert updated.gas_exposure == "low"
    assert updated.weights.gas < default_weights().gas


def test_persona_includes_structured_facts():
    profile = CompanyProfile(description="We are a Bavarian tile maker.")
    persona = profile.persona()
    assert "Bavarian tile maker" in persona
    assert "Handmade Bowl" in persona  # the resolved product name
    assert "5,000" in persona
