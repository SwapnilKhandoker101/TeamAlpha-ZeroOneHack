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
    estimate_product,
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


# --------------------------------------------------------------------------- #
# estimate_product — off-catalog recognition (full-live mode); LLM-estimated INPUT spec
# --------------------------------------------------------------------------- #
def test_estimate_product_falls_back_to_catalog_offline(monkeypatch):
    # No LLM → a committed catalog product, never an estimated spec (offline floor).
    monkeypatch.setattr(llm, "featherless_available", lambda: False)
    product = estimate_product("We make 8000 floor tiles.")
    assert product.id == "tile" and product.estimated is False


def test_estimate_product_returns_catalog_when_llm_recognises_one(monkeypatch):
    monkeypatch.setattr(llm, "featherless_available", lambda: True)
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: {"catalog_id": "dinnerware"})
    product = estimate_product("We make dinner sets.")
    assert product.id == "dinnerware" and product.estimated is False


def test_estimate_product_builds_an_estimated_custom_spec(monkeypatch):
    monkeypatch.setattr(llm, "featherless_available", lambda: True)
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: {
        "catalog_id": None, "name": "Ceramic Sink",
        "clay_kg": 9.0, "glaze_kg": 0.8, "kiln_kwh": 14.0, "firing_gas_kwh": 55.0, "ship_kg": 12.0,
    })
    product = estimate_product("We make 10000 ceramic sinks.")
    assert product.id == "custom" and product.estimated is True
    assert product.name == "Ceramic Sink"
    assert product.firing_gas_kwh == 55.0 and product.clay_kg == 9.0


def test_estimate_product_clamps_absurd_values(monkeypatch):
    monkeypatch.setattr(llm, "featherless_available", lambda: True)
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: {
        "catalog_id": None, "name": "Giant thing",
        "clay_kg": 999999, "glaze_kg": -5, "kiln_kwh": 0, "firing_gas_kwh": "bad", "ship_kg": 9e9,
    })
    product = estimate_product("absurd")
    assert 0.1 <= product.clay_kg <= 300.0          # clamped to the sane ceiling
    assert product.glaze_kg == 0.0                  # negative → floor 0
    assert product.kiln_kwh == 0.1                  # 0 → floor 0.1 (must be positive)
    assert product.firing_gas_kwh == 30.0           # non-numeric → the documented default
    assert product.ship_kg == 500.0                 # huge → ceiling
