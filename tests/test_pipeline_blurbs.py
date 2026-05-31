"""Offline tests for the pipeline stage list and its stage blurbs (W2).

The blurbs narrate *what the pipeline is doing right now* during a live forecast.
THE RULE applies: they are status lines only — the generator may rephrase them in
the persona's voice, but a missing key, a bad shape, or no key at all must always
leave the committed template in place. No network, no live model.
"""

import json

from gas_agent import llm

from ceramics_agent import pipeline
from ceramics_agent.pipeline import (
    PIPELINE_STAGES,
    STAGE_KEYS,
    default_blurbs,
    generate_blurbs,
    stage_labels,
)


# --------------------------------------------------------------------------- #
# The stage list itself
# --------------------------------------------------------------------------- #
def test_pipeline_has_the_expected_ordered_stages():
    assert STAGE_KEYS == (
        "classify",
        "forecast_gas",
        "forecast_factors",
        "poll",
        "curate",
        "decide",
        "negotiate",
        "backtest",
    )
    # Every stage carries a fixed label and a non-empty committed template.
    for stage in PIPELINE_STAGES:
        assert stage.label
        assert stage.template
    assert stage_labels()[0] == PIPELINE_STAGES[0].label


def test_default_blurbs_cover_every_stage():
    blurbs = default_blurbs()
    assert set(blurbs) == set(STAGE_KEYS)
    assert all(text for text in blurbs.values())


# --------------------------------------------------------------------------- #
# Template fallback — the offline floor
# --------------------------------------------------------------------------- #
def test_no_key_returns_templates(monkeypatch):
    monkeypatch.setattr(llm, "featherless_available", lambda: False)
    assert generate_blurbs("a Bavarian tile maker") == default_blurbs()


def test_use_llm_false_returns_templates(monkeypatch):
    # Even with a key, use_llm=False must not call the model.
    monkeypatch.setattr(llm, "featherless_available", lambda: True)

    def _boom(*a, **k):
        raise AssertionError("generate_blurbs(use_llm=False) must not call the model")

    monkeypatch.setattr(llm, "chat_text", _boom)
    assert generate_blurbs("anything", use_llm=False) == default_blurbs()


# --------------------------------------------------------------------------- #
# LLM path (monkeypatched — never a live call)
# --------------------------------------------------------------------------- #
def test_llm_rewrites_are_taken_when_valid(monkeypatch):
    monkeypatch.setattr(llm, "featherless_available", lambda: True)
    rewritten = {key: f"We are now on {key}." for key in STAGE_KEYS}
    monkeypatch.setattr(llm, "chat_text", lambda *a, **k: json.dumps(rewritten))
    blurbs = generate_blurbs("a tile maker")
    assert blurbs == rewritten


def test_llm_partial_reply_backfills_missing_keys_from_templates(monkeypatch):
    monkeypatch.setattr(llm, "featherless_available", lambda: True)
    # Model only returned two stages; the rest must keep their templates.
    monkeypatch.setattr(llm, "chat_text", lambda *a, **k: json.dumps({
        "classify": "We are now reading the brief.",
        "backtest": "We are now replaying history.",
    }))
    blurbs = generate_blurbs("a tile maker")
    assert blurbs["classify"] == "We are now reading the brief."
    assert blurbs["backtest"] == "We are now replaying history."
    # An untouched stage keeps its committed template.
    assert blurbs["decide"] == default_blurbs()["decide"]
    assert set(blurbs) == set(STAGE_KEYS)


def test_llm_blank_or_nonstring_values_fall_back(monkeypatch):
    monkeypatch.setattr(llm, "featherless_available", lambda: True)
    monkeypatch.setattr(llm, "chat_text", lambda *a, **k: json.dumps({
        "classify": "   ",      # blank -> ignored
        "forecast_gas": 123,     # non-string -> ignored
    }))
    blurbs = generate_blurbs("a tile maker")
    assert blurbs["classify"] == default_blurbs()["classify"]
    assert blurbs["forecast_gas"] == default_blurbs()["forecast_gas"]


def test_llm_non_dict_payload_falls_back(monkeypatch):
    monkeypatch.setattr(llm, "featherless_available", lambda: True)
    monkeypatch.setattr(llm, "chat_text", lambda *a, **k: json.dumps(["not", "a", "dict"]))
    assert generate_blurbs("a tile maker") == default_blurbs()


def test_llm_invalid_json_falls_back(monkeypatch):
    monkeypatch.setattr(llm, "featherless_available", lambda: True)
    monkeypatch.setattr(llm, "chat_text", lambda *a, **k: "Sorry, here are the lines: ...")
    assert generate_blurbs("a tile maker") == default_blurbs()


def test_llm_failure_falls_back(monkeypatch):
    monkeypatch.setattr(llm, "featherless_available", lambda: True)

    def _boom(*a, **k):
        raise llm.LLMUnavailable("network down")

    monkeypatch.setattr(llm, "chat_text", _boom)
    assert generate_blurbs("a tile maker") == default_blurbs()
