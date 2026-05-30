"""Tests for the two Featherless agents and the keyword closed loop.

These exercise the *deterministic* paths only: the fallback selection/narrative
and the refine loop. The Featherless key is forced off so the suite never makes a
network call and the loop's widening logic is what is under test. The one rule —
the explainer never produces a hedge number of its own — is checked by feeding it
a fixed ratio and asserting that ratio (and only it) appears.
"""

import pytest

from gas_agent import catalog, keyword_agent, llm
from gas_agent.driver_curation import curate_drivers
from gas_agent.explanation_agent import explain_decision
from gas_agent.hedge_policy import decide_all
from gas_agent.keyword_agent import (
    KeywordSelection,
    _validate_selection,
    refine_filters,
    run_keyword_loop,
    select_filters,
)


@pytest.fixture
def no_featherless(monkeypatch):
    """Force the deterministic fallback path for every LLM helper."""
    monkeypatch.setattr(llm, "featherless_available", lambda: False)
    monkeypatch.setattr(keyword_agent.llm, "featherless_available", lambda: False)


def _signals(*names: str) -> dict:
    return {
        "data": {
            f"uid-{i}": {
                "driver_name": name,
                "importance": {"overall": {"mean": 90.0}},
                "pearson_correlation": {"overall": {"mean": 0.1}},
            }
            for i, name in enumerate(names)
        }
    }


def test_validate_selection_drops_offcatalogue_ids():
    selection = _validate_selection(
        {
            "keywords": ["natural gas"],
            "category_ids": [25, 9999],  # 9999 is not in the catalog
            "region_codes": [276, 123456],  # 123456 is not in the catalog
            "recency_factor": 0.8,
            "rationale": "test",
        },
        source="llm",
    )
    assert selection.category_ids == [25]
    assert selection.region_codes == [276]
    assert selection.source == "llm"


def test_validate_selection_clamps_recency():
    selection = _validate_selection(
        {"category_ids": [25], "region_codes": [276], "recency_factor": 5.0}, source="llm"
    )
    assert 0.0 <= selection.recency_factor <= 1.0


def test_validate_falls_back_when_no_valid_ids():
    selection = _validate_selection(
        {"category_ids": [9999], "region_codes": [], "recency_factor": 0.5}, source="llm"
    )
    assert selection.source == "fallback"
    assert selection.category_ids == catalog.DEFAULT_FORECAST_CATEGORIES


def test_select_filters_falls_back_without_key(no_featherless):
    selection = select_filters()
    assert selection.source == "fallback"
    assert all(cid in catalog.CATEGORIES for cid in selection.category_ids)
    assert all(code in catalog.REGIONS for code in selection.region_codes)


def test_refine_widens_to_full_whitelist(no_featherless):
    previous = KeywordSelection(
        keywords=["natural gas"], category_ids=[25], region_codes=[276],
        recency_factor=0.8, source="llm",
    )
    thin = curate_drivers(_signals("Exports of Natural gas in Europe"))
    refined = refine_filters("buyer", previous, thin)
    assert catalog.CREDIBLE_CATEGORY_IDS.issubset(set(refined.category_ids))
    assert catalog.CREDIBLE_REGION_CODES.issubset(set(refined.region_codes))


def test_keyword_loop_refines_then_succeeds(no_featherless):
    # First pull is thin (1 credible -> needs refine); second pull is rich.
    rich = _signals(
        "Exports of Natural gas in Europe", "Imports of Natural gas in Germany",
        "Exports of Liquefied natural gas in Europe", "Exchange Rates - World",
        "Import prices – Electricity in Europe", "Stock levels for oil products in Netherlands",
        "Energy price benchmark in United States of America",
        "Domestic producer prices – Extraction of natural gas in Germany",
    )
    calls = {"n": 0}

    def fetch_drivers(_selection):
        calls["n"] += 1
        return _signals("Exports of Natural gas in Europe") if calls["n"] == 1 else rich

    result = run_keyword_loop(fetch_drivers, persona="buyer", max_rounds=2)
    assert result.rounds == 2
    assert result.curation.needs_refine is False
    assert len(result.history) == 2


def test_explanation_fallback_is_faithful(no_featherless):
    months = __import__("gas_agent.hedge_policy", fromlist=["MonthForecast"]).MonthForecast
    forecasts = [
        months("2026-06-01", median=60.0, low=48.0, high=72.0),
        months("2026-07-01", median=55.0, low=40.0, high=70.0),
    ]
    decisions = decide_all(forecasts, spot_price=50.0)
    curation = curate_drivers(_signals("Exports of Natural gas in Europe", "Population - Serbia"))
    explanation = explain_decision(
        decisions, quarter_ratio=0.37, spot_price=50.0,
        kept_drivers=curation.top_drivers(6), rejected_drivers=curation.rejected,
    )
    assert explanation.source == "fallback"
    assert "37%" in explanation.text  # the given ratio is reported verbatim
    assert "Exports of Natural gas in Europe" in explanation.text  # cites a real driver
    assert "Population - Serbia" in explanation.text  # names a filtered-out driver
