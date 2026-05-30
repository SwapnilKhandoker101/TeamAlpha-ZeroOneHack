"""Unit tests for the driver-globe geography.

Coordinate lookup and aggregation are pure and tested directly. The per-country
brief is tested on both paths by monkeypatching the LLM call, so no network is
ever required.
"""

import gas_agent.llm as llm_module
from gas_agent.driver_curation import CuratedDriver, CurationResult
from gas_agent.geo import (
    _FALLBACK_BRIEFS,
    aggregate_drivers,
    coords_for,
    country_brief,
)


def _curation() -> CurationResult:
    kept = [
        CuratedDriver("Exports of Natural gas - Russian Federation", 90.0, 0.5,
                      "Russian Federation", "natural gas", "keep", "x"),
        CuratedDriver("Imports of Natural gas - Russian Federation", 60.0, 0.4,
                      "Russian Federation", "energy trade flow", "keep", "x"),
        CuratedDriver("Electricity prices - Germany", 50.0, 0.4, "Germany",
                      "electricity & power", "keep", "x"),
        CuratedDriver("Exchange Rates - Europe", 40.0, 0.3, "Europe",
                      "exchange rates (FX)", "keep", "x"),
        CuratedDriver("Exchange Rates - World", 30.0, 0.3, "World", "commodities", "keep", "x"),
    ]
    rejected = [
        CuratedDriver("Population - Sri Lanka", 70.0, 0.6, "Sri Lanka", "", "reject", "demographic"),
        CuratedDriver("Population - Europe", 65.0, 0.6, "Europe", "", "reject", "demographic"),
    ]
    return CurationResult(kept=kept, rejected=rejected, min_kept_drivers=8)


def test_coords_for_known_aggregate_and_omitted():
    assert coords_for("Norway") is not None
    assert coords_for("Europe") == coords_for("European Union")  # aggregates share the centroid
    assert coords_for("World") is None  # global aggregate is dropped
    assert coords_for("") is None  # unknown region is dropped


def test_aggregate_sums_kept_importance_per_country():
    countries = {c.region: c for c in aggregate_drivers(_curation())}
    russia = countries["Russian Federation"]
    assert russia.kept_importance == 150.0  # 90 + 60 summed onto one column
    assert russia.kept_count == 2
    assert russia.has_kept is True


def test_world_driver_is_dropped_from_the_globe():
    regions = {c.region for c in aggregate_drivers(_curation())}
    assert "World" not in regions  # the only World driver had no placeable point


def test_europe_aggregate_merges_kept_and_rejected_on_one_point():
    countries = {c.region: c for c in aggregate_drivers(_curation())}
    europe = countries["Europe (aggregate)"]
    assert europe.kept_count == 1  # the FX driver
    assert europe.rejected_count == 1  # the population proxy
    assert europe.has_kept is True  # a kept driver present -> green column, not red


def test_rejected_only_country_is_flagged_red():
    countries = {c.region: c for c in aggregate_drivers(_curation())}
    sri_lanka = countries["Sri Lanka"]
    assert sri_lanka.rejected_only is True
    assert sri_lanka.has_kept is False


def test_columns_are_sorted_tallest_first():
    countries = aggregate_drivers(_curation())
    importances = [c.kept_importance for c in countries]
    assert importances == sorted(importances, reverse=True)
    assert countries[0].region == "Russian Federation"  # the heaviest kept country


def test_country_brief_uses_llm_when_available(monkeypatch):
    monkeypatch.setattr(llm_module, "chat_text", lambda **kwargs: "Norway pipelines matter.")
    text, source = country_brief("Norway", ["Exports of Natural gas - Norway"])
    assert text == "Norway pipelines matter."
    assert source == "llm"


def test_country_brief_falls_back_to_static_on_error(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("no key")

    monkeypatch.setattr(llm_module, "chat_text", boom)
    text, source = country_brief("Norway", [])
    assert text == _FALLBACK_BRIEFS["Norway"]
    assert source == "fallback"


def test_country_brief_generic_fallback_for_unknown_region(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("no key")

    monkeypatch.setattr(llm_module, "chat_text", boom)
    text, source = country_brief("Atlantis", [])
    assert "Atlantis" in text
    assert source == "fallback"


def test_spurious_brief_explains_the_rejection(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("no key")

    monkeypatch.setattr(llm_module, "chat_text", boom)
    text, source = country_brief("Sri Lanka", ["Population - Sri Lanka"], credible=False)
    assert source == "fallback"
    assert "Sri Lanka" in text
    assert "spurious" in text or "coincidence" in text or "no cause" in text


def test_credible_flag_picks_the_right_system_prompt(monkeypatch):
    seen = {}

    def capture(**kwargs):
        seen["system"] = kwargs["system"]
        return "ok"

    monkeypatch.setattr(llm_module, "chat_text", capture)
    country_brief("Norway", [], credible=True)
    assert "influences" in seen["system"]
    country_brief("Sri Lanka", [], credible=False)
    assert "no credible causal link" in seen["system"]
