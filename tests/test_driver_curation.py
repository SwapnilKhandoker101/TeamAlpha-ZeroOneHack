"""Unit tests for the deterministic driver curation — the agent's edge.

Curation is what separates this from a forecasting wrapper, so its verdicts must
be predictable: the same driver name always gets the same keep/reject decision,
and the kept-count threshold is what trips the keyword agent's refine loop.
"""

from gas_agent.driver_curation import (
    DEFAULT_CURATION_PARAMS,
    GLOBAL_RISK_THEME,
    CurationParams,
    classify_driver,
    curate_drivers,
    extract_region,
)


def _signals(*names: str) -> dict:
    """Build a minimal external_signals.json-shaped dict from driver names."""
    return {
        "data": {
            f"uid-{i}": {
                "driver_name": name,
                "importance": {"overall": {"mean": 90.0 - i}},
                "pearson_correlation": {"overall": {"mean": 0.1}},
            }
            for i, name in enumerate(names)
        }
    }


def test_population_is_rejected_as_spurious():
    verdict, theme, reason = classify_driver("Population - Sri Lanka")
    assert verdict == "reject"
    assert "demographic" in reason


def test_natural_gas_is_kept():
    verdict, theme, _ = classify_driver("Exports of Natural gas in Europe")
    assert verdict == "keep"
    assert theme == "natural gas"


def test_exchange_rate_is_kept():
    verdict, theme, _ = classify_driver("Exchange Rates - World")
    assert verdict == "keep"
    assert theme == "exchange rates (FX)"


def test_energy_trade_flow_is_kept():
    verdict, theme, _ = classify_driver(
        "EU27 (from 2020) trade – Imports (Russia, Raw materials) in Russian Federation"
    )
    assert verdict == "keep"
    assert theme == "energy trade flow"


def test_unrelated_macro_series_is_rejected():
    verdict, _, reason = classify_driver("Labour market - United States of America")
    assert verdict == "reject"
    assert "whitelist" in reason


def test_extract_region_prefers_longest_match():
    assert extract_region("Energy price benchmark in United States of America") == (
        "United States of America"
    )
    assert extract_region("Exports of Natural gas in Europe") == "Europe"
    assert extract_region("Population - Sri Lanka") == "Sri Lanka"


def test_curation_splits_kept_and_rejected():
    signals = _signals(
        "Exports of Natural gas in Europe",
        "Exchange Rates - World",
        "Population - Belarus",
        "Population - Serbia",
    )
    result = curate_drivers(signals)
    assert result.kept_count == 2
    assert result.rejected_count == 2
    assert {d.name for d in result.rejected} == {"Population - Belarus", "Population - Serbia"}


def test_kept_is_sorted_by_importance_descending():
    signals = _signals(
        "Imports of Natural gas in Germany",  # importance 90
        "Exchange Rates - World",  # importance 89
        "Exports of Liquefied natural gas in Europe",  # importance 88
    )
    result = curate_drivers(signals)
    importances = [d.importance for d in result.kept]
    assert importances == sorted(importances, reverse=True)


def test_needs_refine_trips_below_threshold():
    one_credible = _signals("Exports of Natural gas in Europe")
    result = curate_drivers(one_credible, CurationParams(min_kept_drivers=8))
    assert result.needs_refine is True

    eight_credible = _signals(
        "Exports of Natural gas in Europe",
        "Imports of Natural gas in Germany",
        "Exports of Liquefied natural gas in Europe",
        "Exchange Rates - World",
        "Import prices – Electricity in Europe",
        "Stock levels for oil products in Netherlands",
        "Energy price benchmark in United States of America",
        "Domestic producer prices – Extraction of natural gas in Germany",
    )
    result = curate_drivers(eight_credible, CurationParams(min_kept_drivers=8))
    assert result.needs_refine is False


def test_top_drivers_dedupes_by_name():
    signals = _signals(
        "Exports of Natural gas in Europe",
        "Exports of Natural gas in Europe",
        "Exchange Rates - World",
    )
    result = curate_drivers(signals)
    top = result.top_drivers(3, unique_names=True)
    assert len(top) == 2
    assert len({d.name for d in top}) == 2


# --------------------------------------------------------------------------- #
# Future-proof whitelist keywords (edges.md) — Brent, global-risk/volatility,
# and macro indicators classify as credible the day Sybilion surfaces them.
# --------------------------------------------------------------------------- #
def test_brent_is_kept_as_oil():
    # "Brent" with no other oil cue must still match via the new keyword.
    verdict, theme, _ = classify_driver("Brent front-month settlement")
    assert verdict == "keep"
    assert theme == "oil & petroleum products"


def test_vix_is_kept_as_global_risk():
    verdict, theme, _ = classify_driver("VIX volatility index")
    assert verdict == "keep"
    assert theme == GLOBAL_RISK_THEME


def test_geopolitical_risk_index_is_kept_as_global_risk():
    verdict, theme, _ = classify_driver("Global risk indicator - World")
    assert verdict == "keep"
    assert theme == GLOBAL_RISK_THEME


def test_pmi_is_kept_as_macro_indicator():
    verdict, theme, _ = classify_driver("Eurozone Manufacturing PMI")
    assert verdict == "keep"
    assert theme == "macro indicators"


def test_inflation_is_kept_as_macro_indicator():
    verdict, theme, _ = classify_driver("Inflation rate - euro area")
    assert verdict == "keep"
    assert theme == "macro indicators"


def test_consumer_price_does_not_collide_with_producer_prices():
    # The macro "consumer price" keyword must not be swallowed by, or swallow, the
    # distinct "producer & import prices" theme.
    assert classify_driver("Consumer price index - euro area")[1] == "macro indicators"
    assert classify_driver("Domestic producer prices in Germany")[1] == "producer & import prices"
