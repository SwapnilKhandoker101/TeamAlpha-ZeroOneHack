"""Offline tests for the unified driver-impact map (W4).

``ceramics_agent.impact`` is the pure, Streamlit-free formatter behind the app's
"what moves what" panel: it turns the gas agent's curated drivers and the ceramics
per-factor drivers into one list of rows (driver · factor · importance · which
decision it feeds · the factor's forecast direction). These tests pin that shape and
the headline invariant — **it only describes existing evidence, it computes no
decision number** — using a hand-built gas curation and the committed ceramics mock,
so they run in a clean checkout with no gas cache and no network.
"""

from gas_agent.driver_curation import CuratedDriver, CurationResult
from gas_agent.hedge_policy import MonthForecast

from ceramics_agent import forecast as cforecast
from ceramics_agent.cost_policy import CostWeights
from ceramics_agent.forecast import FACTORS
from ceramics_agent.impact import (
    CERAMICS_DECISION,
    FACTOR_LABELS,
    GAS_DECISION,
    ceramics_impact_rows,
    combined_impact_rows,
    gas_impact_rows,
)
from ceramics_agent.impact import _trend_direction


def _series(levels: list[float]) -> list[MonthForecast]:
    """A factor's months at the given medians (band is irrelevant to the trend read)."""
    return [
        MonthForecast(f"2026-{m + 1:02d}-01", median=v, low=v * 0.9, high=v * 1.1)
        for m, v in enumerate(levels)
    ]


def _kept(name: str, importance: float) -> CuratedDriver:
    return CuratedDriver(
        name=name, importance=importance, correlation=0.0, region="", theme="energy",
        verdict="keep", reason="credible",
    )


def _curation(kept: list[CuratedDriver]) -> CurationResult:
    return CurationResult(kept=kept, rejected=[], min_kept_drivers=8)


# --------------------------------------------------------------------------- #
# The trend read — a pure first→last comparison, never a decision
# --------------------------------------------------------------------------- #
def test_trend_direction_classifies_rising_easing_flat():
    assert _trend_direction(_series([100, 130])).startswith("↑")
    assert _trend_direction(_series([100, 70])).startswith("↓")
    assert _trend_direction(_series([100, 100.5])).startswith("→")  # within the flat band


def test_trend_direction_is_safe_on_degenerate_input():
    assert _trend_direction([]) == "→ flat"
    assert _trend_direction(_series([100])) == "→ flat"  # single month
    assert _trend_direction(_series([0.0, 5.0])) == "→ flat"  # zero base guard


# --------------------------------------------------------------------------- #
# Gas rows — one per KEPT driver, all feeding the gas hedge
# --------------------------------------------------------------------------- #
def test_gas_rows_map_every_kept_driver_to_the_hedge():
    curation = _curation([_kept("Exports of Natural gas in Europe", 100.0),
                          _kept("EU gas storage fill level", 73.6)])
    rows = gas_impact_rows(curation, _series([100, 60]))  # easing gas band
    assert len(rows) == 2
    assert all(r.agent == "gas" for r in rows)
    assert all(r.explains == "TTF gas price" for r in rows)
    assert all(GAS_DECISION in r.feeds for r in rows)
    assert all(r.direction.startswith("↓") for r in rows)  # the band eases
    assert rows[1].importance == 74  # rounded to a whole importance


# --------------------------------------------------------------------------- #
# Ceramics rows — one per factor driver, weighted into the lock %
# --------------------------------------------------------------------------- #
def test_ceramics_rows_from_the_committed_mock_are_well_formed():
    factors, _ = cforecast.load_ceramics_forecast(None)
    drivers_by_factor = cforecast.load_factor_drivers(None)
    weights = CostWeights(gas=0.40, clay=0.35, energy=0.15, transport=0.10)
    rows = ceramics_impact_rows(drivers_by_factor, factors, weights)

    # Every mock factor driver becomes one row, attributed to the lock decision.
    expected = sum(len(drivers_by_factor[f]) for f in FACTORS)
    assert len(rows) == expected and expected > 0
    assert all(r.agent == "ceramics" for r in rows)
    assert all(CERAMICS_DECISION in r.feeds for r in rows)
    assert {r.explains for r in rows} <= set(FACTOR_LABELS.values())
    assert all(0.0 <= r.importance <= 100.0 for r in rows)
    # The feeds string carries the factor's normalized weight as a percentage.
    assert any("40%" in r.feeds for r in rows)  # gas weight, normalized from the inputs


def test_ceramics_weight_in_the_feeds_string_follows_the_weights():
    factors, _ = cforecast.load_ceramics_forecast(None)
    drivers_by_factor = cforecast.load_factor_drivers(None)
    clay_heavy = CostWeights(gas=0.05, clay=0.90, energy=0.03, transport=0.02)
    rows = ceramics_impact_rows(drivers_by_factor, factors, clay_heavy)
    clay_rows = [r for r in rows if r.explains == FACTOR_LABELS["clay"]]
    assert clay_rows and all("90%" in r.feeds for r in clay_rows)


# --------------------------------------------------------------------------- #
# Combined — gas first, then ceramics; gas capped, ceramics complete
# --------------------------------------------------------------------------- #
def test_combined_orders_gas_first_then_ceramics_and_caps_gas():
    gas = gas_impact_rows(
        _curation([_kept(f"gas driver {i}", float(i)) for i in range(20)]),
        _series([100, 90]),
    )
    factors, _ = cforecast.load_ceramics_forecast(None)
    ceramics = ceramics_impact_rows(
        cforecast.load_factor_drivers(None), factors, CostWeights())
    combined = combined_impact_rows(gas, ceramics, gas_limit=10)

    gas_part = [r for r in combined if r.agent == "gas"]
    cer_part = [r for r in combined if r.agent == "ceramics"]
    assert len(gas_part) == 10  # capped
    assert len(cer_part) == len(ceramics)  # every ceramics driver kept
    # Gas rows come first, and they are the highest-importance gas drivers, descending.
    assert combined[: len(gas_part)] == gas_part
    assert [r.importance for r in gas_part] == sorted(
        (r.importance for r in gas_part), reverse=True)
    assert gas_part[0].importance == 19  # the strongest of the 20


def test_combined_no_gas_limit_keeps_all_rows():
    gas = gas_impact_rows(_curation([_kept("g", 50.0)]), _series([1, 1]))
    combined = combined_impact_rows(gas, [], gas_limit=None)
    assert len(combined) == 1
