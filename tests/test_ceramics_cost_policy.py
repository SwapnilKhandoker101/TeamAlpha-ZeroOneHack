"""Offline tests for the cost aggregation + lock-% decision.

The lock % is decided by the *reused* gas hedge engine on a synthetic blended
forecast, so these tests pin the ceramics-specific contract: weights normalise and
map to factors, the weighted band width is what the policy reads, a tight blended
band locks more than a wide one, a rising blended cost tilts the lock up, and the
lock always lands inside the inherited [0.10, 0.90] clamp. The physical per-unit
cost (a separate aggregation) is checked for internal consistency.
"""

import pytest

from gas_agent.hedge_policy import MonthForecast

from ceramics_agent.catalog import get_product
from ceramics_agent.cost_policy import (
    CostWeights,
    component_unit_costs,
    decide_procurement,
    physical_cost_band,
    quarter_band_width,
    quarter_lock_ratio,
    unit_cost_estimate,
)

MONTHS = ["2026-06-01", "2026-07-01", "2026-08-01"]


def _factor(levels: list[float], rel_band: float) -> list[MonthForecast]:
    """A factor whose every month sits at the given level with a symmetric
    relative 80% band ``rel_band`` (so (high-low)/median == rel_band)."""
    out = []
    for month, level in zip(MONTHS, levels):
        half = level * rel_band / 2.0
        out.append(MonthForecast(month, median=level, low=level - half, high=level + half))
    return out


def _flat_factors(rel_band: float, level: float = 100.0) -> dict[str, list[MonthForecast]]:
    return {f: _factor([level, level, level], rel_band) for f in ("gas", "clay", "power", "shipping")}


# --------------------------------------------------------------------------- #
# Weights
# --------------------------------------------------------------------------- #
def test_weights_normalize_to_one():
    norm = CostWeights(2.0, 2.0, 2.0, 2.0).normalized()
    assert sum(norm.as_dict().values()) == pytest.approx(1.0)
    assert norm.gas == pytest.approx(0.25)


def test_all_zero_weights_fall_back_to_equal_split():
    norm = CostWeights(0.0, 0.0, 0.0, 0.0).normalized()
    assert norm.as_dict() == {"gas": 0.25, "clay": 0.25, "energy": 0.25, "transport": 0.25}


def test_as_factor_weights_renames_buckets_to_series():
    fw = CostWeights(gas=0.4, clay=0.3, energy=0.2, transport=0.1).as_factor_weights()
    assert set(fw) == {"gas", "clay", "power", "shipping"}
    assert fw["power"] == pytest.approx(0.2)  # energy -> power
    assert fw["shipping"] == pytest.approx(0.1)  # transport -> shipping


# --------------------------------------------------------------------------- #
# Aggregation 1 — the weighted band width drives the lock %
# --------------------------------------------------------------------------- #
def test_band_width_is_the_weighted_blend_of_factor_bands():
    # gas band 0.10, clay band 0.50; split the weight 50/50 across just those two.
    factors = {
        "gas": _factor([100, 100, 100], 0.10),
        "clay": _factor([100, 100, 100], 0.50),
        "power": _factor([100, 100, 100], 0.10),
        "shipping": _factor([100, 100, 100], 0.10),
    }
    weights = CostWeights(gas=0.5, clay=0.5, energy=0.0, transport=0.0)
    decisions = decide_procurement(factors, weights)
    # Month 0 band width == 0.5*0.10 + 0.5*0.50 == 0.30.
    assert decisions[0].band_width == pytest.approx(0.30)


def test_tight_band_locks_more_than_wide_band():
    tight = decide_procurement(_flat_factors(0.10), CostWeights())
    wide = decide_procurement(_flat_factors(0.80), CostWeights())
    assert quarter_lock_ratio(tight) > quarter_lock_ratio(wide)
    assert tight[0].band_regime == "tight"
    assert wide[0].band_regime == "wide"


def test_lock_ratio_stays_inside_the_clamp_for_any_band():
    for rel in (0.0, 0.10, 0.40, 0.80, 3.0):
        ratio = quarter_lock_ratio(decide_procurement(_flat_factors(rel), CostWeights()))
        assert 0.10 <= ratio <= 0.90


def test_rising_blended_cost_tilts_the_lock_up():
    # Same mid band; one series is flat, the other rises month over month.
    rising = {f: _factor([100, 110, 120], 0.40) for f in ("gas", "clay", "power", "shipping")}
    flat = _flat_factors(0.40, level=100.0)
    rising_decisions = decide_procurement(rising, CostWeights())
    assert quarter_lock_ratio(rising_decisions) >= quarter_lock_ratio(decide_procurement(flat, CostWeights()))
    # The later months read as rising vs the horizon-start anchor.
    assert rising_decisions[-1].direction == "rising"
    assert rising_decisions[-1].drift_pct > 0


def test_quarter_band_width_averages_the_first_three_months():
    decisions = decide_procurement(_flat_factors(0.20), CostWeights())
    assert quarter_band_width(decisions) == pytest.approx(0.20)


def test_empty_forecast_yields_no_decisions():
    assert decide_procurement({}, CostWeights()) == []


# --------------------------------------------------------------------------- #
# Aggregation 2 — the physical per-unit cost
# --------------------------------------------------------------------------- #
def _physical_factors() -> dict[str, list[MonthForecast]]:
    return {
        "gas": _factor([50.0, 50.0, 50.0], 0.20),       # EUR/MWh
        "clay": _factor([0.20, 0.20, 0.20], 0.20),      # EUR/kg
        "power": _factor([0.18, 0.18, 0.18], 0.20),     # EUR/kWh
        "shipping": _factor([0.15, 0.15, 0.15], 0.20),  # EUR/kg
    }


def test_physical_cost_band_is_ordered_and_per_month():
    band = physical_cost_band(get_product("tile"), _physical_factors())
    assert len(band) == len(MONTHS)
    for row in band:
        assert row.q10 < row.q50 < row.q90


def test_component_costs_sum_to_the_unit_cost_estimate():
    product = get_product("bowl")
    factors = _physical_factors()
    components = component_unit_costs(product, factors, month_index=0)
    assert set(components) == {"clay", "power", "gas", "shipping"}
    assert sum(components.values()) == pytest.approx(unit_cost_estimate(product, factors, 0))
