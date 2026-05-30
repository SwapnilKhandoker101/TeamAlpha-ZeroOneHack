"""Offline tests for the committed ceramics business catalog.

Pure data-shape checks — no network, no model. They pin the invariants the rest
of the package relies on: every product has a positive bill of materials, every
supplier prices every cost component, every channel carries a full-year
seasonality curve, and each historical month's channel mix sums to 1.0.
"""

import pytest

from ceramics_agent.catalog import (
    CHANNELS,
    COST_COMPONENTS,
    HISTORICAL_SALES,
    PRODUCTS,
    SUPPLIERS,
    get_product,
    list_channels,
    list_products,
    list_suppliers,
    quarter_of,
)


# --------------------------------------------------------------------------- #
# Products / bill of materials
# --------------------------------------------------------------------------- #
def test_every_product_has_a_positive_bill_of_materials():
    assert PRODUCTS  # not empty
    for product in PRODUCTS.values():
        assert product.clay_kg > 0
        assert product.kiln_kwh > 0
        assert product.firing_gas_kwh > 0
        assert product.ship_kg > 0
        assert product.glaze_kg >= 0  # carried for specialty match; may be small


def test_get_product_round_trips_each_id():
    for pid, product in PRODUCTS.items():
        assert get_product(pid) is product
    assert {p.id for p in list_products()} == set(PRODUCTS)


# --------------------------------------------------------------------------- #
# Suppliers
# --------------------------------------------------------------------------- #
def test_supplier_factor_defaults_to_one_for_unspecified_component():
    alpine = SUPPLIERS["alpine"]
    # Alpine specifies shipping (0.95) but every component resolves to a number.
    for component in COST_COMPONENTS:
        assert alpine.factor(component) > 0
    # A component the supplier never lists falls back to the 1.0 baseline.
    assert alpine.factor("not-a-real-component") == 1.0


def test_average_price_factor_spans_all_components():
    for supplier in list_suppliers():
        expected = sum(supplier.factor(c) for c in COST_COMPONENTS) / len(COST_COMPONENTS)
        assert supplier.average_price_factor() == pytest.approx(expected)


def test_eastern_is_cheapest_and_premium_is_dearest():
    # A documented invariant the backtest's "cheapest supplier" heuristic leans on.
    cheapest = min(list_suppliers(), key=lambda s: s.average_price_factor())
    dearest = max(list_suppliers(), key=lambda s: s.average_price_factor())
    assert cheapest.id == "eastern"
    assert dearest.id == "premium"


# --------------------------------------------------------------------------- #
# Channels
# --------------------------------------------------------------------------- #
def test_every_channel_has_a_full_year_seasonality_curve():
    for channel in list_channels():
        for quarter in ("Q1", "Q2", "Q3", "Q4"):
            assert channel.season_factor(quarter) > 0
        assert 0.0 < channel.target_margin <= 1.0
        assert channel.min_order >= 1


def test_season_factor_defaults_to_one_for_unknown_quarter():
    assert CHANNELS["wholesale"].season_factor("Q9") == 1.0


# --------------------------------------------------------------------------- #
# Historical sales (what the backtest replays)
# --------------------------------------------------------------------------- #
def test_historical_channel_mix_sums_to_one_and_units_positive():
    assert len(HISTORICAL_SALES) == 12
    for record in HISTORICAL_SALES:
        assert record.product_id in PRODUCTS
        assert record.units > 0
        assert sum(record.channel_mix.values()) == pytest.approx(1.0)
        assert all(channel_id in CHANNELS for channel_id in record.channel_mix)
        assert 0.0 <= record.achieved_margin <= 1.0


# --------------------------------------------------------------------------- #
# quarter_of
# --------------------------------------------------------------------------- #
def test_quarter_of_maps_months_to_quarters():
    assert quarter_of("2026-01") == "Q1"
    assert quarter_of("2026-03-01") == "Q1"
    assert quarter_of("2026-04") == "Q2"
    assert quarter_of("2026-09-01") == "Q3"
    assert quarter_of("2026-11-01") == "Q4"
