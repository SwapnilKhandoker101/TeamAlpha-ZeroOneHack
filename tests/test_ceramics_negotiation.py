"""Offline tests for the two-round buy/sell negotiation.

Every quote is pure arithmetic on the committed catalog and the physical cost
band, so the tests recompute the expected opening positions from the same
component costs and check the documented round-2 moves: +5% supplier surcharge on
hot demand, +10% on a rush order, and the competition-driven channel discount
(-7% / -3% / 0). The market-anchored sell side means a cheaper supplier widens the
margin, and identical inputs reproduce identical rows.
"""

import pytest

from gas_agent.hedge_policy import MonthForecast

from ceramics_agent.catalog import CHANNELS, SUPPLIERS, get_product
from ceramics_agent.cost_policy import component_unit_costs
from ceramics_agent.negotiation import (
    COMPETITION_DISCOUNT,
    DEMAND_PREMIUM,
    RUSH_PREMIUM,
    SUPPLIER_BASE_MARGIN,
    negotiate,
)

MONTH = "2026-06-01"


def _factors() -> dict[str, list[MonthForecast]]:
    def one(med: float) -> list[MonthForecast]:
        half = med * 0.20 / 2.0
        return [MonthForecast(MONTH, median=med, low=med - half, high=med + half)]

    return {"gas": one(50.0), "clay": one(0.20), "power": one(0.18), "shipping": one(0.15)}


def _expected_opens(product_id: str, supplier_id: str, channel_id: str):
    product = get_product(product_id)
    supplier = SUPPLIERS[supplier_id]
    channel = CHANNELS[channel_id]
    comp = component_unit_costs(product, _factors(), 0)
    supplier_materials = sum(comp[c] * supplier.factor(c) for c in comp)
    market_materials = sum(comp.values())
    supplier_open = supplier_materials * (1.0 + SUPPLIER_BASE_MARGIN)
    channel_open = market_materials * (1.0 + SUPPLIER_BASE_MARGIN) * (1.0 + channel.target_margin)
    return supplier_open, channel_open


# --------------------------------------------------------------------------- #
# Round 1 — opening positions
# --------------------------------------------------------------------------- #
def test_round1_openings_match_the_cost_arithmetic():
    supplier_open, channel_open = _expected_opens("bowl", "alpine", "wholesale")
    # Q2 target month + a comfortable timeline => no round-2 surcharge.
    result = negotiate(
        get_product("bowl"), SUPPLIERS["alpine"], CHANNELS["wholesale"],
        _factors(), target_month="2026-05-01", quantity=1000, timeline_days=14, competition="low",
    )
    assert result.rows[0].value == pytest.approx(supplier_open)  # R1 supplier
    assert result.rows[1].value == pytest.approx(channel_open)   # R1 channel
    assert len(result.rows) == 5


def test_sell_side_is_supplier_independent_so_cheaper_supplier_widens_margin():
    kwargs = dict(
        channel=CHANNELS["wholesale"], factors=_factors(), target_month="2026-05-01",
        quantity=1000, timeline_days=14, competition="low",
    )
    cheap = negotiate(get_product("bowl"), SUPPLIERS["eastern"], **kwargs)  # cheapest
    dear = negotiate(get_product("bowl"), SUPPLIERS["premium"], **kwargs)   # dearest
    # Same sell price; the cheaper buy side leaves a bigger margin.
    assert cheap.sell_price == pytest.approx(dear.sell_price)
    assert cheap.buy_price < dear.buy_price
    assert cheap.unit_margin > dear.unit_margin


# --------------------------------------------------------------------------- #
# Round 2 — supplier surcharges
# --------------------------------------------------------------------------- #
def test_hot_demand_adds_the_demand_premium():
    supplier_open, _ = _expected_opens("bowl", "alpine", "online")
    # Online's Q4 seasonality is 1.6 (>= 1.2) -> hot demand; timeline 14 -> not a rush.
    result = negotiate(
        get_product("bowl"), SUPPLIERS["alpine"], CHANNELS["online"],
        _factors(), target_month="2026-10-01", quantity=1000, timeline_days=14, competition="low",
    )
    assert result.buy_price == pytest.approx(supplier_open * (1.0 + DEMAND_PREMIUM))


def test_rush_order_adds_the_rush_premium():
    supplier_open, _ = _expected_opens("bowl", "alpine", "wholesale")
    # Q2 wholesale (season 1.0, not hot) + a 5-day rush (< 7) -> rush premium only.
    result = negotiate(
        get_product("bowl"), SUPPLIERS["alpine"], CHANNELS["wholesale"],
        _factors(), target_month="2026-05-01", quantity=1000, timeline_days=5, competition="low",
    )
    assert result.buy_price == pytest.approx(supplier_open * (1.0 + RUSH_PREMIUM))


def test_calm_normal_order_holds_the_opening_quote():
    supplier_open, _ = _expected_opens("bowl", "alpine", "wholesale")
    result = negotiate(
        get_product("bowl"), SUPPLIERS["alpine"], CHANNELS["wholesale"],
        _factors(), target_month="2026-05-01", quantity=1000, timeline_days=14, competition="low",
    )
    assert result.buy_price == pytest.approx(supplier_open)


# --------------------------------------------------------------------------- #
# Round 2 — channel competition discount
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("competition", ["high", "medium", "low"])
def test_competition_discounts_the_sell_price(competition):
    _, channel_open = _expected_opens("bowl", "alpine", "wholesale")
    result = negotiate(
        get_product("bowl"), SUPPLIERS["alpine"], CHANNELS["wholesale"],
        _factors(), target_month="2026-05-01", quantity=1000, timeline_days=14, competition=competition,
    )
    expected = channel_open * (1.0 - COMPETITION_DISCOUNT[competition])
    assert result.sell_price == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Settlement + determinism
# --------------------------------------------------------------------------- #
def test_unit_margin_and_total_are_consistent():
    qty = 2500
    result = negotiate(
        get_product("tile"), SUPPLIERS["alpine"], CHANNELS["wholesale"],
        _factors(), target_month="2026-05-01", quantity=qty, timeline_days=14, competition="medium",
    )
    assert result.unit_margin == pytest.approx(result.sell_price - result.buy_price)
    assert result.total_margin == pytest.approx(result.unit_margin * qty)


def test_identical_inputs_reproduce_identical_rows():
    args = (
        get_product("tile"), SUPPLIERS["eastern"], CHANNELS["online"],
        _factors(), "2026-10-01", 5000, 6, "high",
    )
    first = negotiate(*args)
    second = negotiate(*args)
    assert [(r.round, r.party, r.kind, r.value, r.reason) for r in first.rows] == \
           [(r.round, r.party, r.kind, r.value, r.reason) for r in second.rows]
    assert (first.buy_price, first.sell_price) == (second.buy_price, second.sell_price)
