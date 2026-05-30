"""Offline end-to-end tests for the recommendation orchestrator.

``build_recommendation`` runs the whole deterministic pipeline (forecast -> lock %
-> curation -> negotiation -> backtest) with no LLM and, on the default path, no
network. These tests pin the end-to-end shape and — the headline invariant — that
identical inputs produce byte-identical core numbers.
"""

from ceramics_agent.cost_policy import default_weights
from ceramics_agent.recommend import available_months, build_recommendation


def _core(rec) -> tuple:
    """The decision fingerprint used for the determinism check."""
    return (
        round(rec.lock_ratio, 6),
        round(rec.band_width, 6),
        round(rec.unit_cost, 6),
        rec.chosen_supplier.name if rec.chosen_supplier else None,
        rec.chosen_channel.name if rec.chosen_channel else None,
        round(rec.negotiation.buy_price, 6),
        round(rec.negotiation.sell_price, 6),
        round(rec.negotiation.unit_margin, 6),
        round(rec.backtest.agent.mean, 6),
    )


def test_build_recommendation_produces_a_complete_decision():
    rec = build_recommendation("tile", 5000, 14, default_weights(), "medium")
    assert rec.source == "mock"
    assert 0.10 <= rec.lock_ratio <= 0.90
    assert rec.chosen_supplier is not None
    assert rec.chosen_channel is not None
    assert rec.unit_cost > 0
    assert len(rec.cost_band) == 6
    assert rec.backtest.n_months == 12
    # The negotiated total is internally consistent.
    neg = rec.negotiation
    assert neg.total_margin == neg.unit_margin * rec.quantity


def test_target_month_defaults_to_the_nearest_forecast_month():
    months = available_months()
    rec = build_recommendation("tile", 5000, 14, default_weights(), "medium")
    assert rec.target_month == months[0]


def test_invalid_target_month_falls_back_to_the_nearest():
    months = available_months()
    rec = build_recommendation(
        "tile", 5000, 14, default_weights(), "medium", target_month="1999-01-01",
    )
    assert rec.target_month == months[0]


def test_same_inputs_yield_identical_core_numbers():
    a = build_recommendation("tile", 5000, 14, default_weights(), "medium")
    b = build_recommendation("tile", 5000, 14, default_weights(), "medium")
    assert _core(a) == _core(b)


def test_different_weights_can_move_the_decision():
    from ceramics_agent.cost_policy import CostWeights

    gas_heavy = build_recommendation(
        "tile", 5000, 14, CostWeights(gas=0.9, clay=0.05, energy=0.03, transport=0.02), "medium",
    )
    clay_heavy = build_recommendation(
        "tile", 5000, 14, CostWeights(gas=0.05, clay=0.9, energy=0.03, transport=0.02), "medium",
    )
    # The blended band differs, so at least the volatility readout moves.
    assert gas_heavy.band_width != clay_heavy.band_width


def test_recommendation_labels_are_well_formed():
    rec = build_recommendation("bowl", 2000, 10, default_weights(), "low")
    assert rec.band_regime in {"tight", "moderate", "wide"}
    assert "lock" in rec.lock_label.lower()
