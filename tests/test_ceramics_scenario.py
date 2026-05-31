"""Offline tests for the ceramics shock — one headline, both agents (W6).

No network, no live model. The headline classifier is the gas agent's, reused
verbatim and exercised here on its deterministic keyword path (``use_llm=False``).
THE RULE holds: routing and the re-decide are pure arithmetic — the same shock
always reproduces the same lock %, and the calm path stays byte-identical.

What's pinned here:
  * factor routing maps a headline to the cost factor(s) it hits (gas-only for a
    geopolitical/gas headline, so the gas agent's behaviour is unchanged),
  * the re-decide raises the lock and only moves the affected factor's band,
  * the additive ``risk_premium`` / ``anchor`` on ``decide_procurement`` are
    behaviour-preserving at their defaults,
  * identical inputs → identical numbers.
"""

import pytest

from gas_agent.hedge_policy import MonthForecast

from ceramics_agent.cost_policy import CostWeights, decide_procurement, quarter_lock_ratio
from ceramics_agent.forecast import FACTORS, load_ceramics_forecast
from ceramics_agent.scenario import (
    CERAMICS_SHOCK_PARAMS,
    affected_factors_for,
    ceramics_shock_from_message,
    route_factors,
    run_ceramics_shock,
)

MONTHS = ["2026-06-01", "2026-07-01", "2026-08-01"]


def _factor(level: float, rel_band: float) -> list[MonthForecast]:
    half = level * rel_band / 2.0
    return [MonthForecast(m, median=level, low=level - half, high=level + half) for m in MONTHS]


def _flat_factors(rel_band: float = 0.40, level: float = 100.0) -> dict[str, list[MonthForecast]]:
    return {f: _factor(level, rel_band) for f in FACTORS}


# --------------------------------------------------------------------------- #
# Factor routing — which factor(s) a headline hits
# --------------------------------------------------------------------------- #
def test_geopolitical_gas_headline_routes_to_gas_only():
    # The canonical demo shock — preserves the gas agent's behaviour exactly.
    assert route_factors("Iran closes the Strait of Hormuz") == ("gas",)
    assert route_factors("EU sanctions cut Russian pipeline gas") == ("gas",)


def test_logistics_headline_routes_to_shipping():
    assert route_factors("Houthi attacks shut the Red Sea to container ships") == ("shipping",)
    assert route_factors("a port strike halts freight") == ("shipping",)


def test_power_and_clay_headlines_route_to_their_factor():
    assert route_factors("national grid blackout hits industry") == ("power",)
    assert route_factors("a kaolin mine closure squeezes clay supply") == ("clay",)


def test_multi_factor_headline_returns_all_hits_in_canonical_order():
    # A Red Sea crisis lifts freight and, through fear, gas.
    assert route_factors("Red Sea crisis spikes freight and gas prices") == ("gas", "shipping")


def test_unrouted_headline_is_empty_but_defaults_to_gas():
    assert route_factors("a general supply shock looms") == ()
    assert affected_factors_for("a general supply shock looms") == ("gas",)
    assert affected_factors_for("a general supply shock looms", default_to_gas=False) == ()


# --------------------------------------------------------------------------- #
# The deterministic re-decide
# --------------------------------------------------------------------------- #
def test_shock_raises_the_lock_vs_calm():
    factors, weights = _flat_factors(), CostWeights()
    calm = quarter_lock_ratio(decide_procurement(factors, weights))
    outcome = run_ceramics_shock(factors, weights, magnitude=1.0, affected_factors=["gas"])
    assert outcome.lock_ratio > calm
    assert outcome.affected_factors == ("gas",)
    assert outcome.risk_premium == pytest.approx(CERAMICS_SHOCK_PARAMS.risk_premium)


def test_higher_magnitude_locks_at_least_as_much():
    factors, weights = _flat_factors(), CostWeights()
    low = run_ceramics_shock(factors, weights, 0.35, ["gas"]).lock_ratio
    high = run_ceramics_shock(factors, weights, 1.0, ["gas"]).lock_ratio
    assert high >= low


def test_zero_magnitude_is_a_noop():
    factors, weights = _flat_factors(), CostWeights()
    calm = decide_procurement(factors, weights)
    outcome = run_ceramics_shock(factors, weights, magnitude=0.0, affected_factors=["gas"])
    assert outcome.decisions == calm
    assert outcome.risk_premium == 0.0


def test_empty_affected_set_is_a_noop():
    factors, weights = _flat_factors(), CostWeights()
    calm = decide_procurement(factors, weights)
    outcome = run_ceramics_shock(factors, weights, magnitude=1.0, affected_factors=[])
    assert outcome.decisions == calm
    assert outcome.affected_factors == ()
    assert outcome.risk_premium == 0.0


def test_only_the_affected_band_moves():
    factors, weights = _flat_factors(), CostWeights()
    outcome = run_ceramics_shock(factors, weights, 1.0, ["shipping"])
    # Untouched factors pass through byte-identical; shipping's band is re-cast.
    for factor in ("gas", "clay", "power"):
        assert outcome.factors[factor] == factors[factor]
    assert outcome.factors["shipping"] != factors["shipping"]
    assert outcome.factors["shipping"][0].median > factors["shipping"][0].median


def test_affected_factors_are_normalized_and_ordered():
    factors, weights = _flat_factors(), CostWeights()
    outcome = run_ceramics_shock(factors, weights, 1.0, ["shipping", "gas", "gas", "bogus"])
    assert outcome.affected_factors == ("gas", "shipping")  # deduped, canonical order, unknown dropped


def test_base_premium_composes_and_is_capped():
    factors, weights = _flat_factors(), CostWeights()
    # A standing base premium plus the shock marginal still re-decides without error,
    # and the shock's OWN marginal is what the outcome reports for the panel.
    outcome = run_ceramics_shock(factors, weights, 1.0, ["gas"], base_risk_premium=0.08)
    assert outcome.risk_premium == pytest.approx(CERAMICS_SHOCK_PARAMS.risk_premium)
    assert 0.10 <= outcome.lock_ratio <= 0.90


def test_shock_is_deterministic():
    factors, weights = _flat_factors(), CostWeights()
    a = run_ceramics_shock(factors, weights, 0.6, ["gas", "shipping"])
    b = run_ceramics_shock(factors, weights, 0.6, ["gas", "shipping"])
    assert a.decisions == b.decisions
    assert a.lock_ratio == b.lock_ratio


def test_shock_runs_on_the_committed_mock_and_raises_the_lock():
    factors, _ = load_ceramics_forecast()  # the offline 4-factor mock
    weights = CostWeights()
    calm = quarter_lock_ratio(decide_procurement(factors, weights))
    outcome = run_ceramics_shock(factors, weights, 1.0, ["gas"])
    assert outcome.lock_ratio >= calm
    assert set(outcome.factors) == set(FACTORS)


# --------------------------------------------------------------------------- #
# decide_procurement additive params — behaviour-preserving at defaults
# --------------------------------------------------------------------------- #
def test_risk_premium_default_is_byte_identical():
    factors, weights = _flat_factors(), CostWeights()
    assert decide_procurement(factors, weights) == decide_procurement(factors, weights, risk_premium=0.0)


def test_risk_premium_lifts_every_unclamped_month():
    factors, weights = _flat_factors(0.40), CostWeights()  # mid band, not at a clamp
    base = decide_procurement(factors, weights)
    lifted = decide_procurement(factors, weights, risk_premium=0.20)
    for b, l in zip(base, lifted):
        assert l.hedge_ratio > b.hedge_ratio


def test_anchor_default_is_byte_identical():
    factors, weights = _flat_factors(), CostWeights()
    assert decide_procurement(factors, weights) == decide_procurement(factors, weights, anchor=None)


def test_lower_anchor_reads_as_rising_drift():
    factors, weights = _flat_factors(level=100.0), CostWeights()
    # Anchoring below the blended level makes every month read as rising vs "today".
    decisions = decide_procurement(factors, weights, anchor=80.0)
    assert all(d.direction == "rising" and d.drift_pct > 0 for d in decisions)


# --------------------------------------------------------------------------- #
# Chat entry point — classify (gas classifier, offline) → route → re-decide
# --------------------------------------------------------------------------- #
def test_chat_entry_returns_none_for_a_non_shock():
    factors, weights = _flat_factors(), CostWeights()
    assert ceramics_shock_from_message("how is the weather today?", factors, weights) is None


def test_chat_entry_routes_and_redecides_a_real_shock():
    factors, weights = _flat_factors(), CostWeights()
    calm = quarter_lock_ratio(decide_procurement(factors, weights))
    outcome = ceramics_shock_from_message("Iran closes the Strait of Hormuz", factors, weights)
    assert outcome is not None
    assert outcome.affected_factors == ("gas",)
    assert outcome.lock_ratio > calm
