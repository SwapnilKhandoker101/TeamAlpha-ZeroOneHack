"""Unit tests for the deterministic hedge policy.

The policy is the one part of the agent that must be perfectly predictable. The
tests are written against the *parameters* (not hard-coded magic numbers) so they
keep passing when the band thresholds are recalibrated: we construct forecasts
with a known relative band width and assert the documented behavior.
"""

import pytest

from gas_agent.hedge_policy import (
    DEFAULT_PARAMS,
    MonthForecast,
    clamp,
    decide_all,
    decide_month,
    quarter_hedge_ratio,
)

P = DEFAULT_PARAMS


def forecast_with_band(median: float, band_width: float) -> MonthForecast:
    """A forecast month whose (high-low)/median equals ``band_width`` exactly,
    centered on the median."""
    half = band_width * median / 2
    return MonthForecast("m", median=median, low=median - half, high=median + half)


def test_clamp_bounds():
    assert clamp(1.5, 0.0, 1.0) == 1.0
    assert clamp(-0.5, 0.0, 1.0) == 0.0
    assert clamp(0.3, 0.0, 1.0) == 0.3


def test_band_at_low_threshold_saturates_lock():
    # band width == low_band -> band component is 1.0 (lock the most)
    forecast = forecast_with_band(50.0, P.low_band)
    decision = decide_month(forecast, spot_price=50.0)  # flat -> no tilt
    assert decision.ratio_from_band == pytest.approx(1.0)
    assert decision.band_regime == "tight"


def test_band_at_high_threshold_zeroes_lock():
    # band width == high_band -> band component is 0.0 (keep optionality)
    forecast = forecast_with_band(50.0, P.high_band)
    decision = decide_month(forecast, spot_price=50.0)
    assert decision.ratio_from_band == pytest.approx(0.0)
    assert decision.band_regime == "wide"


def test_band_at_midpoint_is_half():
    midpoint = (P.low_band + P.high_band) / 2
    forecast = forecast_with_band(50.0, midpoint)
    decision = decide_month(forecast, spot_price=50.0)
    assert decision.ratio_from_band == pytest.approx(0.5)
    assert decision.band_regime == "moderate"


def test_tighter_band_locks_more_than_wider_band():
    tight = decide_month(forecast_with_band(50.0, P.low_band + 0.1), spot_price=50.0)
    wide = decide_month(forecast_with_band(50.0, P.high_band - 0.1), spot_price=50.0)
    assert tight.hedge_ratio > wide.hedge_ratio


def test_forward_above_spot_tilts_up():
    forecast = forecast_with_band(60.0, P.low_band)  # median 60 above spot 50
    decision = decide_month(forecast, spot_price=50.0)
    assert decision.direction == "rising"
    assert decision.direction_tilt > 0


def test_forward_below_spot_tilts_down():
    forecast = forecast_with_band(40.0, P.low_band)  # median 40 below spot 50
    decision = decide_month(forecast, spot_price=50.0)
    assert decision.direction == "falling"
    assert decision.direction_tilt < 0


def test_tilt_is_capped():
    # A huge gap vs spot must not push the tilt past max_tilt.
    forecast = forecast_with_band(500.0, P.low_band)
    decision = decide_month(forecast, spot_price=50.0)
    assert decision.direction_tilt == pytest.approx(P.max_tilt)


def test_shock_like_jump_raises_the_hedge():
    # Same (wide) band, but the forward jumps well above spot: hedge must rise,
    # demonstrating the geopolitical-shock response is driven by the drift tilt.
    band = (P.low_band + P.high_band) / 2
    calm = decide_month(forecast_with_band(45.0, band), spot_price=47.0)  # below spot
    shock = decide_month(forecast_with_band(70.0, band), spot_price=47.0)  # spike above spot
    assert shock.hedge_ratio > calm.hedge_ratio


def test_decide_all_compares_every_month_to_same_spot():
    forecasts = [forecast_with_band(60.0, P.low_band), forecast_with_band(40.0, P.low_band)]
    decisions = decide_all(forecasts, spot_price=50.0)
    assert decisions[0].drift_pct == pytest.approx((60.0 - 50.0) / 50.0)
    assert decisions[1].drift_pct == pytest.approx((40.0 - 50.0) / 50.0)


def test_quarter_hedge_ratio_is_the_average():
    forecasts = [forecast_with_band(60.0, P.low_band), forecast_with_band(40.0, P.high_band)]
    decisions = decide_all(forecasts, spot_price=50.0)
    expected = (decisions[0].hedge_ratio + decisions[1].hedge_ratio) / 2
    assert quarter_hedge_ratio(decisions) == pytest.approx(expected)


def test_hedge_ratio_always_within_bounds():
    for band_width in (0.01, 0.5, 3.0):
        for spot in (10.0, 50.0, 500.0):
            decision = decide_month(forecast_with_band(50.0, band_width), spot_price=spot)
            assert P.min_hedge <= decision.hedge_ratio <= P.max_hedge
