"""Unit tests for the decision backtest.

These use small synthetic trajectories with hand-chosen spot/median/actual values
so the realized-cost arithmetic and the spot-anchor lookup are checked exactly,
independent of the cached real data.
"""

import pytest

from gas_agent.decision_backtest import run_decision_backtest
from gas_agent.hedge_policy import MonthForecast, decide_month


def quantiles(median: float, low: float, high: float) -> dict:
    """A quantile_forecast block carrying just the three levels the policy reads."""
    return {"0.10": low, "0.50": median, "0.90": high}


def window(forecast_end: str, months: dict) -> dict:
    """months: {date: (median, low, high, actual_or_None)}."""
    series = {
        date: {
            "actual": actual,
            "quantile_forecast": quantiles(median, low, high),
        }
        for date, (median, low, high, actual) in months.items()
    }
    return {"forecast_end": forecast_end, "forecast_series": series}


def ttf(timeseries: dict) -> dict:
    return {"timeseries": timeseries}


def test_spot_anchor_is_the_month_before_the_window():
    trajectories = {"data": [window("2025-03-01", {"2025-02-01": (40.0, 39.0, 41.0, 50.0)})]}
    result = run_decision_backtest(trajectories, ttf({"2025-01-01": 30.0}))
    assert result.rows[0].spot_at_decision == pytest.approx(30.0)


def test_cost_formulas_match_the_policy():
    # Tight band + median far above spot -> the policy locks the max (0.90).
    trajectories = {"data": [window("2025-03-01", {"2025-02-01": (40.0, 39.0, 41.0, 50.0)})]}
    result = run_decision_backtest(trajectories, ttf({"2025-01-01": 30.0}))
    row = result.rows[0]

    expected_ratio = decide_month(MonthForecast("2025-02-01", 40.0, 39.0, 41.0), 30.0).hedge_ratio
    assert row.hedge_ratio == pytest.approx(expected_ratio)
    assert row.cost_always_spot == pytest.approx(50.0)  # always-spot pays the actual
    assert row.cost_always_half == pytest.approx(0.5 * 30.0 + 0.5 * 50.0)  # lock at spot, rest spot
    assert row.cost_policy == pytest.approx(expected_ratio * 30.0 + (1 - expected_ratio) * 50.0)


def test_correct_bullish_call_beats_both_baselines_on_cost():
    # Forecast says "well above spot" and the market then rises: locking early wins.
    trajectories = {"data": [window("2025-03-01", {"2025-02-01": (40.0, 39.0, 41.0, 50.0)})]}
    result = run_decision_backtest(trajectories, ttf({"2025-01-01": 30.0}))
    assert result.cost_saving_vs_spot > 0  # cheaper than buying spot
    assert result.cost_gap_vs_half < 0  # and cheaper than locking a flat 50%


def test_null_actuals_are_skipped():
    trajectories = {
        "data": [
            window(
                "2025-04-01",
                {
                    "2025-02-01": (40.0, 39.0, 41.0, 50.0),
                    "2025-03-01": (41.0, 40.0, 42.0, None),  # not realized yet
                },
            )
        ]
    }
    result = run_decision_backtest(trajectories, ttf({"2025-01-01": 30.0}))
    assert result.n_months == 1
    assert result.rows[0].month == "2025-02-01"


def test_spot_falls_back_to_first_median_when_history_missing():
    # No 2025-01-01 point in the series -> anchor on the first month's median (45).
    trajectories = {"data": [window("2025-03-01", {"2025-02-01": (45.0, 44.0, 46.0, 48.0)})]}
    result = run_decision_backtest(trajectories, ttf({"2019-01-01": 20.0}))
    assert result.rows[0].spot_at_decision == pytest.approx(45.0)


def test_backtest_is_deterministic():
    trajectories = {
        "data": [
            window("2025-04-01", {"2025-02-01": (40.0, 35.0, 45.0, 50.0),
                                  "2025-03-01": (42.0, 36.0, 48.0, 38.0)}),
            window("2025-05-01", {"2025-03-01": (30.0, 25.0, 35.0, 33.0)}),
        ]
    }
    history = ttf({"2025-01-01": 30.0, "2025-02-01": 31.0})
    first = run_decision_backtest(trajectories, history)
    second = run_decision_backtest(trajectories, history)
    assert first.rows == second.rows
    assert first.verdict == second.verdict


def test_empty_trajectories_give_an_empty_result():
    result = run_decision_backtest({"data": []}, ttf({"2025-01-01": 30.0}))
    assert result.n_months == 0
    assert "No backtest windows" in result.verdict
