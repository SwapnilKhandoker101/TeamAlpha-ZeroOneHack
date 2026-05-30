"""Unit tests for the adaptive shock engine.

All deterministic — no LLM is exercised here (``parse_shock_request`` is tested on
its keyword path). The arithmetic is checked against the same hedge policy the
calm path uses, so a shock provably *raises* the hedge ratio rather than just
relabelling it.
"""

import pytest

from gas_agent.driver_curation import CuratedDriver, CurationResult
from gas_agent.hedge_policy import MonthForecast, decide_all, quarter_hedge_ratio
from gas_agent.scenario import (
    DEFAULT_SHOCK_PARAMS,
    ShockParams,
    apply_shock,
    parse_shock_request,
    run_shock,
    shock_forecast_json,
    shock_risk_premium,
    shocked_curation,
)


# Two mid-band months whose baseline ratios sit clear of the [0.10, 0.90] clamps,
# so a shock has visible room to move them.
MONTHS = [
    MonthForecast("2026-06-01", median=40.0, low=28.0, high=60.0),
    MonthForecast("2026-07-01", median=38.0, low=30.0, high=50.0),
]
SPOT = 40.0


def _curation() -> CurationResult:
    kept = [
        CuratedDriver("Exports of Natural gas in Europe", 80.0, 0.5, "Europe", "natural gas", "keep", "x"),
        CuratedDriver("Imports of Natural gas - Iran", 30.0, 0.3, "Iran", "energy trade flow", "keep", "x"),
        CuratedDriver("Electricity prices - Germany", 50.0, 0.4, "Germany", "electricity & power", "keep", "x"),
    ]
    rejected = [
        CuratedDriver("Population - Sri Lanka", 70.0, 0.6, "Sri Lanka", "", "reject", "demographic proxy"),
    ]
    return CurationResult(kept=kept, rejected=rejected, min_kept_drivers=8)


# --------------------------------------------------------------------------- #
# apply_shock + the hedge policy
# --------------------------------------------------------------------------- #
def test_shock_raises_the_quarter_hedge_ratio():
    baseline = quarter_hedge_ratio(decide_all(MONTHS, SPOT))
    shocked_months = apply_shock(MONTHS, magnitude=1.0)
    premium = shock_risk_premium(1.0)
    shocked = quarter_hedge_ratio(decide_all(shocked_months, SPOT, risk_premium=premium))
    assert shocked > baseline


def test_bigger_shock_locks_at_least_as_much_as_a_smaller_one():
    spot = SPOT

    def ratio_at(mag: float) -> float:
        months = apply_shock(MONTHS, mag)
        return quarter_hedge_ratio(decide_all(months, spot, risk_premium=shock_risk_premium(mag)))

    assert ratio_at(1.0) >= ratio_at(0.5) >= ratio_at(0.0)


def test_apply_shock_lifts_median_and_widens_band():
    shocked = apply_shock(MONTHS, magnitude=1.0)
    for before, after in zip(MONTHS, shocked):
        assert after.median > before.median  # forward lifts above the calm median
        before_width = before.high - before.low
        after_width = after.high - after.low
        assert after_width > before_width  # the 80% band gets wider


def test_zero_magnitude_is_a_no_op():
    assert apply_shock(MONTHS, magnitude=0.0) == MONTHS
    assert shocked_curation(_curation(), magnitude=0.0) == _curation()
    assert shock_risk_premium(0.0) == 0.0


def test_risk_premium_scales_linearly_and_clamps():
    assert shock_risk_premium(1.0) == pytest.approx(DEFAULT_SHOCK_PARAMS.risk_premium)
    assert shock_risk_premium(0.5) == pytest.approx(DEFAULT_SHOCK_PARAMS.risk_premium * 0.5)
    assert shock_risk_premium(5.0) == pytest.approx(DEFAULT_SHOCK_PARAMS.risk_premium)  # clamped to 1.0


def test_magnitude_is_clamped_into_range():
    # An out-of-range magnitude must behave like the in-range extreme, not blow up.
    assert apply_shock(MONTHS, magnitude=9.0) == apply_shock(MONTHS, magnitude=1.0)


# --------------------------------------------------------------------------- #
# shock_forecast_json (chart parity)
# --------------------------------------------------------------------------- #
def _forecast_json() -> dict:
    return {
        "data": {
            "forecast_series": {
                "2026-06-01": {
                    "forecast": 40.0,
                    "quantile_forecast": {
                        "0.05": 22.0, "0.10": 28.0, "0.50": 40.0, "0.90": 60.0, "0.95": 66.0,
                    },
                }
            }
        }
    }


def test_shock_forecast_json_matches_apply_shock_on_the_inner_band():
    shocked_doc = shock_forecast_json(_forecast_json(), magnitude=1.0)
    q = shocked_doc["data"]["forecast_series"]["2026-06-01"]["quantile_forecast"]
    # The q10/q50/q90 it produces must equal what the policy path (apply_shock) uses.
    forecast = MonthForecast("2026-06-01", median=40.0, low=28.0, high=60.0)
    shocked = apply_shock([forecast], magnitude=1.0)[0]
    assert q["0.50"] == pytest.approx(shocked.median)
    assert q["0.10"] == pytest.approx(shocked.low)
    assert q["0.90"] == pytest.approx(shocked.high)
    # Outer band stays ordered and wider than the inner band.
    assert q["0.05"] < q["0.10"] < q["0.50"] < q["0.90"] < q["0.95"]


def test_shock_forecast_json_zero_is_a_no_op():
    doc = _forecast_json()
    assert shock_forecast_json(doc, magnitude=0.0) == doc


# --------------------------------------------------------------------------- #
# shocked_curation
# --------------------------------------------------------------------------- #
def test_shocked_curation_puts_a_risk_driver_on_top():
    shocked = shocked_curation(_curation(), magnitude=1.0)
    assert shocked.kept[0].theme == "geopolitical supply risk"
    assert shocked.kept[0].importance > _curation().kept[0].importance


def test_shocked_curation_boosts_risk_region_suppliers():
    before = {d.name: d.importance for d in _curation().kept}
    shocked = shocked_curation(_curation(), magnitude=1.0)
    iran = next(d for d in shocked.kept if d.region == "Iran" and d.theme != "geopolitical supply risk")
    germany = next(d for d in shocked.kept if d.region == "Germany")
    assert iran.importance > before["Imports of Natural gas - Iran"]  # risk supplier lifted
    assert germany.importance == before["Electricity prices - Germany"]  # non-risk unchanged


def test_shocked_curation_keeps_rejections_intact():
    shocked = shocked_curation(_curation(), magnitude=1.0)
    assert [d.name for d in shocked.rejected] == [d.name for d in _curation().rejected]


# --------------------------------------------------------------------------- #
# parse_shock_request (keyword path — no network)
# --------------------------------------------------------------------------- #
def test_parse_detects_a_full_severity_shock():
    request = parse_shock_request("Iran closes the Strait of Hormuz")
    assert request.is_shock is True
    assert request.magnitude == pytest.approx(1.0)
    assert "Hormuz" in request.label


def test_parse_reads_a_low_intensity_threat():
    request = parse_shock_request("rumours of a possible pipeline outage")
    assert request.is_shock is True
    assert request.magnitude == pytest.approx(0.35)


def test_parse_uses_a_default_magnitude_for_an_uncued_shock():
    request = parse_shock_request("new sanctions on Russian gas exports")
    assert request.is_shock is True
    assert request.magnitude == pytest.approx(0.6)
    assert "Russia" in request.label


def test_parse_rejects_a_non_shock_message():
    request = parse_shock_request("what does the calm forecast say for the summer?")
    assert request.is_shock is False
    assert request.magnitude == 0.0


def test_parse_handles_empty_input():
    request = parse_shock_request("   ")
    assert request.is_shock is False
    assert request.source == "none"


# --------------------------------------------------------------------------- #
# run_shock orchestration
# --------------------------------------------------------------------------- #
def test_run_shock_bundles_a_consistent_outcome():
    outcome = run_shock(MONTHS, SPOT, _curation(), magnitude=1.0)
    baseline = quarter_hedge_ratio(decide_all(MONTHS, SPOT))
    assert quarter_hedge_ratio(outcome.decisions) > baseline
    assert outcome.risk_premium == pytest.approx(DEFAULT_SHOCK_PARAMS.risk_premium)
    assert outcome.curation.kept[0].theme == "geopolitical supply risk"
    assert all(decision.risk_premium == outcome.risk_premium for decision in outcome.decisions)


def test_run_shock_is_deterministic():
    first = run_shock(MONTHS, SPOT, _curation(), magnitude=0.7, label="X")
    second = run_shock(MONTHS, SPOT, _curation(), magnitude=0.7, label="X")
    assert first.decisions == second.decisions
    assert [d.name for d in first.curation.kept] == [d.name for d in second.curation.kept]


def test_run_shock_respects_custom_params():
    gentle = ShockParams(median_bump=0.0, band_widen=0.0, risk_premium=0.0)
    outcome = run_shock(MONTHS, SPOT, _curation(), magnitude=1.0, params=gentle)
    # With every shock term zeroed, the decision must match the calm baseline.
    assert quarter_hedge_ratio(outcome.decisions) == pytest.approx(
        quarter_hedge_ratio(decide_all(MONTHS, SPOT))
    )
