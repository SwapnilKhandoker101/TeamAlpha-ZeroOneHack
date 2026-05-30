"""The deterministic hedge-policy engine.

This is the heart of the agent and the one rule that keeps it honest: the LLMs
prepare inputs and explain outputs, but the hedge ratio is decided *here*, by
plain arithmetic on the forecast's confidence band and drift. No model, no
randomness — the same forecast always yields the same decision, and every
number that goes into that decision is reported back for inspection.

The decision answers: "what share of next quarter's gas should we lock in
forward now, versus leave to buy later on the spot market?" — a hedge ratio
between a floor and a cap, per forecast month.

Intuition:
  * A *narrow* confidence band means the forecast is confident → little reason
    to stay flexible → lock in a high share now.
  * A *wide* band means the future is uncertain → keep optionality → lock less.
  * On top of that, if a forecast month's median sits *above* today's spot price,
    waiting to buy means paying more, so tilt toward locking more now; if it sits
    *below* spot, the model expects cheaper spot later, so tilt toward locking less.

The band thresholds below are calibrated against this series' backtest: ``low_band``
sits near the 15th percentile of historical band widths (the model is unusually
confident) and ``high_band`` near the top of the observed range (almost no
confidence), so "tight" and "wide" are relative to what this model realistically
produces rather than arbitrary constants.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HedgePolicyParams:
    """Tunable knobs. Defaults are sensible; ``low_band``/``high_band`` get
    calibrated against the backtest band widths so the band maps onto the
    realistic range this series actually produces."""

    low_band: float = 0.30  # at/below this relative band width, lock the most (~backtest p15)
    high_band: float = 1.10  # at/above this, the band is so wide we keep optionality (~backtest max)
    tilt_gain: float = 1.0  # how strongly the forward-vs-spot gap shifts the ratio
    max_tilt: float = 0.30  # cap on the drift adjustment, either direction
    min_hedge: float = 0.10  # never lock less than this (some baseline coverage)
    max_hedge: float = 0.90  # never lock more than this (always keep some spot)


DEFAULT_PARAMS = HedgePolicyParams()


@dataclass(frozen=True)
class MonthForecast:
    """One forecast month, reduced to the three quantiles the policy needs."""

    month: str  # "2026-06-01"
    median: float  # quantile 0.50
    low: float  # quantile 0.10
    high: float  # quantile 0.90


@dataclass(frozen=True)
class MonthDecision:
    """The decision for one month, with every component exposed for tracing."""

    month: str
    median: float
    low: float
    high: float
    band_width: float  # (high - low) / median — relative width of the 80% band
    drift_pct: float  # (median - spot_price) / spot_price — forward vs today's spot
    ratio_from_band: float  # band component of the decision, before the tilt
    direction_tilt: float  # drift adjustment (+ locks more, - locks less)
    hedge_ratio: float  # final decision, clamped to [min_hedge, max_hedge]
    band_regime: str  # "tight" | "moderate" | "wide"
    direction: str  # "rising" | "falling" | "flat"
    reason: str  # one-line human-readable trace

    def as_dict(self) -> dict:
        return asdict(self)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _classify_band(band_width: float, params: HedgePolicyParams) -> str:
    if band_width <= params.low_band:
        return "tight"
    if band_width >= params.high_band:
        return "wide"
    return "moderate"


def _classify_direction(drift_pct: float) -> str:
    if drift_pct > 0.01:
        return "rising"
    if drift_pct < -0.01:
        return "falling"
    return "flat"


def decide_month(
    forecast: MonthForecast,
    spot_price: float,
    params: HedgePolicyParams = DEFAULT_PARAMS,
) -> MonthDecision:
    """Compute the hedge ratio for a single month.

    ``spot_price`` is today's observed price. The drift compares this month's
    forecast median to that spot: a forward above spot means buying later is
    expected to cost more (tilt toward locking), below spot the reverse.
    """
    band_width = (forecast.high - forecast.low) / forecast.median

    # Band component: tight band -> ~1.0 (lock a lot), wide band -> ~0.0 (stay flexible).
    band_span = params.high_band - params.low_band
    ratio_from_band = clamp(1.0 - (band_width - params.low_band) / band_span, 0.0, 1.0)

    # Drift component: forward above spot tilts toward locking more, below toward less.
    drift_pct = (forecast.median - spot_price) / spot_price
    direction_tilt = clamp(drift_pct * params.tilt_gain, -params.max_tilt, params.max_tilt)

    hedge_ratio = clamp(ratio_from_band + direction_tilt, params.min_hedge, params.max_hedge)

    band_regime = _classify_band(band_width, params)
    direction = _classify_direction(drift_pct)
    reason = (
        f"{band_regime} band ({band_width:.0%} of median) "
        f"+ median {direction} vs spot ({drift_pct:+.0%}) "
        f"-> lock {hedge_ratio:.0%}"
    )

    return MonthDecision(
        month=forecast.month,
        median=forecast.median,
        low=forecast.low,
        high=forecast.high,
        band_width=band_width,
        drift_pct=drift_pct,
        ratio_from_band=ratio_from_band,
        direction_tilt=direction_tilt,
        hedge_ratio=hedge_ratio,
        band_regime=band_regime,
        direction=direction,
        reason=reason,
    )


def decide_all(
    forecasts: list[MonthForecast],
    spot_price: float,
    params: HedgePolicyParams = DEFAULT_PARAMS,
) -> list[MonthDecision]:
    """Decide every forecast month, each compared against today's spot price."""
    return [decide_month(forecast, spot_price, params) for forecast in forecasts]


def quarter_hedge_ratio(decisions: list[MonthDecision]) -> float:
    """Average hedge ratio across the given months — the single headline number
    for 'what share of next quarter do we lock in now?'."""
    if not decisions:
        return 0.0
    return sum(decision.hedge_ratio for decision in decisions) / len(decisions)
