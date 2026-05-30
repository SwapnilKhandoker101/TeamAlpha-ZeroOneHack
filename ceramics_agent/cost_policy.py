"""Cost aggregation + the lock-% decision — the ceramics analogue of the hedge.

The ceramics buyer faces the same question the gas buyer does, one level up:
*what share of next quarter's input cost should we lock in forward now versus
leave to float?* The four factor forecasts (gas / clay / power / shipping) are
blended under the user's cost-factor weights, and the **lock %** is decided by
the **same deterministic engine the gas agent uses** — :func:`gas_agent.hedge_policy.decide_month`
— so THE RULE holds verbatim: a model never sets the number, the band does.

Two distinct aggregations live here, deliberately kept apart:

1. **Decision volatility** → drives the lock %. Per month we compute a
   weighted relative band ``band_width = Σ wₖ·(q90ₖ−q10ₖ)/q50ₖ`` and a blended
   cost index ``level = Σ wₖ·q50ₖ``. We pack those into a *synthetic*
   :class:`~gas_agent.hedge_policy.MonthForecast` (median = level, band centered
   so ``(high−low)/median == band_width``) and hand it to ``decide_month``. The
   returned ``MonthDecision.hedge_ratio`` **is the lock %**; the band thresholds
   are tuned for ceramics (tight ≤ 25%, wide ≥ 60%). A tight blended band → lock
   a lot; a wide one → keep flexibility.

2. **Physical per-unit cost** → drives the cost chart and the negotiation floor.
   This is a *real* EUR-per-unit number from the bill of materials:
   ``clay_kg·clay + kiln_kwh·power + firing_gas_kwh·gas + ship_kg·shipping``,
   evaluated at each factor's q10/q50/q90. It is not weighted — weights only
   shape the *decision*, not the actual cost a unit incurs.

The blended index mixes factor units (gas EUR/MWh, clay EUR/kg, …) on purpose:
its *absolute* value is meaningless, but its band width and its month-over-month
drift — the only two things the policy reads — are unit-free and well-defined.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from gas_agent import config
from gas_agent.hedge_policy import (
    HedgePolicyParams,
    MonthDecision,
    MonthForecast,
    decide_all,
    quarter_hedge_ratio,
)

from ceramics_agent.catalog import Product

# Ceramics-tuned band thresholds: the data-source doc's "tight ≤ 25% / wide ≥ 60%".
# Everything else (tilt gain, clamps) inherits the gas defaults, so the lock %
# lives on [0.10, 0.90] just like the hedge ratio.
CERAMICS_POLICY_PARAMS = HedgePolicyParams(low_band=0.25, high_band=0.60)

# The user weights are named for the *business* cost buckets; the forecast is
# named for the *series*. This is the one mapping between them.
WEIGHT_TO_FACTOR: dict[str, str] = {
    "gas": "gas",
    "clay": "clay",
    "energy": "power",
    "transport": "shipping",
}


@dataclass(frozen=True)
class CostWeights:
    """User-set importance of each cost bucket. Need not sum to 1 — :meth:`normalized`
    rescales — so the UI sliders can move freely."""

    gas: float = 0.40
    clay: float = 0.35
    energy: float = 0.15
    transport: float = 0.10

    def normalized(self) -> "CostWeights":
        """Rescale to sum to 1.0. All-zero (or negative) input falls back to an
        equal split, so a blended index always exists."""
        parts = [max(0.0, self.gas), max(0.0, self.clay), max(0.0, self.energy), max(0.0, self.transport)]
        total = sum(parts)
        if total <= 0:
            return CostWeights(0.25, 0.25, 0.25, 0.25)
        return CostWeights(parts[0] / total, parts[1] / total, parts[2] / total, parts[3] / total)

    def as_factor_weights(self) -> dict[str, float]:
        """Normalized weights keyed by *factor* name (energy→power, transport→shipping)."""
        norm = self.normalized()
        bucket = {"gas": norm.gas, "clay": norm.clay, "energy": norm.energy, "transport": norm.transport}
        return {WEIGHT_TO_FACTOR[name]: value for name, value in bucket.items()}

    def as_dict(self) -> dict[str, float]:
        return {"gas": self.gas, "clay": self.clay, "energy": self.energy, "transport": self.transport}


def default_weights() -> CostWeights:
    """The calm starting weights from config (gas-dominant)."""
    cfg = config.CERAMICS_DEFAULT_WEIGHTS
    return CostWeights(gas=cfg["gas"], clay=cfg["clay"], energy=cfg["energy"], transport=cfg["transport"])


# --------------------------------------------------------------------------- #
# Aggregation 1 — the weighted blended index that drives the lock %
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BlendedMonth:
    """One month of the weighted blended cost index (the decision input)."""

    month: str
    level: float  # Σ wₖ·q50ₖ — a unit-mixed index; only its drift + band matter
    band_width: float  # Σ wₖ·(q90ₖ−q10ₖ)/q50ₖ — the weighted relative 80% band


def _aligned_months(factors: dict[str, list[MonthForecast]]) -> list[str]:
    """The common month grid (factors all share it; sorted). Empty when there are
    no factors, so the callers' empty-input guards (e.g. ``decide_procurement``'s
    ``if not blended``) actually fire instead of tripping on an empty iterator."""
    any_factor = next(iter(factors.values()), [])
    return [m.month for m in sorted(any_factor, key=lambda f: f.month)]


def _by_month(months: list[MonthForecast]) -> dict[str, MonthForecast]:
    return {m.month: m for m in months}


def blended_index(
    factors: dict[str, list[MonthForecast]],
    weights: CostWeights,
) -> list[BlendedMonth]:
    """Blend the per-factor bands into the weighted index the policy decides on."""
    factor_weights = weights.as_factor_weights()
    indexed = {factor: _by_month(months) for factor, months in factors.items()}

    blended: list[BlendedMonth] = []
    for month in _aligned_months(factors):
        level = 0.0
        band = 0.0
        for factor, weight in factor_weights.items():
            entry = indexed.get(factor, {}).get(month)
            if entry is None or entry.median == 0:
                continue
            level += weight * entry.median
            band += weight * (entry.high - entry.low) / entry.median
        blended.append(BlendedMonth(month=month, level=level, band_width=band))
    return blended


def _synthetic_forecast(month: BlendedMonth) -> MonthForecast:
    """Pack a blended month into a MonthForecast whose 80% band, centered on the
    level, reproduces the weighted band width exactly: ``(high−low)/median == band_width``."""
    half = month.level * month.band_width / 2.0
    return MonthForecast(
        month=month.month,
        median=month.level,
        low=month.level - half,
        high=month.level + half,
    )


def decide_procurement(
    factors: dict[str, list[MonthForecast]],
    weights: CostWeights,
    params: HedgePolicyParams = CERAMICS_POLICY_PARAMS,
) -> list[MonthDecision]:
    """Decide the per-month lock % from the weighted blended band + its drift.

    Reuses the gas hedge engine verbatim: each blended month becomes a synthetic
    forecast, decided against a single horizon-start anchor (the first month's
    blended level — the nearest-future cost, the ceramics analogue of "today's
    spot"). The resulting ``MonthDecision.hedge_ratio`` is the lock %.
    """
    blended = blended_index(factors, weights)
    if not blended:
        return []
    synthetic = [_synthetic_forecast(month) for month in blended]
    anchor = blended[0].level  # nearest-future blended cost — the "today" reference
    decisions = decide_all(synthetic, anchor, params)
    # Relabel the reason so it reads as a cost-lock, not a gas hedge.
    return [_relabel(decision) for decision in decisions]


def _relabel(decision: MonthDecision) -> MonthDecision:
    """Reword the engine's gas-flavoured trace into ceramics cost-lock language,
    without touching a single number (the decision is final)."""
    reason = (
        f"{decision.band_regime} blended band ({decision.band_width:.0%}) "
        f"+ cost {decision.direction} vs horizon start ({decision.drift_pct:+.0%}) "
        f"-> lock {decision.hedge_ratio:.0%}"
    )
    return replace(decision, reason=reason)


def quarter_lock_ratio(decisions: list[MonthDecision]) -> float:
    """Average lock % across the next quarter (first three months) — the headline."""
    return quarter_hedge_ratio(decisions[:3])


def quarter_band_width(decisions: list[MonthDecision]) -> float:
    """Mean weighted blended band width over the next quarter — the 'how volatile
    is our cost' readout shown beside the lock %."""
    quarter = decisions[:3]
    if not quarter:
        return 0.0
    return sum(d.band_width for d in quarter) / len(quarter)


# --------------------------------------------------------------------------- #
# Aggregation 2 — the real physical per-unit cost (chart + negotiation floor)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PhysicalCost:
    """Real EUR-per-unit cost for one month, with its uncertainty band.

    ``q10``/``q90`` sum each factor's own q10/q90 contribution — a transparent,
    deterministic outer band (it assumes the factors move together, so it is a
    slight over-estimate of spread, stated rather than hidden)."""

    month: str
    q10: float
    q50: float
    q90: float

    @property
    def band_width(self) -> float:
        return (self.q90 - self.q10) / self.q50 if self.q50 else 0.0


def _gas_eur_per_kwh(gas_eur_per_mwh: float) -> float:
    """Gas is forecast in EUR/MWh (TTF); the BOM burns it in kWh."""
    return gas_eur_per_mwh / 1000.0


def _unit_cost_at(product: Product, factor_quantile: dict[str, float]) -> float:
    """One unit's cost given a price for each factor (all at the same quantile)."""
    return (
        product.clay_kg * factor_quantile.get("clay", 0.0)
        + product.kiln_kwh * factor_quantile.get("power", 0.0)
        + product.firing_gas_kwh * _gas_eur_per_kwh(factor_quantile.get("gas", 0.0))
        + product.ship_kg * factor_quantile.get("shipping", 0.0)
    )


def physical_cost_band(
    product: Product,
    factors: dict[str, list[MonthForecast]],
) -> list[PhysicalCost]:
    """Per-month physical cost (q10/q50/q90 EUR per unit) from the BOM × factor prices."""
    indexed = {factor: _by_month(months) for factor, months in factors.items()}
    costs: list[PhysicalCost] = []
    for month in _aligned_months(factors):
        lows = {f: indexed[f][month].low for f in factors if month in indexed[f]}
        meds = {f: indexed[f][month].median for f in factors if month in indexed[f]}
        highs = {f: indexed[f][month].high for f in factors if month in indexed[f]}
        costs.append(
            PhysicalCost(
                month=month,
                q10=_unit_cost_at(product, lows),
                q50=_unit_cost_at(product, meds),
                q90=_unit_cost_at(product, highs),
            )
        )
    return costs


def unit_cost_estimate(product: Product, factors: dict[str, list[MonthForecast]], month_index: int = 0) -> float:
    """The median physical unit cost for one month (default: the first) — the
    anchor the negotiation builds its quotes from."""
    band = physical_cost_band(product, factors)
    if not band:
        return 0.0
    index = max(0, min(month_index, len(band) - 1))
    return band[index].q50


def component_unit_costs(
    product: Product,
    factors: dict[str, list[MonthForecast]],
    month_index: int = 0,
) -> dict[str, float]:
    """Median EUR-per-unit cost split by physical component for one month.

    Returns ``{"clay", "power", "gas", "shipping"}`` → EUR/unit at the q50 factor
    price — the per-component breakdown whose sum equals
    :func:`unit_cost_estimate`. The negotiation multiplies each component by a
    supplier's per-component price factor, so a supplier that is cheap on clay but
    dear on shipping reprices the quote correctly. Gas is converted EUR/MWh→EUR/kWh
    on the way in (the BOM burns kWh), exactly as the total cost does."""
    indexed = {factor: _by_month(months) for factor, months in factors.items()}
    months = _aligned_months(factors) if factors else []
    if not months:
        return {"clay": 0.0, "power": 0.0, "gas": 0.0, "shipping": 0.0}
    index = max(0, min(month_index, len(months) - 1))
    month = months[index]

    def median(factor: str) -> float:
        entry = indexed.get(factor, {}).get(month)
        return entry.median if entry is not None else 0.0

    return {
        "clay": product.clay_kg * median("clay"),
        "power": product.kiln_kwh * median("power"),
        "gas": product.firing_gas_kwh * _gas_eur_per_kwh(median("gas")),
        "shipping": product.ship_kg * median("shipping"),
    }
