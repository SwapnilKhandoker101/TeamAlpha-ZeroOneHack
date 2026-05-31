"""The recommendation orchestrator — every deterministic piece, assembled once.

This is the ceramics analogue of the gas dashboard's decision assembly: it pulls
the 4-factor forecast, runs the lock-% policy, the supplier/channel curation, the
two-round negotiation and the three-strategy backtest, and packs them into one
:class:`Recommendation` the dashboard renders. It is **pure and deterministic** —
no LLM, no network on the default (mock) path — so the same inputs always produce
the same recommendation, and the end-to-end flow is testable offline.

The narrative is deliberately *not* built here: :func:`ceramics_agent.explanation.explain_recommendation`
takes a finished :class:`Recommendation` and explains it. Keeping the (slow,
optional) LLM call out of this function lets the dashboard cache it separately and
lets the determinism test exercise the whole pipeline without touching a model
(the same separation the gas agent keeps between its policy and its explainer).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from gas_agent.hedge_policy import MonthDecision, MonthForecast

from ceramics_agent.backtest import CeramicsBacktestResult, run_ceramics_backtest
from ceramics_agent.catalog import Product, get_product
from ceramics_agent.cost_policy import (
    CostWeights,
    PhysicalCost,
    decide_procurement,
    physical_cost_band,
    quarter_band_width,
    quarter_lock_ratio,
    unit_cost_estimate,
)
from ceramics_agent.curation import (
    ChannelCuration,
    CuratedChannel,
    CuratedSupplier,
    SupplierCuration,
    curate_channels,
    curate_suppliers,
    select_channel,
    select_supplier,
)
from ceramics_agent.forecast import load_ceramics_forecast
from ceramics_agent.negotiation import NegotiationResult, negotiate
from ceramics_agent.scenario import run_ceramics_shock


@dataclass(frozen=True)
class Recommendation:
    """Everything the dashboard needs for one (product, quantity, timeline, weights,
    competition, target-month) request — deterministic, assembled from the catalog
    and the committed forecast. The explanation is added separately."""

    # --- inputs (echoed back for the trace) ---
    product: Product
    quantity: int
    timeline_days: int
    weights: CostWeights
    competition: str
    target_month: str
    source: str  # "mock" | "live" — drives the dashboard's offline banner

    # --- the lock-% decision (C3) ---
    decisions: list[MonthDecision]  # one per forecast month
    lock_ratio: float  # next-quarter average lock % — the headline
    band_width: float  # mean weighted blended band over the quarter (the volatility readout)
    unit_cost: float  # nearest-future median physical EUR/unit
    cost_band: list[PhysicalCost]  # per-month q10/q50/q90 EUR/unit (the chart)

    # --- curation + choices (C4) ---
    supplier_curation: SupplierCuration
    channel_curation: ChannelCuration
    chosen_supplier: CuratedSupplier | None
    chosen_channel: CuratedChannel | None

    # --- negotiation + backtest (C5, C6) ---
    negotiation: NegotiationResult
    backtest: CeramicsBacktestResult

    # --- optional supply-shock overlay (W6/W7 — additive; defaults = calm path) ---
    # When a chat headline shocks the ceramics line, these record what moved so the
    # dashboard can draw a scenario banner and the calm→shocked lock delta. All
    # defaults leave the calm recommendation byte-identical to before.
    calm_lock_ratio: float = 0.0  # the pre-shock lock (the delta's baseline)
    shock_magnitude: float = 0.0
    shock_label: str = ""
    shock_affected: tuple[str, ...] = ()
    scenario_premium: float = 0.0  # the shock's OWN marginal floor (for the readout)

    @property
    def shock_active(self) -> bool:
        """True when a supply shock re-decided this recommendation."""
        return self.shock_magnitude > 0.0 and bool(self.shock_affected)

    @property
    def lock_delta(self) -> float:
        """How much the shock lifted the lock vs the calm baseline (0.0 when calm)."""
        return self.lock_ratio - self.calm_lock_ratio

    @property
    def band_regime(self) -> str:
        """The quarter's volatility regime label, straight from the lead month's
        decision (tight / moderate / wide) — reused from the gas hedge engine."""
        return self.decisions[0].band_regime if self.decisions else "normal"

    @property
    def lock_label(self) -> str:
        """Which lock band routed the supplier pick — for the explanation/trace."""
        from ceramics_agent.curation import LOCK_HIGH, LOCK_LOW

        if self.lock_ratio >= LOCK_HIGH:
            return "high lock — commit to the best-ranked supplier"
        if self.lock_ratio <= LOCK_LOW:
            return "low lock — stay flexible, favour the cheapest supplier"
        return "mid lock — take the balanced, mid-ranked supplier"


def _forecast_months(factors: dict[str, list[MonthForecast]]) -> list[str]:
    """The shared month grid (any factor carries it; sorted)."""
    any_factor = next(iter(factors.values()), [])
    return [m.month for m in sorted(any_factor, key=lambda f: f.month)]


def available_months(job_id: str | None = None) -> list[str]:
    """The forecast's month grid — lets the dashboard offer a target-month picker
    before building the full recommendation."""
    factors, _ = load_ceramics_forecast(job_id)
    return _forecast_months(factors)


def build_recommendation(
    product_id: str,
    quantity: int,
    timeline_days: int,
    weights: CostWeights,
    competition: str,
    *,
    target_month: str | None = None,
    job_id: str | None = None,
    shock_magnitude: float = 0.0,
    shock_affected: Iterable[str] = (),
    shock_label: str = "",
    base_risk_premium: float = 0.0,
) -> Recommendation:
    """Run the full deterministic pipeline and assemble the recommendation.

    ``target_month`` defaults to the nearest forecast month (the next production
    run). Everything is pure arithmetic on the committed catalog + forecast; the
    only optional, off-path I/O is a live forecast behind ``job_id`` (else mock).

    A chat-driven supply shock is applied via the optional ``shock_*`` params
    (default = no shock → byte-identical calm recommendation). When active, the
    affected factor band(s) are re-cast by :func:`ceramics_agent.scenario.run_ceramics_shock`
    and the **shocked** bands drive the cost chart, the supplier routing (a higher
    lock favours the top-ranked supplier), the negotiation and the backtest — so one
    headline moves the whole ceramics decision, mirroring the gas hedge. THE RULE
    still holds: every number here is deterministic arithmetic, never an LLM output."""
    factors, source = load_ceramics_forecast(job_id)
    product = get_product(product_id)
    months = _forecast_months(factors)
    target = target_month if target_month in months else (months[0] if months else "")

    # --- the calm baseline (always computed — the pre-shock reference for the delta) ---
    calm_decisions = decide_procurement(factors, weights)
    calm_lock_ratio = quarter_lock_ratio(calm_decisions)

    # --- C3: the lock-% decision + the physical cost band (shocked when a shock is live) ---
    outcome = (
        run_ceramics_shock(
            factors, weights, shock_magnitude, shock_affected,
            label=shock_label, base_risk_premium=base_risk_premium,
        )
        if shock_magnitude > 0.0 else None
    )
    shock_on = outcome is not None and bool(outcome.affected_factors)
    eff_factors = outcome.factors if shock_on else factors
    decisions = outcome.decisions if shock_on else calm_decisions
    lock_ratio = outcome.lock_ratio if shock_on else calm_lock_ratio
    band_width = outcome.band_width if shock_on else quarter_band_width(calm_decisions)
    cost_band = physical_cost_band(product, eff_factors)
    unit_cost = unit_cost_estimate(product, eff_factors)

    # --- C4: curate + choose, routed by the (possibly shocked) lock stance ---
    supplier_curation = curate_suppliers(timeline_days)
    channel_curation = curate_channels(target, quantity)
    chosen_supplier = select_supplier(supplier_curation.kept, lock_ratio)
    chosen_channel = select_channel(channel_curation.kept)

    # --- C5: negotiate the chosen pairing (fall back to the first kept if needed) ---
    supplier_obj = (
        chosen_supplier.supplier
        if chosen_supplier
        else (supplier_curation.kept[0].supplier if supplier_curation.kept else None)
    )
    channel_obj = (
        chosen_channel.channel
        if chosen_channel
        else (channel_curation.kept[0].channel if channel_curation.kept else None)
    )
    if supplier_obj is None or channel_obj is None:
        raise ValueError("No credible supplier or channel survived curation — cannot negotiate.")
    negotiation = negotiate(
        product, supplier_obj, channel_obj, eff_factors, target, quantity, timeline_days, competition
    )

    # --- C6: replay the three strategies for the same (possibly shocked) lock stance ---
    backtest = run_ceramics_backtest(eff_factors, lock_ratio)

    return Recommendation(
        product=product,
        quantity=quantity,
        timeline_days=timeline_days,
        weights=weights,
        competition=competition,
        target_month=target,
        source=source,
        decisions=decisions,
        lock_ratio=lock_ratio,
        band_width=band_width,
        unit_cost=unit_cost,
        cost_band=cost_band,
        supplier_curation=supplier_curation,
        channel_curation=channel_curation,
        chosen_supplier=chosen_supplier,
        chosen_channel=chosen_channel,
        negotiation=negotiation,
        backtest=backtest,
        calm_lock_ratio=calm_lock_ratio,
        shock_magnitude=outcome.magnitude if shock_on else 0.0,
        shock_label=outcome.label if shock_on else "",
        shock_affected=outcome.affected_factors if shock_on else (),
        scenario_premium=outcome.risk_premium if shock_on else 0.0,
    )
