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
) -> Recommendation:
    """Run the full deterministic pipeline and assemble the recommendation.

    ``target_month`` defaults to the nearest forecast month (the next production
    run). Everything is pure arithmetic on the committed catalog + forecast; the
    only optional, off-path I/O is a live forecast behind ``job_id`` (else mock)."""
    factors, source = load_ceramics_forecast(job_id)
    product = get_product(product_id)
    months = _forecast_months(factors)
    target = target_month if target_month in months else (months[0] if months else "")

    # --- C3: the lock-% decision + the physical cost band ---
    decisions = decide_procurement(factors, weights)
    lock_ratio = quarter_lock_ratio(decisions)
    band_width = quarter_band_width(decisions)
    cost_band = physical_cost_band(product, factors)
    unit_cost = unit_cost_estimate(product, factors)

    # --- C4: curate + choose, routed by the lock stance ---
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
        product, supplier_obj, channel_obj, factors, target, quantity, timeline_days, competition
    )

    # --- C6: replay the three strategies for the same lock stance ---
    backtest = run_ceramics_backtest(factors, lock_ratio)

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
    )
