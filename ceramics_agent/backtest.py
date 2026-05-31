"""Three-strategy backtest — does the agent's policy beat a coin flip?

The ceramics analogue of :mod:`gas_agent.decision_backtest`. There, the question
is "did the hedge policy beat always-spot / always-50%?"; here it is "did the
agent's supplier + channel + negotiation choices book more margin than naive
alternatives?" We replay the twelve committed months of :data:`HISTORICAL_SALES`
and, for each, run three strategies through the *same* deterministic negotiation:

* **S1 — agent.** The real policy: pick the supplier the lock-% routes to
  (:func:`ceramics_agent.curation.select_supplier`), the top-ranked credible
  channel for that month's season (:func:`~ceramics_agent.curation.select_channel`),
  then negotiate.
* **S2 — random.** A coin flip over suppliers and channels — but seeded with
  ``config.CERAMICS_BACKTEST_SEED`` so it **reproduces run-to-run**. Determinism
  is the selling point, even for the foil (THE RULE): a "random" baseline that
  shuffled every run could not be audited.
* **S3 — cheap + best-margin.** A static heuristic: always the cheapest supplier
  and the highest-target-margin channel. A *strong* baseline, not a naive one.

The headline comparison is **agent vs. random** (the naive baseline the agent must
beat); the cheap heuristic is shown as a third reference bar. As in the gas
backtest, the honest framing matters more than a flattering one — the verdict
reports the measured deltas, whatever they are.

Replay convention: each historical month contributes its *product*, *quantity* and
*season*; the per-unit materials cost and the policy's lock stance come from the
current forecast (today's cost structure applied to each past month), and the
replay uses a fixed representative timeline + competition since history carries
neither. Everything is pure arithmetic on the committed catalog — no model, no
network, identical inputs → identical result.

Reliability haircut: the realized margin each strategy books is the negotiated
nominal margin **scaled by the chosen supplier's reliability** (a 96%-reliable
supplier realizes 96% of the nominal margin; the rest is lost to late/short
deliveries). This is the backtest's expected-value model — the analogue of the
gas backtest pricing realized cost at the actual outturn — and it is *why* the
agent's reliability-aware supplier choice can out-earn a blind chase of the
cheapest input. Like the gas backtest's stated lock-price proxy, it is an explicit
assumption, not a hidden one.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from statistics import mean, pstdev

from gas_agent import config
from gas_agent.hedge_policy import MonthForecast

from ceramics_agent.catalog import (
    HISTORICAL_SALES,
    SalesChannel,
    SalesRecord,
    Supplier,
    get_product,
    list_channels,
    list_suppliers,
)
from ceramics_agent.curation import (
    curate_channels,
    curate_suppliers,
    select_channel,
    select_supplier,
)
from ceramics_agent.negotiation import negotiate

# Replay constants — history carries no timeline or competition, so the replay
# fixes a representative pair (documented). Both feed every strategy equally, so
# the comparison stays fair.
REPLAY_TIMELINE_DAYS = 14
REPLAY_COMPETITION = "medium"


@dataclass(frozen=True)
class CeramicsBacktestRow:
    """One replayed month: each strategy's realized margin and the agent's picks.

    Margins are *realized* (the negotiated nominal margin × the chosen supplier's
    reliability), so a less-reliable supplier's headline margin is discounted."""

    month: str
    product: str
    units: int
    agent_margin: float  # EUR realized total margin (nominal × supplier reliability)
    random_margin: float
    cheap_margin: float
    agent_supplier: str
    agent_channel: str
    # --- extra baseline (W12 — additive; default 0.0 leaves the row otherwise unchanged) ---
    top_ranked_margin: float = 0.0  # always the best-SCORED supplier, ignoring the lock routing


@dataclass(frozen=True)
class StrategyMargin:
    """Aggregate realized margin for one strategy across all replayed months."""

    name: str
    mean: float  # mean total margin per month (EUR)
    std: float  # population std — the month-to-month swing the business feels
    min: float
    max: float


@dataclass(frozen=True)
class CeramicsBacktestResult:
    """The full comparison: per-month rows plus per-strategy aggregates."""

    rows: list[CeramicsBacktestRow]
    agent: StrategyMargin
    random_: StrategyMargin
    cheap: StrategyMargin
    # --- extra baseline (W12 — additive; the headline trio above is unchanged) ---
    top_ranked: StrategyMargin = field(
        default_factory=lambda: StrategyMargin("always top-ranked", 0.0, 0.0, 0.0, 0.0))

    @property
    def n_months(self) -> int:
        return len(self.rows)

    @property
    def agent_vs_random_pct(self) -> float:
        """How much higher the agent's mean margin is than random (% of random; +ve = better)."""
        if self.random_.mean == 0:
            return 0.0
        return 100.0 * (self.agent.mean - self.random_.mean) / abs(self.random_.mean)

    @property
    def agent_vs_cheap_pct(self) -> float:
        """Agent mean margin vs the cheap+best-margin heuristic (% of cheap; +ve = better)."""
        if self.cheap.mean == 0:
            return 0.0
        return 100.0 * (self.agent.mean - self.cheap.mean) / abs(self.cheap.mean)

    @property
    def agent_vs_top_ranked_pct(self) -> float:
        """Agent mean margin vs always committing to the best-SCORED supplier, ignoring
        the lock routing (% of top-ranked; +ve = the lock-routed pick books more). This
        isolates the value of routing the supplier by the lock stance: at a mid lock the
        agent takes the balanced, more-reliable mid supplier, whose reliability haircut is
        gentler than the top-scored (cheapest) supplier's."""
        if self.top_ranked.mean == 0:
            return 0.0
        return 100.0 * (self.agent.mean - self.top_ranked.mean) / abs(self.top_ranked.mean)

    @property
    def extended_verdict(self) -> str:
        """The extra baseline as a second line under :pyattr:`verdict` (additive — ``verdict``
        stays byte-identical). Reports how the lock-routed supplier pick compares to blindly
        always taking the top-scored supplier."""
        if not self.rows:
            return ""
        delta = self.agent_vs_top_ranked_pct
        word = "above" if delta >= 0 else "below"
        return (
            f"Versus always committing to the top-scored supplier (ignoring the lock "
            f"routing), the agent books €{self.top_ranked.mean:,.0f}/month there and is "
            f"{abs(delta):.0f}% {word} it — the lock stance routes to the balanced, more-"
            f"reliable supplier whose margin survives the reliability haircut better."
        )

    @property
    def volatility_drop_vs_random(self) -> float:
        """How much tighter the agent's month-to-month swing is than random
        (% of random's std; +ve = steadier)."""
        if self.random_.std == 0:
            return 0.0
        return 100.0 * (self.random_.std - self.agent.std) / self.random_.std

    @property
    def verdict(self) -> str:
        """One honest line for the dashboard. Ceramics maximizes *margin* (unlike
        gas, which minimizes cost variance), so this leads with the margin deltas
        against both baselines, not with steadiness."""
        if not self.rows:
            return "No historical months available to replay."
        vs_random = self.agent_vs_random_pct
        vs_cheap = self.agent_vs_cheap_pct
        random_word = "more than" if vs_random >= 0 else "less than"
        cheap_word = "above" if vs_cheap >= 0 else "below"
        return (
            f"Across {self.n_months} replayed months the agent books €{self.agent.mean:,.0f} "
            f"realized margin/month on average — {abs(vs_random):.0f}% {random_word} a random "
            f"supplier/channel pick (€{self.random_.mean:,.0f}), and {abs(vs_cheap):.0f}% "
            f"{cheap_word} a static cheapest-supplier + highest-margin-channel heuristic "
            f"(€{self.cheap.mean:,.0f}), which it edges by weighting supplier reliability, "
            f"not price alone."
        )


def _margin_stats(name: str, values: list[float]) -> StrategyMargin:
    if not values:
        return StrategyMargin(name=name, mean=0.0, std=0.0, min=0.0, max=0.0)
    spread = pstdev(values) if len(values) > 1 else 0.0
    return StrategyMargin(name=name, mean=mean(values), std=spread, min=min(values), max=max(values))


def _cheapest_supplier() -> Supplier:
    return min(list_suppliers(), key=lambda s: s.average_price_factor())


def _best_margin_channel() -> SalesChannel:
    return max(list_channels(), key=lambda c: c.target_margin)


def _realized(margin: float, supplier: Supplier) -> float:
    """Nominal margin discounted by the supplier's reliability (the haircut)."""
    return margin * (supplier.reliability_pct / 100.0)


def run_ceramics_backtest(
    factors: dict[str, list[MonthForecast]],
    lock_ratio: float,
    *,
    timeline_days: int = REPLAY_TIMELINE_DAYS,
    competition: str = REPLAY_COMPETITION,
    seed: int = config.CERAMICS_BACKTEST_SEED,
    records: list[SalesRecord] | None = None,
) -> CeramicsBacktestResult:
    """Replay the three strategies over the historical months and aggregate margins.

    ``lock_ratio`` is the agent's policy stance (from :func:`cost_policy.decide_procurement`),
    routing its supplier pick; ``factors`` provides the per-unit cost basis for the
    negotiation. The random baseline is seeded with ``seed`` so it reproduces."""
    history = records if records is not None else HISTORICAL_SALES
    rng = random.Random(seed)  # seeded once → the whole random sequence is reproducible
    suppliers = list_suppliers()
    channels = list_channels()

    cheapest = _cheapest_supplier()
    best_margin = _best_margin_channel()

    # The agent's supplier follows the lock stance + timeline, so it is fixed across
    # the replay; its channel is re-picked per month (season-aware).
    kept_suppliers = curate_suppliers(timeline_days).kept
    agent_supplier_pick = select_supplier(kept_suppliers, lock_ratio)
    # S4 baseline: always the best-SCORED supplier, ignoring the lock routing.
    top_ranked_supplier = kept_suppliers[0].supplier if kept_suppliers else suppliers[0]

    rows: list[CeramicsBacktestRow] = []
    agent_margins: list[float] = []
    random_margins: list[float] = []
    cheap_margins: list[float] = []
    top_ranked_margins: list[float] = []

    for record in history:
        product = get_product(record.product_id)

        # --- S1 agent --------------------------------------------------------
        kept_channels = curate_channels(record.month, record.units).kept
        agent_channel_pick = select_channel(kept_channels)
        agent_supplier = agent_supplier_pick.supplier if agent_supplier_pick else suppliers[0]
        agent_channel = agent_channel_pick.channel if agent_channel_pick else channels[0]
        agent = negotiate(
            product, agent_supplier, agent_channel, factors,
            record.month, record.units, timeline_days, competition,
        )

        # --- S2 random (seeded) ---------------------------------------------
        rnd_supplier = rng.choice(suppliers)
        rnd_channel = rng.choice(channels)
        rnd = negotiate(
            product, rnd_supplier, rnd_channel, factors,
            record.month, record.units, timeline_days, competition,
        )

        # --- S3 cheap + best-margin -----------------------------------------
        cheap = negotiate(
            product, cheapest, best_margin, factors,
            record.month, record.units, timeline_days, competition,
        )

        # --- S4 always top-ranked supplier (+ the season channel) -----------
        # No RNG here, so S2's seeded sequence is untouched and S1-S3 stay byte-identical.
        top = negotiate(
            product, top_ranked_supplier, agent_channel, factors,
            record.month, record.units, timeline_days, competition,
        )

        # Realized margin = nominal × the chosen supplier's reliability (the haircut).
        agent_realized = _realized(agent.total_margin, agent_supplier)
        random_realized = _realized(rnd.total_margin, rnd_supplier)
        cheap_realized = _realized(cheap.total_margin, cheapest)
        top_ranked_realized = _realized(top.total_margin, top_ranked_supplier)

        agent_margins.append(agent_realized)
        random_margins.append(random_realized)
        cheap_margins.append(cheap_realized)
        top_ranked_margins.append(top_ranked_realized)
        rows.append(
            CeramicsBacktestRow(
                month=record.month,
                product=product.name,
                units=record.units,
                agent_margin=agent_realized,
                random_margin=random_realized,
                cheap_margin=cheap_realized,
                agent_supplier=agent_supplier.name,
                agent_channel=agent_channel.name,
                top_ranked_margin=top_ranked_realized,
            )
        )

    return CeramicsBacktestResult(
        rows=rows,
        agent=_margin_stats("agent policy", agent_margins),
        random_=_margin_stats("random pick", random_margins),
        cheap=_margin_stats("cheap + best margin", cheap_margins),
        top_ranked=_margin_stats("always top-ranked", top_ranked_margins),
    )


if __name__ == "__main__":  # quick manual check off the committed mock forecast
    from ceramics_agent.catalog import EXTENDED_HISTORICAL_SALES
    from ceramics_agent.cost_policy import decide_procurement, default_weights, quarter_lock_ratio
    from ceramics_agent.forecast import load_ceramics_forecast

    factors, source = load_ceramics_forecast()
    decisions = decide_procurement(factors, default_weights())
    lock = quarter_lock_ratio(decisions)
    result = run_ceramics_backtest(factors, lock)
    print(f"source={source}  lock_ratio={lock:.0%}  months={result.n_months}")
    for row in result.rows:
        print(
            f"  {row.month}  {row.product:22s} x{row.units:5d}  "
            f"agent €{row.agent_margin:10,.0f}  random €{row.random_margin:10,.0f}  "
            f"cheap €{row.cheap_margin:10,.0f}  top €{row.top_ranked_margin:10,.0f}  "
            f"[{row.agent_supplier} / {row.agent_channel}]"
        )
    print()
    print(result.verdict)
    print(result.extended_verdict)
    print()
    longer = run_ceramics_backtest(factors, lock, records=EXTENDED_HISTORICAL_SALES)
    print(f"Robustness — {longer.n_months}-month replay: agent €{longer.agent.mean:,.0f}/mo, "
          f"{longer.agent_vs_random_pct:+.0f}% vs random, {longer.agent_vs_cheap_pct:+.0f}% vs cheap, "
          f"{longer.agent_vs_top_ranked_pct:+.0f}% vs top-ranked.")
