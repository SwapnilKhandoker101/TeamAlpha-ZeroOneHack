"""Two-round buy/sell negotiation — the margin, made traceable.

This is the ceramics agent's "what to pay / what to charge" step. It models a
short, deterministic negotiation between three parties from the manufacturer's
seat:

* the **supplier** quotes a per-unit materials price (what the manufacturer
  *pays* to buy), and
* the **sales channel** offers a per-unit price (what the manufacturer *gets*
  when it sells),

with the gap between them being the manufacturer's **unit margin**. Two rounds:

* **Round 1 — opening positions.** The supplier prices the bill of materials at
  its own per-component factors and adds a fixed 15% supplier margin (the buy
  side). The channel opens at the **market** production cost (the same BOM at the
  1.0 baseline factors) marked up by its target margin — a market-anchored
  willingness-to-pay, *independent of which supplier was chosen*. This is the
  economically honest tie: a cheaper supplier lowers the buy price without lowering
  the sell price, so it correctly *widens* the margin.
* **Round 2 — counters.** The supplier pushes the buy price *up* when demand is
  hot (the target quarter's channel seasonality ≥ 1.2, +5%) or the order is a rush
  (timeline < 7 days, +10%). The channel pushes the sell price *down* the more
  competition the user reports (high −7%, medium −3%, low 0%). Both moves squeeze
  the margin, so the user's market read actually changes the outcome.

Every number is pure arithmetic on the committed catalog and the physical cost
band (:func:`ceramics_agent.cost_policy.component_unit_costs`); no model is
consulted. Identical inputs always produce identical rows — the same determinism
guarantee the gas agent's decision path makes (THE RULE).
"""

from __future__ import annotations

from dataclasses import dataclass

from gas_agent.hedge_policy import MonthForecast

from ceramics_agent.catalog import Product, SalesChannel, Supplier, quarter_of
from ceramics_agent.cost_policy import component_unit_costs

# Negotiation constants (deterministic; tuned for a legible demo).
SUPPLIER_BASE_MARGIN = 0.15  # the supplier's own markup over its materials cost
DEMAND_PREMIUM = 0.05  # supplier surcharge when the target quarter is hot
RUSH_PREMIUM = 0.10  # supplier surcharge for a rush order (short timeline)
HIGH_DEMAND_SEASON = 1.2  # channel seasonality at/above this counts as "hot demand"
RUSH_TIMELINE_DAYS = 7  # a timeline strictly under this is a rush order

# How hard the channel pushes the sell price down, by the competition the user
# reports facing. More competition for the channel's shelf → lower price to us.
COMPETITION_DISCOUNT: dict[str, float] = {"high": 0.07, "medium": 0.03, "low": 0.0}


@dataclass(frozen=True)
class QuoteRow:
    """One logged move in the negotiation, for the dashboard's audit table.

    ``kind`` is ``"price"`` when ``value`` is a EUR-per-unit quote, or ``"margin"``
    when ``value`` is a fraction (the closing unit margin)."""

    round: int  # 1 (opening) or 2 (counter)
    party: str  # "supplier", "channel", or "manufacturer" (the closing summary)
    kind: str  # "price" | "margin"
    value: float
    reason: str


@dataclass(frozen=True)
class NegotiationResult:
    """The settled deal: the full trace plus the final per-unit and total margin."""

    rows: list[QuoteRow]
    buy_price: float  # EUR/unit the manufacturer pays the supplier (round-2 quote)
    sell_price: float  # EUR/unit the channel pays the manufacturer (round-2 offer)
    unit_margin: float  # sell_price − buy_price
    total_margin: float  # unit_margin × quantity


def _supplier_materials_cost(product: Product, supplier: Supplier, month_index: int,
                             factors: dict[str, list[MonthForecast]]) -> float:
    """Per-unit materials cost repriced at one supplier's per-component factors.

    Takes the median physical cost split by component (clay / power / gas /
    shipping) and multiplies each by that supplier's factor for the component, so
    a supplier cheap on clay but dear on shipping lands at the right total."""
    components = component_unit_costs(product, factors, month_index)
    return sum(cost * supplier.factor(component) for component, cost in components.items())


def _market_materials_cost(product: Product, month_index: int,
                           factors: dict[str, list[MonthForecast]]) -> float:
    """Per-unit materials cost at the 1.0 baseline factors — the market reference
    the channel's sell price is anchored to (supplier-independent)."""
    return sum(component_unit_costs(product, factors, month_index).values())


def negotiate(
    product: Product,
    supplier: Supplier,
    channel: SalesChannel,
    factors: dict[str, list[MonthForecast]],
    target_month: str,
    quantity: int,
    timeline_days: int,
    competition: str,
    *,
    month_index: int = 0,
) -> NegotiationResult:
    """Run the deterministic two-round negotiation and return the settled deal.

    ``factors`` is the 4-factor forecast; ``month_index`` selects which forecast
    month prices the materials (default 0 = nearest future, the cost basis the
    recommendation anchors on). ``target_month`` ("YYYY-MM") sets the demand
    quarter; ``competition`` is the user's market read ("high"/"medium"/"low")."""
    rows: list[QuoteRow] = []

    # --- Round 1: opening positions ---------------------------------------- #
    # Buy side: this supplier's repriced materials + the supplier's own margin.
    materials = _supplier_materials_cost(product, supplier, month_index, factors)
    supplier_open = materials * (1.0 + SUPPLIER_BASE_MARGIN)
    rows.append(
        QuoteRow(
            1, "supplier", "price", supplier_open,
            f"opening quote: €{materials:.2f} materials × (1 + {SUPPLIER_BASE_MARGIN:.0%} supplier margin)",
        )
    )

    # Sell side: anchored to the *market* production cost (1.0-baseline materials +
    # a typical maker margin), marked up by the channel's target margin — so the
    # sell price does not depend on which supplier was picked.
    reference_cost = _market_materials_cost(product, month_index, factors) * (1.0 + SUPPLIER_BASE_MARGIN)
    channel_open = reference_cost * (1.0 + channel.target_margin)
    rows.append(
        QuoteRow(
            1, "channel", "price", channel_open,
            f"opening offer: €{reference_cost:.2f} market cost × (1 + {channel.target_margin:.0%} channel margin)",
        )
    )

    # --- Round 2: counters -------------------------------------------------- #
    quarter = quarter_of(target_month)
    season = channel.season_factor(quarter)
    demand_hot = season >= HIGH_DEMAND_SEASON
    is_rush = timeline_days < RUSH_TIMELINE_DAYS

    supplier_premium = (DEMAND_PREMIUM if demand_hot else 0.0) + (RUSH_PREMIUM if is_rush else 0.0)
    supplier_final = supplier_open * (1.0 + supplier_premium)
    if supplier_premium > 0:
        drivers = []
        if demand_hot:
            drivers.append(f"+{DEMAND_PREMIUM:.0%} hot demand ({quarter} ×{season:.2f})")
        if is_rush:
            drivers.append(f"+{RUSH_PREMIUM:.0%} rush (<{RUSH_TIMELINE_DAYS}d lead)")
        supplier_reason = "counters up " + ", ".join(drivers)
    else:
        supplier_reason = "holds opening quote (normal demand, standard lead time)"
    rows.append(QuoteRow(2, "supplier", "price", supplier_final, supplier_reason))

    competition_key = competition.strip().lower()
    discount = COMPETITION_DISCOUNT.get(competition_key, 0.0)
    channel_final = channel_open * (1.0 - discount)
    if discount > 0:
        channel_reason = f"counters down −{discount:.0%} ({competition_key} competition for shelf space)"
    else:
        channel_reason = "holds opening offer (low competition)"
    rows.append(QuoteRow(2, "channel", "price", channel_final, channel_reason))

    # --- Settle ------------------------------------------------------------- #
    buy_price = supplier_final
    sell_price = channel_final
    unit_margin = sell_price - buy_price
    total_margin = unit_margin * quantity
    margin_fraction = unit_margin / buy_price if buy_price else 0.0
    rows.append(
        QuoteRow(
            2, "manufacturer", "margin", margin_fraction,
            f"unit margin €{unit_margin:.2f} = sell €{sell_price:.2f} − buy €{buy_price:.2f}",
        )
    )

    return NegotiationResult(
        rows=rows,
        buy_price=buy_price,
        sell_price=sell_price,
        unit_margin=unit_margin,
        total_margin=total_margin,
    )
