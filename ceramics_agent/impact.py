"""The unified driver-impact map — *what moves what*, across BOTH decisions (W4).

A pure, Streamlit-free formatter. It turns the gas agent's curated drivers and the
ceramics agent's per-factor drivers into one flat list of rows, each mapping a driver
to the factor it explains, its Sybilion importance, **which decision it feeds**, and
**which way that factor's forecast is heading** — so the app can render a single
"what moves what" table over evidence that already exists.

**No decision math lives here.** The hedge ratio and the lock % are decided elsewhere
(``gas_agent.hedge_policy`` / ``ceramics_agent.cost_policy``); this module only
*describes* the evidence behind them. The "direction" is a plain read of a factor's
median trajectory over the horizon — the same first-vs-last comparison the cost chart
draws — never a re-decision. THE RULE holds: this is formatting, not deciding.

It lives in ``ceramics_agent`` because that is the package allowed to import
``gas_agent`` (the dependency arrow is ceramics → gas); the app is the only other
importer. Gas modules stay byte-identical.
"""

from __future__ import annotations

from dataclasses import dataclass

from gas_agent.driver_curation import CurationResult
from gas_agent.hedge_policy import MonthForecast

from ceramics_agent.cost_policy import CostWeights
from ceramics_agent.forecast import FACTORS, FactorDriver

# Series name → the business cost bucket a judge recognises. Kept here so the panel
# and any future export share one mapping.
FACTOR_LABELS: dict[str, str] = {
    "gas": "Kiln gas",
    "clay": "Clay & kaolin",
    "power": "Grid power",
    "shipping": "Freight",
}

# How each decision is referred to in the "feeds" column (the circled numbers match
# the two section headers on the page).
GAS_DECISION = "① Gas hedge"
CERAMICS_DECISION = "② Ceramics lock"

# Below this relative move over the horizon a factor reads as flat (not rising/easing).
_FLAT_BAND = 0.02


@dataclass(frozen=True)
class ImpactRow:
    """One driver's line in the cross-decision map."""

    driver: str  # the external series' name
    explains: str  # the factor / price series it informs
    importance: float  # Sybilion importance, 0..100
    feeds: str  # which decision it feeds, and how
    direction: str  # the factor's forecast outlook (arrow + words)
    agent: str  # "gas" | "ceramics" — for filtering / styling


def _trend_direction(months: list[MonthForecast]) -> str:
    """Read a factor's median trajectory (first → last) into a short outlook label.

    A pure read of the forecast — the same first-vs-last comparison the cost-band
    chart uses — so it never touches a decision number."""
    medians = [m.median for m in months]
    if len(medians) < 2 or medians[0] <= 0:
        return "→ flat"
    change = (medians[-1] - medians[0]) / medians[0]
    if change > _FLAT_BAND:
        return f"↑ rising {change:+.0%}"
    if change < -_FLAT_BAND:
        return f"↓ easing {change:+.0%}"
    return "→ flat"


def gas_impact_rows(
    curation: CurationResult, gas_months: list[MonthForecast]
) -> list[ImpactRow]:
    """One row per **kept** gas driver — it informs the TTF price band the hedge %
    is sized from. (Rejected drivers are shown in the gas section's curation view;
    this overview carries only the evidence the decision actually trusts.)"""
    outlook = _trend_direction(gas_months)
    return [
        ImpactRow(
            driver=d.name,
            explains="TTF gas price",
            importance=round(d.importance, 0),
            feeds=f"{GAS_DECISION} — sizes the forecast price band",
            direction=outlook,
            agent="gas",
        )
        for d in curation.kept
    ]


def ceramics_impact_rows(
    drivers_by_factor: dict[str, list[FactorDriver]],
    factors: dict[str, list[MonthForecast]],
    weights: CostWeights,
) -> list[ImpactRow]:
    """One row per ceramics factor driver — it informs its factor's band, which enters
    the blended cost band **weighted by the user's cost mix**, and that band sets the
    lock %. The weight shown is the normalized weight for the driver's factor."""
    factor_weights = weights.as_factor_weights()
    rows: list[ImpactRow] = []
    for factor in FACTORS:
        outlook = _trend_direction(factors.get(factor, []))
        weight = factor_weights.get(factor, 0.0)
        label = FACTOR_LABELS.get(factor, factor)
        for d in drivers_by_factor.get(factor, []):
            rows.append(
                ImpactRow(
                    driver=d.name,
                    explains=label,
                    importance=round(d.importance, 0),
                    feeds=f"{CERAMICS_DECISION} — {weight:.0%} of the blended cost band",
                    direction=outlook,
                    agent="ceramics",
                )
            )
    return rows


def combined_impact_rows(
    gas_rows: list[ImpactRow],
    ceramics_rows: list[ImpactRow],
    *,
    gas_limit: int | None = None,
) -> list[ImpactRow]:
    """The cross-decision map: gas rows first (the headline cost), then ceramics rows,
    each sorted by importance descending. ``gas_limit`` caps the gas rows so the table
    stays scannable — the gas section below shows the full curation — while every
    ceramics factor driver is kept (there are only a handful)."""
    gas_sorted = sorted(gas_rows, key=lambda r: r.importance, reverse=True)
    if gas_limit is not None:
        gas_sorted = gas_sorted[:gas_limit]
    ceramics_sorted = sorted(ceramics_rows, key=lambda r: r.importance, reverse=True)
    return gas_sorted + ceramics_sorted
