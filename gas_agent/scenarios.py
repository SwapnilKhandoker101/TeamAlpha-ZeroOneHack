"""The scenario library — match a company profile to a pre-fetched real forecast (W17).

A live Sybilion forecast can take ~11 minutes, which is too slow for a live stage. So we
pre-fetch **real** Sybilion forecasts for the whole input space ahead of time, commit them
under ``scenarios/`` (a clean checkout ships them, offline-reproducible), and at demo time
retrieve the nearest match — instant, and genuinely real data.

**The key insight that makes "cover everything" cheap:** a forecast depends only on
**(product × gas_exposure)** — that pair flavours the persona → Sybilion driver/filter pick
(gas) and the shared 4-factor cost forecast is product-independent (the BOM differs per
product, but that's deterministic *downstream* math). Competition, quantity and timeline never
change a forecast — they're applied deterministically downstream. So a **9-cell grid (3
products × 3 gas-exposures) of gas forecasts + one shared ceramics forecast covers the entire
input space exactly**, and matching is an exact lookup on those two dims (with a graceful
nearest-fallback while the library is still being populated).

This module is pure (no Streamlit, no network) so the matcher is unit-testable; the artifacts
themselves are read through ``gas_agent.sybilion_client``'s cache→scenarios resolver, so a
matched scenario flows through the existing render path unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from gas_agent import config

INDEX_PATH = config.SCENARIOS_DIR / "index.json"
CERAMICS_SHARED_SLUG = "_shared"  # the product-independent 4-factor ceramics forecast

# The forecast-relevant dimensions (everything else is deterministic downstream).
PRODUCTS: tuple[str, ...] = ("bowl", "dinnerware", "tile")
GAS_EXPOSURES: tuple[str, ...] = ("low", "medium", "high")
_EXPOSURE_RANK = {"low": 0, "medium": 1, "high": 2}
_CROSS_PRODUCT_PENALTY = 10  # a same-product, dearer-exposure match always beats a cross-product one


@dataclass(frozen=True)
class Scenario:
    """One committed library cell: a real gas forecast for a (product, gas_exposure)."""

    product: str
    gas_exposure: str
    slug: str  # the gas job dir under scenarios/
    ceramics_ref: str = CERAMICS_SHARED_SLUG  # the shared 4-factor ceramics forecast dir
    label: str = ""


@dataclass(frozen=True)
class ScenarioMatch:
    """The retrieval result: which scenario, and how close it is to the request."""

    scenario: Scenario
    exact: bool  # product AND gas_exposure both matched
    distance: int  # 0 = exact; larger = further (exposure steps, +penalty across products)


def slug_for(product: str, gas_exposure: str) -> str:
    """The canonical library slug for a forecast cell."""
    return f"scn-{product}-{gas_exposure}"


def load_index(path=INDEX_PATH) -> list[Scenario]:
    """Load the committed scenario index (``[]`` when the library isn't built yet)."""
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    return [Scenario(**entry) for entry in raw.get("scenarios", [])]


def match(
    product: str,
    gas_exposure: str,
    scenarios: list[Scenario] | None = None,
) -> ScenarioMatch | None:
    """Deterministic nearest-match for a (product, gas_exposure) request.

    Prefers the same product; within it, the nearest gas-exposure. If the product has no
    cell yet, falls back to the nearest gas-exposure of any product (so any request maps to
    *something* real while the library is still filling in). ``None`` only when the library
    is empty."""
    pool = scenarios if scenarios is not None else load_index()
    if not pool:
        return None
    target = _EXPOSURE_RANK.get(gas_exposure, 1)

    def distance(scenario: Scenario) -> int:
        steps = abs(_EXPOSURE_RANK.get(scenario.gas_exposure, 1) - target)
        return steps + (0 if scenario.product == product else _CROSS_PRODUCT_PENALTY)

    # min() is stable (keeps the first of equal-distance cells → deterministic ties).
    best = min(pool, key=distance)
    dist = distance(best)
    return ScenarioMatch(
        scenario=best,
        exact=best.product == product and best.gas_exposure == gas_exposure,
        distance=dist,
    )


def nearest_options(product: str, scenarios: list[Scenario] | None = None, limit: int = 3) -> list[Scenario]:
    """A few ready scenarios to suggest when there's no confident match — same product
    first, then the rest, capped at ``limit`` (for the 'try one of these' chips)."""
    pool = scenarios if scenarios is not None else load_index()
    same = [s for s in pool if s.product == product]
    others = [s for s in pool if s.product != product]
    return (same + others)[:limit]
