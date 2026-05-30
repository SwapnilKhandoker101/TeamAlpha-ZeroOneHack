"""Geography for the driver globe — where each driver's country sits, and a
one-paragraph "why it matters" brief for it.

The curation step already split Sybilion's ranked drivers into the credible
energy/markets series and the spurious demographic proxies. This module pins each
driver's country onto the globe so that split becomes *spatial*: the suppliers and
hubs that genuinely move European gas rise as green columns (taller = more kept
importance), while the countries that only showed up through spurious correlations
(Sri Lanka, Bangladesh, Serbia, Belarus) sit as flat red markers. One look says
"the agent trusts the gas routes and threw out the population proxies."

Coordinates are representative points (a capital or a gas-relevant hub), good
enough to read on a sphere — not survey-grade. Pan-regional aggregates ("Europe",
"World") have no single point: Europe-style aggregates are pinned to a central-
Europe centroid so they still show, and truly global ones are dropped.

The per-country brief is the one place an LLM speaks here, and it only *explains*
("why does Norway move European gas?") — it never touches the hedge ratio.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gas_agent.driver_curation import CurationResult

# Representative (lat, lon) per region display name (as produced by
# driver_curation.extract_region). Capitals or gas-relevant hubs.
REGION_COORDS: dict[str, tuple[float, float]] = {
    "United States of America": (39.8, -98.6),
    "Russian Federation": (55.75, 37.62),  # Moscow — anchors near the European gas routes
    "Russia": (55.75, 37.62),
    "United Kingdom": (52.5, -1.5),
    "Netherlands": (52.13, 5.29),  # the TTF hub itself
    "Germany": (51.16, 10.45),
    "Belgium": (50.85, 4.35),  # Zeebrugge LNG
    "France": (46.6, 2.2),
    "Poland": (51.9, 19.1),
    "Austria": (47.6, 14.1),  # Baumgarten hub
    "Hungary": (47.16, 19.5),
    "Czechia": (49.8, 15.5),
    "Italy": (41.9, 12.5),
    "Spain": (40.0, -3.7),
    "Norway": (60.5, 8.5),  # the EU's top pipeline supplier
    "Ukraine": (48.4, 31.2),  # transit country
    "Turkey": (39.0, 35.2),  # TurkStream
    "Algeria": (28.0, 2.6),  # Transmed / Medgaz pipelines
    "Qatar": (25.3, 51.2),  # LNG shipped through the Strait of Hormuz
    "Iran": (32.4, 53.7),  # the Strait of Hormuz sits off its coast
    "Serbia": (44.0, 21.0),
    "Belarus": (53.7, 27.95),
    "Sri Lanka": (7.87, 80.77),
    "Bangladesh": (23.7, 90.35),
}

# Pan-European aggregates with no single point — pin to a central-Europe centroid
# so they still appear rather than vanishing.
EUROPE_AGGREGATES: frozenset[str] = frozenset({"Europe", "European Union", "euro area"})
_EUROPE_CENTROID: tuple[float, float] = (50.5, 9.0)

# Regions we cannot meaningfully place on a sphere — dropped from the globe.
OMIT_REGIONS: frozenset[str] = frozenset({"World", ""})


def coords_for(region: str) -> tuple[float, float] | None:
    """(lat, lon) for a region display name, or ``None`` if it can't be placed."""
    if region in OMIT_REGIONS:
        return None
    if region in EUROPE_AGGREGATES:
        return _EUROPE_CENTROID
    return REGION_COORDS.get(region)


@dataclass
class CountryAggregate:
    """All drivers that resolved to one country, split kept vs rejected."""

    region: str
    lat: float
    lon: float
    kept_importance: float = 0.0  # summed importance of kept drivers — the column height
    kept_names: list[str] = field(default_factory=list)
    rejected_names: list[str] = field(default_factory=list)

    @property
    def kept_count(self) -> int:
        return len(self.kept_names)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected_names)

    @property
    def has_kept(self) -> bool:
        """A country with at least one credible driver — drawn as a green column."""
        return self.kept_count > 0

    @property
    def rejected_only(self) -> bool:
        """Only spurious drivers landed here — drawn as a flat red marker."""
        return self.kept_count == 0 and self.rejected_count > 0


def aggregate_drivers(curation: CurationResult) -> list[CountryAggregate]:
    """Group curated drivers by placeable country, summing kept importance.

    Co-located aggregates (the Europe-style names) merge onto the centroid, so
    "Europe" and "European Union" share one column rather than stacking. Drivers
    whose region can't be placed are silently skipped — they still live in the
    curation table, just not on the sphere.
    """
    by_region: dict[tuple[float, float], CountryAggregate] = {}

    def bucket(region: str) -> CountryAggregate | None:
        point = coords_for(region)
        if point is None:
            return None
        existing = by_region.get(point)
        if existing is None:
            display = "Europe (aggregate)" if region in EUROPE_AGGREGATES else region
            existing = CountryAggregate(region=display, lat=point[0], lon=point[1])
            by_region[point] = existing
        return existing

    for driver in curation.kept:
        target = bucket(driver.region)
        if target is not None:
            target.kept_importance += max(0.0, driver.importance)
            target.kept_names.append(driver.name)

    for driver in curation.rejected:
        target = bucket(driver.region)
        if target is not None:
            target.rejected_names.append(driver.name)

    # Tallest columns first — stable, and handy for any "top supplier" readout.
    return sorted(by_region.values(), key=lambda c: c.kept_importance, reverse=True)


# --------------------------------------------------------------------------- #
# Per-country "why it matters" brief (explanation only — never a decision).
# --------------------------------------------------------------------------- #
_BRIEF_SYSTEM = (
    "You are a concise European energy-markets analyst. In at most 3 sentences, "
    "explain why the given country's economy, exports or infrastructure influences "
    "the European natural-gas (TTF) price — be concrete about pipelines, LNG, "
    "storage or demand. Do NOT give hedging advice, do NOT mention a hedge ratio, "
    "and do NOT tell the reader what to do; only explain the linkage."
)

_SPURIOUS_SYSTEM = (
    "You are a careful European energy-markets analyst explaining a data-quality "
    "decision. In at most 2 sentences, explain why a statistical series from the "
    "given country (e.g. population or other demographic data) can correlate with "
    "the European gas price in-sample yet have no credible causal link, so a hedging "
    "agent is right to drop it. Do NOT give hedging advice or mention a hedge ratio."
)

# Deterministic fallbacks so the panel always says something useful offline.
_FALLBACK_BRIEFS: dict[str, str] = {
    "Norway": "Norway is the EU's largest pipeline gas supplier since 2022, so its "
              "output and field maintenance move European prices directly.",
    "Russian Federation": "Russia was the dominant pipeline supplier to Europe; "
                           "sanctions and flow cuts since 2022 are a primary driver of TTF volatility.",
    "Netherlands": "The Dutch TTF hub is the European gas benchmark itself, and Dutch "
                   "storage and LNG terminals set the marginal price.",
    "Qatar": "Qatar is a top global LNG exporter whose cargoes — shipped through the "
             "Strait of Hormuz — compete for the same European demand.",
    "Iran": "Iran sits on the Strait of Hormuz, the chokepoint for a large share of "
            "seaborne LNG and oil; disruption there spikes global energy risk premia.",
    "Algeria": "Algeria supplies southern Europe via the Transmed and Medgaz pipelines, "
               "so its exports shape Mediterranean gas balances.",
    "Germany": "Germany is Europe's largest gas consumer; its industrial demand and "
               "storage levels are a core signal for TTF.",
}


def country_brief(
    region: str,
    kept_names: list[str] | None = None,
    credible: bool = True,
) -> tuple[str, str]:
    """Return ``(text, source)`` for ``region``.

    When ``credible`` is true the brief explains why the country *moves* European
    gas; when false it explains why its series was *dropped* as a spurious correlate.
    Tries Featherless (grounded on the driver names); falls back to a static
    one-liner — or a generic templated sentence — if the model is unavailable.
    """
    try:
        from gas_agent import config, llm

        drivers = ", ".join(kept_names or []) or "(none)"
        system = _BRIEF_SYSTEM if credible else _SPURIOUS_SYSTEM
        prefix = "kept here" if credible else "dropped here"
        text = llm.chat_text(
            model=config.EXPLANATION_MODEL,
            system=system,
            user=f"Country/region: {region}. Sybilion driver series {prefix}: {drivers}.",
            temperature=0.3,
            max_tokens=160,
        )
        if text:
            return text, "llm"
    except Exception:
        pass

    if not credible:
        return (
            f"{region} only entered the ranking through demographic-style series that "
            "track the gas price by coincidence, not cause — so the agent drops them.",
            "fallback",
        )
    fallback = _FALLBACK_BRIEFS.get(region)
    if fallback:
        return fallback, "fallback"
    return (
        f"{region} appears in Sybilion's driver ranking for European gas; the agent "
        "keeps it only where there is a plausible supply, trade or demand linkage.",
        "fallback",
    )
