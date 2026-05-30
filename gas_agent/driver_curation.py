"""Curation of Sybilion's ranked drivers — the agent's substantive edge.

Sybilion returns the external series it found most predictive of the gas price,
each with an importance score. The catch: a high importance score only says a
series *correlates* in-sample, not that it can *cause* gas to move. The ranking
duly mixes genuinely causal energy/markets series ("Exports of Natural gas in
Europe", "Exchange Rates - World") with high-scoring nonsense ("Population -
Sri Lanka") — classic spurious correlation.

This module is the filter that separates them. It is **deterministic**: a driver
is judged purely by reading its name against an energy/markets whitelist and a
demographic-proxy blacklist. No model decides what is credible — the same driver
name always gets the same verdict, and every verdict carries a one-line reason
so the curation is fully inspectable in the dashboard.

If too few credible drivers survive (the forecast filters were too narrow or
off-topic), :attr:`CurationResult.needs_refine` flips on — the signal the
keyword agent uses to widen its filters and re-pull. That is the closed loop.
"""

from __future__ import annotations

from dataclasses import dataclass

# The theme a driver matches when it traces to a geopolitical / market-risk signal
# (Brent-style risk premia, the VIX, named conflicts). Defined as a module constant
# because the scenario engine imports it by name to compute the standing supply-risk
# premium — keeping the two modules in lockstep instead of duplicating a string.
GLOBAL_RISK_THEME = "global risk & volatility"

# --------------------------------------------------------------------------- #
# What counts as a credible European-gas driver, by theme.
# Ordered most-specific first so the *cited* theme is the meaningful one
# (first match wins). Matching is case-insensitive substring.
# --------------------------------------------------------------------------- #
CREDIBLE_THEMES: dict[str, tuple[str, ...]] = {
    "natural gas": ("natural gas", "extraction of natural gas"),
    "LNG": ("liquefied natural gas",),
    "petroleum gas (LPG)": ("liquefied petroleum", "petroleum gas"),
    "oil & petroleum products": ("petroleum", "gasoil", "crude", "oil", "brent"),
    "electricity & power": ("electricity", "power"),
    "energy prices & benchmarks": ("energy",),
    "coal & carbon": ("coal", "carbon", "emission"),
    "extraction & mining": ("mining", "quarrying", "extraction"),
    "exchange rates (FX)": ("exchange rate",),
    "interest rates": ("interest rate",),
    # Geopolitical / volatility risk and macro demand indicators. Placed after the
    # energy themes (most-specific-first) so an energy series still cites its energy
    # theme; these catch the risk/macro series Sybilion may surface in future pulls.
    GLOBAL_RISK_THEME: (
        "global risk", "geopolitical", "volatility index", "vix", "risk index", "conflict",
    ),
    "macro indicators": ("purchasing managers", "pmi", "inflation", "consumer price", "cpi"),
    "producer & import prices": ("producer price", "import price"),
    "stocks & storage": ("stock level", "stocks of", "storage"),
    "commodities": ("commodit",),
}

# Energy/commodity terms that make an import/export/trade series an *energy*
# trade flow rather than generic trade.
ENERGY_TRADE_TERMS: tuple[str, ...] = (
    "gas", "oil", "petroleum", "lng", "lpg", "gasoil", "energy",
    "electricity", "coal", "carbon", "raw material", "commodit", "crude", "fuel",
)

# Demographic / unrelated proxies — high in-sample correlation, no causal path
# to the gas price. These are the rejections that prove curation is doing work.
SPURIOUS_KEYWORDS: tuple[str, ...] = (
    "population", "fertility", "birth rate", "mortality", "life expectancy",
    "traffic accident", "tourism", "accommodation",
)

# Region names we recognise, for display only (longest first so
# "United States of America" wins over "United States", etc.).
_REGION_DISPLAY_NAMES: tuple[str, ...] = (
    "United States of America", "Russian Federation", "European Union",
    "United Kingdom", "Netherlands", "Sri Lanka", "Bangladesh", "Czechia",
    "Belgium", "Belarus", "Germany", "Hungary", "Algeria", "Ukraine",
    "Austria", "Serbia", "France", "Poland", "Norway", "Turkey", "Europe",
    "Russia", "Italy", "Spain", "Qatar", "World", "Iran", "euro area",
)


@dataclass(frozen=True)
class CurationParams:
    """Tunable knobs for curation."""

    min_kept_drivers: int = 8  # below this, ask the keyword agent to widen filters
    top_n_for_explanation: int = 6  # how many kept drivers the explainer gets


DEFAULT_CURATION_PARAMS = CurationParams()


@dataclass(frozen=True)
class CuratedDriver:
    """One ranked driver after curation, with the verdict and its reason."""

    name: str
    importance: float  # Sybilion's overall mean importance (~0-100)
    correlation: float  # overall mean Pearson correlation with the gas price
    region: str  # parsed from the name, display only ("" if unknown)
    theme: str  # the credible theme it matched ("" when rejected)
    verdict: str  # "keep" | "reject"
    reason: str  # one-line human-readable justification


@dataclass(frozen=True)
class CurationResult:
    """The outcome of curating one driver ranking."""

    kept: list[CuratedDriver]  # sorted by importance, descending
    rejected: list[CuratedDriver]  # sorted by importance, descending
    min_kept_drivers: int

    @property
    def kept_count(self) -> int:
        return len(self.kept)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    @property
    def needs_refine(self) -> bool:
        """True when too few credible drivers survived — the closed-loop trigger
        telling the keyword agent its filters were too narrow or off-topic."""
        return self.kept_count < self.min_kept_drivers

    def top_drivers(self, count: int, unique_names: bool = True) -> list[CuratedDriver]:
        """The strongest kept drivers, for handing to the explanation agent."""
        chosen: list[CuratedDriver] = []
        seen: set[str] = set()
        for driver in self.kept:  # already sorted by importance
            if unique_names and driver.name in seen:
                continue
            seen.add(driver.name)
            chosen.append(driver)
            if len(chosen) >= count:
                break
        return chosen


def extract_region(driver_name: str) -> str:
    """Best-effort parse of the region from a driver name (display only)."""
    lower = driver_name.lower()
    for display_name in _REGION_DISPLAY_NAMES:
        if display_name.lower() in lower:
            return display_name
    return ""


def classify_driver(driver_name: str) -> tuple[str, str, str]:
    """Judge one driver by its name. Returns ``(verdict, theme, reason)``."""
    lower = driver_name.lower()

    # 1. Demographic / unrelated proxies — the headline rejection.
    for keyword in SPURIOUS_KEYWORDS:
        if keyword in lower:
            return (
                "reject",
                "",
                f"demographic proxy ('{keyword}') — high in-sample correlation but "
                "no causal link to the gas price",
            )

    # 2. Direct energy / markets themes.
    for theme, keywords in CREDIBLE_THEMES.items():
        for keyword in keywords:
            if keyword in lower:
                return "keep", theme, f"credible {theme} driver"

    # 3. Energy trade flows: import/export/trade that names an energy commodity,
    #    or an aggregate trade index (a macro demand proxy).
    if any(word in lower for word in ("import", "export", "trade")):
        for term in ENERGY_TRADE_TERMS:
            if term in lower:
                return "keep", "energy trade flow", "credible energy trade-flow driver"
        if any(word in lower for word in ("index", "indices", "value, volume")):
            return (
                "keep",
                "trade index (macro demand proxy)",
                "aggregate trade index — weak but plausible macro demand proxy",
            )

    # 4. Everything else falls outside the European-gas whitelist.
    return (
        "reject",
        "",
        "outside the energy/markets whitelist for European gas",
    )


def _overall_mean(block: dict, default: float = 0.0) -> float:
    overall = block.get("overall") if isinstance(block, dict) else None
    if isinstance(overall, dict) and "mean" in overall:
        return float(overall["mean"])
    return default


def parse_drivers(external_signals_json: dict) -> list[CuratedDriver]:
    """Read external_signals.json into curated drivers (unsorted)."""
    data = external_signals_json.get("data", external_signals_json)
    drivers: list[CuratedDriver] = []
    for entry in data.values():
        if not isinstance(entry, dict) or "driver_name" not in entry:
            continue
        name = entry["driver_name"]
        importance = _overall_mean(entry.get("importance", {}))
        correlation = _overall_mean(entry.get("pearson_correlation", {}))
        verdict, theme, reason = classify_driver(name)
        drivers.append(
            CuratedDriver(
                name=name,
                importance=importance,
                correlation=correlation,
                region=extract_region(name),
                theme=theme,
                verdict=verdict,
                reason=reason,
            )
        )
    return drivers


def _dedupe_by_name(drivers: list[CuratedDriver]) -> list[CuratedDriver]:
    """Sybilion can return the same series more than once; keep the
    highest-importance instance of each name (input must be sorted desc)."""
    seen: set[str] = set()
    unique: list[CuratedDriver] = []
    for driver in drivers:
        if driver.name in seen:
            continue
        seen.add(driver.name)
        unique.append(driver)
    return unique


def curate_drivers(
    external_signals_json: dict,
    params: CurationParams = DEFAULT_CURATION_PARAMS,
) -> CurationResult:
    """Curate a Sybilion driver ranking into kept vs rejected *distinct* drivers."""
    drivers = parse_drivers(external_signals_json)
    by_importance = _dedupe_by_name(sorted(drivers, key=lambda d: d.importance, reverse=True))
    kept = [d for d in by_importance if d.verdict == "keep"]
    rejected = [d for d in by_importance if d.verdict == "reject"]
    return CurationResult(kept=kept, rejected=rejected, min_kept_drivers=params.min_kept_drivers)
