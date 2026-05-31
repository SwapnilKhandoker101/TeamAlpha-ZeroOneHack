"""Geography for the ceramics globe — *where to buy* and *where to sell* (W5).

The gas globe turns the curated drivers into a spatial story (trusted suppliers rise
as green columns, spurious proxies sit as flat red markers). This module does the
same for the ceramics decision, but with **two switchable layers** over the same
sphere:

* **Where to buy** — the kept suppliers, placed at their sourcing country (Austria /
  Poland / Italy), each column sized by its **curation score**. One look says which
  countries the agent would source from, and how strongly.
* **Where to sell** — the demand markets, placed where each sales channel's buyers
  sit (EU core for wholesale, DACH for hospitality, global hubs for online export),
  each column sized by **demand potential** (historical channel mix × the target
  quarter's seasonality × the channel's margin).

Both layers are emitted as :class:`gas_agent.geo.CountryAggregate` points so the same
orthographic-globe renderer draws them unchanged — the engine is reused, only the
data differs. Coordinates come from the gas module's shared table (capitals / hubs),
with a tiny local supplement for markets it doesn't carry.

**No decision math lives here.** Scores and demand weights are computed upstream
(``curation`` / ``catalog``); this module only *places* them. The per-country brief
is the one place an LLM speaks, and it only *explains* why a country is a credible
source or a strong market — it never touches the lock %, a score, or a quote.

It lives in ``ceramics_agent`` because that is the package allowed to import
``gas_agent`` (the dependency arrow is ceramics → gas). Gas modules stay byte-identical.
"""

from __future__ import annotations

from gas_agent.geo import CountryAggregate, coords_for as _gas_coords_for

from ceramics_agent.catalog import (
    CHANNEL_DEMAND_REGIONS,
    SUPPLIER_REGION_NAMES,
    SalesChannel,
    average_channel_mix,
    list_channels,
    quarter_of,
)
from ceramics_agent.curation import SupplierCuration

# Markets the gas coordinate table doesn't carry (it is gas-supplier focused).
# Representative points, good enough to read on a sphere.
_EXTRA_COORDS: dict[str, tuple[float, float]] = {
    "Switzerland": (46.8, 8.2),
}


def coords_for(region: str) -> tuple[float, float] | None:
    """(lat, lon) for a region, reusing the gas table first then the local supplement."""
    point = _gas_coords_for(region)
    if point is not None:
        return point
    return _EXTRA_COORDS.get(region)


def supplier_sourcing_points(
    supplier_curation: SupplierCuration,
    chosen_name: str | None = None,
) -> list[CountryAggregate]:
    """Place each **kept** supplier at its sourcing country, sized by curation score.

    Rejected (off-domain) suppliers are omitted — like the gas globe, this layer shows
    only the sources the agent trusts. Suppliers that share a country merge onto one
    column (the higher score wins the height); the chosen supplier is flagged in its
    label so the renderer / brief can call it out."""
    by_region: dict[str, CountryAggregate] = {}
    for curated in supplier_curation.kept:
        region = SUPPLIER_REGION_NAMES.get(curated.supplier.region, curated.supplier.region)
        point = coords_for(region)
        if point is None:
            continue
        label = curated.supplier.name + (" ✓ chosen" if curated.supplier.name == chosen_name else "")
        existing = by_region.get(region)
        if existing is None:
            by_region[region] = CountryAggregate(
                region=region, lat=point[0], lon=point[1],
                kept_importance=round(curated.total_score, 1), kept_names=[label],
            )
        else:
            existing.kept_names.append(label)
            existing.kept_importance = max(existing.kept_importance, round(curated.total_score, 1))
    return sorted(by_region.values(), key=lambda c: c.kept_importance, reverse=True)


def demand_market_points(
    target_month: str,
    channels: list[SalesChannel] | None = None,
) -> list[CountryAggregate]:
    """Place demand on the globe: each channel's potential, split across its markets.

    Demand potential for a channel = its mean historical share × the target quarter's
    seasonality × its margin; that potential is divided across the channel's
    destination markets by the catalog weights, and markets shared by several channels
    sum. Heights are scaled so the strongest market reads 100 (the same 0..100 scale as
    the supplier scores), keeping the two layers visually comparable."""
    chans = channels if channels is not None else list_channels()
    avg_mix = average_channel_mix()
    quarter = quarter_of(target_month) if target_month else "Q1"

    raw: dict[str, float] = {}
    names: dict[str, list[str]] = {}
    coords: dict[str, tuple[float, float]] = {}
    for channel in chans:
        regions = CHANNEL_DEMAND_REGIONS.get(channel.id)
        if not regions:
            continue
        potential = avg_mix.get(channel.id, 0.0) * channel.season_factor(quarter) * channel.target_margin
        for region, weight in regions:
            point = coords_for(region)
            if point is None:
                continue
            raw[region] = raw.get(region, 0.0) + potential * weight
            names.setdefault(region, []).append(channel.name)
            coords[region] = point

    top = max(raw.values(), default=0.0)
    if top <= 0:
        return []
    points = [
        CountryAggregate(
            region=region, lat=coords[region][0], lon=coords[region][1],
            kept_importance=round(100.0 * value / top, 1), kept_names=names[region],
        )
        for region, value in raw.items()
    ]
    return sorted(points, key=lambda c: c.kept_importance, reverse=True)


# --------------------------------------------------------------------------- #
# Per-country "why it matters" brief (explanation only — never a decision).
# --------------------------------------------------------------------------- #
_BUY_SYSTEM = (
    "You are a concise ceramics procurement analyst. In at most 2 sentences, explain "
    "why the given country is a credible place to source raw materials (clay, glaze, "
    "kiln supplies) for a German ceramics maker — be concrete about proximity, "
    "materials, logistics or reliability. Do NOT give a recommendation, do NOT mention "
    "a lock percentage, score or price; only explain the sourcing rationale."
)
_SELL_SYSTEM = (
    "You are a concise ceramics sales analyst. In at most 2 sentences, explain why the "
    "given country/market is a strong destination to sell handmade ceramics into — be "
    "concrete about demand, channels (wholesale, hospitality, online export) or "
    "seasonality. Do NOT give a recommendation, do NOT mention a lock percentage, score "
    "or price; only explain the demand rationale."
)

_BUY_FALLBACKS: dict[str, str] = {
    "Austria": "Austria borders the German plant, so an Austrian clay supplier offers "
               "the shortest, most reliable lead time and low inbound freight.",
    "Poland": "Poland offers the cheapest raw clay and minerals in the region, trading "
              "a longer lead time and dearer logistics for the lowest material cost.",
    "Italy": "Italy is the European centre for premium glazes and kiln furniture, the "
             "most reliable source where colour and finish quality matter most.",
    "Germany": "Germany hosts the plant itself, so a domestic source minimises lead time "
               "and cross-border logistics risk.",
}
_SELL_FALLBACKS: dict[str, str] = {
    "Germany": "Germany is the home market — the deepest, most reachable demand across "
               "wholesale, hospitality and online channels.",
    "France": "France is a large neighbouring EU market for wholesale ceramics, reached "
              "without customs friction.",
    "Italy": "Italy is a design-led ceramics market that values handmade quality, a "
             "strong wholesale destination.",
    "Austria": "Austria anchors the DACH hospitality demand the restaurant-supply channel "
               "serves, close to the plant.",
    "Switzerland": "Switzerland's high-end hospitality sector pays premium margins for "
                   "handmade tableware, a small but rich DACH market.",
    "United States of America": "The US is the largest online-export market for handmade "
                                "ceramics, peaking sharply into the year-end gifting season.",
    "United Kingdom": "The UK is a major English-language online-export market for "
                      "handmade homeware, strong through the holiday quarter.",
}


def ceramics_country_brief(
    region: str,
    layer: str,
    names: list[str] | None = None,
) -> tuple[str, str]:
    """Return ``(text, source)`` explaining a country on the buy or sell layer.

    ``layer`` is ``"buy"`` (why it's a credible source) or ``"sell"`` (why it's a
    strong market). Tries Featherless (grounded on the supplier/channel names landing
    there); falls back to a static one-liner — or a generic templated sentence — when
    the model is unavailable, so the panel always says something useful offline.
    Explanation only: it never emits a lock %, score or quote (THE RULE)."""
    selling = layer == "sell"
    try:
        from gas_agent import config, llm

        detail = ", ".join(names or []) or "(none)"
        system = _SELL_SYSTEM if selling else _BUY_SYSTEM
        noun = "Sales channels serving it" if selling else "Suppliers / materials sourced here"
        text = llm.chat_text(
            model=config.EXPLANATION_MODEL,
            system=system,
            user=f"Country/region: {region}. {noun}: {detail}.",
            temperature=0.3,
            max_tokens=140,
        )
        if text:
            return text, "llm"
    except Exception:
        pass

    fallbacks = _SELL_FALLBACKS if selling else _BUY_FALLBACKS
    hit = fallbacks.get(region)
    if hit:
        return hit, "fallback"
    if selling:
        return (f"{region} is one of the destination markets the chosen sales channel "
                "serves for this production run.", "fallback")
    return (f"{region} is one of the sourcing countries a kept supplier delivers from "
            "for this production run.", "fallback")
