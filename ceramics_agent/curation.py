"""Supplier and channel curation — the ceramics edge, made auditable.

This is the ceramics analogue of :mod:`gas_agent.driver_curation`. Two jobs:

1. **Credibility** — a deterministic keep/reject verdict with a one-line reason,
   from a substring whitelist/blacklist on a supplier's specialties or a
   channel's type. A textile or food vendor is rejected as off-domain; a clay /
   glaze / kiln supplier is kept. No model decides credibility (THE RULE).

2. **Scoring & ranking** — each kept supplier and channel gets transparent
   sub-scores that combine into one total, so the dashboard can show *why* one
   ranks above another:

   * **Supplier** — cost 50% (cheapest in the set → 100, dearest → 0),
     reliability 30% (= on-time %), lead-time fit 20% (100 at exactly the
     requested timeline, falling off linearly with the gap).
   * **Channel** — margin 40% (scaled to the richest channel), seasonality 30%
     (the target month's quarter demand factor), order-fit 30% (100 when the
     quantity clears the channel's minimum, reduced below it).

Finally, :func:`select_supplier` ties the choice back to the lock %: a confident
high lock commits to the best-ranked supplier; a low, stay-flexible lock favours
the cheapest; in between, the mid-ranked one. Everything here is pure arithmetic
on the committed catalog, so the same inputs always pick the same supplier.
"""

from __future__ import annotations

from dataclasses import dataclass

from ceramics_agent.catalog import (
    CHANNEL_BLACKLIST,
    CHANNEL_WHITELIST,
    SUPPLIER_BLACKLIST,
    SUPPLIER_WHITELIST,
    SalesChannel,
    Supplier,
    list_channels,
    list_suppliers,
    quarter_of,
)

# Lock-band thresholds that route the supplier choice (match the cost-policy feel).
LOCK_HIGH = 0.60  # at/above: commit to the best-ranked supplier
LOCK_LOW = 0.30  # at/below: minimise cost exposure — pick the cheapest

# Strongest seasonality multiplier in the catalog — the scale the season score
# normalises against, so the peak channel-quarter tops out at 100.
_PEAK_SEASONALITY = 1.6


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


# --------------------------------------------------------------------------- #
# Credibility classifiers (deterministic substring rules)
# --------------------------------------------------------------------------- #
def classify_supplier(specialties: tuple[str, ...]) -> tuple[str, str]:
    """Judge a supplier by its specialties. Returns ``(verdict, reason)``.

    Blacklist hit → reject (off-domain). Whitelist hit → keep (credible ceramics
    input). Neither → keep by default (a generic materials vendor gets the benefit
    of the doubt; only a clear off-domain signal rejects)."""
    blob = " ".join(specialties).lower()
    for term in SUPPLIER_BLACKLIST:
        if term in blob:
            return "reject", f"off-domain supplier ('{term}') — not a ceramics input"
    for term in SUPPLIER_WHITELIST:
        if term in blob:
            return "keep", f"credible ceramics supplier (matched '{term}')"
    return "keep", "generic supplier kept by default — no off-domain signal"


def classify_channel(channel_type: str) -> tuple[str, str]:
    """Judge a channel by its type. Returns ``(verdict, reason)`` (same rule shape)."""
    lower = channel_type.lower()
    for term in CHANNEL_BLACKLIST:
        if term in lower:
            return "reject", f"off-domain channel ('{term}') — wrong route to market"
    for term in CHANNEL_WHITELIST:
        if term in lower:
            return "keep", f"credible sales channel (matched '{term}')"
    return "keep", "generic channel kept by default — no off-domain signal"


# --------------------------------------------------------------------------- #
# Suppliers — scoring, curation, selection
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CuratedSupplier:
    """A supplier after curation: verdict + reason + the sub-scores and total."""

    supplier: Supplier
    verdict: str
    reason: str
    cost_score: float  # 0..100, cheapest in the set = 100
    reliability_score: float  # 0..100 (= reliability_pct)
    lead_time_score: float  # 0..100, 100 at the requested timeline
    total_score: float  # 0.5·cost + 0.3·reliability + 0.2·lead

    @property
    def name(self) -> str:
        return self.supplier.name


@dataclass(frozen=True)
class SupplierCuration:
    kept: list[CuratedSupplier]  # sorted by total_score, descending
    rejected: list[CuratedSupplier]


def _cost_score(average_factor: float, cheapest: float, dearest: float) -> float:
    """Cheapest avg price factor in the set → 100, dearest → 0, linear."""
    if dearest <= cheapest:
        return 100.0
    return _clamp(100.0 * (dearest - average_factor) / (dearest - cheapest))


def _lead_time_score(lead_time_days: int, timeline_days: int) -> float:
    """100 when lead time equals the requested timeline, falling off linearly with
    the absolute gap (relative to the timeline so it scales to any horizon)."""
    horizon = max(1, timeline_days)
    return _clamp(100.0 * (1.0 - abs(lead_time_days - timeline_days) / horizon))


def curate_suppliers(
    timeline_days: int,
    suppliers: list[Supplier] | None = None,
) -> SupplierCuration:
    """Classify, score and rank suppliers for a requested delivery timeline."""
    candidates = suppliers if suppliers is not None else list_suppliers()
    classified = [(s, *classify_supplier(s.specialties)) for s in candidates]
    kept_suppliers = [s for s, verdict, _ in classified if verdict == "keep"]

    factors = [s.average_price_factor() for s in kept_suppliers]
    cheapest = min(factors) if factors else 0.0
    dearest = max(factors) if factors else 0.0

    kept: list[CuratedSupplier] = []
    rejected: list[CuratedSupplier] = []
    for supplier, verdict, reason in classified:
        if verdict == "reject":
            rejected.append(
                CuratedSupplier(supplier, verdict, reason, 0.0, 0.0, 0.0, 0.0)
            )
            continue
        cost = _cost_score(supplier.average_price_factor(), cheapest, dearest)
        reliability = _clamp(supplier.reliability_pct)
        lead = _lead_time_score(supplier.lead_time_days, timeline_days)
        total = 0.5 * cost + 0.3 * reliability + 0.2 * lead
        kept.append(
            CuratedSupplier(supplier, verdict, reason, cost, reliability, lead, total)
        )

    kept.sort(key=lambda c: c.total_score, reverse=True)
    return SupplierCuration(kept=kept, rejected=rejected)


def select_supplier(kept: list[CuratedSupplier], lock_ratio: float) -> CuratedSupplier | None:
    """Pick a supplier given the lock %.

    A confident, high lock (≥ ``LOCK_HIGH``) commits volume, so take the
    best-ranked supplier (reliability + balance matter). A low, stay-flexible lock
    (≤ ``LOCK_LOW``) minimises cost, so take the cheapest. In between, the
    mid-ranked one. Deterministic given the (sorted) kept list."""
    if not kept:
        return None
    if lock_ratio >= LOCK_HIGH:
        return kept[0]
    if lock_ratio <= LOCK_LOW:
        return max(kept, key=lambda c: c.cost_score)  # cheapest
    return kept[len(kept) // 2]  # mid-ranked


# --------------------------------------------------------------------------- #
# Channels — scoring, curation, selection
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CuratedChannel:
    """A channel after curation: verdict + reason + the sub-scores and total."""

    channel: SalesChannel
    verdict: str
    reason: str
    margin_score: float  # 0..100, scaled to the richest channel in the set
    season_score: float  # 0..100, the target quarter's demand factor
    order_score: float  # 0..100, 100 when quantity clears the minimum order
    total_score: float  # 0.4·margin + 0.3·season + 0.3·order

    @property
    def name(self) -> str:
        return self.channel.name


@dataclass(frozen=True)
class ChannelCuration:
    kept: list[CuratedChannel]  # sorted by total_score, descending
    rejected: list[CuratedChannel]


def _margin_score(target_margin: float, richest: float) -> float:
    if richest <= 0:
        return 0.0
    return _clamp(100.0 * target_margin / richest)


def _season_score(season_factor: float) -> float:
    """Normalise the quarter demand factor so the catalog's peak (≈1.6) tops at 100."""
    return _clamp(100.0 * season_factor / _PEAK_SEASONALITY)


def _order_score(quantity: int, min_order: int) -> float:
    """100 when the order clears the channel minimum; linearly reduced below it."""
    if quantity >= min_order:
        return 100.0
    if min_order <= 0:
        return 100.0
    return _clamp(100.0 * quantity / min_order)


def curate_channels(
    target_month: str,
    quantity: int,
    channels: list[SalesChannel] | None = None,
) -> ChannelCuration:
    """Classify, score and rank sales channels for a target month + order size."""
    candidates = channels if channels is not None else list_channels()
    classified = [(c, *classify_channel(c.channel_type)) for c in candidates]
    kept_channels = [c for c, verdict, _ in classified if verdict == "keep"]

    richest = max((c.target_margin for c in kept_channels), default=0.0)
    quarter = quarter_of(target_month)

    kept: list[CuratedChannel] = []
    rejected: list[CuratedChannel] = []
    for channel, verdict, reason in classified:
        if verdict == "reject":
            rejected.append(CuratedChannel(channel, verdict, reason, 0.0, 0.0, 0.0, 0.0))
            continue
        margin = _margin_score(channel.target_margin, richest)
        season = _season_score(channel.season_factor(quarter))
        order = _order_score(quantity, channel.min_order)
        total = 0.4 * margin + 0.3 * season + 0.3 * order
        kept.append(CuratedChannel(channel, verdict, reason, margin, season, order, total))

    kept.sort(key=lambda c: c.total_score, reverse=True)
    return ChannelCuration(kept=kept, rejected=rejected)


def select_channel(kept: list[CuratedChannel]) -> CuratedChannel | None:
    """The top-ranked credible channel (highest total score)."""
    return kept[0] if kept else None
