"""Offline tests for the ceramics globe's two layers (W5).

``ceramics_agent.geo`` places the *already-decided* ceramics evidence onto a sphere:
the kept suppliers at their sourcing country (sized by curation score) and the demand
markets where each channel's buyers sit (sized by demand potential). These tests pin
that both layers emit valid :class:`gas_agent.geo.CountryAggregate` points with real
coordinates, that the headline invariant holds — **it only places existing numbers, it
computes no decision** — and that the per-country brief is explanation-only with a
deterministic offline fallback. They run in a clean checkout: the supplier/channel
catalog is committed and the LLM is forced off.
"""

from gas_agent.geo import CountryAggregate

from ceramics_agent import geo
from ceramics_agent.catalog import SUPPLIER_REGION_NAMES, Supplier, list_suppliers
from ceramics_agent.curation import curate_suppliers


def _valid_point(p: CountryAggregate) -> bool:
    """A point a globe can actually draw: real, in-range coordinates + a non-empty label."""
    return (
        isinstance(p, CountryAggregate)
        and -90.0 <= p.lat <= 90.0
        and -180.0 <= p.lon <= 180.0
        and not (p.lat == 0.0 and p.lon == 0.0)  # not the null-island placeholder
        and bool(p.region)
        and p.kept_importance >= 0.0
    )


# --------------------------------------------------------------------------- #
# coords_for — reuses the gas table, plus the local supplement
# --------------------------------------------------------------------------- #
def test_coords_reuse_gas_table_and_local_supplement():
    assert geo.coords_for("Germany") == (51.16, 10.45)  # straight from the gas table
    assert geo.coords_for("Switzerland") == (46.8, 8.2)  # ceramics-local supplement
    assert geo.coords_for("Atlantis") is None  # unplaceable → dropped, never null-island


# --------------------------------------------------------------------------- #
# Where to buy — kept suppliers at their sourcing country, sized by score
# --------------------------------------------------------------------------- #
def test_supplier_layer_places_every_kept_supplier_with_real_coords():
    curation = curate_suppliers(14)
    points = geo.supplier_sourcing_points(curation, chosen_name="Alpine Clay Works")

    assert len(points) == len(curation.kept) > 0
    assert all(_valid_point(p) for p in points)
    # Each kept supplier's sourcing country is represented.
    placed = {p.region for p in points}
    assert {SUPPLIER_REGION_NAMES[c.supplier.region] for c in curation.kept} <= placed
    # Height == curation score, and the layer is sorted strongest-first.
    assert [p.kept_importance for p in points] == sorted(
        (p.kept_importance for p in points), reverse=True)
    by_region = {p.region: p for p in points}
    for c in curation.kept:
        assert by_region[SUPPLIER_REGION_NAMES[c.supplier.region]].kept_importance == round(
            c.total_score, 1)


def test_supplier_layer_flags_the_chosen_supplier_and_drops_rejected():
    # A blacklisted (off-domain) supplier is rejected by curation, so it must not appear.
    textile = Supplier(
        id="textile", name="Bav Textiles", region="DE", specialties=("textile",),
        reliability_pct=90.0, lead_time_days=10,
    )
    curation = curate_suppliers(14, suppliers=[*list_suppliers(), textile])
    assert any(c.supplier.id == "textile" for c in curation.rejected)

    points = geo.supplier_sourcing_points(curation, chosen_name="Alpine Clay Works")
    assert "Bav Textiles" not in {n for p in points for n in p.kept_names}  # rejected dropped
    chosen_labels = [n for p in points for n in p.kept_names if "Alpine Clay Works" in n]
    assert chosen_labels and chosen_labels[0].endswith("✓ chosen")  # the pick is flagged


# --------------------------------------------------------------------------- #
# Where to sell — demand markets, sized by demand potential, top scaled to 100
# --------------------------------------------------------------------------- #
def test_demand_layer_places_markets_with_real_coords_and_scales_to_100():
    points = geo.demand_market_points("2026-06-01")
    assert points and all(_valid_point(p) for p in points)
    # Germany is in every channel's destination set, so it is the strongest market.
    assert points[0].region == "Germany"
    assert points[0].kept_importance == 100.0  # top scaled to 100
    assert [p.kept_importance for p in points] == sorted(
        (p.kept_importance for p in points), reverse=True)
    # A shared market carries every contributing channel's name.
    germany = next(p for p in points if p.region == "Germany")
    assert len(germany.kept_names) >= 2


def test_demand_layer_is_deterministic_and_quarter_sensitive():
    # Identical inputs → identical points (no RNG, no clock).
    a = geo.demand_market_points("2026-06-01")
    b = geo.demand_market_points("2026-06-01")
    assert [(p.region, p.kept_importance) for p in a] == [(p.region, p.kept_importance) for p in b]
    # A different quarter re-weights demand via channel seasonality, so the mix shifts.
    q4 = geo.demand_market_points("2026-12-01")  # online/holiday peak quarter
    assert {p.region for p in q4} == {p.region for p in a}
    assert any(
        round(x.kept_importance, 1) != round(y.kept_importance, 1)
        for x, y in zip(sorted(a, key=lambda p: p.region), sorted(q4, key=lambda p: p.region))
    )


# --------------------------------------------------------------------------- #
# The brief — explanation only, deterministic fallback, never a decision number
# --------------------------------------------------------------------------- #
def _force_llm_unavailable(monkeypatch):
    """Make the (real or stubbed) Featherless call fail, so the brief takes its
    deterministic fallback — the same idiom the gas geo tests use."""
    import gas_agent.llm as llm

    def boom(**kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(llm, "chat_text", boom)


def test_country_brief_falls_back_offline_for_both_layers(monkeypatch):
    _force_llm_unavailable(monkeypatch)

    buy_text, buy_src = geo.ceramics_country_brief("Austria", "buy", ["Alpine Clay Works"])
    sell_text, sell_src = geo.ceramics_country_brief("Germany", "sell", ["Online Retail Export"])
    assert buy_src == "fallback" and sell_src == "fallback"
    assert buy_text and sell_text
    # Explanation only — no lock %, score or price leaks into a brief.
    for text in (buy_text, sell_text):
        assert "%" not in text and "€" not in text


def test_country_brief_handles_unknown_region_gracefully(monkeypatch):
    _force_llm_unavailable(monkeypatch)

    text, source = geo.ceramics_country_brief("Atlantis", "sell", [])
    assert source == "fallback" and "Atlantis" in text


def test_country_brief_uses_llm_when_available_and_picks_the_layer_prompt(monkeypatch):
    import gas_agent.llm as llm

    captured = {}

    def capture(**kwargs):
        captured["system"] = kwargs.get("system", "")
        return "A grounded, decision-free sentence."

    monkeypatch.setattr(llm, "chat_text", capture)
    text, source = geo.ceramics_country_brief("Poland", "buy", ["Eastern European Materials"])
    assert source == "llm" and text == "A grounded, decision-free sentence."
    assert "source" in captured["system"].lower()  # the buy-layer system prompt was used
