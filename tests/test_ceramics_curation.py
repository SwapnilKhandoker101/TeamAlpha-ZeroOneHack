"""Offline tests for supplier + channel curation, scoring and selection.

Deterministic substring credibility (textile rejected, clay kept, a generic
vendor kept by default), the transparent sub-scores (cheapest supplier = cost 100,
exact lead-time fit = 100, richest channel = margin 100), descending ranking, and
the lock-%-routed supplier choice (high lock -> best-ranked, low lock -> cheapest,
mid lock -> the middle one).
"""

from ceramics_agent.catalog import SUPPLIERS, get_product  # noqa: F401  (kept for clarity)
from ceramics_agent.curation import (
    LOCK_HIGH,
    LOCK_LOW,
    classify_channel,
    classify_supplier,
    curate_channels,
    curate_suppliers,
    select_channel,
    select_supplier,
)


# --------------------------------------------------------------------------- #
# Credibility classifiers
# --------------------------------------------------------------------------- #
def test_textile_supplier_is_rejected_as_off_domain():
    verdict, reason = classify_supplier(("synthetic textile fibres",))
    assert verdict == "reject"
    assert "textile" in reason


def test_clay_supplier_is_kept_as_credible():
    verdict, reason = classify_supplier(("kaolin clay", "refractory materials"))
    assert verdict == "keep"
    assert "credible" in reason


def test_generic_supplier_is_kept_by_default():
    verdict, reason = classify_supplier(("logistics services",))
    assert verdict == "keep"
    assert "default" in reason


def test_channel_classifier_keeps_wholesale_rejects_automotive():
    assert classify_channel("wholesale")[0] == "keep"
    assert classify_channel("automotive parts")[0] == "reject"


# --------------------------------------------------------------------------- #
# Supplier scoring + ranking
# --------------------------------------------------------------------------- #
def test_cheapest_supplier_scores_cost_100_dearest_scores_0():
    import pytest

    curation = curate_suppliers(timeline_days=10)
    by_id = {c.supplier.id: c for c in curation.kept}
    assert by_id["eastern"].cost_score == pytest.approx(100.0)  # cheapest avg price factor
    assert by_id["premium"].cost_score == pytest.approx(0.0)     # dearest
    assert 0.0 < by_id["alpine"].cost_score < 100.0


def test_exact_lead_time_match_scores_100():
    # Alpine's lead time is 7 days; requesting exactly 7 gives a perfect lead score.
    curation = curate_suppliers(timeline_days=7)
    alpine = next(c for c in curation.kept if c.supplier.id == "alpine")
    assert alpine.lead_time_score == 100.0


def test_kept_suppliers_are_ranked_by_total_descending():
    curation = curate_suppliers(timeline_days=10)
    totals = [c.total_score for c in curation.kept]
    assert totals == sorted(totals, reverse=True)
    assert len(curation.kept) == 3  # none of the three catalog suppliers is off-domain


def test_reliability_score_equals_reliability_pct():
    curation = curate_suppliers(timeline_days=10)
    for c in curation.kept:
        assert c.reliability_score == c.supplier.reliability_pct


# --------------------------------------------------------------------------- #
# select_supplier — routed by the lock %
# --------------------------------------------------------------------------- #
def test_high_lock_takes_the_best_ranked_supplier():
    kept = curate_suppliers(timeline_days=10).kept
    chosen = select_supplier(kept, lock_ratio=LOCK_HIGH + 0.05)
    assert chosen is kept[0]


def test_low_lock_takes_the_cheapest_supplier():
    kept = curate_suppliers(timeline_days=10).kept
    chosen = select_supplier(kept, lock_ratio=LOCK_LOW - 0.05)
    assert chosen.supplier.id == "eastern"  # max cost_score
    assert chosen is max(kept, key=lambda c: c.cost_score)


def test_mid_lock_takes_the_middle_ranked_supplier():
    kept = curate_suppliers(timeline_days=10).kept
    chosen = select_supplier(kept, lock_ratio=(LOCK_LOW + LOCK_HIGH) / 2)
    assert chosen is kept[len(kept) // 2]


def test_select_supplier_handles_empty_list():
    assert select_supplier([], lock_ratio=0.5) is None


# --------------------------------------------------------------------------- #
# Channel scoring + selection
# --------------------------------------------------------------------------- #
def test_richest_margin_channel_scores_margin_100():
    curation = curate_channels(target_month="2026-11-01", quantity=5000)
    online = next(c for c in curation.kept if c.channel.id == "online")
    assert online.margin_score == 100.0  # target margin 0.80 is the richest


def test_order_below_minimum_reduces_order_fit():
    # Wholesale needs >=100; a 10-unit order cannot clear it.
    curation = curate_channels(target_month="2026-11-01", quantity=10)
    wholesale = next(c for c in curation.kept if c.channel.id == "wholesale")
    assert wholesale.order_score < 100.0


def test_channels_ranked_descending_and_select_channel_takes_the_top():
    curation = curate_channels(target_month="2026-11-01", quantity=5000)
    totals = [c.total_score for c in curation.kept]
    assert totals == sorted(totals, reverse=True)
    assert select_channel(curation.kept) is curation.kept[0]


def test_select_channel_handles_empty_list():
    assert select_channel([]) is None
