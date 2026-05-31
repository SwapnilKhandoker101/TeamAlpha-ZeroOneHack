"""Offline tests for the scenario library + nearest-match retrieval (W17).

The library pre-fetches real Sybilion forecasts for the (product × gas_exposure) grid and
commits them under ``scenarios/``; at intake we match a parsed profile to the nearest cell.
These tests pin the **deterministic** matcher (product-dominant, nearest gas-exposure, stable
ties, graceful fallback while the library fills in) and the artifact-loader indirection that
lets a matched scenario flow through the existing render path (cache → scenarios). No network.
"""

from gas_agent import config
from gas_agent import scenarios as scn
from gas_agent import sybilion_client as sc


def _cell(product: str, gas_exposure: str) -> scn.Scenario:
    return scn.Scenario(product=product, gas_exposure=gas_exposure,
                        slug=scn.slug_for(product, gas_exposure), label=f"{product} · {gas_exposure}")


# --------------------------------------------------------------------------- #
# The matcher
# --------------------------------------------------------------------------- #
def test_exact_match_when_product_and_exposure_present():
    pool = [_cell("bowl", "low"), _cell("bowl", "high"), _cell("tile", "medium")]
    m = scn.match("bowl", "high", pool)
    assert m is not None and m.exact and m.distance == 0
    assert m.scenario.slug == "scn-bowl-high"


def test_nearest_exposure_within_the_same_product():
    pool = [_cell("bowl", "low"), _cell("tile", "high")]
    m = scn.match("bowl", "high", pool)  # bowl has only 'low' → nearest exposure, same product
    assert m.scenario.product == "bowl" and not m.exact
    assert m.scenario.gas_exposure == "low" and m.distance == 2  # low..high = 2 steps


def test_same_product_beats_a_closer_other_product():
    # tile/high is exposure-exact but cross-product; bowl/low is same-product → must win.
    pool = [_cell("bowl", "low"), _cell("tile", "high")]
    m = scn.match("bowl", "high", pool)
    assert m.scenario.product == "bowl"  # cross-product penalty keeps us on the same product


def test_cross_product_fallback_when_product_absent():
    pool = [_cell("tile", "medium"), _cell("dinnerware", "low")]
    m = scn.match("bowl", "medium", pool)  # no bowl cell → nearest exposure of any product
    assert m is not None and not m.exact
    assert m.scenario.gas_exposure == "medium"  # exposure-exact cross-product


def test_empty_library_returns_none():
    assert scn.match("bowl", "medium", []) is None


def test_ties_are_stable():
    # Two equally-distant cells → the first in the pool wins (deterministic).
    pool = [_cell("dinnerware", "medium"), _cell("tile", "medium")]
    assert scn.match("bowl", "medium", pool).scenario.product == "dinnerware"


def test_nearest_options_prefers_same_product_then_caps():
    pool = [_cell("tile", "low"), _cell("bowl", "low"), _cell("bowl", "high"), _cell("dinnerware", "low")]
    opts = scn.nearest_options("bowl", pool, limit=3)
    assert opts[0].product == "bowl" and opts[1].product == "bowl"  # same product first
    assert len(opts) == 3


# --------------------------------------------------------------------------- #
# Index round-trip
# --------------------------------------------------------------------------- #
def test_load_index_round_trips(tmp_path, monkeypatch):
    index = tmp_path / "index.json"
    index.write_text(
        '{"scenarios": [{"product": "bowl", "gas_exposure": "medium", '
        '"slug": "scn-bowl-medium", "ceramics_ref": "_shared", "label": "bowl · medium-gas"}]}'
    )
    loaded = scn.load_index(index)
    assert len(loaded) == 1
    assert loaded[0].product == "bowl" and loaded[0].slug == "scn-bowl-medium"
    assert loaded[0].ceramics_ref == "_shared"


def test_load_index_empty_when_absent(tmp_path):
    assert scn.load_index(tmp_path / "nope.json") == []


# --------------------------------------------------------------------------- #
# Loader indirection — a scenario dir is read like a job dir (cache → scenarios)
# --------------------------------------------------------------------------- #
def test_artifact_loader_resolves_cache_first_then_scenarios(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "SCENARIOS_DIR", tmp_path / "scenarios")

    # A scenario-only artifact (no cache copy) resolves out of scenarios/.
    (config.SCENARIOS_DIR / "scn-x").mkdir(parents=True)
    (config.SCENARIOS_DIR / "scn-x" / "forecast.json").write_text('{"from": "scenarios"}')
    assert sc.has_artifact("scn-x", "forecast")
    assert sc.load_artifact("scn-x", "forecast") == {"from": "scenarios"}

    # A cache copy shadows the scenario (cache wins); writes always land in cache.
    sc.save_artifact("scn-x", "forecast", {"from": "cache"})
    assert sc.load_artifact("scn-x", "forecast") == {"from": "cache"}
    assert sc.cache_artifact_path("scn-x", "forecast").exists()  # the write went to cache


def test_committed_seed_library_matches_the_demo_default():
    # The committed library (seeded offline) must at least carry the default cell, so the
    # blank-submit demo (bowl · medium) resolves to a real pre-fetched forecast.
    index = scn.load_index()
    if not index:  # library not built in this checkout — skip rather than fail
        return
    m = scn.match("bowl", "medium", index)
    assert m is not None
    assert sc.scenario_artifact_path(m.scenario.slug, "forecast").exists()
