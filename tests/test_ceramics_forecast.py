"""Offline tests for the 4-factor forecast loader and the opt-in live path.

No network. The committed mock is the source of truth, so the default load is
exercised directly; the live path is driven through a fully faked Sybilion layer
(submit/poll/fetch monkeypatched) and the contract under test is the same one the
gas agent keeps — it caches a combined ``ceramics_forecast.json`` and **never**
repoints ``latest_job.txt``.
"""

import pytest

from gas_agent import sybilion_client as sc

from ceramics_agent import forecast as cf
from ceramics_agent.forecast import (
    FACTORS,
    load_ceramics_forecast,
    load_factor_drivers,
    parse_factor_months,
)


# --------------------------------------------------------------------------- #
# Loading the committed mock
# --------------------------------------------------------------------------- #
def test_mock_loads_four_factors_each_with_six_ordered_months():
    factors, source = load_ceramics_forecast()
    assert source == "mock"
    assert set(factors) == set(FACTORS)
    for months in factors.values():
        assert len(months) == 6
        # Ordered by month and quantiles correctly nested q10 <= q50 <= q90.
        assert months == sorted(months, key=lambda m: m.month)
        for m in months:
            assert m.low <= m.median <= m.high


def test_missing_job_falls_back_to_mock():
    factors, source = load_ceramics_forecast("no-such-job-id")
    assert source == "mock"
    assert set(factors) == set(FACTORS)


def test_parse_factor_months_sorts_and_reduces_quantiles():
    factor_json = {
        "forecast_series": {
            "2026-07-01": {"quantile_forecast": {"0.10": 8.0, "0.50": 10.0, "0.90": 12.0}},
            "2026-06-01": {"quantile_forecast": {"0.10": 4.0, "0.50": 5.0, "0.90": 6.0}},
        }
    }
    months = parse_factor_months(factor_json)
    assert [m.month for m in months] == ["2026-06-01", "2026-07-01"]
    assert months[0].median == 5.0 and months[0].low == 4.0 and months[0].high == 6.0


def test_load_factor_drivers_returns_a_list_per_factor():
    drivers = load_factor_drivers()
    assert set(drivers) == set(FACTORS)
    for factor, entries in drivers.items():
        for d in entries:
            assert d.factor == factor
            assert d.name  # named provenance, never blank


# --------------------------------------------------------------------------- #
# Opt-in live path (faked Sybilion — no network, never repoints the pointer)
# --------------------------------------------------------------------------- #
def test_live_path_caches_combined_artifact_without_touching_pointer(monkeypatch):
    submitted_payloads: list[dict] = []
    saved: dict = {}

    def fake_run_live_forecast(client, payload, **kwargs):
        submitted_payloads.append(payload)
        return "factor-job"

    def fake_load_artifact(job_id, name):
        if name == "forecast.json":
            return {"data": {"forecast_series": {
                "2026-06-01": {"quantile_forecast": {"0.10": 1.0, "0.50": 2.0, "0.90": 3.0}},
            }}}
        raise FileNotFoundError(name)  # no external_signals -> drivers fall back to []

    def fake_save_artifact(job_id, name, content):
        saved["job_id"] = job_id
        saved["name"] = name
        saved["content"] = content

    def fail_if_repointed(job_id):
        raise AssertionError("run_live_ceramics_forecast must never repoint latest_job.txt")

    monkeypatch.setattr(sc, "run_live_forecast", fake_run_live_forecast)
    monkeypatch.setattr(sc, "load_artifact", fake_load_artifact)
    monkeypatch.setattr(sc, "save_artifact", fake_save_artifact)
    monkeypatch.setattr(sc, "set_latest_job", fail_if_repointed)

    base = {factor: {"2026-01-01": 1.0, "2026-02-01": 1.1, "2026-03-01": 1.2} for factor in FACTORS}
    job_id = cf.run_live_ceramics_forecast(client=object(), base_series_by_factor=base)

    # A fresh, session-only combined job — distinct from the pinned demo.
    assert job_id.startswith("ceramics-")
    # One Sybilion submit per factor.
    assert len(submitted_payloads) == len(FACTORS)
    # The combined artifact is cached under the new job, all four factors present.
    assert saved["name"] == "ceramics_forecast"
    assert saved["job_id"] == job_id
    assert set(saved["content"]["factors"]) == set(FACTORS)
    for factor in FACTORS:
        assert "forecast_series" in saved["content"]["factors"][factor]


def test_live_path_skips_factors_with_no_history(monkeypatch):
    saved: dict = {}
    monkeypatch.setattr(sc, "run_live_forecast", lambda *a, **k: "factor-job")
    monkeypatch.setattr(
        sc, "load_artifact",
        lambda j, n: {"data": {"forecast_series": {"2026-06-01": {"quantile_forecast": {}}}}}
        if n == "forecast.json" else (_ for _ in ()).throw(FileNotFoundError(n)),
    )
    monkeypatch.setattr(sc, "save_artifact", lambda j, n, c: saved.update(content=c))
    monkeypatch.setattr(sc, "set_latest_job", lambda j: (_ for _ in ()).throw(AssertionError("no repoint")))

    # Only gas has history -> only gas ends up in the combined artifact.
    job_id = cf.run_live_ceramics_forecast(object(), {"gas": {"2026-01-01": 1.0}})
    assert job_id.startswith("ceramics-")
    assert list(saved["content"]["factors"]) == ["gas"]


# --------------------------------------------------------------------------- #
# Cache-first plumbing — a deterministic combined-job id so re-submitting the
# same profile reuses the cached artifact instead of re-polling four jobs.
# --------------------------------------------------------------------------- #
def test_combined_job_id_is_deterministic_and_signature_sensitive():
    a = cf.combined_job_id("bowl|5000|14d|de")
    assert a == cf.combined_job_id("bowl|5000|14d|de")  # same signature -> same id
    assert a != cf.combined_job_id("tile|5000|14d|de")  # different signature -> different id
    assert a.startswith("ceramics-")


def test_live_artifact_exists_false_for_none_and_missing():
    assert cf.live_artifact_exists(None) is False
    assert cf.live_artifact_exists("definitely-not-a-cached-job") is False


def test_ensure_returns_cached_job_without_polling(monkeypatch):
    # Cache hit: ensure must return the deterministic id and never run the live path.
    monkeypatch.setattr(cf, "live_artifact_exists", lambda job_id: True)

    def fail_if_polled(*a, **k):
        raise AssertionError("ensure_ceramics_forecast must not re-poll on a cache hit")

    monkeypatch.setattr(cf, "run_live_ceramics_forecast", fail_if_polled)

    job_id = cf.ensure_ceramics_forecast(object(), {}, signature="bowl|5000|14d|de")
    assert job_id == cf.combined_job_id("bowl|5000|14d|de")


def test_ensure_runs_live_under_the_deterministic_id_on_a_miss(monkeypatch):
    # Cache miss: ensure must run the live path under the signature's stable id.
    monkeypatch.setattr(cf, "live_artifact_exists", lambda job_id: False)
    captured: dict = {}

    def fake_run(client, base, *, job_id, soft_horizon):
        captured["job_id"] = job_id
        return job_id

    monkeypatch.setattr(cf, "run_live_ceramics_forecast", fake_run)
    sig = "tile|8000|21d|de"
    job_id = cf.ensure_ceramics_forecast(object(), {}, signature=sig)
    assert job_id == cf.combined_job_id(sig)
    assert captured["job_id"] == cf.combined_job_id(sig)


def test_ensure_refresh_repolls_even_on_a_cache_hit(monkeypatch):
    # refresh=True (a chat impact invalidated a factor) forces a re-poll.
    monkeypatch.setattr(cf, "live_artifact_exists", lambda job_id: True)
    called = {"ran": False}

    def fake_run(client, base, *, job_id, soft_horizon):
        called["ran"] = True
        return job_id

    monkeypatch.setattr(cf, "run_live_ceramics_forecast", fake_run)
    cf.ensure_ceramics_forecast(object(), {}, signature="bowl|5000|14d|de", refresh=True)
    assert called["ran"] is True
