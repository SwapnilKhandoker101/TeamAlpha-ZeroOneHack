"""Offline tests for the non-blocking live-forecast building blocks (W18).

A real Sybilion job can take ~11 minutes, so the in-app live run must submit all jobs
up front and poll one tick per rerun rather than block. These tests pin the split
helpers that make that possible — ``submit`` / ``poll_once`` / ``fetch_forecast_artifacts``
in the client, and ``submit_factor_jobs`` / ``assemble_ceramics_from_jobs`` for the four
ceramics factors — using a fake client and a redirected disk cache. No network, no sleep.
The blocking ``run_live_forecast`` (used by the offline scenario batch) is covered by
``test_sybilion_client.py`` and stays behaviour-identical after the refactor.
"""

import pytest

from gas_agent import config
from gas_agent import sybilion_client as sc

from ceramics_agent import forecast as cforecast
from ceramics_agent.forecast import FACTORS


# --------------------------------------------------------------------------- #
# Descriptor readers + single-step poll
# --------------------------------------------------------------------------- #
def test_descriptor_readers_and_terminality():
    assert sc.job_id_of({"job_id": "j1"}) == "j1"
    assert sc.job_id_of({"id": "j2"}) == "j2"
    assert sc.job_id_of({"forecast_id": "j3"}) == "j3"
    assert sc.job_id_of({"nope": 1}) is None
    assert sc.status_of({"status": "RUNNING"}) == "running"  # lowercased
    assert sc.status_of({"state": "Completed"}) == "completed"
    assert sc.status_of({}) == ""
    assert sc.is_terminal("completed") and sc.is_terminal("failed")
    assert not sc.is_terminal("running") and not sc.is_terminal("")


def test_poll_once_reads_one_status_without_sleeping():
    class _C:
        def __init__(self):
            self.calls = 0

        def get_forecast(self, job_id):
            self.calls += 1
            return {"job_id": job_id, "status": "running"}

    client = _C()
    assert sc.poll_once(client, "j1") == "running"
    assert client.calls == 1  # exactly one poll, no loop


def test_fetch_forecast_artifacts_caches_all_four(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)

    canned = {name: {"data": {name: True}} for name in sc.FORECAST_ARTIFACTS}

    class _C:
        def __init__(self):
            self.fetched = []

        def get_artifact(self, job_id, name):
            self.fetched.append(name)
            return canned[name]

    client = _C()
    sc.fetch_forecast_artifacts(client, "job-7")
    assert sorted(client.fetched) == sorted(sc.FORECAST_ARTIFACTS)
    for name in sc.FORECAST_ARTIFACTS:
        assert sc.has_artifact("job-7", name)
        assert sc.load_artifact("job-7", name) == canned[name]


# --------------------------------------------------------------------------- #
# Ceramics: submit all factors up front, then assemble from cached jobs
# --------------------------------------------------------------------------- #
class _FactorClient:
    """Returns a distinct job id per submit, so the four factors get four jobs."""

    def __init__(self):
        self.n = 0
        self.submitted: list[dict] = []

    def submit_forecast(self, payload):
        self.n += 1
        self.submitted.append(payload)
        return {"job_id": f"factor-job-{self.n}", "status": "queued"}


def test_submit_factor_jobs_returns_a_job_per_factor():
    client = _FactorClient()
    jobs = cforecast.submit_factor_jobs(client, cforecast.default_factor_history())
    assert set(jobs) == set(FACTORS)  # one job id per cost factor
    assert len(set(jobs.values())) == len(FACTORS)  # all distinct
    assert len(client.submitted) == len(FACTORS)
    # Each payload carries that factor's keywords (reuses the gas payload builder).
    assert all("timeseries_metadata" in p for p in client.submitted)


def test_assemble_ceramics_from_jobs_builds_the_combined_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)

    # Stand in a minimal forecast.json for each factor job under the redirected cache.
    factor_jobs = {factor: f"fjob-{factor}" for factor in FACTORS}
    for factor, job in factor_jobs.items():
        sc.save_artifact(job, "forecast", {
            "data": {"forecast_series": {"2026-06-01": {"quantile_forecast": {
                "0.10": 1.0, "0.50": 2.0, "0.90": 3.0}}}}})

    combined = cforecast.assemble_ceramics_from_jobs(factor_jobs, job_id="ceramics-test")
    assert combined == "ceramics-test"
    artifact = sc.load_artifact(combined, "ceramics_forecast")
    assert artifact["meta"]["source"] == "live"
    assert set(artifact["factors"]) == set(FACTORS)
    # The forecast series flows through to the combined artifact.
    assert artifact["factors"]["gas"]["forecast_series"]["2026-06-01"]["quantile_forecast"]["0.50"] == 2.0


def test_assemble_then_load_round_trips_through_the_normal_loader(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    factor_jobs = {factor: f"rj-{factor}" for factor in FACTORS}
    for factor, job in factor_jobs.items():
        sc.save_artifact(job, "forecast", {
            "data": {"forecast_series": {
                "2026-06-01": {"quantile_forecast": {"0.10": 10.0, "0.50": 12.0, "0.90": 14.0}},
                "2026-07-01": {"quantile_forecast": {"0.10": 11.0, "0.50": 13.0, "0.90": 15.0}},
            }}})
    combined = cforecast.assemble_ceramics_from_jobs(factor_jobs, job_id="ceramics-rt")

    # The normal loader reads the assembled live artifact back as MonthForecast lists.
    factors, source = cforecast.load_ceramics_forecast(combined)
    assert source == "live"
    assert set(factors) == set(FACTORS)
    assert factors["gas"][0].median == 12.0
