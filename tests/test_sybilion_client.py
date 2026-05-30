"""Offline tests for the live-forecast orchestration (Workstream 3).

No network: a fake client stands in for the REST layer, and the disk cache is
redirected to ``tmp_path``. The contract under test is that ``run_live_forecast``
submits, polls until the job finishes, caches all four artifacts under
``cache/<job_id>/``, and — crucially — never repoints ``latest_job.txt`` so the
pinned demo job stays the default.
"""

import pytest

from gas_agent import config
from gas_agent import sybilion_client as sc


class _FakeClient:
    """Stands in for SybilionClient: canned submit / poll / artifact responses."""

    def __init__(self, job_id: str, statuses: list[str], artifacts: dict[str, dict]):
        self._job_id = job_id
        self._statuses = list(statuses)
        self._artifacts = artifacts
        self.submitted: dict | None = None
        self.fetched: list[str] = []

    def submit_forecast(self, payload: dict) -> dict:
        self.submitted = payload
        return {"job_id": self._job_id, "status": "queued"}

    def get_forecast(self, job_id: str) -> dict:
        status = self._statuses.pop(0) if self._statuses else "completed"
        return {"job_id": job_id, "status": status}

    def get_artifact(self, job_id: str, name: str) -> dict:
        self.fetched.append(name)
        return self._artifacts[name]


def test_build_forecast_payload_matches_documented_shape():
    payload = sc.build_forecast_payload(
        {"2017-11-01": 20.31, "2017-12-01": 19.62},
        keywords=["European natural gas", "Dutch TTF"],
        category_ids=[25, 46],
        region_codes=[1003, 276],
        soft_horizon=6,
    )
    assert payload["soft_horizon"] == 6
    assert payload["hard_horizon"] is None
    assert payload["backtest"] is True
    assert payload["frequency"] == "monthly"
    assert payload["pipeline_version"] == "v1"
    # Timeseries passes through untouched.
    assert payload["timeseries"] == {"2017-11-01": 20.31, "2017-12-01": 19.62}
    # Keywords live under metadata (no top-level keywords field), title required.
    assert payload["timeseries_metadata"]["keywords"] == ["European natural gas", "Dutch TTF"]
    assert 20 <= len(payload["timeseries_metadata"]["title"]) <= 511
    assert "timeseries" not in payload["filters"]
    assert payload["filters"] == {"categories": [25, 46], "regions": [1003, 276], "limit": 1000}


def test_build_forecast_payload_defaults_are_safe_when_unspecified():
    payload = sc.build_forecast_payload({"2017-11-01": 20.31})
    assert payload["timeseries_metadata"]["keywords"] == []
    assert payload["filters"]["categories"] == []
    assert payload["filters"]["regions"] == []
    # The title/description defaults satisfy the API length bounds.
    assert payload["timeseries_metadata"]["title"]
    assert len(payload["timeseries_metadata"]["description"]) <= 2048


def test_run_live_forecast_caches_artifacts_without_touching_pointer(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    pointer = tmp_path / "latest_job.txt"
    monkeypatch.setattr(sc, "LATEST_JOB_POINTER", pointer)
    monkeypatch.setattr(sc.time, "sleep", lambda *_: None)

    canned = {
        "forecast": {"data": {"forecast_series": {"2026-06-01": {"quantile_forecast": {}}}}},
        "external_signals": {"data": {"uid-0": {"driver_name": "x"}}},
        "backtest_metrics": {"data": {"mape": 0.28}},
        "backtest_trajectories": {"data": {"rows": []}},
    }
    client = _FakeClient("job-live-123", ["running", "running", "completed"], canned)

    job_id = sc.run_live_forecast(client, {"any": "payload"}, poll_interval=0.0, timeout=5.0)

    assert job_id == "job-live-123"
    # All four artifacts fetched and persisted under the live job's cache dir.
    assert sorted(client.fetched) == sorted(sc.FORECAST_ARTIFACTS)
    for name in sc.FORECAST_ARTIFACTS:
        assert sc.has_artifact("job-live-123", name)
        assert sc.load_artifact("job-live-123", name) == canned[name]
    # Non-destructive: the pinned demo pointer is never written.
    assert not pointer.exists()
    assert sc.get_latest_job() is None


def test_run_live_forecast_raises_on_failed_job(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(sc, "LATEST_JOB_POINTER", tmp_path / "latest_job.txt")
    monkeypatch.setattr(sc.time, "sleep", lambda *_: None)

    client = _FakeClient("job-x", ["running", "failed"], {})
    with pytest.raises(RuntimeError, match="failure"):
        sc.run_live_forecast(client, {}, poll_interval=0.0, timeout=5.0)


def test_run_live_forecast_raises_without_job_id():
    class _NoId:
        def submit_forecast(self, payload):
            return {"status": "queued"}

    with pytest.raises(RuntimeError, match="no job id"):
        sc.run_live_forecast(_NoId(), {})


def test_run_live_forecast_times_out_when_never_terminal(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(sc, "LATEST_JOB_POINTER", tmp_path / "latest_job.txt")
    monkeypatch.setattr(sc.time, "sleep", lambda *_: None)

    # monotonic jumps past the deadline on the first check inside the loop.
    ticks = iter([0.0, 100.0, 200.0])
    monkeypatch.setattr(sc.time, "monotonic", lambda: next(ticks))

    client = _FakeClient("job-stuck", ["running", "running", "running"], {})
    with pytest.raises(TimeoutError):
        sc.run_live_forecast(client, {}, poll_interval=0.0, timeout=50.0)
