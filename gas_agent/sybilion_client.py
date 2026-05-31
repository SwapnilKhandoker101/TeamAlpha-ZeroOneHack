"""Talking to Sybilion, and remembering what it said.

Two jobs live here:

1. A thin REST client over the documented Sybilion endpoints (submit a forecast,
   poll it, pull artifacts, rank drivers, list the catalog). It reads its key and
   base URL from :mod:`gas_agent.config`.

2. A disk cache + small parsers. Every artifact we fetch is written under
   ``cache/<job_id>/`` so the dashboard runs off real, frozen forecast data
   without needing live API access during a demo. The parsers reduce the raw
   artifact JSON to the handful of fields the hedge policy and charts need.

During development we drive the real API through the Sybilion MCP tools and save
the artifacts into this same cache, so the REST methods are a faithful
documented client even when the demo path only ever reads the cache.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

from gas_agent import config
from gas_agent.hedge_policy import MonthForecast

LATEST_JOB_POINTER = config.CACHE_DIR / "latest_job.txt"


# --------------------------------------------------------------------------- #
# REST client
# --------------------------------------------------------------------------- #
class SybilionClient:
    """Minimal client over the documented /api/v1 endpoints."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None, timeout: float = 30.0):
        self.api_key = api_key if api_key is not None else config.SYBILION_API_KEY
        self.base_url = (base_url or config.SYBILION_BASE_URL).rstrip("/")
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _get(self, path: str) -> dict:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(f"{self.base_url}{path}", headers=self._headers())
            response.raise_for_status()
            return response.json()

    def _post(self, path: str, payload: dict) -> dict:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.base_url}{path}", headers=self._headers(), json=payload)
            response.raise_for_status()
            return response.json()

    def submit_forecast(self, payload: dict) -> dict:
        """POST /api/v1/forecasts — returns the job descriptor (incl. job_id)."""
        return self._post("/api/v1/forecasts", payload)

    def get_forecast(self, job_id: str) -> dict:
        """GET /api/v1/forecasts/{id} — status descriptor."""
        return self._get(f"/api/v1/forecasts/{job_id}")

    def get_artifact(self, job_id: str, name: str) -> dict:
        """GET /api/v1/forecasts/{id}/artifacts/{name}."""
        return self._get(f"/api/v1/forecasts/{job_id}/artifacts/{name}")

    def rank_drivers(self, payload: dict) -> dict:
        """POST /api/v1/drivers — driver ranking for a series + filters."""
        return self._post("/api/v1/drivers", payload)

    def list_categories(self) -> dict:
        return self._get("/api/v1/categories")

    def list_regions(self) -> dict:
        return self._get("/api/v1/regions")

    def whoami(self) -> dict:
        return self._get("/api/v1/me")


# --------------------------------------------------------------------------- #
# Disk cache
# --------------------------------------------------------------------------- #
def job_cache_dir(job_id: str) -> Path:
    return config.CACHE_DIR / job_id


def _artifact_name(name: str) -> str:
    return name if name.endswith(".json") else f"{name}.json"


def cache_artifact_path(job_id: str, name: str) -> Path:
    """The WRITE path for an artifact — always under the working ``cache/`` (so a live
    refresh never overwrites a committed scenario)."""
    return job_cache_dir(job_id) / _artifact_name(name)


def scenario_artifact_path(job_id: str, name: str) -> Path:
    """The committed scenario-library path for a slug (read-only, W17)."""
    return config.SCENARIOS_DIR / job_id / _artifact_name(name)


def artifact_path(job_id: str, name: str) -> Path:
    """READ-resolve an artifact: the working ``cache/`` first, then the committed
    ``scenarios/`` library, else the cache path (so a matched library scenario flows
    through the existing render path unchanged — a scenario dir is just a job dir)."""
    cache_path = cache_artifact_path(job_id, name)
    if cache_path.exists():
        return cache_path
    scenario_path = scenario_artifact_path(job_id, name)
    if scenario_path.exists():
        return scenario_path
    return cache_path


def save_artifact(job_id: str, name: str, content: dict) -> Path:
    path = cache_artifact_path(job_id, name)  # writes always go to the working cache
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2))
    return path


def has_artifact(job_id: str, name: str) -> bool:
    return artifact_path(job_id, name).exists()


def load_artifact(job_id: str, name: str) -> dict:
    return json.loads(artifact_path(job_id, name).read_text())


def set_latest_job(job_id: str) -> None:
    LATEST_JOB_POINTER.parent.mkdir(parents=True, exist_ok=True)
    LATEST_JOB_POINTER.write_text(job_id.strip())


def get_latest_job() -> str | None:
    if LATEST_JOB_POINTER.exists():
        value = LATEST_JOB_POINTER.read_text().strip()
        return value or None
    return None


# --------------------------------------------------------------------------- #
# Live forecast orchestration (opt-in; off the default demo path)
# --------------------------------------------------------------------------- #
# The four artifacts the dashboard reads for any job. A live refresh must fetch
# and cache all of them so every downstream panel (forecast band, driver
# curation, backtest verdict, globe) has the data it expects.
FORECAST_ARTIFACTS: tuple[str, ...] = (
    "forecast",
    "external_signals",
    "backtest_metrics",
    "backtest_trajectories",
)

# Statuses Sybilion reports for a job; these two are terminal.
TERMINAL_STATUSES: frozenset[str] = frozenset({"completed", "failed"})

# The TTF document already carries a good title/description; these are the
# defaults a live caller inherits when it does not pass its own. The API
# requires a title (20-511 chars) and accepts a description (<=2048 chars).
DEFAULT_SERIES_TITLE = (
    "European natural gas price — Dutch TTF front-month futures, monthly close in EUR/MWh"
)
DEFAULT_SERIES_DESCRIPTION = (
    "Monthly closing price of Dutch TTF natural gas front-month futures in EUR/MWh. "
    "TTF is the benchmark wholesale gas price for continental Europe. Used by an "
    "energy-intensive EU industrial buyer (a German glass/ceramics manufacturer) to "
    "decide what share of next quarter's gas to lock in forward versus leave to spot."
)


def build_forecast_payload(
    timeseries: dict,
    *,
    title: str = DEFAULT_SERIES_TITLE,
    description: str = DEFAULT_SERIES_DESCRIPTION,
    keywords: list[str] | None = None,
    category_ids: list[int] | None = None,
    region_codes: list[int] | None = None,
    soft_horizon: int = 6,
    recency_factor: float = 0.5,
    backtest: bool = True,
    limit: int = 1000,
) -> dict:
    """Assemble the documented Sybilion forecast request body.

    Takes primitives (not a ``KeywordSelection``) so this module never imports
    :mod:`gas_agent.keyword_agent`, keeping the dependency arrow one-directional.
    ``timeseries`` is the ``{month: price}`` map from ``ttf_series.json``;
    keywords / categories / regions come from the selected filters. The keywords
    live under ``timeseries_metadata`` (the API has no top-level keywords field).
    """
    return {
        "soft_horizon": soft_horizon,
        "hard_horizon": None,
        "backtest": backtest,
        "frequency": "monthly",
        "timeseries": dict(timeseries),
        "recency_factor": recency_factor,
        "strictly_positive": False,
        "timeseries_metadata": {
            "title": title,
            "description": description,
            "keywords": list(keywords or []),
        },
        "filters": {
            "categories": list(category_ids or []),
            "regions": list(region_codes or []),
            "limit": limit,
        },
        "pipeline_version": "v1",
    }


def _job_id_from_descriptor(descriptor: dict) -> str | None:
    """Pull the new job id out of a submit/status descriptor, tolerating the
    documented key spellings."""
    for key in ("job_id", "id", "forecast_id"):
        value = descriptor.get(key)
        if value:
            return str(value)
    return None


def _status_from_descriptor(descriptor: dict) -> str:
    raw = descriptor.get("status") or descriptor.get("state") or ""
    return str(raw).strip().lower()


# Public descriptor readers + single-step building blocks, so a non-blocking caller
# (the app's poll-across-reruns live run, W18) can submit once and poll one tick per
# rerun without the blocking loop below. The blocking helper now composes these too.
def job_id_of(descriptor: dict) -> str | None:
    """The job id from a submit/status descriptor (tolerant of key spellings)."""
    return _job_id_from_descriptor(descriptor)


def status_of(descriptor: dict) -> str:
    """The lowercased status from a descriptor (``""`` when unknown)."""
    return _status_from_descriptor(descriptor)


def is_terminal(status: str) -> bool:
    """True once a job has reached a terminal status (completed/failed)."""
    return status in TERMINAL_STATUSES


def poll_once(client: SybilionClient, job_id: str) -> str:
    """One non-blocking status poll — the across-reruns building block (no sleep)."""
    return _status_from_descriptor(client.get_forecast(job_id))


def fetch_forecast_artifacts(client: SybilionClient, job_id: str) -> None:
    """Fetch + cache the four artifacts for a finished job under ``cache/<job_id>/``.
    Shared by the blocking helper and the non-blocking live run."""
    for name in FORECAST_ARTIFACTS:
        save_artifact(job_id, name, client.get_artifact(job_id, name))


def run_live_forecast(
    client: SybilionClient,
    payload: dict,
    *,
    poll_interval: float = 3.0,
    timeout: float = 900.0,
) -> str:
    """Submit a forecast, poll until it finishes, cache its artifacts, return the id.

    BLOCKING — used by the offline batch (``scripts/build_scenarios.py``) where a long
    wait is fine; the in-app live refresh uses the non-blocking submit/``poll_once``
    building blocks instead (W18). The default ``timeout`` is 15 minutes because a real
    Sybilion job can take ~11 minutes.

    Non-destructive by design: this fetches and saves the four artifacts under
    ``cache/<job_id>/`` but deliberately does **not** call :func:`set_latest_job`,
    so the pinned demo job remains the default and toggling live refresh off
    instantly restores the deterministic numbers.

    Raises ``RuntimeError`` if the submit returns no id or the job fails, and
    ``TimeoutError`` if it has not reached a terminal status within ``timeout``.
    """
    descriptor = client.submit_forecast(payload)
    job_id = _job_id_from_descriptor(descriptor)
    if not job_id:
        raise RuntimeError(f"Sybilion submit returned no job id: {descriptor!r}")

    deadline = time.monotonic() + timeout
    status = _status_from_descriptor(descriptor)
    while status not in TERMINAL_STATUSES:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Sybilion job {job_id} did not finish within {timeout:.0f}s "
                f"(last status: {status or 'unknown'})"
            )
        time.sleep(poll_interval)
        status = poll_once(client, job_id)

    if status == "failed":
        raise RuntimeError(f"Sybilion job {job_id} reported failure")

    fetch_forecast_artifacts(client, job_id)
    return job_id


# --------------------------------------------------------------------------- #
# TTF input series
# --------------------------------------------------------------------------- #
def load_ttf_series() -> dict:
    """Load the committed monthly TTF series document (meta + timeseries)."""
    return json.loads((config.DATA_DIR / "ttf_series.json").read_text())


def last_actual_price(ttf_document: dict | None = None) -> float:
    """The most recent observed monthly price — the spot anchor for drift."""
    document = ttf_document or load_ttf_series()
    timeseries = document["timeseries"]
    last_month = list(timeseries)[-1]
    return float(timeseries[last_month])


# --------------------------------------------------------------------------- #
# Artifact parsers
# --------------------------------------------------------------------------- #
def _quantile(entry: dict, level: str) -> float:
    return float(entry["quantile_forecast"][level])


def parse_forecast_months(forecast_json: dict) -> list[MonthForecast]:
    """Reduce forecast.json to the median / q10 / q90 the hedge policy needs."""
    series = forecast_json["data"]["forecast_series"]
    months: list[MonthForecast] = []
    for month in sorted(series):
        entry = series[month]
        months.append(
            MonthForecast(
                month=month,
                median=_quantile(entry, "0.50"),
                low=_quantile(entry, "0.10"),
                high=_quantile(entry, "0.90"),
            )
        )
    return months


def forecast_band_table(forecast_json: dict) -> list[dict]:
    """Full per-month quantile rows for charting: outer (q05/q95) and inner
    (q10/q90) bands plus the median and point forecast."""
    series = forecast_json["data"]["forecast_series"]
    rows: list[dict] = []
    for month in sorted(series):
        entry = series[month]
        rows.append(
            {
                "month": month,
                "forecast": float(entry.get("forecast", entry["quantile_forecast"]["0.50"])),
                "q05": _quantile(entry, "0.05"),
                "q10": _quantile(entry, "0.10"),
                "q50": _quantile(entry, "0.50"),
                "q90": _quantile(entry, "0.90"),
                "q95": _quantile(entry, "0.95"),
            }
        )
    return rows
