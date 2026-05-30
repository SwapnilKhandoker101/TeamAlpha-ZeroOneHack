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


def artifact_path(job_id: str, name: str) -> Path:
    name = name if name.endswith(".json") else f"{name}.json"
    return job_cache_dir(job_id) / name


def save_artifact(job_id: str, name: str, content: dict) -> Path:
    path = artifact_path(job_id, name)
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
