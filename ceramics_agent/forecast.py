"""The 4-factor cost forecast — mock by default, live Sybilion as opt-in.

The ceramics lock decision rides on four input-cost factors, each with its own
probabilistic band:

    gas       — kiln firing gas (Dutch TTF, EUR/MWh)        — the dominant cost
    clay      — body clay / kaolin (EUR/kg)
    power     — grid electricity for the kiln (EUR/kWh)
    shipping  — inbound/outbound freight (EUR/kg)

Each factor is stored in the **same quantile shape Sybilion emits** — a
``forecast_series`` of months, each with a ``quantile_forecast`` dict — so the
same :class:`gas_agent.hedge_policy.MonthForecast` reducer the gas agent uses
parses them unchanged.

Source of truth, like the gas agent, is a **committed mock artifact**
(``cache/mock_ceramics_forecast.json``, built by
``scripts/build_ceramics_forecast.py`` — its gas factor is the *real* cached TTF
band, the other three are deterministic seasonal mocks). The demo therefore runs
with no keys and identical numbers every time.

An **opt-in live path** (:func:`run_live_ceramics_forecast`) submits one Sybilion
job per factor — reusing the gas agent's :func:`gas_agent.sybilion_client.build_forecast_payload`
and :func:`~gas_agent.sybilion_client.run_live_forecast` verbatim — and assembles
the four results into a single ``ceramics_forecast.json`` under a fresh job dir.
It is off the default path, guarded by ``SYBILION_API_KEY``, and (like the gas
live refresh) **never repoints ``latest_job.txt``**.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass

from gas_agent import config
from gas_agent import sybilion_client as sc
from gas_agent.hedge_policy import MonthForecast

# The four cost factors, in display order. "gas" first — it is the dominant,
# most-volatile cost and the one whose band most moves the lock decision.
FACTORS: tuple[str, ...] = ("gas", "clay", "power", "shipping")

# Native price units per factor, for labels and the (unit-aware) physical-cost
# conversion in cost_policy. Kept here so the artifact stays faithful to source
# (gas is EUR/MWh straight off TTF, never silently rescaled).
FACTOR_UNITS: dict[str, str] = {
    "gas": "EUR/MWh",
    "clay": "EUR/kg",
    "power": "EUR/kWh",
    "shipping": "EUR/kg",
}


@dataclass(frozen=True)
class FactorDriver:
    """One external driver behind a factor's forecast (for display / curation parity)."""

    name: str
    importance: float  # 0..100, mirrors Sybilion's importance scale
    source: str  # short provenance string (the dataset it came from)
    factor: str  # which factor it explains


# --------------------------------------------------------------------------- #
# Loading the committed mock / cached live artifact
# --------------------------------------------------------------------------- #
def _ceramics_artifact_path(job_id: str | None):
    """Resolve which artifact to read and label its source.

    Returns ``(path, source, job_label)``. A live job whose
    ``cache/<job>/ceramics_forecast.json`` exists wins; otherwise we fall back to
    the committed mock — exactly the gas agent's cache-first, offline-safe pattern.
    """
    if job_id:
        live_path = sc.artifact_path(job_id, "ceramics_forecast")
        if live_path.exists():
            return live_path, "live", job_id
    return config.CERAMICS_MOCK_FORECAST, "mock", "mock"


def load_ceramics_artifact(job_id: str | None = None) -> tuple[dict, str, str]:
    """Read the raw 4-factor artifact dict plus ``(source, job_label)``."""
    import json

    path, source, job_label = _ceramics_artifact_path(job_id)
    return json.loads(path.read_text()), source, job_label


def _quantile(entry: dict, level: str) -> float:
    return float(entry["quantile_forecast"][level])


def parse_factor_months(factor_json: dict) -> list[MonthForecast]:
    """Reduce one factor's ``forecast_series`` to the median / q10 / q90 the policy
    needs — the exact reducer shape :func:`gas_agent.sybilion_client.parse_forecast_months`
    produces, so a factor band and the gas band are interchangeable downstream."""
    series = factor_json["forecast_series"]
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


def factor_band_table(factor_json: dict) -> list[dict]:
    """Full per-month quantile rows for one factor (charting / debug)."""
    series = factor_json["forecast_series"]
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


def load_ceramics_forecast(
    job_id: str | None = None,
) -> tuple[dict[str, list[MonthForecast]], str]:
    """Load the 4-factor forecast as ``{factor: [MonthForecast, ...]}`` + source.

    ``source`` is ``"live"`` (a cached Sybilion 4-job run) or ``"mock"`` (the
    committed artifact) — the dashboard shows an offline banner when it is mock.
    This is the documented entry point the cost policy consumes.
    """
    raw, source, _ = load_ceramics_artifact(job_id)
    factors = raw["factors"]
    parsed = {factor: parse_factor_months(factors[factor]) for factor in FACTORS if factor in factors}
    return parsed, source


def load_factor_drivers(job_id: str | None = None) -> dict[str, list[FactorDriver]]:
    """Per-factor external drivers, for the dashboard's "what's behind each factor" view."""
    raw, _, _ = load_ceramics_artifact(job_id)
    drivers_block = raw.get("external_drivers", {})
    out: dict[str, list[FactorDriver]] = {}
    for factor in FACTORS:
        entries = drivers_block.get(factor, [])
        out[factor] = [
            FactorDriver(
                name=str(entry.get("name", "")),
                importance=float(entry.get("importance", 0.0)),
                source=str(entry.get("source", "")),
                factor=factor,
            )
            for entry in entries
        ]
    return out


# --------------------------------------------------------------------------- #
# Opt-in live path — one Sybilion job per factor, assembled into one artifact
# --------------------------------------------------------------------------- #
# Per-factor Sybilion request scaffolding. Titles satisfy the API's 20-511-char
# rule; category/region ids are real Sybilion catalog codes from gas_agent.catalog.
FACTOR_REQUEST_META: dict[str, dict] = {
    "gas": {
        "title": "Kiln firing gas cost — Dutch TTF front-month futures, monthly close in EUR/MWh",
        "keywords": ["natural gas", "TTF", "energy price"],
        "category_ids": [25, 46],  # Energy, Commodities
        "region_codes": [1003, 276, 528],  # Europe, Germany, Netherlands
    },
    "clay": {
        "title": "Ceramic body clay & kaolin raw-material cost index for an EU ceramics maker",
        "keywords": ["kaolin", "clay", "minerals", "raw materials"],
        "category_ids": [19, 46],  # Minerals, Commodities
        "region_codes": [1003, 276],
    },
    "power": {
        "title": "Industrial grid electricity cost for kiln operation, EU wholesale power prices",
        "keywords": ["electricity", "power price", "energy"],
        "category_ids": [25],  # Energy
        "region_codes": [1003, 276],
    },
    "shipping": {
        "title": "Inbound and outbound freight cost index for an EU ceramics manufacturer",
        "keywords": ["freight", "shipping", "transport", "logistics"],
        "category_ids": [30, 46],  # Transport, Commodities
        "region_codes": [1003, 276],
    },
}


# --------------------------------------------------------------------------- #
# Cache-first plumbing — a deterministic combined-job id so re-submitting the same
# profile reuses the cached live artifact instead of re-polling 4 jobs (~1-4 min).
# --------------------------------------------------------------------------- #
def combined_job_id(signature: str) -> str:
    """A stable combined-job id derived from a signature (the company description +
    the chosen Sybilion filters). Same signature → same job dir → cache hit, so a
    live forecast is polled once per distinct profile and instant on every reload."""
    salt = getattr(config, "CERAMICS_CACHE_SALT", "ceramics-v1")
    digest = hashlib.sha1(f"{salt}|{signature}".encode("utf-8")).hexdigest()[:12]
    return f"ceramics-{digest}"


def live_artifact_exists(job_id: str | None) -> bool:
    """True when a combined live artifact is already cached for ``job_id``."""
    if not job_id:
        return False
    return sc.artifact_path(job_id, "ceramics_forecast").exists()


def ensure_ceramics_forecast(
    client: sc.SybilionClient,
    base_series_by_factor: dict[str, dict],
    *,
    signature: str,
    refresh: bool = False,
    soft_horizon: int = 6,
) -> str:
    """Cache-first live forecast. Returns the cached combined job for this signature
    when present (instant, deterministic for the session), otherwise runs the four
    live jobs and caches them under the signature's deterministic id. ``refresh=True``
    forces a re-poll (e.g. a chat impact invalidated a factor)."""
    job_id = combined_job_id(signature)
    if not refresh and live_artifact_exists(job_id):
        return job_id
    return run_live_ceramics_forecast(
        client, base_series_by_factor, job_id=job_id, soft_horizon=soft_horizon
    )


def run_live_ceramics_forecast(
    client: sc.SybilionClient,
    base_series_by_factor: dict[str, dict],
    *,
    job_id: str | None = None,
    soft_horizon: int = 6,
) -> str:
    """Forecast all four factors live and cache one combined ``ceramics_forecast.json``.

    For each factor it builds a Sybilion payload (reusing the gas agent's
    :func:`~gas_agent.sybilion_client.build_forecast_payload`) from that factor's
    historical ``{month: price}`` series in ``base_series_by_factor`` and runs the
    same submit→poll→cache loop (:func:`~gas_agent.sybilion_client.run_live_forecast`).
    The four per-factor forecasts are then assembled into a single artifact under a
    combined job dir (``job_id`` when given — the cache-first deterministic id —
    else a fresh random one).

    Non-destructive by design (mirrors the gas live refresh): it caches under
    ``cache/<combined_job>/`` and **never** calls ``set_latest_job``, so toggling
    live off instantly restores the committed mock. Returns the combined job id.
    """
    factors_block: dict[str, dict] = {}
    drivers_block: dict[str, list] = {}

    for factor in FACTORS:
        series = base_series_by_factor.get(factor)
        if not series:
            continue
        meta = FACTOR_REQUEST_META[factor]
        payload = sc.build_forecast_payload(
            series,
            title=meta["title"],
            keywords=meta["keywords"],
            category_ids=meta["category_ids"],
            region_codes=meta["region_codes"],
            soft_horizon=soft_horizon,
        )
        factor_job = sc.run_live_forecast(client, payload)
        forecast_json = sc.load_artifact(factor_job, "forecast.json")
        factors_block[factor] = {"forecast_series": forecast_json["data"]["forecast_series"]}
        drivers_block[factor] = _live_drivers_for(factor, factor_job)

    combined_job = job_id or f"ceramics-{uuid.uuid4().hex[:8]}"
    artifact = {
        "meta": {"source": "live", "factors": list(factors_block), "units": FACTOR_UNITS},
        "factors": factors_block,
        "external_drivers": drivers_block,
    }
    sc.save_artifact(combined_job, "ceramics_forecast", artifact)
    return combined_job


def _live_drivers_for(factor: str, factor_job: str) -> list[dict]:
    """Best-effort: read a factor job's external_signals (if the live run cached it)
    into the artifact's driver shape. Silent fallback to an empty list keeps the
    assembly robust when a factor job carried no signals artifact."""
    try:
        signals = sc.load_artifact(factor_job, "external_signals.json")
    except Exception:
        return []
    data = signals.get("data", {}) if isinstance(signals, dict) else {}
    drivers: list[dict] = []
    for entry in list(data.values())[:5]:
        if not isinstance(entry, dict):
            continue
        importance = entry.get("importance", {})
        overall = importance.get("overall", {}) if isinstance(importance, dict) else {}
        drivers.append(
            {
                "name": str(entry.get("driver_name", "")),
                "importance": float(overall.get("mean", 0.0)) if isinstance(overall, dict) else 0.0,
                "source": "Sybilion live",
            }
        )
    return drivers


def default_factor_history() -> dict[str, dict]:
    """Deterministic historical input series per factor for the opt-in live path.

    Gas reuses the real committed TTF history; clay/power/shipping use short
    synthetic histories anchored on the mock's base levels (so a live submit has
    a credible series to forecast from). Off the demo path — only invoked when a
    user opts into a live refresh with a Sybilion key.
    """
    ttf = sc.load_ttf_series()["timeseries"]
    months = list(ttf)[-24:]  # last two years is plenty of history for Sybilion

    def synth(base: float, quarter_factor: dict[str, float]) -> dict:
        return {
            m: round(base * quarter_factor.get(gas_catalog_quarter(m), 1.0), 4) for m in months
        }

    return {
        "gas": {m: ttf[m] for m in months},
        "clay": synth(0.18, {"Q1": 1.0, "Q2": 1.0, "Q3": 1.0, "Q4": 1.02}),
        "power": synth(0.17, {"Q1": 1.08, "Q2": 0.95, "Q3": 0.95, "Q4": 1.15}),
        "shipping": synth(0.12, {"Q1": 1.0, "Q2": 1.0, "Q3": 1.05, "Q4": 1.20}),
    }


def gas_catalog_quarter(month: str) -> str:
    """Quarter of a 'YYYY-MM-01' month — local helper so this module needs no
    ceramics_agent.catalog import (keeps the dependency arrow clean)."""
    month_number = int(month.split("-")[1])
    return f"Q{(month_number - 1) // 3 + 1}"
