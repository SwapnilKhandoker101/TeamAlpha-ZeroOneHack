"""Generate and commit the ceramics 4-factor mock forecast.

The ceramics demo's source of truth is a single committed artifact,
``cache/mock_ceramics_forecast.json``, mirroring how the gas demo runs off a
cached Sybilion job. This script builds it deterministically (no RNG):

* **gas** factor — the *real* cached TTF band from the pinned gas job, so the
  ceramics gas cost is grounded in the same Sybilion forecast the gas tab shows;
* **clay / power / shipping** factors — deterministic seasonal mock bands
  (fixed base level × per-quarter factor × a small monthly trend, with fixed
  relative band widths), over the *same* months as the gas factor;
* per-factor **external drivers** named after the real public datasets in the
  ceramics data-source doc (IMF/FRED clay, ENTSO-E/Eurostat power, Baltic Dry
  shipping, TTF/GIE/ENTSOG gas).

Run:  uv run python scripts/build_ceramics_forecast.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on the path

from ceramics_agent.catalog import quarter_of
from ceramics_agent.forecast import FACTOR_UNITS
from gas_agent import config
from gas_agent import sybilion_client as sc

GAS_JOB = "f445eec1-62e2-43e1-9995-18ba7ee668c3"  # the pinned cached gas forecast


def gas_factor_series() -> dict:
    """The gas factor straight from the real cached TTF forecast band."""
    forecast_json = sc.load_artifact(GAS_JOB, "forecast.json")
    series: dict[str, dict] = {}
    for row in sc.forecast_band_table(forecast_json):
        series[row["month"]] = {
            "quantile_forecast": {
                "0.05": round(row["q05"], 4),
                "0.10": round(row["q10"], 4),
                "0.50": round(row["q50"], 4),
                "0.90": round(row["q90"], 4),
                "0.95": round(row["q95"], 4),
            },
            "forecast": round(row["forecast"], 4),
        }
    return {"forecast_series": series}


def seasonal_factor_series(
    months: list[str],
    base: float,
    quarter_factor: dict[str, float],
    band10_frac: float,
    monthly_trend: float,
) -> dict:
    """A deterministic factor band over ``months``.

    ``median = base × quarter_factor[quarter] × (1 + monthly_trend × month_index)``.
    The 80% band is ``median × (1 ± band10_frac)``; the 90% band is 1.6× wider.
    Pure arithmetic — no randomness — so the committed artifact is reproducible.
    """
    band05_frac = band10_frac * 1.6
    series: dict[str, dict] = {}
    for index, month in enumerate(months):
        median = base * quarter_factor.get(quarter_of(month), 1.0) * (1 + monthly_trend * index)
        series[month] = {
            "quantile_forecast": {
                "0.05": round(median * (1 - band05_frac), 4),
                "0.10": round(median * (1 - band10_frac), 4),
                "0.50": round(median, 4),
                "0.90": round(median * (1 + band10_frac), 4),
                "0.95": round(median * (1 + band05_frac), 4),
            },
            "forecast": round(median, 4),
        }
    return {"forecast_series": series}


# Per-factor external drivers — named after the real public datasets in the
# ceramics data-source doc. Importances are fixed (deterministic), on Sybilion's
# 0-100 scale, descending by how directly each dataset drives the factor.
FACTOR_DRIVERS: dict[str, list[dict]] = {
    "gas": [
        {"name": "Dutch TTF front-month futures (ICE Endex)", "importance": 92.0, "source": "ICE / Yahoo TTF=F"},
        {"name": "EU gas storage fill level", "importance": 74.0, "source": "GIE AGSI+"},
        {"name": "Norwegian pipeline flows to the EU", "importance": 61.0, "source": "ENTSOG"},
    ],
    "clay": [
        {"name": "IMF kaolin & clay price index", "importance": 70.0, "source": "IMF Primary Commodity Prices"},
        {"name": "PPI: kaolin & ball clay mining", "importance": 58.0, "source": "FRED / BLS PPI"},
        {"name": "World clay & kaolin output", "importance": 44.0, "source": "World Mineral Statistics (BGS)"},
    ],
    "power": [
        {"name": "EU wholesale day-ahead electricity price", "importance": 80.0, "source": "ENTSO-E Transparency"},
        {"name": "EU electricity price index", "importance": 66.0, "source": "Eurostat"},
        {"name": "Heating-degree-days demand proxy", "importance": 49.0, "source": "Copernicus / Eurostat"},
    ],
    "shipping": [
        {"name": "Baltic Dry Index", "importance": 77.0, "source": "Baltic Exchange"},
        {"name": "EU freight transport cost index", "importance": 59.0, "source": "Eurostat"},
        {"name": "Road & container freight benchmark", "importance": 42.0, "source": "Eurostat COMEXT proxy"},
    ],
}


def build_artifact() -> dict:
    gas = gas_factor_series()
    months = sorted(gas["forecast_series"])  # all factors share the gas month grid

    factors = {
        "gas": gas,
        # Clay: a stable mineral — narrow band, gentle upward trend, near-flat season.
        "clay": seasonal_factor_series(
            months, base=0.18, quarter_factor={"Q4": 1.02}, band10_frac=0.06, monthly_trend=0.004
        ),
        # Power: winter-peaking and moderately volatile — Q4 heating lifts the level.
        "power": seasonal_factor_series(
            months,
            base=0.17,
            quarter_factor={"Q1": 1.08, "Q2": 0.95, "Q3": 0.95, "Q4": 1.15},
            band10_frac=0.12,
            monthly_trend=0.003,
        ),
        # Shipping: the most volatile non-gas factor — freight peaks into year-end.
        "shipping": seasonal_factor_series(
            months,
            base=0.12,
            quarter_factor={"Q3": 1.05, "Q4": 1.20},
            band10_frac=0.15,
            monthly_trend=0.0,
        ),
    }

    return {
        "meta": {
            "source": "mock",
            "built_from_gas_job": GAS_JOB,
            "units": FACTOR_UNITS,
            "note": "Deterministic ceramics 4-factor mock. Gas = real cached TTF band; "
            "clay/power/shipping = seasonal mock. Regenerate via scripts/build_ceramics_forecast.py.",
        },
        "factors": factors,
        "external_drivers": FACTOR_DRIVERS,
    }


def main() -> None:
    artifact = build_artifact()
    out = config.CERAMICS_MOCK_FORECAST
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2))
    months = sorted(artifact["factors"]["gas"]["forecast_series"])
    print(f"Wrote {out} ({out.stat().st_size} bytes)")
    print(f"  factors: {list(artifact['factors'])}")
    print(f"  months : {months[0]} … {months[-1]} ({len(months)} months)")
    for factor, block in artifact["factors"].items():
        first = block["forecast_series"][months[0]]["quantile_forecast"]
        bw = (first["0.90"] - first["0.10"]) / first["0.50"]
        print(f"  {factor:8s} {months[0]} q50={first['0.50']:.4f} band={bw:.0%} ({FACTOR_UNITS[factor]})")


if __name__ == "__main__":
    main()
