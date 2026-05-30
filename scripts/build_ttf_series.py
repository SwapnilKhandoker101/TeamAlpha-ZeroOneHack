"""Build the monthly TTF natural gas series for Sybilion.

Source: Yahoo Finance ``TTF=F`` — Dutch TTF natural gas front-month futures,
quoted in EUR/MWh. We take the monthly close, then fill the handful of months
Yahoo leaves empty (contract-roll / low-liquidity artifacts) by linear
interpolation so the series is contiguous, which Sybilion requires.

Run:  uv run python scripts/build_ttf_series.py
Output: data/ttf_series.json  (committed; this is the demo's source of truth)
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "ttf_series.json"

START = "2017-01-01"
TICKER = "TTF=F"


def fetch_monthly_close() -> pd.Series:
    raw = yf.download(TICKER, start=START, interval="1mo", progress=False, auto_adjust=True)
    close = raw["Close"][TICKER].dropna()
    close.index = pd.to_datetime(close.index).to_period("M")
    return close


def fill_gaps(close: pd.Series) -> tuple[pd.Series, list[str]]:
    """Reindex to a contiguous monthly range and linearly interpolate gaps."""
    full_range = pd.period_range(close.index[0], close.index[-1], freq="M")
    missing = [p.to_timestamp().strftime("%Y-%m-01") for p in full_range if p not in close.index]
    filled = close.reindex(full_range).interpolate(method="linear")
    return filled, missing


def main() -> None:
    close = fetch_monthly_close()
    filled, interpolated_months = fill_gaps(close)

    timeseries = {
        period.to_timestamp().strftime("%Y-%m-01"): round(float(value), 2)
        for period, value in filled.items()
    }

    document = {
        "meta": {
            "title": "European natural gas — Dutch TTF front-month futures, monthly close",
            "units": "EUR/MWh",
            "source": "Yahoo Finance TTF=F (Dutch TTF natural gas front-month futures)",
            "frequency": "monthly",
            "first_month": next(iter(timeseries)),
            "last_month": list(timeseries)[-1],
            "point_count": len(timeseries),
            "interpolated_months": interpolated_months,
            "generated_at": date.today().isoformat(),
            "note": (
                "Monthly close in EUR/MWh. Months absent from the Yahoo feed were "
                "linearly interpolated to keep the series contiguous (Sybilion rejects gaps). "
                "Interpolated months are listed for transparency."
            ),
        },
        "timeseries": timeseries,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(document, indent=2))
    print(f"Wrote {len(timeseries)} monthly points to {OUTPUT_PATH}")
    print(f"Range: {document['meta']['first_month']} .. {document['meta']['last_month']}")
    print(f"Interpolated {len(interpolated_months)} months: {interpolated_months}")


if __name__ == "__main__":
    main()
