"""Build the committed scenario library (W17).

A live Sybilion forecast can take ~11 minutes, so we pre-fetch **real** forecasts for the
whole input space once, offline, and commit them under ``scenarios/`` — a clean checkout then
ships an instant, offline-reproducible, real-data demo.

The forecast depends only on **(product × gas_exposure)** (everything else is deterministic
downstream), so the library is a **9-cell grid of gas forecasts + one shared 4-factor ceramics
forecast**. This script is **idempotent and resumable**: re-running skips cells already present,
so the ~9 long jobs can be built over several sessions.

Usage
-----
    # No network — seed the (bowl, medium) cell from the committed cached job + mock,
    # so the library is immediately demoable offline. Safe to run anytime.
    uv run python scripts/build_scenarios.py --seed

    # Live — fetch every missing cell from Sybilion (needs SYBILION_API_KEY; slow).
    uv run python scripts/build_scenarios.py
    uv run python scripts/build_scenarios.py --only bowl:high dinnerware:low   # a subset
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on the path

from gas_agent import config
from gas_agent import scenarios as scn
from gas_agent import sybilion_client as sc
from gas_agent.keyword_agent import select_filters

from ceramics_agent import forecast as cforecast
from ceramics_agent.intake import CompanyProfile

# A short, distinct description per cell so the persona → Sybilion driver pick is well-grounded.
_GAS_PHRASE = {
    "low": "fires on an electric kiln, so natural-gas exposure is low",
    "medium": "uses gas firing as a significant but not dominant cost",
    "high": "is energy-intensive: gas firing is by far its largest volatile cost",
}
_PRODUCT_PHRASE = {
    "bowl": "a German pottery making handmade bowls",
    "dinnerware": "a German maker of dinnerware sets",
    "tile": "a German floor-tile works (high-temperature firing)",
}


def _profile(product: str, gas_exposure: str) -> CompanyProfile:
    desc = (f"{_PRODUCT_PHRASE[product]} that {_GAS_PHRASE[gas_exposure]}. "
            "Buys gas forward each quarter on the TTF market.")
    return CompanyProfile(product_id=product, gas_exposure=gas_exposure, description=desc,
                          source="scenario")


def _copy_job_into_library(job_id: str, dest_slug: str, names: tuple[str, ...]) -> None:
    """Copy a freshly-cached job's artifacts from cache/<job_id>/ into the committed
    scenarios/<dest_slug>/ (writes always land in cache; the library is the copy)."""
    dest = config.SCENARIOS_DIR / dest_slug
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = sc.cache_artifact_path(job_id, name)
        if src.exists():
            shutil.copyfile(src, dest / sc._artifact_name(name))


def _build_gas_cell(product: str, gas_exposure: str) -> None:
    slug = scn.slug_for(product, gas_exposure)
    if sc.scenario_artifact_path(slug, "forecast").exists():
        print(f"  {slug}: already built — skip")
        return
    persona = _profile(product, gas_exposure).persona()
    selection = select_filters(persona)
    ttf = sc.load_ttf_series()
    payload = sc.build_forecast_payload(
        ttf["timeseries"],
        title=ttf.get("meta", {}).get("title") or sc.DEFAULT_SERIES_TITLE,
        keywords=selection.keywords, category_ids=selection.category_ids,
        region_codes=selection.region_codes,
    )
    print(f"  {slug}: submitting gas forecast (this can take ~11 min)…")
    job_id = sc.run_live_forecast(sc.SybilionClient(), payload)  # blocking; 15-min timeout
    _copy_job_into_library(job_id, slug, sc.FORECAST_ARTIFACTS)
    print(f"  {slug}: cached real forecast (from job {job_id[:10]}…)")


def _build_shared_ceramics() -> None:
    dest = sc.scenario_artifact_path(scn.CERAMICS_SHARED_SLUG, "ceramics_forecast")
    if dest.exists():
        print(f"  {scn.CERAMICS_SHARED_SLUG}: ceramics forecast already built — skip")
        return
    print(f"  {scn.CERAMICS_SHARED_SLUG}: submitting the 4 ceramics factor forecasts…")
    combined = cforecast.run_live_ceramics_forecast(
        sc.SybilionClient(), cforecast.default_factor_history())
    src = sc.cache_artifact_path(combined, "ceramics_forecast")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    print(f"  {scn.CERAMICS_SHARED_SLUG}: cached shared 4-factor ceramics forecast")


def _seed_from_cache() -> None:
    """No-network seed: the committed cached gas job → the (bowl, medium) cell, and the
    committed mock ceramics → the shared cell. Makes the library demoable offline now."""
    latest = sc.get_latest_job()
    if latest and sc.cache_artifact_path(latest, "forecast").exists():
        _copy_job_into_library(latest, scn.slug_for("bowl", "medium"), sc.FORECAST_ARTIFACTS)
        print(f"  seeded scn-bowl-medium from cached job {latest[:10]}…")
    else:
        print("  no cached gas job to seed from — run a live build instead", file=sys.stderr)
    dest = sc.scenario_artifact_path(scn.CERAMICS_SHARED_SLUG, "ceramics_forecast")
    if config.CERAMICS_MOCK_FORECAST.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(config.CERAMICS_MOCK_FORECAST, dest)
        print("  seeded the shared ceramics forecast from the committed mock")


def _write_index() -> None:
    """Rebuild scenarios/index.json from whatever cells are present on disk."""
    entries: list[dict] = []
    for product in scn.PRODUCTS:
        for gas_exposure in scn.GAS_EXPOSURES:
            slug = scn.slug_for(product, gas_exposure)
            if sc.scenario_artifact_path(slug, "forecast").exists():
                entries.append({
                    "product": product, "gas_exposure": gas_exposure, "slug": slug,
                    "ceramics_ref": scn.CERAMICS_SHARED_SLUG,
                    "label": f"{product} · {gas_exposure}-gas",
                })
    config.SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
    scn.INDEX_PATH.write_text(json.dumps({"scenarios": entries}, indent=2))
    print(f"index.json: {len(entries)} scenario cell(s) committed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the committed scenario library (W17).")
    parser.add_argument("--seed", action="store_true",
                        help="No network: seed (bowl, medium) + shared ceramics from cache/mock.")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Build only these cells, e.g. bowl:high tile:low (live).")
    args = parser.parse_args()

    if args.seed:
        print("Seeding the scenario library from the committed cache + mock (no network)…")
        _seed_from_cache()
        _write_index()
        return

    if not config.have_sybilion_key():
        print("No SYBILION_API_KEY — cannot build live cells. Use --seed for an offline seed.",
              file=sys.stderr)
        sys.exit(1)

    cells = [(p, g) for p in scn.PRODUCTS for g in scn.GAS_EXPOSURES]
    if args.only:
        wanted = {tuple(spec.split(":", 1)) for spec in args.only}
        cells = [c for c in cells if c in wanted]

    print(f"Building {len(cells)} gas cell(s) + the shared ceramics forecast (resumable)…")
    _build_shared_ceramics()
    for product, gas_exposure in cells:
        _build_gas_cell(product, gas_exposure)
    _write_index()
    print("Done. Commit the scenarios/ directory to ship the library.")


if __name__ == "__main__":
    main()
