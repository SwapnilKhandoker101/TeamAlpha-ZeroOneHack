#!/usr/bin/env python3
import json
import os
import sys
import uuid

try:
    import requests
except ImportError:
    sys.exit("This script needs `requests`.  Install it with:  pip install requests")


# ===========================================================================
# CONFIG  — the only two things you may need to confirm from your dashboard
# ===========================================================================
BASE_URL = os.environ.get("SYBILION_BASE_URL").rstrip("/")
DRIVERS_PATH = "/api/v1/drivers"

API_KEY = os.environ.get("SYBILION_API_KEY")
# Auth header scheme. Most likely a Bearer token; flip to "x-api-key" if your
# docs specify that instead.
AUTH_SCHEME = "bearer"            # "bearer"  ->  Authorization: Bearer <key>
#                                 # "apikey"  ->  x-api-key: <key>


# ===========================================================================
# STEP 1 — INPUTS / FACTORS  (edit freely)
# ===========================================================================
# Category IDs come from the live GET /api/v1/categories listing.
# Focus = raw materials + labour; sector context keeps the engine on-domain.
CATEGORIES = {
    19: "Minerals",                 # clay, kaolin, ball clay, feldspar, silica  <-- RAW MATERIALS
    33: "Labour costs, non-wage costs",  # wage / on-costs                        <-- LABOUR
    34: "Earnings",                 # wages level                                 <-- LABOUR
    23: "Construction",             # demand pull for structural ceramics (context)
    27: "Industry, manufacturing",  # output / activity (context)
    25: "Energy",                 # the dominant cost driver for ceramics —
    46: "Commodities",            # benchmark commodity prices
}

# Region codes come from the live GET /api/v1/regions listing (use the `code` field).
REGIONS = {
    1003: "Europe",                 # Italy/Spain heartland of structural ceramics
    # 380: "Italy", 724: "Spain", 792: "Turkey", 76: "Brazil", 1001: "World",
}

# Keyword hints steer which driver datasets the engine considers.
KEYWORDS = [
    # raw materials (structural ceramics body + fluxes + refractories)
    "clay", "kaolin", "ball clay", "feldspar", "silica", "alumina", "zircon",
    # labour cost
    "labour cost", "manufacturing wages", "unit labour cost",
    # structural-ceramics sector / products
    "structural ceramics", "ceramic tiles", "sanitaryware", "bricks",
    "refractories", "clay building materials",
]

TITLE = "Structural ceramics: raw-material and labour-cost driver discovery"
RECENCY_FACTOR = 0.4   # slow-moving cost series -> lean toward historical signal (0.0=full archive, 1.0=last ~5 days)
LIMIT = 150            # candidate drivers retrieved before ranking; also the pre-charge ceiling


def build_request() -> dict:
    """Assemble the RecommendRequestV1 body."""
    return {
        "version": "v1",
        "recency_factor": RECENCY_FACTOR,
        "timeseries_metadata": {
            "title": TITLE,
            "keywords": KEYWORDS,
            # time-series data is optional for the drivers endpoint; we omit it
            # because we only want the factor ranking, not a forecast.
        },
        "filters": {
            "categories": sorted(CATEGORIES),
            "regions": sorted(REGIONS),
            "limit": LIMIT,
        },
    }


# ===========================================================================
# STEP 2 — CALL THE API, SHOW RAW DATA + DRIVING FACTORS
# ===========================================================================
def auth_headers() -> dict:
    if not API_KEY:
        sys.exit("Set your key first:  export SYBILION_API_KEY=...")
    if AUTH_SCHEME == "apikey":
        return {"x-api-key": API_KEY}
    return {"Authorization": f"Bearer {API_KEY}"}


def call_drivers(body: dict) -> requests.Response:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Request-ID": str(uuid.uuid4()),  # send the SAME id when retrying (idempotency)
        **auth_headers(),
    }
    return requests.post(BASE_URL + DRIVERS_PATH, headers=headers, json=body, timeout=90)


def explain_status(resp: requests.Response) -> None:
    """Friendly messages for the documented error codes."""
    msgs = {
        402: "Insufficient credits for the worst-case (LIMIT). Top up or lower LIMIT.",
        422: "Validation error. Look at details[0] in the raw body above.",
        429: "Rate limited (per-minute cap on billed calls). Wait, then retry.",
        502: "Engine transport error. Retry with the SAME X-Request-ID.",
        503: "Drivers feature not enabled for this account "
             "(you can still get driver attribution from a forecast's external_signals.json).",
    }
    if resp.status_code in msgs:
        print(f"\n[{resp.status_code}] {msgs[resp.status_code]}", file=sys.stderr)


def _importance(d: dict):
    v = d.get("importance")
    return v if isinstance(v, (int, float)) else float("-inf")


def print_driving_factors(data: dict) -> None:
    """Pretty-print the ranked driving factors. Tolerant of field-name variation."""
    drivers = data.get("drivers") or data.get("items") or data.get("results") or []
    if not isinstance(drivers, list) or not drivers:
        print("\n(No 'drivers' array found — inspect the raw JSON above.)")
        return

    drivers = sorted(drivers, key=_importance, reverse=True)

    print("\n" + "=" * 78)
    print("DRIVING FACTORS (ranked by importance)")
    print("=" * 78)
    print(f"{'#':>3}  {'importance':>11}  {'dir':>5}   driver")
    print("-" * 78)
    for i, d in enumerate(drivers, 1):
        name = d.get("driver_name") or d.get("name") or d.get("title") or "?"
        imp = d.get("importance", "?")
        direction = d.get("direction", d.get("sign", "?"))
        imp_str = f"{imp:.4f}" if isinstance(imp, (int, float)) else str(imp)
        print(f"{i:>3}  {imp_str:>11}  {str(direction):>5}   {name}")

    # show correlation columns too, if the engine returns them
    extra_keys = [k for k in ("correlation", "lag", "category", "region")
                  if any(k in d for d in drivers)]
    if extra_keys:
        print("\n(Each driver also carries: " + ", ".join(extra_keys) + " — see raw JSON.)")
    print(f"\nTotal driving factors returned: {len(drivers)}")


def main() -> None:
    body = build_request()

    print("REQUEST BODY")
    print("-" * 78)
    print(json.dumps(body, indent=2))
    print(f"\nPOST {BASE_URL + DRIVERS_PATH}")

    resp = call_drivers(body)
    print(f"HTTP {resp.status_code}")

    # ---- raw data ----
    try:
        data = resp.json()
    except ValueError:
        print("\nRAW RESPONSE (non-JSON):")
        print(resp.text[:4000])
        explain_status(resp)
        return

    print("\nRAW RESPONSE")
    print("-" * 78)
    print(json.dumps(data, indent=2)[:8000])  # truncate very large bodies

    explain_status(resp)

    # ---- driving factors ----
    if resp.ok:
        print_driving_factors(data)


if __name__ == "__main__":
    main()