"""The Sybilion catalog, plus our view of what counts as a credible gas driver.

Two things live here:

* The real catalog IDs (categories and the regions that matter for European
  gas), copied from the Sybilion ``/categories`` and ``/regions`` endpoints.
  The keyword agent is handed these so it *selects* real IDs instead of
  inventing them.

* The credibility whitelist — the category x region combinations that can
  plausibly move the European gas price. This is the backbone of driver
  curation: a ranked driver that falls outside the whitelist (say,
  "Population - Belarus") is treated as a spurious correlation and rejected,
  while "Energy - Norway" or "Commodities - World" is kept.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Categories — full catalog (id -> name). id doubles as the filter code.
# --------------------------------------------------------------------------- #
CATEGORIES: dict[int, str] = {
    1: "Society and environment",
    2: "Economy",
    3: "Economic sectors and enterprises",
    4: "Labour",
    5: "Government",
    6: "Financial Data",
    7: "Population",
    8: "Education, research and culture",
    9: "Income, consumption and living conditions",
    10: "Health",
    11: "Chemicals",
    12: "Environment, Environmental Economic Accounting",
    13: "Traffic accidents",
    14: "Housing",
    15: "Sustainable development indicators",
    16: "Textile and fibres",
    17: "Global risk",
    18: "Wholesale and retail trade",
    19: "Minerals",
    20: "Processed food and consumables",
    23: "Construction",
    24: "Services",
    25: "Energy",
    26: "Accommodation and food service activities, tourism",
    27: "Industry, manufacturing",
    28: "Agriculture and forestry, fisheries",
    29: "Crafts",
    30: "Transport",
    31: "Enterprises",
    32: "Labour market",
    33: "Labour costs, non-wage costs",
    34: "Earnings",
    35: "Justice",
    36: "Public finance",
    37: "Taxes",
    38: "Public service",
    39: "Bureaucracy costs",
    40: "Equities",
    41: "Bonds",
    42: "Market Indices",
    43: "Trade Performance",
    44: "Sector Indices",
    45: "Interest Rates",
    46: "Commodities",
    47: "Exchange Rates",
    48: "Investment Funds",
}

# --------------------------------------------------------------------------- #
# Regions — the subset that can plausibly drive European gas (code -> name).
# Codes are the Sybilion region `code` field (ISO numeric for countries,
# 1001-1006 for the world/continents).
# --------------------------------------------------------------------------- #
REGIONS: dict[int, str] = {
    1001: "World",
    1003: "Europe",
    276: "Germany",
    528: "Netherlands",
    56: "Belgium",
    250: "France",
    380: "Italy",
    724: "Spain",
    616: "Poland",
    40: "Austria",
    203: "Czechia",
    348: "Hungary",
    578: "Norway",  # #1 pipeline supplier to the EU
    826: "United Kingdom",
    643: "Russian Federation",  # historical supplier, key geopolitical driver
    804: "Ukraine",  # transit + war risk
    840: "United States of America",  # LNG exporter
    634: "Qatar",  # LNG exporter
    12: "Algeria",  # pipeline + LNG supplier to southern Europe
    364: "Iran (Islamic Republic of)",  # Strait of Hormuz / geopolitical risk
    792: "Turkey",  # transit hub
}

# --------------------------------------------------------------------------- #
# Credibility whitelist for European gas
# --------------------------------------------------------------------------- #
# Categories whose datasets can plausibly correlate with the gas price.
CREDIBLE_CATEGORY_IDS: set[int] = {
    25,  # Energy — the obvious one (power, gas, coal, oil products)
    46,  # Commodities — oil, coal, carbon, correlated energy commodities
    47,  # Exchange Rates — LNG is priced in USD; EUR/USD matters
    17,  # Global risk — geopolitical / supply-shock signals
    45,  # Interest Rates — macro demand + cost of carry / storage financing
    40,  # Equities — energy-sector equities track the complex
    27,  # Industry, manufacturing — industrial gas demand
    19,  # Minerals — adjacent extractive commodities
    11,  # Chemicals — gas-intensive industry, demand proxy
}

# Regions whose datasets can plausibly relate to European gas supply/demand.
CREDIBLE_REGION_CODES: set[int] = {
    1001,  # World
    1003,  # Europe
    276,  # Germany
    528,  # Netherlands
    56,  # Belgium
    250,  # France
    380,  # Italy
    724,  # Spain
    616,  # Poland
    40,  # Austria
    203,  # Czechia
    348,  # Hungary
    578,  # Norway
    826,  # United Kingdom
    643,  # Russian Federation
    804,  # Ukraine
    840,  # United States of America
    634,  # Qatar
    12,  # Algeria
    364,  # Iran
    792,  # Turkey
}

# --------------------------------------------------------------------------- #
# Default forecast filters for the TTF gas demo (the curated starting point the
# keyword agent refines from).
# --------------------------------------------------------------------------- #
DEFAULT_FORECAST_CATEGORIES: list[int] = [25, 46, 47, 17, 45]
DEFAULT_FORECAST_REGIONS: list[int] = [1003, 276, 528, 56, 578, 643, 804, 840]


def category_name(category_id: int) -> str:
    return CATEGORIES.get(category_id, f"category {category_id}")


def region_name(region_code: int) -> str:
    return REGIONS.get(region_code, f"region {region_code}")


def is_credible_category(category_id: int | None) -> bool:
    return category_id is not None and category_id in CREDIBLE_CATEGORY_IDS


def is_credible_region(region_code: int | None) -> bool:
    # An unknown / missing region is not grounds for rejection on its own;
    # category credibility carries more weight for gas. Treat None as neutral.
    if region_code is None:
        return True
    return region_code in CREDIBLE_REGION_CODES
