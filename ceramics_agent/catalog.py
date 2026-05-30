"""The ceramics business catalog — products, suppliers, channels, and history.

This is the deterministic, committed "world" the optimizer reasons over. It is
the ceramics analogue of :mod:`gas_agent.catalog`: static module-level data the
rest of the package treats as read-only, so the same inputs always produce the
same recommendation and the demo runs with no keys and no network.

Four things live here:

* :data:`PRODUCTS` — three products, each with a **bill of materials** (BOM):
  how much clay, glaze, kiln electricity, kiln **gas**, and shipping weight one
  unit consumes. The BOM is what turns the four forecast factors
  (gas / clay / power / shipping) into a real EUR-per-unit cost downstream.
* :data:`SUPPLIERS` — three raw-material suppliers, each with per-component
  ``price_factors`` (multipliers vs. a 1.0 baseline), reliability, and lead time.
* :data:`CHANNELS` — three sales channels, each with a target margin, a minimum
  order, and a per-quarter seasonality multiplier.
* :data:`HISTORICAL_SALES` — twelve months of past sales the backtest replays.

The whitelist / blacklist tuples mirror :func:`gas_agent.driver_curation.classify_driver`:
substring rules that let :mod:`ceramics_agent.curation` keep credible ceramics
suppliers / channels and reject off-domain ones (a textile or food supplier),
deterministically and with a one-line reason.

Note on the BOM: gas is modelled as an explicit ``firing_gas_kwh`` field rather
than being folded into ``kiln_kwh``. The persona's whole point is that **gas is
the largest volatile cost**, so it must be a first-class, separately-forecast
input — kept distinct from grid electricity (``kiln_kwh`` → the *power* factor).
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Products + bill of materials (BOM)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Product:
    """One sellable product and the per-unit resources it consumes.

    Quantities are per single unit (one bowl, one 6-piece set, one square metre
    of tile). ``glaze_kg`` is carried for supplier-specialty matching and realism;
    the forecast-driven physical cost itself uses the four forecast factors
    (clay, kiln power, firing gas, shipping) — see :mod:`ceramics_agent.cost_policy`.
    """

    id: str
    name: str
    clay_kg: float  # body clay per unit -> the "clay" factor
    glaze_kg: float  # glaze per unit (specialty match + realism; not a forecast factor)
    kiln_kwh: float  # grid electricity per unit -> the "power" factor
    firing_gas_kwh: float  # kiln gas burned per unit -> the "gas" factor (the dominant cost)
    ship_kg: float  # shipped weight per unit -> the "shipping" factor


PRODUCTS: dict[str, Product] = {
    "bowl": Product(
        id="bowl",
        name="Handmade Bowl",
        clay_kg=0.5,
        glaze_kg=0.10,
        kiln_kwh=2.0,
        firing_gas_kwh=6.0,
        ship_kg=0.7,
    ),
    "dinnerware": Product(
        id="dinnerware",
        name="6-Piece Dinnerware Set",
        clay_kg=3.0,
        glaze_kg=0.6,
        kiln_kwh=8.0,
        firing_gas_kwh=22.0,
        ship_kg=4.5,
    ),
    "tile": Product(
        id="tile",
        name="Floor Tile (1 m²)",
        clay_kg=25.0,
        glaze_kg=2.0,
        kiln_kwh=15.0,
        firing_gas_kwh=120.0,  # high-temperature firing — the most gas-intensive product
        ship_kg=30.0,
    ),
}


# --------------------------------------------------------------------------- #
# Suppliers
# --------------------------------------------------------------------------- #
# Physical cost components a supplier's price factors can apply to. Keys missing
# from a supplier's ``price_factors`` default to 1.0 (the neutral baseline).
COST_COMPONENTS: tuple[str, ...] = ("clay", "glaze", "gas", "power", "shipping")


@dataclass(frozen=True)
class Supplier:
    """A raw-material supplier.

    ``price_factors`` are multipliers against a 1.0 market baseline, per cost
    component (see :data:`COST_COMPONENTS`); a value below 1.0 is cheaper than
    market, above 1.0 dearer. ``reliability_pct`` is on-time/in-full delivery,
    ``lead_time_days`` is quoted lead time. ``specialties`` feed the credibility
    classifier in :mod:`ceramics_agent.curation`.
    """

    id: str
    name: str
    region: str  # ISO-ish short region code, display only
    specialties: tuple[str, ...]
    reliability_pct: float  # 0..100
    lead_time_days: int
    price_factors: dict[str, float] = field(default_factory=dict)

    def factor(self, component: str) -> float:
        """The price multiplier for one cost component (1.0 when unspecified)."""
        return float(self.price_factors.get(component, 1.0))

    def average_price_factor(self) -> float:
        """Mean multiplier across all cost components — the supplier's overall
        price level, used by curation's cost score. Components the supplier does
        not specify count as the 1.0 baseline so every supplier is comparable."""
        return sum(self.factor(c) for c in COST_COMPONENTS) / len(COST_COMPONENTS)


SUPPLIERS: dict[str, Supplier] = {
    "alpine": Supplier(
        id="alpine",
        name="Alpine Clay Works",
        region="AT",  # Austria — near the German plant
        specialties=("clay", "kaolin clay", "refractory materials"),
        reliability_pct=96.0,
        lead_time_days=7,
        # Balanced, near-market pricing; slight shipping edge for proximity.
        price_factors={"clay": 1.0, "glaze": 1.0, "gas": 1.0, "power": 1.0, "shipping": 0.95},
    ),
    "eastern": Supplier(
        id="eastern",
        name="Eastern European Materials",
        region="PL",  # Poland — cheapest materials, dearer/slower logistics
        specialties=("clay", "feldspar", "raw materials", "minerals"),
        reliability_pct=92.0,
        lead_time_days=14,
        price_factors={"clay": 0.85, "glaze": 0.90, "gas": 0.95, "power": 0.95, "shipping": 1.15},
    ),
    "premium": Supplier(
        id="premium",
        name="Premium Ceramics Supply",
        region="IT",  # Italy — most reliable, premium glaze/colours
        specialties=("glaze", "ceramic colours", "kiln furniture", "heat-resistant materials"),
        reliability_pct=98.0,
        lead_time_days=10,
        price_factors={"clay": 1.05, "glaze": 1.15, "gas": 1.0, "power": 1.0, "shipping": 1.05},
    ),
}


# --------------------------------------------------------------------------- #
# Sales channels
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SalesChannel:
    """A route to market.

    ``target_margin`` is the channel's typical gross margin (fraction of cost the
    buyer is willing to pay on top). ``min_order`` is the minimum order quantity
    that channel accepts. ``seasonality`` maps quarter ("Q1".."Q4") to a demand
    multiplier (1.0 = baseline), which feeds both channel scoring and the demand
    side of negotiation.
    """

    id: str
    name: str
    channel_type: str
    min_order: int
    target_margin: float  # 0..1, fraction over cost
    seasonality: dict[str, float] = field(default_factory=dict)

    def season_factor(self, quarter: str) -> float:
        """Demand multiplier for a quarter (1.0 when unspecified)."""
        return float(self.seasonality.get(quarter, 1.0))


CHANNELS: dict[str, SalesChannel] = {
    "wholesale": SalesChannel(
        id="wholesale",
        name="European Ceramics Wholesale",
        channel_type="wholesale",
        min_order=100,
        target_margin=0.35,
        seasonality={"Q1": 0.9, "Q2": 1.0, "Q3": 1.0, "Q4": 1.3},  # builds into year-end
    ),
    "hospitality": SalesChannel(
        id="hospitality",
        name="Hospitality & Restaurant Supply",
        channel_type="hospitality",
        min_order=50,
        target_margin=0.45,
        seasonality={"Q1": 0.9, "Q2": 1.1, "Q3": 1.3, "Q4": 1.0},  # summer refit season
    ),
    "online": SalesChannel(
        id="online",
        name="Online Retail Export",
        channel_type="online retail export",
        min_order=1,
        target_margin=0.80,
        seasonality={"Q1": 0.8, "Q2": 0.9, "Q3": 1.0, "Q4": 1.6},  # holiday gifting peak
    ),
}


# --------------------------------------------------------------------------- #
# Historical sales — what the backtest replays
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SalesRecord:
    """One past month of sales for one product.

    ``channel_mix`` maps channel id -> share of that month's units (sums to 1.0).
    ``achieved_margin`` is the gross margin actually realized that month — the
    historical reference the backtest shows alongside each strategy's replay.
    """

    month: str  # "YYYY-MM"
    product_id: str
    units: int
    channel_mix: dict[str, float]
    achieved_margin: float  # 0..1, realized gross margin


# Twelve months ending at the demo "today" (2026-05). Hand-set, deterministic,
# no RNG. Tile volume swells into Q4 (wholesale + online seasonality); achieved
# margins drift with the channel mix. Every channel_mix sums to exactly 1.0.
HISTORICAL_SALES: list[SalesRecord] = [
    SalesRecord("2025-06", "tile", 4200, {"wholesale": 0.6, "hospitality": 0.3, "online": 0.1}, 0.31),
    SalesRecord("2025-07", "dinnerware", 1800, {"wholesale": 0.4, "hospitality": 0.5, "online": 0.1}, 0.38),
    SalesRecord("2025-08", "tile", 3900, {"wholesale": 0.5, "hospitality": 0.4, "online": 0.1}, 0.33),
    SalesRecord("2025-09", "tile", 4600, {"wholesale": 0.7, "hospitality": 0.2, "online": 0.1}, 0.30),
    SalesRecord("2025-10", "bowl", 5200, {"wholesale": 0.5, "hospitality": 0.2, "online": 0.3}, 0.46),
    SalesRecord("2025-11", "tile", 5400, {"wholesale": 0.7, "hospitality": 0.1, "online": 0.2}, 0.34),
    SalesRecord("2025-12", "bowl", 6100, {"wholesale": 0.4, "hospitality": 0.1, "online": 0.5}, 0.52),
    SalesRecord("2026-01", "dinnerware", 1500, {"wholesale": 0.5, "hospitality": 0.4, "online": 0.1}, 0.36),
    SalesRecord("2026-02", "tile", 3600, {"wholesale": 0.6, "hospitality": 0.3, "online": 0.1}, 0.32),
    SalesRecord("2026-03", "tile", 4100, {"wholesale": 0.6, "hospitality": 0.3, "online": 0.1}, 0.33),
    SalesRecord("2026-04", "dinnerware", 2000, {"wholesale": 0.4, "hospitality": 0.5, "online": 0.1}, 0.39),
    SalesRecord("2026-05", "tile", 4400, {"wholesale": 0.6, "hospitality": 0.3, "online": 0.1}, 0.32),
]


# --------------------------------------------------------------------------- #
# Credibility whitelists / blacklists (substring, case-insensitive)
# --------------------------------------------------------------------------- #
# Mirrors gas_agent.driver_curation: a supplier whose specialties hit the
# whitelist is a credible ceramics input; a blacklist hit (a textile / food /
# automotive / chemical / electronics vendor) is rejected as off-domain.
SUPPLIER_WHITELIST: tuple[str, ...] = (
    "clay", "glaze", "kiln", "ceramic", "pottery", "refractory", "material", "heat-resistant",
    "kaolin", "feldspar", "mineral",
)
SUPPLIER_BLACKLIST: tuple[str, ...] = (
    "textile", "food", "automotive", "chemical", "electronic",
)

CHANNEL_WHITELIST: tuple[str, ...] = (
    "wholesale", "hospitality", "ceramics retail", "retail", "export", "d2c", "direct-to-consumer",
)
CHANNEL_BLACKLIST: tuple[str, ...] = (
    "food retail", "automotive", "chemical",
)


# --------------------------------------------------------------------------- #
# Small accessors / helpers
# --------------------------------------------------------------------------- #
def quarter_of(month: str) -> str:
    """Map a month string ("YYYY-MM" or "YYYY-MM-01") to its quarter ("Q1".."Q4")."""
    month_number = int(month.split("-")[1])
    return f"Q{(month_number - 1) // 3 + 1}"


def get_product(product_id: str) -> Product:
    return PRODUCTS[product_id]


def list_products() -> list[Product]:
    return list(PRODUCTS.values())


def list_suppliers() -> list[Supplier]:
    return list(SUPPLIERS.values())


def list_channels() -> list[SalesChannel]:
    return list(CHANNELS.values())
