"""Company-description intake — free text → one complete CompanyProfile.

The unified app opens with a single free-text box: the user describes their
ceramics company. That one description is the **base case for both agents** — it
becomes the Sybilion filter *persona* (the gas forecast and the four ceramics
factors) and it fills the ceramics decision inputs (product, quantity, timeline,
channel competition, cost-factor weights).

THE RULE holds exactly as in :mod:`gas_agent.keyword_agent`: the LLM here only
*extracts stated facts* (temperature 0, a strict prompt, every value validated
against the committed catalog). It never invents an input the user didn't state,
and it never computes a decision number — the lock %, the scores, the quotes and
the margins are all decided downstream by deterministic code. Anything the
description does not state is returned as a **missing field**; the profile is
still completed with the committed ceramics defaults so the pipeline can always
run, and :func:`follow_up_questions` turns the missing list into 1-3 plain
clarifiers the UI can optionally ask before forecasting.

Offline floor: with no Featherless key (or on any LLM failure)
:func:`parse_description` falls back to a deterministic keyword scan over the
text plus the defaults, so the whole intake runs with no keys.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from gas_agent import llm
from gas_agent.config import KEYWORD_MODEL

from ceramics_agent.catalog import PRODUCTS, Product, get_product
from ceramics_agent.cost_policy import CostWeights, default_weights

# Allowed enum values the rest of the pipeline understands.
VALID_PRODUCT_IDS: tuple[str, ...] = tuple(PRODUCTS.keys())  # ("bowl", "dinnerware", "tile")
VALID_COMPETITION: tuple[str, ...] = ("low", "medium", "high")
VALID_GAS_EXPOSURE: tuple[str, ...] = ("low", "medium", "high")

# The committed defaults — the demo's calm base case (Handmade Bowl / 5000 / 14d /
# medium). Anything the description doesn't state falls back to these, so a profile
# is always complete and the forecast can always run.
DEFAULT_PRODUCT_ID = "bowl"
DEFAULT_QUANTITY = 5000
DEFAULT_TIMELINE_DAYS = 14
DEFAULT_COMPETITION = "medium"
# "medium" maps to the committed default weights (gas 0.40 — still the largest single
# bucket), so the default profile reproduces the demo's exact base case. A description
# that stresses gas ("gas-intensive", "energy-intensive") lifts this to "high".
DEFAULT_GAS_EXPOSURE = "medium"

# The fields the agents care about, in the order we'd ask about them. ``missing``
# is reported against this set; :func:`follow_up_questions` walks it in priority
# order and asks about at most three.
CORE_FIELDS: tuple[str, ...] = ("product", "quantity", "timeline_days", "competition")
OPTIONAL_FIELDS: tuple[str, ...] = ("gas_exposure", "sell_regions")
FIELD_PRIORITY: tuple[str, ...] = CORE_FIELDS + OPTIONAL_FIELDS


@dataclass(frozen=True)
class CompanyProfile:
    """The structured base case both agents run on.

    Every field is resolved (defaults applied) so the pipeline can always run; the
    separately-returned ``missing`` list says which were *not* stated by the user,
    so the UI can choose to ask follow-ups. ``weights`` is derived from the stated
    gas exposure (a deterministic mapping — see :func:`weights_for_gas_exposure`),
    never computed by the model."""

    product_id: str = DEFAULT_PRODUCT_ID
    quantity: int = DEFAULT_QUANTITY
    timeline_days: int = DEFAULT_TIMELINE_DAYS
    competition: str = DEFAULT_COMPETITION
    gas_exposure: str = DEFAULT_GAS_EXPOSURE
    weights: CostWeights = field(default_factory=default_weights)
    sell_regions: tuple[str, ...] = ()
    description: str = ""  # the raw free text — the shared persona source
    source: str = "default"  # "llm" | "fallback" | "default"
    notes: str = ""

    @property
    def product_name(self) -> str:
        return get_product(self.product_id).name

    def persona(self) -> str:
        """The plain-English brief handed to ``keyword_agent.select_filters`` — the
        raw description (or the gas default when empty), enriched with the resolved
        structured facts so the Sybilion filter pick is well-grounded for both the
        gas forecast and the four ceramics factors."""
        base = (self.description or "").strip()
        if not base:
            from gas_agent.keyword_agent import DEFAULT_PERSONA

            base = DEFAULT_PERSONA
        facts = (
            f"\n\nStructured profile — producing {self.quantity:,} units of "
            f"{self.product_name} over a {self.timeline_days}-day run; channel "
            f"competition {self.competition}; gas cost exposure {self.gas_exposure}."
        )
        if self.sell_regions:
            facts += f" Sells into: {', '.join(self.sell_regions)}."
        return base + facts


# --------------------------------------------------------------------------- #
# Deterministic gas-exposure → weights mapping (input prep, not a decision)
# --------------------------------------------------------------------------- #
def weights_for_gas_exposure(label: str | None) -> CostWeights:
    """Map a stated gas-exposure label to starting cost-factor weights.

    A pure, deterministic mapping: a manufacturer that says gas is its dominant
    cost gets a gas-heavy blend (which the band-blend then turns into a lock %);
    "low" tilts toward clay. THE RULE holds — this shapes the *decision input*
    (the weights), it does not compute the lock %. Unknown/None → the calm
    gas-dominant default."""
    table = {
        "high": CostWeights(gas=0.55, clay=0.25, energy=0.12, transport=0.08),
        "medium": default_weights(),
        "low": CostWeights(gas=0.22, clay=0.42, energy=0.20, transport=0.16),
    }
    return table.get((label or "").strip().lower(), default_weights())


# --------------------------------------------------------------------------- #
# Deterministic keyword scan — the offline extractor (also coerces LLM output)
# --------------------------------------------------------------------------- #
def _product_from_text(text: str) -> str | None:
    """Best-effort product match from free text. Checks the most specific cues
    first (tile, then dinnerware, then bowl) so "floor tiles" doesn't fall through
    to a generic match."""
    low = text.lower()
    if any(word in low for word in ("tile", "tiling", "floor", "m2", "m²", "square met")):
        return "tile"
    if any(word in low for word in ("dinnerware", "dinner set", "tableware", "plate",
                                    "6-piece", "six-piece", "place setting", "crockery")):
        return "dinnerware"
    if any(word in low for word in ("bowl", "mug", "cup", "vase", "handmade", "pottery", "tableware piece")):
        return "bowl"
    return None


def _quantity_from_text(text: str) -> int | None:
    """First plausible order quantity (>= 50). Handles "5,000", "5000", "5k"."""
    for raw in re.findall(r"(\d[\d,]*\.?\d*)\s*(k\b)?", text.lower()):
        number_text, kilo = raw
        digits = number_text.replace(",", "")
        if not digits or digits == ".":
            continue
        try:
            value = float(digits)
        except ValueError:
            continue
        if kilo:
            value *= 1000
        if value >= 50:
            return int(round(value))
    return None


def _timeline_from_text(text: str) -> int | None:
    """Timeline in days from "14 days" / "2 weeks" / "1 month"."""
    match = re.search(r"(\d+)\s*(day|days|week|weeks|month|months)", text.lower())
    if not match:
        return None
    value = int(match.group(1))
    unit = match.group(2)
    if unit.startswith("week"):
        return value * 7
    if unit.startswith("month"):
        return value * 30
    return value


def _competition_from_text(text: str) -> str | None:
    low = text.lower()
    if any(word in low for word in ("high competition", "very competitive", "crowded",
                                    "saturated", "cut-throat", "fierce")):
        return "high"
    if any(word in low for word in ("low competition", "little competition", "niche",
                                    "uncontested", "few competitors")):
        return "low"
    if "medium competition" in low or "moderate competition" in low:
        return "medium"
    return None


def _gas_exposure_from_text(text: str) -> str | None:
    low = text.lower()
    if any(word in low for word in ("gas is our largest", "gas is our biggest", "gas is the largest",
                                    "gas-intensive", "gas intensive", "energy-intensive",
                                    "energy intensive", "high-temperature firing", "high temperature firing")):
        return "high"
    if any(word in low for word in ("electric kiln", "electric firing", "low gas",
                                    "minimal gas", "renewable", "solar", "little gas")):
        return "low"
    return None


def _coerce_product(value, text: str) -> str | None:
    """A model-returned product string → one of the three catalog ids (or None)."""
    if isinstance(value, str):
        candidate = value.strip().lower()
        if candidate in VALID_PRODUCT_IDS:
            return candidate
        matched = _product_from_text(candidate)
        if matched:
            return matched
    return _product_from_text(text)


def _coerce_enum(value, allowed: tuple[str, ...]) -> str | None:
    if isinstance(value, str) and value.strip().lower() in allowed:
        return value.strip().lower()
    return None


def _enum_from_answer(answer: str, allowed: tuple[str, ...]) -> str | None:
    """Find an allowed enum value appearing as a word in a free-text follow-up
    answer (e.g. "low — we fire electric" → "low"). More lenient than
    :func:`_coerce_enum`, which expects the model to return a clean enum."""
    tokens = set(re.findall(r"[a-z]+", (answer or "").lower()))
    for value in allowed:
        if value in tokens:
            return value
    return None


def _coerce_int(value, *, minimum: int) -> int | None:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return number if number >= minimum else None


def _coerce_regions(value) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())[:8]
    if isinstance(value, str) and value.strip():
        return tuple(part.strip() for part in re.split(r"[,/;]", value) if part.strip())[:8]
    return ()


# --------------------------------------------------------------------------- #
# Assemble a profile from raw (possibly partial) values + record what's missing
# --------------------------------------------------------------------------- #
def _assemble(
    *,
    product: str | None,
    quantity: int | None,
    timeline_days: int | None,
    competition: str | None,
    gas_exposure: str | None,
    sell_regions: tuple[str, ...],
    description: str,
    source: str,
    notes: str,
) -> tuple[CompanyProfile, list[str]]:
    """Fill any unstated field with its default and report it as missing."""
    missing: list[str] = []
    if product is None:
        missing.append("product")
    if quantity is None:
        missing.append("quantity")
    if timeline_days is None:
        missing.append("timeline_days")
    if competition is None:
        missing.append("competition")
    if gas_exposure is None:
        missing.append("gas_exposure")
    if not sell_regions:
        missing.append("sell_regions")

    resolved_exposure = gas_exposure or DEFAULT_GAS_EXPOSURE
    profile = CompanyProfile(
        product_id=product or DEFAULT_PRODUCT_ID,
        quantity=quantity or DEFAULT_QUANTITY,
        timeline_days=timeline_days or DEFAULT_TIMELINE_DAYS,
        competition=competition or DEFAULT_COMPETITION,
        gas_exposure=resolved_exposure,
        weights=weights_for_gas_exposure(resolved_exposure),
        sell_regions=sell_regions,
        description=description,
        source=source,
        notes=notes,
    )
    return profile, missing


def _fallback_parse(text: str) -> tuple[CompanyProfile, list[str]]:
    """Deterministic keyword scan — the no-key / LLM-down path."""
    return _assemble(
        product=_product_from_text(text),
        quantity=_quantity_from_text(text),
        timeline_days=_timeline_from_text(text),
        competition=_competition_from_text(text),
        gas_exposure=_gas_exposure_from_text(text),
        sell_regions=(),
        description=text,
        source="fallback",
        notes="",
    )


_SYSTEM_PROMPT = (
    "You read a manufacturer's free-text company description and extract ONLY the "
    "facts they explicitly state, for a ceramics production-planning agent. You do "
    "NOT make any decision, recommendation, or numeric estimate — you only transcribe "
    "stated facts. If a fact is not stated, use null; never guess or invent a value.\n\n"
    "Return ONLY a JSON object with exactly these keys:\n"
    '  "product": one of "bowl" (handmade bowls/mugs/vases), "dinnerware" (plates/'
    'tableware/dinner sets), "tile" (floor/wall tiles), or null;\n'
    '  "quantity": integer units for the production run, or null;\n'
    '  "timeline_days": integer days until the run must ship, or null;\n'
    '  "competition": one of "low", "medium", "high" (how crowded their sales '
    'channel is), or null;\n'
    '  "gas_exposure": one of "low", "medium", "high" (how dominant natural gas is '
    'in their cost base), or null;\n'
    '  "sell_regions": array of country/region names they sell into, or null;\n'
    '  "notes": one short sentence of anything else relevant, or "".'
)


def parse_description(text: str, *, use_llm: bool = True) -> tuple[CompanyProfile, list[str]]:
    """Extract a :class:`CompanyProfile` from a free-text company description.

    With Featherless available (and ``use_llm``) the model extracts stated facts
    only; every value is validated/coerced against the catalog, and anything it
    leaves null is back-filled deterministically from a keyword scan before
    defaults apply. With no key (or any failure) it drops straight to the
    deterministic scan. Returns ``(profile, missing_fields)`` — the profile is
    always complete; ``missing_fields`` lists what the user never stated."""
    text = (text or "").strip()
    if not text:
        profile, _ = _assemble(
            product=None, quantity=None, timeline_days=None, competition=None,
            gas_exposure=None, sell_regions=(), description="", source="default", notes="",
        )
        return profile, list(FIELD_PRIORITY)

    if not (use_llm and llm.featherless_available()):
        return _fallback_parse(text)

    try:
        payload = llm.chat_json(KEYWORD_MODEL, _SYSTEM_PROMPT, text, temperature=0.0, max_tokens=400)
    except llm.LLMUnavailable:
        return _fallback_parse(text)
    if not isinstance(payload, dict):
        return _fallback_parse(text)

    # Coerce + validate; fall back to the keyword scan field-by-field so a partial
    # model answer is still enriched deterministically rather than left as default.
    return _assemble(
        product=_coerce_product(payload.get("product"), text),
        quantity=_coerce_int(payload.get("quantity"), minimum=50) or _quantity_from_text(text),
        timeline_days=_coerce_int(payload.get("timeline_days"), minimum=1) or _timeline_from_text(text),
        competition=_coerce_enum(payload.get("competition"), VALID_COMPETITION) or _competition_from_text(text),
        gas_exposure=_coerce_enum(payload.get("gas_exposure"), VALID_GAS_EXPOSURE) or _gas_exposure_from_text(text),
        sell_regions=_coerce_regions(payload.get("sell_regions")),
        description=text,
        source="llm",
        notes=str(payload.get("notes", "")).strip(),
    )


# --------------------------------------------------------------------------- #
# Off-catalog product recognition (full-live mode) — LLM-estimated INPUT spec
# --------------------------------------------------------------------------- #
_PRODUCT_SPEC_SYSTEM = (
    "You are a ceramics production engineer preparing physical INPUTS (not a decision, not a "
    "price) for a costing agent. Read a manufacturer's description and identify the product. If "
    "it clearly matches a catalog line, return its id; otherwise estimate a realistic per-unit "
    "bill of materials for the NEW product so it can be costed.\n\n"
    "Return ONLY a JSON object:\n"
    '  "catalog_id": one of "bowl","dinnerware","tile" if it clearly matches, else null;\n'
    '  "name": a short product name (e.g. "Ceramic Sink");\n'
    '  "clay_kg": body clay kg per unit (>0);\n'
    '  "glaze_kg": glaze kg per unit (>=0);\n'
    '  "kiln_kwh": grid electricity kWh per unit (>0);\n'
    '  "firing_gas_kwh": kiln gas kWh per unit (>0);\n'
    '  "ship_kg": shipped weight kg per unit (>0).\n'
    "Reference scale: a small bowl ~0.5kg clay / 6kWh gas; a 6-piece dinnerware set ~3kg / 22kWh; "
    "a floor tile ~25kg / 120kWh; a large ceramic sink is heavier and more gas-intensive. These "
    "are material estimates (inputs the user can edit), never a price or a decision."
)


def _spec_value(payload: dict, key: str, default: float, floor: float, ceiling: float) -> float:
    """Coerce one estimated BOM number into a sane positive range (defends the cost math
    against an absurd LLM value — the user can still edit it)."""
    try:
        value = float(payload.get(key))
    except (TypeError, ValueError):
        value = default
    return max(floor, min(value, ceiling))


def estimate_product(description: str, *, use_llm: bool = True) -> Product:
    """Recognise the product in a description for FULL-LIVE mode.

    Returns a committed catalog :class:`Product` when the description clearly matches one of
    the three lines; otherwise an **LLM-estimated** custom Product (``estimated=True``) carrying a
    per-unit bill of materials. THE RULE holds: the BOM is a *labeled, editable INPUT spec* (like
    the user stating their materials), not a decision — the lock %, supplier and margin are still
    computed deterministically from it downstream. The deterministic fallback (no key / failure)
    is the nearest catalog product, so the offline path never invents a spec."""
    text = (description or "").strip()
    if not (text and use_llm and llm.featherless_available()):
        return get_product(_product_from_text(text))  # offline → catalog only, never estimated
    try:
        payload = llm.chat_json(KEYWORD_MODEL, _PRODUCT_SPEC_SYSTEM, text, temperature=0.0, max_tokens=300)
    except llm.LLMUnavailable:
        return get_product(_product_from_text(text))
    if not isinstance(payload, dict):
        return get_product(_product_from_text(text))

    catalog_id = _coerce_enum(payload.get("catalog_id"), VALID_PRODUCT_IDS)
    if catalog_id:
        return get_product(catalog_id)  # the LLM recognised a catalog product → use the real spec

    name = str(payload.get("name") or "Custom ceramic product").strip()[:60] or "Custom ceramic product"
    return Product(
        id="custom", name=name,
        clay_kg=_spec_value(payload, "clay_kg", 5.0, 0.1, 300.0),
        glaze_kg=_spec_value(payload, "glaze_kg", 0.5, 0.0, 50.0),
        kiln_kwh=_spec_value(payload, "kiln_kwh", 8.0, 0.1, 500.0),
        firing_gas_kwh=_spec_value(payload, "firing_gas_kwh", 30.0, 0.1, 2000.0),
        ship_kg=_spec_value(payload, "ship_kg", 5.0, 0.1, 500.0),
        estimated=True,
    )


# --------------------------------------------------------------------------- #
# Follow-up questions for the missing fields
# --------------------------------------------------------------------------- #
_QUESTIONS: dict[str, str] = {
    "product": "What are you producing — handmade bowls, dinnerware sets, or floor tiles?",
    "quantity": "How many units is this production run?",
    "timeline_days": "How many days until the run needs to ship?",
    "competition": "How competitive is your main sales channel — low, medium, or high?",
    "gas_exposure": "How big a share of your costs is natural gas — low, medium, or high?",
    "sell_regions": "Which regions or countries do you sell into?",
}


def follow_up_questions(missing: list[str], *, limit: int = 3) -> list[str]:
    """Turn the missing-field list into at most ``limit`` plain-English clarifiers,
    asked in priority order (decision-critical fields first). Returns an empty list
    when nothing material is missing, so the UI can skip straight to forecasting."""
    ordered = [field_name for field_name in FIELD_PRIORITY if field_name in set(missing)]
    return [_QUESTIONS[field_name] for field_name in ordered[:limit] if field_name in _QUESTIONS]


def missing_field_for_question(question: str) -> str | None:
    """Reverse-lookup the field a follow-up question is about (so the UI can fold an
    answer back into the right slot)."""
    for field_name, text in _QUESTIONS.items():
        if text == question:
            return field_name
    return None


def apply_answer(profile: CompanyProfile, field_name: str, answer: str) -> CompanyProfile:
    """Fold one free-text follow-up answer back into the profile, parsing it the same
    deterministic way the description is scanned. Unrecognised answers leave the
    field at its default (so the pipeline still runs)."""
    answer = (answer or "").strip()
    if not answer:
        return profile
    if field_name == "product":
        matched = _product_from_text(answer)
        return replace(profile, product_id=matched) if matched else profile
    if field_name == "quantity":
        value = _quantity_from_text(answer)
        return replace(profile, quantity=value) if value else profile
    if field_name == "timeline_days":
        value = _timeline_from_text(answer) or _quantity_from_text(answer)
        return replace(profile, timeline_days=value) if value else profile
    if field_name == "competition":
        value = _competition_from_text(answer) or _enum_from_answer(answer, VALID_COMPETITION)
        return replace(profile, competition=value) if value else profile
    if field_name == "gas_exposure":
        value = _gas_exposure_from_text(answer) or _enum_from_answer(answer, VALID_GAS_EXPOSURE)
        if value:
            return replace(profile, gas_exposure=value, weights=weights_for_gas_exposure(value))
        return profile
    if field_name == "sell_regions":
        regions = _coerce_regions(answer)
        return replace(profile, sell_regions=regions) if regions else profile
    return profile
