"""The keyword agent — Featherless turns a plain-English brief into Sybilion filters.

Given a description of who is forecasting and what they need ("a German glass
manufacturer hedging quarterly gas cost"), this picks the Sybilion category and
region filters and the keywords that steer the forecast toward the right drivers.

Two rules keep it honest:

* It **selects, never invents.** The real Sybilion catalog (category ids, region
  codes) is embedded in the prompt and every id the model returns is validated
  against it — anything off-catalogue is dropped.
* It **never decides the hedge.** It only shapes the API call; the hedge ratio
  is computed downstream by :mod:`gas_agent.hedge_policy`.

It also closes a loop with curation: if a forecast's drivers come back too thin
after :mod:`gas_agent.driver_curation` (``needs_refine``), :func:`refine_filters`
widens the selection and the forecast is re-pulled. Whenever Featherless is
unavailable the agent falls back to the curated default filters, so the pipeline
still runs end to end off the cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gas_agent import catalog, llm
from gas_agent.config import KEYWORD_MODEL
from gas_agent.driver_curation import CurationResult

# The demo persona. The agent is configurable — swap this string to retarget it —
# but the hackathon demo drives this one buyer.
DEFAULT_PERSONA = (
    "A mid-size German glass and ceramics manufacturer. Natural gas is its single "
    "largest volatile cost. Procurement buys gas forward each quarter on the TTF "
    "market and must decide how much of next quarter to lock in now versus leave "
    "to spot. They care about European gas supply, LNG flows, energy and "
    "electricity prices, the EUR/USD rate, and geopolitical supply risk."
)


@dataclass(frozen=True)
class KeywordSelection:
    """The Sybilion filters the agent chose for a forecast call."""

    keywords: list[str]
    category_ids: list[int]
    region_codes: list[int]
    recency_factor: float  # 0-1; higher leans on recent data (used by the shock path)
    source: str  # "llm" | "fallback"
    notes: str = ""  # rationale or fallback reason, for the dashboard

    def category_names(self) -> list[str]:
        return [catalog.category_name(cid) for cid in self.category_ids]

    def region_names(self) -> list[str]:
        return [catalog.region_name(code) for code in self.region_codes]


def _fallback_selection(notes: str, recency_factor: float = 0.7) -> KeywordSelection:
    return KeywordSelection(
        keywords=[
            "natural gas", "TTF", "LNG", "energy prices", "electricity",
            "EUR/USD", "gas storage", "geopolitical supply risk",
        ],
        category_ids=list(catalog.DEFAULT_FORECAST_CATEGORIES),
        region_codes=list(catalog.DEFAULT_FORECAST_REGIONS),
        recency_factor=recency_factor,
        source="fallback",
        notes=notes,
    )


def _catalog_prompt_block() -> str:
    """Compact catalog listing handed to the model so it selects real ids."""
    categories = "\n".join(f"  {cid}: {name}" for cid, name in sorted(catalog.CATEGORIES.items()))
    regions = "\n".join(f"  {code}: {name}" for code, name in catalog.REGIONS.items())
    return f"CATEGORIES (id: name)\n{categories}\n\nREGIONS (code: name)\n{regions}"


_SYSTEM_PROMPT = (
    "You configure a probabilistic forecasting API for a commodity-hedging agent. "
    "Given a buyer's description, choose the category and region filters and search "
    "keywords that will surface the most causally-relevant price drivers for their "
    "forecast target. Select ONLY ids and codes from the provided catalog; never "
    "invent ids. Prefer precision over breadth. You do NOT make any trading or "
    "hedging decision — you only prepare the data request.\n\n"
    "Return ONLY a JSON object with exactly these keys:\n"
    '  "keywords": array of <=20 short strings,\n'
    '  "category_ids": array of integers from the catalog,\n'
    '  "region_codes": array of integers from the catalog,\n'
    '  "recency_factor": number between 0 and 1 (higher = weight recent data more),\n'
    '  "rationale": one short sentence.'
)


def _validate_selection(payload: dict, source: str) -> KeywordSelection:
    """Coerce + validate the model's JSON against the real catalog."""
    raw_categories = payload.get("category_ids", []) or []
    raw_regions = payload.get("region_codes", []) or []
    category_ids = [int(c) for c in raw_categories if int(c) in catalog.CATEGORIES]
    region_codes = [int(r) for r in raw_regions if int(r) in catalog.REGIONS]

    keywords = [str(k).strip() for k in (payload.get("keywords") or []) if str(k).strip()][:20]
    try:
        recency_factor = float(payload.get("recency_factor", 0.7))
    except (TypeError, ValueError):
        recency_factor = 0.7
    recency_factor = max(0.0, min(1.0, recency_factor))

    # If the model gave us nothing usable, fall back rather than submit empties.
    if not category_ids or not region_codes:
        return _fallback_selection(
            "model returned no valid catalog ids; using curated defaults", recency_factor
        )

    rationale = str(payload.get("rationale", "")).strip()
    return KeywordSelection(
        keywords=keywords or _fallback_selection("").keywords,
        category_ids=category_ids,
        region_codes=region_codes,
        recency_factor=recency_factor,
        source=source,
        notes=rationale,
    )


def select_filters(persona: str = DEFAULT_PERSONA) -> KeywordSelection:
    """Pick Sybilion filters for ``persona``. Falls back to curated defaults when
    Featherless is unavailable or returns nothing usable."""
    if not llm.featherless_available():
        return _fallback_selection("Featherless key not configured; using curated defaults")

    user = f"{_catalog_prompt_block()}\n\nBUYER:\n{persona}"
    try:
        payload = llm.chat_json(KEYWORD_MODEL, _SYSTEM_PROMPT, user, temperature=0.0, max_tokens=500)
    except llm.LLMUnavailable as error:
        return _fallback_selection(f"Featherless call failed ({error}); using curated defaults")
    if not isinstance(payload, dict):
        return _fallback_selection("model did not return a JSON object; using curated defaults")
    return _validate_selection(payload, source="llm")


def refine_filters(
    persona: str,
    previous: KeywordSelection,
    curation: CurationResult,
) -> KeywordSelection:
    """Widen the selection when curation kept too few drivers (the closed loop).

    Deterministic widening (union with the full credible whitelist) guarantees
    progress even without the LLM; with Featherless we additionally ask it to add
    keywords. Recency is nudged down so a broader history can surface drivers."""
    widened_categories = sorted(set(previous.category_ids) | catalog.CREDIBLE_CATEGORY_IDS)
    widened_regions = sorted(set(previous.region_codes) | catalog.CREDIBLE_REGION_CODES)
    notes = (
        f"refined: only {curation.kept_count} credible drivers "
        f"(< {curation.min_kept_drivers}); widened to the full gas whitelist"
    )

    keywords = list(previous.keywords)
    if llm.featherless_available():
        try:
            user = (
                f"{_catalog_prompt_block()}\n\nBUYER:\n{persona}\n\n"
                f"The previous filters surfaced only {curation.kept_count} credible "
                "drivers. Broaden the keywords and selection to capture more relevant "
                "energy, commodity, FX and supply-risk drivers."
            )
            payload = llm.chat_json(KEYWORD_MODEL, _SYSTEM_PROMPT, user, temperature=0.0, max_tokens=500)
            if isinstance(payload, dict):
                more = [str(k).strip() for k in (payload.get("keywords") or []) if str(k).strip()]
                keywords = list(dict.fromkeys(keywords + more))[:20]
        except llm.LLMUnavailable:
            pass  # deterministic widening below still applies

    return KeywordSelection(
        keywords=keywords,
        category_ids=widened_categories,
        region_codes=widened_regions,
        recency_factor=max(0.5, previous.recency_factor - 0.1),
        source=previous.source,
        notes=notes,
    )


@dataclass
class KeywordLoopResult:
    """The trace of a select -> curate -> refine loop."""

    selection: KeywordSelection
    curation: CurationResult
    rounds: int
    history: list[KeywordSelection] = field(default_factory=list)


def run_keyword_loop(
    fetch_drivers,
    persona: str = DEFAULT_PERSONA,
    max_rounds: int = 2,
):
    """Run the closed loop: pick filters, pull + curate drivers, and widen and
    retry while too few credible drivers survive.

    ``fetch_drivers`` is a callable taking a :class:`KeywordSelection` and
    returning a Sybilion ``external_signals.json`` dict (live API in production,
    a cached/fake function in tests). Returns a :class:`KeywordLoopResult`.
    """
    from gas_agent.driver_curation import curate_drivers

    selection = select_filters(persona)
    history = [selection]
    curation = curate_drivers(fetch_drivers(selection))
    rounds = 1
    while curation.needs_refine and rounds < max_rounds:
        selection = refine_filters(persona, selection, curation)
        history.append(selection)
        curation = curate_drivers(fetch_drivers(selection))
        rounds += 1
    return KeywordLoopResult(selection=selection, curation=curation, rounds=rounds, history=history)
