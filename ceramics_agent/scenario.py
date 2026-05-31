"""One shock, both agents — the ceramics half of the mid-run assumption shift.

The gas agent already answers "a supply shock just landed — re-decide the hedge."
This module lets the *same* headline re-decide the ceramics input-cost lock at the
same instant, so a single chat message ("Iran closes the Strait of Hormuz") moves
**both** decisions on screen together.

It stays inside THE RULE exactly as the gas scenario does. The only thing an LLM
ever does is classify the free-text headline (is this a shock, how severe, what to
call it) — and that classifier is **reused verbatim** from
:func:`gas_agent.scenario.parse_shock_request`. Everything that moves a number here
is deterministic arithmetic:

  * :func:`route_factors` maps the headline to the cost factor(s) it hits — a pure
    keyword router (Hormuz/Iran/sanctions → gas, Red Sea/freight/port → shipping,
    grid/blackout → power, kaolin/clay → clay). A geopolitical/gas headline routes
    to **gas only**, so the gas agent's behaviour is unchanged and the ceramics
    *gas* factor moves in lockstep with it.
  * :func:`run_ceramics_shock` applies :func:`gas_agent.scenario.apply_shock` to each
    affected factor band (each ceramics factor is already a ``list[MonthForecast]``,
    so the gas transform works **unchanged**), re-blends under the user's weights, and
    re-decides the lock % through the same deterministic ``decide_procurement`` — with
    an additive ``risk_premium`` that lifts the lock floor in step with the gas hedge.

The net effect mirrors the gas story: a credible supply shock skews input cost to the
upside and makes the future less certain, so the buyer **locks more** of next quarter's
cost now even though the central forecast is murkier — and the same shock that did it
to gas did it to ceramics, in one move.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from gas_agent.hedge_policy import HedgePolicyParams, MonthDecision, MonthForecast, clamp
from gas_agent.scenario import (
    AUTO_PREMIUM_CAP,
    DEFAULT_SHOCK_PARAMS,
    ShockParams,
    ShockRequest,
    apply_shock,
    parse_shock_request,
    shock_risk_premium,
)

from ceramics_agent.cost_policy import (
    CERAMICS_POLICY_PARAMS,
    CostWeights,
    blended_index,
    decide_procurement,
    quarter_band_width,
    quarter_lock_ratio,
)
from ceramics_agent.forecast import FACTORS

# Ceramics-tuned shock parameters. The median bump and the additive risk premium
# match the gas defaults (a shock lifts the input's cost and the lock floor by the
# same amount), but the band widening is *gentler*: the ceramics band thresholds are
# narrow (tight ≤ 25% / wide ≥ 60%, a 0.35 span) versus gas's 0.80 span, so the gas
# default widening would crater the band component and fight the premium. At 0.15 the
# band still loosens (the future got less certain) while the premium + rising drift
# clearly win — the lock rises, mirroring the gas hedge.
CERAMICS_SHOCK_PARAMS = ShockParams(
    median_bump=0.12,
    band_widen=0.15,
    risk_premium=0.25,
    label="Input-cost supply shock",
)


# --------------------------------------------------------------------------- #
# Factor routing — which cost factor(s) a headline hits (deterministic, no LLM).
# --------------------------------------------------------------------------- #
# Substring keywords per factor. Designed so the canonical geopolitical/gas
# headlines (Hormuz / Iran / war / sanctions / pipeline) route to *gas only*, keeping
# the gas agent's behaviour identical, while logistics, grid and raw-material
# headlines route to their own factor. A message may legitimately hit several (a Red
# Sea crisis lifts freight *and*, through fear, gas) — every matched factor is shocked.
_FACTOR_KEYWORDS: dict[str, tuple[str, ...]] = {
    "gas": (
        "gas", "ttf", "hormuz", "strait", "iran", "sanction", "embargo", "war",
        "invasion", "invade", "pipeline", "nord stream", "nordstream", "lng",
        "russia", "russian", "qatar", "opec", "energy crisis", "force majeure",
    ),
    "clay": (
        "kaolin", "clay", "feldspar", "mineral", "raw material", "quarry", "ball clay",
    ),
    "power": (
        "power", "grid", "blackout", "brownout", "electricity", "load shedding",
        "load-shedding", "power price", "power outage",
    ),
    "shipping": (
        "red sea", "freight", "shipping", "container", "suez", "canal", "port",
        "dock", "vessel", "houthi", "logistics", "shipping rate", "freight rate",
        "port strike", "port closure",
    ),
}


def route_factors(message: str) -> tuple[str, ...]:
    """Return the cost factor(s) a free-text headline names, in canonical FACTORS
    order. Empty when nothing matches — the caller decides whether to default
    (a recognised-but-unrouted supply shock falls to gas, the dominant energy cost,
    in :func:`affected_factors_for`)."""
    lower = (message or "").lower()
    return tuple(
        factor
        for factor in FACTORS
        if any(keyword in lower for keyword in _FACTOR_KEYWORDS[factor])
    )


def affected_factors_for(message: str, *, default_to_gas: bool = True) -> tuple[str, ...]:
    """The factor(s) a headline should shock. Falls back to ``("gas",)`` when the
    message is a shock with no specific factor cue (geopolitical/energy shocks hit
    gas first), unless ``default_to_gas`` is turned off."""
    hits = route_factors(message)
    if hits:
        return hits
    return ("gas",) if default_to_gas else ()


def _normalize_factors(affected: Iterable[str]) -> tuple[str, ...]:
    """Filter to known factors, dedupe, and keep canonical FACTORS order."""
    chosen = set(affected)
    return tuple(factor for factor in FACTORS if factor in chosen)


# --------------------------------------------------------------------------- #
# The deterministic re-decide — apply_shock to the affected bands, re-blend, re-lock.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CeramicsShockOutcome:
    """Everything the dashboard needs to redraw the ceramics lock under a shock."""

    magnitude: float
    label: str
    affected_factors: tuple[str, ...]
    risk_premium: float  # the shock's OWN marginal floor (for the scenario readout)
    factors: dict[str, list[MonthForecast]]  # per-factor bands, affected ones shocked
    decisions: list[MonthDecision]  # re-decided lock % per month
    lock_ratio: float  # next-quarter average lock % — the headline
    band_width: float  # next-quarter mean weighted blended band


def run_ceramics_shock(
    factors: dict[str, list[MonthForecast]],
    weights: CostWeights,
    magnitude: float,
    affected_factors: Iterable[str],
    *,
    label: str = "",
    base_risk_premium: float = 0.0,
    params: ShockParams = CERAMICS_SHOCK_PARAMS,
    policy_params: HedgePolicyParams = CERAMICS_POLICY_PARAMS,
) -> CeramicsShockOutcome:
    """Re-decide the ceramics lock under a supply shock — pure arithmetic, no LLM.

    Each *affected* factor band is re-cast by :func:`gas_agent.scenario.apply_shock`
    (median bumped up, band widened); untouched factors pass through unchanged. The
    four bands are re-blended under ``weights`` and re-decided by the same
    ``decide_procurement`` engine, with the lock anchored to the **pre-shock** nearest
    level (so a uniform cost bump reads as rising drift) and an additive
    ``risk_premium`` lifting the floor.

    ``base_risk_premium`` is any standing premium already in force; the shock's own
    marginal is added on top and clamped to the same ceiling the gas path uses, so a
    shock raises the lock above the baseline rather than replacing it. ``magnitude == 0``
    or no affected factor returns the calm decisions unchanged (a no-op outcome)."""
    mag = clamp(magnitude, 0.0, 1.0)
    chosen = _normalize_factors(affected_factors)

    shocked: dict[str, list[MonthForecast]] = {}
    for factor, months in factors.items():
        if factor in chosen and mag > 0.0:
            shocked[factor] = apply_shock(months, mag, params)
        else:
            shocked[factor] = list(months)

    # Hold the lock anchor at the pre-shock nearest-future blended cost: a uniform
    # cost bump then reads as rising drift, exactly as the gas shock keeps spot fixed.
    pre_shock_blended = blended_index(factors, weights)
    anchor = pre_shock_blended[0].level if pre_shock_blended else None

    marginal = shock_risk_premium(mag, params) if (chosen and mag > 0.0) else 0.0
    total_premium = min(
        AUTO_PREMIUM_CAP + params.risk_premium,  # standing cap + this shock's max marginal
        max(0.0, base_risk_premium) + marginal,
    )

    decisions = decide_procurement(
        shocked, weights, policy_params, risk_premium=total_premium, anchor=anchor
    )
    return CeramicsShockOutcome(
        magnitude=mag,
        label=label or params.label,
        affected_factors=chosen,
        risk_premium=marginal,
        factors=shocked,
        decisions=decisions,
        lock_ratio=quarter_lock_ratio(decisions),
        band_width=quarter_band_width(decisions),
    )


def ceramics_shock_from_message(
    message: str,
    factors: dict[str, list[MonthForecast]],
    weights: CostWeights,
    *,
    request: ShockRequest | None = None,
    use_llm: bool = False,
    base_risk_premium: float = 0.0,
) -> CeramicsShockOutcome | None:
    """One-call entry for the chat layer: classify the headline (reusing the gas
    classifier), route it to the factor(s) it hits, and re-decide the ceramics lock.

    Returns ``None`` when the message is not a credible supply shock, so the caller
    can leave the calm decision in place. Pass an already-parsed ``request`` to avoid
    classifying the same message twice (the gas and ceramics shocks share one)."""
    req = request if request is not None else parse_shock_request(message, use_llm=use_llm)
    if not req.is_shock:
        return None
    affected = affected_factors_for(message)
    return run_ceramics_shock(
        factors,
        weights,
        req.magnitude,
        affected,
        label=req.label,
        base_risk_premium=base_risk_premium,
    )
