"""The forecast pipeline's stage list and the short "what's happening now" blurbs.

While a live forecast runs (five sequential Sybilion polls — one gas band plus the
four ceramics factors), the page shows a staged progress animation. Each stage has
a committed one-line **template** blurb so the animation always has something to say
offline; when a Featherless key is present, :func:`generate_blurbs` rewrites those
lines once, in the persona's voice, in a single call.

THE RULE holds here exactly as everywhere else: these blurbs are **status narration
only** — present-continuous "we are now doing X" lines. The generator's system
prompt forbids it from emitting any number, ratio, supplier, price, or verdict; the
decision is made downstream by the deterministic engines, never described into being
here. A malformed or missing line silently falls back to its committed template, so
the animation degrades gracefully and the no-keys demo is unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from gas_agent import config, llm


@dataclass(frozen=True)
class PipelineStage:
    """One step of the forecast pipeline.

    ``key``      — stable id used to look a blurb up and to drive the progress bar.
    ``label``    — the short fixed heading shown on the stage (never LLM-written).
    ``template`` — the committed fallback blurb (used offline / on any LLM hiccup).
    """

    key: str
    label: str
    template: str


# The ordered pipeline, mirroring what actually runs in recommend.py / the live
# forecast path: classify the drivers, submit the gas band, submit the four factor
# bands, poll them, then the deterministic decide→negotiate→backtest tail. The labels
# are fixed; only the blurbs are (optionally) rephrased by the LLM.
PIPELINE_STAGES: tuple[PipelineStage, ...] = (
    PipelineStage(
        "classify",
        "Reading your description",
        "Pulling the stated facts out of your company description.",
    ),
    PipelineStage(
        "forecast_gas",
        "Forecasting gas",
        "Submitting the kiln-gas band to Sybilion for a probabilistic forecast.",
    ),
    PipelineStage(
        "forecast_factors",
        "Forecasting clay, power & freight",
        "Submitting the clay, power and freight bands alongside the gas one.",
    ),
    PipelineStage(
        "poll",
        "Waiting on the bands",
        "Polling Sybilion until every cost-factor band comes back.",
    ),
    PipelineStage(
        "curate",
        "Curating suppliers & channels",
        "Keeping the credible suppliers and sales channels and scoring them.",
    ),
    PipelineStage(
        "decide",
        "Deciding the lock",
        "Blending the four bands to settle how much input cost to lock now.",
    ),
    PipelineStage(
        "negotiate",
        "Negotiating the deal",
        "Walking two rounds of buy and sell quotes to a working margin.",
    ),
    PipelineStage(
        "backtest",
        "Backtesting the policy",
        "Replaying past months to check the policy beats the baselines.",
    ),
)

STAGE_KEYS: tuple[str, ...] = tuple(stage.key for stage in PIPELINE_STAGES)


def stage_labels() -> list[str]:
    """The fixed stage headings, in order (for a progress legend)."""
    return [stage.label for stage in PIPELINE_STAGES]


def default_blurbs() -> dict[str, str]:
    """The committed template blurb for every stage — the offline floor."""
    return {stage.key: stage.template for stage in PIPELINE_STAGES}


_SYSTEM_PROMPT = (
    "You write very short status lines for a forecasting app's progress animation. "
    "The app is running a multi-step pipeline for a manufacturer; you will be given "
    "the stage keys and a one-line description of each stage. Rewrite each line as a "
    "single calm present-continuous status line (\"We are now ...\"), at most 12 words, "
    "in plain English, tuned to the company described.\n\n"
    "HARD RULES:\n"
    "- Output ONLY a JSON object mapping each given stage key to its rewritten line.\n"
    "- Never invent or state any number, percentage, ratio, price, supplier name, "
    "channel, or recommendation. You describe the *activity*, never a result.\n"
    "- No markdown, no preamble, no extra keys. Keep every line under 12 words."
)


def _stage_brief() -> str:
    """The compact stage list handed to the model (keys + their template lines)."""
    return "\n".join(f"{stage.key}: {stage.template}" for stage in PIPELINE_STAGES)


def generate_blurbs(persona: str, *, use_llm: bool = True) -> dict[str, str]:
    """Return ``{stage_key: blurb}`` for every stage.

    With no Featherless key (or ``use_llm=False``) this is just
    :func:`default_blurbs`. With a key it makes **one** ``chat_text`` call asking the
    model to rephrase the committed lines in the persona's voice, then validates the
    reply: only non-empty string values for known stage keys are taken, and any stage
    the model dropped or mangled keeps its committed template. The call is wrapped so
    any failure (no JSON, network, bad shape) falls back to the full template set —
    the animation never blocks on the model.
    """
    blurbs = default_blurbs()
    if not use_llm or not llm.featherless_available():
        return blurbs

    user = (
        f"Company description:\n{persona.strip() or '(none given)'}\n\n"
        f"Stages to rewrite (key: current line):\n{_stage_brief()}"
    )
    try:
        raw = llm.chat_text(
            config.CERAMICS_STAGE_BLURB_MODEL,
            _SYSTEM_PROMPT,
            user,
            temperature=0.3,
            max_tokens=300,
        )
        payload = json.loads(llm.strip_json_fence(raw))
    except (llm.LLMUnavailable, json.JSONDecodeError, ValueError):
        return blurbs

    if not isinstance(payload, dict):
        return blurbs

    for key in STAGE_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            blurbs[key] = value.strip()
    return blurbs
