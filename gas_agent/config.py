"""Central configuration.

Reads secrets and model choices from the environment (optionally via a local
.env file). Every value has a safe fallback so the app still imports and runs
off cached forecast artifacts even when no keys are wired up.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Folder layout
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
# Committed, read-only library of pre-fetched real Sybilion scenarios (W17). Each
# scenario dir mirrors a cache job dir; the artifact loaders read cache first, then
# here, so a matched scenario flows through the existing render path unchanged.
SCENARIOS_DIR = PROJECT_ROOT / "scenarios"

# Sybilion forecasting API
SYBILION_API_KEY = os.getenv("SYBILION_API_KEY", "")
SYBILION_BASE_URL = os.getenv("SYBILION_BASE_URL", "https://api.sybilion.dev")

# Featherless (OpenAI-compatible) inference endpoint
FEATHERLESS_API_KEY = os.getenv("FEATHERLESS_API_KEY", "")
FEATHERLESS_BASE_URL = os.getenv("FEATHERLESS_BASE_URL", "https://api.featherless.ai/v1")

# Model choices — swappable strings, defaulting to Mistral (European AI).
# The tag-picker needs reliable structured output; the explainer needs fluent,
# faithful prose. Both IDs are verified live on the Featherless catalog.
KEYWORD_MODEL = os.getenv("KEYWORD_MODEL", "mistralai/Mistral-Small-3.2-24B-Instruct-2506")
EXPLANATION_MODEL = os.getenv("EXPLANATION_MODEL", "mistralai/Mistral-Large-Instruct-2411")

# --------------------------------------------------------------------------- #
# Voice narration (optional). Provider chosen here via VOICE_PROVIDER:
#   * "auto"/"nvidia" → prefer NVIDIA (Riva fastpitch-hifigan-tts over gRPC),
#     fall back to the local OS voice when the free-tier rate limit is hit or a
#     call fails;
#   * "local"         → local OS voice only (macOS `say`).
# (Featherless serves text only — it has no /v1/audio/speech endpoint — so the
# always-on fallback is the local engine.) Everything has a safe fallback, so the
# app imports and runs with no voice setup: voice.synthesize() returns None and
# the dashboard simply hides the player.
# --------------------------------------------------------------------------- #
VOICE_PROVIDER = os.getenv("VOICE_PROVIDER", "auto").lower()

# NVIDIA Riva TTS (gRPC via NVIDIA Cloud Functions). The TTS models are NOT served
# on the OpenAI-compatible integrate.api.nvidia.com endpoint — they need the Riva
# client (`pip install nvidia-riva-client`) talking to the NVCF gRPC host with a
# per-model function id. All optional; absence just routes to the local engine.
# Defaults point at magpie-tts-multilingual: the older fastpitch/radtts gRPC
# functions are now NOT_FOUND on current NVCF accounts, so magpie's public function
# id is the verified-working default (override via env for a different model/voice).
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_RIVA_URI = os.getenv("NVIDIA_RIVA_URI", "grpc.nvcf.nvidia.com:443")
NVIDIA_TTS_MODEL = os.getenv("NVIDIA_TTS_MODEL", "magpie-tts-multilingual")
NVIDIA_TTS_FUNCTION_ID = os.getenv("NVIDIA_TTS_FUNCTION_ID", "877104f7-e885-42b9-8de8-f6e4c6303969")
NVIDIA_TTS_VOICE = os.getenv("NVIDIA_TTS_VOICE", "Magpie-Multilingual.EN-US.Sofia")

try:
    NVIDIA_RPM_LIMIT = int(os.getenv("NVIDIA_RPM_LIMIT", "40"))  # free tier ≈ 40 req/min
except ValueError:
    NVIDIA_RPM_LIMIT = 40

# Local OS voice (macOS `say`) — the always-available fallback. Empty = system default.
LOCAL_TTS_VOICE = os.getenv("LOCAL_TTS_VOICE", "")

# --------------------------------------------------------------------------- #
# Speech-to-text for the push-to-talk voice assistant (optional). Mirrors the
# TTS ladder: NVIDIA Riva ASR first (same NVCF gRPC host + NVIDIA_API_KEY as TTS,
# but its own ASR function id), then a HuggingFace Whisper model when the NVIDIA
# free tier is exhausted, then nothing (the mic widget simply hides). ASR_PROVIDER
# picks the order: "auto" (NVIDIA→HF), "nvidia", or "hf". All optional — with no
# keys, transcribe() returns None and the dashboard runs text-only as before.
# --------------------------------------------------------------------------- #
ASR_PROVIDER = os.getenv("ASR_PROVIDER", "auto").lower()
NVIDIA_ASR_FUNCTION_ID = os.getenv("NVIDIA_ASR_FUNCTION_ID", "")

# Local Whisper (faster-whisper) — the offline, no-cloud-key ASR fallback. Optional
# dependency (`uv sync --extra localasr`); when installed it joins the "auto" ladder
# after the cloud backends, so voice input works even with no/broken cloud keys. The
# model downloads once (~145 MB for "base") then runs on CPU (~1-3s per short clip).
LOCAL_ASR_MODEL = os.getenv("LOCAL_ASR_MODEL", "base")

# HuggingFace Inference API (text-only models over plain HTTPS via httpx — no new
# hard dependency). Used for the Whisper ASR fallback when NVIDIA is unavailable.
HF_API_KEY = os.getenv("HF_API_KEY", "")
# The legacy api-inference.huggingface.co host is RETIRED (NXDOMAIN). The serverless
# Inference API now routes through router.huggingface.co/hf-inference. transcribe.py
# rewrites a stale legacy base to the router automatically, so an old .env still works.
HF_BASE_URL = os.getenv("HF_BASE_URL", "https://router.huggingface.co")
HF_ASR_MODEL = os.getenv("HF_ASR_MODEL", "openai/whisper-large-v3")


# --------------------------------------------------------------------------- #
# Ceramics supply-chain optimizer (the second decision agent). Additive only —
# the gas agent never reads these. Defaults keep the ceramics tab fully
# deterministic and offline, mirroring the gas demo's "runs with no keys" rule.
# --------------------------------------------------------------------------- #
# Starting cost-factor weights for the ceramics lock decision (sum to 1.0). The
# UI lets the user re-weight; these are the calm defaults — gas-dominant, because
# firing gas is the largest volatile cost for the persona.
CERAMICS_DEFAULT_WEIGHTS: dict[str, float] = {
    "gas": 0.40,
    "clay": 0.35,
    "energy": 0.15,
    "transport": 0.10,
}

# Seed for the "random" backtest baseline. Seeded so the random strategy
# reproduces run to run — determinism is the selling point, even for the foil.
CERAMICS_BACKTEST_SEED = 7

# Seed for the gas backtest's random-hedge-ratio baseline (the "beats a coin flip"
# foil for the hedge ratio). Seeded for the same reproducibility reason as above.
GAS_BACKTEST_SEED = 7

# Committed mock 4-factor forecast — the ceramics demo's source of truth, the
# analogue of the cached gas job. Regenerate with scripts/build_ceramics_forecast.py.
CERAMICS_MOCK_FORECAST = CACHE_DIR / "mock_ceramics_forecast.json"

# Cache-key salt for the opt-in live 4-factor forecast. The combined job id is a
# hash of (this salt | the company description + chosen filters), so re-submitting
# the same profile reuses the cached artifact (no re-poll). Bump the salt to force
# every live profile to re-forecast (e.g. after changing the request scaffolding).
CERAMICS_CACHE_SALT = os.getenv("CERAMICS_CACHE_SALT", "ceramics-v1")

# Model that writes the one-line pipeline stage blurbs shown during a live
# forecast. Reuses the fast tag-picker model by default; falls back to committed
# templates with no key (THE RULE: these are status lines, never a decision).
CERAMICS_STAGE_BLURB_MODEL = os.getenv("CERAMICS_STAGE_BLURB_MODEL", KEYWORD_MODEL)


def have_sybilion_key() -> bool:
    return bool(SYBILION_API_KEY)


def have_featherless_key() -> bool:
    return bool(FEATHERLESS_API_KEY)


def have_nvidia_key() -> bool:
    return bool(NVIDIA_API_KEY)


def have_hf_key() -> bool:
    return bool(HF_API_KEY)
