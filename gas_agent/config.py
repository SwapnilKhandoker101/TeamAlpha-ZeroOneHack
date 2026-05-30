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

# HuggingFace Inference API (text-only models over plain HTTPS via httpx — no new
# hard dependency). Used for the Whisper ASR fallback when NVIDIA is unavailable.
HF_API_KEY = os.getenv("HF_API_KEY", "")
HF_BASE_URL = os.getenv("HF_BASE_URL", "https://api-inference.huggingface.co")
HF_ASR_MODEL = os.getenv("HF_ASR_MODEL", "openai/whisper-large-v3")


def have_sybilion_key() -> bool:
    return bool(SYBILION_API_KEY)


def have_featherless_key() -> bool:
    return bool(FEATHERLESS_API_KEY)


def have_nvidia_key() -> bool:
    return bool(NVIDIA_API_KEY)


def have_hf_key() -> bool:
    return bool(HF_API_KEY)
