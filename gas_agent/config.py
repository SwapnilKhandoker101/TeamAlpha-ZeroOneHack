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


def have_sybilion_key() -> bool:
    return bool(SYBILION_API_KEY)


def have_featherless_key() -> bool:
    return bool(FEATHERLESS_API_KEY)
