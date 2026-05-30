"""Optional speech-to-text for the push-to-talk voice assistant.

Mirrors :mod:`gas_agent.voice`: an NVIDIA-first ladder that degrades gracefully
and returns ``None`` when nothing is reachable, so the mic widget simply hides
and the dashboard runs text-only exactly as before.

* **NVIDIA Riva ASR** — ``offline_recognize`` over the same NVCF gRPC host and
  ``NVIDIA_API_KEY`` as TTS, with its own ``NVIDIA_ASR_FUNCTION_ID``. The
  ``nvidia-riva-client`` package is optional and imported lazily; on a 429/quota
  error it saturates the **shared** 60 s window in :mod:`gas_agent.voice` and
  falls through, so TTS and ASR together stay under the ~40/min free tier.
* **HuggingFace Whisper** — POSTs the WAV bytes to the HF Inference API over the
  existing ``httpx`` dependency (no new hard dep): the "switch to a HuggingFace
  model when the NVIDIA tier runs out" path.
* else ``None`` — the mic disables with a hint to set a key.

Like :mod:`gas_agent.voice`, this only transcribes audio to text; it never makes
or alters a decision (THE RULE). The transcript is fed to the *same* chat
branches a typed message would hit.
"""

from __future__ import annotations

import io
import wave

import httpx

from gas_agent import config
from gas_agent import voice  # share the NVIDIA 60 s rate-limit window


class TranscriptionUnavailable(RuntimeError):
    """A backend could not transcribe; the ladder falls through to the next."""


class TranscriptionRateLimited(TranscriptionUnavailable):
    """NVIDIA refused for rate/quota reasons — saturate the shared window and let
    the HuggingFace fallback take over for the rest of the minute."""


# Quota/rate flags shared with voice.py's NVIDIA error classification.
_RATE_FLAGS = ("429", "rate", "quota", "exhaust", "resource_exhausted")


def provider_order() -> list[str]:
    """Resolve ASR preference from ``ASR_PROVIDER``: ``auto`` (nvidia→hf),
    ``nvidia``, or ``hf``."""
    choice = (config.ASR_PROVIDER or "auto").lower()
    if choice == "nvidia":
        return ["nvidia"]
    if choice == "hf":
        return ["hf"]
    return ["nvidia", "hf"]  # "auto"


def available() -> bool:
    """True when at least one configured backend could plausibly transcribe — the
    UI uses this to decide whether to show the mic widget at all."""
    order = provider_order()
    if "nvidia" in order and config.have_nvidia_key() and config.NVIDIA_ASR_FUNCTION_ID:
        return True
    if "hf" in order and config.have_hf_key():
        return True
    return False


def transcribe(wav_bytes: bytes) -> str | None:
    """Transcribe recorded WAV audio to text, honouring the provider order and the
    shared NVIDIA rate-limit window.

    Returns the transcript, or ``None`` when every eligible backend is unavailable
    (no key/function-id, over the shared limit, package missing, or a failed call)
    — the caller then hides or disables the mic.
    """
    if not wav_bytes:
        return None

    for provider in provider_order():
        if provider == "nvidia":
            if not (config.have_nvidia_key() and config.NVIDIA_ASR_FUNCTION_ID):
                continue
            if voice.nvidia_rate_limited():
                continue  # over the shared free-tier window — let HF take it
            try:
                text = _transcribe_nvidia(wav_bytes)
            except TranscriptionRateLimited:
                voice._saturate_nvidia_window()  # a 429 — back off NVIDIA for the minute
                continue
            except Exception:
                continue
            voice._record_nvidia_call()
            if text:
                return text
            continue
        if provider == "hf":
            if not config.have_hf_key():
                continue
            try:
                text = _transcribe_hf(wav_bytes)
            except Exception:
                continue
            if text:
                return text
            continue
    return None


# --------------------------------------------------------------------------- #
# Provider backends. Each raises TranscriptionUnavailable (or
# TranscriptionRateLimited) on failure so the ladder can fall through.
# --------------------------------------------------------------------------- #
def _wav_sample_rate(wav_bytes: bytes, default: int = 16_000) -> int:
    """Read the sample rate from a WAV header, defaulting if it can't be parsed."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            return wav.getframerate() or default
    except Exception:
        return default


def _extract_transcript(response) -> str:
    """Pull the best transcript out of a Riva ASR response."""
    results = getattr(response, "results", None) or []
    parts: list[str] = []
    for result in results:
        alternatives = getattr(result, "alternatives", None) or []
        if alternatives:
            parts.append(str(getattr(alternatives[0], "transcript", "") or ""))
    return " ".join(p.strip() for p in parts if p.strip()).strip()


def _transcribe_nvidia(wav_bytes: bytes) -> str:
    """Transcribe via NVIDIA Riva ASR over gRPC (NVIDIA Cloud Functions).

    ``nvidia-riva-client`` is optional and imported lazily — a missing package or
    auth/function-id raises so the ladder falls through to HuggingFace.
    """
    try:
        import riva.client  # type: ignore
    except ImportError as error:
        raise TranscriptionUnavailable("nvidia-riva-client not installed") from error

    if not config.NVIDIA_ASR_FUNCTION_ID:
        raise TranscriptionUnavailable("NVIDIA_ASR_FUNCTION_ID not configured")

    try:
        auth = riva.client.Auth(
            uri=config.NVIDIA_RIVA_URI,
            use_ssl=True,
            metadata_args=[
                ["function-id", config.NVIDIA_ASR_FUNCTION_ID],
                ["authorization", f"Bearer {config.NVIDIA_API_KEY}"],
            ],
        )
        service = riva.client.ASRService(auth)
        recognition_config = riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=_wav_sample_rate(wav_bytes),
            language_code="en-US",
            max_alternatives=1,
            enable_automatic_punctuation=True,
        )
        response = service.offline_recognize(wav_bytes, recognition_config)
    except Exception as error:
        message = str(error).lower()
        if any(flag in message for flag in _RATE_FLAGS):
            raise TranscriptionRateLimited(str(error)) from error
        raise TranscriptionUnavailable(str(error)) from error

    return _extract_transcript(response)


def _transcribe_hf(wav_bytes: bytes) -> str:
    """Transcribe via a HuggingFace Inference API Whisper model (plain HTTPS).

    Raises on any HTTP error so the ladder falls through to ``None``.
    """
    url = f"{config.HF_BASE_URL.rstrip('/')}/models/{config.HF_ASR_MODEL}"
    headers = {
        "Authorization": f"Bearer {config.HF_API_KEY}",
        "Content-Type": "audio/wav",
    }
    response = httpx.post(url, headers=headers, content=wav_bytes, timeout=60.0)
    response.raise_for_status()
    data = response.json()
    # HF ASR returns {"text": "..."}; some pipelines wrap it in a list.
    if isinstance(data, dict):
        return str(data.get("text", "")).strip()
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return str(data[0].get("text", "")).strip()
    return ""
