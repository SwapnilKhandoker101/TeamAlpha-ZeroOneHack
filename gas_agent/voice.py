"""Optional text-to-speech narration for the dashboard.

Turns the agent's already-written explanation into a short spoken clip, so the
demo can *say* its decision out loud. Two providers, chosen in ``.env`` via
``VOICE_PROVIDER`` (``auto`` | ``nvidia`` | ``local``):

* **NVIDIA Riva** (``fastpitch-hifigan-tts``) — the primary cloud voice, but the
  free tier is rate-limited (~40 requests/minute). A process-local sliding window
  counts NVIDIA calls in the last 60 s; once the limit is hit — or a call comes
  back 429 / fails — the next clip is synthesised by the local fallback instead.
  The NVIDIA TTS models are served over gRPC (NVIDIA Cloud Functions), not the
  OpenAI-compatible chat endpoint, so the client is imported lazily and its
  absence simply routes to the fallback.
* **Local system TTS** — the always-available fallback, the OS ``say`` voice
  (macOS), needing no key or network. (Featherless, which the rest of the agent
  uses for text, has no speech endpoint — ``/v1/audio/speech`` 404s — so it can't
  voice anything; the local engine fills that role and keeps the demo audible
  off-line.)

Like everything else in this agent, this module never makes a decision — it only
voices text the deterministic policy and the explainer already produced. If no
provider is reachable (e.g. NVIDIA unset on a non-macOS host), :func:`synthesize`
returns ``None`` and the UI hides the player; the demo still runs.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from gas_agent import config


@dataclass
class VoiceClip:
    """A synthesised narration clip, ready to hand to ``st.audio``."""

    audio_bytes: bytes
    mime: str  # e.g. "audio/mpeg" or "audio/wav"
    provider: str  # "nvidia" | "featherless" — shown in the UI caption
    voice: str


class VoiceUnavailable(RuntimeError):
    """A provider backend could not synthesise. The router catches it and tries
    the next provider, returning ``None`` once none remain."""


class VoiceRateLimited(VoiceUnavailable):
    """NVIDIA refused the call for rate/quota reasons (a 429 or an exhausted
    free tier). The router treats this like a full window and lets the local
    fallback take over for the rest of the minute."""


# --------------------------------------------------------------------------- #
# NVIDIA free-tier rate limiter — a process-local sliding 60 s window.
# Streamlit is single-process, so this deque is shared across reruns and chat
# turns within a session, which is exactly the scope the free tier cares about.
# --------------------------------------------------------------------------- #
_WINDOW_SECONDS = 60.0
_NVIDIA_CALLS: deque[float] = deque()


def _evict_old(now: float) -> None:
    while _NVIDIA_CALLS and now - _NVIDIA_CALLS[0] > _WINDOW_SECONDS:
        _NVIDIA_CALLS.popleft()


def nvidia_calls_in_window(now: float | None = None) -> int:
    """How many NVIDIA calls landed in the last 60 s (evicting older ones)."""
    now = time.monotonic() if now is None else now
    _evict_old(now)
    return len(_NVIDIA_CALLS)


def nvidia_rate_limited(now: float | None = None) -> bool:
    """True when another NVIDIA call right now would exceed ``NVIDIA_RPM_LIMIT``."""
    return nvidia_calls_in_window(now) >= config.NVIDIA_RPM_LIMIT


def _record_nvidia_call(now: float | None = None) -> None:
    _NVIDIA_CALLS.append(time.monotonic() if now is None else now)


def _saturate_nvidia_window(now: float | None = None) -> None:
    """Fill the window to the limit — used after a 429 so we stop hammering
    NVIDIA and route to Featherless for the rest of the minute."""
    now = time.monotonic() if now is None else now
    _evict_old(now)
    while len(_NVIDIA_CALLS) < config.NVIDIA_RPM_LIMIT:
        _NVIDIA_CALLS.append(now)


def reset_rate_limiter() -> None:
    """Clear the sliding window (used by tests and a fresh session)."""
    _NVIDIA_CALLS.clear()


def provider_order() -> list[str]:
    """Resolve the provider preference from ``VOICE_PROVIDER``.

    ``auto``/``nvidia`` both prefer NVIDIA and fall back to the local engine;
    ``local`` forces the local engine only. ``featherless`` is accepted as a
    legacy alias for ``local`` (Featherless has no speech endpoint).
    """
    choice = (config.VOICE_PROVIDER or "auto").lower()
    if choice in ("local", "featherless"):
        return ["local"]
    return ["nvidia", "local"]  # "auto" and "nvidia"


def synthesize(text: str, *, providers: list[str] | None = None) -> VoiceClip | None:
    """Narrate ``text``, honouring the provider order and the NVIDIA rate limit.

    Returns a :class:`VoiceClip`, or ``None`` when every eligible provider is
    unavailable (no key, over the limit, no local engine, or a failed call) — the
    caller then just hides the audio player.

    ``providers`` overrides the configured order for this one call (default
    ``None`` → :func:`provider_order`, i.e. today's behaviour unchanged). The
    pipeline filler passes ``["local"]`` to force the free, instant local ``say``
    voice so it never burns the NVIDIA free-tier window on throwaway chatter, while
    the headline narration keeps using the configured order.
    """
    clean = (text or "").strip()
    if not clean:
        return None

    for provider in (providers or provider_order()):
        if provider == "nvidia":
            if not config.have_nvidia_key():
                continue
            if nvidia_rate_limited():
                continue  # over the free-tier window — let the local engine take it
            try:
                clip = _synthesize_nvidia(clean)
            except VoiceRateLimited:
                _saturate_nvidia_window()  # a 429 — back off NVIDIA for the minute
                continue
            except Exception:
                continue
            _record_nvidia_call()
            return clip
        if provider == "local":
            if not _local_available():
                continue
            try:
                return _synthesize_local(clean)
            except Exception:
                continue
    return None


# --------------------------------------------------------------------------- #
# Provider backends. Each raises VoiceUnavailable (or VoiceRateLimited) on any
# failure so the router can fall through; they never return None themselves.
# --------------------------------------------------------------------------- #
def _pcm_to_wav(pcm_bytes: bytes, sample_rate_hz: int, channels: int = 1,
                sample_width_bytes: int = 2) -> bytes:
    """Wrap raw little-endian PCM (what Riva LINEAR_PCM returns) in a WAV header
    so the browser's ``st.audio`` can play it without any decoding step."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width_bytes)
        wav.setframerate(sample_rate_hz)
        wav.writeframes(pcm_bytes)
    return buffer.getvalue()


def _synthesize_nvidia(text: str) -> VoiceClip:
    """Synthesise via NVIDIA Riva over gRPC (NVIDIA Cloud Functions).

    The ``nvidia-riva-client`` package is optional and imported lazily — if it
    is not installed, or auth/function-id is missing, this raises so the router
    falls back to Featherless.
    """
    try:
        import riva.client  # type: ignore
    except ImportError as error:
        raise VoiceUnavailable("nvidia-riva-client not installed") from error

    if not config.NVIDIA_TTS_FUNCTION_ID:
        raise VoiceUnavailable("NVIDIA_TTS_FUNCTION_ID not configured")

    sample_rate_hz = 44_100
    try:
        auth = riva.client.Auth(
            uri=config.NVIDIA_RIVA_URI,
            use_ssl=True,
            metadata_args=[
                ["function-id", config.NVIDIA_TTS_FUNCTION_ID],
                ["authorization", f"Bearer {config.NVIDIA_API_KEY}"],
            ],
        )
        service = riva.client.SpeechSynthesisService(auth)
        response = service.synthesize(
            text=text,
            voice_name=config.NVIDIA_TTS_VOICE,
            language_code="en-US",
            sample_rate_hz=sample_rate_hz,
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
        )
    except Exception as error:
        message = str(error).lower()
        if any(flag in message for flag in ("429", "rate", "quota", "exhaust", "resource_exhausted")):
            raise VoiceRateLimited(str(error)) from error
        raise VoiceUnavailable(str(error)) from error

    audio = getattr(response, "audio", b"")
    if not audio:
        raise VoiceUnavailable("NVIDIA returned no audio")
    wav_bytes = _pcm_to_wav(audio, sample_rate_hz=sample_rate_hz)
    return VoiceClip(wav_bytes, "audio/wav", "nvidia", config.NVIDIA_TTS_VOICE)


def _local_available() -> bool:
    """True when an OS text-to-speech binary is on PATH (macOS ``say``)."""
    return shutil.which("say") is not None


def _synthesize_local(text: str) -> VoiceClip:
    """Synthesise with the local OS voice (macOS ``say``) to a WAV — no key, no
    network, so the demo is always audible on the presenter's machine."""
    if shutil.which("say") is None:
        raise VoiceUnavailable("no local TTS engine (say) on PATH")

    voice_name = config.LOCAL_TTS_VOICE
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "clip.wav"
        command = ["say", "-o", str(out),
                   "--file-format=WAVE", "--data-format=LEI16@22050"]
        if voice_name:
            command += ["-v", voice_name]
        command.append(text)
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=60)
        except Exception as error:
            raise VoiceUnavailable(f"local TTS failed: {error}") from error
        audio = out.read_bytes()
    if not audio:
        raise VoiceUnavailable("local TTS produced no audio")
    return VoiceClip(audio, "audio/wav", "local", voice_name or "system")
