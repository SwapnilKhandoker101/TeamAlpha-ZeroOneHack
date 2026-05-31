"""Unit tests for the voice-narration router and its NVIDIA rate limiter.

All deterministic — no network and no TTS provider is ever called. The provider
backends are monkeypatched, and the sliding-window limiter is driven with
injected timestamps, so the routing logic (NVIDIA primary, Featherless fallback,
429 back-off, graceful None) is proven without a key.
"""

import io
import wave

import pytest

import gas_agent.config as config
import gas_agent.voice as voice
from gas_agent.voice import VoiceClip, VoiceRateLimited, VoiceUnavailable


@pytest.fixture(autouse=True)
def _clean_window():
    voice.reset_rate_limiter()
    yield
    voice.reset_rate_limiter()


def _clip(provider: str) -> VoiceClip:
    return VoiceClip(b"\x00\x01", "audio/wav", provider, "test-voice")


# --------------------------------------------------------------------------- #
# provider_order
# --------------------------------------------------------------------------- #
def test_auto_and_nvidia_prefer_nvidia_then_local(monkeypatch):
    monkeypatch.setattr(config, "VOICE_PROVIDER", "auto")
    assert voice.provider_order() == ["nvidia", "local"]
    monkeypatch.setattr(config, "VOICE_PROVIDER", "nvidia")
    assert voice.provider_order() == ["nvidia", "local"]


def test_local_choice_forces_local_only(monkeypatch):
    monkeypatch.setattr(config, "VOICE_PROVIDER", "local")
    assert voice.provider_order() == ["local"]
    # "featherless" is a legacy alias — Featherless has no TTS, so it means local.
    monkeypatch.setattr(config, "VOICE_PROVIDER", "featherless")
    assert voice.provider_order() == ["local"]


# --------------------------------------------------------------------------- #
# Sliding-window rate limiter
# --------------------------------------------------------------------------- #
def test_window_counts_recent_calls_and_evicts_old_ones(monkeypatch):
    monkeypatch.setattr(config, "NVIDIA_RPM_LIMIT", 40)
    voice._record_nvidia_call(now=1000.0)
    voice._record_nvidia_call(now=1001.0)
    assert voice.nvidia_calls_in_window(now=1002.0) == 2
    # 61 s later the first two have aged out of the 60 s window.
    assert voice.nvidia_calls_in_window(now=1062.0) == 0


def test_rate_limited_flips_exactly_at_the_limit(monkeypatch):
    monkeypatch.setattr(config, "NVIDIA_RPM_LIMIT", 3)
    for offset in range(2):
        voice._record_nvidia_call(now=2000.0 + offset)
    assert voice.nvidia_rate_limited(now=2002.0) is False  # 2 < 3
    voice._record_nvidia_call(now=2002.0)
    assert voice.nvidia_rate_limited(now=2002.5) is True  # 3 >= 3


# --------------------------------------------------------------------------- #
# synthesize routing
# --------------------------------------------------------------------------- #
def test_empty_text_returns_none():
    assert voice.synthesize("   ") is None


def test_uses_nvidia_when_available_and_under_limit(monkeypatch):
    monkeypatch.setattr(config, "VOICE_PROVIDER", "auto")
    monkeypatch.setattr(config, "have_nvidia_key", lambda: True)
    monkeypatch.setattr(voice, "_local_available", lambda: True)
    monkeypatch.setattr(voice, "_synthesize_nvidia", lambda text: _clip("nvidia"))
    monkeypatch.setattr(voice, "_synthesize_local",
                        lambda text: pytest.fail("should not reach the local engine"))

    clip = voice.synthesize("hello")
    assert clip is not None and clip.provider == "nvidia"
    assert voice.nvidia_calls_in_window() == 1  # the successful call was recorded


def test_falls_back_to_local_when_over_the_nvidia_limit(monkeypatch):
    monkeypatch.setattr(config, "VOICE_PROVIDER", "auto")
    monkeypatch.setattr(config, "NVIDIA_RPM_LIMIT", 2)
    monkeypatch.setattr(config, "have_nvidia_key", lambda: True)
    monkeypatch.setattr(voice, "_local_available", lambda: True)
    monkeypatch.setattr(voice, "_synthesize_nvidia",
                        lambda text: pytest.fail("NVIDIA should be skipped over the limit"))
    monkeypatch.setattr(voice, "_synthesize_local", lambda text: _clip("local"))

    voice._saturate_nvidia_window()  # window now full
    clip = voice.synthesize("hello")
    assert clip is not None and clip.provider == "local"


def test_429_from_nvidia_saturates_window_then_uses_local(monkeypatch):
    monkeypatch.setattr(config, "VOICE_PROVIDER", "auto")
    monkeypatch.setattr(config, "NVIDIA_RPM_LIMIT", 3)
    monkeypatch.setattr(config, "have_nvidia_key", lambda: True)
    monkeypatch.setattr(voice, "_local_available", lambda: True)

    def boom(text):
        raise VoiceRateLimited("429 too many requests")

    monkeypatch.setattr(voice, "_synthesize_nvidia", boom)
    monkeypatch.setattr(voice, "_synthesize_local", lambda text: _clip("local"))

    clip = voice.synthesize("hello")
    assert clip is not None and clip.provider == "local"
    assert voice.nvidia_rate_limited() is True  # the 429 backed NVIDIA off for the minute


def test_nvidia_generic_error_falls_through_without_saturating(monkeypatch):
    monkeypatch.setattr(config, "VOICE_PROVIDER", "auto")
    monkeypatch.setattr(config, "NVIDIA_RPM_LIMIT", 3)
    monkeypatch.setattr(config, "have_nvidia_key", lambda: True)
    monkeypatch.setattr(voice, "_local_available", lambda: True)

    def boom(text):
        raise VoiceUnavailable("connection reset")

    monkeypatch.setattr(voice, "_synthesize_nvidia", boom)
    monkeypatch.setattr(voice, "_synthesize_local", lambda text: _clip("local"))

    clip = voice.synthesize("hello")
    assert clip is not None and clip.provider == "local"
    # A plain failure must NOT saturate the window — NVIDIA can be retried.
    assert voice.nvidia_rate_limited() is False


def test_returns_none_when_no_provider_is_available(monkeypatch):
    monkeypatch.setattr(config, "VOICE_PROVIDER", "auto")
    monkeypatch.setattr(config, "have_nvidia_key", lambda: False)
    monkeypatch.setattr(voice, "_local_available", lambda: False)
    assert voice.synthesize("hello") is None


def test_local_only_never_touches_nvidia(monkeypatch):
    monkeypatch.setattr(config, "VOICE_PROVIDER", "local")
    monkeypatch.setattr(config, "have_nvidia_key", lambda: True)  # key present but must be ignored
    monkeypatch.setattr(voice, "_local_available", lambda: True)
    monkeypatch.setattr(voice, "_synthesize_nvidia",
                        lambda text: pytest.fail("local-only must skip NVIDIA"))
    monkeypatch.setattr(voice, "_synthesize_local", lambda text: _clip("local"))

    clip = voice.synthesize("hello")
    assert clip is not None and clip.provider == "local"


def test_providers_override_forces_local_even_with_nvidia(monkeypatch):
    # The pipeline filler passes providers=["local"] so it never burns the NVIDIA
    # free-tier window on throwaway chatter — even with a key and room under the limit.
    monkeypatch.setattr(config, "VOICE_PROVIDER", "auto")
    monkeypatch.setattr(config, "have_nvidia_key", lambda: True)
    monkeypatch.setattr(voice, "_local_available", lambda: True)
    monkeypatch.setattr(voice, "_synthesize_nvidia",
                        lambda text: pytest.fail("providers=['local'] must skip NVIDIA"))
    monkeypatch.setattr(voice, "_synthesize_local", lambda text: _clip("local"))

    clip = voice.synthesize("filler", providers=["local"])
    assert clip is not None and clip.provider == "local"
    assert voice.nvidia_calls_in_window() == 0  # NVIDIA window untouched


def test_providers_none_keeps_configured_order(monkeypatch):
    # Default providers=None must reproduce today's behaviour (configured order).
    monkeypatch.setattr(config, "VOICE_PROVIDER", "auto")
    monkeypatch.setattr(config, "have_nvidia_key", lambda: True)
    monkeypatch.setattr(voice, "_local_available", lambda: True)
    monkeypatch.setattr(voice, "_synthesize_nvidia", lambda text: _clip("nvidia"))
    monkeypatch.setattr(voice, "_synthesize_local",
                        lambda text: pytest.fail("default order should prefer NVIDIA"))
    clip = voice.synthesize("hello")
    assert clip is not None and clip.provider == "nvidia"


# --------------------------------------------------------------------------- #
# PCM → WAV helper (NVIDIA returns raw LINEAR_PCM)
# --------------------------------------------------------------------------- #
def test_pcm_to_wav_wraps_a_playable_header():
    pcm = b"\x00\x01" * 100  # 100 frames of 16-bit mono
    wav_bytes = voice._pcm_to_wav(pcm, sample_rate_hz=44_100)
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 44_100
        assert wav.getnframes() == 100
