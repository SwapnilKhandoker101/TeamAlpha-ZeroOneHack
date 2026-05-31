"""Offline tests for the speech-to-text ladder (Workstream 4).

No network and no real Riva: a fake ``riva.client`` is injected into
``sys.modules`` and the HuggingFace call is monkeypatched at ``httpx.post``. The
contract: NVIDIA is tried first; a 429/quota error saturates the *shared* rate
window and falls through to HuggingFace Whisper; with neither backend configured
``transcribe`` returns ``None`` (so the mic widget hides).
"""

import io
import sys
import types
import wave

from gas_agent import config
from gas_agent import transcribe
from gas_agent import voice


def _wav_bytes(sample_rate: int = 16_000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * 16)
    return buf.getvalue()


class _FakeResp:
    def __init__(self, payload, status: int = 200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _install_fake_riva(monkeypatch, *, transcript: str | None = None, error: Exception | None = None):
    """Inject a minimal fake ``riva.client`` so ``import riva.client`` succeeds."""
    riva_mod = types.ModuleType("riva")
    client_mod = types.ModuleType("riva.client")

    class AudioEncoding:
        LINEAR_PCM = "LINEAR_PCM"

    class Auth:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class RecognitionConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _Alt:
        def __init__(self, text):
            self.transcript = text

    class _Result:
        def __init__(self, text):
            self.alternatives = [_Alt(text)]

    class _Response:
        def __init__(self, text):
            self.results = [_Result(text)]

    class ASRService:
        def __init__(self, auth):
            self.auth = auth

        def offline_recognize(self, audio, recognition_config):
            if error is not None:
                raise error
            return _Response(transcript or "")

    client_mod.AudioEncoding = AudioEncoding
    client_mod.Auth = Auth
    client_mod.RecognitionConfig = RecognitionConfig
    client_mod.ASRService = ASRService
    riva_mod.client = client_mod
    monkeypatch.setitem(sys.modules, "riva", riva_mod)
    monkeypatch.setitem(sys.modules, "riva.client", client_mod)


def test_transcribe_returns_none_without_any_backend(monkeypatch):
    # No cloud keys AND no local Whisper installed → the mic hides, transcribe() is None.
    monkeypatch.setattr(config, "ASR_PROVIDER", "auto")
    monkeypatch.setattr(config, "NVIDIA_API_KEY", "")
    monkeypatch.setattr(config, "NVIDIA_ASR_FUNCTION_ID", "")
    monkeypatch.setattr(config, "HF_API_KEY", "")
    monkeypatch.setattr(transcribe, "_local_available", lambda: False)  # faster-whisper absent
    assert transcribe.available() is False
    assert transcribe.transcribe(_wav_bytes()) is None


def test_nvidia_is_tried_first_and_succeeds(monkeypatch):
    voice.reset_rate_limiter()
    monkeypatch.setattr(config, "ASR_PROVIDER", "auto")
    monkeypatch.setattr(config, "NVIDIA_API_KEY", "nvapi-x")
    monkeypatch.setattr(config, "NVIDIA_ASR_FUNCTION_ID", "fn-asr")
    monkeypatch.setattr(config, "HF_API_KEY", "")  # HF off — proves the NVIDIA path
    _install_fake_riva(monkeypatch, transcript="why this hedge ratio")

    assert transcribe.transcribe(_wav_bytes()) == "why this hedge ratio"
    # The successful call is recorded against the window shared with TTS.
    assert voice.nvidia_calls_in_window() == 1


def test_nvidia_429_falls_through_to_huggingface(monkeypatch):
    voice.reset_rate_limiter()
    monkeypatch.setattr(config, "ASR_PROVIDER", "auto")
    monkeypatch.setattr(config, "NVIDIA_API_KEY", "nvapi-x")
    monkeypatch.setattr(config, "NVIDIA_ASR_FUNCTION_ID", "fn-asr")
    monkeypatch.setattr(config, "HF_API_KEY", "hf-x")
    _install_fake_riva(monkeypatch, error=RuntimeError("StatusCode.RESOURCE_EXHAUSTED 429"))

    seen: dict[str, str] = {}

    def fake_post(url, headers=None, content=None, timeout=None):
        seen["url"] = url
        return _FakeResp({"text": "which supplier matters most"})

    monkeypatch.setattr(transcribe.httpx, "post", fake_post)

    assert transcribe.transcribe(_wav_bytes()) == "which supplier matters most"
    assert config.HF_ASR_MODEL in seen["url"]
    # The 429 saturated the shared window, so further NVIDIA calls back off.
    assert voice.nvidia_rate_limited() is True


def test_hf_handles_list_shaped_response(monkeypatch):
    voice.reset_rate_limiter()
    monkeypatch.setattr(config, "ASR_PROVIDER", "hf")
    monkeypatch.setattr(config, "HF_API_KEY", "hf-x")
    monkeypatch.setattr(
        transcribe.httpx, "post",
        lambda *a, **k: _FakeResp([{"text": "explain the decision"}]),
    )
    assert transcribe.transcribe(_wav_bytes()) == "explain the decision"


def test_hf_failure_returns_none(monkeypatch):
    voice.reset_rate_limiter()
    monkeypatch.setattr(config, "ASR_PROVIDER", "hf")
    monkeypatch.setattr(config, "HF_API_KEY", "hf-x")
    monkeypatch.setattr(
        transcribe.httpx, "post",
        lambda *a, **k: _FakeResp({"error": "model loading"}, status=503),
    )
    assert transcribe.transcribe(_wav_bytes()) is None


# --------------------------------------------------------------------------- #
# The HF endpoint fix (the retired api-inference host → the router) + error surfacing
# --------------------------------------------------------------------------- #
def test_hf_url_rewrites_the_retired_host_to_the_router(monkeypatch):
    # A stale .env still pointing at the dead host is transparently rewritten.
    monkeypatch.setattr(config, "HF_BASE_URL", "https://api-inference.huggingface.co")
    monkeypatch.setattr(config, "HF_ASR_MODEL", "openai/whisper-large-v3")
    assert transcribe._hf_url() == (
        "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3")


def test_hf_url_adds_provider_path_for_the_router_and_leaves_custom_bases(monkeypatch):
    monkeypatch.setattr(config, "HF_ASR_MODEL", "openai/whisper-large-v3")
    monkeypatch.setattr(config, "HF_BASE_URL", "https://router.huggingface.co")
    assert transcribe._hf_url().endswith("/hf-inference/models/openai/whisper-large-v3")
    # A custom / self-hosted base is used as-is with the plain /models path.
    monkeypatch.setattr(config, "HF_BASE_URL", "https://my-endpoint.example.com")
    assert transcribe._hf_url() == "https://my-endpoint.example.com/models/openai/whisper-large-v3"


def test_last_error_is_surfaced_when_all_backends_fail(monkeypatch):
    voice.reset_rate_limiter()
    monkeypatch.setattr(config, "ASR_PROVIDER", "hf")
    monkeypatch.setattr(config, "HF_API_KEY", "hf-x")
    monkeypatch.setattr(
        transcribe.httpx, "post",
        lambda *a, **k: _FakeResp({"error": "no permission"}, status=403),
    )
    assert transcribe.transcribe(_wav_bytes()) is None
    assert "HuggingFace" in transcribe.last_error()  # the reason is now visible to the UI


def test_last_error_clears_on_success(monkeypatch):
    voice.reset_rate_limiter()
    monkeypatch.setattr(config, "ASR_PROVIDER", "hf")
    monkeypatch.setattr(config, "HF_API_KEY", "hf-x")
    monkeypatch.setattr(transcribe.httpx, "post", lambda *a, **k: _FakeResp({"text": "ok"}))
    assert transcribe.transcribe(_wav_bytes()) == "ok"
    assert transcribe.last_error() == ""


# --------------------------------------------------------------------------- #
# Local Whisper — the offline, no-cloud-key fallback (faster-whisper, optional)
# --------------------------------------------------------------------------- #
class _FakeSeg:
    def __init__(self, text):
        self.text = text


class _FakeWhisperModel:
    def __init__(self, *args, **kwargs):
        pass

    def transcribe(self, path, **kwargs):
        return [_FakeSeg("local transcript here")], None


def _install_fake_faster_whisper(monkeypatch):
    module = types.ModuleType("faster_whisper")
    module.WhisperModel = _FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    monkeypatch.setattr(transcribe, "_local_available", lambda: True)
    monkeypatch.setattr(transcribe, "_local_model", None)


def test_provider_order_includes_local_in_auto(monkeypatch):
    monkeypatch.setattr(config, "ASR_PROVIDER", "auto")
    assert transcribe.provider_order() == ["nvidia", "hf", "local"]
    monkeypatch.setattr(config, "ASR_PROVIDER", "local")
    assert transcribe.provider_order() == ["local"]


def test_local_whisper_works_with_no_cloud_keys(monkeypatch):
    voice.reset_rate_limiter()
    monkeypatch.setattr(config, "ASR_PROVIDER", "auto")
    monkeypatch.setattr(config, "NVIDIA_API_KEY", "")
    monkeypatch.setattr(config, "NVIDIA_ASR_FUNCTION_ID", "")
    monkeypatch.setattr(config, "HF_API_KEY", "")  # NO cloud keys at all
    _install_fake_faster_whisper(monkeypatch)

    assert transcribe.available() is True  # the mic shows on local Whisper alone
    assert transcribe.transcribe(_wav_bytes()) == "local transcript here"
    assert transcribe.last_error() == ""


def test_local_whisper_is_the_last_resort_after_cloud(monkeypatch):
    # Cloud backends fail → the ladder falls through to local Whisper.
    voice.reset_rate_limiter()
    monkeypatch.setattr(config, "ASR_PROVIDER", "auto")
    monkeypatch.setattr(config, "NVIDIA_API_KEY", "")
    monkeypatch.setattr(config, "NVIDIA_ASR_FUNCTION_ID", "")
    monkeypatch.setattr(config, "HF_API_KEY", "hf-x")
    monkeypatch.setattr(transcribe.httpx, "post",
                        lambda *a, **k: _FakeResp({"error": "boom"}, status=500))
    _install_fake_faster_whisper(monkeypatch)
    assert transcribe.transcribe(_wav_bytes()) == "local transcript here"
