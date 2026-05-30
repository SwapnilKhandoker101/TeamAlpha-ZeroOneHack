"""Pre-synthesise the demo narration to ``cache/audio/`` for stage reliability.

Rebuilds the same two explanations the dashboard shows — the calm "golden" path
and the Strait-of-Hormuz shock — straight from the cached Sybilion job (no
Streamlit, no decision made here), then voices each via :mod:`gas_agent.voice`.
Each clip is written next to a small JSON sidecar recording the provider/voice
actually used, so the app can play it instantly and still caption it truthfully.

If no TTS provider is reachable, ``voice.synthesize`` returns ``None`` and this
script says so and exits cleanly — the dashboard then just hides the player.

Run:  uv run python scripts/build_voiceover.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on the path

from gas_agent import config
from gas_agent import sybilion_client as sc
from gas_agent import voice
from gas_agent.driver_curation import curate_drivers
from gas_agent.explanation_agent import explain_decision
from gas_agent.hedge_policy import DEFAULT_PARAMS, decide_all, quarter_hedge_ratio
from gas_agent.keyword_agent import DEFAULT_PERSONA
from gas_agent.scenario import run_shock

SHOCK_LABEL = "Strait of Hormuz disruption"
AUDIO_DIR = config.CACHE_DIR / "audio"
_EXT = {"audio/mpeg": ".mp3", "audio/wav": ".wav"}


def _golden_text(job_id: str) -> str:
    """The calm narration — mirrors app.cached_explanation."""
    ttf = sc.load_ttf_series()
    forecast_json = sc.load_artifact(job_id, "forecast.json")
    signals = sc.load_artifact(job_id, "external_signals.json")
    spot = sc.last_actual_price(ttf)
    quarter = decide_all(sc.parse_forecast_months(forecast_json), spot, DEFAULT_PARAMS)[:3]
    curation = curate_drivers(signals)
    return explain_decision(
        quarter, quarter_hedge_ratio(quarter), spot,
        curation.top_drivers(6), curation.rejected, DEFAULT_PERSONA,
    ).text


def _shock_text(job_id: str) -> str:
    """The shock narration — mirrors app.cached_shock_explanation at magnitude 1.0."""
    ttf = sc.load_ttf_series()
    forecast_json = sc.load_artifact(job_id, "forecast.json")
    signals = sc.load_artifact(job_id, "external_signals.json")
    spot = sc.last_actual_price(ttf)
    months = sc.parse_forecast_months(forecast_json)
    outcome = run_shock(months, spot, curate_drivers(signals), 1.0, SHOCK_LABEL)
    quarter = outcome.decisions[:3]
    return explain_decision(
        quarter, quarter_hedge_ratio(quarter), spot,
        outcome.curation.top_drivers(6), outcome.curation.rejected, DEFAULT_PERSONA,
    ).text


def _save(name: str, text: str) -> bool:
    clip = voice.synthesize(text)
    if clip is None:
        print(f"  {name}: no TTS provider available — skipped")
        return False
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    audio_path = AUDIO_DIR / f"{name}{_EXT.get(clip.mime, '.bin')}"
    audio_path.write_bytes(clip.audio_bytes)
    sidecar = AUDIO_DIR / f"{name}.json"
    sidecar.write_text(json.dumps({
        "file": audio_path.name, "mime": clip.mime,
        "provider": clip.provider, "voice": clip.voice,
    }, indent=2))
    print(f"  {name}: wrote {audio_path.name} via {clip.provider}/{clip.voice} "
          f"({len(clip.audio_bytes):,} bytes)")
    return True


def main() -> None:
    job_id = sc.get_latest_job()
    if not job_id:
        print("No cached Sybilion job found (cache/latest_job.txt). Run a forecast first.")
        return

    print(f"Building voiceover from job {job_id} "
          f"(VOICE_PROVIDER={config.VOICE_PROVIDER}, "
          f"nvidia_key={'yes' if config.have_nvidia_key() else 'no'}, "
          f"featherless_key={'yes' if config.have_featherless_key() else 'no'})")
    golden, shock = _golden_text(job_id), _shock_text(job_id)
    print(f"  golden text: {golden[:120]}…")
    print(f"  shock  text: {shock[:120]}…")

    wrote_any = _save("golden", golden)
    wrote_any = _save("shock", shock) or wrote_any
    if not wrote_any:
        print("No audio written. Use the local voice (macOS `say`, the default fallback) or "
              "configure NVIDIA (NVIDIA_API_KEY + NVIDIA_TTS_FUNCTION_ID and "
              "`uv pip install nvidia-riva-client`). The dashboard runs fine without voice — "
              "it just hides the player.")


if __name__ == "__main__":
    main()
