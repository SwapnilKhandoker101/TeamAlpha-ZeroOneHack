"""The voice-guided tour — a spoken answer that drives the page (W16).

When a question is asked **by voice**, the answer should not just play: the page should
*scroll to the elements it's talking about* and the 3D globe should *rotate/zoom to the
country it names*, in sync with the narration. This module is the pure, Streamlit-free,
JS-free core of that: it turns an already-written answer into an ordered list of **beats**,
each = a short sentence + a DOM anchor to scroll to + an optional globe pose + (after
synthesis) its audio. The app renders the beats through one ``st.components.v1.html`` JS
bridge that talks to the parent page.

**THE RULE holds.** The answer text is produced by the explanation-only layer
(``voice_chat``) — this module only *splits* it into sentences (plain regex) and maps each
to a section anchor; it never computes or alters a decision number. The beat list is fully
deterministic, so it is unit-testable without a browser.

Tour audio uses the **local** voice (``voice.synthesize(providers=["local"])``) — instant,
offline, and free of the NVIDIA rate limit — so a tour never burns the headline-clip budget.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, replace

from gas_agent import voice
from gas_agent.geo import coords_for as _gas_coords_for

# Anchor slugs — must match the invisible ``anchor_html`` markers the app places before
# each section heading. Five gas-side, two ceramics-side.
GAS = "gas"
HEDGE = "hedge"
BACKTEST = "backtest"
WHY = "why"
DRIVERS_GLOBE = "drivers_globe"
CERAMICS = "ceramics"
CER_MAP = "cer_map"
GLOBE_ANCHORS = frozenset({DRIVERS_GLOBE, CER_MAP})

# Which sections a kind of answer walks through, in order. Sentences are zipped onto these
# (the last anchor repeats if there are more sentences than anchors).
ANCHOR_TEMPLATES: dict[str, list[str]] = {
    "shock": [WHY, HEDGE, BACKTEST, DRIVERS_GLOBE],
    "why": [WHY, HEDGE, DRIVERS_GLOBE],
    "country": [DRIVERS_GLOBE],  # "which supplier/country" — walk the globe per named region
    "ceramics": [CERAMICS, CER_MAP],
    "about": [GAS, CERAMICS],
}
_DEFAULT_TEMPLATE = ANCHOR_TEMPLATES["why"]

_MAX_BEATS = 6  # keep a tour under ~25s and the inlined audio small

# Globe pose tuning. Orthographic projection centres on the rotation lon/lat; scale zooms in.
_DEFAULT_SCALE = 2.0
_SCALE_OVERRIDE: dict[str, float] = {
    "Netherlands": 2.4, "Switzerland": 2.4, "Qatar": 2.5, "Belgium": 2.4,  # small countries
    "Russian Federation": 1.5, "Russia": 1.5, "Europe (aggregate)": 1.5,  # broad stories
}
# A few coordinates the gas table doesn't carry (a ceramics demand market).
_EXTRA_COORDS: dict[str, tuple[float, float]] = {"Switzerland": (46.8, 8.2)}


@dataclass(frozen=True)
class Beat:
    """One step of the tour: a sentence, where to look, and (after synth) its audio."""

    say: str
    anchor: str
    pose: dict | None = None  # {"lon","lat","scale"} for a globe beat, else None
    audio_b64: str | None = None
    mime: str = "audio/wav"

    def to_dict(self) -> dict:
        return {"say": self.say, "anchor": self.anchor, "pose": self.pose,
                "audio_b64": self.audio_b64, "mime": self.mime}


def anchor_html(slug: str) -> str:
    """A zero-height, invisible scroll target placed before a section heading. The negative
    offset keeps the heading clear of any sticky top bar when scrolled into view."""
    return (f"<div data-tour-anchor='{slug}' "
            "style='position:relative;top:-72px;height:0;visibility:hidden'></div>")


def _coords(region: str) -> tuple[float, float] | None:
    point = _gas_coords_for(region)
    return point if point is not None else _EXTRA_COORDS.get(region)


def pose_for(region: str, scale: float | None = None) -> dict | None:
    """A globe pose {lon, lat, scale} for a region, or ``None`` if it can't be placed.
    (``geo.REGION_COORDS`` is (lat, lon); Plotly's rotation wants {lon, lat} — converted
    here, in one place.)"""
    point = _coords(region)
    if point is None:
        return None
    lat, lon = point
    return {"lon": lon, "lat": lat, "scale": scale or _SCALE_OVERRIDE.get(region, _DEFAULT_SCALE)}


def _sentences(text: str, limit: int = _MAX_BEATS) -> list[str]:
    """Split an answer into at most ``limit`` sentences (plain punctuation split)."""
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()][:limit]


def _regions_in(text: str, regions: list[str]) -> list[str]:
    """Region names from ``regions`` that appear in ``text`` (longest first, so e.g.
    'United States of America' wins over a bare substring)."""
    low = (text or "").lower()
    return [r for r in sorted(regions, key=len, reverse=True) if r and r.lower() in low]


def build_beats(route_kind: str, answer_text: str, *, regions: list[str] | None = None) -> list[Beat]:
    """Turn an answer into an ordered beat list — deterministic, no LLM, no number touched.

    ``route_kind`` selects the section walk (see :data:`ANCHOR_TEMPLATES`); each sentence is
    pinned to the next anchor, and a globe beat is given the pose of the region named in that
    sentence (falling back to the regions named anywhere in the answer). ``regions`` is the
    candidate set to scan (e.g. the kept drivers' regions) — empty ⇒ no globe poses."""
    sentences = _sentences(answer_text)
    if not sentences:
        return []
    template = ANCHOR_TEMPLATES.get(route_kind, _DEFAULT_TEMPLATE)
    candidates = list(regions or [])
    answer_regions = _regions_in(answer_text, candidates)
    cursor = 0  # walks answer_regions so successive globe beats visit different countries

    beats: list[Beat] = []
    for index, sentence in enumerate(sentences):
        anchor = template[min(index, len(template) - 1)]
        pose = None
        if anchor in GLOBE_ANCHORS and candidates:
            here = _regions_in(sentence, candidates)
            region = here[0] if here else (answer_regions[cursor % len(answer_regions)]
                                           if answer_regions else None)
            if region:
                pose = pose_for(region)
                if not here and answer_regions:
                    cursor += 1
        beats.append(Beat(say=sentence, anchor=anchor, pose=pose))
    return beats


def synth_beats(beats: list[Beat]) -> list[Beat]:
    """Fill each beat's ``audio_b64`` via the local voice (instant, offline, no rate limit).
    A beat whose synthesis fails keeps ``audio_b64=None`` — the JS player then advances it on
    an estimated-read timer instead of an audio ``ended`` event, so the tour never stalls."""
    out: list[Beat] = []
    for beat in beats:
        clip = voice.synthesize(beat.say, providers=["local"])
        if clip is not None:
            out.append(replace(
                beat, audio_b64=base64.b64encode(clip.audio_bytes).decode("ascii"), mime=clip.mime))
        else:
            out.append(beat)
    return out


def route_kind_for(prompt: str, *, is_shock: bool, is_about: bool) -> str:
    """Classify a chat prompt into a tour walk. Reuses the caller's already-computed shock /
    about flags (so the LLM still classifies nothing new here); the rest is keyword-simple."""
    if is_shock:
        return "shock"
    if is_about:
        return "about"
    low = (prompt or "").lower()
    if any(w in low for w in ("ceramic", "supplier", "channel", "lock", "margin")):
        return "ceramics" if "ceramic" in low else "country"
    if any(w in low for w in ("which", "country", "where", "driver", "region")):
        return "country"
    return "why"
