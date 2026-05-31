"""Offline tests for the voice-guided tour's deterministic core (W16).

``gas_agent.tour`` turns an already-written (explanation-only) answer into an ordered list of
beats — sentence + scroll anchor + optional globe pose — that the app's JS bridge plays in
sync with the narration. These tests pin that the beat list is **deterministic** (same answer
→ same beats), that it only *re-uses* the answer text (THE RULE — it computes no number), that
globe beats get real poses for the regions the answer names, and that synthesis degrades when
the voice is unavailable. No browser, no network (the voice call is monkeypatched).
"""

from dataclasses import replace

from gas_agent import tour
from gas_agent import voice


# --------------------------------------------------------------------------- #
# Poses — reuse the gas coordinate table; (lat,lon) → {lon,lat,scale}
# --------------------------------------------------------------------------- #
def test_pose_for_known_region_converts_latlon_to_lonlat():
    pose = tour.pose_for("Iran")
    assert pose["lon"] == 53.7 and pose["lat"] == 32.4  # geo has (32.4, 53.7) as (lat, lon)
    assert pose["scale"] == tour._DEFAULT_SCALE


def test_pose_for_uses_scale_overrides_and_local_extra_coords():
    assert tour.pose_for("Russian Federation")["scale"] < tour._DEFAULT_SCALE  # broad story, zoom out
    assert tour.pose_for("Netherlands")["scale"] > tour._DEFAULT_SCALE  # small country, zoom in
    assert tour.pose_for("Switzerland") is not None  # the one ceramics market gas geo lacks
    assert tour.pose_for("Atlantis") is None  # unplaceable → no pose (beat just scrolls + speaks)


# --------------------------------------------------------------------------- #
# Beat decomposition — deterministic, template-driven, answer-text-only
# --------------------------------------------------------------------------- #
_ANSWER = ("The agent locks 30% of next quarter forward. The confidence band is moderate. "
           "Norway is the strongest credible driver, with Russia close behind.")


def test_tour_starts_at_the_chat_then_walks_the_template():
    # Every tour begins at the chat (so the user sees the answer), then scrolls.
    beats = tour.build_beats("why", _ANSWER, regions=["Norway", "Russia"])
    assert [b.anchor for b in beats] == [tour.CHAT, tour.WHY, tour.HEDGE]
    assert beats[0].pose is None  # the chat beat never moves the globe


def test_country_walk_rotates_the_globe_after_the_chat_intro():
    answer = "Here is what I found. Norway supplies the most gas. Russia is next."
    beats = tour.build_beats("country", answer, regions=["Norway", "Russia"])
    assert beats[0].anchor == tour.CHAT and beats[0].pose is None  # intro, no scroll
    # The remaining beats are globe beats, each rotating to the country IT names.
    assert beats[1].anchor == tour.DRIVERS_GLOBE and beats[1].pose == tour.pose_for("Norway")
    assert beats[2].anchor == tour.DRIVERS_GLOBE and beats[2].pose == tour.pose_for("Russia")


def test_beats_are_capped_and_deterministic():
    long_answer = " ".join(f"Sentence number {i} ends here." for i in range(20))
    a = tour.build_beats("why", long_answer, regions=[])
    b = tour.build_beats("why", long_answer, regions=[])
    assert len(a) == tour._MAX_BEATS  # capped
    assert [x.to_dict() for x in a] == [x.to_dict() for x in b]  # identical run-to-run


def test_the_rule_beats_only_reuse_the_answer_text():
    # Every beat sentence must be a verbatim slice of the answer — no invented numbers/claims.
    beats = tour.build_beats("shock", _ANSWER, regions=["Norway", "Russia"])
    for beat in beats:
        assert beat.say in _ANSWER


def test_empty_answer_yields_no_beats():
    assert tour.build_beats("why", "", regions=["Norway"]) == []
    assert tour.build_beats("why", "   ", regions=[]) == []


def test_no_regions_means_no_globe_poses():
    beats = tour.build_beats("country", _ANSWER, regions=[])
    assert all(b.pose is None for b in beats)


# --------------------------------------------------------------------------- #
# route_kind_for — reuses the caller's shock/about flags, keyword rest
# --------------------------------------------------------------------------- #
def test_route_kind_classification():
    assert tour.route_kind_for("Iran closes Hormuz", is_shock=True, is_about=False) == "shock"
    assert tour.route_kind_for("what is this app?", is_shock=False, is_about=True) == "about"
    assert tour.route_kind_for("what about ceramics?", is_shock=False, is_about=False) == "ceramics"
    # Supplier / lock / margin are CERAMICS-decision concepts → the ceramics section
    # (the bug fix: these used to mis-route to the gas drivers globe).
    assert tour.route_kind_for("which supplier matters?", is_shock=False, is_about=False) == "ceramics"
    assert tour.route_kind_for("why this lock percentage?", is_shock=False, is_about=False) == "ceramics"
    assert tour.route_kind_for("what's the margin?", is_shock=False, is_about=False) == "ceramics"
    # Gas-driver / geography questions → the gas drivers globe.
    assert tour.route_kind_for("which country drives gas?", is_shock=False, is_about=False) == "country"
    assert tour.route_kind_for("why this ratio?", is_shock=False, is_about=False) == "why"


# --------------------------------------------------------------------------- #
# Synthesis — local voice fills audio; failure degrades to a timer-advance beat
# --------------------------------------------------------------------------- #
def test_synth_fills_audio_and_forces_the_local_voice(monkeypatch):
    captured = {}

    def fake_synth(text, *, providers=None):
        captured["providers"] = providers
        return voice.VoiceClip(audio_bytes=b"RIFFfake", mime="audio/wav", provider="local", voice="sys")

    monkeypatch.setattr(voice, "synthesize", fake_synth)
    beats = tour.synth_beats([tour.Beat(say="Hello.", anchor=tour.WHY)])
    assert captured["providers"] == ["local"]  # never burns the NVIDIA budget on a tour
    assert beats[0].audio_b64 and beats[0].mime == "audio/wav"


def test_synth_degrades_when_voice_unavailable(monkeypatch):
    monkeypatch.setattr(voice, "synthesize", lambda *a, **k: None)
    beats = tour.synth_beats([tour.Beat(say="Hello.", anchor=tour.WHY)])
    assert beats[0].audio_b64 is None  # JS will timer-advance this beat


def test_anchor_html_carries_the_slug():
    html = tour.anchor_html("drivers_globe")
    assert "data-tour-anchor='drivers_globe'" in html and "height:0" in html
