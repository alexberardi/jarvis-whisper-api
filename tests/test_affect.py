"""Tests for the acoustic affect (prosodic mood) analysis.

The DSP core is exercised on SYNTHETIC signals with relative assertions (an
animated voice reads more aroused than a flat one) rather than brittle absolute
labels — the exact thresholds are meant to be tuned on real clips. The contract
that must hold regardless of tuning: never raise, degrade to ``unknown`` when
there isn't enough voiced speech, and keep confidence in [0, 1].
"""
import numpy as np
import pytest

from app.affect import (
    AffectResult,
    SR,
    analyze_affect,
    analyze_affect_from_signal,
)


def _sine(freq: float, dur: float = 3.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _sweep(f0: float, f1: float, dur: float = 3.0, amp: float = 0.3,
           tremolo: float = 0.0) -> np.ndarray:
    """A pitch-sweeping tone with optional amplitude tremolo — a crude 'animated'
    voice: changing pitch (high pitch variability) and energy dynamics."""
    t = np.arange(int(dur * SR)) / SR
    freq = np.linspace(f0, f1, len(t))
    phase = 2 * np.pi * np.cumsum(freq) / SR
    sig = np.sin(phase)
    if tremolo:
        sig = sig * (1.0 + tremolo * np.sin(2 * np.pi * 5.0 * t))
    return (amp * sig).astype(np.float32)


class TestAnalyzeAffectFromSignal:
    def test_silence_is_unknown(self):
        r = analyze_affect_from_signal(np.zeros(SR * 2, dtype=np.float32))
        assert r.arousal == "unknown"
        assert r.confidence == 0.0
        assert r.to_response() is None

    def test_too_short_is_unknown(self):
        r = analyze_affect_from_signal(_sine(200, dur=0.3))
        assert r.arousal == "unknown"

    def test_returns_valid_result_shape(self):
        r = analyze_affect_from_signal(_sine(180))
        assert isinstance(r, AffectResult)
        assert r.arousal in {"low", "neutral", "high", "unknown"}
        assert 0.0 <= r.confidence <= 1.0
        assert "duration_s" in r.features

    def test_analyzable_tone_is_not_unknown(self):
        # A sustained voiced tone has enough signal to read.
        r = analyze_affect_from_signal(_sine(200))
        assert r.arousal != "unknown"
        assert "arousal_score" in r.features

    def test_animated_reads_more_aroused_than_flat(self):
        # The load-bearing relative property: a pitch-varying, dynamic voice
        # scores higher arousal than a flat monotone. Robust to threshold tuning.
        flat = analyze_affect_from_signal(_sine(200))
        animated = analyze_affect_from_signal(_sweep(120, 380, tremolo=0.6))
        assert flat.features.get("arousal_score") is not None
        assert animated.features.get("arousal_score") is not None
        assert animated.features["arousal_score"] > flat.features["arousal_score"]

    def test_pitch_variability_tracks_animation(self):
        flat = analyze_affect_from_signal(_sine(200))
        animated = analyze_affect_from_signal(_sweep(120, 380))
        assert animated.features["pitch_var_st"] > flat.features["pitch_var_st"]

    @pytest.mark.parametrize("bad", [
        np.array([], dtype=np.float32),
        np.full(SR, np.nan, dtype=np.float32),
        np.array([np.inf, -np.inf] * SR, dtype=np.float32),
        np.zeros(10, dtype=np.float32),
    ])
    def test_never_raises_on_garbage(self, bad):
        r = analyze_affect_from_signal(bad)
        assert r.arousal in {"low", "neutral", "high", "unknown"}
        assert 0.0 <= r.confidence <= 1.0

    def test_stereo_is_downmixed(self):
        mono = _sine(200)
        stereo = np.stack([mono, mono], axis=1)
        r = analyze_affect_from_signal(stereo)
        assert r.arousal != "unknown"

    def test_confidence_zero_for_unknown(self):
        r = analyze_affect_from_signal(np.zeros(SR, dtype=np.float32))
        assert r.confidence == 0.0

    @pytest.mark.parametrize("amp", [1e-3, 1e-2, 0.1, 0.3])
    def test_broadband_noise_is_not_surfaced(self, amp):
        # THE regression for the confirmed high-sev finding: quiet/loud room noise
        # (a VAD false-trigger) must NOT surface a confident "energized" read.
        # The node's AGC makes absolute level meaningless, so any amplitude of
        # noise must be rejected by the speech-presence (spectral-flatness) gate.
        rng = np.random.default_rng(1)
        noise = (amp * rng.standard_normal(SR * 3)).astype(np.float32)
        r = analyze_affect_from_signal(noise)
        # Either unknown, or at minimum never surfaced at the real default gate.
        assert r.to_response(min_confidence=0.45) is None, (
            f"noise@{amp} surfaced a read: {r.read!r} arousal={r.arousal} conf={r.confidence}"
        )

    def test_real_speech_still_passes_the_gate(self):
        # Guard against the noise gate being so strict it rejects real speech.
        import os
        wav = os.path.join(os.path.dirname(os.path.dirname(__file__)), "jfk_half.wav")
        if not os.path.exists(wav):
            pytest.skip("jfk_half.wav sample not present")
        r = analyze_affect(wav)
        # Real speech must remain analyzable (features computed, not rejected as
        # noise) — the arousal_score is only present when the gates were cleared.
        assert "arousal_score" in r.features, f"real speech rejected: {r.features}"


class TestSpeakerRelativePitch:
    """A rise above the speaker's OWN baseline pitch reads as energy — the cue a
    flat pitch-variability measure misses (raised-but-steady voice)."""

    def test_rise_above_baseline_raises_arousal(self):
        # Same 200 Hz voice reads more aroused when the speaker's resting pitch is
        # low (130 → big rise) than when their baseline IS 200 (no rise).
        y = _sine(200, dur=3.0)
        low_base = analyze_affect_from_signal(y, baseline_pitch=130.0)
        at_base = analyze_affect_from_signal(y, baseline_pitch=200.0)
        assert low_base.features["arousal_score"] > at_base.features["arousal_score"]
        assert "pitch_rise" in low_base.features
        assert low_base.features["pitch_rise"] > 0.3

    def test_no_baseline_means_no_rise_cue(self):
        r = analyze_affect_from_signal(_sine(200, dur=3.0))
        assert "pitch_rise" not in r.features


class TestPitchBaselineStore:
    def _clear(self):
        from app.affect import _pitch_baselines
        _pitch_baselines.clear()

    def test_ema_seeds_then_smooths(self):
        from app.affect import _get_pitch_baseline, _update_pitch_baseline
        self._clear()
        assert _get_pitch_baseline(1) is None
        _update_pitch_baseline(1, 130.0)
        assert _get_pitch_baseline(1) == 130.0  # first observation seeds
        _update_pitch_baseline(1, 200.0)         # EMA: 0.75*130 + 0.25*200
        assert abs(_get_pitch_baseline(1) - 147.5) < 0.01

    def test_none_user_and_bad_pitch_are_noops(self):
        from app.affect import _get_pitch_baseline, _update_pitch_baseline
        self._clear()
        _update_pitch_baseline(None, 130.0)
        assert _get_pitch_baseline(None) is None
        _update_pitch_baseline(2, 0.0)
        assert _get_pitch_baseline(2) is None


class TestToResponse:
    def test_unknown_is_suppressed(self):
        assert AffectResult("", "unknown", 0.0, {}).to_response() is None

    def test_empty_read_is_suppressed(self):
        assert AffectResult("", "low", 0.9, {}).to_response() is None

    def test_below_min_confidence_is_suppressed(self):
        r = AffectResult("subdued — flat", "low", 0.30, {})
        assert r.to_response(min_confidence=0.45) is None

    def test_surfaces_above_threshold(self):
        r = AffectResult("subdued — flat", "low", 0.80, {"x": 1})
        out = r.to_response(min_confidence=0.45)
        # features are NOT surfaced downstream — only the read/arousal/confidence.
        assert out == {"read": "subdued — flat", "arousal": "low", "confidence": 0.80}


class TestAnalyzeAffectPath:
    def test_missing_file_is_unknown(self):
        assert analyze_affect("/nonexistent/definitely/not/here.wav").arousal == "unknown"

    def test_garbage_file_is_unknown(self, tmp_path):
        p = tmp_path / "bad.wav"
        p.write_bytes(b"RIFF" + b"\x00" * 64)
        assert analyze_affect(str(p)).arousal == "unknown"
