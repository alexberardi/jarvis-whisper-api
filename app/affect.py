"""Acoustic affect (mood) analysis — a prosodic read of *how* something was said.

This does NOT try to classify a discrete emotion (happy/sad/angry) — that path is
brittle and misfires constantly. Instead it produces an **acoustic-texture
descriptor**: how energized/animated the voice is (arousal), plus a short
plain-language read ("subdued, flat pitch, halting"). The downstream LLM fuses
that texture with *what* was said — "sounds tired" + "how do I get the baby to
sleep" → be gentle. So the bar here is "describe the voice decently + report
honest confidence," not "classify emotion correctly."

Design notes:
- **Arousal is the reliable axis from prosody** (energy dynamics, pitch
  variability, pace, brightness). Valence (positive vs. negative) is weak from
  audio alone — that's the part the *content* carries, so we don't guess it here.
- **Gain-robust features only.** The node applies AGC/gain, so ABSOLUTE loudness
  is unreliable. We lean on pitch *variability* (semitones, speaker-relative),
  energy *dynamics* (coefficient of variation), pause ratio, and spectral
  centroid — all robust to a flat gain change.
- **Fails safe + never raises.** Any error, or too little voiced speech, yields
  ``arousal="unknown"`` / ``confidence=0`` so the caller simply surfaces nothing.
- **Thresholds below are a first pass to be TUNED on real clips** (the whole
  point of shipping this behind an opt-in flag first): the caller logs every
  read + raw features so the bands can be calibrated against actual telemetry.

Zero new heavy deps: uses librosa/numpy/scipy, already installed for the stack.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np

logger = logging.getLogger(__name__)

SR = 16000

# Below this, there isn't enough speech for an honest prosodic read.
_MIN_DURATION_S = 0.5
# Need at least this fraction voiced (and this many voiced frames) to trust pitch.
_MIN_VOICED_FRAC = 0.15
_MIN_VOICED_FRAMES = 5

_FRAME = 1024
_HOP = 256

# Speech-presence gate. Spectral flatness (Wiener entropy, 0..1) is near 1 for
# broadband noise and low for voiced speech (formant/harmonic peaks). Above this,
# the clip is treated as non-speech (noise / VAD false-trigger) and yields
# ``unknown`` — the energy gates alone can't tell noise from speech, and a
# confident read on noise is exactly the false positive this must never surface.
_MAX_FLATNESS = 0.35

# Cap analysis length — commands are short; this bounds worst-case latency on a
# stray long clip so the affect pass can't outrun the speaker embed it overlaps.
_MAX_SAMPLES = SR * 12

# Arousal score → label thresholds. Recalibrated on real telemetry (2026-07-23):
# real commands topped out around ~0.5, so the "high" bar was lowered from 0.62.
_AROUSAL_LOW = 0.35
_AROUSAL_HIGH = 0.55

# Per-speaker pitch baseline (in-process EMA of each speaker's median pitch, Hz).
# Absolute pitch is speaker-dependent, so "energized" can only be judged RELATIVE
# to a speaker's OWN resting pitch — a rise above their baseline is the strongest
# arousal cue (and the one a flat pitch-variability measure misses). Learned over
# a few utterances; resets on restart and rebuilds fast. Keyed by speaker user_id
# from the recognition pass; unknown speaker → no rise cue (base formula only).
_BASELINE_ALPHA = 0.25
_pitch_baselines: dict[int, float] = {}
_baseline_lock = threading.Lock()


def _get_pitch_baseline(user_id: int | None) -> float | None:
    if user_id is None:
        return None
    with _baseline_lock:
        return _pitch_baselines.get(user_id)


def _update_pitch_baseline(user_id: int | None, pitch_median: float) -> None:
    """EMA-update the speaker's resting-pitch baseline. Called AFTER the rise cue
    is computed from the prior baseline, so an utterance never inflates its own."""
    if user_id is None or not pitch_median or pitch_median <= 0:
        return
    with _baseline_lock:
        prev = _pitch_baselines.get(user_id)
        _pitch_baselines[user_id] = (
            pitch_median if prev is None
            else (1.0 - _BASELINE_ALPHA) * prev + _BASELINE_ALPHA * pitch_median
        )


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


@dataclass
class AffectResult:
    """A prosodic read of an utterance.

    ``arousal`` is one of ``low`` | ``neutral`` | ``high`` | ``unknown``.
    ``unknown`` means "not analyzable" (too short, too little voiced speech, or
    an error) — the caller should surface nothing. ``read`` is a short human
    descriptor for the prompt; ``confidence`` gates whether it's worth using.
    ``features`` carries the raw measurements for logging + threshold tuning.
    """

    read: str
    arousal: str
    confidence: float
    features: dict = field(default_factory=dict)

    def to_response(self, min_confidence: float = 0.0) -> dict | None:
        """Wire shape for the STT response, or ``None`` when not worth surfacing.

        Suppresses unanalyzable reads (``unknown``) and anything below
        ``min_confidence`` — a false read that stays silent costs nothing, a
        shaky one spoken aloud breaks trust, so we bias toward silence.
        """
        if self.arousal == "unknown" or not self.read:
            return None
        if self.confidence < min_confidence:
            return None
        return {
            "read": self.read,
            "arousal": self.arousal,
            "confidence": self.confidence,
        }


def _unknown(features: dict) -> AffectResult:
    return AffectResult(read="", arousal="unknown", confidence=0.0, features=features)


def _describe(arousal: str, pitch_var: float, pause_ratio: float,
              centroid: float, energy_cv: float,
              pitch_rise: float | None = None) -> str:
    """Compose a short human descriptor from the dominant prosodic cues."""
    cues: list[str] = []
    # Speaker-relative pitch rise leads when it's the salient signal.
    if pitch_rise is not None and pitch_rise > 0.2:
        cues.append("pitched up / energized")
    elif pitch_rise is not None and pitch_rise < -0.12:
        cues.append("pitched down / subdued")
    if pitch_var < 1.3:
        cues.append("flat, even pitch")
    elif pitch_var > 3.0:
        cues.append("animated pitch")
    if pause_ratio > 0.35:
        cues.append("halting, lots of pauses")
    elif pause_ratio < 0.12:
        cues.append("brisk, few pauses")
    if centroid < 1300:
        cues.append("dark, muted tone")
    elif centroid > 2400:
        cues.append("bright, tense tone")
    if energy_cv > 1.2:
        cues.append("emphatic delivery")
    elif energy_cv < 0.5:
        cues.append("low, steady energy")
    if not cues:
        cues.append("even and steady")
    head = {
        "low": "subdued / low-energy",
        "high": "animated / energized",
        "neutral": "even",
    }.get(arousal, "even")
    return f"{head} — " + ", ".join(cues[:3])


def analyze_affect_from_signal(
    y: np.ndarray, sr: int = SR, baseline_pitch: float | None = None
) -> AffectResult:
    """Pure prosodic analysis of a mono waveform. The testable DSP core.

    ``baseline_pitch`` (Hz), when given, is the speaker's learned resting pitch;
    a rise above it feeds a speaker-relative arousal cue. Never raises: on any
    internal failure it returns an ``unknown`` result.
    """
    features: dict = {}
    try:
        y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1).astype(np.float32)
        if len(y) > _MAX_SAMPLES:
            y = y[:_MAX_SAMPLES]

        dur = len(y) / float(sr) if sr else 0.0
        features["duration_s"] = round(dur, 2)
        if dur < _MIN_DURATION_S or not np.any(np.isfinite(y)):
            return _unknown(features)

        # RMS energy per frame (+eps so silent frames don't divide-by-zero).
        rms = librosa.feature.rms(y=y, frame_length=_FRAME, hop_length=_HOP)[0]
        rms = np.asarray(rms, dtype=np.float64) + 1e-9
        peak = float(rms.max())
        if peak <= 1e-8:  # essentially silent
            return _unknown(features)

        # Pitch via YIN (fast; no probabilistic voicing), voicing-gated by energy.
        f0 = librosa.yin(y, fmin=70, fmax=400, sr=sr,
                         frame_length=_FRAME, hop_length=_HOP)
        m = min(len(f0), len(rms))
        f0 = np.asarray(f0[:m], dtype=np.float64)
        rms_a = rms[:m]

        # A frame is "voiced" when it carries real energy (>10% of the clip peak).
        voiced_mask = rms_a > (peak * 0.10)
        voiced_frac = float(voiced_mask.mean()) if m else 0.0
        voiced_f0 = f0[voiced_mask & np.isfinite(f0) & (f0 > 0)]

        # Pause ratio: fraction of frames that are near-silent within the clip.
        pause_ratio = float((rms_a < peak * 0.10).mean()) if m else 1.0

        features["voiced_frac"] = round(voiced_frac, 3)
        features["pause_ratio"] = round(pause_ratio, 3)

        if voiced_frac < _MIN_VOICED_FRAC or len(voiced_f0) < _MIN_VOICED_FRAMES:
            return _unknown(features)

        # SPEECH-PRESENCE gate. Everything above tests only the energy
        # DISTRIBUTION, not that the input is speech — so broadband noise (a VAD
        # false-trigger, room hiss; the node's AGC makes its absolute level
        # meaningless) sails through with voiced_frac≈1 and would yield a
        # confident, WRONG "energized" read. Spectral flatness separates them:
        # ~1 for noise, low for voiced speech. Reject noise-like clips outright.
        flat = librosa.feature.spectral_flatness(y=y, hop_length=_HOP)[0]
        flat = np.asarray(flat, dtype=np.float64)
        vmask = voiced_mask[:len(flat)]
        flat_v = flat[:len(vmask)][vmask]
        flatness = float(np.mean(flat_v)) if len(flat_v) else 1.0
        features["flatness"] = round(flatness, 3)
        if flatness > _MAX_FLATNESS:
            return _unknown(features)

        # Pitch variability in semitones around the speaker's own median — this
        # is animated-vs-flat, and being relative it's robust across speakers
        # (a naturally high-pitched voice isn't read as "aroused").
        median_f0 = float(np.median(voiced_f0))
        semitones = 12.0 * np.log2(voiced_f0 / median_f0)
        pitch_var = float(np.std(semitones))

        # Energy dynamics as a coefficient of variation — gain-invariant (a flat
        # gain change scales mean and std together, leaving the ratio fixed).
        energy_cv = float(np.std(rms_a) / (np.mean(rms_a) + 1e-9))

        # Spectral centroid over VOICED frames only — librosa reads silence/pause
        # frames as 0 Hz, and averaging those in would drag the mean down and flip
        # the dark/bright tone cue (rms & f0 are already voiced-masked, so the
        # centroid must be too, for consistency).
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=_HOP)[0]
        cent = np.asarray(cent, dtype=np.float64)
        cmask = voiced_mask[:len(cent)]
        cent_v = cent[:len(cmask)][cmask]
        centroid = float(np.mean(cent_v)) if len(cent_v) else float(np.mean(cent))

        features.update({
            "pitch_var_st": round(pitch_var, 2),
            "pitch_median_hz": round(median_f0, 1),
            "energy_cv": round(energy_cv, 3),
            "centroid_hz": round(centroid, 1),
        })

        # Arousal from gain-robust cues, each squashed to ~[0,1] over a working
        # band, then a WEIGHTED mean. Bands recalibrated on real telemetry
        # (2026-07-23): real-speech centroids sit ~850-1150 Hz (the old 1200-2800
        # band was always ~0), and short commands run ~0.45-0.70 pause ratio (the
        # old 0.5 pivot pinned pace to ~0).
        s_pitchvar = _clip01((pitch_var - 1.0) / 3.0)       # ~1..4 semitone spread
        s_pace = _clip01((0.65 - pause_ratio) / 0.40)       # real commands pause a lot
        s_centroid = _clip01((centroid - 850.0) / 700.0)    # ~850..1550 Hz (real speech)
        s_dynamics = _clip01((energy_cv - 0.5) / 1.0)       # emphatic delivery
        weighted: list[tuple[float, float]] = [
            (s_pitchvar, 1.0), (s_pace, 1.0), (s_centroid, 0.7), (s_dynamics, 1.0),
        ]

        # Speaker-RELATIVE pitch rise — the strongest energy cue, and the one the
        # semitone-spread measure above misses (a raised-but-steady voice reads as
        # low pitch_var). Only usable with a learned baseline for THIS speaker,
        # since absolute pitch is speaker-specific: a jump above their resting
        # pitch reads as excitement. Weighted heavily when available.
        pitch_rise: float | None = None
        if baseline_pitch and baseline_pitch > 0:
            pitch_rise = (median_f0 - baseline_pitch) / baseline_pitch
            s_pitchrise = _clip01(pitch_rise / 0.35)        # +35% over baseline → full
            weighted.append((s_pitchrise, 1.8))
            features["pitch_rise"] = round(pitch_rise, 3)

        arousal_score = float(
            sum(w * s for s, w in weighted) / sum(w for _, w in weighted)
        )
        features["arousal_score"] = round(arousal_score, 3)

        if arousal_score < _AROUSAL_LOW:
            arousal = "low"
        elif arousal_score > _AROUSAL_HIGH:
            arousal = "high"
        else:
            arousal = "neutral"

        read = _describe(arousal, pitch_var, pause_ratio, centroid, energy_cv,
                         pitch_rise)

        # Confidence = enough signal AND a clear (non-neutral) read. A short clip
        # or a near-neutral score → low confidence → the caller surfaces nothing.
        signal_q = _clip01(dur / 2.5) * _clip01(voiced_frac / 0.3)
        extremity = _clip01(abs(arousal_score - 0.5) / 0.32)
        confidence = round(float(signal_q * extremity), 2)

        return AffectResult(read=read, arousal=arousal,
                            confidence=confidence, features=features)
    except Exception as e:  # noqa: BLE001 — affect must NEVER break the STT path
        logger.warning("affect analysis failed on signal: %s", e)
        return _unknown(features)


def warmup() -> None:
    """Trigger librosa/numba JIT compilation on a short synthetic tone.

    The first real ``analyze_affect`` call JITs for ~1-2s (numba compiles yin +
    spectral kernels); running this at startup moves that cost off the first
    live command. Never raises.
    """
    try:
        # A ~3s SPEECH-LIKE signal: a pitch-varying fundamental + harmonics +
        # light broadband noise. A pure tone leaves YIN's real-audio branches
        # (trough search / parabolic interpolation over a noisy difference
        # function) uncompiled, so the first live utterance would still pay
        # ~0.8s of JIT. The richer signal exercises those paths here instead.
        rng = np.random.default_rng(0)
        t = np.arange(int(SR * 3.0)) / SR
        freq = 160.0 + 40.0 * np.sin(2 * np.pi * 0.5 * t)
        phase = 2 * np.pi * np.cumsum(freq) / SR
        y = (
            0.30 * np.sin(phase)
            + 0.15 * np.sin(2 * phase)
            + 0.08 * np.sin(3 * phase)
            + 0.02 * rng.standard_normal(len(t))
        ).astype(np.float32)
        analyze_affect_from_signal(y, SR)
    except Exception as e:  # noqa: BLE001
        logger.warning("affect warmup failed: %s", e)


def analyze_affect(path: str, user_id: int | None = None) -> AffectResult:
    """Load a WAV (16 kHz mono) and run :func:`analyze_affect_from_signal`.

    ``user_id`` (the recognized speaker) keys the per-speaker pitch baseline: the
    prior baseline feeds this read's rise cue, then this clip's pitch updates it.
    Never raises — a bad/missing/garbage file yields an ``unknown`` result so the
    STT response is unaffected.
    """
    try:
        # Lazy import: reuse the speaker path's decoder (16 kHz mono float32) so
        # affect sees exactly what the rest of the pipeline does. Imported here,
        # not at module top, to keep app.utils' heavy deps off affect's import.
        from app.utils import _load_wav_mono_16k

        y = _load_wav_mono_16k(Path(path))
        baseline = _get_pitch_baseline(user_id)
        result = analyze_affect_from_signal(y, SR, baseline_pitch=baseline)
        # Learn/update AFTER using the prior baseline, so an utterance never
        # inflates its own rise cue. Only updates on a real voiced pitch.
        _update_pitch_baseline(user_id, result.features.get("pitch_median_hz", 0.0))
        return result
    except Exception as e:  # noqa: BLE001
        logger.warning("affect analysis failed to load %s: %s", path, e)
        return _unknown({})
