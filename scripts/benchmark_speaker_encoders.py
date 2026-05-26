"""Benchmark Resemblyzer vs ECAPA-TDNN for speaker recognition on short utterances.

Goal: validate whether switching encoder is worth doing (per the secret-scopes /
speaker-rec plan). Inform encoder choice and threshold tuning before committing
to a refactor of jarvis-whisper-api/app/utils.py.

## Recording protocol

On the dev Pi (or any mic-equipped machine), record yourself + one other
household member saying each of these phrases 5-10 times:

  - "do I have any emails"        (~1.5-2.0s — typical short command)
  - "what's on my calendar today"  (~2.0-2.5s — typical medium command)
  - "delete it"                    (~0.5-0.8s — short follow-up)

Save WAVs to a directory with naming:

  {speaker}_{phrase_slug}_{take}.wav

  alex_emails_1.wav
  alex_emails_2.wav
  ...
  sara_calendar_1.wav
  ...
  alex_delete_it_1.wav
  ...

Aim for 60+ clips total (2 speakers × 3 phrases × 10 takes). The more takes per
phrase per speaker, the better the variance estimate.

## Running

  python scripts/benchmark_speaker_encoders.py \\
      --audio-dir /path/to/recordings \\
      --enrollment-count 3

## Output

Per-encoder, for each length bucket (short / medium / long), reports:
  - Accuracy (top-1 prediction)
  - Mean separation margin (best score - second-best score)
  - Equal Error Rate (EER) sweep
  - Recommended threshold

Decision rule from the plan: if ECAPA shows >=15% absolute accuracy improvement
on 1-2s clips, proceed with Phase 2 of the refactor. If marginal, drop Phase 2
and adopt strict no-fallback semantics for user-scope commands.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import wave
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass
class Clip:
    path: Path
    speaker: str
    phrase: str
    take: int
    duration_s: float

    @property
    def length_bucket(self) -> str:
        if self.duration_s < 1.0:
            return "short"
        if self.duration_s <= 3.0:
            return "medium"
        return "long"


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
        return frames / float(rate) if rate else 0.0


_NAME_RE = re.compile(r"^([a-zA-Z]+)_([a-zA-Z_]+)_(\d+)\.wav$")


def discover_clips(audio_dir: Path) -> list[Clip]:
    clips: list[Clip] = []
    for path in sorted(audio_dir.glob("*.wav")):
        m = _NAME_RE.match(path.name)
        if not m:
            print(f"  skip (unparseable name): {path.name}", file=sys.stderr)
            continue
        speaker, phrase, take = m.group(1), m.group(2).rstrip("_"), int(m.group(3))
        try:
            duration = _wav_duration(path)
        except Exception as e:
            print(f"  skip (unreadable wav): {path.name} — {e}", file=sys.stderr)
            continue
        clips.append(Clip(path=path, speaker=speaker, phrase=phrase, take=take, duration_s=duration))
    return clips


# ---- Encoder facades ------------------------------------------------------


@dataclass
class EncoderResult:
    name: str
    embed: Callable[[Path], np.ndarray]


def _load_resemblyzer() -> EncoderResult | None:
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except ImportError as e:
        print(f"resemblyzer unavailable: {e}", file=sys.stderr)
        return None
    encoder = VoiceEncoder(device="cpu", verbose=False)

    def embed(path: Path) -> np.ndarray:
        wav = preprocess_wav(path)
        return encoder.embed_utterance(wav)

    return EncoderResult(name="resemblyzer", embed=embed)


def _load_ecapa() -> EncoderResult | None:
    try:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError as e:
        print(f"speechbrain unavailable (pip install speechbrain torch): {e}", file=sys.stderr)
        return None
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"},
        savedir="/tmp/ecapa-spkrec",
    )

    def embed(path: Path) -> np.ndarray:
        signal, sr = _load_wav_mono_16k(path)
        emb = classifier.encode_batch(torch.tensor(signal).unsqueeze(0))
        vec = emb.squeeze().detach().cpu().numpy()
        # L2-normalize so inner-product == cosine similarity
        return vec / (np.linalg.norm(vec) + 1e-9)

    return EncoderResult(name="ecapa-tdnn", embed=embed)


def _load_wav_mono_16k(path: Path) -> tuple[np.ndarray, int]:
    import scipy.signal as sps

    with wave.open(str(path), "rb") as wav:
        sr = wav.getframerate()
        n_channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())

    if sampwidth == 2:
        dtype = np.int16
        scale = 32768.0
    elif sampwidth == 4:
        dtype = np.int32
        scale = 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    audio = np.frombuffer(frames, dtype=dtype).astype(np.float32) / scale
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    if sr != 16000:
        # polyphase resample
        from math import gcd
        g = gcd(sr, 16000)
        audio = sps.resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
        sr = 16000

    return audio, sr


# ---- Evaluation -----------------------------------------------------------


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (na * nb))


@dataclass
class BucketStats:
    bucket: str
    n: int = 0
    correct: int = 0
    margins: list[float] = field(default_factory=list)
    same_speaker_scores: list[float] = field(default_factory=list)
    cross_speaker_scores: list[float] = field(default_factory=list)


def _eer(positive_scores: list[float], negative_scores: list[float]) -> tuple[float, float]:
    """Sweep threshold to find Equal Error Rate. Returns (eer, threshold_at_eer)."""
    if not positive_scores or not negative_scores:
        return float("nan"), float("nan")
    thresholds = sorted(set(positive_scores + negative_scores))
    best_eer = 1.0
    best_t = thresholds[0]
    for t in thresholds:
        far = sum(1 for s in negative_scores if s >= t) / len(negative_scores)
        frr = sum(1 for s in positive_scores if s < t) / len(positive_scores)
        if abs(far - frr) < abs(best_eer * 2 - 1):  # rough — closest crossover
            best_eer = (far + frr) / 2
            best_t = t
    return best_eer, best_t


def evaluate(
    encoder: EncoderResult,
    clips: list[Clip],
    enrollment_count: int,
    seed: int,
) -> dict[str, BucketStats]:
    rng = random.Random(seed)
    by_speaker: dict[str, list[Clip]] = defaultdict(list)
    for c in clips:
        by_speaker[c.speaker].append(c)

    # Build enrollment reference per speaker (average of N embeddings)
    references: dict[str, np.ndarray] = {}
    test_clips: list[Clip] = []
    for speaker, sclips in by_speaker.items():
        if len(sclips) <= enrollment_count:
            print(f"  warn: speaker '{speaker}' has only {len(sclips)} clips — using all for enrollment, no test set", file=sys.stderr)
            continue
        rng.shuffle(sclips)
        enroll = sclips[:enrollment_count]
        test_clips.extend(sclips[enrollment_count:])
        embs = [encoder.embed(c.path) for c in enroll]
        ref = np.mean(embs, axis=0)
        references[speaker] = ref / (np.linalg.norm(ref) + 1e-9)

    buckets: dict[str, BucketStats] = {b: BucketStats(b) for b in ("short", "medium", "long")}

    for clip in test_clips:
        emb = encoder.embed(clip.path)
        scores = {sp: _cosine(emb, ref) for sp, ref in references.items()}
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        predicted = ranked[0][0]
        best_score = ranked[0][1]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = best_score - second_score

        b = buckets[clip.length_bucket]
        b.n += 1
        if predicted == clip.speaker:
            b.correct += 1
        b.margins.append(margin)
        for sp, sc in scores.items():
            (b.same_speaker_scores if sp == clip.speaker else b.cross_speaker_scores).append(sc)

    return buckets


# ---- Reporting ------------------------------------------------------------


def report_text(label: str, buckets: dict[str, BucketStats]) -> str:
    lines = [f"\n=== {label} ===", f"{'bucket':<8} {'n':>4} {'accuracy':>10} {'mean_margin':>12} {'EER':>8} {'threshold@EER':>14}"]
    overall_pos: list[float] = []
    overall_neg: list[float] = []
    overall_n = 0
    overall_correct = 0
    for name in ("short", "medium", "long"):
        b = buckets[name]
        if b.n == 0:
            lines.append(f"{name:<8} {'-':>4} {'-':>10} {'-':>12} {'-':>8} {'-':>14}")
            continue
        acc = b.correct / b.n
        margin = np.mean(b.margins) if b.margins else float("nan")
        eer, t_eer = _eer(b.same_speaker_scores, b.cross_speaker_scores)
        lines.append(f"{name:<8} {b.n:>4} {acc:>10.1%} {margin:>12.3f} {eer:>8.1%} {t_eer:>14.3f}")
        overall_pos.extend(b.same_speaker_scores)
        overall_neg.extend(b.cross_speaker_scores)
        overall_n += b.n
        overall_correct += b.correct
    if overall_n:
        eer, t_eer = _eer(overall_pos, overall_neg)
        acc = overall_correct / overall_n
        lines.append(f"{'overall':<8} {overall_n:>4} {acc:>10.1%} {'':>12} {eer:>8.1%} {t_eer:>14.3f}")
    return "\n".join(lines)


def report_json(buckets_by_encoder: dict[str, dict[str, BucketStats]]) -> str:
    out: dict = {}
    for enc_name, buckets in buckets_by_encoder.items():
        out[enc_name] = {}
        for b_name, b in buckets.items():
            eer, t_eer = _eer(b.same_speaker_scores, b.cross_speaker_scores)
            out[enc_name][b_name] = {
                "n": b.n,
                "accuracy": (b.correct / b.n) if b.n else None,
                "mean_margin": (float(np.mean(b.margins)) if b.margins else None),
                "eer": eer,
                "threshold_at_eer": t_eer,
            }
    return json.dumps(out, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio-dir", required=True, type=Path, help="Directory of {speaker}_{phrase}_{take}.wav files")
    parser.add_argument("--enrollment-count", type=int, default=3, help="Clips per speaker used for enrollment (default 3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for enrollment split")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable table")
    parser.add_argument("--skip-resemblyzer", action="store_true")
    parser.add_argument("--skip-ecapa", action="store_true")
    args = parser.parse_args()

    if not args.audio_dir.is_dir():
        print(f"audio dir not found: {args.audio_dir}", file=sys.stderr)
        return 2

    clips = discover_clips(args.audio_dir)
    if not clips:
        print("no clips discovered — check naming convention {speaker}_{phrase}_{take}.wav", file=sys.stderr)
        return 2

    speakers = sorted({c.speaker for c in clips})
    print(f"discovered {len(clips)} clips across {len(speakers)} speakers: {speakers}")
    bucket_counts = defaultdict(int)
    for c in clips:
        bucket_counts[c.length_bucket] += 1
    print(f"length distribution: {dict(bucket_counts)}")

    results: dict[str, dict[str, BucketStats]] = {}
    if not args.skip_resemblyzer:
        print("loading resemblyzer encoder...")
        enc = _load_resemblyzer()
        if enc is not None:
            print("evaluating resemblyzer...")
            results[enc.name] = evaluate(enc, clips, args.enrollment_count, args.seed)
    if not args.skip_ecapa:
        print("loading ecapa-tdnn encoder...")
        enc = _load_ecapa()
        if enc is not None:
            print("evaluating ecapa...")
            results[enc.name] = evaluate(enc, clips, args.enrollment_count, args.seed)

    if not results:
        print("no encoders loaded successfully", file=sys.stderr)
        return 2

    if args.json:
        print(report_json(results))
    else:
        for enc_name, buckets in results.items():
            print(report_text(enc_name, buckets))
        # Decision hint
        if "resemblyzer" in results and "ecapa-tdnn" in results:
            print("\n--- decision hint (per plan) ---")
            for bucket in ("short", "medium"):
                r_b = results["resemblyzer"].get(bucket)
                e_b = results["ecapa-tdnn"].get(bucket)
                if r_b and e_b and r_b.n > 0 and e_b.n > 0:
                    r_acc = r_b.correct / r_b.n
                    e_acc = e_b.correct / e_b.n
                    delta = e_acc - r_acc
                    verdict = "PROCEED with Phase 2" if delta >= 0.15 else "marginal — consider dropping Phase 2"
                    print(f"  {bucket}: resemblyzer {r_acc:.1%} → ecapa {e_acc:.1%}  (Δ={delta:+.1%})  ⇒ {verdict}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
