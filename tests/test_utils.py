"""Tests for utils module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.exceptions import WhisperTranscriptionError
from app.utils import (
    SpeakerResult,
    _load_for_whisper,
    hash_user_id,
    invalidate_household_cache,
    load_household_profiles,
    recognize_speaker,
    run_whisper,
)


def _make_segment(text: str, t0: int = 0, t1: int = 100) -> MagicMock:
    """Build a fake pywhispercpp Segment with .text, .t0, .t1 attributes.

    t0/t1 mimic whisper.cpp's centisecond units.
    """
    seg = MagicMock()
    seg.text = text
    seg.t0 = t0
    seg.t1 = t1
    return seg


class TestSpeakerResult:
    """Test SpeakerResult dataclass."""

    def test_speaker_result_creation(self) -> None:
        """SpeakerResult should store user_id and confidence."""
        result = SpeakerResult(user_id=42, confidence=0.92)
        assert result.user_id == 42
        assert result.confidence == 0.92

    def test_speaker_result_unknown(self) -> None:
        """SpeakerResult should support unknown speaker with None user_id."""
        result = SpeakerResult(user_id=None, confidence=0.0)
        assert result.user_id is None
        assert result.confidence == 0.0

    def test_speaker_result_equality(self) -> None:
        """SpeakerResult should support equality comparison."""
        result1 = SpeakerResult(user_id=123, confidence=0.85)
        result2 = SpeakerResult(user_id=123, confidence=0.85)
        assert result1 == result2


class TestHashUserId:
    """Test hash_user_id function."""

    def test_hash_user_id_deterministic(self) -> None:
        """hash_user_id should return same hash for same input."""
        hash1 = hash_user_id(42)
        hash2 = hash_user_id(42)
        assert hash1 == hash2

    def test_hash_user_id_different_ids_different_hashes(self) -> None:
        """hash_user_id should return different hashes for different inputs."""
        hash1 = hash_user_id(1)
        hash2 = hash_user_id(2)
        assert hash1 != hash2

    def test_hash_user_id_length(self) -> None:
        """hash_user_id should return 16-character string."""
        result = hash_user_id(12345)
        assert len(result) == 16

    def test_hash_user_id_is_hex(self) -> None:
        """hash_user_id should return valid hex string."""
        result = hash_user_id(999)
        # Should not raise ValueError
        int(result, 16)


class TestLoadHouseholdProfiles:
    """Test load_household_profiles function."""

    def teardown_method(self) -> None:
        """Clear cache after each test."""
        invalidate_household_cache()

    @patch("app.utils.PROFILE_DIR", Path("/nonexistent"))
    def test_load_household_profiles_missing_directory(self) -> None:
        """load_household_profiles should return empty dict for missing directory."""
        result = load_household_profiles("missing-household", [1, 2, 3])
        assert result == {}

    def test_load_household_profiles_caches_per_member(self) -> None:
        """Embeddings cache per (household, user); a repeat call re-globs nothing."""
        from app import utils

        household_id = "cache-test-household"
        with patch.object(
            utils, "user_profile_paths", return_value=[]
        ) as mock_paths:
            result1 = load_household_profiles(household_id, [1])
            result2 = load_household_profiles(household_id, [1])

        # No profiles on disk → empty result both times.
        assert result1 == {}
        assert result2 == {}
        # The second call is served from the per-member (negative) cache, so the
        # filesystem is consulted exactly once for user 1.
        assert mock_paths.call_count == 1
        assert utils._member_embedding_cache.get((household_id, 1)) is None

    def test_load_household_profiles_empty_member_list(self) -> None:
        """An empty member list returns {} and caches nothing (no poison)."""
        from app import utils

        result = load_household_profiles("empty-member-household", [])
        assert result == {}
        # Critically, a memberless request must not write any cache entry for
        # the household — otherwise it would poison a later, fuller scope.
        assert not any(
            k[0] == "empty-member-household" for k in utils._member_embedding_cache
        )

    def test_prep_fingerprint_change_invalidates_cache(self, monkeypatch) -> None:
        """A change in embed-preprocess settings should drop cached centroids."""
        from app import utils

        utils.invalidate_household_cache()
        utils._cache_prep_fingerprint = None
        # Seed a fake cached member embedding and pin its fingerprint.
        utils._member_embedding_cache[("hh", 1)] = np.array([1.0])
        monkeypatch.setattr(utils, "_embed_prep_settings", lambda: (True, -23.0, -40.0))
        utils._cache_prep_fingerprint = utils._embed_prep_fingerprint()

        # Change the settings → new fingerprint → the cache must be invalidated
        # so the stale centroid is re-embedded under the new regime.
        monkeypatch.setattr(utils, "_embed_prep_settings", lambda: (True, -20.0, -40.0))
        with patch.object(utils, "PROFILE_DIR", Path("/nonexistent")):
            result = utils.load_household_profiles("hh", [1])

        assert result == {}
        # The stale centroid was dropped; the re-load found nothing on disk and
        # recorded a negative-cache entry instead.
        assert utils._member_embedding_cache.get(("hh", 1)) is None
        utils._cache_prep_fingerprint = None


class TestInvalidateHouseholdCache:
    """Test invalidate_household_cache function."""

    def teardown_method(self) -> None:
        """Clear cache after each test."""
        invalidate_household_cache()

    def test_invalidate_household_cache_specific(self) -> None:
        """invalidate_household_cache should clear specific household."""
        # Add to cache manually
        from app import utils
        utils._member_embedding_cache[("household-1", 1)] = np.array([1.0])
        utils._member_embedding_cache[("household-2", 2)] = np.array([2.0])

        invalidate_household_cache("household-1")

        assert ("household-1", 1) not in utils._member_embedding_cache
        assert ("household-2", 2) in utils._member_embedding_cache

    def test_invalidate_household_cache_all(self) -> None:
        """invalidate_household_cache should clear all when no ID specified."""
        from app import utils
        utils._member_embedding_cache[("household-1", 1)] = np.array([1.0])
        utils._member_embedding_cache[("household-2", 2)] = np.array([2.0])

        invalidate_household_cache()

        assert utils._member_embedding_cache == {}

    def test_invalidate_household_cache_nonexistent(self) -> None:
        """invalidate_household_cache should not error for nonexistent household."""
        # Should not raise
        invalidate_household_cache("nonexistent-household")


class TestMemberCacheIsolation:
    """Regression tests for the (household, user) cache-poison fix.

    The prod incident: a memberless mobile /stt request landed first after a
    restart and, under the old household-keyed cache, poisoned the household
    with ``{}`` so every later node command got no profiles back without ever
    running recognition. The per-member cache must make that impossible.
    """

    def teardown_method(self) -> None:
        invalidate_household_cache()

    def test_memberless_request_does_not_poison_populated_scope(self) -> None:
        """An empty-scope request must not block a later [1, 4] request."""
        from app import utils

        vec = np.array([1.0, 0.0], dtype=np.float32)
        with patch.object(
            utils,
            "_load_member_embedding",
            side_effect=lambda hh, uid: vec if uid in (1, 4) else None,
        ):
            # Memberless request first (the poison seed in prod).
            assert load_household_profiles("hh", []) == {}
            # The real node command must still resolve both members.
            result = load_household_profiles("hh", [1, 4])

        assert set(result) == {1, 4}

    def test_partial_then_full_scope_loads_missing_member(self) -> None:
        """A [1] request then [1, 4] must return both — not a cached partial."""
        from app import utils

        vec = np.array([1.0, 0.0], dtype=np.float32)
        with patch.object(
            utils,
            "_load_member_embedding",
            side_effect=lambda hh, uid: vec if uid in (1, 4) else None,
        ):
            load_household_profiles("hh", [1])
            result = load_household_profiles("hh", [1, 4])

        assert set(result) == {1, 4}

    def test_per_user_embedding_loaded_once(self) -> None:
        """Each member is embedded at most once per process across scopes."""
        from app import utils

        vec = np.array([1.0, 0.0], dtype=np.float32)
        loader = MagicMock(side_effect=lambda hh, uid: vec)
        with patch.object(utils, "_load_member_embedding", loader):
            load_household_profiles("hh", [1])
            load_household_profiles("hh", [1, 4])
            load_household_profiles("hh", [1, 4])

        # user 1 reused across all three calls, user 4 loaded once → 2 total.
        loaded = sorted(call.args[1] for call in loader.call_args_list)
        assert loaded == [1, 4]

    def test_returned_dict_only_contains_requested_members(self) -> None:
        """Member-scope isolation: never return a user outside member_ids."""
        from app import utils

        vec = np.array([1.0, 0.0], dtype=np.float32)
        with patch.object(
            utils,
            "_load_member_embedding",
            side_effect=lambda hh, uid: vec if uid in (1, 4, 7) else None,
        ):
            result = load_household_profiles("hh", [1, 4])

        assert set(result) == {1, 4}  # user 7 enrolled but not requested

    def test_negative_cache_does_not_block_other_member(self) -> None:
        """A None-cached unenrolled user must not affect a different member."""
        from app import utils

        vec = np.array([1.0, 0.0], dtype=np.float32)
        with patch.object(
            utils,
            "_load_member_embedding",
            side_effect=lambda hh, uid: vec if uid == 1 else None,
        ):
            assert load_household_profiles("hh", [99]) == {}  # unenrolled → {}
            result = load_household_profiles("hh", [1])

        assert set(result) == {1}


class TestCachedProfileCounts:
    """speaker_recognition_status's cached_profile_counts shape preservation."""

    def teardown_method(self) -> None:
        invalidate_household_cache()

    def test_counts_group_by_household_and_ignore_negative(self) -> None:
        """{household_id: count of members with a non-None profile}."""
        from app import utils

        utils._member_embedding_cache[("hh1", 1)] = np.array([1.0])
        utils._member_embedding_cache[("hh1", 2)] = np.array([2.0])
        utils._member_embedding_cache[("hh2", 3)] = None  # negative-cached

        counts = utils._cached_profile_counts()
        assert counts == {"hh1": 2, "hh2": 0}


class TestLoadForWhisper:
    """Test the resample/mono helper that feeds pywhispercpp."""

    def _write_wav(self, path: Path, sr: int, channels: int = 1, seconds: float = 0.5) -> None:
        from scipy.io import wavfile
        n = int(sr * seconds)
        # Simple sine so resampling has signal to chew on
        t = np.linspace(0.0, seconds, n, endpoint=False)
        sig = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        if channels == 2:
            sig = np.stack([sig, sig], axis=1)
        wavfile.write(str(path), sr, (sig * 32767).astype(np.int16))

    def test_resamples_48khz_to_16khz(self, tmp_path: Path) -> None:
        wav = tmp_path / "48k.wav"
        self._write_wav(wav, sr=48000, seconds=1.0)

        out = _load_for_whisper(str(wav))

        assert out.dtype == np.float32
        assert out.ndim == 1
        assert abs(len(out) - 16000) < 5  # 1 second @ 16 kHz, ±a sample of slop

    def test_passthrough_at_16khz(self, tmp_path: Path) -> None:
        wav = tmp_path / "16k.wav"
        self._write_wav(wav, sr=16000, seconds=0.5)

        out = _load_for_whisper(str(wav))

        assert out.dtype == np.float32
        assert out.ndim == 1
        assert abs(len(out) - 8000) < 5  # 0.5 second @ 16 kHz

    def test_mono_mixdown(self, tmp_path: Path) -> None:
        wav = tmp_path / "stereo.wav"
        self._write_wav(wav, sr=16000, channels=2, seconds=0.5)

        out = _load_for_whisper(str(wav))

        assert out.ndim == 1


class TestRunWhisper:
    """Test run_whisper function."""

    @patch("app.utils._load_for_whisper", return_value=np.zeros(16000, dtype=np.float32))
    @patch("app.utils.get_model")
    def test_run_whisper_success(self, mock_get_model: MagicMock, mock_load: MagicMock) -> None:
        """run_whisper should join segment texts and return text + segments."""
        model = MagicMock()
        model.transcribe.return_value = [
            _make_segment("Hello ", t0=0, t1=50),
            _make_segment("world", t0=80, t1=120),
        ]
        mock_get_model.return_value = model

        text, segments = run_whisper("/tmp/test.wav")

        assert text == "Hello  world"
        assert segments == [
            {"t0_ms": 0, "t1_ms": 500, "text": "Hello "},
            {"t0_ms": 800, "t1_ms": 1200, "text": "world"},
        ]
        model.transcribe.assert_called_once()

    @patch("app.utils._load_for_whisper", return_value=np.zeros(16000, dtype=np.float32))
    @patch("app.utils.get_model")
    def test_run_whisper_failure_raises_transcription_error(
        self, mock_get_model: MagicMock, mock_load: MagicMock
    ) -> None:
        """run_whisper should wrap underlying errors in WhisperTranscriptionError."""
        model = MagicMock()
        model.transcribe.side_effect = RuntimeError("Model file not found")
        mock_get_model.return_value = model

        with pytest.raises(WhisperTranscriptionError) as exc_info:
            run_whisper("/tmp/test.wav")

        assert "Model file not found" in str(exc_info.value)
        assert "Model file not found" in (exc_info.value.stderr or "")

    @patch("app.utils._load_for_whisper", return_value=np.zeros(16000, dtype=np.float32))
    @patch("app.utils.get_model")
    def test_run_whisper_with_prompt(self, mock_get_model: MagicMock, mock_load: MagicMock) -> None:
        """run_whisper should pass prompt as initial_prompt."""
        model = MagicMock()
        model.transcribe.return_value = [_make_segment("Jarvis turn on lights")]
        mock_get_model.return_value = model

        text, _ = run_whisper("/tmp/test.wav", prompt="Jarvis commands")

        assert text == "Jarvis turn on lights"
        kwargs = model.transcribe.call_args.kwargs
        assert kwargs["initial_prompt"] == "Jarvis commands"

    @patch("app.utils._load_for_whisper", return_value=np.zeros(16000, dtype=np.float32))
    @patch("app.utils.get_model")
    def test_run_whisper_without_prompt(self, mock_get_model: MagicMock, mock_load: MagicMock) -> None:
        """run_whisper should pass empty initial_prompt when prompt is None."""
        model = MagicMock()
        model.transcribe.return_value = [_make_segment("Hello")]
        mock_get_model.return_value = model

        run_whisper("/tmp/test.wav")

        kwargs = model.transcribe.call_args.kwargs
        assert kwargs["initial_prompt"] == ""

    @patch("app.utils._load_for_whisper", return_value=np.zeros(16000, dtype=np.float32))
    @patch("app.utils.get_model")
    def test_run_whisper_with_temperature(self, mock_get_model: MagicMock, mock_load: MagicMock) -> None:
        """run_whisper should pass temperature to the model."""
        model = MagicMock()
        model.transcribe.return_value = [_make_segment("Hello world")]
        mock_get_model.return_value = model

        run_whisper("/tmp/test.wav", temperature=0.3)

        kwargs = model.transcribe.call_args.kwargs
        assert kwargs["temperature"] == 0.3

    @patch("app.utils._load_for_whisper", return_value=np.zeros(16000, dtype=np.float32))
    @patch("app.utils.get_model")
    def test_run_whisper_with_temperature_inc(self, mock_get_model: MagicMock, mock_load: MagicMock) -> None:
        """run_whisper should pass temperature_inc to the model."""
        model = MagicMock()
        model.transcribe.return_value = [_make_segment("Hello world")]
        mock_get_model.return_value = model

        run_whisper("/tmp/test.wav", temperature_inc=0.1)

        kwargs = model.transcribe.call_args.kwargs
        assert kwargs["temperature_inc"] == 0.1

    @patch("app.utils._load_for_whisper", return_value=np.zeros(16000, dtype=np.float32))
    @patch("app.utils.get_model")
    def test_run_whisper_with_beam_size(self, mock_get_model: MagicMock, mock_load: MagicMock) -> None:
        """run_whisper should pass beam_size inside the beam_search dict."""
        model = MagicMock()
        model.transcribe.return_value = [_make_segment("Hello world")]
        mock_get_model.return_value = model

        run_whisper("/tmp/test.wav", beam_size=3)

        kwargs = model.transcribe.call_args.kwargs
        assert kwargs["beam_search"] == {"beam_size": 3, "patience": -1.0}

    @patch("app.utils._load_for_whisper", return_value=np.zeros(16000, dtype=np.float32))
    @patch("app.utils.get_model")
    def test_run_whisper_default_params(self, mock_get_model: MagicMock, mock_load: MagicMock) -> None:
        """run_whisper should use sane defaults for temperature/beam params."""
        model = MagicMock()
        model.transcribe.return_value = [_make_segment("Hello world")]
        mock_get_model.return_value = model

        run_whisper("/tmp/test.wav")

        kwargs = model.transcribe.call_args.kwargs
        assert kwargs["temperature"] == 0.0
        assert kwargs["temperature_inc"] == 0.2
        assert kwargs["beam_search"] == {"beam_size": 5, "patience": -1.0}
        assert kwargs["language"] == "en"

    @patch("app.utils._load_for_whisper", return_value=np.zeros(16000, dtype=np.float32))
    @patch("app.utils.get_model")
    def test_run_whisper_all_params_together(self, mock_get_model: MagicMock, mock_load: MagicMock) -> None:
        """run_whisper should forward all params together."""
        model = MagicMock()
        model.transcribe.return_value = [_make_segment("Hello world")]
        mock_get_model.return_value = model

        text, _ = run_whisper(
            "/tmp/test.wav",
            prompt="Test prompt",
            temperature=0.5,
            temperature_inc=0.15,
            beam_size=8,
        )

        assert text == "Hello world"
        kwargs = model.transcribe.call_args.kwargs
        assert kwargs["initial_prompt"] == "Test prompt"
        assert kwargs["temperature"] == 0.5
        assert kwargs["temperature_inc"] == 0.15
        assert kwargs["beam_search"] == {"beam_size": 8, "patience": -1.0}


class TestRecognizeSpeaker:
    """Test recognize_speaker function."""

    def teardown_method(self) -> None:
        """Clear cache after each test."""
        invalidate_household_cache()

    @patch("app.utils.load_household_profiles")
    def test_recognize_speaker_no_profiles(
        self, mock_load: MagicMock
    ) -> None:
        """recognize_speaker should return None user_id when no profiles exist."""
        mock_load.return_value = {}

        result = recognize_speaker("/tmp/test.wav", "household-1", [1, 2])

        assert result.user_id is None
        assert result.confidence == 0.0

    @patch("app.utils.load_household_profiles")
    @patch("app.utils._get_encoder")
    def test_recognize_speaker_matches_correct_user(
        self, mock_encoder: MagicMock, mock_load: MagicMock
    ) -> None:
        """recognize_speaker should return matched user_id above threshold."""
        mock_load.return_value = {42: np.array([1.0, 0.0, 0.0])}
        # Return embedding very similar to user 42's profile via the
        # encoder facade introduced in Phase 2a.
        mock_encoder.return_value.embed.return_value = np.array([0.95, 0.0, 0.0])

        # Explicit threshold pins the assertion regardless of any future
        # length-adaptive default tweaks.
        result = recognize_speaker("/tmp/test.wav", "household-1", [42], threshold=0.5)

        assert result.user_id == 42
        assert result.confidence > 0.5

    @patch("app.utils.load_household_profiles")
    @patch("app.utils._get_encoder")
    def test_recognize_speaker_below_threshold_returns_none(
        self, mock_encoder: MagicMock, mock_load: MagicMock
    ) -> None:
        """recognize_speaker should return None user_id when below threshold."""
        mock_load.return_value = {42: np.array([1.0, 0.0, 0.0])}
        # Embedding orthogonal to the profile → cosine ~0.0
        mock_encoder.return_value.embed.return_value = np.array([0.0, 1.0, 0.0])

        result = recognize_speaker("/tmp/test.wav", "household-1", [42], threshold=0.5)

        assert result.user_id is None
        # Confidence should still be reported (the best score found)
        assert result.confidence < 0.5

    @patch("app.utils.load_household_profiles")
    @patch("app.utils._get_encoder")
    def test_recognize_speaker_processing_error(
        self, mock_encoder: MagicMock, mock_load: MagicMock
    ) -> None:
        """recognize_speaker should return None user_id on processing error."""
        mock_load.return_value = {1: np.array([1.0, 0.0, 0.0])}
        mock_encoder.return_value.embed.side_effect = RuntimeError("Audio file corrupted")

        result = recognize_speaker("/tmp/test.wav", "household-1", [1])

        assert result.user_id is None
        assert result.confidence == 0.0

    @patch("app.utils.load_household_profiles")
    @patch("app.utils._get_encoder")
    def test_recognize_speaker_custom_threshold(
        self, mock_encoder: MagicMock, mock_load: MagicMock
    ) -> None:
        """recognize_speaker should respect explicit threshold override."""
        mock_load.return_value = {99: np.array([1.0, 0.0, 0.0])}
        # Embedding has 0.8 cosine similarity to user 99's profile
        mock_encoder.return_value.embed.return_value = np.array([0.8, 0.6, 0.0])

        # With threshold 0.5, should match
        result_lenient = recognize_speaker(
            "/tmp/test.wav", "household-1", [99], threshold=0.5
        )
        assert result_lenient.user_id == 99

        # With higher threshold 0.9, should not match
        result_strict = recognize_speaker(
            "/tmp/test.wav", "household-1", [99], threshold=0.9
        )
        assert result_strict.user_id is None

    @patch("app.utils.load_household_profiles")
    @patch("app.utils._get_encoder")
    def test_recognize_speaker_multiple_users_best_match(
        self, mock_encoder: MagicMock, mock_load: MagicMock
    ) -> None:
        """recognize_speaker should return best matching user among multiple."""
        mock_load.return_value = {
            1: np.array([1.0, 0.0, 0.0]),
            2: np.array([0.0, 1.0, 0.0]),
            3: np.array([0.0, 0.0, 1.0]),
        }
        # Embedding most similar to user 2's profile
        mock_encoder.return_value.embed.return_value = np.array([0.1, 0.95, 0.1])

        result = recognize_speaker("/tmp/test.wav", "household-1", [1, 2, 3], threshold=0.5)

        assert result.user_id == 2
        assert result.confidence > 0.5

    @patch("app.services.settings_service.get_settings_service")
    @patch("app.utils.load_household_profiles")
    @patch("app.utils._get_encoder")
    def test_recognize_speaker_margin_gate_rejects_ambiguous(
        self, mock_encoder: MagicMock, mock_load: MagicMock, mock_settings: MagicMock
    ) -> None:
        """With 2+ profiles, a winner-vs-runner-up margin below
        voice.min_speaker_margin returns unknown even when above threshold —
        the clip is too close to another member to safely commit an identity."""
        mock_settings.return_value.get_float.return_value = 0.05
        mock_load.return_value = {
            1: np.array([1.0, 0.0, 0.0]),
            4: np.array([0.0, 1.0, 0.0]),
        }
        # Query nearly equidistant: score(1)=0.72, score(4)=0.70 -> margin 0.02 < 0.05
        mock_encoder.return_value.embed.return_value = np.array([0.72, 0.70, 0.0])
        result = recognize_speaker("/tmp/test.wav", "household-1", [1, 4], threshold=0.5)
        assert result.user_id is None  # abstain, don't guess Alex
        assert result.confidence == pytest.approx(0.72)

    @patch("app.services.settings_service.get_settings_service")
    @patch("app.utils.load_household_profiles")
    @patch("app.utils._get_encoder")
    def test_recognize_speaker_margin_gate_accepts_clear_winner(
        self, mock_encoder: MagicMock, mock_load: MagicMock, mock_settings: MagicMock
    ) -> None:
        """A clear winner (margin >= voice.min_speaker_margin) still matches."""
        mock_settings.return_value.get_float.return_value = 0.05
        mock_load.return_value = {
            1: np.array([1.0, 0.0, 0.0]),
            4: np.array([0.0, 1.0, 0.0]),
        }
        # score(1)=0.95, score(4)=0.10 -> margin 0.85 >> 0.05
        mock_encoder.return_value.embed.return_value = np.array([0.95, 0.10, 0.0])
        result = recognize_speaker("/tmp/test.wav", "household-1", [1, 4], threshold=0.5)
        assert result.user_id == 1

    @patch("app.services.settings_service.get_settings_service")
    @patch("app.utils.load_household_profiles")
    @patch("app.utils._get_encoder")
    def test_recognize_speaker_margin_gate_skipped_single_profile(
        self, mock_encoder: MagicMock, mock_load: MagicMock, mock_settings: MagicMock
    ) -> None:
        """Single-profile households have no runner-up — the gate is skipped so
        a lone enrolled user still matches on threshold alone."""
        mock_settings.return_value.get_float.return_value = 0.05
        mock_load.return_value = {1: np.array([1.0, 0.0, 0.0])}
        mock_encoder.return_value.embed.return_value = np.array([0.72, 0.0, 0.0])
        result = recognize_speaker("/tmp/test.wav", "household-1", [1], threshold=0.5)
        assert result.user_id == 1

    @patch("app.utils.load_household_profiles")
    @patch("app.utils._get_encoder")
    def test_recognize_speaker_logs_per_member_telemetry(
        self, mock_encoder: MagicMock, mock_load: MagicMock, caplog
    ) -> None:
        """recognize_speaker should log runner-up, margin, and per-member scores."""
        import logging as _logging

        mock_load.return_value = {
            1: np.array([1.0, 0.0, 0.0]),
            2: np.array([0.0, 1.0, 0.0]),
        }
        mock_encoder.return_value.embed.return_value = np.array([0.9, 0.1, 0.0])
        mock_encoder.return_value.name = "ecapa"

        with caplog.at_level(_logging.INFO, logger="app.utils"):
            recognize_speaker("/tmp/x.wav", "hh", [1, 2], threshold=0.5)

        text = caplog.text
        assert "scores=[" in text
        assert "second=" in text
        assert "margin=" in text
        assert "encoder=ecapa" in text


class TestResolveVoiceDevice:
    """Test _resolve_voice_device branches."""

    def test_explicit_cpu_returns_cpu(self, monkeypatch):
        from app.utils import _resolve_voice_device
        monkeypatch.setenv("JARVIS_VOICE_DEVICE", "cpu")
        assert _resolve_voice_device() == "cpu"

    def test_auto_with_cuda_returns_cuda(self, monkeypatch):
        from app import utils as utils_mod
        monkeypatch.setenv("JARVIS_VOICE_DEVICE", "auto")
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
        assert utils_mod._resolve_voice_device() == "cuda"

    def test_auto_without_cuda_returns_cpu(self, monkeypatch):
        from app import utils as utils_mod
        monkeypatch.setenv("JARVIS_VOICE_DEVICE", "auto")
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
        assert utils_mod._resolve_voice_device() == "cpu"

    def test_cuda_requested_falls_back_to_cpu_when_unavailable(self, monkeypatch, caplog):
        from app import utils as utils_mod
        monkeypatch.setenv("JARVIS_VOICE_DEVICE", "cuda")
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
        assert utils_mod._resolve_voice_device() == "cpu"

    def test_cuda_requested_and_available_returns_cuda(self, monkeypatch):
        from app import utils as utils_mod
        monkeypatch.setenv("JARVIS_VOICE_DEVICE", "cuda")
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
        assert utils_mod._resolve_voice_device() == "cuda"

    def test_torch_import_failure_falls_back_to_cpu(self, monkeypatch):
        from app import utils as utils_mod
        monkeypatch.setenv("JARVIS_VOICE_DEVICE", "auto")
        # Force `import torch` inside the function to raise
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("torch not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert utils_mod._resolve_voice_device() == "cpu"


class TestPrepForEmbed:
    """Test _prep_for_embed — the symmetric embed-preprocessing helper.

    These cover the fix for the enroll↔runtime loudness asymmetry: enrollment
    is captured raw/quiet while live commands are gain-normalized, so both must
    be RMS-normalized + DC-removed + silence-trimmed before ECAPA sees them.
    No torch/speechbrain needed — this is pure numpy/scipy.
    """

    @staticmethod
    def _write_wav(path: Path, sr: int, audio: np.ndarray) -> None:
        from scipy.io import wavfile
        wavfile.write(str(path), sr, (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16))

    def test_disabled_is_passthrough(self, tmp_path: Path, monkeypatch) -> None:
        """With preprocessing disabled, output equals the raw 16k mono load."""
        from app import utils
        monkeypatch.setattr(utils, "_embed_prep_settings", lambda: (False, -23.0, -40.0))
        sr = 16000
        t = np.linspace(0.0, 1.0, sr, endpoint=False)
        sig = (0.05 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        wav = tmp_path / "q.wav"
        self._write_wav(wav, sr, sig)

        out = utils._prep_for_embed(wav)

        assert np.allclose(out, utils._load_wav_mono_16k(wav))

    def test_normalizes_quiet_clip_up_to_target(self, tmp_path: Path, monkeypatch) -> None:
        """A quiet clip is RMS-normalized up to the target level."""
        from app import utils
        monkeypatch.setattr(utils, "_embed_prep_settings", lambda: (True, -20.0, -60.0))
        sr = 16000
        t = np.linspace(0.0, 1.0, sr, endpoint=False)
        sig = (0.02 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)  # ~-34 dBFS
        wav = tmp_path / "quiet.wav"
        self._write_wav(wav, sr, sig)

        out = utils._prep_for_embed(wav)

        rms_db = 20.0 * np.log10(float(np.sqrt(np.mean(out**2))) + 1e-12)
        assert abs(rms_db - (-20.0)) < 1.5

    def test_removes_dc_offset(self, tmp_path: Path, monkeypatch) -> None:
        """A DC offset is removed before embedding."""
        from app import utils
        monkeypatch.setattr(utils, "_embed_prep_settings", lambda: (True, -20.0, -60.0))
        sr = 16000
        t = np.linspace(0.0, 1.0, sr, endpoint=False)
        sig = (0.1 * np.sin(2 * np.pi * 220.0 * t) + 0.2).astype(np.float32)
        wav = tmp_path / "dc.wav"
        self._write_wav(wav, sr, sig)

        out = utils._prep_for_embed(wav)

        assert abs(float(np.mean(out))) < 0.01

    def test_trims_padding_silence(self, tmp_path: Path, monkeypatch) -> None:
        """Leading/trailing silence is trimmed."""
        from app import utils
        monkeypatch.setattr(utils, "_embed_prep_settings", lambda: (True, -20.0, -40.0))
        sr = 16000
        silence = np.zeros(sr, dtype=np.float32)
        t = np.linspace(0.0, 1.0, sr, endpoint=False)
        tone = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        sig = np.concatenate([silence, tone, silence])
        wav = tmp_path / "pad.wav"
        self._write_wav(wav, sr, sig)

        out = utils._prep_for_embed(wav)

        assert len(out) < len(sig)

    def test_all_silence_does_not_crash(self, tmp_path: Path, monkeypatch) -> None:
        """An all-silence clip falls back to the untrimmed signal, not a sliver."""
        from app import utils
        monkeypatch.setattr(utils, "_embed_prep_settings", lambda: (True, -20.0, -40.0))
        sr = 16000
        wav = tmp_path / "sil.wav"
        self._write_wav(wav, sr, np.zeros(sr, dtype=np.float32))

        out = utils._prep_for_embed(wav)

        assert len(out) >= utils._MIN_PREP_SAMPLES
