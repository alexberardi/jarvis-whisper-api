"""Tests for the speaker-encoder dispatch in app/utils.py.

Covers:
- `_get_encoder()` honors voice.encoder setting
- ECAPA → resemblyzer fallback when speechbrain is unavailable
- Cache invalidation when encoder switches
- SpeakerEncoder.embed returns L2-normalized output
"""
from __future__ import annotations

import numpy as np
import pytest

from app import utils as utils_mod
from app.utils import SpeakerEncoder, _get_encoder


@pytest.fixture(autouse=True)
def _reset_encoder_state(monkeypatch):
    """Wipe encoder singletons between tests."""
    monkeypatch.setattr(utils_mod, "_encoder", None)
    monkeypatch.setattr(utils_mod, "_encoder_fingerprint", None)
    monkeypatch.setattr(utils_mod, "_household_profiles_cache", {})
    yield


def _stub_encoder(name: str) -> SpeakerEncoder:
    def embed(path):
        # Deterministic L2-normalized vector keyed off the path string
        rng = np.random.default_rng(abs(hash(str(path))) % (2**32))
        v = rng.standard_normal(32).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)

    return SpeakerEncoder(name=name, embed=embed)


class TestEncoderSelection:
    def test_default_to_ecapa(self, monkeypatch):
        called = {}

        def fake_ecapa(device):
            called["ecapa"] = True
            return _stub_encoder("ecapa")

        def fake_resemblyzer(device):
            called["resemblyzer"] = True
            return _stub_encoder("resemblyzer")

        monkeypatch.setattr(utils_mod, "_load_ecapa_encoder", fake_ecapa)
        monkeypatch.setattr(utils_mod, "_load_resemblyzer_encoder", fake_resemblyzer)
        monkeypatch.setattr(utils_mod, "_desired_encoder_name", lambda: "ecapa")
        monkeypatch.setattr(utils_mod, "_resolve_voice_device", lambda: "cpu")

        enc = _get_encoder()
        assert enc.name == "ecapa"
        assert called == {"ecapa": True}

    def test_select_resemblyzer_via_setting(self, monkeypatch):
        monkeypatch.setattr(
            utils_mod, "_load_resemblyzer_encoder",
            lambda device: _stub_encoder("resemblyzer"),
        )
        monkeypatch.setattr(utils_mod, "_desired_encoder_name", lambda: "resemblyzer")
        monkeypatch.setattr(utils_mod, "_resolve_voice_device", lambda: "cpu")

        enc = _get_encoder()
        assert enc.name == "resemblyzer"

    def test_unknown_encoder_falls_back_to_ecapa(self, monkeypatch):
        monkeypatch.setattr(
            utils_mod, "_load_ecapa_encoder",
            lambda device: _stub_encoder("ecapa"),
        )
        monkeypatch.setattr(utils_mod, "_desired_encoder_name", lambda: "unknownmodel")
        monkeypatch.setattr(utils_mod, "_resolve_voice_device", lambda: "cpu")

        enc = _get_encoder()
        assert enc.name == "ecapa"


class TestEncoderFallback:
    def test_ecapa_unavailable_falls_back_to_resemblyzer(self, monkeypatch):
        def fake_ecapa(device):
            raise ImportError("speechbrain missing")

        monkeypatch.setattr(utils_mod, "_load_ecapa_encoder", fake_ecapa)
        monkeypatch.setattr(
            utils_mod, "_load_resemblyzer_encoder",
            lambda device: _stub_encoder("resemblyzer"),
        )
        monkeypatch.setattr(utils_mod, "_desired_encoder_name", lambda: "ecapa")
        monkeypatch.setattr(utils_mod, "_resolve_voice_device", lambda: "cpu")

        enc = _get_encoder()
        assert enc.name == "resemblyzer"

    def test_resemblyzer_unavailable_raises(self, monkeypatch):
        # Resemblyzer being unavailable is the user's explicit choice — no fallback
        def fake_resemblyzer(device):
            raise ImportError("resemblyzer missing")

        monkeypatch.setattr(utils_mod, "_load_resemblyzer_encoder", fake_resemblyzer)
        monkeypatch.setattr(utils_mod, "_desired_encoder_name", lambda: "resemblyzer")
        monkeypatch.setattr(utils_mod, "_resolve_voice_device", lambda: "cpu")

        with pytest.raises(ImportError):
            _get_encoder()


class TestDesiredEncoderName:
    def test_reads_voice_encoder_from_settings(self, monkeypatch):
        class _FakeSvc:
            def get_string(self, key, default):
                assert key == "voice.encoder"
                return "resemblyzer"

        monkeypatch.setattr(
            "app.services.settings_service.get_settings_service",
            lambda: _FakeSvc(),
        )
        assert utils_mod._desired_encoder_name() == "resemblyzer"

    def test_falls_back_to_ecapa_when_setting_blank(self, monkeypatch):
        class _FakeSvc:
            def get_string(self, key, default):
                return ""

        monkeypatch.setattr(
            "app.services.settings_service.get_settings_service",
            lambda: _FakeSvc(),
        )
        assert utils_mod._desired_encoder_name() == "ecapa"


class TestLengthAdaptiveThreshold:
    def test_short_clip_uses_short_threshold(self, monkeypatch):
        # Drive _pick_threshold through a fake settings service that returns
        # the same values previous tests asserted against.
        values = {
            "voice.similarity_threshold": 0.5,
            "voice.threshold_short": 0.65,
            "voice.threshold_long": 0.4,
            "voice.short_cutoff_seconds": 1.0,
            "voice.long_cutoff_seconds": 3.0,
        }

        class _FakeSvc:
            def get_float(self, key, default):
                return values.get(key, default)

        monkeypatch.setattr(
            "app.services.settings_service.get_settings_service",
            lambda: _FakeSvc(),
        )

        assert utils_mod._pick_threshold(0.5) == 0.65   # short
        assert utils_mod._pick_threshold(2.0) == 0.5    # normal
        assert utils_mod._pick_threshold(5.0) == 0.4    # long
        assert utils_mod._pick_threshold(0.0) == 0.5    # unknown/zero duration → normal


class TestCacheInvalidation:
    def test_fingerprint_caches_same_config(self, monkeypatch):
        call_count = {"n": 0}

        def fake_ecapa(device):
            call_count["n"] += 1
            return _stub_encoder("ecapa")

        monkeypatch.setattr(utils_mod, "_load_ecapa_encoder", fake_ecapa)
        monkeypatch.setattr(utils_mod, "_desired_encoder_name", lambda: "ecapa")
        monkeypatch.setattr(utils_mod, "_resolve_voice_device", lambda: "cpu")

        _get_encoder()
        _get_encoder()
        _get_encoder()
        assert call_count["n"] == 1

    def test_switching_encoder_invalidates_profile_cache(self, monkeypatch):
        # Seed the cache as if profiles were loaded
        utils_mod._household_profiles_cache["h1"] = {42: np.zeros(32, dtype=np.float32)}

        # First call: load ecapa
        monkeypatch.setattr(
            utils_mod, "_load_ecapa_encoder",
            lambda device: _stub_encoder("ecapa"),
        )
        monkeypatch.setattr(
            utils_mod, "_load_resemblyzer_encoder",
            lambda device: _stub_encoder("resemblyzer"),
        )
        monkeypatch.setattr(utils_mod, "_resolve_voice_device", lambda: "cpu")

        monkeypatch.setattr(utils_mod, "_desired_encoder_name", lambda: "ecapa")
        _get_encoder()
        # Cache still has the seed (first-time load doesn't invalidate)
        assert "h1" in utils_mod._household_profiles_cache

        # Switch encoder — should invalidate
        monkeypatch.setattr(utils_mod, "_desired_encoder_name", lambda: "resemblyzer")
        _get_encoder()
        assert utils_mod._household_profiles_cache == {}
