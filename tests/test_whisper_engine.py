"""Tests for whisper_engine module (settings-driven Model with reload)."""
from unittest.mock import MagicMock, patch

import pytest

from app import whisper_engine
from app.whisper_engine import (
    WhisperEngine,
    _read_fingerprint,
    get_model,
    reset_model_for_tests,
)


def _settings_returning(model_path: str) -> MagicMock:
    """Build a mock SettingsService that returns the given model_path."""
    settings = MagicMock()
    settings.get_str.return_value = model_path
    return settings


class TestReadFingerprint:
    """Tests for _read_fingerprint() — pulls config from settings + env."""

    def test_raises_when_model_path_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty model_path setting should raise — operator hasn't configured."""
        with patch(
            "app.services.settings_service.get_settings_service",
            return_value=_settings_returning(""),
        ):
            with pytest.raises(RuntimeError, match="whisper.model_path is not set"):
                _read_fingerprint()

    def test_reads_model_path_from_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fingerprint should reflect the settings value."""
        monkeypatch.delenv("WHISPER_N_THREADS", raising=False)
        with patch(
            "app.services.settings_service.get_settings_service",
            return_value=_settings_returning("/tmp/model.bin"),
        ):
            fp = _read_fingerprint()
        assert fp.model_path == "/tmp/model.bin"
        assert fp.n_threads == 4

    def test_n_threads_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """WHISPER_N_THREADS overrides the n_threads default of 4."""
        monkeypatch.setenv("WHISPER_N_THREADS", "12")
        with patch(
            "app.services.settings_service.get_settings_service",
            return_value=_settings_returning("/tmp/model.bin"),
        ):
            fp = _read_fingerprint()
        assert fp.n_threads == 12


class TestEngineCaching:
    """Tests for WhisperEngine.get() reload-on-fingerprint-change behavior."""

    def teardown_method(self) -> None:
        reset_model_for_tests()

    def _patch_settings(self, model_path: str):
        return patch(
            "app.services.settings_service.get_settings_service",
            return_value=_settings_returning(model_path),
        )

    def _patch_module(self, *models: MagicMock):
        fake_module = MagicMock()
        fake_module.Model.side_effect = list(models)
        return patch.dict("sys.modules", {"pywhispercpp.model": fake_module}), fake_module

    def test_loads_once_when_fingerprint_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two calls with identical settings should reuse the cached model."""
        monkeypatch.delenv("WHISPER_N_THREADS", raising=False)
        first_model, second_model = MagicMock(), MagicMock()
        mod_patch, mod = self._patch_module(first_model, second_model)
        with self._patch_settings("/tmp/a.bin"), mod_patch:
            engine = WhisperEngine()
            a = engine.get()
            b = engine.get()
        assert a is b is first_model
        assert mod.Model.call_count == 1

    def test_reloads_when_model_path_changes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Changing whisper.model_path should trigger a fresh load."""
        monkeypatch.delenv("WHISPER_N_THREADS", raising=False)
        first_model, second_model = MagicMock(), MagicMock()
        mod_patch, mod = self._patch_module(first_model, second_model)
        with mod_patch:
            engine = WhisperEngine()
            with self._patch_settings("/tmp/a.bin"):
                a = engine.get()
            with self._patch_settings("/tmp/b.bin"):
                b = engine.get()
        assert a is first_model
        assert b is second_model
        assert mod.Model.call_count == 2

    def test_keeps_previous_model_when_reload_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bad path setting must not blow away the working model."""
        monkeypatch.delenv("WHISPER_N_THREADS", raising=False)
        good_model = MagicMock()
        fake_module = MagicMock()
        fake_module.Model.side_effect = [good_model, OSError("bad path")]
        with patch.dict("sys.modules", {"pywhispercpp.model": fake_module}):
            engine = WhisperEngine()
            with self._patch_settings("/tmp/a.bin"):
                a = engine.get()
            with self._patch_settings("/tmp/missing.bin"):
                b = engine.get()
        assert a is b is good_model

    def test_reset_clears_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """reset() should let the next get() re-construct from settings."""
        monkeypatch.delenv("WHISPER_N_THREADS", raising=False)
        m1, m2 = MagicMock(), MagicMock()
        mod_patch, mod = self._patch_module(m1, m2)
        with self._patch_settings("/tmp/a.bin"), mod_patch:
            engine = WhisperEngine()
            a = engine.get()
            engine.reset()
            b = engine.get()
        assert a is m1 and b is m2
        assert mod.Model.call_count == 2

    def test_get_model_uses_singleton(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The module-level get_model() should construct a singleton WhisperEngine."""
        monkeypatch.delenv("WHISPER_N_THREADS", raising=False)
        m = MagicMock()
        mod_patch, mod = self._patch_module(m)
        with self._patch_settings("/tmp/a.bin"), mod_patch:
            assert get_model() is m
            assert get_model() is m  # second call hits cache
        assert mod.Model.call_count == 1
        # Singleton is alive after first construction
        assert whisper_engine._engine is not None
