"""Tests for whisper_engine module (settings-driven Model with reload)."""
import os
from unittest.mock import MagicMock, patch

import pytest

from app import whisper_engine
from app.whisper_engine import (
    WhisperEngine,
    _build_model,
    _read_fingerprint,
    get_model,
    reset_model_for_tests,
)


def _settings_returning(
    model_path: str, allow_autodownload: bool = True
) -> MagicMock:
    """Build a mock SettingsService that returns the given model_path.

    `allow_autodownload` defaults to True so the existing caching tests (which
    use non-existent /tmp paths) still construct the Model regardless of disk
    state. The egress-gate tests set it False explicitly.
    """
    settings = MagicMock()
    settings.get_str.return_value = model_path
    settings.get_bool.return_value = allow_autodownload
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

    def _patch_settings(self, model_path: str, allow_autodownload: bool = True):
        return patch(
            "app.services.settings_service.get_settings_service",
            return_value=_settings_returning(model_path, allow_autodownload),
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


class TestAutodownloadGate:
    """Tests for the model auto-download egress gate (fail closed by default).

    The gate must NEVER let pywhispercpp egress to huggingface.co when
    auto-download is disabled and no local model exists. We prove no egress by
    asserting the fake Model is never constructed.
    """

    def teardown_method(self) -> None:
        reset_model_for_tests()

    def _patch_settings(self, model_path: str, allow_autodownload: bool):
        return patch(
            "app.services.settings_service.get_settings_service",
            return_value=_settings_returning(model_path, allow_autodownload),
        )

    def _patch_module(self):
        fake_module = MagicMock()
        fake_module.Model.return_value = MagicMock()
        return patch.dict("sys.modules", {"pywhispercpp.model": fake_module}), fake_module

    def test_disabled_and_no_local_file_raises_and_never_constructs_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Gate off + missing file → guidance RuntimeError, Model NEVER built."""
        monkeypatch.delenv("WHISPER_N_THREADS", raising=False)
        mod_patch, mod = self._patch_module()
        missing = "/tmp/definitely-not-a-real-whisper-model-xyz.bin"
        assert not os.path.exists(missing)
        with self._patch_settings(missing, allow_autodownload=False), mod_patch:
            engine = WhisperEngine()
            with pytest.raises(RuntimeError, match="auto-download is disabled"):
                engine.get()
        # No egress: pywhispercpp Model was never constructed.
        assert mod.Model.call_count == 0

    def test_disabled_with_existing_file_constructs_with_resolved_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """Gate off + file exists → Model built with the expanduser'd path."""
        monkeypatch.delenv("WHISPER_N_THREADS", raising=False)
        model_file = tmp_path / "ggml-base.en.bin"
        model_file.write_bytes(b"fake-ggml")
        mod_patch, mod = self._patch_module()
        with self._patch_settings(str(model_file), allow_autodownload=False), mod_patch:
            engine = WhisperEngine()
            engine.get()
        assert mod.Model.call_count == 1
        _, kwargs = mod.Model.call_args
        assert kwargs["model"] == str(model_file)

    def test_enabled_and_no_local_file_constructs_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Gate ON + missing file → download path permitted, Model IS built."""
        monkeypatch.delenv("WHISPER_N_THREADS", raising=False)
        mod_patch, mod = self._patch_module()
        missing = "/tmp/definitely-not-a-real-whisper-model-abc.bin"
        assert not os.path.exists(missing)
        with self._patch_settings(missing, allow_autodownload=True), mod_patch:
            engine = WhisperEngine()
            engine.get()
        # Download permitted: Model constructed (mock — no real network).
        assert mod.Model.call_count == 1
        _, kwargs = mod.Model.call_args
        assert kwargs["model"] == missing

    def test_tilde_path_is_expanded_before_model_load(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """A '~'-prefixed model_path must be expanded to an absolute path."""
        monkeypatch.delenv("WHISPER_N_THREADS", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        # Touch the file at the expanded location so the gate passes too.
        model_file = tmp_path / "x.bin"
        model_file.write_bytes(b"fake-ggml")
        mod_patch, mod = self._patch_module()
        with self._patch_settings("~/x.bin", allow_autodownload=False), mod_patch:
            engine = WhisperEngine()
            engine.get()
        assert mod.Model.call_count == 1
        _, kwargs = mod.Model.call_args
        assert kwargs["model"] == str(model_file)
        assert "~" not in kwargs["model"]
