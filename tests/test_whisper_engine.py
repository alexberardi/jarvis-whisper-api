"""Tests for whisper_engine module (lazy Model singleton)."""
from unittest.mock import MagicMock, patch

import pytest

from app import whisper_engine
from app.whisper_engine import get_model, reset_model_for_tests


class TestGetModel:
    """Tests for the lazy get_model() singleton."""

    def teardown_method(self) -> None:
        reset_model_for_tests()

    def test_raises_when_model_path_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_model should error clearly if WHISPER_MODEL is not set."""
        monkeypatch.delenv("WHISPER_MODEL", raising=False)
        with pytest.raises(RuntimeError, match="WHISPER_MODEL"):
            get_model()

    def test_loads_once_and_caches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_model should construct the Model exactly once across calls."""
        monkeypatch.setenv("WHISPER_MODEL", "/fake/model.bin")

        fake_model = MagicMock()
        fake_module = MagicMock()
        fake_module.Model.return_value = fake_model

        with patch.dict("sys.modules", {"pywhispercpp.model": fake_module}):
            first = get_model()
            second = get_model()

        assert first is fake_model
        assert second is fake_model
        assert fake_module.Model.call_count == 1

    def test_passes_n_threads_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """WHISPER_N_THREADS should be forwarded as n_threads to Model()."""
        monkeypatch.setenv("WHISPER_MODEL", "/fake/model.bin")
        monkeypatch.setenv("WHISPER_N_THREADS", "8")

        fake_module = MagicMock()
        fake_module.Model.return_value = MagicMock()

        with patch.dict("sys.modules", {"pywhispercpp.model": fake_module}):
            get_model()

        kwargs = fake_module.Model.call_args.kwargs
        assert kwargs["model"] == "/fake/model.bin"
        assert kwargs["n_threads"] == 8
        assert kwargs["print_progress"] is False
        assert kwargs["print_realtime"] is False

    def test_default_n_threads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If WHISPER_N_THREADS is unset, n_threads should default to 4."""
        monkeypatch.setenv("WHISPER_MODEL", "/fake/model.bin")
        monkeypatch.delenv("WHISPER_N_THREADS", raising=False)

        fake_module = MagicMock()
        fake_module.Model.return_value = MagicMock()

        with patch.dict("sys.modules", {"pywhispercpp.model": fake_module}):
            get_model()

        assert fake_module.Model.call_args.kwargs["n_threads"] == 4

    def test_reset_clears_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """reset_model_for_tests should let the next call re-construct."""
        monkeypatch.setenv("WHISPER_MODEL", "/fake/model.bin")

        fake_module = MagicMock()
        fake_module.Model.side_effect = [MagicMock(), MagicMock()]

        with patch.dict("sys.modules", {"pywhispercpp.model": fake_module}):
            first = get_model()
            reset_model_for_tests()
            second = get_model()

        assert first is not second
        assert fake_module.Model.call_count == 2

    def test_module_state_starts_unloaded(self) -> None:
        """The module-level _model should be None until first call."""
        reset_model_for_tests()
        assert whisper_engine._model is None
