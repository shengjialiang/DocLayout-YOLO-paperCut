"""Tests for service.config.Config and load_config."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from service.config import Config, load_config


@pytest.fixture
def model_file(tmp_path: Path) -> str:
    f = tmp_path / "model.pt"
    f.write_bytes(b"")
    return str(f)


def test_load_config_minimal(model_file: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODEL_PATH", model_file)
    monkeypatch.delenv("DOCRECT_MODEL_PATH", raising=False)
    monkeypatch.delenv("DEVICE", raising=False)
    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.delenv("MAX_CONCURRENT", raising=False)
    monkeypatch.delenv("MAX_FILE_SIZE_MB", raising=False)

    cfg = load_config()
    assert cfg.model_path == model_file
    assert cfg.device is None
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 8000
    assert cfg.max_concurrent == 2
    assert cfg.max_file_size_mb == 20
    assert cfg.docrect_model_path is None


def test_load_config_with_docrect(model_file: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    docrect = tmp_path / "docrect.onnx"
    docrect.write_bytes(b"")
    monkeypatch.setenv("MODEL_PATH", model_file)
    monkeypatch.setenv("DOCRECT_MODEL_PATH", str(docrect))

    cfg = load_config()
    assert cfg.docrect_model_path == str(docrect)


def test_load_config_docrect_missing_file(model_file: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    missing = tmp_path / "does-not-exist.onnx"
    monkeypatch.setenv("MODEL_PATH", model_file)
    monkeypatch.setenv("DOCRECT_MODEL_PATH", str(missing))

    cfg = load_config()
    # Missing file → None with a warning, not an exception.
    assert cfg.docrect_model_path is None