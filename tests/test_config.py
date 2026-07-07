import pytest
from service.config import Config, load_config


def test_load_config_with_env(monkeypatch, tmp_path):
    """环境变量优先于默认。"""
    model_file = tmp_path / "m.pt"
    model_file.touch()
    monkeypatch.setenv("MODEL_PATH", str(model_file))
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setenv("MAX_CONCURRENT", "3")
    monkeypatch.setenv("MAX_FILE_SIZE_MB", "10")

    cfg = load_config()

    assert cfg.model_path == str(model_file)
    assert cfg.device == "cpu"
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 9000
    assert cfg.max_concurrent == 3
    assert cfg.max_file_size_mb == 10


def test_load_config_defaults(monkeypatch, tmp_path):
    """未设置环境变量时使用默认。"""
    model_file = tmp_path / "m.pt"
    model_file.touch()
    monkeypatch.setenv("MODEL_PATH", str(model_file))
    for k in ("DEVICE", "HOST", "PORT", "MAX_CONCURRENT", "MAX_FILE_SIZE_MB"):
        monkeypatch.delenv(k, raising=False)

    cfg = load_config()

    assert cfg.device is None  # 表示自动检测
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 8000
    assert cfg.max_concurrent == 1
    assert cfg.max_file_size_mb == 20


def test_load_config_missing_model_path(monkeypatch, tmp_path):
    """MODEL_PATH 未设置应抛错。"""
    monkeypatch.delenv("MODEL_PATH", raising=False)
    with pytest.raises(ValueError, match="MODEL_PATH"):
        load_config()


def test_load_config_nonexistent_model(tmp_path):
    """模型文件不存在应抛错。"""
    import os
    os.environ["MODEL_PATH"] = str(tmp_path / "nonexistent.pt")
    try:
        with pytest.raises(FileNotFoundError):
            load_config()
    finally:
        del os.environ["MODEL_PATH"]
