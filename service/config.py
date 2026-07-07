"""Service configuration loaded from environment variables."""
from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Config:
    model_path: str
    device: str | None  # None means auto-detect
    host: str
    port: int
    max_concurrent: int
    max_file_size_mb: int


def load_config() -> Config:
    model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        raise ValueError(
            "MODEL_PATH environment variable is required. "
            "Set it to your .pt model file, e.g. "
            "MODEL_PATH=/path/to/doclayout_yolo_docstructbench_imgsz1024.pt"
        )
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = os.environ.get("DEVICE") or None  # empty string -> None

    return Config(
        model_path=model_path,
        device=device,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        max_concurrent=int(os.environ.get("MAX_CONCURRENT", "1")),
        max_file_size_mb=int(os.environ.get("MAX_FILE_SIZE_MB", "20")),
    )
