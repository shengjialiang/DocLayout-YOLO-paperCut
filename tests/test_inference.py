import asyncio
import io
from unittest import mock

import numpy as np
import pytest
from PIL import Image

from service.inference import load_model, parse_detections, predict_one


class FakeBoxes:
    """Stand-in for doclayout_yolo.engine.results.Boxes."""

    def __init__(self, xyxy, conf, cls, names):
        self.xyxy = xyxy      # numpy array of shape (N, 4)
        self.conf = conf      # numpy array of shape (N,)
        self.cls = cls        # numpy array of shape (N,) int
        self._names = names

    def __len__(self):
        return len(self.xyxy)

    @property
    def names(self):
        # API: results.boxes.names
        return self._names


class FakeResult:
    def __init__(self, orig_shape, names, boxes):
        self.orig_shape = orig_shape  # (h, w)
        self.names = names
        self.boxes = boxes


def test_parse_detections_empty():
    """Empty boxes returns empty list."""
    boxes = FakeBoxes(
        xyxy=np.zeros((0, 4)),
        conf=np.zeros((0,)),
        cls=np.zeros((0,), dtype=int),
        names={0: "title"},
    )
    res = FakeResult(orig_shape=(100, 200), names={0: "title"}, boxes=boxes)

    detections = parse_detections(res)

    assert detections == []


def test_parse_detections_returns_expected_fields():
    """Each detection has id, class_id, class_name, bbox_xyxy, bbox_xywh, score."""
    boxes = FakeBoxes(
        xyxy=np.array([[10.0, 20.0, 110.0, 70.0], [200.0, 50.0, 300.0, 150.0]]),
        conf=np.array([0.95, 0.80]),
        cls=np.array([0, 1], dtype=int),
        names={0: "title", 1: "text"},
    )
    res = FakeResult(orig_shape=(200, 400), names={0: "title", 1: "text"}, boxes=boxes)

    detections = parse_detections(res)

    assert len(detections) == 2
    assert detections[0]["id"] == 0
    assert detections[0]["class_id"] == 0
    assert detections[0]["class_name"] == "title"
    assert detections[0]["bbox_xyxy"] == [10.0, 20.0, 110.0, 70.0]
    assert detections[0]["bbox_xywh"] == [60.0, 45.0, 100.0, 50.0]  # xc, yc, w, h
    assert detections[0]["score"] == pytest.approx(0.95)
    assert detections[1]["class_name"] == "text"
    assert detections[1]["bbox_xyxy"] == [200.0, 50.0, 300.0, 150.0]


def test_load_model_calls_yolov10(monkeypatch, tmp_path):
    """load_model instantiates YOLOv10 with the given path and returns it."""
    model_file = tmp_path / "model.pt"
    model_file.touch()

    fake_instance = object()
    fake_yolov10_cls = mock.MagicMock(return_value=fake_instance)

    with mock.patch("service.inference.YOLOv10", fake_yolov10_cls):
        result = load_model(str(model_file))

    assert result is fake_instance
    fake_yolov10_cls.assert_called_once_with(str(model_file))


class _FakeResults:
    """Mimics doclayout_yolo Results object."""

    def __init__(self, names, boxes):
        self.names = names
        self.boxes = boxes
        self.orig_shape = (100, 200)

    def plot(self, pil=True, line_width=5, font_size=20):
        # Return a small RGB numpy array
        return np.zeros((100, 200, 3), dtype=np.uint8)


def _fake_image_bytes():
    """Build a 100x200 white PNG in memory."""
    img = np.ones((100, 200, 3), dtype=np.uint8) * 255
    pil = Image.fromarray(img[:, :, ::-1])  # BGR->RGB
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def test_predict_one_runs_model_and_returns_dict(monkeypatch):
    """predict_one calls model.predict, parses result, encodes image, returns metadata."""
    image_bytes = _fake_image_bytes()

    fake_box = mock.MagicMock()
    fake_box.xyxy = np.array([[10.0, 20.0, 110.0, 70.0]])
    fake_box.conf = np.array([0.95])
    fake_box.cls = np.array([0])
    fake_results = [_FakeResults(names={0: "title"}, boxes=fake_box)]

    fake_model = mock.MagicMock()
    fake_model.predict = mock.MagicMock(return_value=fake_results)
    fake_semaphore = mock.MagicMock()
    fake_semaphore.__aenter__ = mock.AsyncMock(return_value=None)
    fake_semaphore.__aexit__ = mock.AsyncMock(return_value=None)

    out = asyncio.run(predict_one(
        image_bytes=image_bytes,
        model=fake_model,
        semaphore=fake_semaphore,
        conf=0.2,
        imgsz=1024,
        line_width=5,
        font_size=20,
        device="cpu",
    ))

    # model.predict was called with decoded BGR numpy array
    fake_model.predict.assert_called_once()
    call_args = fake_model.predict.call_args
    # First positional arg should be a numpy array (decoded image)
    img_arg = call_args.args[0]
    assert isinstance(img_arg, np.ndarray)
    assert call_args.kwargs["conf"] == 0.2
    assert call_args.kwargs["imgsz"] == 1024
    assert call_args.kwargs["device"] == "cpu"

    assert out["width"] == 200
    assert out["height"] == 100
    assert out["num_detections"] == 1
    assert out["conf_threshold"] == 0.2
    assert out["model_imgsz"] == 1024
    assert out["device"] == "cpu"
    assert out["detections"][0]["class_name"] == "title"
    assert "annotated_base64" in out
    assert out["annotated_base64"].startswith("data:image/jpeg;base64,")
    assert out["inference_time_ms"] >= 0


def test_predict_one_empty_detections(monkeypatch):
    """No detections returns empty list and num_detections=0."""
    image_bytes = _fake_image_bytes()

    fake_box = mock.MagicMock()
    fake_box.xyxy = np.zeros((0, 4))
    fake_box.conf = np.zeros((0,))
    fake_box.cls = np.zeros((0,), dtype=int)
    fake_results = [_FakeResults(names={0: "title"}, boxes=fake_box)]

    fake_model = mock.MagicMock()
    fake_model.predict = mock.MagicMock(return_value=fake_results)
    fake_semaphore = mock.MagicMock()
    fake_semaphore.__aenter__ = mock.AsyncMock(return_value=None)
    fake_semaphore.__aexit__ = mock.AsyncMock(return_value=None)

    out = asyncio.run(predict_one(
        image_bytes=image_bytes,
        model=fake_model,
        semaphore=fake_semaphore,
        conf=0.2,
        imgsz=1024,
        line_width=5,
        font_size=20,
        device="cpu",
    ))
    assert out["num_detections"] == 0
    assert out["detections"] == []
