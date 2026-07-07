import numpy as np
import pytest

from service.inference import parse_detections


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
