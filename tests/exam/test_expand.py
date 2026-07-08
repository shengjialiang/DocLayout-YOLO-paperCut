"""Box expansion tests (pixel / ratio, 4-direction, clamping)."""
import pytest

from service.exam.expand import expand_boxes


def make_box(x1, y1, x2, y2, cls="plain text"):
    return {
        "id": 0,
        "class_id": 0,
        "class_name": cls,
        "bbox_xyxy": [x1, y1, x2, y2],
        "bbox_xywh": [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1],
        "score": 0.9,
    }


def test_pixel_mode_expands_all_four_directions():
    boxes = [make_box(100, 100, 200, 200)]
    out = expand_boxes(
        boxes, mode="pixel",
        top=10, bottom=20, left=5, right=15,
        img_width=1000, img_height=1000,
    )
    assert len(out) == 1
    x1, y1, x2, y2 = out[0]["bbox_xyxy"]
    assert (x1, y1, x2, y2) == (95, 90, 215, 220)


def test_pixel_mode_skips_non_plain_text():
    boxes = [
        make_box(100, 100, 200, 200, cls="plain text"),
        make_box(50, 50, 80, 80, cls="figure"),
    ]
    out = expand_boxes(
        boxes, mode="pixel",
        top=10, bottom=10, left=10, right=10,
        img_width=1000, img_height=1000,
    )
    # figure unchanged
    assert out[1]["bbox_xyxy"] == [50, 50, 80, 80]
    # plain text expanded
    assert out[0]["bbox_xyxy"] == [90, 90, 210, 210]


def test_pixel_mode_clamps_to_image_bounds():
    boxes = [make_box(5, 5, 10, 10)]
    out = expand_boxes(
        boxes, mode="pixel",
        top=100, bottom=100, left=100, right=100,
        img_width=50, img_height=50,
    )
    assert out[0]["bbox_xyxy"] == [0, 0, 50, 50]
