"""Box expansion: pixel or ratio mode, 4-direction independent values."""
from __future__ import annotations

from typing import Literal


def expand_boxes(
    boxes: list[dict],
    *,
    mode: Literal["pixel", "ratio"],
    top: float,
    bottom: float,
    left: float,
    right: float,
    img_width: int,
    img_height: int,
) -> list[dict]:
    """Expand plain text boxes outward; other classes untouched.

    Args:
        boxes: Detection dicts with `class_name` and `bbox_xyxy`.
        mode: "pixel" — values are absolute pixels. "ratio" — values are fractions of box size.
        top/bottom/left/right: Expansion per direction (in chosen mode).
        img_width, img_height: Image dimensions for clamping.

    Returns:
        New list of boxes with expanded `bbox_xyxy`. Only `plain text` class is modified.
    """
    out = []
    for box in boxes:
        if box.get("class_name") != "plain text":
            out.append(box)
            continue
        x1, y1, x2, y2 = box["bbox_xyxy"]
        w, h = x2 - x1, y2 - y1
        if mode == "pixel":
            dx_left, dx_right = left, right
            dy_top, dy_bottom = top, bottom
        elif mode == "ratio":
            dx_left, dx_right = left * w, right * w
            dy_top, dy_bottom = top * h, bottom * h
        else:
            raise ValueError(f"unsupported mode: {mode}")
        nx1 = max(0.0, x1 - dx_left)
        ny1 = max(0.0, y1 - dy_top)
        nx2 = min(float(img_width), x2 + dx_right)
        ny2 = min(float(img_height), y2 + dy_bottom)
        new_box = dict(box)
        new_box["bbox_xyxy"] = [nx1, ny1, nx2, ny2]
        new_box["bbox_xywh"] = [
            (nx1 + nx2) / 2, (ny1 + ny2) / 2,
            nx2 - nx1, ny2 - ny1,
        ]
        out.append(new_box)
    return out
