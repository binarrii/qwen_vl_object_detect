import ast
import json
from dataclasses import dataclass
from typing import Any, Dict, List

from utils.json import parse_json


@dataclass
class DetectedBox:
    bbox: List[int]
    score: float
    label: str = ""


class BBoxesToSAM2:
    """Convert a list of bounding boxes to the format expected by SAM2 nodes."""

    def convert(self, bboxes):
        if not isinstance(bboxes, list):
            raise ValueError("bboxes must be a list")

        # If already batched, return as-is
        if (
            bboxes
            and isinstance(bboxes[0], (list, tuple))
            and bboxes[0]
            and isinstance(bboxes[0][0], (list, tuple))
        ):
            return bboxes

        return [bboxes]


def parse_boxes(
    text: str,
    img_width: int,
    img_height: int,
    input_w: int,
    input_h: int,
    score_threshold: float = 0.6,
) -> List[Dict[str, Any]]:
    """Return bounding boxes parsed from the model's raw JSON output."""
    text = parse_json(text)
    try:
        data = json.loads(text)
    except Exception:
        try:
            data = ast.literal_eval(text)
        except Exception:
            end_idx = text.rfind('"}') + len('"}')
            truncated = text[:end_idx] + "]"
            data = ast.literal_eval(truncated)
    if isinstance(data, dict):
        inner = data.get("content")
        if isinstance(inner, str):
            try:
                data = ast.literal_eval(inner)
            except Exception:
                data = []
        else:
            data = []
    items: List[DetectedBox] = []
    x_scale = img_width / input_w
    y_scale = img_height / input_h

    for item in data:
        box = item.get("bbox_2d") or item.get("bbox") or item
        label = item.get("label", "")
        score = float(item.get("score", 1.0))
        y1, x1, y2, x2 = box[1], box[0], box[3], box[2]
        abs_y1 = int(y1 * y_scale)
        abs_x1 = int(x1 * x_scale)
        abs_y2 = int(y2 * y_scale)
        abs_x2 = int(x2 * x_scale)
        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1
        if score >= score_threshold:
            items.append(DetectedBox([abs_x1, abs_y1, abs_x2, abs_y2], score, label))
    items.sort(key=lambda x: x.score, reverse=True)
    return [{"score": b.score, "bbox": b.bbox, "label": b.label} for b in items]
