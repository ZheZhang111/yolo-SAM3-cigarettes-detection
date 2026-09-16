"""Post-processing for multiclass cigarette detection."""

from __future__ import annotations

import numpy as np

CIGARETTES_BUTTS = 0
CIG_PACK = 1
CIGARETTE = 2
LIGHTER = 3

CLASS_NAMES = {
    0: "cigarettes_butts",
    1: "cig-pack",
    2: "cigarette",
    3: "Lighter",
}

NAME_TO_ID = {v: k for k, v in CLASS_NAMES.items()}
NAME_TO_ID.update({
    "lighter": LIGHTER,
    "cig_pack": CIG_PACK,
    "cig-pack": CIG_PACK,
    "cigarettes_butts": CIGARETTES_BUTTS,
    "cigarette": CIGARETTE,
})


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two xyxy boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _suppress_lower_priority_overlaps(
    cls_ids: np.ndarray,
    boxes_xyxy: np.ndarray,
    priority_cls: int,
    drop_cls: int,
    iou_thresh: float,
) -> np.ndarray:
    n = len(cls_ids)
    keep = np.ones(n, dtype=bool)
    pri_idx = [i for i, c in enumerate(cls_ids) if int(c) == priority_cls]
    drop_idx = [i for i, c in enumerate(cls_ids) if int(c) == drop_cls]

    for di in drop_idx:
        for pi in pri_idx:
            if box_iou(boxes_xyxy[di], boxes_xyxy[pi]) >= iou_thresh:
                keep[di] = False
                break
    return keep


def suppress_butts_overlapping_lighter(
    cls_ids: np.ndarray,
    boxes_xyxy: np.ndarray,
    iou_thresh: float = 0.3,
) -> np.ndarray:
    """Drop cigarettes_butts when overlapping Lighter."""
    return _suppress_lower_priority_overlaps(
        cls_ids, boxes_xyxy, LIGHTER, CIGARETTES_BUTTS, iou_thresh
    )


def suppress_pack_overlapping_lighter(
    cls_ids: np.ndarray,
    boxes_xyxy: np.ndarray,
    iou_thresh: float = 0.3,
) -> np.ndarray:
    """Drop cig-pack when overlapping Lighter."""
    return _suppress_lower_priority_overlaps(
        cls_ids, boxes_xyxy, LIGHTER, CIG_PACK, iou_thresh
    )


def box_aspect_ratio(box: np.ndarray) -> float:
    """Width / height for xyxy box."""
    w = max(1.0, box[2] - box[0])
    h = max(1.0, box[3] - box[1])
    return w / h
