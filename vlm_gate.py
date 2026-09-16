"""Gate YOLO boxes that need Qwen3-VL arbitration (lighter vs pack)."""

from __future__ import annotations

import numpy as np

from postprocess import CIG_PACK, CIGARETTES_BUTTS, LIGHTER, box_aspect_ratio, box_iou


def needs_vlm_arbitration(
    cls_ids: np.ndarray,
    confs: np.ndarray,
    boxes_xyxy: np.ndarray,
    *,
    low_conf: float = 0.65,
    tall_aspect: float = 0.85,
    overlap_iou: float = 0.3,
) -> list[int]:
    """Return indices of boxes that should be sent to VLM."""
    n = len(cls_ids)
    if n == 0:
        return []

    candidates: set[int] = set()
    lighter_idx = [i for i, c in enumerate(cls_ids) if int(c) == LIGHTER]
    pack_idx = [i for i, c in enumerate(cls_ids) if int(c) == CIG_PACK]
    butt_idx = [i for i, c in enumerate(cls_ids) if int(c) == CIGARETTES_BUTTS]

    for i in range(n):
        cls = int(cls_ids[i])
        conf = float(confs[i])
        ar = box_aspect_ratio(boxes_xyxy[i])

        if cls == CIG_PACK:
            if conf < low_conf or ar < tall_aspect:
                candidates.add(i)
        elif cls == CIGARETTES_BUTTS:
            if conf < low_conf or ar < tall_aspect:
                candidates.add(i)
        elif cls == LIGHTER and conf < low_conf:
            candidates.add(i)

    for pi in pack_idx:
        for li in lighter_idx:
            if box_iou(boxes_xyxy[pi], boxes_xyxy[li]) >= overlap_iou:
                candidates.add(pi)
                candidates.add(li)

    for bi in butt_idx:
        for li in lighter_idx:
            if box_iou(boxes_xyxy[bi], boxes_xyxy[li]) >= overlap_iou:
                candidates.add(bi)
                candidates.add(li)

    if len(pack_idx) >= 2:
        for i in range(len(pack_idx)):
            for j in range(i + 1, len(pack_idx)):
                a, b = pack_idx[i], pack_idx[j]
                if box_iou(boxes_xyxy[a], boxes_xyxy[b]) >= overlap_iou:
                    candidates.add(a)
                    candidates.add(b)

    return sorted(candidates)
