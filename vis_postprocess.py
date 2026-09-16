"""Visualization post-process: bold Chinese labels and percent confidence."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from vis_labels import class_name_zh, draw_chinese_label

# BGR colors per class (same as predict_multiclass)
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 140, 255),
    1: (0, 0, 255),
    2: (255, 180, 0),
    3: (0, 220, 0),
}


@dataclass
class VisPostprocessConfig:
    """Control output image label style."""

    enabled: bool = True
    use_zh: bool = True
    percent_conf: bool = True
    bold_zh: bool = True
    font_scale: float = 1.0


def format_confidence(conf: float, *, as_percent: bool) -> str:
    if as_percent:
        return f"{round(conf * 100)}%"
    return f"{conf:.2f}"


def format_box_label(
    cls: int,
    conf: float,
    names: dict,
    *,
    cfg: VisPostprocessConfig,
) -> str:
    if cfg.enabled and cfg.use_zh:
        name = class_name_zh(cls, names)
    else:
        name = str(names[cls])
    as_percent = cfg.enabled and cfg.percent_conf
    return f"{name} {format_confidence(conf, as_percent=as_percent)}"


def vis_scale(h: int, w: int, *, cfg: VisPostprocessConfig) -> tuple[int, int, float, int]:
    """line_width, font_size (zh px), font_scale (en), font_thickness (en)."""
    base = max(h, w)
    line_w = max(5, int(base * 0.005))
    use_zh = cfg.enabled and cfg.use_zh
    font_size = max(28, int(base * 0.04 * cfg.font_scale))
    if use_zh and cfg.enabled and cfg.bold_zh:
        font_size = int(font_size * 1.12)
    font_scale = max(1.2, base * 0.0018)
    font_th = max(3, int(font_scale * 2.5))
    if cfg.enabled and cfg.bold_zh and not use_zh:
        font_th = max(font_th + 1, int(font_th * 1.3))
    return line_w, font_size, font_scale, font_th


def draw_detection_image(
    img_bgr: np.ndarray,
    boxes: list[dict],
    names: dict,
    *,
    cfg: VisPostprocessConfig | None = None,
) -> np.ndarray:
    """Draw boxes on image.

    Each box dict: cls (int), conf (float), xyxy (x1,y1,x2,y2).
    """
    cfg = cfg or VisPostprocessConfig()
    out = img_bgr.copy()
    if not boxes:
        return out

    h, w = out.shape[:2]
    line_w, font_size, font_scale, font_th = vis_scale(h, w, cfg=cfg)
    use_zh = cfg.enabled and cfg.use_zh

    for box in boxes:
        cls = int(box["cls"])
        conf = float(box["conf"])
        x1, y1, x2, y2 = map(int, box["xyxy"])
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, line_w)
        label = format_box_label(cls, conf, names, cfg=cfg)
        if use_zh:
            ty = max(y1 - font_size - 20, 0)
            out = draw_chinese_label(
                out,
                label,
                (x1, ty),
                font_size=font_size,
                color_bgr=color,
                bold=cfg.bold_zh,
            )
        else:
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_th
            )
            ty = max(y1 - 8, th + 8)
            cv2.rectangle(out, (x1, ty - th - 8), (x1 + tw + 8, ty + baseline), color, -1)
            cv2.putText(
                out,
                label,
                (x1 + 4, ty - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_th,
                cv2.LINE_AA,
            )
    return out


def boxes_from_yolo_result(result) -> list[dict]:
    items: list[dict] = []
    if result.boxes is None or len(result.boxes) == 0:
        return items
    for b in result.boxes:
        items.append(
            {
                "cls": int(b.cls),
                "conf": float(b.conf),
                "xyxy": tuple(float(v) for v in b.xyxy[0].tolist()),
            }
        )
    return items
