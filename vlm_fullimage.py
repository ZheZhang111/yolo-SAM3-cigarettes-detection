"""Full-image VLM detection (OpenAI-compatible vLLM). ROI prompt lives in vlm_arbitrate.py."""

from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from postprocess import CLASS_NAMES, NAME_TO_ID
from vlm_arbitrate import DEFAULT_API_BASE, DEFAULT_MODEL, _map_class

# Only used when full-image VLM is invoked — not for ROI arbitration.
FULLIMAGE_PROMPT = """你是巡检图像上的目标检测助手。请分析**整张图**（不是裁剪区域），
数清可见的烟盒(cig-pack)和打火机(Lighter)，为**每个独立物体**各输出一个边界框。

只输出 JSON，不要其他文字：
{
  "summary": "一句话描述场景",
  "objects": [
    {
      "class": "cig-pack" | "Lighter" | "cigarette" | "cigarettes_butts",
      "bbox_xyxy": [x1, y1, x2, y2],
      "confidence": 0.0到1.0
    }
  ]
}

bbox_xyxy 规则（必须遵守）：
- 格式 [x1, y1, x2, y2]，(x1,y1) 左上角，(x2,y2) 右下角
- 坐标使用 0~1000 相对刻度（x 相对图宽×1000，y 相对图高×1000），例如 x1=377 表示 37.7% 图宽
- x2>x1，y2>y1；每个物体单独一个框

类别要点：
- cig-pack：长方形纸盒/金属箔烟盒，有品牌、警示语；红色/金色/绿色包装都是烟盒
- Lighter：一次性打火机，细长，常见绿色/红色，有金属点火罩或黑色点火按钮

特殊情况：
1. 打火机压在烟盒上、或贴靠：仍是两个物体 → 输出两个框
2. 不要把绿色烟盒误判为打火机；不要把整图只标成一个 Lighter
3. 若某类不存在，objects 中不要包含该类"""


@dataclass
class FullImageBox:
    class_id: int | None
    class_name: str
    confidence: float
    xc: float
    yc: float
    w: float
    h: float

    def xyxy(self, img_w: int, img_h: int) -> np.ndarray:
        x1 = (self.xc - self.w / 2) * img_w
        y1 = (self.yc - self.h / 2) * img_h
        x2 = (self.xc + self.w / 2) * img_w
        y2 = (self.yc + self.h / 2) * img_h
        return np.array([x1, y1, x2, y2], dtype=np.float32)


@dataclass
class FullImageResult:
    summary: str
    objects: list[FullImageBox]
    raw: dict[str, Any]


def _image_to_b64_full(image_bgr: np.ndarray, max_side: int = 1280) -> str:
    h, w = image_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    img = image_bgr
    if scale < 1.0:
        img = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("failed to encode full image jpeg")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _parse_json_text(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"no JSON in response: {text[:300]}")
    return json.loads(m.group())


def _normalize_bbox(
    v0: float,
    v1: float,
    v2: float,
    v3: float,
    *,
    img_w: int,
    img_h: int,
    is_xyxy: bool,
) -> tuple[float, float, float, float] | None:
    """Return (xc, yc, w, h) normalized 0~1. Handles 0~1, 0~1000, or pixel coords."""
    vals = [v0, v1, v2, v3]
    mx = max(abs(v) for v in vals)

    def to_norm_x(x: float) -> float:
        if mx > 1.0:
            if mx <= 1000.0:
                return x / 1000.0
            return x / img_w
        return x

    def to_norm_y(y: float) -> float:
        if mx > 1.0:
            if mx <= 1000.0:
                return y / 1000.0
            return y / img_h
        return y

    if is_xyxy:
        x1, y1, x2, y2 = to_norm_x(v0), to_norm_y(v1), to_norm_x(v2), to_norm_y(v3)
        if x2 <= x1 or y2 <= y1:
            return None
        xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = x2 - x1, y2 - y1
    else:
        xc, yc = to_norm_x(v0), to_norm_y(v1)
        if mx <= 1.0:
            bw, bh = v2, v3
        elif mx <= 1000.0:
            bw, bh = v2 / 1000.0, v3 / 1000.0
        else:
            bw, bh = v2 / img_w, v3 / img_h

    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    bw = min(max(bw, 0.005), 1.0)
    bh = min(max(bh, 0.005), 1.0)
    return xc, yc, bw, bh


class QwenFullImageClient:
    """Full-image detection client. Uses FULLIMAGE_PROMPT only."""

    def __init__(
        self,
        api_base: str = DEFAULT_API_BASE,
        model: str = DEFAULT_MODEL,
        timeout: float = 120.0,
        max_side: int = 1280,
    ):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_side = max_side

    def health_ok(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.api_base.replace('/v1', '')}/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except (urllib.error.URLError, TimeoutError):
            return False

    def detect_full_image(self, image_bgr: np.ndarray) -> FullImageResult:
        img_h, img_w = image_bgr.shape[:2]
        b64 = _image_to_b64_full(image_bgr, max_side=self.max_side)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                        {"type": "text", "text": FULLIMAGE_PROMPT},
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": 512,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        text = body["choices"][0]["message"]["content"]
        parsed = _parse_json_text(text)

        objects: list[FullImageBox] = []
        for item in parsed.get("objects", []):
            if not isinstance(item, dict):
                continue
            cls_name = str(item.get("class", "other"))
            cls_id, cls_norm_name = _map_class(cls_name)
            if cls_id is None:
                continue
            bbox = item.get("bbox_xyxy") or item.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            v0, v1, v2, v3 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            is_xyxy = item.get("bbox_xyxy") is not None or (v2 > v0 and v3 > v1 and max(v0, v1, v2, v3) > 1.0)
            box_norm = _normalize_bbox(v0, v1, v2, v3, img_w=img_w, img_h=img_h, is_xyxy=is_xyxy)
            if box_norm is None:
                continue
            xc, yc, bw, bh = box_norm
            conf = float(item.get("confidence", 0.5))
            objects.append(
                FullImageBox(
                    class_id=cls_id,
                    class_name=cls_norm_name,
                    confidence=conf,
                    xc=xc,
                    yc=yc,
                    w=bw,
                    h=bh,
                )
            )
        return FullImageResult(
            summary=str(parsed.get("summary", "")),
            objects=objects,
            raw=parsed,
        )
