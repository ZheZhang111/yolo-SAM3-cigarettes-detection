"""Qwen3-VL client for ROI classification (OpenAI-compatible vLLM API)."""

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

DEFAULT_API_BASE = "http://127.0.0.1:8001/v1"
DEFAULT_MODEL = "qwen3-vl-idc"

PROMPT = """你是物体识别助手。图中是从监控/巡检照片裁剪出的候选区域（可能含少量背景）。
请判断该裁剪区域内**最主要、最居中**的物体属于哪一类，只输出 JSON，不要其他文字：

{
  "class": "Lighter" | "cig-pack" | "cigarette" | "cigarettes_butts" | "other",
  "confidence": 0.0到1.0的数字,
  "multiple_objects": true或false,
  "objects_present": ["cig-pack", "Lighter"] 或 [],
  "reason": "一句话说明"
}

判断要点：
- Lighter：一次性塑料/金属打火机，有点火按钮/金属罩，细长，常见绿/红/透明；表面可有条形码
- cig-pack：长方形纸盒或金属箔烟盒，有品牌文字、健康警示、条码；即使包装为绿色也仍是烟盒
- cigarettes_butts：短小的滤嘴烟头残留
- cigarette：完整未点燃的单根香烟
- other：以上都不是或无法判断

【特殊情况 — 务必注意】
1. 打火机在烟盒上面/旁边/贴靠：裁剪框内可能同时出现两个物体。
   - 设 multiple_objects=true
   - objects_present 列出能看到的全部类别（如 ["cig-pack","Lighter"]）
   - class 填**占据框内面积最大或最居中**的那个；不要只因看到打火机就把整块烟盒判成打火机
2. 检测器只给出一个大框，但框内明显有烟盒+打火机两个物体：
   - multiple_objects=true，objects_present 写全
   - 若 YOLO 初判为 cig-pack 且能看到纸盒平面/品牌文字，优先保持 class="cig-pack"
   - 若 YOLO 初判为 Lighter 且能看到点火按钮/金属罩，优先保持 class="Lighter"
3. 两个绿色物体并排时：烟盒=扁平长方体纸盒；打火机=更细长、有金属点火结构。不要把绿色烟盒误判为打火机。
4. 若裁剪区域主要是烟盒（可见盒面、警示语、品牌），即使旁边紧贴打火机，class 仍应为 cig-pack。
5. 若裁剪区域主要是打火机（可见点火按钮），即使下方有烟盒边缘，class 仍应为 Lighter。
6. 只有当你非常确定裁剪内仅有单一物体时，才设 multiple_objects=false。

输出要求：严格 JSON，objects_present 无多余物体时用 []。"""

PROMPT_WITH_YOLO = """你是物体识别助手。图中是从监控/巡检照片裁剪出的候选区域（可能含少量背景）。
YOLO 初判类别：{yolo_class}（置信度 {yolo_conf:.2f}）。请复核该裁剪内**最主要、最居中**的物体，只输出 JSON：

{{
  "class": "Lighter" | "cig-pack" | "cigarette" | "cigarettes_butts" | "other",
  "confidence": 0.0到1.0的数字,
  "multiple_objects": true或false,
  "objects_present": ["cig-pack", "Lighter"] 或 [],
  "reason": "一句话说明"
}}

判断要点：
- Lighter：一次性塑料/金属打火机，有点火按钮/金属罩，细长，常见绿/红/透明
- cig-pack：长方形纸盒或金属箔烟盒，有品牌文字、健康警示；**绿色包装仍是烟盒，不是打火机**
- cigarettes_butts：短小的滤嘴烟头残留
- cigarette：完整未点燃的单根香烟
- other：以上都不是或无法判断

【特殊情况 — 务必注意】
1. 打火机在烟盒上面/旁边/贴靠：框内可能有两个物体 → multiple_objects=true，objects_present 列全
2. 只有一个大框但明显含烟盒+打火机 → multiple_objects=true；不要整框统一判成 Lighter
3. 两个绿色物体：烟盒=扁平长方纸盒；打火机=更细长、有金属点火结构
4. YOLO 初判 cig-pack 且能看到纸盒/品牌/警示语 → 倾向保持 cig-pack，除非裁剪内明确只有打火机
5. YOLO 初判 Lighter 且能看到点火按钮/金属罩 → 倾向保持 Lighter
6. 只有非常确定裁剪内仅有单一物体时，才设 multiple_objects=false

输出要求：严格 JSON。"""


@dataclass
class VlmResult:
    class_id: int | None
    class_name: str
    confidence: float
    reason: str
    raw: dict[str, Any]
    multiple_objects: bool = False
    objects_present: list[str] | None = None


def crop_roi(
    image: np.ndarray,
    box_xyxy: np.ndarray,
    pad_ratio: float = 0.15,
) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad_ratio, bh * pad_ratio
    x1 = int(max(0, x1 - px))
    y1 = int(max(0, y1 - py))
    x2 = int(min(w, x2 + px))
    y2 = int(min(h, y2 + py))
    return image[y1:y2, x1:x2].copy()


def _image_to_b64(image: np.ndarray, max_side: int = 512) -> str:
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("failed to encode ROI jpeg")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _parse_json_text(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"no JSON in response: {text[:200]}")
    return json.loads(m.group())


def _map_class(name: str) -> tuple[int | None, str]:
    key = name.strip().lower().replace("_", "-")
    if key in ("other", "none", "unknown"):
        return None, "other"
    for k, cid in NAME_TO_ID.items():
        if k.lower().replace("_", "-") == key:
            return cid, CLASS_NAMES[cid]
    return None, name


class QwenVlmClient:
    def __init__(
        self,
        api_base: str = DEFAULT_API_BASE,
        model: str = DEFAULT_MODEL,
        timeout: float = 60.0,
    ):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout = timeout

    def health_ok(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.api_base.replace('/v1', '')}/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except (urllib.error.URLError, TimeoutError):
            return False

    def classify_roi(
        self,
        image_bgr: np.ndarray,
        *,
        yolo_class: str | None = None,
        yolo_conf: float | None = None,
    ) -> VlmResult:
        b64 = _image_to_b64(image_bgr)
        if yolo_class is not None and yolo_conf is not None:
            prompt = PROMPT_WITH_YOLO.format(yolo_class=yolo_class, yolo_conf=yolo_conf)
        else:
            prompt = PROMPT
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
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": 320,
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
        cls_name = str(parsed.get("class", "other"))
        conf = float(parsed.get("confidence", 0.0))
        reason = str(parsed.get("reason", ""))
        multiple = bool(parsed.get("multiple_objects", False))
        present_raw = parsed.get("objects_present", [])
        objects_present: list[str] = []
        if isinstance(present_raw, list):
            for item in present_raw:
                _, norm = _map_class(str(item))
                if norm != "other":
                    objects_present.append(norm)
        cls_id, norm_name = _map_class(cls_name)
        return VlmResult(
            class_id=cls_id,
            class_name=norm_name,
            confidence=conf,
            reason=reason,
            raw=parsed,
            multiple_objects=multiple,
            objects_present=objects_present or None,
        )


def should_apply_vlm_update(
    old_cls: int,
    vlm_res: VlmResult,
    *,
    vlm_min_conf: float = 0.5,
    n_boxes: int = 1,
) -> tuple[bool, str]:
    """Conservative gate: avoid wrong pack<->lighter flips when multiple objects visible."""
    from postprocess import CIG_PACK, LIGHTER

    if vlm_res.class_id is None or vlm_res.confidence < vlm_min_conf:
        return False, "vlm conf too low"

    if vlm_res.class_id == old_cls:
        return False, "same class"

    present = set(vlm_res.objects_present or [])
    pack_lighter = {CIG_PACK, LIGHTER}

    # Single box covering pack+lighters: VLM cannot split; keep YOLO class
    if vlm_res.multiple_objects and n_boxes == 1:
        if present >= {"cig-pack", "Lighter"} or present == {"cig-pack", "Lighter"}:
            return False, "merged box: pack+lighters co-visible, keep yolo"
        if old_cls in pack_lighter and present & pack_lighter:
            return False, "merged box: multi-object, keep yolo"

    # YOLO said pack, VLM wants lighter, but pack still visible in crop
    if old_cls == CIG_PACK and vlm_res.class_id == LIGHTER:
        if "cig-pack" in present or vlm_res.multiple_objects:
            return False, "pack visible in crop, reject lighter flip"
        if vlm_res.confidence < 0.75:
            return False, "pack->lighter needs higher conf"

    # YOLO said lighter, VLM wants pack, but lighter still visible
    if old_cls == LIGHTER and vlm_res.class_id == CIG_PACK:
        if "Lighter" in present or vlm_res.multiple_objects:
            return False, "lighter visible in crop, reject pack flip"
        if vlm_res.confidence < 0.75:
            return False, "lighter->pack needs higher conf"

    return True, "apply"
