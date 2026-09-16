#!/usr/bin/env python3
"""SAM3 text-prompt inference on New_test/image_0828-v2 images."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

SAM3_ROOT = Path("/data/zhangzhe/do-as-i-do/reconstruction/modules/sam3")
sys.path.insert(0, str(SAM3_ROOT))

from sam3.model_builder import build_sam3_image_model  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402

ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT / "New_test" / "image_0828-v2"
OUT_DIR = ROOT / "runs/detect/New_test" / "image_0828_v2_sam3"
CKPT = Path("/data/zhangzhe/do-as-i-do/reconstruction/checkpoints/sam3/sam3.pt")

PROMPTS = [
    ("cigarette pack", (0, 0, 255)),
    ("lighter", (0, 220, 0)),
    ("cigarette", (255, 180, 0)),
    ("cigarette butt", (0, 140, 255)),
]

IMAGES = sorted(IMG_DIR.glob("*.jpg"))
CONF = 0.25
DEVICE = "cuda"


def box_xyxy_to_norm(box: torch.Tensor, w: int, h: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box.tolist()
    xc = ((x1 + x2) / 2) / w
    yc = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return xc, yc, bw, bh


def draw_result(img_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    out = img_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        color = det["color"]
        label = f"{det['prompt']} {det['score']:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        cv2.putText(out, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"checkpoint: {CKPT}")
    print(f"device    : {DEVICE}")
    print(f"images    : {len(IMAGES)}")
    print(f"output    : {OUT_DIR}\n")

    model = build_sam3_image_model(
        checkpoint_path=str(CKPT),
        load_from_HF=False,
        device=DEVICE,
    )
    model = model.to(DEVICE)
    processor = Sam3Processor(model, device=DEVICE, confidence_threshold=CONF)

    summary_lines: list[str] = []
    for img_path in IMAGES:
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        all_dets: list[dict] = []

        print(img_path.name)
        for prompt, color in PROMPTS:
            processor.reset_all_prompts({})
            state = processor.set_image(image)
            state = processor.set_text_prompt(prompt=prompt, state=state)
            boxes = state.get("boxes")
            scores = state.get("scores")
            if boxes is None or len(boxes) == 0:
                print(f"  {prompt:16s} -> (none)")
                continue
            for box, score in zip(boxes, scores):
                score_f = float(score)
                xc, yc, bw, bh = box_xyxy_to_norm(box, w, h)
                x1, y1, x2, y2 = box.tolist()
                all_dets.append(
                    {
                        "prompt": prompt,
                        "score": score_f,
                        "box": (x1, y1, x2, y2),
                        "norm": (xc, yc, bw, bh),
                        "color": color,
                    }
                )
                print(f"  {prompt:16s} conf={score_f:.3f}  box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

        vis = draw_result(img_bgr, all_dets)
        out_path = OUT_DIR / f"{img_path.stem}_sam3.jpg"
        cv2.imwrite(str(out_path), vis)
        print(f"  saved: {out_path}\n")
        summary_lines.append(f"{img_path.name}: {len(all_dets)} detections")

    (OUT_DIR / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("Done.")


if __name__ == "__main__":
    main()
