#!/usr/bin/env python3
"""SAM3 text-prompt inference with Chinese percent labels."""

from __future__ import annotations

import argparse
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

from vis_postprocess import VisPostprocessConfig, draw_detection_image, format_confidence
from vis_labels import class_name_zh

ROOT = Path(__file__).resolve().parent
CKPT = Path("/data/zhangzhe/do-as-i-do/reconstruction/checkpoints/sam3/sam3.pt")
DEVICE = "cuda"
CONF = 0.25

PACK_PROMPTS = [
    "red cigarette pack",
    "gold cigarette pack",
    "cigarette pack",
    "cigarette box",
    "tobacco pack",
]

# SAM3 open-vocab prompt -> YOLO class id for visualization
PROMPT_TO_CLS = {
    "lighter": 3,
    "red cigarette pack": 1,
    "gold cigarette pack": 1,
    "cigarette pack": 1,
    "cigarette box": 1,
    "tobacco pack": 1,
}

YOLO_NAMES = {
    0: "cigarettes_butts",
    1: "cig-pack",
    2: "cigarette",
    3: "Lighter",
}


def iter_images(source: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if source.is_file():
        return [source]
    return sorted(p for p in source.iterdir() if p.suffix.lower() in exts)


def best_pack_detection(processor: Sam3Processor, image: Image.Image) -> tuple[str, float, tuple] | None:
    best = None
    for prompt in PACK_PROMPTS:
        processor.reset_all_prompts({})
        state = processor.set_image(image)
        state = processor.set_text_prompt(prompt=prompt, state=state)
        boxes = state.get("boxes")
        scores = state.get("scores")
        if boxes is None or len(boxes) == 0:
            continue
        idx = int(torch.argmax(scores))
        score = float(scores[idx])
        box = tuple(float(v) for v in boxes[idx].tolist())
        if best is None or score > best[1]:
            best = (prompt, score, box)
    return best


def sam3_detections(processor: Sam3Processor, image: Image.Image) -> list[dict]:
    """Run SAM3 lighter + best pack prompt; return vis-ready box dicts."""
    dets: list[dict] = []

    processor.reset_all_prompts({})
    state = processor.set_image(image)
    state = processor.set_text_prompt(prompt="lighter", state=state)
    if state.get("boxes") is not None and len(state["boxes"]):
        for box, score in zip(state["boxes"], state["scores"]):
            dets.append(
                {
                    "prompt": "lighter",
                    "cls": PROMPT_TO_CLS["lighter"],
                    "conf": float(score),
                    "xyxy": tuple(float(v) for v in box.tolist()),
                }
            )

    pack = best_pack_detection(processor, image)
    if pack:
        prompt, score, box = pack
        dets.append(
            {
                "prompt": prompt,
                "cls": PROMPT_TO_CLS.get(prompt, 1),
                "conf": score,
                "xyxy": box,
            }
        )
    return dets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="image file or directory")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=CONF)
    parser.add_argument("--bold", action="store_true", help="bold Chinese labels")
    parser.add_argument(
        "--font-scale",
        type=float,
        default=0.5,
        help="label font scale (default 0.5)",
    )
    args = parser.parse_args()

    images = iter_images(args.source)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    vis_cfg = VisPostprocessConfig(
        enabled=True,
        use_zh=True,
        percent_conf=True,
        bold_zh=args.bold,
        font_scale=args.font_scale,
    )

    print(f"checkpoint: {CKPT}")
    print(f"images    : {len(images)} from {args.source}")
    print(f"output    : {args.out_dir}")
    print(f"labels    : Chinese, percent, bold={vis_cfg.bold_zh}\n")

    model = build_sam3_image_model(
        checkpoint_path=str(CKPT),
        load_from_HF=False,
        device=DEVICE,
    ).to(DEVICE)
    processor = Sam3Processor(model, device=DEVICE, confidence_threshold=args.conf)

    lines: list[str] = []
    for img_path in images:
        image = Image.open(img_path).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        all_dets = sam3_detections(processor, image)

        lines.append(img_path.name)
        print(img_path.name)
        if not all_dets:
            msg = "  (no detections)"
            print(msg)
            lines.append(msg)
        else:
            for det in all_dets:
                cls = det["cls"]
                name = class_name_zh(cls, YOLO_NAMES)
                conf_s = format_confidence(det["conf"], as_percent=True)
                x1, y1, x2, y2 = det["xyxy"]
                msg = (
                    f"  {name:6s} ({det['prompt']}) conf={conf_s} "
                    f"box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})"
                )
                print(msg)
                lines.append(msg)

        vis_boxes = [{"cls": d["cls"], "conf": d["conf"], "xyxy": d["xyxy"]} for d in all_dets]
        out_img = draw_detection_image(img_bgr, vis_boxes, YOLO_NAMES, cfg=vis_cfg)
        out_path = args.out_dir / f"{img_path.stem}_sam3.jpg"
        cv2.imwrite(str(out_path), out_img)
        print(f"  saved: {out_path}\n")
        lines.append("")

    (args.out_dir / "results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Done.")


if __name__ == "__main__":
    main()
