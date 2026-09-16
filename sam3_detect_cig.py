#!/usr/bin/env python3
"""Standalone SAM3 open-vocab detection: 烟盒 / 打火机 / 烟头.

Uses Meta SAM3 text prompts (no YOLO). One concept family per class;
keeps top box(es) per class after simple NMS.

Example:
  conda activate sam3
  CUDA_VISIBLE_DEVICES=5 python sam3_detect_cig.py \\
    --source New_test/image_0901_v2 \\
    --out-dir runs/detect/New_test/sam3_cig_demo
"""

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

ROOT = Path(__file__).resolve().parent
DEFAULT_CKPT = Path(
    "/data/zhangzhe/do-as-i-do/reconstruction/checkpoints/sam3/sam3.pt"
)

# class_id aligned with YOLO multiclass for visualization
CLASS_NAMES = {
    0: "cigarettes_butts",
    1: "cig-pack",
    3: "Lighter",
}
CLASS_NAMES_ZH = {
    0: "烟头",
    1: "烟盒",
    3: "打火机",
}
# BGR colors
CLASS_COLORS = {
    0: (0, 140, 255),
    1: (0, 0, 255),
    3: (0, 220, 0),
}

# Multiple English phrases per class (open-vocab is prompt-sensitive)
CLASS_PROMPTS: dict[int, list[str]] = {
    3: [
        "lighter",
        "cigarette lighter",
        "disposable lighter",
    ],
    1: [
        "cigarette pack",
        "red cigarette pack",
        "gold cigarette pack",
        "cigarette box",
        "tobacco pack",
    ],
    0: [
        "cigarette butt",
        "cigarette butts",
        "cigarette filter tip",
        "smoked cigarette butt",
    ],
}


def iter_images(source: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if source.is_file():
        return [source]
    return sorted(p for p in source.iterdir() if p.suffix.lower() in exts)


def box_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-6)


def nms_keep(dets: list[dict], iou_thr: float = 0.5) -> list[dict]:
    dets = sorted(dets, key=lambda d: -d["conf"])
    keep: list[dict] = []
    for d in dets:
        if all(box_iou(d["xyxy"], k["xyxy"]) < iou_thr for k in keep):
            keep.append(d)
    return keep


def detect_class(
    processor: Sam3Processor,
    image: Image.Image,
    cls_id: int,
    prompts: list[str],
    conf_thr: float,
    max_per_class: int,
) -> list[dict]:
    """Run all prompts for one class; keep top boxes after NMS."""
    cands: list[dict] = []
    for prompt in prompts:
        processor.reset_all_prompts({})
        state = processor.set_image(image)
        state = processor.set_text_prompt(prompt=prompt, state=state)
        boxes = state.get("boxes")
        scores = state.get("scores")
        if boxes is None or len(boxes) == 0:
            continue
        for box, score in zip(boxes, scores):
            sc = float(score)
            if sc < conf_thr:
                continue
            xyxy = tuple(float(v) for v in box.tolist())
            cands.append(
                {
                    "cls": cls_id,
                    "conf": sc,
                    "xyxy": xyxy,
                    "prompt": prompt,
                }
            )
    kept = nms_keep(cands, iou_thr=0.5)
    return kept[:max_per_class]


def draw_dets(img_bgr: np.ndarray, dets: list[dict], font_scale: float = 0.6) -> np.ndarray:
    out = img_bgr.copy()
    for d in dets:
        cls = d["cls"]
        x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        name = CLASS_NAMES_ZH.get(cls, CLASS_NAMES.get(cls, str(cls)))
        label = f"{name} {round(d['conf'] * 100)}%"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        # simple ASCII-safe fallback if Chinese font missing: use PIL when available
        try:
            from vis_labels import draw_chinese_label

            out = draw_chinese_label(
                out, label, (x1, max(0, y1 - 28)), font_size=int(22 * font_scale / 0.6),
                color_bgr=color,
            )
        except Exception:
            cv2.putText(
                out, f"{CLASS_NAMES[cls]} {round(d['conf']*100)}%",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA,
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM3: pack / lighter / butts")
    parser.add_argument("--source", type=Path, required=True, help="image or directory")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-per-class", type=int, default=3, help="max boxes per class")
    parser.add_argument("--font-scale", type=float, default=0.6)
    parser.add_argument(
        "--classes",
        type=str,
        default="pack,lighter,butts",
        help="subset: pack,lighter,butts (comma-separated)",
    )
    args = parser.parse_args()

    name_to_id = {"butts": 0, "pack": 1, "lighter": 3}
    selected = []
    for tok in args.classes.split(","):
        tok = tok.strip().lower()
        if tok not in name_to_id:
            raise SystemExit(f"unknown class '{tok}', use pack/lighter/butts")
        selected.append(name_to_id[tok])

    images = iter_images(args.source)
    if not images:
        raise SystemExit(f"no images in {args.source}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"checkpoint: {args.ckpt}")
    print(f"images    : {len(images)} from {args.source}")
    print(f"classes   : {[CLASS_NAMES_ZH[c] for c in selected]}")
    print(f"output    : {args.out_dir}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model(
        checkpoint_path=str(args.ckpt),
        load_from_HF=False,
        device=device,
    ).to(device)
    processor = Sam3Processor(model, device=device, confidence_threshold=args.conf)

    lines: list[str] = []
    for img_path in images:
        image = Image.open(img_path).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        dets: list[dict] = []
        for cls_id in selected:
            dets.extend(
                detect_class(
                    processor,
                    image,
                    cls_id,
                    CLASS_PROMPTS[cls_id],
                    conf_thr=args.conf,
                    max_per_class=args.max_per_class,
                )
            )
        # cross-class NMS: if pack and lighter heavily overlap, keep higher conf
        dets = nms_keep(dets, iou_thr=0.7)

        print(img_path.name)
        lines.append(img_path.name)
        if not dets:
            print("  (no detections)")
            lines.append("  (no detections)")
        else:
            for d in sorted(dets, key=lambda x: -x["conf"]):
                zh = CLASS_NAMES_ZH[d["cls"]]
                x1, y1, x2, y2 = d["xyxy"]
                msg = (
                    f"  {zh:6s} conf={round(d['conf']*100)}% "
                    f"prompt={d['prompt']!r} "
                    f"box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})"
                )
                print(msg)
                lines.append(msg)

        vis = draw_dets(img_bgr, dets, font_scale=args.font_scale)
        out_path = args.out_dir / f"{img_path.stem}_sam3_cig.jpg"
        cv2.imwrite(str(out_path), vis)
        print(f"  saved: {out_path}\n")
        lines.append(f"  saved: {out_path}\n")

    (args.out_dir / "results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Done.")


if __name__ == "__main__":
    main()
