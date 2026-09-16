#!/usr/bin/env python3
"""SAM3 multi-task open-vocab detector.

Tasks:
  - cig     : 烟盒 / 打火机 / 烟头
  - wire    : 电线绝缘层破损（局部框）
  - hazard  : 积水 / 火花 / 烟雾
  - water / spark / smoke : 单项

Examples:
  CUDA_VISIBLE_DEVICES=5 python sam3_detect_multi.py \\
    --source New_test/image-0914 --out-dir runs/detect/sam3_hazard \\
    --tasks water,spark,smoke

  CUDA_VISIBLE_DEVICES=5 python sam3_detect_multi.py \\
    --source New_test/image_0901_v2 --out-dir runs/detect/sam3_all \\
    --tasks cig,wire,hazard
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

# ---- class registry ----
# id -> meta
CLASSES: dict[int, dict] = {
    0: {
        "name": "cigarettes_butts",
        "zh": "烟头",
        "color": (0, 140, 255),
        "task": "cig",
        "prompts": [
            "cigarette butt",
            "cigarette butts",
            "cigarette filter tip",
            "smoked cigarette butt",
        ],
        "mode": "object",  # standard NMS top-k
    },
    1: {
        "name": "cig-pack",
        "zh": "烟盒",
        "color": (0, 0, 255),
        "task": "cig",
        "prompts": [
            "cigarette pack",
            "red cigarette pack",
            "gold cigarette pack",
            "cigarette box",
            "tobacco pack",
        ],
        "mode": "object",
    },
    3: {
        "name": "Lighter",
        "zh": "打火机",
        "color": (0, 220, 0),
        "task": "cig",
        "prompts": [
            "lighter",
            "cigarette lighter",
            "disposable lighter",
        ],
        "mode": "object",
    },
    10: {
        "name": "wire_insulation_damage",
        "zh": "绝缘破损",
        "color": (0, 0, 255),
        "task": "wire",
        "prompts": [
            "exposed copper wire",
            "bare copper strands",
            "exposed wires",
            "torn insulation exposing wires",
            "hole in cable insulation",
            "damaged insulation spot",
        ],
        "mode": "damage",  # prefer compact localized box
    },
    11: {
        "name": "standing_water",
        "zh": "积水",
        "color": (255, 180, 0),  # cyan-ish BGR
        "task": "hazard",
        "prompts": [
            "puddle of water",
            "standing water on floor",
            "water puddle",
            "water leak on ground",
            "wet floor puddle",
            "pool of water on pavement",
        ],
        "mode": "object",
        # 积水只保留一个最大框，避免碎小框
        "select": "largest",
        "max_boxes": 1,
    },
    12: {
        "name": "spark",
        "zh": "火花",
        "color": (0, 255, 255),  # yellow
        "task": "hazard",
        "prompts": [
            "electrical spark",
            "sparks",
            "welding spark",
            "bright spark flash",
            "electric arc spark",
        ],
        "mode": "object",
    },
    13: {
        "name": "smoke",
        "zh": "烟雾",
        "color": (160, 160, 160),
        "task": "hazard",
        "prompts": [
            "smoke",
            "smoke plume",
            "smoke in air",
            "fire smoke",
            "gray smoke",
            "white smoke",
        ],
        "mode": "object",
        "select": "largest",
        "max_boxes": 1,
    },
}

TASK_ALIASES = {
    "cig": [0, 1, 3],
    "wire": [10],
    "butts": [0],
    "pack": [1],
    "lighter": [3],
    "damage": [10],
    "wire_damage": [10],
    "hazard": [11, 12, 13],
    "water": [11],
    "puddle": [11],
    "spark": [12],
    "smoke": [13],
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


def center_score(xyxy: tuple[float, ...], w: int, h: int) -> float:
    x1, y1, x2, y2 = xyxy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = abs(cx - w / 2) / max(w / 2, 1), abs(cy - h / 2) / max(h / 2, 1)
    return max(0.0, 1.0 - 0.5 * (dx + dy))


def collect_prompt_boxes(
    processor: Sam3Processor,
    image: Image.Image,
    prompts: list[str],
    conf_thr: float,
) -> list[dict]:
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
            cands.append(
                {
                    "conf": sc,
                    "xyxy": tuple(float(v) for v in box.tolist()),
                    "prompt": prompt,
                }
            )
    return cands


def detect_object_class(
    processor: Sam3Processor,
    image: Image.Image,
    cls_id: int,
    conf_thr: float,
    max_per_class: int,
) -> list[dict]:
    meta = CLASSES[cls_id]
    cands = collect_prompt_boxes(processor, image, meta["prompts"], conf_thr)
    for c in cands:
        c["cls"] = cls_id
    kept = nms_keep(cands, iou_thr=0.5)
    limit = int(meta.get("max_boxes", max_per_class))
    select = meta.get("select", "topk")
    if select == "largest" and kept:
        # keep the single largest box (prefer big puddle over tiny edge fragments)
        def area(d: dict) -> float:
            x1, y1, x2, y2 = d["xyxy"]
            return max(0.0, x2 - x1) * max(0.0, y2 - y1)

        kept = sorted(kept, key=lambda d: (area(d), d["conf"]), reverse=True)[:limit]
    else:
        kept = kept[:limit]
    return kept


def detect_damage_class(
    processor: Sam3Processor,
    image: Image.Image,
    cls_id: int,
    conf_thr: float,
    max_boxes: int,
    max_area_ratio: float,
    min_area_ratio: float,
) -> list[dict]:
    """Prefer compact localized damage; drop near-full-image boxes."""
    meta = CLASSES[cls_id]
    w, h = image.size
    img_area = float(w * h)
    raw = collect_prompt_boxes(processor, image, meta["prompts"], conf_thr)
    ranked: list[dict] = []
    for c in raw:
        x1, y1, x2, y2 = c["xyxy"]
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(w), x2), min(float(h), y2)
        area_r = max(0.0, x2 - x1) * max(0.0, y2 - y1) / img_area
        if not (min_area_ratio <= area_r <= max_area_ratio):
            continue
        cs = center_score((x1, y1, x2, y2), w, h)
        rank = c["conf"] * (1.0 - area_r) * (0.5 + 0.5 * cs)
        ranked.append(
            {
                "cls": cls_id,
                "conf": c["conf"],
                "xyxy": (x1, y1, x2, y2),
                "prompt": c["prompt"],
                "rank": rank,
                "area_r": area_r,
            }
        )
    ranked.sort(key=lambda d: -d["rank"])
    return nms_keep(ranked, iou_thr=0.3)[:max_boxes]


def draw_dets(
    img_bgr: np.ndarray,
    dets: list[dict],
    *,
    font_scale: float = 0.6,
    banner: str | None = None,
) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]
    if banner:
        color = (0, 0, 180) if "损坏" in banner or "破损" in banner else (0, 140, 0)
        cv2.rectangle(out, (0, 0), (w, 36), color, -1)
        try:
            from vis_labels import draw_chinese_label

            out = draw_chinese_label(out, banner, (8, 4), font_size=22, color_bgr=(255, 255, 255))
        except Exception:
            cv2.putText(
                out, banner, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

    for d in dets:
        cls = d["cls"]
        meta = CLASSES[cls]
        x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
        color = meta["color"]
        label = f"{meta['zh']} {round(d['conf'] * 100)}%"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        try:
            from vis_labels import draw_chinese_label

            out = draw_chinese_label(
                out,
                label,
                (x1, max(40 if banner else 0, y1 - 28)),
                font_size=int(22 * font_scale / 0.6),
                color_bgr=color,
            )
        except Exception:
            cv2.putText(
                out,
                f"{meta['name']} {round(d['conf']*100)}%",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                2,
                cv2.LINE_AA,
            )
    return out


def parse_tasks(s: str) -> list[int]:
    ids: list[int] = []
    for tok in s.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok not in TASK_ALIASES:
            raise SystemExit(
                "unknown task '{}'. use: cig,wire,hazard,water,spark,smoke,"
                "pack,lighter,butts,damage".format(tok)
            )
        for cid in TASK_ALIASES[tok]:
            if cid not in ids:
                ids.append(cid)
    if not ids:
        raise SystemExit("no tasks selected")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAM3 multi-task: cig + wire + water/spark/smoke"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument(
        "--tasks",
        type=str,
        default="cig,wire,hazard",
        help="comma list: cig,wire,hazard,water,spark,smoke,pack,lighter,butts,damage",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--wire-conf", type=float, default=0.15, help="wire damage conf floor")
    parser.add_argument(
        "--hazard-conf",
        type=float,
        default=0.20,
        help="water/spark/smoke conf floor",
    )
    parser.add_argument("--max-per-class", type=int, default=2)
    parser.add_argument("--max-damage-boxes", type=int, default=1)
    parser.add_argument("--max-area-ratio", type=float, default=0.25)
    parser.add_argument("--min-area-ratio", type=float, default=0.008)
    parser.add_argument("--font-scale", type=float, default=0.6)
    args = parser.parse_args()

    class_ids = parse_tasks(args.tasks)
    images = iter_images(args.source)
    if not images:
        raise SystemExit(f"no images in {args.source}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"checkpoint: {args.ckpt}")
    print(f"images    : {len(images)} from {args.source}")
    print(f"classes   : {[CLASSES[c]['zh'] for c in class_ids]}")
    print(f"output    : {args.out_dir}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model(
        checkpoint_path=str(args.ckpt),
        load_from_HF=False,
        device=device,
    ).to(device)
    proc_conf = min(args.conf, args.wire_conf, args.hazard_conf)
    processor = Sam3Processor(model, device=device, confidence_threshold=proc_conf)

    hazard_ids = {11, 12, 13}

    lines: list[str] = []
    for img_path in images:
        image = Image.open(img_path).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        dets: list[dict] = []

        for cls_id in class_ids:
            mode = CLASSES[cls_id]["mode"]
            if mode == "object":
                conf_use = args.hazard_conf if cls_id in hazard_ids else args.conf
                dets.extend(
                    detect_object_class(
                        processor, image, cls_id, conf_use, args.max_per_class
                    )
                )
            else:
                dets.extend(
                    detect_damage_class(
                        processor,
                        image,
                        cls_id,
                        conf_thr=args.wire_conf,
                        max_boxes=args.max_damage_boxes,
                        max_area_ratio=args.max_area_ratio,
                        min_area_ratio=args.min_area_ratio,
                    )
                )

        object_dets = [d for d in dets if CLASSES[d["cls"]]["mode"] == "object"]
        damage_dets = [d for d in dets if CLASSES[d["cls"]]["mode"] == "damage"]
        object_dets = nms_keep(object_dets, iou_thr=0.7)
        dets = object_dets + damage_dets

        banner_parts: list[str] = []
        if 10 in class_ids:
            has_wire = any(d["cls"] == 10 for d in dets)
            banner_parts.append("绝缘损坏" if has_wire else "无绝缘破损")
        if any(c in class_ids for c in hazard_ids):
            hit = sorted({CLASSES[d["cls"]]["zh"] for d in dets if d["cls"] in hazard_ids})
            banner_parts.append("检出:" + "/".join(hit) if hit else "无积水火花烟雾")
        banner = " | ".join(banner_parts) if banner_parts else None

        print(img_path.name)
        lines.append(img_path.name)
        if banner:
            print(f"  [{banner}]")
            lines.append(f"  [{banner}]")
        if not dets:
            print("  (no detections)")
            lines.append("  (no detections)")
        else:
            for d in sorted(dets, key=lambda x: -x["conf"]):
                zh = CLASSES[d["cls"]]["zh"]
                x1, y1, x2, y2 = d["xyxy"]
                msg = (
                    f"  {zh:8s} conf={round(d['conf']*100)}% "
                    f"prompt={d['prompt']!r} "
                    f"box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})"
                )
                print(msg)
                lines.append(msg)

        vis = draw_dets(img_bgr, dets, font_scale=args.font_scale, banner=banner)
        out_path = args.out_dir / f"{img_path.stem}_sam3_multi.jpg"
        cv2.imwrite(str(out_path), vis)
        print(f"  saved: {out_path}\n")
        lines.append(f"  saved: {out_path}\n")

    (args.out_dir / "results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Done.")


if __name__ == "__main__":
    main()
