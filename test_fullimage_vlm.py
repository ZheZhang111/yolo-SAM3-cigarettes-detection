#!/usr/bin/env python3
"""Standalone trial: full-image VLM vs YOLO vs GT on one hard-case image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from postprocess import CLASS_NAMES
from predict_multiclass import DEFAULT_MODEL, apply_rule_postprocess
from vlm_fullimage import QwenFullImageClient

ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE = ROOT / "New_test/image_0828-v2/3a901419830a64008f4202477923ee68.jpg"
DEFAULT_LABEL = ROOT / "New_test/image_0828-v2/labels/3a901419830a64008f4202477923ee68.txt"

COLORS = {
    "gt": (255, 255, 0),      # cyan-ish yellow
    "yolo": (0, 0, 255),      # red
    "vlm": (0, 220, 0),       # green
}


def load_yolo_labels(path: Path, w: int, h: int) -> list[tuple[int, np.ndarray]]:
    out: list[tuple[int, np.ndarray]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        c = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:5])
        x1 = (xc - bw / 2) * w
        y1 = (yc - bh / 2) * h
        x2 = (xc + bw / 2) * w
        y2 = (yc + bh / 2) * h
        out.append((c, np.array([x1, y1, x2, y2], dtype=np.float32)))
    return out


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
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


def match_boxes(
    preds: list[tuple[int, np.ndarray, float]],
    gts: list[tuple[int, np.ndarray]],
    iou_thr: float = 0.5,
) -> dict:
    """Greedy match preds to GT; report class-correct hits."""
    used_gt: set[int] = set()
    matches: list[dict] = []
    for pi, (p_cls, p_box, p_conf) in enumerate(preds):
        best_i, best_iou = -1, 0.0
        for gi, (g_cls, g_box) in enumerate(gts):
            if gi in used_gt:
                continue
            iou = box_iou(p_box, g_box)
            if iou > best_iou:
                best_iou = iou
                best_i = gi
        if best_i >= 0 and best_iou >= iou_thr:
            used_gt.add(best_i)
            g_cls = gts[best_i][0]
            matches.append(
                {
                    "pred_idx": pi,
                    "gt_idx": best_i,
                    "iou": round(float(best_iou), 3),
                    "pred_class": CLASS_NAMES[p_cls],
                    "gt_class": CLASS_NAMES[g_cls],
                    "class_ok": p_cls == g_cls,
                }
            )
        else:
            matches.append(
                {
                    "pred_idx": pi,
                    "gt_idx": None,
                    "iou": round(float(best_iou), 3),
                    "pred_class": CLASS_NAMES[p_cls],
                    "gt_class": None,
                    "class_ok": False,
                }
            )
    missed_gt = [i for i in range(len(gts)) if i not in used_gt]
    return {
        "matches": matches,
        "missed_gt": missed_gt,
        "pred_count": len(preds),
        "gt_count": len(gts),
        "matched_class_correct": sum(1 for m in matches if m.get("class_ok")),
    }


def draw_boxes(
    img: np.ndarray,
    boxes: list[tuple[int, np.ndarray, float]],
    prefix: str,
    color: tuple[int, int, int],
) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    lw = max(3, int(max(h, w) * 0.004))
    for i, (cls, box, conf) in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, lw)
        label = f"{prefix}{CLASS_NAMES[cls]} {conf:.2f}"
        cv2.putText(
            out, label, (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
        )
    return out


def class_counts(boxes: list[tuple[int, ...]]) -> dict[str, int]:
    from collections import Counter
    c = Counter(CLASS_NAMES[b[0]] for b in boxes)
    return dict(c)


def verdict(yolo_stat: dict, vlm_stat: dict, yolo_preds, vlm_preds, gt_boxes) -> str:
    gt_n = yolo_stat["gt_count"]
    yolo_ok = yolo_stat["matched_class_correct"]
    vlm_ok = vlm_stat["matched_class_correct"]
    gt_classes = class_counts(gt_boxes)
    yolo_classes = class_counts(yolo_preds)
    vlm_classes = class_counts(vlm_preds)

    lines = [
        f"GT classes: {gt_classes}",
        f"YOLO classes: {yolo_classes} | bbox+class match @IoU0.5: {yolo_ok}/{gt_n}",
        f"VLM classes: {vlm_classes} | bbox+class match @IoU0.5: {vlm_ok}/{gt_n}",
    ]

    if vlm_classes == gt_classes and vlm_ok >= gt_n and vlm_ok > yolo_ok:
        lines.append("PASS — full-image VLM matches GT classes and bboxes; worth integrating.")
    elif vlm_classes == gt_classes and vlm_ok > yolo_ok:
        lines.append("PARTIAL — VLM class count/types correct, bboxes improved but not perfect.")
    elif vlm_classes == gt_classes:
        lines.append(
            "PARTIAL — VLM class count/types correct (1 pack + 1 lighter), "
            "but bbox IoU poor; cannot replace YOLO boxes yet."
        )
    elif vlm_ok > yolo_ok:
        lines.append("PARTIAL — VLM slightly better bbox match, class set still wrong.")
    else:
        lines.append("FAIL — full-image VLM not better than YOLO; do not integrate yet.")
    return "\n  ".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trial full-image VLM on one image")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--label", type=Path, default=DEFAULT_LABEL)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/detect/fullimage_vlm_test")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(args.image))
    if img is None:
        raise FileNotFoundError(args.image)
    h, w = img.shape[:2]

    # GT
    gt_boxes = load_yolo_labels(args.label, w, h)
    gt_preds = [(c, b, 1.0) for c, b in gt_boxes]

    # YOLO
    model = YOLO(str(args.model))
    result = model.predict(source=str(args.image), imgsz=args.imgsz, conf=args.conf, verbose=False, device=args.device)[0]
    result = apply_rule_postprocess(result, iou_thresh=0.3)
    yolo_preds: list[tuple[int, np.ndarray, float]] = []
    if result.boxes is not None and len(result.boxes) > 0:
        for b in result.boxes:
            yolo_preds.append(
                (int(b.cls), b.xyxy[0].cpu().numpy().astype(np.float32), float(b.conf))
            )

    # Full-image VLM
    client = QwenFullImageClient()
    if not client.health_ok():
        raise RuntimeError("Qwen3-VL service not ready at http://127.0.0.1:8001")
    print("Calling full-image VLM (FULLIMAGE_PROMPT only)...")
    vlm_res = client.detect_full_image(img)
    vlm_preds: list[tuple[int, np.ndarray, float]] = []
    for obj in vlm_res.objects:
        if obj.class_id is None:
            continue
        vlm_preds.append((obj.class_id, obj.xyxy(w, h), obj.confidence))

    yolo_stat = match_boxes(yolo_preds, gt_boxes)
    vlm_stat = match_boxes(vlm_preds, gt_boxes)
    v = verdict(yolo_stat, vlm_stat, yolo_preds, vlm_preds, gt_boxes)

    report = {
        "image": str(args.image),
        "gt": [{"class": CLASS_NAMES[c], "xyxy": [float(x) for x in b]} for c, b in gt_boxes],
        "yolo": {
            "boxes": [
                {"class": CLASS_NAMES[c], "conf": conf, "xyxy": [float(x) for x in b]}
                for c, b, conf in yolo_preds
            ],
            "match": yolo_stat,
        },
        "vlm_fullimage": {
            "summary": vlm_res.summary,
            "boxes": [
                {"class": CLASS_NAMES[c], "conf": conf, "xyxy": [float(x) for x in b]}
                for c, b, conf in vlm_preds
            ],
            "match": vlm_stat,
            "raw": vlm_res.raw,
        },
        "verdict": v,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Visualizations
    cv2.imwrite(str(args.out / "00_gt.jpg"), draw_boxes(img, gt_preds, "GT:", COLORS["gt"]))
    cv2.imwrite(str(args.out / "01_yolo.jpg"), draw_boxes(img, yolo_preds, "YOLO:", COLORS["yolo"]))
    cv2.imwrite(str(args.out / "02_vlm_full.jpg"), draw_boxes(img, vlm_preds, "VLM:", COLORS["vlm"]))

    # Side-by-side
    panels = [
        draw_boxes(img, gt_preds, "GT:", COLORS["gt"]),
        draw_boxes(img, yolo_preds, "YOLO:", COLORS["yolo"]),
        draw_boxes(img, vlm_preds, "VLM:", COLORS["vlm"]),
    ]
    scale = 640 / max(h, w)
    panels = [cv2.resize(p, (int(w * scale), int(h * scale))) for p in panels]
    cv2.imwrite(str(args.out / "03_compare.jpg"), cv2.hconcat(panels))

    print("\n=== Ground Truth ===")
    for c, b in gt_boxes:
        print(f"  {CLASS_NAMES[c]:15s} {b.astype(int).tolist()}")

    print("\n=== YOLO ===")
    for c, b, conf in yolo_preds:
        print(f"  {CLASS_NAMES[c]:15s} conf={conf:.3f}  {b.astype(int).tolist()}")
    print(f"  match: {yolo_stat['matched_class_correct']}/{yolo_stat['gt_count']} class-correct @ IoU0.5")

    print("\n=== Full-image VLM ===")
    print(f"  summary: {vlm_res.summary}")
    for c, b, conf in vlm_preds:
        print(f"  {CLASS_NAMES[c]:15s} conf={conf:.3f}  {b.astype(int).tolist()}")
    print(f"  match: {vlm_stat['matched_class_correct']}/{vlm_stat['gt_count']} class-correct @ IoU0.5")

    print(f"\n=== VERDICT ===\n  {v}\n")
    print(f"Saved to {args.out}/")


if __name__ == "__main__":
    main()
