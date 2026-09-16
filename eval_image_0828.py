#!/usr/bin/env python3
"""Benchmark image_0828 (5 hard cases): no-VLM vs VLM + per-image latency."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics, box_iou

from add_newtest_to_train import ANNOTATIONS
from predict_multiclass import DEFAULT_MODEL, apply_rule_postprocess, apply_vlm_postprocess
from vlm_arbitrate import QwenVlmClient
from vlm_gate import needs_vlm_arbitration

ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT / "New_test/image_0828"
LBL_DIR = ROOT / "New_test/image_0828/labels"
CLASS_NAMES = {0: "cigarettes_butts", 1: "cig-pack", 2: "cigarette", 3: "Lighter"}


def export_gt_labels() -> None:
    LBL_DIR.mkdir(parents=True, exist_ok=True)
    for fname, boxes in ANNOTATIONS.items():
        lines = [f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for cid, xc, yc, w, h in boxes]
        (LBL_DIR / f"{Path(fname).stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_gt(label_path: Path, img_w: int, img_h: int) -> tuple[np.ndarray, np.ndarray]:
    if not label_path.exists():
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)
    cls_list: list[int] = []
    boxes: list[list[float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        c = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:5])
        cls_list.append(c)
        boxes.append([
            (xc - bw / 2) * img_w, (yc - bh / 2) * img_h,
            (xc + bw / 2) * img_w, (yc + bh / 2) * img_h,
        ])
    if not cls_list:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)
    return np.array(cls_list, dtype=np.int64), np.array(boxes, dtype=np.float32)


def match_predictions(pred_cls, gt_cls, iou_mat, iouv) -> np.ndarray:
    n_pred = len(pred_cls)
    correct = np.zeros((n_pred, len(iouv)), dtype=bool)
    if n_pred == 0 or len(gt_cls) == 0:
        return correct
    correct_class = gt_cls[:, None] == pred_cls[None, :]
    iou = (iou_mat * torch.from_numpy(correct_class).to(iou_mat.device)).cpu().numpy()
    for i, thr in enumerate(iouv.cpu().tolist()):
        matches = np.nonzero(iou >= thr)
        matches = np.array(matches).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


def run_timed(
    model: YOLO,
    img_path: Path,
    *,
    imgsz: int,
    conf: float,
    iou_thresh: float,
    device: int,
    vlm: QwenVlmClient | None,
    vlm_min_conf: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    t0 = time.perf_counter()
    result = model.predict(
        source=str(img_path), imgsz=imgsz, conf=conf, verbose=False, device=device,
    )[0]
    result = apply_rule_postprocess(result, iou_thresh=iou_thresh)
    vlm_calls = 0
    if vlm is not None and result.boxes is not None and len(result.boxes) > 0:
        cls_ids = result.boxes.cls.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        vlm_calls = len(needs_vlm_arbitration(cls_ids, confs, boxes_xyxy))
        result, _ = apply_vlm_postprocess(result, vlm, vlm_min_conf=vlm_min_conf)
        result = apply_rule_postprocess(result, iou_thresh=iou_thresh)
    elapsed = time.perf_counter() - t0

    if result.boxes is None or len(result.boxes) == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
            vlm_calls,
            elapsed,
        )
    return (
        result.boxes.cls.cpu().numpy().astype(np.int64),
        result.boxes.conf.cpu().numpy().astype(np.float32),
        result.boxes.xyxy.cpu().numpy().astype(np.float32),
        vlm_calls,
        elapsed,
    )


def per_image_class_match(pred_cls, gt_cls, pred_boxes, gt_boxes, iou_thr=0.5) -> dict:
    """Greedy IoU match; return class-correct count and details."""
    if len(pred_cls) == 0:
        return {"matched": 0, "gt_n": len(gt_cls), "pred_n": 0, "details": []}
    if len(gt_cls) == 0:
        return {"matched": 0, "gt_n": 0, "pred_n": len(pred_cls), "details": []}
    iou_mat = box_iou(torch.from_numpy(gt_boxes).float(), torch.from_numpy(pred_boxes).float()).numpy()
    used_gt: set[int] = set()
    details: list[dict] = []
    matched = 0
    for pi in range(len(pred_cls)):
        best_gi, best_iou = -1, 0.0
        for gi in range(len(gt_cls)):
            if gi in used_gt:
                continue
            if pred_cls[pi] != gt_cls[gi]:
                continue
            if iou_mat[gi, pi] > best_iou:
                best_iou = iou_mat[gi, pi]
                best_gi = gi
        ok = best_gi >= 0 and best_iou >= iou_thr
        if ok:
            used_gt.add(best_gi)
            matched += 1
        details.append({
            "pred": CLASS_NAMES[int(pred_cls[pi])],
            "gt_match": CLASS_NAMES[int(gt_cls[best_gi])] if best_gi >= 0 else None,
            "iou": round(float(best_iou), 3),
            "class_ok": ok,
        })
    return {
        "matched": matched,
        "gt_n": len(gt_cls),
        "pred_n": len(pred_cls),
        "missed_gt": len(gt_cls) - len(used_gt),
        "details": details,
    }


def evaluate_split(*, use_vlm: bool, model_path, imgsz, conf, iou_thresh, device, vlm_min_conf) -> dict:
    export_gt_labels()
    images = sorted(p for p in IMG_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    model = YOLO(str(model_path))
    vlm = None
    if use_vlm:
        vlm = QwenVlmClient()
        if not vlm.health_ok():
            raise RuntimeError("Qwen3-VL not ready")

    metrics = DetMetrics(names=CLASS_NAMES)
    iouv = torch.linspace(0.5, 0.95, 10)
    per_image: list[dict] = []
    total_vlm_calls = 0
    total_time = 0.0

    for img_path in images:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        gt_cls, gt_boxes = load_gt(LBL_DIR / f"{img_path.stem}.txt", w, h)
        pred_cls, pred_conf, pred_boxes, vlm_calls, elapsed = run_timed(
            model, img_path,
            imgsz=imgsz, conf=conf, iou_thresh=iou_thresh, device=device,
            vlm=vlm, vlm_min_conf=vlm_min_conf,
        )
        total_vlm_calls += vlm_calls
        total_time += elapsed

        if len(pred_cls) == 0 or len(gt_cls) == 0:
            tp = np.zeros((len(pred_cls), len(iouv)), dtype=bool)
        else:
            iou_mat = box_iou(torch.from_numpy(gt_boxes).float(), torch.from_numpy(pred_boxes).float())
            tp = match_predictions(pred_cls, gt_cls, iou_mat, iouv)

        metrics.update_stats({
            "tp": tp, "conf": pred_conf, "pred_cls": pred_cls,
            "target_cls": gt_cls,
            "target_img": np.unique(gt_cls) if len(gt_cls) else np.array([], dtype=np.int64),
            "im_name": img_path.name,
        })

        pm = per_image_class_match(pred_cls, gt_cls, pred_boxes, gt_boxes)
        pred_str = ", ".join(
            f"{CLASS_NAMES[int(c)]}({conf:.2f})" for c, conf in zip(pred_cls, pred_conf)
        ) or "(none)"
        gt_str = ", ".join(f"{CLASS_NAMES[int(c)]}" for c in gt_cls) or "(none)"

        per_image.append({
            "image": img_path.name,
            "gt": gt_str,
            "pred": pred_str,
            "time_sec": round(elapsed, 3),
            "vlm_calls": vlm_calls,
            "class_match_iou50": f"{pm['matched']}/{pm['gt_n']}",
            "pred_n": pm["pred_n"],
            "missed_gt": pm["missed_gt"],
        })

    metrics.process(plot=False)
    p, r, map50, map5095 = metrics.mean_results()
    return {
        "pipeline": "yolo+rules+vlm" if use_vlm else "yolo+rules",
        "images": len(images),
        "instances": int(metrics.nt_per_class.sum()) if metrics.nt_per_class is not None else 0,
        "precision": float(p),
        "recall": float(r),
        "mAP50": float(map50),
        "mAP50-95": float(map5095),
        "total_time_sec": round(total_time, 3),
        "avg_time_sec": round(total_time / len(images), 3) if images else 0,
        "vlm_calls": total_vlm_calls,
        "per_class": {
            CLASS_NAMES[int(c)]: {
                "instances": int(metrics.nt_per_class[c]),
                "mAP50": float(metrics.class_result(i)[2]),
            }
            for i, c in enumerate(metrics.ap_class_index)
        },
        "per_image": per_image,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou-thresh", type=float, default=0.3)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--vlm-min-conf", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/detect/eval_image_0828")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    kw = dict(
        model_path=args.model, imgsz=args.imgsz, conf=args.conf,
        iou_thresh=args.iou_thresh, device=args.device, vlm_min_conf=args.vlm_min_conf,
    )
    print("Evaluating no-VLM ...")
    no_vlm = evaluate_split(use_vlm=False, **kw)
    print("Evaluating with-VLM ...")
    with_vlm = evaluate_split(use_vlm=True, **kw)

    report = {"no_vlm": no_vlm, "with_vlm": with_vlm}
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Overall ===")
    print(f"{'':12} {'no VLM':>12} {'with VLM':>12}")
    for k in ("mAP50", "mAP50-95", "precision", "recall", "avg_time_sec", "vlm_calls"):
        print(f"{k:12} {no_vlm[k]:>12.3f} {with_vlm[k]:>12.3f}")

    print("\n=== Per-image time (sec) ===")
    print(f"{'image':40} {'no VLM':>8} {'+VLM':>8} {'VLM calls':>10}")
    for a, b in zip(no_vlm["per_image"], with_vlm["per_image"]):
        print(f"{a['image'][:38]:40} {a['time_sec']:>8.3f} {b['time_sec']:>8.3f} {b['vlm_calls']:>10}")

    print("\n=== Per-image detection ===")
    for a, b in zip(no_vlm["per_image"], with_vlm["per_image"]):
        print(f"\n{a['image']}")
        print(f"  GT:     {a['gt']}")
        print(f"  no VLM: {a['pred']}  match={a['class_match_iou50']}")
        print(f"  +VLM:   {b['pred']}  match={b['class_match_iou50']}")

    print(f"\nSaved: {args.out}/metrics.json")


if __name__ == "__main__":
    main()
