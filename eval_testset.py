#!/usr/bin/env python3
"""Evaluate YOLO pipeline on merged_dataset test split (with/without Qwen3-VL)."""

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

from predict_multiclass import DEFAULT_MODEL, apply_rule_postprocess, apply_vlm_postprocess
from vlm_arbitrate import QwenVlmClient
from vlm_gate import needs_vlm_arbitration

ROOT = Path(__file__).resolve().parent
TEST_IMG_DIR = ROOT / "merged_dataset/test/images"
TEST_LBL_DIR = ROOT / "merged_dataset/test/labels"
CLASS_NAMES = {
    0: "cigarettes_butts",
    1: "cig-pack",
    2: "cigarette",
    3: "Lighter",
}


def load_gt(label_path: Path, img_w: int, img_h: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (cls, xyxy) arrays."""
    if not label_path.exists():
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)
    cls_list: list[int] = []
    boxes: list[list[float]] = []
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        c = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:5])
        x1 = (xc - bw / 2) * img_w
        y1 = (yc - bh / 2) * img_h
        x2 = (xc + bw / 2) * img_w
        y2 = (yc + bh / 2) * img_h
        cls_list.append(c)
        boxes.append([x1, y1, x2, y2])
    if not cls_list:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)
    return np.array(cls_list, dtype=np.int64), np.array(boxes, dtype=np.float32)


def match_predictions(
    pred_cls: np.ndarray,
    gt_cls: np.ndarray,
    iou_mat: torch.Tensor,
    iouv: torch.Tensor,
) -> np.ndarray:
    """Return tp matrix (N_preds, len(iouv))."""
    n_pred = len(pred_cls)
    n_iou = len(iouv)
    correct = np.zeros((n_pred, n_iou), dtype=bool)
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


def run_pipeline(
    model: YOLO,
    img_path: Path,
    *,
    imgsz: int,
    conf: float,
    iou_thresh: float,
    device: int,
    vlm: QwenVlmClient | None,
    vlm_min_conf: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return pred_cls, pred_conf, pred_xyxy, vlm_calls."""
    result = model.predict(
        source=str(img_path),
        imgsz=imgsz,
        conf=conf,
        verbose=False,
        device=device,
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

    if result.boxes is None or len(result.boxes) == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
            vlm_calls,
        )
    return (
        result.boxes.cls.cpu().numpy().astype(np.int64),
        result.boxes.conf.cpu().numpy().astype(np.float32),
        result.boxes.xyxy.cpu().numpy().astype(np.float32),
        vlm_calls,
    )


def evaluate(
    *,
    model_path: Path,
    imgsz: int,
    conf: float,
    iou_thresh: float,
    device: int,
    use_vlm: bool,
    vlm_api: str,
    vlm_model: str,
    vlm_min_conf: float,
    limit: int | None,
    img_dir: Path = TEST_IMG_DIR,
    label_dir: Path = TEST_LBL_DIR,
) -> dict:
    images = sorted(
        p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if limit:
        images = images[:limit]

    model = YOLO(str(model_path))
    vlm: QwenVlmClient | None = None
    if use_vlm:
        vlm = QwenVlmClient(api_base=vlm_api, model=vlm_model)
        if not vlm.health_ok():
            raise RuntimeError(f"Qwen3-VL not ready at {vlm_api}")

    metrics = DetMetrics(names=CLASS_NAMES)
    iouv = torch.linspace(0.5, 0.95, 10)
    total_vlm_calls = 0
    t0 = time.perf_counter()

    for img_idx, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt_cls, gt_boxes = load_gt(label_dir / f"{img_path.stem}.txt", w, h)

        pred_cls, pred_conf, pred_boxes, vlm_calls = run_pipeline(
            model,
            img_path,
            imgsz=imgsz,
            conf=conf,
            iou_thresh=iou_thresh,
            device=device,
            vlm=vlm,
            vlm_min_conf=vlm_min_conf,
        )
        total_vlm_calls += vlm_calls

        if len(pred_cls) == 0 or len(gt_cls) == 0:
            tp = np.zeros((len(pred_cls), len(iouv)), dtype=bool)
        else:
            iou_mat = box_iou(torch.from_numpy(gt_boxes).float(), torch.from_numpy(pred_boxes).float())
            tp = match_predictions(pred_cls, gt_cls, iou_mat, iouv)

        metrics.update_stats(
            {
                "tp": tp,
                "conf": pred_conf,
                "pred_cls": pred_cls,
                "target_cls": gt_cls,
                "target_img": np.unique(gt_cls) if len(gt_cls) else np.array([], dtype=np.int64),
                "im_name": img_path.name,
            }
        )

        if (img_idx + 1) % 50 == 0:
            print(f"  processed {img_idx + 1}/{len(images)} images...")

    elapsed = time.perf_counter() - t0
    metrics.process(plot=False)
    p, r, map50, map5095 = metrics.mean_results()

    per_class: dict[str, dict] = {}
    for i, c in enumerate(metrics.ap_class_index):
        per_class[CLASS_NAMES[int(c)]] = {
            "images": int(metrics.nt_per_image[c]),
            "instances": int(metrics.nt_per_class[c]),
            "precision": float(metrics.class_result(i)[0]),
            "recall": float(metrics.class_result(i)[1]),
            "mAP50": float(metrics.class_result(i)[2]),
            "mAP50-95": float(metrics.class_result(i)[3]),
        }

    return {
        "pipeline": "yolo+rules+vlm" if use_vlm else "yolo+rules",
        "images": len(images),
        "instances": int(metrics.nt_per_class.sum()) if metrics.nt_per_class is not None else 0,
        "precision": float(p),
        "recall": float(r),
        "mAP50": float(map50),
        "mAP50-95": float(map5095),
        "per_class": per_class,
        "vlm_calls": total_vlm_calls,
        "elapsed_sec": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate test set with/without VLM")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou-thresh", type=float, default=0.3)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--vlm", action="store_true")
    parser.add_argument("--vlm-api", type=str, default="http://127.0.0.1:8001/v1")
    parser.add_argument("--vlm-model", type=str, default="qwen3-vl-idc")
    parser.add_argument("--vlm-min-conf", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None, help="limit images for quick test")
    parser.add_argument("--img-dir", type=Path, default=TEST_IMG_DIR)
    parser.add_argument("--label-dir", type=Path, default=TEST_LBL_DIR)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/detect/eval_testset")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tag = "with_vlm" if args.vlm else "no_vlm"
    print(f"Evaluating test set [{tag}] ...")

    result = evaluate(
        model_path=args.model,
        imgsz=args.imgsz,
        conf=args.conf,
        iou_thresh=args.iou_thresh,
        device=args.device,
        use_vlm=args.vlm,
        vlm_api=args.vlm_api,
        vlm_model=args.vlm_model,
        vlm_min_conf=args.vlm_min_conf,
        limit=args.limit,
        img_dir=args.img_dir,
        label_dir=args.label_dir,
    )

    out_json = args.out / f"metrics_{tag}.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
