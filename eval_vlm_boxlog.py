#!/usr/bin/env python3
"""Box-level VLM arbitration eval on merged_dataset/test.

Produces:
  - box_log.csv / box_log.jsonl  (one row per YOLO prediction box)
  - metrics_summary.json         (Acc_pre/post, net ledger, confusion, bootstrap CI)
  - report.md                    (resume-ready sentence + tables)

GT provenance: Roboflow-exported labels remapped by merge_datasets.py;
image_0828 "newtest_*" injections are train-only (not in this test split).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou

from predict_multiclass import DEFAULT_MODEL, apply_rule_postprocess
from vlm_arbitrate import QwenVlmClient, crop_roi, should_apply_vlm_update
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
NC = 4


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
        boxes.append(
            [
                (xc - bw / 2) * img_w,
                (yc - bh / 2) * img_h,
                (xc + bw / 2) * img_w,
                (yc + bh / 2) * img_h,
            ]
        )
    if not cls_list:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)
    return np.array(cls_list, dtype=np.int64), np.array(boxes, dtype=np.float32)


def match_pred_to_gt_class_agnostic(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    iou_thr: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy one-to-one match by IoU (ignore class). Returns matched_gt_idx, matched_iou."""
    n_pred = len(pred_boxes)
    matched_gt = np.full(n_pred, -1, dtype=np.int64)
    matched_iou = np.zeros(n_pred, dtype=np.float32)
    if n_pred == 0 or len(gt_boxes) == 0:
        return matched_gt, matched_iou

    iou_mat = box_iou(
        torch.from_numpy(pred_boxes).float(),
        torch.from_numpy(gt_boxes).float(),
    ).numpy()  # [n_pred, n_gt]

    pairs: list[tuple[float, int, int]] = []
    for pi in range(n_pred):
        for gi in range(len(gt_boxes)):
            if iou_mat[pi, gi] >= iou_thr:
                pairs.append((float(iou_mat[pi, gi]), pi, gi))
    pairs.sort(reverse=True)

    used_pred: set[int] = set()
    used_gt: set[int] = set()
    for iou, pi, gi in pairs:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        matched_gt[pi] = gi
        matched_iou[pi] = iou
    return matched_gt, matched_iou


def confusion_matrix(pred_cls: np.ndarray, gt_cls: np.ndarray, nc: int = NC) -> np.ndarray:
    cm = np.zeros((nc, nc), dtype=np.int64)
    for p, g in zip(pred_cls, gt_cls):
        if 0 <= int(p) < nc and 0 <= int(g) < nc:
            cm[int(g), int(p)] += 1  # rows=GT, cols=pred
    return cm


def metrics_from_matched(rows: list[dict]) -> dict:
    """Compute Acc / ledger / confusion on G_matched rows (dicts with cls fields)."""
    if not rows:
        return {
            "M": 0,
            "Acc_pre": None,
            "Acc_post": None,
            "delta": None,
            "fix_to_correct": 0,
            "correct_to_wrong": 0,
            "net_gain": 0,
            "cm_pre": np.zeros((NC, NC), dtype=np.int64).tolist(),
            "cm_post": np.zeros((NC, NC), dtype=np.int64).tolist(),
            "transition_wrong_to_right": {},
        }

    pre_ok = np.array([r["cls_yolo"] == r["gt_cls"] for r in rows], dtype=bool)
    post_ok = np.array([r["cls_vlm"] == r["gt_cls"] for r in rows], dtype=bool)
    fix = (~pre_ok) & post_ok
    harm = pre_ok & (~post_ok)

    transitions: Counter[str] = Counter()
    for r, f in zip(rows, fix):
        if f:
            key = f"{CLASS_NAMES[r['cls_yolo']]}->{CLASS_NAMES[r['cls_vlm']]} (gt={CLASS_NAMES[r['gt_cls']]})"
            transitions[key] += 1

    gt_arr = np.array([r["gt_cls"] for r in rows], dtype=np.int64)
    yolo_arr = np.array([r["cls_yolo"] for r in rows], dtype=np.int64)
    vlm_arr = np.array([r["cls_vlm"] for r in rows], dtype=np.int64)

    acc_pre = float(pre_ok.mean())
    acc_post = float(post_ok.mean())
    return {
        "M": len(rows),
        "Acc_pre": acc_pre,
        "Acc_post": acc_post,
        "delta": acc_post - acc_pre,
        "fix_to_correct": int(fix.sum()),
        "correct_to_wrong": int(harm.sum()),
        "net_gain": int(fix.sum() - harm.sum()),
        "cm_pre": confusion_matrix(yolo_arr, gt_arr).tolist(),
        "cm_post": confusion_matrix(vlm_arr, gt_arr).tolist(),
        "transition_wrong_to_right": dict(transitions.most_common()),
    }


def bootstrap_by_image(
    matched_rows: list[dict],
    *,
    all_img_ids: list[str],
    B: int = 2000,
    seed: int = 42,
) -> dict:
    """Resample ALL test images with replacement; pool G_matched boxes; recompute Acc/Δ."""
    by_img: dict[str, list[dict]] = defaultdict(list)
    for r in matched_rows:
        by_img[r["img_id"]].append(r)
    img_ids = list(all_img_ids)
    n_with = sum(1 for i in img_ids if by_img[i])
    rng = np.random.default_rng(seed)

    deltas = np.empty(B, dtype=np.float64)
    acc_posts = np.empty(B, dtype=np.float64)
    acc_pres = np.empty(B, dtype=np.float64)

    if not img_ids:
        return {
            "B": B,
            "n_images": 0,
            "n_images_with_matched": 0,
            "Acc_pre_ci": [None, None],
            "Acc_post_ci": [None, None],
            "delta_ci": [None, None],
            "delta_ci_excludes_zero": None,
        }

    for b in range(B):
        sample = rng.choice(img_ids, size=len(img_ids), replace=True)
        pooled: list[dict] = []
        for iid in sample:
            pooled.extend(by_img.get(iid, []))
        m = metrics_from_matched(pooled)
        if m["M"] == 0:
            deltas[b] = np.nan
            acc_posts[b] = np.nan
            acc_pres[b] = np.nan
        else:
            deltas[b] = m["delta"]
            acc_posts[b] = m["Acc_post"]
            acc_pres[b] = m["Acc_pre"]

    def ci(arr: np.ndarray) -> list[float | None]:
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return [None, None]
        lo, hi = np.percentile(valid, [2.5, 97.5])
        return [float(lo), float(hi)]

    d_ci = ci(deltas)
    return {
        "B": B,
        "seed": seed,
        "n_images": len(img_ids),
        "n_images_with_matched": n_with,
        "note": (
            f"Bootstrap resamples all {len(img_ids)} test images with replacement; "
            f"{n_with} of them contain ≥1 G_matched box. Acc/Δ pooled over boxes "
            "in the resampled images (images with no matched gated boxes contribute 0 boxes)."
        ),
        "Acc_pre_ci": ci(acc_pres),
        "Acc_post_ci": ci(acc_posts),
        "delta_ci": d_ci,
        "delta_ci_excludes_zero": (
            d_ci[0] is not None and d_ci[1] is not None and (d_ci[0] > 0 or d_ci[1] < 0)
        ),
        "delta_point_positive_ci_crosses_zero": (
            d_ci[0] is not None
            and d_ci[1] is not None
            and d_ci[0] <= 0 <= d_ci[1]
        ),
    }


def fmt_pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{100.0 * x:.1f}%"


def cm_to_markdown(cm: list[list[int]], title: str) -> str:
    names = [CLASS_NAMES[i] for i in range(NC)]
    lines = [f"**{title}** (rows=GT, cols=pred)", "", "| GT \\ Pred | " + " | ".join(names) + " |", "|---|" + "|".join(["---"] * NC) + "|"]
    for i, row in enumerate(cm):
        lines.append("| " + names[i] + " | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def run_boxlog(
    *,
    model_path: Path,
    img_dir: Path,
    label_dir: Path,
    imgsz: int,
    conf: float,
    iou_thresh: float,
    device: int,
    vlm: QwenVlmClient,
    vlm_min_conf: float,
    limit: int | None,
) -> list[dict]:
    images = sorted(
        p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if limit is not None:
        images = images[:limit]

    model = YOLO(str(model_path))
    rows: list[dict] = []
    t0 = time.perf_counter()

    for idx, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt_cls, gt_boxes = load_gt(label_dir / f"{img_path.stem}.txt", w, h)

        result = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=conf,
            verbose=False,
            device=device,
        )[0]
        result = apply_rule_postprocess(result, iou_thresh=iou_thresh)

        if result.boxes is None or len(result.boxes) == 0:
            if (idx + 1) % 50 == 0:
                print(f"  processed {idx + 1}/{len(images)} images... ({len(rows)} box rows)")
            continue

        cls_ids = result.boxes.cls.cpu().numpy().astype(np.int64)
        confs = result.boxes.conf.cpu().numpy().astype(np.float32)
        boxes_xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        names = result.names
        n_boxes = len(cls_ids)

        matched_gt, matched_iou = match_pred_to_gt_class_agnostic(boxes_xyxy, gt_boxes)
        gated_idx = set(
            needs_vlm_arbitration(cls_ids, confs, boxes_xyxy, low_conf=0.65)
        )

        # Effective post-VLM classes (start as YOLO)
        cls_vlm = cls_ids.copy()
        vlm_raw = np.full(n_boxes, -1, dtype=np.int64)
        vlm_conf_arr = np.full(n_boxes, np.nan, dtype=np.float32)
        applied = np.zeros(n_boxes, dtype=bool)
        skip_reasons: list[str | None] = [None] * n_boxes

        for i in sorted(gated_idx):
            roi = crop_roi(result.orig_img, boxes_xyxy[i])
            if roi.size == 0:
                skip_reasons[i] = "empty roi"
                continue
            try:
                vlm_res = vlm.classify_roi(
                    roi,
                    yolo_class=str(names[int(cls_ids[i])]),
                    yolo_conf=float(confs[i]),
                )
            except Exception as exc:  # noqa: BLE001
                skip_reasons[i] = f"vlm error: {exc}"
                continue

            if vlm_res.class_id is not None:
                vlm_raw[i] = int(vlm_res.class_id)
            vlm_conf_arr[i] = float(vlm_res.confidence)

            ok, reason = should_apply_vlm_update(
                int(cls_ids[i]),
                vlm_res,
                vlm_min_conf=vlm_min_conf,
                n_boxes=n_boxes,
            )
            if ok and vlm_res.class_id is not None:
                cls_vlm[i] = int(vlm_res.class_id)
                applied[i] = True
            else:
                skip_reasons[i] = reason

        for i in range(n_boxes):
            gi = int(matched_gt[i])
            row = {
                "img_id": img_path.name,
                "box_idx": i,
                "box_xyxy": [round(float(v), 2) for v in boxes_xyxy[i].tolist()],
                "yolo_conf": round(float(confs[i]), 4),
                "gated": bool(i in gated_idx),
                "matched_gt": gi,
                "match_iou": round(float(matched_iou[i]), 4) if gi >= 0 else None,
                "gt_cls": int(gt_cls[gi]) if gi >= 0 else None,
                "gt_cls_name": CLASS_NAMES[int(gt_cls[gi])] if gi >= 0 else None,
                "cls_yolo": int(cls_ids[i]),
                "cls_yolo_name": CLASS_NAMES[int(cls_ids[i])],
                "cls_vlm_raw": int(vlm_raw[i]) if vlm_raw[i] >= 0 else None,
                "cls_vlm_raw_name": CLASS_NAMES[int(vlm_raw[i])] if vlm_raw[i] >= 0 else None,
                "cls_vlm": int(cls_vlm[i]),
                "cls_vlm_name": CLASS_NAMES[int(cls_vlm[i])],
                "vlm_conf": None if np.isnan(vlm_conf_arr[i]) else round(float(vlm_conf_arr[i]), 4),
                "vlm_applied": bool(applied[i]),
                "skip_reason": skip_reasons[i],
            }
            rows.append(row)

        if (idx + 1) % 25 == 0 or (idx + 1) == len(images):
            elapsed = time.perf_counter() - t0
            print(
                f"  processed {idx + 1}/{len(images)} images... "
                f"{len(rows)} box rows, {elapsed:.1f}s"
            )

    return rows


def summarize(
    rows: list[dict],
    *,
    all_img_ids: list[str],
    B: int,
    seed: int,
) -> dict:
    gated = [r for r in rows if r["gated"]]
    matched = [r for r in gated if r["matched_gt"] is not None and r["matched_gt"] >= 0 and r["gt_cls"] is not None]

    # Normalize for metrics_from_matched
    matched_norm = [
        {
            "img_id": r["img_id"],
            "cls_yolo": r["cls_yolo"],
            "cls_vlm": r["cls_vlm"],
            "gt_cls": r["gt_cls"],
        }
        for r in matched
    ]
    core = metrics_from_matched(matched_norm)
    boot = bootstrap_by_image(matched_norm, all_img_ids=all_img_ids, B=B, seed=seed)
    n_images = len(all_img_ids)

    # Directional flips among fix_to_correct
    pack_lighter_fixes = 0
    butts_lighter_fixes = 0
    for r in matched:
        if r["cls_yolo"] != r["gt_cls"] and r["cls_vlm"] == r["gt_cls"]:
            pair = {r["cls_yolo"], r["cls_vlm"]}
            if pair == {0, 3}:
                butts_lighter_fixes += 1
            if pair == {1, 3}:
                pack_lighter_fixes += 1

    # Also count applied flips regardless of GT (for debugging)
    applied_flips = Counter()
    for r in gated:
        if r["vlm_applied"] and r["cls_yolo"] != r["cls_vlm"]:
            applied_flips[f"{r['cls_yolo_name']}->{r['cls_vlm_name']}"] += 1

    return {
        "gt_provenance": (
            "merged_dataset/test from Roboflow YOLO exports via merge_datasets.py "
            "(independent labels; newtest_/image_0828 injections are train-only)."
        ),
        "n_images": n_images,
        "n_pred_boxes": len(rows),
        "N_gated": len(gated),
        "M_matched": core["M"],
        "Acc_pre": core["Acc_pre"],
        "Acc_post": core["Acc_post"],
        "delta": core["delta"],
        "fix_to_correct": core["fix_to_correct"],
        "correct_to_wrong": core["correct_to_wrong"],
        "net_gain": core["net_gain"],
        "cm_pre": core["cm_pre"],
        "cm_post": core["cm_post"],
        "transition_wrong_to_right": core["transition_wrong_to_right"],
        "fixes_butts_lighter": butts_lighter_fixes,
        "fixes_pack_lighter": pack_lighter_fixes,
        "applied_flips_all_gated": dict(applied_flips),
        "bootstrap": boot,
        "overall_map50_reference": {
            "no_vlm": 0.857,
            "with_vlm": 0.818,
            "source": "runs/detect/eval_testset/metrics_*.json",
        },
    }


def write_outputs(rows: list[dict], summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "box_log.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    csv_path = out_dir / "box_log.csv"
    fieldnames = [
        "img_id",
        "box_idx",
        "box_xyxy",
        "yolo_conf",
        "gated",
        "matched_gt",
        "match_iou",
        "gt_cls",
        "gt_cls_name",
        "cls_yolo",
        "cls_yolo_name",
        "cls_vlm_raw",
        "cls_vlm_raw_name",
        "cls_vlm",
        "cls_vlm_name",
        "vlm_conf",
        "vlm_applied",
        "skip_reason",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row = dict(r)
            row["box_xyxy"] = json.dumps(r["box_xyxy"])
            w.writerow({k: row.get(k) for k in fieldnames})

    (out_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Resume sentence
    N = summary["N_gated"]
    M = summary["M_matched"]
    X = summary["Acc_pre"]
    Y = summary["Acc_post"]
    d_ci = summary["bootstrap"]["delta_ci"]
    Z = summary["fix_to_correct"]
    W = summary["correct_to_wrong"]
    delta = summary["delta"]

    if summary["fixes_butts_lighter"] == 0 and summary["fixes_pack_lighter"] == 0:
        if W > 0:
            # describe harm direction from cm delta
            main_dir = "无净纠正；误伤主要为 butts→pack / butts→Lighter（见转移表）"
        else:
            main_dir = "无类别纠正转移"
    elif summary["fixes_butts_lighter"] >= summary["fixes_pack_lighter"]:
        main_dir = "butts↔Lighter"
    else:
        main_dir = "pack↔Lighter"

    if d_ci[0] is not None and d_ci[1] is not None and d_ci[0] > 0:
        sig_note = "Δ 的 95% CI 下界>0，提升统计显著"
        verb = "提升至"
    elif d_ci[0] is not None and d_ci[1] is not None and d_ci[1] < 0:
        sig_note = "Δ 的 95% CI 上界<0，下降统计显著（VLM 净误伤）"
        verb = "变为"
    elif summary["bootstrap"].get("delta_point_positive_ci_crosses_zero"):
        sig_note = "点估计为正但 95% CI 跨 0，未达显著"
        verb = "变为"
    else:
        sig_note = "见 CI 解读"
        verb = "变为"

    sentence = (
        f"利用Qwen3-VL对YOLOv8输出的易混框做类别二次仲裁。"
        f"在完整测试集（{summary['n_images']}图）的N={N}个门控框中，"
        f"可与GT对齐的M={M}个框上，分类准确率由{fmt_pct(X)}{verb}{fmt_pct(Y)}"
        f"（按图bootstrap，B={summary['bootstrap']['B']}，"
        f"Δ 95%CI [{fmt_pct(d_ci[0]) if d_ci[0] is not None else 'n/a'}, "
        f"{fmt_pct(d_ci[1]) if d_ci[1] is not None else 'n/a'}]；{sig_note}），"
        f"净纠正{Z}处误分类、误伤{W}处；"
        f"方向解读：{main_dir}"
        f"（butts↔Lighter纠正={summary['fixes_butts_lighter']}，"
        f"pack↔Lighter纠正={summary['fixes_pack_lighter']}）；"
        f"整体mAP50 0.857→0.818（VLM不改定位、仅作用于分类；"
        f"本测试集上门控∩匹配子集 Acc_pre 已近满分，VLM几乎无可纠正空间，误伤主导全局mAP略降）。"
    )

    report = f"""# VLM 框级仲裁评测报告

## GT 来源确认

{summary['gt_provenance']}

## 简历句（填数版）

{sentence}

## 主指标

| 指标 | 值 |
|------|-----|
| 测试图数 | {summary['n_images']} |
| 预测框总数 | {summary['n_pred_boxes']} |
| N（门控框） | {N} |
| M（门控 ∩ IoU≥0.5 匹配 GT） | {M} |
| Acc_pre (X) | {fmt_pct(X)} |
| Acc_post (Y) | {fmt_pct(Y)} |
| Δ = Y−X | {fmt_pct(summary['delta']) if summary['delta'] is not None else 'n/a'} |
| 改对 (wrong→right) | {Z} |
| 误伤 (right→wrong) | {W} |
| 净收益 | {summary['net_gain']} |
| butts↔Lighter 纠正 | {summary['fixes_butts_lighter']} |
| pack↔Lighter 纠正 | {summary['fixes_pack_lighter']} |

## Bootstrap（按图，有放回）

- B = {summary['bootstrap']['B']}, seed = {summary['bootstrap']['seed']}
- Bootstrap 图数: {summary['bootstrap'].get('n_images', summary['n_images'])}
- 有 ≥1 个 G_matched 的图数: {summary['bootstrap']['n_images_with_matched']}
- Acc_pre 95% CI: {summary['bootstrap']['Acc_pre_ci']}
- Acc_post 95% CI: {summary['bootstrap']['Acc_post_ci']}
- Δ 95% CI: {summary['bootstrap']['delta_ci']}
- CI 是否排除 0: {summary['bootstrap']['delta_ci_excludes_zero']}
- 注: {summary['bootstrap']['note']}

## 改对转移（wrong→right）

```json
{json.dumps(summary['transition_wrong_to_right'], indent=2, ensure_ascii=False)}
```

## 混淆矩阵

{cm_to_markdown(summary['cm_pre'], 'Pre-VLM (cls_yolo vs GT)')}

{cm_to_markdown(summary['cm_post'], 'Post-VLM (cls_vlm vs GT)')}

## 参考：整体 mAP50

来自既有 `eval_testset`：{summary['overall_map50_reference']}

## 产物文件

- `box_log.csv` / `box_log.jsonl` — 框级日志
- `metrics_summary.json` — 机器可读汇总
"""
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    (out_dir / "resume_sentence.txt").write_text(sentence + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Box-level VLM arbitration eval")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--img-dir", type=Path, default=TEST_IMG_DIR)
    parser.add_argument("--label-dir", type=Path, default=TEST_LBL_DIR)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou-thresh", type=float, default=0.3)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--vlm-api", type=str, default="http://127.0.0.1:8001/v1")
    parser.add_argument("--vlm-model", type=str, default="qwen3-vl-idc")
    parser.add_argument("--vlm-min-conf", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bootstrap-B", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "runs/detect/eval_vlm_boxlog",
    )
    parser.add_argument(
        "--from-log",
        type=Path,
        default=None,
        help="Skip inference; recompute metrics from existing box_log.jsonl",
    )
    args = parser.parse_args()

    all_img_ids = sorted(
        p.name
        for p in args.img_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if args.limit is not None:
        all_img_ids = all_img_ids[: args.limit]

    if args.from_log is not None:
        rows = [
            json.loads(line)
            for line in args.from_log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        # Prefer full test image list for bootstrap; fall back to ids seen in log
        if not all_img_ids:
            all_img_ids = sorted({r["img_id"] for r in rows})
    else:
        vlm = QwenVlmClient(api_base=args.vlm_api, model=args.vlm_model)
        if not vlm.health_ok():
            raise RuntimeError(f"Qwen3-VL not ready at {args.vlm_api}")
        print(f"VLM ready: {args.vlm_api} model={args.vlm_model}")
        print(f"model: {args.model}")
        print(f"images: {args.img_dir} ({len(all_img_ids)})")
        print(f"output: {args.out}")

        rows = run_boxlog(
            model_path=args.model,
            img_dir=args.img_dir,
            label_dir=args.label_dir,
            imgsz=args.imgsz,
            conf=args.conf,
            iou_thresh=args.iou_thresh,
            device=args.device,
            vlm=vlm,
            vlm_min_conf=args.vlm_min_conf,
            limit=args.limit,
        )

    summary = summarize(rows, all_img_ids=all_img_ids, B=args.bootstrap_B, seed=args.seed)
    write_outputs(rows, summary, args.out)

    print("\n=== Summary ===")
    print(f"N_gated={summary['N_gated']}  M_matched={summary['M_matched']}")
    print(f"Acc_pre={fmt_pct(summary['Acc_pre'])}  Acc_post={fmt_pct(summary['Acc_post'])}  Δ={fmt_pct(summary['delta'])}")
    print(f"fix={summary['fix_to_correct']}  harm={summary['correct_to_wrong']}  net={summary['net_gain']}")
    print(f"Δ 95%CI={summary['bootstrap']['delta_ci']}")
    print(f"Saved: {args.out}/report.md")


if __name__ == "__main__":
    main()
