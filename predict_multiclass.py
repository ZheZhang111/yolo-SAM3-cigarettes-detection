#!/usr/bin/env python3
"""Multiclass detection with rule + Qwen3-VL post-processing."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from postprocess import (
    CIG_PACK,
    CIGARETTES_BUTTS,
    LIGHTER,
    suppress_butts_overlapping_lighter,
    suppress_pack_overlapping_lighter,
)
from vlm_arbitrate import QwenVlmClient, crop_roi, should_apply_vlm_update
from vlm_gate import needs_vlm_arbitration
from vis_labels import class_name_zh, en_name_zh
from vis_postprocess import (
    VisPostprocessConfig,
    boxes_from_yolo_result,
    draw_detection_image,
    format_confidence,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "runs/detect/cig_multiclass_ft_newtest/weights/best.pt"


def apply_rule_postprocess(result, iou_thresh: float = 0.3):
    """IoU rules: drop butts/pack overlapping lighter."""
    if result.boxes is None or len(result.boxes) == 0:
        return result
    cls_ids = result.boxes.cls.cpu().numpy()
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    keep = suppress_butts_overlapping_lighter(cls_ids, boxes_xyxy, iou_thresh)
    keep &= suppress_pack_overlapping_lighter(cls_ids, boxes_xyxy, iou_thresh)
    if keep.all():
        return result
    result.boxes = result.boxes[keep]
    return result


def apply_vlm_postprocess(
    result,
    vlm: QwenVlmClient,
    *,
    vlm_min_conf: float = 0.5,
    low_conf: float = 0.65,
) -> tuple[object, list[dict]]:
    """Run VLM on gated boxes and update class/conf."""
    logs: list[dict] = []
    if result.boxes is None or len(result.boxes) == 0:
        return result, logs

    cls_ids = result.boxes.cls.cpu().numpy().copy()
    confs = result.boxes.conf.cpu().numpy().copy()
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    names = result.names

    indices = needs_vlm_arbitration(cls_ids, confs, boxes_xyxy, low_conf=low_conf)
    n_boxes = len(cls_ids)
    for i in indices:
        roi = crop_roi(result.orig_img, boxes_xyxy[i])
        if roi.size == 0:
            continue
        try:
            vlm_res = vlm.classify_roi(
                roi,
                yolo_class=str(names[int(cls_ids[i])]),
                yolo_conf=float(confs[i]),
            )
        except Exception as exc:  # noqa: BLE001
            logs.append({"index": i, "error": str(exc)})
            continue

        old_cls = int(cls_ids[i])
        old_name = names[old_cls]
        log = {
            "index": i,
            "yolo_class": old_name,
            "yolo_conf": float(confs[i]),
            "vlm_class": vlm_res.class_name,
            "vlm_conf": vlm_res.confidence,
            "reason": vlm_res.reason,
            "multiple_objects": vlm_res.multiple_objects,
            "objects_present": vlm_res.objects_present,
        }
        logs.append(log)

        apply, gate_reason = should_apply_vlm_update(
            old_cls, vlm_res, vlm_min_conf=vlm_min_conf, n_boxes=n_boxes
        )
        if not apply:
            log["skipped"] = gate_reason
            continue
        cls_ids[i] = vlm_res.class_id
        confs[i] = max(float(confs[i]), vlm_res.confidence)
        log["updated"] = True

    device = result.boxes.data.device
    dtype = result.boxes.data.dtype
    data = result.boxes.data.clone()
    data[:, 4] = torch.tensor(confs, device=device, dtype=dtype)
    data[:, 5] = torch.tensor(cls_ids, device=device, dtype=dtype)
    result.boxes.data = data
    return result, logs


def draw_and_save(
    result,
    out_img: Path,
    names: dict,
    *,
    vis_cfg: VisPostprocessConfig,
    use_zh: bool = False,
):
    out_img.parent.mkdir(parents=True, exist_ok=True)
    cfg = VisPostprocessConfig(
        enabled=vis_cfg.enabled,
        use_zh=vis_cfg.use_zh if vis_cfg.enabled else use_zh,
        percent_conf=vis_cfg.percent_conf,
        bold_zh=vis_cfg.bold_zh,
    )
    img = draw_detection_image(
        result.orig_img.copy(),
        boxes_from_yolo_result(result),
        names,
        cfg=cfg,
    )
    cv2.imwrite(str(out_img), img)


def _display_name(cls: int, names: dict, use_zh: bool) -> str:
    if use_zh:
        return class_name_zh(cls, names)
    return str(names[cls])


def _format_conf_for_console(conf: float, *, vis_cfg: VisPostprocessConfig) -> str:
    if vis_cfg.enabled and vis_cfg.percent_conf:
        return format_confidence(conf, as_percent=True)
    return f"{conf:.3f}"


def save_label(result, out_txt: Path, img_w: int, img_h: int):
    lines: list[str] = []
    if result.boxes is not None and len(result.boxes) > 0:
        for b in result.boxes:
            cls = int(b.cls)
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            xc = ((x1 + x2) / 2) / img_w
            yc = ((y1 + y2) / 2) / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="YOLO + rule + Qwen3-VL predict")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou-thresh", type=float, default=0.3)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/detect/New_test")
    parser.add_argument("--name", type=str, default="predict_vlm")
    parser.add_argument("--vlm", action="store_true", help="enable Qwen3-VL arbitration")
    parser.add_argument("--vlm-api", type=str, default="http://127.0.0.1:8001/v1")
    parser.add_argument("--vlm-model", type=str, default="qwen3-vl-idc")
    parser.add_argument("--vlm-min-conf", type=float, default=0.5)
    parser.add_argument(
        "--zh",
        action="store_true",
        help="use Chinese labels when --no-vis-postprocess (legacy English default)",
    )
    parser.add_argument(
        "--no-vis-postprocess",
        action="store_true",
        help="disable vis postprocess (bold Chinese labels + percent confidence)",
    )
    args = parser.parse_args()

    vis_cfg = VisPostprocessConfig(enabled=not args.no_vis_postprocess)
    console_zh = vis_cfg.enabled or args.zh

    source = args.source
    if source.is_dir():
        images = sorted(
            p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
    else:
        images = [source]

    out_dir = args.project / args.name
    label_dir = out_dir / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))
    names = model.names

    vlm: QwenVlmClient | None = None
    if args.vlm:
        vlm = QwenVlmClient(api_base=args.vlm_api, model=args.vlm_model)
        if not vlm.health_ok():
            raise RuntimeError(
                f"Qwen3-VL service not ready at {args.vlm_api}. "
                "Start vLLM first (see doc or tmux qwen3vl)."
            )
        print(f"VLM   : {args.vlm_api} model={args.vlm_model}")

    print(f"model : {args.model}")
    print(f"source: {source}")
    print(f"output: {out_dir}")
    print(f"labels: {'Chinese+bold+%' if vis_cfg.enabled else ('Chinese' if args.zh else 'English')}")
    if vis_cfg.enabled:
        print("vis   : postprocess ON (bold Chinese, percent confidence)")
    print()

    for img_path in images:
        results = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            verbose=False,
            device=args.device,
        )
        result = results[0]
        result = apply_rule_postprocess(result, iou_thresh=args.iou_thresh)

        vlm_logs: list[dict] = []
        if vlm is not None:
            result, vlm_logs = apply_vlm_postprocess(
                result, vlm, vlm_min_conf=args.vlm_min_conf
            )
            result = apply_rule_postprocess(result, iou_thresh=args.iou_thresh)

        h, w = result.orig_shape
        stem = img_path.stem
        draw_and_save(result, out_dir / f"{stem}.jpg", names, vis_cfg=vis_cfg, use_zh=args.zh)
        save_label(result, label_dir / f"{stem}.txt", w, h)

        print(img_path.name)
        if result.boxes is None or len(result.boxes) == 0:
            print("  (no detections)")
        else:
            for b in result.boxes:
                cls = int(b.cls)
                print(
                    f"  {_display_name(cls, names, console_zh):18s} "
                    f"conf={_format_conf_for_console(float(b.conf), vis_cfg=vis_cfg)}"
                )
        for log in vlm_logs:
            if "error" in log:
                print(f"  [VLM err] {log['error']}")
            elif log.get("updated"):
                y0 = en_name_zh(log["yolo_class"]) if console_zh else log["yolo_class"]
                y1 = en_name_zh(log["vlm_class"]) if console_zh else log["vlm_class"]
                print(
                    f"  [VLM] {y0} -> {y1} "
                    f"({log['vlm_conf']:.2f}) {log.get('reason', '')}"
                )
            elif log.get("skipped"):
                y0 = en_name_zh(log["yolo_class"]) if console_zh else log["yolo_class"]
                y1 = en_name_zh(log["vlm_class"]) if console_zh else log["vlm_class"]
                multi = log.get("multiple_objects")
                present = log.get("objects_present")
                extra = ""
                if multi:
                    extra = f" multi={present}"
                if console_zh:
                    print(f"  [VLM] 保持 {y0}；VLM={y1} ({log['vlm_conf']:.2f}) [{log['skipped']}]{extra}")
                else:
                    print(f"  [VLM] kept {y0}; vlm={y1} ({log['vlm_conf']:.2f}) [{log['skipped']}]{extra}")
            elif "vlm_class" in log:
                y0 = en_name_zh(log["yolo_class"]) if console_zh else log["yolo_class"]
                y1 = en_name_zh(log["vlm_class"]) if console_zh else log["vlm_class"]
                if console_zh:
                    print(f"  [VLM] 保持 {y0}；VLM={y1} ({log['vlm_conf']:.2f})")
                else:
                    print(f"  [VLM] kept {y0}; vlm={y1} ({log['vlm_conf']:.2f})")
        print()


if __name__ == "__main__":
    main()
