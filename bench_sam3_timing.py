#!/usr/bin/env python3
"""Accurate SAM3 inference timing (warmup + CUDA sync).

Reports:
  - optimized: 1× set_image + text(lighter) + 5× pack text prompts
  - legacy:    same as run_sam3_predict.py (set_image again for every pack prompt)

Example:
  CUDA_VISIBLE_DEVICES=5 python bench_sam3_timing.py \\
    --source New_test/image_0901_v2 --warmup 3 --repeat 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from PIL import Image

SAM3_ROOT = Path("/data/zhangzhe/do-as-i-do/reconstruction/modules/sam3")
sys.path.insert(0, str(SAM3_ROOT))

from sam3.model_builder import build_sam3_image_model  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402

ROOT = Path(__file__).resolve().parent
CKPT = Path("/data/zhangzhe/do-as-i-do/reconstruction/checkpoints/sam3/sam3.pt")

PACK_PROMPTS = [
    "red cigarette pack",
    "gold cigarette pack",
    "cigarette pack",
    "cigarette box",
    "tobacco pack",
]


def iter_images(source: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if source.is_file():
        return [source]
    return sorted(p for p in source.iterdir() if p.suffix.lower() in exts)


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    sync()
    t0 = time.perf_counter()
    out = fn()
    sync()
    return out, time.perf_counter() - t0


def ms(x: float) -> float:
    return round(1000.0 * x, 2)


def summarize(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean_ms": ms(statistics.mean(xs)),
        "median_ms": ms(statistics.median(xs)),
        "stdev_ms": ms(statistics.stdev(xs)) if len(xs) > 1 else 0.0,
        "min_ms": ms(min(xs)),
        "max_ms": ms(max(xs)),
    }


def run_optimized_once(processor: Sam3Processor, image: Image.Image) -> dict:
    """Encode image once; then lighter + each pack text prompt."""
    parts: dict = {}
    state, parts["set_image"] = timed(lambda: processor.set_image(image))
    state, parts["text_lighter"] = timed(
        lambda: processor.set_text_prompt(prompt="lighter", state=state)
    )

    pack_times: list[float] = []
    best_score = -1.0
    best_prompt = None
    for prompt in PACK_PROMPTS:
        state, t_txt = timed(
            lambda p=prompt, s=state: processor.set_text_prompt(prompt=p, state=s)
        )
        pack_times.append(t_txt)
        scores = state.get("scores")
        if scores is not None and len(scores):
            sc = float(torch.max(scores))
            if sc > best_score:
                best_score = sc
                best_prompt = prompt

    parts["text_pack_sum"] = sum(pack_times)
    parts["text_pack_mean"] = statistics.mean(pack_times) if pack_times else 0.0
    parts["text_pack_each_ms"] = [ms(t) for t in pack_times]
    parts["best_pack_prompt"] = best_prompt
    parts["n_image_encodes"] = 1
    parts["total"] = parts["set_image"] + parts["text_lighter"] + parts["text_pack_sum"]
    return parts


def run_legacy_once(processor: Sam3Processor, image: Image.Image) -> dict:
    """Matches run_sam3_predict.py: re-set_image for every pack prompt."""
    parts: dict = {}
    total = 0.0

    processor.reset_all_prompts({})
    state, t0 = timed(lambda: processor.set_image(image))
    state, t1 = timed(lambda: processor.set_text_prompt(prompt="lighter", state=state))
    parts["lighter_set_image"] = t0
    parts["lighter_text"] = t1
    total += t0 + t1

    pack_img: list[float] = []
    pack_txt: list[float] = []
    for prompt in PACK_PROMPTS:
        processor.reset_all_prompts({})
        st, ti = timed(lambda: processor.set_image(image))
        st, tt = timed(lambda p=prompt, s=st: processor.set_text_prompt(prompt=p, state=s))
        pack_img.append(ti)
        pack_txt.append(tt)
        total += ti + tt

    parts["pack_set_image_sum"] = sum(pack_img)
    parts["pack_text_sum"] = sum(pack_txt)
    parts["n_image_encodes"] = 1 + len(PACK_PROMPTS)
    parts["total"] = total
    return parts


def warmup(processor: Sam3Processor, image: Image.Image, n: int) -> None:
    for _ in range(n):
        state = processor.set_image(image)
        processor.set_text_prompt(prompt="lighter", state=state)
        sync()


def bench_yolo(
    image_paths: list[Path],
    *,
    device: int,
    imgsz: int,
    warmup_n: int,
    repeat: int,
    weights: Path,
) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    for _ in range(warmup_n):
        model.predict(
            source=str(image_paths[0]),
            imgsz=imgsz,
            conf=0.25,
            verbose=False,
            device=device,
        )
        sync()

    per_image = []
    all_times: list[float] = []
    for p in image_paths:
        times = []
        for _ in range(repeat):
            _, t = timed(
                lambda path=p: model.predict(
                    source=str(path),
                    imgsz=imgsz,
                    conf=0.25,
                    verbose=False,
                    device=device,
                )
            )
            times.append(t)
            all_times.append(t)
        per_image.append({"image": p.name, **summarize(times)})
    return {"per_image": per_image, "overall": summarize(all_times), "imgsz": imgsz}


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM3 accurate timing bench")
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "New_test/image_0901_v2",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--yolo", action="store_true", help="also bench YOLO (needs ultralytics)")
    parser.add_argument("--yolo-device", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        default=ROOT / "runs/detect/cig_multiclass_ft_newtest/weights/best.pt",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "runs/detect/bench_sam3_timing/image_0901_v2.json",
    )
    args = parser.parse_args()

    images = iter_images(args.source)
    if not images:
        raise SystemExit(f"no images in {args.source}")

    print(f"source : {args.source} ({len(images)} images)")
    print(f"ckpt   : {CKPT}")
    print(f"device : {args.device}  warmup={args.warmup} repeat={args.repeat}")
    if torch.cuda.is_available():
        print(f"cuda   : {torch.cuda.get_device_name(0)}")
    print()

    model = build_sam3_image_model(
        checkpoint_path=str(CKPT),
        load_from_HF=False,
        device=args.device,
    ).to(args.device)
    processor = Sam3Processor(model, device=args.device, confidence_threshold=args.conf)

    img0 = Image.open(images[0]).convert("RGB")
    print(f"warmup {args.warmup}× on {images[0].name} ...")
    warmup(processor, img0, args.warmup)
    print("warmup done.\n")

    report: dict = {
        "source": str(args.source),
        "n_images": len(images),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "pack_prompts": PACK_PROMPTS,
        "timing_method": "perf_counter + torch.cuda.synchronize; exclude read/draw/save",
        "note": (
            "optimized = 1× set_image + lighter + 5 pack texts. "
            "legacy = run_sam3_predict style (set_image per pack prompt → 6 encodes/image)."
        ),
        "images": [],
    }

    opt_totals: list[float] = []
    leg_totals: list[float] = []
    encode_times: list[float] = []
    lighter_times: list[float] = []
    pack_sum_times: list[float] = []

    for img_path in images:
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        print(f"=== {img_path.name} ({w}x{h}) ===")

        opt_runs = []
        leg_runs = []
        for i in range(args.repeat):
            opt = run_optimized_once(processor, image)
            leg = run_legacy_once(processor, image)
            opt_runs.append(opt)
            leg_runs.append(leg)
            print(
                f"  rep{i + 1}: optimized={ms(opt['total']):7.1f} ms  "
                f"(img={ms(opt['set_image']):6.1f} light={ms(opt['text_lighter']):6.1f} "
                f"packs={ms(opt['text_pack_sum']):6.1f})  "
                f"legacy={ms(leg['total']):7.1f} ms ({leg['n_image_encodes']}× encode)"
            )

        opt_total = [r["total"] for r in opt_runs]
        leg_total = [r["total"] for r in leg_runs]
        enc = [r["set_image"] for r in opt_runs]
        light = [r["text_lighter"] for r in opt_runs]
        packs = [r["text_pack_sum"] for r in opt_runs]

        opt_totals.extend(opt_total)
        leg_totals.extend(leg_total)
        encode_times.extend(enc)
        lighter_times.extend(light)
        pack_sum_times.extend(packs)

        entry = {
            "image": img_path.name,
            "size": [w, h],
            "optimized": {
                "total": summarize(opt_total),
                "set_image": summarize(enc),
                "text_lighter": summarize(light),
                "text_pack_sum": summarize(packs),
                "best_pack_prompt": opt_runs[-1].get("best_pack_prompt"),
                "text_pack_each_ms_last": opt_runs[-1].get("text_pack_each_ms"),
            },
            "legacy_run_sam3_predict": {
                "total": summarize(leg_total),
                "n_image_encodes": 1 + len(PACK_PROMPTS),
            },
        }
        report["images"].append(entry)
        print(
            f"  -> optimized median {entry['optimized']['total']['median_ms']} ms | "
            f"legacy median {entry['legacy_run_sam3_predict']['total']['median_ms']} ms\n"
        )

    report["overall"] = {
        "optimized_total": summarize(opt_totals),
        "legacy_total": summarize(leg_totals),
        "set_image": summarize(encode_times),
        "text_lighter": summarize(lighter_times),
        "text_pack_sum": summarize(pack_sum_times),
    }

    if args.yolo:
        print("=== YOLO timing (same images) ===")
        try:
            report["yolo"] = bench_yolo(
                images,
                device=args.yolo_device,
                imgsz=args.imgsz,
                warmup_n=args.warmup,
                repeat=args.repeat,
                weights=args.yolo_weights,
            )
            print(f"  YOLO overall: {report['yolo']['overall']}")
        except Exception as exc:  # noqa: BLE001
            report["yolo_error"] = str(exc)
            print(f"  YOLO bench skipped: {exc}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== OVERALL (all images × repeats) ===")
    print(f"optimized (1 encode + 6 texts): {report['overall']['optimized_total']}")
    print(f"  set_image:     {report['overall']['set_image']}")
    print(f"  text_lighter:  {report['overall']['text_lighter']}")
    print(f"  text_pack_sum: {report['overall']['text_pack_sum']}")
    print(f"legacy (6 encodes):             {report['overall']['legacy_total']}")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
