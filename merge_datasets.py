#!/usr/bin/env python3
"""Merge three Roboflow YOLO datasets into one 4-class dataset.

Unified classes:
  0: cigarettes_butts
  1: cig-pack
  2: cigarette
  3: Lighter

Source class remaps (unlisted classes are dropped):
  Butts:          cigarettes_butts(0) → 0   | drop others
  SmokingAndPack: cig-pack(0) → 1, cigarette(1) → 2  | drop face, smoking
  Lighter:        Lighter(0) → 3
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
OUT = ROOT / "merged_dataset"

UNIFIED_NAMES = ["cigarettes_butts", "cig-pack", "cigarette", "Lighter"]

# (src_dir relative to DATASET, {src_cls_id: unified_cls_id})
SOURCES = [
    (
        "Yolo Cigarettes Butts.v1i.yolov8",
        {0: 0},  # cigarettes_butts only
    ),
    (
        "SmokingAndPack.v1i.yolov8",
        {0: 1, 1: 2},  # cig-pack, cigarette
    ),
    (
        "Lighter.v1i.yolov8",
        {0: 3},  # Lighter
    ),
]

SPLITS = ("train", "valid", "test")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def remap_label_file(src: Path, dst: Path, class_map: dict[int, int]) -> int:
    """Write remapped YOLO label; return number of kept boxes. Skip file if empty."""
    lines_out: list[str] = []
    if src.is_file():
        for line in src.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                cid = int(float(parts[0]))
            except (ValueError, IndexError):
                continue
            if cid not in class_map:
                continue
            parts[0] = str(class_map[cid])
            lines_out.append(" ".join(parts))
    if not lines_out:
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return len(lines_out)


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.is_file():
            return p
        p2 = images_dir / f"{stem}{ext.upper()}"
        if p2.is_file():
            return p2
    # fallback: any file with same stem
    matches = list(images_dir.glob(f"{stem}.*"))
    return matches[0] if matches else None


def merge() -> dict:
    if OUT.exists():
        shutil.rmtree(OUT)
    stats = {
        split: {"images": 0, "boxes": 0, "per_class": {i: 0 for i in range(4)}}
        for split in SPLITS
    }
    file_idx = {split: 0 for split in SPLITS}

    for src_name, class_map in SOURCES:
        src_root = DATASET / src_name
        if not src_root.is_dir():
            raise FileNotFoundError(src_root)
        print(f"\n=== merging {src_name} ===")
        for split in SPLITS:
            labels_dir = src_root / split / "labels"
            images_dir = src_root / split / "images"
            if not labels_dir.is_dir():
                print(f"  skip missing {split}/labels")
                continue
            n_kept_img = 0
            n_kept_box = 0
            for lab in sorted(labels_dir.glob("*.txt")):
                stem = lab.stem
                img = find_image(images_dir, stem)
                if img is None:
                    continue
                # unique prefix avoids name collisions across sources
                file_idx[split] += 1
                new_stem = f"{src_name.split('.')[0].replace(' ', '_')}_{file_idx[split]:06d}_{stem}"
                out_lab = OUT / split / "labels" / f"{new_stem}.txt"
                n_box = remap_label_file(lab, out_lab, class_map)
                if n_box == 0:
                    continue
                out_img = OUT / split / "images" / f"{new_stem}{img.suffix.lower()}"
                out_img.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img, out_img)
                n_kept_img += 1
                n_kept_box += n_box
                # recount classes from written label
                for line in out_lab.read_text(encoding="utf-8").splitlines():
                    cid = int(float(line.split()[0]))
                    stats[split]["per_class"][cid] += 1
            stats[split]["images"] += n_kept_img
            stats[split]["boxes"] += n_kept_box
            print(f"  {split}: +{n_kept_img} images, +{n_kept_box} boxes")

    yaml_text = (
        f"path: {OUT.resolve()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        f"\nnc: {len(UNIFIED_NAMES)}\n"
        f"names: {UNIFIED_NAMES}\n"
    )
    (OUT / "data.yaml").write_text(yaml_text, encoding="utf-8")
    print(f"\nwrote {OUT / 'data.yaml'}")
    return stats


def main():
    stats = merge()
    print("\n========== SUMMARY ==========")
    for split in SPLITS:
        s = stats[split]
        print(f"{split}: images={s['images']} boxes={s['boxes']}")
        for i, name in enumerate(UNIFIED_NAMES):
            print(f"  [{i}] {name}: {s['per_class'][i]}")


if __name__ == "__main__":
    main()
