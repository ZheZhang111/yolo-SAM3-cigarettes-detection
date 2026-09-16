#!/usr/bin/env python3
"""Append aTarah Lighter-only images into merged_dataset for fine-tuning.

Split plan:
  - train: aTarah train + valid  (511 images)
  - valid: 25 images from aTarah test (deterministic sample)
  - test:  remaining aTarah test images (holdout, appended to merged test)
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ATARAH = ROOT / "dataset" / "aTarah.v1i.yolov8"
MERGED = ROOT / "merged_dataset"

LIGHTER_SRC_ID = 4  # aTarah class id
LIGHTER_DST_ID = 3  # unified class id
VALID_FROM_TEST = 25
SEED = 42

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        for p in (images_dir / f"{stem}{ext}", images_dir / f"{stem}{ext.upper()}"):
            if p.is_file():
                return p
    matches = list(images_dir.glob(f"{stem}.*"))
    return matches[0] if matches else None


def is_lighter_only(label_file: Path) -> bool:
    classes: set[int] = set()
    for line in label_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        classes.add(int(float(line.split()[0])))
    return classes == {LIGHTER_SRC_ID}


def remap_label(src: Path, dst: Path) -> int:
    lines_out: list[str] = []
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cid = int(float(parts[0]))
        if cid != LIGHTER_SRC_ID:
            continue
        parts[0] = str(LIGHTER_DST_ID)
        lines_out.append(" ".join(parts))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return len(lines_out)


def collect_lighter_items(split: str) -> list[tuple[Path, Path, str]]:
    labels_dir = ATARAH / split / "labels"
    images_dir = ATARAH / split / "images"
    items: list[tuple[Path, Path, str]] = []
    for lab in sorted(labels_dir.glob("*.txt")):
        if not is_lighter_only(lab):
            continue
        img = find_image(images_dir, lab.stem)
        if img is None:
            continue
        items.append((img, lab, lab.stem))
    return items


def next_index(split: str) -> int:
    labels_dir = MERGED / split / "labels"
    if not labels_dir.is_dir():
        return 0
    max_idx = 0
    for lf in labels_dir.glob("atarah_*.txt"):
        try:
            max_idx = max(max_idx, int(lf.stem.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return max_idx


def copy_item(
    split: str,
    idx: int,
    img: Path,
    lab: Path,
    stem: str,
) -> int:
    new_stem = f"atarah_{idx:06d}_{stem}"
    out_img = MERGED / split / "images" / f"{new_stem}{img.suffix.lower()}"
    out_lab = MERGED / split / "labels" / f"{new_stem}.txt"
    out_img.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img, out_img)
    return remap_label(lab, out_lab)


def main() -> None:
    if not MERGED.is_dir():
        raise FileNotFoundError(f"merged_dataset not found: {MERGED}")

    train_items = collect_lighter_items("train") + collect_lighter_items("valid")
    test_items = collect_lighter_items("test")

    rng = random.Random(SEED)
    test_items_shuffled = test_items.copy()
    rng.shuffle(test_items_shuffled)
    valid_items = test_items_shuffled[:VALID_FROM_TEST]
    holdout_test_items = test_items_shuffled[VALID_FROM_TEST:]

    stats = {"train": 0, "valid": 0, "test": 0}
    boxes = {"train": 0, "valid": 0, "test": 0}

    for split, items in (
        ("train", train_items),
        ("valid", valid_items),
        ("test", holdout_test_items),
    ):
        idx = next_index(split)
        for img, lab, stem in items:
            idx += 1
            n_box = copy_item(split, idx, img, lab, stem)
            stats[split] += 1
            boxes[split] += n_box

    print("========== aTarah Lighter merge ==========")
    print(f"train: +{stats['train']} images, +{boxes['train']} boxes (expected 511)")
    print(f"valid: +{stats['valid']} images, +{boxes['valid']} boxes (expected {VALID_FROM_TEST})")
    print(f"test : +{stats['test']} images, +{boxes['test']} boxes (holdout)")
    print(f"total added: {sum(stats.values())} images")


if __name__ == "__main__":
    main()
