#!/usr/bin/env python3
"""Add New_test/image_0828 images with corrected labels into merged_dataset/train."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "New_test" / "image_0828"
MERGED = ROOT / "merged_dataset"

# (class_id, x_center, y_center, width, height) — from pred boxes corrected by class
# 0=butts 1=cig-pack 2=cigarette 3=Lighter
ANNOTATIONS: dict[str, list[tuple[int, float, float, float, float]]] = {
    # lighter + pack; top object was mislabeled as cigarettes_butts
    "26ce65209e520107e1c34dd3a7f78a8f.jpg": [
        (3, 0.417933, 0.276527, 0.203343, 0.191097),
        (1, 0.558055, 0.715377, 0.322134, 0.257604),
    ],
    # gold pack only
    "4e069066e24f6e8e5f91db2a272914f6.jpg": [
        (1, 0.487332, 0.501222, 0.207834, 0.405535),
    ],
    # lighter + pack; drop false cigarettes_butts
    "58396953409ca201cdcbb13abecdd4fa.jpg": [
        (3, 0.693888, 0.264058, 0.216813, 0.245743),
        (1, 0.458478, 0.722301, 0.416966, 0.232830),
    ],
    # lighter on pack; vertical pred -> Lighter, horizontal pack estimated
    "cbad7dae5b818e030b2ca74c94473b8b.jpg": [
        (3, 0.450107, 0.500717, 0.132401, 0.386912),
        (1, 0.395000, 0.545000, 0.420000, 0.300000),
    ],
    # lighter only; cig-pack pred relabeled as Lighter
    "fd5e67ebaa6f835a1f943130f1cfd4c0.jpg": [
        (3, 0.572797, 0.348215, 0.330831, 0.255460),
    ],
}


def next_index() -> int:
    labels_dir = MERGED / "train" / "labels"
    max_idx = 0
    for lf in labels_dir.glob("newtest_*.txt"):
        try:
            max_idx = max(max_idx, int(lf.stem.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return max_idx


def main() -> None:
    idx = next_index()
    added = 0
    for fname, boxes in ANNOTATIONS.items():
        src_img = SRC_DIR / fname
        if not src_img.is_file():
            raise FileNotFoundError(src_img)
        idx += 1
        stem = f"newtest_{idx:04d}_{src_img.stem}"
        dst_img = MERGED / "train" / "images" / f"{stem}{src_img.suffix.lower()}"
        dst_lab = MERGED / "train" / "labels" / f"{stem}.txt"
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_img, dst_img)
        lines = [f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for cid, xc, yc, w, h in boxes]
        dst_lab.write_text("\n".join(lines) + "\n", encoding="utf-8")
        added += 1
        print(f"  {fname} -> {stem} ({len(boxes)} boxes)")

    print(f"\nAdded {added} images to merged_dataset/train")


if __name__ == "__main__":
    print("=== Add New_test/image_0828 to train ===")
    main()
