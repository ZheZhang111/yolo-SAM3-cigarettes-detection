#!/usr/bin/env python3
"""Multiclass train: butts + pack + cigarette + lighter, init from cig_butts best.pt."""

from ultralytics import YOLO

DATA = "/data/zhangzhe/Yolo_Cigarettes_dectection/merged_dataset/data.yaml"
WEIGHTS = "/data/zhangzhe/Yolo_Cigarettes_dectection/runs/detect/cig_butts_1280/weights/best.pt"

model = YOLO(WEIGHTS)

model.train(
    data=DATA,
    epochs=200,
    imgsz=1280,
    batch=32,
    device=4,  # free H800 (same as previous butts run)
    workers=16,
    patience=40,
    cache="ram",
    project="/data/zhangzhe/Yolo_Cigarettes_dectection/runs/detect",
    name="cig_multiclass",
    mosaic=1.0,
    close_mosaic=20,
)
