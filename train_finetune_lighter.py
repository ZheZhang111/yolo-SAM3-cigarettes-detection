#!/usr/bin/env python3
"""Fine-tune multiclass model after adding aTarah Lighter data."""

from ultralytics import YOLO

DATA = "/data/zhangzhe/Yolo_Cigarettes_dectection/merged_dataset/data.yaml"
WEIGHTS = (
    "/data/zhangzhe/Yolo_Cigarettes_dectection/"
    "runs/detect/runs/cig_multiclass/weights/best.pt"
)

model = YOLO(WEIGHTS)

model.train(
    data=DATA,
    epochs=50,
    imgsz=1280,
    batch=32,
    device=0,  # use with CUDA_VISIBLE_DEVICES
    workers=16,
    patience=15,
    cache="ram",
    project="/data/zhangzhe/Yolo_Cigarettes_dectection/runs/detect",
    name="cig_multiclass_ft_lighter",
    mosaic=1.0,
    close_mosaic=10,
)
