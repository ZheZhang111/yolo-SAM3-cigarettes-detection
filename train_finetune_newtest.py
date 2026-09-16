#!/usr/bin/env python3
"""Second fine-tune after adding New_test/image_0828 ground truth."""

from ultralytics import YOLO

DATA = "/data/zhangzhe/Yolo_Cigarettes_dectection/merged_dataset/data.yaml"
WEIGHTS = (
    "/data/zhangzhe/Yolo_Cigarettes_dectection/"
    "runs/detect/cig_multiclass_ft_lighter/weights/best.pt"
)

model = YOLO(WEIGHTS)

model.train(
    data=DATA,
    epochs=30,
    imgsz=1280,
    batch=32,
    device=0,
    workers=16,
    patience=10,
    cache="ram",
    project="/data/zhangzhe/Yolo_Cigarettes_dectection/runs/detect",
    name="cig_multiclass_ft_newtest",
    mosaic=1.0,
    close_mosaic=5,
)
