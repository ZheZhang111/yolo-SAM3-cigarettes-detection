# 烟头检测 YOLO 训练说明

## 1. 项目概述

本项目基于 **Ultralytics YOLOv8** 对 Roboflow 上的烟头（cigarette butts）数据集进行目标检测训练，产出可用于部署的 `best.pt` 权重。

| 项 | 说明 |
|---|---|
| 项目路径 | `/data/zhangzhe/Yolo_Cigarettes_dectection` |
| 任务类型 | 目标检测（detect） |
| 预训练模型 | **YOLOv8m**（`yolov8m.pt`） |
| 框架 | Ultralytics 8.4.130 |
| 后端 | PyTorch 2.5.1+cu121 |

---

## 2. 数据集基本情况

### 2.1 来源与格式

| 项 | 说明 |
|---|---|
| 来源 | [Roboflow Universe](https://universe.roboflow.com/-os52u/yolo-cigarettes-butts-vz7jv) |
| Workspace | `-os52u` |
| Project | `yolo-cigarettes-butts-vz7jv` |
| Version | 1 |
| 许可 | CC BY 4.0 |
| 标注格式 | YOLOv8（images + labels 目录结构） |

### 2.2 数据划分

| 划分 | 图片数 | 标注数 |
|---|---:|---:|
| train | 791 | 791 |
| valid | 226 | 226 |
| test | 113 | 113 |
| **合计** | **1130** | **1130** |

### 2.3 类别

共 **2 类**（`nc: 2`）：

| ID | 类别名 | 说明 |
|---|---|---|
| 0 | `cigarettes_butts` | 烟头（主要检测目标） |
| 1 | `others` | 其他 |

### 2.4 目录结构

```
dataset/Yolo Cigarettes Butts.v1i.yolov8/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### 2.5 data.yaml 配置

路径：`dataset/Yolo Cigarettes Butts.v1i.yolov8/data.yaml`

```yaml
train: train/images
val: valid/images
test: test/images

nc: 2
names: ['cigarettes_butts', 'others']
```

> 注意：路径为相对 `data.yaml` 所在目录，不要使用 `../train/images` 这类多一层 `..` 的写法。

### 2.6 数据特点

- 总量约 1130 张，规模较小
- 烟头属于**小目标**，对输入分辨率（`imgsz`）较敏感
- 原数据集 baseline mAP 约 52%，可通过更高分辨率与更大模型提升

---

## 3. 训练环境与硬件

### 3.1 Python 虚拟环境

环境安装在项目目录内（不在 home）：

```
/data/zhangzhe/Yolo_Cigarettes_dectection/.conda/cig
```

| 组件 | 版本 |
|---|---|
| Python | 3.10 |
| PyTorch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| ultralytics | 8.4.130 |

激活环境：

```bash
conda activate /data/zhangzhe/Yolo_Cigarettes_dectection/.conda/cig
```

### 3.2 GPU

| 项 | 说明 |
|---|---|
| 型号 | NVIDIA H800 × 8 |
| 显存 | 约 80 GB / 卡 |
| 训练用卡 | **第 5 张卡**（`device=4`，0 起算） |

---

## 4. 训练参数

### 4.1 参数一览

| 参数 | 值 | 说明 |
|---|---|---|
| `model` | `yolov8m.pt` | 中等规模，精度与速度平衡；H800 显存充足 |
| `epochs` | 200 | 总训练轮数 |
| `imgsz` | 1280 | 输入分辨率，利于小目标检测 |
| `batch` | 32 | H800 80GB 可承受；OOM 时降至 16 |
| `device` | 4 | 使用第 5 张 GPU |
| `workers` | 16 | 数据加载线程数 |
| `patience` | 40 | 早停：40 轮无提升则停止 |
| `cache` | ram | 全量缓存到内存，加速 IO |
| `name` | `cig_butts_1280` | 实验名称 |
| `mosaic` | 1.0 | Mosaic 数据增强 |
| `close_mosaic` | 20 | 最后 20 轮关闭 mosaic，稳定收敛 |

### 4.2 选型理由

- **YOLOv8m**：比 `s` 更准，H800 显存足够
- **imgsz=1280**：烟头为小目标，高分辨率有助于 recall / mAP
- **batch=32**：1280 分辨率下 H800 约占用 45–60 GB 显存
- **cache=ram**：1130 张图可全部载入内存，减少数据读取瓶颈

### 4.3 预期训练时长

H800 + 1130 张 + `imgsz=1280`，约 **20–40 分钟**（含早停可能更早结束）。

---

## 5. 如何启动训练

### 5.1 推荐：tmux 后台训练（断线/关电脑继续）

```bash
# 新建 tmux 会话
tmux new -s cig

# 激活环境
conda activate /data/zhangzhe/Yolo_Cigarettes_dectection/.conda/cig

# 进入项目目录
cd /data/zhangzhe/Yolo_Cigarettes_dectection

# 启动训练（第 5 张 GPU，device=4）
yolo detect train \
  data="dataset/Yolo Cigarettes Butts.v1i.yolov8/data.yaml" \
  model=yolov8m.pt \
  epochs=200 \
  imgsz=1280 \
  batch=32 \
  device=4 \
  workers=16 \
  patience=40 \
  cache=ram \
  name=cig_butts_1280 \
  mosaic=1.0 \
  close_mosaic=20
```

**tmux 常用操作：**

| 操作 | 命令 / 快捷键 |
|---|---|
| 脱离会话（训练继续） | `Ctrl+b` 然后 `d` |
| 重新进入 | `tmux attach -t cig` |
| 查看会话列表 | `tmux ls` |
| 结束会话 | `tmux kill-session -t cig` |

### 5.2 可选：限定可见 GPU

若希望进程只能看到第 5 张物理卡：

```bash
CUDA_VISIBLE_DEVICES=4 yolo detect train \
  data="dataset/Yolo Cigarettes Butts.v1i.yolov8/data.yaml" \
  model=yolov8m.pt \
  epochs=200 \
  imgsz=1280 \
  batch=32 \
  device=0 \
  workers=16 \
  patience=40 \
  cache=ram \
  name=cig_butts_1280 \
  mosaic=1.0 \
  close_mosaic=20
```

此时 `CUDA_VISIBLE_DEVICES=4` 将物理 GPU 4 映射为逻辑 `device=0`。

### 5.3 开训前检查

```bash
# 确认 GPU 空闲
nvidia-smi

# 确认环境与 CUDA
conda activate /data/zhangzhe/Yolo_Cigarettes_dectection/.conda/cig
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(4))"
```

---

## 6. 训练产出

### 6.1 权重路径

```
runs/detect/cig_butts_1280/weights/best.pt   # 验证集最优，部署使用
runs/detect/cig_butts_1280/weights/last.pt   # 最后一轮，用于断点续训
```

### 6.2 其他产物

```
runs/detect/cig_butts_1280/
├── results.csv          # 指标曲线数据
├── results.png          # 训练曲线
├── confusion_matrix.png
├── val_batch*.jpg       # 验证可视化
└── args.yaml            # 本次训练完整参数快照
```

---

## 7. 训练结果（2026-08-27 实测）

### 7.1 训练概况

| 项 | 结果 |
|---|---|
| 实验名称 | `cig_butts_1280` |
| 开始时间 | 2026-08-27 10:54 |
| 结束时间 | 2026-08-27 11:14 |
| 计划 epoch | 200 |
| 实际 epoch | **87**（早停触发） |
| 最佳 epoch | **第 47 轮**（保存为 `best.pt`） |
| 总耗时 | **0.271 小时（约 16 分钟）** |
| 使用 GPU | 第 5 张卡（`device=4`） |
| 权重大小 | `best.pt` / `last.pt` 各约 50 MB |

### 7.2 最佳轮次指标（epoch 47，验证集）

| 指标 | 值 |
|---|---:|
| Precision (B) | 0.672 |
| Recall (B) | 0.730 |
| mAP50 (B) | **0.682** |
| mAP50-95 (B) | 0.434 |

### 7.3 训练结束时指标（epoch 87，验证集）

| 类别 | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| **all** | 226 | 412 | 0.672 | 0.730 | **0.681** | 0.434 |
| cigarettes_butts | 226 | 377 | 0.845 | 0.859 | **0.889** | 0.648 |
| others | 9 | 35 | 0.499 | 0.600 | 0.472 | 0.220 |

推理速度（训练内置 val）：preprocess 0.3ms / inference 2.4ms / postprocess 0.3ms per image

### 7.4 与 baseline 对比

| 指标 | 数据集 baseline | 本次训练（best） |
|---|---:|---:|
| mAP50（整体） | ~52% | **68.1%** |
| mAP50（cigarettes_butts） | — | **88.9%** |

### 7.5 训练曲线与可视化

路径：`runs/detect/cig_butts_1280/`

| 文件 | 说明 |
|---|---|
| `results.csv` | 逐 epoch 损失与 mAP 数据 |
| `results.png` | 训练曲线总览 |
| `confusion_matrix.png` | 混淆矩阵 |
| `confusion_matrix_normalized.png` | 归一化混淆矩阵 |
| `val_batch0_pred.jpg` 等 | 验证集预测可视化 |

---

## 8. 验证结果（2026-08-27 实测）

使用 `best.pt` 在验证集上单独运行 `yolo detect val`：

```bash
yolo detect val \
  model=runs/detect/cig_butts_1280/weights/best.pt \
  data="dataset/Yolo Cigarettes Butts.v1i.yolov8/data.yaml" \
  imgsz=1280 \
  device=4
```

### 8.1 验证指标

| 类别 | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| **all** | 226 | 412 | 0.692 | 0.698 | **0.685** | 0.439 |
| cigarettes_butts | 226 | 377 | 0.852 | 0.854 | **0.892** | 0.648 |
| others | 9 | 35 | 0.532 | 0.543 | 0.477 | 0.229 |

推理速度：preprocess 3.8ms / inference 5.3ms / postprocess 1.6ms per image

### 8.2 验证产出

路径：`runs/detect/val/`

| 文件 | 说明 |
|---|---|
| `val_batch0_labels.jpg` / `val_batch0_pred.jpg` | 标注 vs 预测对比 |
| `confusion_matrix.png` | 混淆矩阵 |
| `BoxP_curve.png` / `BoxR_curve.png` / `BoxPR_curve.png` | P/R/PR 曲线 |

### 8.3 结果分析

- **cigarettes_butts（烟头）**：mAP50 达 **0.892**，Precision / Recall 均 > 0.85，主任务效果良好。
- **others**：验证集仅 9 张图 / 35 个实例，样本极少，指标偏低属预期现象；若部署只需检烟头，可考虑 `classes=0` 过滤或改为单类训练。

---

## 9. 推理结果（2026-08-27 实测）

在 **test 集（113 张）** 上运行批量推理：

```bash
yolo detect predict \
  model=runs/detect/cig_butts_1280/weights/best.pt \
  source="dataset/Yolo Cigarettes Butts.v1i.yolov8/test/images" \
  conf=0.25 \
  save=True \
  device=4
```

### 9.1 推理概况

| 项 | 结果 |
|---|---|
| 输入 | test/images，共 **113 张** |
| 置信度阈值 | 0.25 |
| 输入尺寸 | 736×1280（letterbox 后） |
| 输出目录 | `runs/detect/predict/` |
| 输出图片数 | **113 张**（带检测框可视化） |

### 9.2 推理速度

| 阶段 | 耗时 |
|---|---:|
| preprocess | 4.2 ms / image |
| inference | 9.4 ms / image |
| postprocess | 4.8 ms / image |
| **合计** | **约 18.4 ms / image**（~54 FPS） |

### 9.3 终端输出示例

```
image 109/113 .../test/images/kvIIju4fG6G59MB0F3sE_....jpg: 736x1280 1 cigarettes_butts, 6.9ms
image 110/113 .../test/images/l857Tf0AyP3wQy0lkA6E_....jpg: 736x1280 1 cigarettes_butts, 6.2ms
image 111/113 .../test/images/lQJhL361s4RIQytdyHEH_....jpg: 736x1280 1 cigarettes_butts, 5.0ms
image 112/113 .../test/images/r8uaDHaEuZNqlozd340r_....jpg: 736x1280 1 cigarettes_butts, 7.6ms
image 113/113 .../test/images/t3s7r7tiHI9uPjZB9738_....jpg: 736x1280 1 cigarettes_butts, 5.7ms
Speed: 4.2ms preprocess, 9.4ms inference, 4.8ms postprocess per image at shape (1, 3, 736, 1280)
Results saved to /data/zhangzhe/Yolo_Cigarettes_dectection/runs/detect/predict
```

### 9.4 推理产出示例

路径：`runs/detect/predict/`

部分输出文件名示例：

- `cig_butts1_001_jpg.rf.4636b7cabd320fb07f334e8f805c352f.jpg`
- `cigbutt00015_jpg.rf.15a2c82dea137e72d8cb0fe708f6d308.jpg`
- `5wadmVWy612ABJXxittv_jpg.rf.1e074d0c05269f5f4a458bd8955eabca.jpg`

---

## 10. 断点续训

训练中断后，从 `last.pt` 继续：

```bash
conda activate /data/zhangzhe/Yolo_Cigarettes_dectection/.conda/cig
cd /data/zhangzhe/Yolo_Cigarettes_dectection

yolo detect train resume model=runs/detect/cig_butts_1280/weights/last.pt
```

---

## 11. 验证与推理命令

### 11.1 验证集评估

```bash
yolo detect val \
  model=runs/detect/cig_butts_1280/weights/best.pt \
  data="dataset/Yolo Cigarettes Butts.v1i.yolov8/data.yaml" \
  imgsz=1280 \
  device=4
```

### 11.2 单张 / 批量推理

```bash
# 单张图片
yolo detect predict \
  model=runs/detect/cig_butts_1280/weights/best.pt \
  source=test.jpg \
  conf=0.25 \
  save=True \
  device=4

# 测试集批量
yolo detect predict \
  model=runs/detect/cig_butts_1280/weights/best.pt \
  source="dataset/Yolo Cigarettes Butts.v1i.yolov8/test/images" \
  conf=0.25 \
  save=True \
  device=4
```

推理结果默认保存在 `runs/detect/predict/`。

---

## 12. 常见问题

| 问题 | 处理 |
|---|---|
| `CUDA out of memory` | 将 `batch` 从 32 降到 16 或 8 |
| 找不到图片 | 检查 `data.yaml` 路径是否为 `train/images`（非 `../train/images`） |
| 训练中断 | 使用 `resume` 从 `last.pt` 续训 |
| 想进一步提高精度 | 尝试 `imgsz=1536`、`batch=16`，或换 `yolov8l.pt` |

---

*文档创建：2026-08-27 | 训练/验证/推理结果更新：2026-08-27*
