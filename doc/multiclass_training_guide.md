# 烟头/烟盒/香烟/打火机 多类检测训练说明

本文档记录 **路线 A**：将三个 Roboflow 数据集合并为 4 类，用已训好的烟头模型 `best.pt` 做迁移学习起点，重新训练统一检测模型的完整流程。

相关文件：

| 文件 | 说明 |
|---|---|
| `doc/training_guide.md` | 第一阶段：仅烟头检测（`cig_butts_1280`） |
| `doc/multiclass_training_guide.md` | 本文：四类合并训练（`cig_multiclass`） |
| `merge_datasets.py` | 三源合并脚本 |
| `train_multiclass.py` | 多类训练入口 |
| `merged_dataset/` | 合并后的训练数据 |

---

## 1. 目标与策略

**目标**：一个模型同时检测烟头、烟盒、香烟、打火机。

**策略（迁移学习，不是增量加类）**：

1. 合并三个 YOLO 数据集，统一为 4 类标签；
2. 以第一阶段烟头模型 `runs/detect/cig_butts_1280/weights/best.pt` 为初始权重；
3. Ultralytics 按新 `nc=4` **重建检测头**，主干特征被继承；
4. 比从 COCO 预训练 `yolov8m.pt` 冷启动更快、通常更稳。

日志关键信息：

```
Overriding model.yaml nc=2 with nc=4
Remapped 1/4 cls head rows from pretrained weights by class name
Transferred 475/475 items from pretrained weights
```

---

## 2. 源数据集

根目录：`/data/zhangzhe/Yolo_Cigarettes_dectection/dataset/`

| 源目录 | Roboflow 项目 | 原类别 | 合并时处理 |
|---|---|---|---|
| `Yolo Cigarettes Butts.v1i.yolov8` | yolo-cigarettes-butts-vz7jv | `cigarettes_butts`, `others` | 只保留 `cigarettes_butts` → 0；丢弃 `others` |
| `SmokingAndPack.v1i.yolov8` | smokingandpack-stnjr | `cig-pack`, `cigarette`, `face`, `smoking` | `cig-pack`→1，`cigarette`→2；丢弃 `face`/`smoking` |
| `Lighter.v1i.yolov8` | lighter-nvtss-2vq4d | `Lighter` | `Lighter` → 3 |

各源原始规模（合并前）：

| 源 | train | valid | test |
|---|---:|---:|---:|
| Butts | 791 | 226 | 113 |
| SmokingAndPack | 2290 | 571 | 280 |
| Lighter | 86 | 11 | 11 |

目录均为 Roboflow 标准结构：`{train,valid,test}/{images,labels}/`。

---

## 3. 统一类别定义

```yaml
nc: 4
names: ['cigarettes_butts', 'cig-pack', 'cigarette', 'Lighter']
```

| ID | 类名 | 含义 |
|---|---|---|
| 0 | `cigarettes_butts` | 烟头 |
| 1 | `cig-pack` | 烟盒 |
| 2 | `cigarette` | 香烟（整支/未吸完等） |
| 3 | `Lighter` | 打火机 |

---

## 4. 合并流程

### 4.1 脚本

```bash
cd /data/zhangzhe/Yolo_Cigarettes_dectection
conda activate /data/zhangzhe/Yolo_Cigarettes_dectection/.conda/cig   # 或用该 env 的 python
python merge_datasets.py
```

脚本行为：

- 按源重映射类别 ID，丢弃不在统一表中的框；
- 仅当重映射后仍有框时，才复制对应图片；
- 文件名加源前缀，避免跨数据集重名冲突；
- 写出 `merged_dataset/data.yaml`（含绝对 `path`）。

### 4.2 合并后规模（实测）

| split | 图片数 | 框数 |
|---|---:|---:|
| train | 3160 | 4041 |
| valid | 804 | 1002 |
| test | 401 | 545 |
| **合计** | **4365** | **5588** |

### 4.3 各类框数分布（重点）

**train：**

| ID | 类名 | 框数 |
|---|---|---:|
| 0 | cigarettes_butts | 1319 |
| 1 | cig-pack | 897 |
| 2 | cigarette | 1713 |
| 3 | Lighter | **112** |

**valid / test：**

| 类名 | valid | test |
|---|---:|---:|
| cigarettes_butts | 377 | 220 |
| cig-pack | 132 | 72 |
| cigarette | 479 | 240 |
| Lighter | **14** | **13** |

> **Lighter 样本极少**，预期该类 mAP 最弱。`cigarette` 与 `cigarettes_butts` 外观相近，可能互相误检。

快速复检：

```bash
for split in train valid test; do
  echo "=== $split ==="
  cat merged_dataset/$split/labels/*.txt | awk '{print $1}' | sort | uniq -c
done
```

### 4.4 数据质量备注

- SmokingAndPack 中少量标签混有 segment 行，训练时 Ultralytics 忽略约 3 张 corrupt 样本，并提示只用 boxes、去掉 segments。
- 仅含 `face`/`smoking` 的图在合并时被跳过（无保留框）。

---

## 5. 模型

| 项 | 说明 |
|---|---|
| 架构 | YOLOv8m |
| 第一阶段预训练 | `yolov8m.pt` → 烟头二类微调 |
| 第一阶段权重 | `runs/detect/cig_butts_1280/weights/best.pt`（约 50MB） |
| 第二阶段起点 | 上述 `best.pt`（`nc=2` → 覆盖为 `nc=4`） |
| 框架 | Ultralytics 8.4.130 |
| 后端 | PyTorch 2.5.1+cu121 |
| 参数量 | ~25.9M（Detect 头 4 类） |

环境：

```
/data/zhangzhe/Yolo_Cigarettes_dectection/.conda/cig
```

激活：

```bash
conda activate /data/zhangzhe/Yolo_Cigarettes_dectection/.conda/cig
```

---

## 6. 训练参数

入口脚本：`train_multiclass.py`

| 参数 | 值 | 说明 |
|---|---|---|
| `model` | `cig_butts_1280/weights/best.pt` | 迁移起点 |
| `data` | `merged_dataset/data.yaml` | 四类合并集 |
| `epochs` | 200 | 最大轮数 |
| `imgsz` | 1280 | 小目标友好 |
| `batch` | 32 | H800 80GB |
| `device` | 4 | 第 5 张 GPU（0 起算） |
| `workers` | 16 | DataLoader 线程 |
| `patience` | 40 | 早停 |
| `cache` | ram | 全量进内存 |
| `mosaic` | 1.0 | 训练期 mosaic |
| `close_mosaic` | 20 | 最后 20 轮关闭 mosaic |
| `name` | `cig_multiclass` | 实验名 |

### 6.1 启动命令（tmux，断线可续）

```bash
tmux new -s cig_mc

conda activate /data/zhangzhe/Yolo_Cigarettes_dectection/.conda/cig
cd /data/zhangzhe/Yolo_Cigarettes_dectection
python train_multiclass.py 2>&1 | tee train_multiclass.log

# 脱离：Ctrl+b 然后 d
# 重连：tmux attach -t cig_mc
```

等价 CLI（参考）：

```bash
yolo detect train \
  model=runs/detect/cig_butts_1280/weights/best.pt \
  data=merged_dataset/data.yaml \
  epochs=200 imgsz=1280 batch=32 device=4 \
  workers=16 patience=40 cache=ram \
  name=cig_multiclass mosaic=1.0 close_mosaic=20
```

### 6.2 本次实际运行记录（2026-08-27）

| 项 | 值 |
|---|---|
| 启动方式 | tmux 会话 `cig_mc` |
| 日志 | `train_multiclass.log` |
| GPU | CUDA:4，NVIDIA H800 |
| AMP | 已开启并通过检查 |
| 权重落盘（本次） | `runs/detect/runs/cig_multiclass/weights/` |

> 说明：本次 `train_multiclass.py` 初版 `project="runs"`，Ultralytics 默认再套一层 `runs/detect/`，故实际目录为 `runs/detect/runs/cig_multiclass/`。脚本已改为显式 `project=.../runs/detect`，**下次重跑**会落到 `runs/detect/cig_multiclass/`。

断点续训：

```bash
yolo detect train resume \
  model=runs/detect/runs/cig_multiclass/weights/last.pt
```

---

## 7. 产出路径

### 7.1 权重（本次）

```
runs/detect/runs/cig_multiclass/weights/best.pt   # 部署用
runs/detect/runs/cig_multiclass/weights/last.pt   # 续训用
```

### 7.2 曲线与可视化

```
runs/detect/runs/cig_multiclass/
├── results.csv / results.png
├── confusion_matrix.png
├── args.yaml
└── val_batch*_pred.jpg
```

---

## 8. 验证与推理（训完必做）

### 8.1 按类看测试集指标

```bash
conda activate /data/zhangzhe/Yolo_Cigarettes_dectection/.conda/cig
cd /data/zhangzhe/Yolo_Cigarettes_dectection

yolo detect val \
  model=runs/detect/runs/cig_multiclass/weights/best.pt \
  data=merged_dataset/data.yaml \
  imgsz=1280 \
  split=test \
  device=4
```

日志会逐类打印 Precision / Recall / mAP50 / mAP50-95。重点看：

- `Lighter` 是否因样本少拖后腿；
- `cigarette` vs `cigarettes_butts` 是否混淆（看 confusion matrix）。

### 8.2 批量推理

```bash
yolo detect predict \
  model=runs/detect/runs/cig_multiclass/weights/best.pt \
  source=merged_dataset/test/images \
  conf=0.25 \
  save=True \
  device=4
```

---

## 9. 端到端流程清单

1. 准备三个源数据集目录（Butts / SmokingAndPack / Lighter）
2. 准备第一阶段烟头权重 `cig_butts_1280/weights/best.pt`
3. `python merge_datasets.py` → 检查各类框数
4. `python train_multiclass.py`（tmux / GPU 4）
5. `yolo detect val ... split=test` 看每类指标
6. `yolo detect predict` 抽查可视化
7. 部署使用 `best.pt`

---

## 10. 已知风险与后续可选项

| 风险 | 应对 |
|---|---|
| Lighter 框过少（train 112） | 增补数据；或对该类降低 conf / 单独微调 |
| cigarette ↔ cigarettes_butts 混淆 | 查混淆矩阵；必要时收紧标注定义或加 hard negative |
| 类不平衡 | 可试 `cls` 损失加权、过采样少数类图 |
| 混有 segment 标签 | 已用检测框；导出时尽量只选 detect 格式 |

---

## 11. 与第一阶段对照

| | 烟头单任务 | 四类合并 |
|---|---|---|
| 实验名 | `cig_butts_1280` | `cig_multiclass` |
| 类别数 | 2（实际主用 butts；others 弱） | 4（统一表） |
| 训练图 | ~791 train | 3160 train |
| 初始权重 | `yolov8m.pt` | 烟头 `best.pt` |
| imgsz / batch | 1280 / 32 | 同左 |
| 烟头 valid mAP50（一阶段） | ~0.89 | 待本次 val 确认 |

---

*文档创建：2026-08-27 | 对应合并与训练启动当日*
