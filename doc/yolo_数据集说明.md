# YOLO 检测所用数据集说明

> 整理本仓库（及关联积水项目）训练 / 评测用到的 YOLO 数据集：**来源、类别、规模、磁盘路径**。  
> 数字以 **2026-09-16** 磁盘实测为准；与早期三源合并文档不一致时，以本文为准。

---

## 1. 总览

| 用途 | 路径 | 说明 |
|------|------|------|
| 源数据（Roboflow 导出） | `/data/zhangzhe/Yolo_Cigarettes_dectection/dataset/` | 烟头 / 烟盒香烟 / 打火机 / aTarah |
| **训练用合并集** | `/data/zhangzhe/Yolo_Cigarettes_dectection/merged_dataset/` | 统一 4 类；`data.yaml` 指向本目录 |
| 实拍增广（写入 train） | `merged_dataset/train/` 下 `newtest_*` | 来自 `New_test/image_0828`，5 张 |
| 推理样例（非训练集） | `/data/zhangzhe/Yolo_Cigarettes_dectection/New_test/` | 机房/手机实拍，用于演示与难例 |
| 积水专用 YOLO | `/data/zhangzhe/Yolo_Water_Leak/dataset/water-leakage--1/` | 单类 Standing Water（另一项目） |

合并脚本：`/data/zhangzhe/Yolo_Cigarettes_dectection/merge_datasets.py`  
训练配置入口：`merged_dataset/data.yaml`

**统一 4 类（合并后）：**

| ID | 英文名 | 中文 |
|----|--------|------|
| 0 | cigarettes_butts | 烟头 |
| 1 | cig-pack | 烟盒 |
| 2 | cigarette | 香烟 |
| 3 | Lighter | 打火机 |

```yaml
# merged_dataset/data.yaml
path: /data/zhangzhe/Yolo_Cigarettes_dectection/merged_dataset
train: train/images
val: valid/images
test: test/images
nc: 4
names: ['cigarettes_butts', 'cig-pack', 'cigarette', 'Lighter']
```

---

## 2. 源数据集（Roboflow → `dataset/`）

根目录：

```text
/data/zhangzhe/Yolo_Cigarettes_dectection/dataset/
├── Yolo Cigarettes Butts.v1i.yolov8/
├── SmokingAndPack.v1i.yolov8/
├── Lighter.v1i.yolov8/
└── aTarah.v1i.yolov8/
```

标准结构：`{train,valid,test}/{images,labels}/` + `data.yaml`。  
许可均为 **CC BY 4.0**（Roboflow Universe）。

### 2.1 烟头 — `Yolo Cigarettes Butts.v1i.yolov8`

| 项 | 内容 |
|----|------|
| 绝对路径 | `/data/zhangzhe/Yolo_Cigarettes_dectection/dataset/Yolo Cigarettes Butts.v1i.yolov8/` |
| Roboflow | workspace `-os52u` · project `yolo-cigarettes-butts-vz7jv` · version 1 |
| URL | https://universe.roboflow.com/-os52u/yolo-cigarettes-butts-vz7jv/dataset/1 |
| 原始类别 | `cigarettes_butts`(0), `others`(1) |
| 合并映射 | **仅保留** butts→统一类 **0**；丢弃 `others` |
| 规模 | train **791** / valid **226** / test **113**（图=标） |

首阶段单类训练（`cig_butts_1280`）直接使用本集。

### 2.2 烟盒与香烟 — `SmokingAndPack.v1i.yolov8`

| 项 | 内容 |
|----|------|
| 绝对路径 | `/data/zhangzhe/Yolo_Cigarettes_dectection/dataset/SmokingAndPack.v1i.yolov8/` |
| Roboflow | `-os52u` · `smokingandpack-stnjr` · v1 |
| URL | https://universe.roboflow.com/-os52u/smokingandpack-stnjr/dataset/1 |
| 原始类别 | `cig-pack`, `cigarette`, `face`, `smoking` |
| 合并映射 | pack→**1**，cigarette→**2**；丢弃 face / smoking |
| 规模 | train **2290** / valid **571** / test **280** |

### 2.3 打火机 — `Lighter.v1i.yolov8`

| 项 | 内容 |
|----|------|
| 绝对路径 | `/data/zhangzhe/Yolo_Cigarettes_dectection/dataset/Lighter.v1i.yolov8/` |
| Roboflow | `-os52u` · `lighter-nvtss-2vq4d` · v1 |
| URL | https://universe.roboflow.com/-os52u/lighter-nvtss-2vq4d/dataset/1 |
| 原始类别 | `Lighter`(0) |
| 合并映射 | Lighter→**3** |
| 规模 | train **86** / valid **11** / test **11**（偏少，后期靠 aTarah + 实拍补） |

### 2.4 杂物多类（补打火机）— `aTarah.v1i.yolov8`

| 项 | 内容 |
|----|------|
| 绝对路径 | `/data/zhangzhe/Yolo_Cigarettes_dectection/dataset/aTarah.v1i.yolov8/` |
| Roboflow | `-os52u` · `atarah-ezziq-jomia` · v1 |
| URL | https://universe.roboflow.com/-os52u/atarah-ezziq-jomia/dataset/1 |
| 原始类别（12 类） | Bucket, Cutter, Electrical Outlet, Knife, **Lighter**, Matches, Nail Cutter, Rocky Road, Scissor, Stairs, Stapler, Swimming Pool |
| 合并用法 | **只抽取 `Lighter`（源 id=4）→ 统一类 3**；其余类丢弃 |
| 规模（全图） | train **5199** / valid **1474** / test **729** |
| 其中 Lighter 框 | train **864** / valid **262** / test **140** |

说明：仓库根目录的 `merge_datasets.py` 当前仍只声明三源（Butts / SmokingAndPack / Lighter）；磁盘上 `merged_dataset` 已额外并入 **aTarah（Lighter）** 与 **newtest_***，故合并规模大于「仅三源」时期。

---

## 3. 合并训练集 — `merged_dataset/`

| 项 | 路径 |
|----|------|
| 根目录 | `/data/zhangzhe/Yolo_Cigarettes_dectection/merged_dataset/` |
| 配置 | `/data/zhangzhe/Yolo_Cigarettes_dectection/merged_dataset/data.yaml` |
| 图片/标签 | `{train,valid,test}/images/` · `{train,valid,test}/labels/` |

### 3.1 当前规模（按框实测）

| split | 图片数 | 总框数 | 烟头(0) | 烟盒(1) | 香烟(2) | 打火机(3) |
|-------|-------:|-------:|--------:|--------:|--------:|----------:|
| train | 3676 | 5175 | 1319 | 901 | 1713 | 1242 |
| valid | 829 | 1070 | 377 | 132 | 479 | 82 |
| test | 429 | 617 | 220 | 72 | 240 | 85 |
| **合计** | **4934** | **6862** | **1916** | **1105** | **2432** | **1409** |

全库框占比：香烟 35.4% · 烟头 27.9% · 打火机 20.5% · 烟盒 16.1%。  
**正式评测用 test**：429 图 / 617 框（Roboflow 导出重映射，**不含** `newtest_*`）。

### 3.2 train 文件名前缀（溯源）

合并后标签 stem 带源前缀，例如：

| 前缀 | 来源 | train 约略张数 |
|------|------|----------------|
| `Yolo_Cigarettes_Butts_*` | 烟头集 | 791 |
| `SmokingAndPack_*` | 烟盒/香烟集 | 2283 |
| `Lighter_*` | 打火机小集 | 86 |
| `atarah_*` | aTarah 中保留含 Lighter 的图 | 511 |
| `newtest_*` | 机房实拍 image_0828 | **5** |

### 3.3 实拍增广 `newtest_*`

| 项 | 说明 |
|----|------|
| 源图目录 | `/data/zhangzhe/Yolo_Cigarettes_dectection/New_test/image_0828/`（5 张 JPG） |
| 写入位置 | `merged_dataset/train/images|labels/newtest_0001~0005_*` |
| 用途 | 打火机/烟盒难例微调（`cig_multiclass_ft_newtest`） |
| 注意 | GT 曾按预测框校正，**有乐观偏差**；仅进 train，**不作正式 test** |

---

## 4. 推理 / 难例目录（一般不用于训练）

根路径：`/data/zhangzhe/Yolo_Cigarettes_dectection/New_test/`

| 子目录 | 用途（简要） |
|--------|----------------|
| `image_0828/` | 早期打火机误检难例（5 张） |
| `image_0828-v2/` | 含 Roboflow 导出 `Cigarette Lighter.v1i.yolov8/`（见下） |
| `image_0901/` · `image_0901_v2/` | 烟盒/打火机实拍；YOLO / SAM3 对比 |
| `image_0907_wire/` | 绝缘破损（SAM3 为主） |
| `image-0914/` | 积水 / 烟雾样例 |

### 4.1 附加导出：`Cigarette Lighter.v1i.yolov8`（v2 目录内）

| 项 | 内容 |
|----|------|
| 路径 | `/data/zhangzhe/Yolo_Cigarettes_dectection/New_test/image_0828-v2/Cigarette Lighter.v1i.yolov8/` |
| `data.yaml` 声明 | nc=4，与统一类名一致；图多放在 `test/` |
| 备注 | 小样本补充/对照用，**不是** `merged_dataset` 主训练来源 |

---

## 5. 关联：积水 YOLO 数据集

香烟四类之外，巡检里积水走**独立 YOLO 项目**（非 `merged_dataset`）。

| 项 | 内容 |
|----|------|
| 项目根 | `/data/zhangzhe/Yolo_Water_Leak/` |
| 数据集 | `/data/zhangzhe/Yolo_Water_Leak/dataset/water-leakage--1/` |
| Roboflow | `-os52u` · `water-leakage-khoez` · v1 |
| URL | https://universe.roboflow.com/-os52u/water-leakage-khoez/dataset/1 |
| 类别 | 单类 `Standing Water`（nc=1） |
| 规模 | train **1958** / valid **1240** / test **344** |
| 部署权重 | `/data/zhangzhe/Yolo_Water_Leak/runs/detect/water_leak_yolov8m/weights/best.pt` |

---

## 6. 与训练权重的对应关系

| 阶段 | 实验名 | 主要数据 | 权重路径 |
|------|--------|----------|----------|
| ① 烟头单类 | `cig_butts_1280` | Butts 源集 | `.../runs/detect/cig_butts_1280/weights/best.pt` |
| ② 四类合并 | `cig_multiclass` | `merged_dataset`（早期三源） | `.../runs/detect/runs/cig_multiclass/weights/best.pt` |
| ③ 打火机微调 | `cig_multiclass_ft_lighter` | 补 Lighter 后 | `.../cig_multiclass_ft_lighter/weights/best.pt` |
| ④ 实拍难例 | `cig_multiclass_ft_newtest` | + `newtest_*` | **`.../cig_multiclass_ft_newtest/weights/best.pt`（部署用）** |

完整绝对前缀：`/data/zhangzhe/Yolo_Cigarettes_dectection/runs/detect/`。

验证 / 评测常用：

```bash
# 正式 test（429 图）
yolo detect val \
  model=runs/detect/cig_multiclass_ft_newtest/weights/best.pt \
  data=merged_dataset/data.yaml imgsz=1280 split=test
```

---

## 7. 路径速查（复制用）

```text
# 四类 YOLO 工程根
/data/zhangzhe/Yolo_Cigarettes_dectection/

# 源
/data/zhangzhe/Yolo_Cigarettes_dectection/dataset/Yolo Cigarettes Butts.v1i.yolov8/
/data/zhangzhe/Yolo_Cigarettes_dectection/dataset/SmokingAndPack.v1i.yolov8/
/data/zhangzhe/Yolo_Cigarettes_dectection/dataset/Lighter.v1i.yolov8/
/data/zhangzhe/Yolo_Cigarettes_dectection/dataset/aTarah.v1i.yolov8/

# 合并训练
/data/zhangzhe/Yolo_Cigarettes_dectection/merged_dataset/
/data/zhangzhe/Yolo_Cigarettes_dectection/merged_dataset/data.yaml

# 推理样例
/data/zhangzhe/Yolo_Cigarettes_dectection/New_test/

# 积水
/data/zhangzhe/Yolo_Water_Leak/dataset/water-leakage--1/
```

相关文档：

- `doc/机房巡检-数据集训练测试与VLM仲裁.md` — 训练指标与 VLM 仲裁  
- `doc/multiclass_training_guide.md` — 早期三源合并流程  
- `doc/training_guide.md` — 烟头单类训练  

---

*文档生成：2026-09-16 · 路径：`doc/yolo_数据集说明.md`*
