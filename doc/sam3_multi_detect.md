# SAM3 多任务检测使用说明

本文说明项目中基于 **Meta SAM3** 的开放词汇检测脚本：用途、环境启动与运行指令。

---

## 1. 代码作用

SAM3 用**文本提示（prompt）**在图中查找目标，**无需**为新类别单独训练 YOLO。本仓库提供：

| 脚本 | 作用 |
|------|------|
| `sam3_detect_multi.py` | **推荐**：多任务合一。可检测 **烟盒/打火机/烟头**、**电线绝缘破损**、**积水/火花/烟雾** |
| `sam3_detect_cig.py` | 仅香烟相关三类（烟盒、打火机、烟头） |
| `bench_sam3_timing.py` | SAM3 推理耗时评测（预热 + CUDA 同步，不画图） |
| `run_sam3_predict.py` | 早期烟盒+打火机试跑脚本（可被 multi 替代） |

### `sam3_detect_multi.py` 能力概要

- **香烟类（`--tasks cig`）**：对每个类别尝试多组英文短语，NMS 后保留每类 Top-K 框。  
- **绝缘破损（`--tasks wire`）**：用「外露铜丝 / 外露芯线」等短语找**局部破损**；过滤过大整缆框，默认每图 **1** 个破损框，并给出「绝缘层损坏 / 未检出」横幅。  
- **多任务（`--tasks cig,wire`）**：同一张图上同时跑两类任务，框分色标注，输出 `*_sam3_multi.jpg` + `results.txt`。

输出目录示例：`runs/detect/New_test/sam3_multi_*/`

---

## 2. 环境启动

### 2.1 Conda 环境

```bash
conda activate sam3
# 或
export PATH=/data/zhangzhe/conda/envs/sam3/bin:$PATH
```

环境路径：`/data/zhangzhe/conda/envs/sam3`  
依赖要点：PyTorch（CUDA）、SAM3 代码库所需包（如 `iopath`、`einops`、`pycocotools` 等）。

### 2.2 权重与代码路径（脚本内默认）

| 项 | 路径 |
|----|------|
| SAM3 权重 | `/data/zhangzhe/do-as-i-do/reconstruction/checkpoints/sam3/sam3.pt` |
| SAM3 源码 | `/data/zhangzhe/do-as-i-do/reconstruction/modules/sam3`（脚本自动加入 `sys.path`） |
| 项目根目录 | `/data/zhangzhe/Yolo_Cigarettes_dectection` |

### 2.3 GPU

建议指定空闲卡，例如物理 GPU 5：

```bash
CUDA_VISIBLE_DEVICES=5
```

脚本内使用 `device="cuda"`（即可见的第 0 张逻辑卡）。

### 2.4 进入项目目录

```bash
cd /data/zhangzhe/Yolo_Cigarettes_dectection
```

---

## 3. 运行指令

### 3.1 多任务（香烟 + 绝缘破损）——推荐入口

```bash
conda activate sam3
cd /data/zhangzhe/Yolo_Cigarettes_dectection

CUDA_VISIBLE_DEVICES=5 python sam3_detect_multi.py \
  --source New_test/image_0907_wire \
  --out-dir runs/detect/New_test/sam3_multi_all \
  --tasks cig,wire \
  --conf 0.25 \
  --wire-conf 0.15 \
  --max-per-class 2 \
  --max-damage-boxes 1
```

### 3.2 仅香烟类（烟盒 / 打火机 / 烟头）

```bash
CUDA_VISIBLE_DEVICES=5 python sam3_detect_multi.py \
  --source New_test/image_0901_v2 \
  --out-dir runs/detect/New_test/sam3_multi_cig \
  --tasks cig
```

或只用旧脚本：

```bash
CUDA_VISIBLE_DEVICES=5 python sam3_detect_cig.py \
  --source New_test/image_0901_v2 \
  --out-dir runs/detect/New_test/sam3_cig_demo \
  --conf 0.25 \
  --max-per-class 2
```

### 3.3 积水 / 火花 / 烟雾

```bash
# 三项一起
CUDA_VISIBLE_DEVICES=5 python sam3_detect_multi.py \
  --source New_test/image-0914 \
  --out-dir runs/detect/New_test/image-0914_sam3 \
  --tasks water,spark,smoke \
  --hazard-conf 0.20 \
  --max-per-class 3

# 或 --tasks hazard（等价于 water+spark+smoke）
```

### 3.4 仅电线绝缘破损

```bash
CUDA_VISIBLE_DEVICES=5 python sam3_detect_multi.py \
  --source New_test/image_0907_wire \
  --out-dir runs/detect/New_test/sam3_multi_wire \
  --tasks wire \
  --wire-conf 0.15 \
  --max-damage-boxes 1
```

### 3.4 细粒度任务选择

`--tasks` 支持逗号分隔：

| 取值 | 含义 |
|------|------|
| `cig` | 烟头 + 烟盒 + 打火机 |
| `wire` / `damage` | 绝缘破损 |
| `hazard` | 积水 + 火花 + 烟雾 |
| `water` / `puddle` | 仅积水 |
| `spark` / `smoke` | 仅火花 / 仅烟雾 |
| `pack` / `lighter` / `butts` | 单一香烟子类 |
| `cig,wire,hazard` | 全部任务 |

示例：只要烟盒和打火机：

```bash
CUDA_VISIBLE_DEVICES=5 python sam3_detect_multi.py \
  --source /path/to/images \
  --out-dir runs/detect/New_test/sam3_pack_lighter \
  --tasks pack,lighter
```

### 3.5 常用参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--source` | （必填） | 单张图片或目录 |
| `--out-dir` | （必填） | 输出目录 |
| `--tasks` | `cig,wire,hazard` | 任务列表 |
| `--conf` | `0.25` | 香烟类置信度阈值 |
| `--wire-conf` | `0.15` | 破损类置信度下限 |
| `--hazard-conf` | `0.20` | 积水/火花/烟雾置信度下限 |
| `--max-per-class` | `2` | 香烟/隐患每类最多框数 |
| `--max-damage-boxes` | `1` | 破损最多框数 |
| `--max-area-ratio` | `0.25` | 破损框最大面积占比（滤掉整缆大框） |
| `--ckpt` | 见上文默认路径 | SAM3 权重 |

### 3.6 耗时评测（可选）

```bash
CUDA_VISIBLE_DEVICES=5 python bench_sam3_timing.py \
  --source New_test/image_0901_v2 \
  --warmup 3 --repeat 5 \
  --out runs/detect/bench_sam3_timing/image_0901_v2.json
```

参考量级（H800，1080p 左右）：优化路径约 **0.26 s/张**（1× 图编码 + 多文本）；当前香烟多 prompt 串行会更慢。YOLO 同图约 **18 ms**，SAM3 适合难例/开放词汇，不宜无脑替代生产线 YOLO。

---

## 4. 输出说明

```
<out-dir>/
├── <stem>_sam3_multi.jpg   # 可视化（中文类名 + 百分比）
└── results.txt             # 文本汇总（类别、conf、prompt、框坐标）
```

启用 `wire` 时，图顶部横幅：

- **绝缘层损坏**：检出破损框  
- **未检出绝缘破损**：未找到满足面积/置信度条件的破损框  

---

## 5. 文本 Prompt 说明（实现内置）

开放词汇对措辞敏感，脚本为每类准备了多组英文短句，再选优：

| 类别 | 代表性 prompt |
|------|----------------|
| 打火机 | `lighter`, `cigarette lighter` |
| 烟盒 | `cigarette pack`, `red/gold cigarette pack` |
| 烟头 | `cigarette butt(s)`, `cigarette filter tip` |
| 绝缘破损 | `bare copper strands`, `exposed wires`, … |
| 积水 | `puddle of water`, `standing water on floor`, … |
| 火花 | `electrical spark`, `bright spark flash`, … |
| 烟雾 | `smoke plume`, `gray smoke`, … |

破损任务额外用面积与中心偏好，避免框住整根电缆。

---

## 6. 常见问题

| 现象 | 处理 |
|------|------|
| `No module named iopath` 等 | 在 `sam3` 环境安装缺失包 |
| CUDA ordinal 错误 | 使用 `CUDA_VISIBLE_DEVICES=N`，不要写死 `cuda:6` |
| 烟头检不出 | 试降低 `--conf`，或确认图中确有烟头；开放词汇对小目标更挑 |
| 破损框太大 | 调低 `--max-area-ratio`（如 `0.15`），保持 `--max-damage-boxes 1` |
| 中文标签方块 | 确认 `assets/fonts/NotoSansCJK-Regular.ttc` 存在 |

---

## 7. 相关文档

- 接口总览：`/data/zhangzhe/docs/cig-yolo-sam3-vlm-interface.md`  
- 数据集/YOLO/VLM：`doc/机房巡检-数据集训练测试与VLM仲裁.md`  

---

*路径：`doc/sam3_multi_detect.md`*
