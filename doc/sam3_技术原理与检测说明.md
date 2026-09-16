# SAM3 目标检测技术文档与原理说明

> 本文面向本仓库的 **SAM3 开放词汇检测**实践：讲清 Meta SAM3 原理，以及我们如何用文本提示做烟盒/打火机/烟头、绝缘破损、积水/火花/烟雾等检测。  
> 配套使用说明见：`doc/sam3_multi_detect.md`。

---

## 1. 文档定位与结论先行

| 问题 | 结论 |
|------|------|
| SAM3 是什么？ | Meta 的**提示式、开放词汇**分割/检测基础模型（Segment Anything with Concepts） |
| 我们有没有自己预训练？ | **没有**。加载官方/本地权重 `sam3.pt`，只做推理与后处理 |
| 方框谁画的？ | SAM3 只返回坐标与分数；**可视化框由脚本后处理绘制** |
| 适合什么？ | 零样本试新概念、难例补漏、多任务快速验证 |
| 不适合什么？ | 替代产线闭集 YOLO 的全量高速检测（更慢、措辞敏感） |

**本机关键路径**

| 项 | 路径 |
|----|------|
| 多任务脚本 | `sam3_detect_multi.py` |
| 香烟专用脚本 | `sam3_detect_cig.py` |
| 计时脚本 | `bench_sam3_timing.py` |
| 权重 | `/data/zhangzhe/do-as-i-do/reconstruction/checkpoints/sam3/sam3.pt` |
| 源码 | `/data/zhangzhe/do-as-i-do/reconstruction/modules/sam3` |
| Conda | `/data/zhangzhe/conda/envs/sam3` |

---

## 2. SAM3 原理（详细）

### 2.1 从 SAM → SAM2 → SAM3

| 代际 | 核心能力 | 「找什么」怎么指定 |
|------|----------|-------------------|
| **SAM** | 交互式分割 | 点、框等**几何提示**；不理解「这是打火机」这类类名 |
| **SAM2** | 图像 + 视频跟踪 | 仍以几何提示为主，跨帧跟踪 |
| **SAM3** | **概念级**开放词汇 | 除点/框外，可用**短文本短语**（如 `"lighter"`）找出该概念的**全部实例** |

官方表述要点（节选自 Meta SAM3 README）：

- 统一基础模型：图像/视频上的 **promptable segmentation**  
- 可用 **text** 或 **visual prompts**（点、框、掩码、exemplar）  
- 相对 SAM2：能按短文本/范例**穷尽检出**开放词汇概念的所有实例  
- SA-CO 基准含约 **27 万**独特概念；数据引擎自动标注超 **400 万**概念量级  
- 架构上引入 **presence token**，更好区分相近文本（如 “穿白衣的球员” vs “穿红衣的球员”）  
- **detector–tracker 解耦**，减轻任务互相干扰  

论文/主页：  
[SAM 3 Paper](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/) · [Project](https://ai.meta.com/sam3) · [Blog](https://ai.meta.com/blog/segment-anything-model-3/)

### 2.2 「开放词汇」是什么意思

- **闭集检测（如本仓库 YOLOv8 四类）**：只能检训练集里写死的类（烟头/烟盒/香烟/打火机）。新类要重新标数据、改 `nc`、再训。  
- **开放词汇（SAM3）**：推理时用自然语言描述目标；模型在大规模图文/概念数据上**预训练**过，能把文本语义对齐到图像区域，因此可对**未见过的业务类名**做零样本尝试（如 `"bare copper strands"`、`"puddle of water"`）。

注意：开放词汇 ≠ 任意中文长句任意理解。本仓库接口实际吃的是**短英文概念短语**；措辞不同，召回差异很大。

### 2.3 推理时的信息流（概念图）

```
输入图像
    │
    ▼
┌─────────────────────┐
│ 视觉骨干（大 ViT）   │  → 稠密图像特征 / backbone_out
└─────────────────────┘
    │
输入文本 prompt（如 "lighter"）
    │
    ▼
┌─────────────────────┐
│ 语言骨干 forward_text│  → 文本特征，写入 backbone_out
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Grounding / 检测头   │  → 与文本匹配的区域
└─────────────────────┘
    │
    ▼
boxes + scores（+ 可选 masks）
```

本仓库调用链（`Sam3Processor`）：

1. `set_image(image)`：图像预处理 + 视觉编码，得到 `state["backbone_out"]`  
2. `set_text_prompt(prompt, state)`：`model.backbone.forward_text([prompt])`，再 `_forward_grounding(state)`  
3. 从 `state` 读取 `boxes`、`scores`

几何提示（点/框）走另一路 API（如 `add_geometric_prompt`）；无文本时可用哑文本 `"visual"` 让模型主要依赖几何。**当前业务脚本以文本提示为主。**

### 2.4 文本提示 vs 几何提示

| 类型 | 例子 | 作用 |
|------|------|------|
| **文本概念** | `"cigarette pack"`、`"exposed wires"` | 指定语义类别/概念 |
| **几何提示** | 点、框（正/负） | 指定「看图像上哪一块」 |

可组合使用。我们巡检脚本目前主要用**多组文本短语 + 后处理选框**。

### 2.5 Presence token 与相近概念

开放词汇容易混淆近义/近形短语。SAM3 用 presence 等相关设计，提高「这个概念在图中是否存在、对应哪片区域」的判别力。对业务的启示是：**prompt 要尽量写成具体视觉外观**（`bare copper strands`），少写抽象标签（`insulation damage`），否则易框成整根电缆或整片背景。

### 2.6 与 YOLO、VLM 的分工（本项目）

```
YOLO（闭集、快）     → 主路径：定位 + 初分类（~十几 ms）
VLM 二次仲裁         → 只改 YOLO 已有框的类别，不补漏检
SAM3（开放词汇）     → 零样本新概念 / 难例补漏 / 绝缘破损·积水·烟雾试探
```

| | YOLO | SAM3 | Qwen3-VL 仲裁 |
|--|------|------|----------------|
| 是否需本业务训练 | 要 | 不要（用预训练） | 不要（用对话/分类） |
| 输出 | 固定类框 | 开放概念框/掩码 | ROI 类别 JSON |
| 速度（本机 H800 量级） | ~18 ms/图 | ~0.26 s+（多 prompt） | ROI 级秒级累计 |
| 漏检 | 依赖训练分布 | 可按文本再找 | **不能**补漏检 |

---

## 3. 本仓库检测流水线（实现层）

### 3.1 总流程（一句话）

**加载预训练 SAM3 → 对每个业务类尝试多组英文 prompt → 收集 boxes/scores → NMS / 面积或最大框策略 → OpenCV/中文标签画框落盘。**

### 3.2 类注册表（`CLASSES`）

`sam3_detect_multi.py` 用字典注册每一类：

| 字段 | 含义 |
|------|------|
| `name` / `zh` | 英文名 / 中文显示名 |
| `color` | 可视化 BGR 颜色 |
| `task` | 任务分组（cig / wire / hazard） |
| `prompts` | 该类别的多条英文短句 |
| `mode` | `object`：常规目标；`damage`：局部破损（面积过滤） |
| `select` | 可选 `largest`：只留面积最大的 1 框（积水、烟雾） |
| `max_boxes` | 该类最多保留框数 |

当前已注册：

| ID | 中文 | 任务 | 模式要点 |
|----|------|------|----------|
| 0 | 烟头 | cig | object，Top-K |
| 1 | 烟盒 | cig | object，多颜色 pack 短语 |
| 3 | 打火机 | cig | object |
| 10 | 绝缘破损 | wire | damage，紧凑局部框 |
| 11 | 积水 | hazard | object + **largest×1** |
| 12 | 火花 | hazard | object |
| 13 | 烟雾 | hazard | object + **largest×1** |

`--tasks` 别名：`cig`、`wire`、`hazard`、`water`、`spark`、`smoke`、`pack`、`lighter`、`butts` 等。

### 3.3 为何一类要用「一组 prompt」而不是一句话

1. **API 是一词一查**：每次 `set_text_prompt` 一个短短语，不是长对话。  
2. **外观多样**：金烟盒 vs 红烟盒、铜芯外露 vs 彩色芯线外露，单一短语难覆盖。  
3. **措辞敏感**：`"exposed copper wire"` 可能无框，`"bare copper strands"` 却高分。  
4. **后处理只留少数框**：多短语负责召回，NMS / largest 负责「看起来只有一个准框」。

### 3.4 后处理策略

#### （1）object 模式（烟盒/打火机/烟头/积水/火花/烟雾）

- 汇总该类所有 prompt 的框  
- IoU NMS（默认 0.5）  
- 默认按置信度 Top-K（`--max-per-class`）  
- 若 `select=largest`：在 NMS 后按**面积×置信度**取最大，强制 `max_boxes=1`（积水、烟雾）

#### （2）damage 模式（绝缘破损）

- 过滤面积占比过大/过小的框（避免整根电缆）  
- 排序：`conf × (1 − area_ratio) × 中心偏好`  
- 默认每图最多 1 个破损框（`--max-damage-boxes`）

#### （3）可视化

- `draw_dets`：OpenCV 画矩形 + 中文标签（`vis_labels.draw_chinese_label`）  
- 横幅：绝缘是否损坏、是否检出积水/火花/烟雾  

**再次强调：框不是 SAM3 渲染的，是后处理画的。**

### 3.5 内置 Prompt 一览（业务常用）

| 类别 | 代表性英文 prompt |
|------|-------------------|
| 打火机 | `lighter`, `cigarette lighter`, `disposable lighter` |
| 烟盒 | `cigarette pack`, `red/gold cigarette pack`, `cigarette box`, `tobacco pack` |
| 烟头 | `cigarette butt(s)`, `cigarette filter tip`, `smoked cigarette butt` |
| 绝缘破损 | `bare copper strands`, `exposed wires`, `torn insulation exposing wires`, … |
| 积水 | `puddle of water`, `standing water on floor`, `water puddle`, … |
| 火花 | `electrical spark`, `sparks`, `bright spark flash`, … |
| 烟雾 | `smoke`, `smoke plume`, `gray smoke`, `white smoke`, … |

经验：

- **具体外观 > 抽象缺陷名**（破损、损坏）  
- 积水/烟雾易碎框 → `largest` 只留大框  
- 绝缘破损要**局部** → `damage` 面积门控  

---

## 4. 环境与运行

### 4.1 环境

```bash
conda activate sam3
# 或
export PATH=/data/zhangzhe/conda/envs/sam3/bin:$PATH

cd /data/zhangzhe/Yolo_Cigarettes_dectection
```

依赖：PyTorch（CUDA）、SAM3 及其依赖（如 `iopath`、`einops`、`pycocotools` 等）。

### 4.2 推荐命令

```bash
# 香烟三类
CUDA_VISIBLE_DEVICES=5 python sam3_detect_multi.py \
  --source New_test/image_0901_v2 \
  --out-dir runs/detect/New_test/sam3_multi_cig \
  --tasks cig

# 绝缘破损
CUDA_VISIBLE_DEVICES=5 python sam3_detect_multi.py \
  --source New_test/image_0907_wire \
  --out-dir runs/detect/New_test/sam3_multi_wire \
  --tasks wire --wire-conf 0.15 --max-damage-boxes 1

# 积水 / 火花 / 烟雾
CUDA_VISIBLE_DEVICES=5 python sam3_detect_multi.py \
  --source New_test/image-0914 \
  --out-dir runs/detect/New_test/image-0914_sam3 \
  --tasks water,spark,smoke --hazard-conf 0.20

# 全部任务
CUDA_VISIBLE_DEVICES=5 python sam3_detect_multi.py \
  --source /path/to/images \
  --out-dir runs/detect/New_test/sam3_all \
  --tasks cig,wire,hazard
```

### 4.3 主要参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--tasks` | `cig,wire,hazard` | 任务组合 |
| `--conf` | 0.25 | 香烟类阈值 |
| `--wire-conf` | 0.15 | 绝缘破损阈值 |
| `--hazard-conf` | 0.20 | 积水/火花/烟雾阈值 |
| `--max-per-class` | 2 | object 类默认 Top-K（可被类内 `max_boxes` 覆盖） |
| `--max-damage-boxes` | 1 | 破损最多框数 |
| `--max-area-ratio` | 0.25 | 破损最大面积占比 |

输出：`<out-dir>/<stem>_sam3_multi.jpg` + `results.txt`。

---

## 5. 性能与计时口径

### 5.1 本机实测（`bench_sam3_timing.py`，`image_0901_v2`，H800）

预热 + `torch.cuda.synchronize()` + `perf_counter`，不含读盘/画框：

| 模式 | 中位耗时 | 说明 |
|------|---------:|------|
| YOLO imgsz=1280 | ~18.5 ms | 同图对比 |
| SAM3 optimized | ~258 ms | 1× `set_image` + lighter + 5× pack 文本 |
| SAM3 legacy（旧脚本每 pack 重编码） | ~515 ms | 约 6× 图像编码 |

分段（optimized）：`set_image` ≈ 51 ms；单次文本 grounding ≈ 35 ms；5 个 pack 合计 ≈ 172 ms。

### 5.2 如何准确计时（规范）

1. GPU 预热若干次  
2. 计时前后 `torch.cuda.synchronize()`  
3. 分段：`set_image` vs 每次 `set_text_prompt`  
4. 报清分辨率与 prompt 次数  
5. 不要把首次加载权重、写 JPG 算进「模型推理」  

公平对比应：**图像只编码一次，再换多个文本**；旧 `run_sam3_predict` 对每个 pack prompt 都 `set_image`，会人为翻倍耗时。

---

## 6. 验证样例（本仓库已跑）

| 场景 | 目录/图 | 现象 |
|------|---------|------|
| 烟盒+打火机 | `New_test/image_0901_v2` | 开放词汇检出稳定，conf 可到 80%+ |
| 绝缘破损 | `New_test/image_0907_wire` | 具体短语 + 面积过滤后局部框准 |
| 积水 | `New_test/image-0914` | `puddle of water`；`largest` 后每图 1 大框 |
| 烟雾 | `烟雾jpg.jpg` | `smoke`；`largest` 后 1 大框 |

积水产线更推荐已有 YOLO：`Yolo_Water_Leak/.../water_leak_yolov8m/weights/best.pt`（更快、专用数据训过）；SAM3 适合统一入口试探或多任务演示。

---

## 7. 扩展新检测类（怎么改代码）

1. 在 `CLASSES` 增加新 id、`zh`、`color`、`prompts`、`mode`  
2. 在 `TASK_ALIASES` 挂上别名（如 `flood: [新id]`）  
3. 若只需 1 个大区域：`"select": "largest", "max_boxes": 1`  
4. 若只要局部小缺陷：`"mode": "damage"` 并调面积阈值  
5. 用真实图试 prompt，删掉弱短语  

详见使用说明：`doc/sam3_multi_detect.md`。

---

## 8. 局限与风险（写材料/面试建议主动讲）

1. **措辞敏感**：同一概念换说法，结果可能差很多。  
2. **速度**：大 ViT + 多 prompt 串行，远慢于 YOLO。  
3. **不是业务专用检测器**：未在你们绝缘/积水/烟雾大数据上 fine-tune；零样本有上限。  
4. **小目标/瞬态**（火花、远距离烟头）易漏或假阳。  
5. **烟雾/积水**边界模糊，易出大框或碎框 → 需 `largest` 等后处理。  
6. **指标**：开放词汇试探不能直接当 mAP 主结论；正式指标仍应挂闭集 YOLO 或专用模型。  

---

## 9. 与相关文档索引

| 文档 | 内容 |
|------|------|
| `doc/sam3_multi_detect.md` | 启动环境、命令、参数速查 |
| `doc/机房巡检-数据集训练测试与VLM仲裁.md` | YOLO/VLM 数据集与仲裁 |
| `/data/zhangzhe/docs/cig-yolo-sam3-vlm-interface.md` | YOLO+SAM3+VLM 接口总览 |
| Meta SAM3 README | 官方原理与安装 |

---

## 10. 参考引用（SAM3）

请以官方页面最新 BibTeX 为准。模型全称：**SAM 3: Segment Anything with Concepts**（Meta Superintelligence Labs）。

---

*文档路径：`doc/sam3_技术原理与检测说明.md`*  
*维护：与 `sam3_detect_multi.py` 类表保持同步；增类时请同步更新 §3.2 / §3.5。*
