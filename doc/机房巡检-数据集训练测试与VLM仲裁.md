# 机房巡检异常检测：数据集 · 训练 · 测试 · 推理 · VLM 二次仲裁

> 技术说明文档（与面试备战稿 `机房巡检-VLM感知模块.md` 互补）。  
> 本文只写**已落地、可复现**的路径、超参与实测数字；参考/未跑数字单独标注。  
> 项目根目录：`/data/zhangzhe/Yolo_Cigarettes_dectection/`

---

## 1. 系统概览

```
输入图片
  → YOLOv8 多类检测（烟头/烟盒/香烟/打火机）
  → 规则后处理（IoU：lighter 优先于 butts/pack）
  → [可选] 门控 → Qwen3-VL ROI 类别二次仲裁（只改类、不改框）
  → 可视化 + YOLO labels 落盘
```

| 模块 | 角色 | 主指标 |
|------|------|--------|
| YOLO | 定位 + 初分类 | **mAP@0.5**（检测） |
| VLM 仲裁 | 易混框类别复核 | **分类准确率 Acc**（门控∩IoU≥0.5 对齐子集） |
| 机柜状态 VLM | 柜门/指示灯/温湿度 → JSON | 端到端延迟（另见 `docs/qwen3-vl-*.md`） |

**当前生产检测权重：**

`runs/detect/cig_multiclass_ft_newtest/weights/best.pt`

---

## 2. 数据集

### 2.1 源数据集（Roboflow YOLO 导出）

路径：`dataset/`

| 源目录 | 原任务类别 | 合并映射 | 合并前规模 (train/valid/test) |
|--------|------------|----------|------------------------------|
| `Yolo Cigarettes Butts.v1i.yolov8` | cigarettes_butts, others | 仅保留 butts→**0**；丢弃 others | 791 / 226 / 113 |
| `SmokingAndPack.v1i.yolov8` | cig-pack, cigarette, face, smoking | pack→**1**，cigarette→**2**；丢弃 face/smoking | 2290 / 571 / 280 |
| `Lighter.v1i.yolov8` | Lighter | Lighter→**3** | 86 / 11 / 11 |
| `aTarah.v1i.yolov8`（后续补充） | 含打火机等 | 并入统一 4 类 | 5199 / 1474 / 729 |

合并脚本：`merge_datasets.py`（类别重映射、丢弃无关类、文件名加源前缀防冲突）。

另有机房实拍难例 `New_test/image_0828`（5 张）经人工校正后以 `newtest_*` 前缀**写入 train**（不进 test），用于微调；GT 曾按预测框改类，**有乐观偏差，不作正式测试集**。

### 2.2 统一类别

```yaml
nc: 4
names: ['cigarettes_butts', 'cig-pack', 'cigarette', 'Lighter']
```

| ID | 英文 | 中文 |
|----|------|------|
| 0 | cigarettes_butts | 烟头 |
| 1 | cig-pack | 烟盒 |
| 2 | cigarette | 香烟 |
| 3 | Lighter | 打火机 |

配置：`merged_dataset/data.yaml`

### 2.3 合并后规模（当前磁盘实测）

**按 split × 类别（标注框数）：**

| split | 图片数 | 总框数 | 烟头 (0) | 烟盒 (1) | 香烟 (2) | 打火机 (3) |
|-------|-------:|-------:|---------:|---------:|---------:|-----------:|
| train | 3676 | 5175 | 1319 | 901 | 1713 | 1242 |
| valid | 829 | 1070 | 377 | 132 | 479 | 82 |
| test | 429 | 617 | 220 | 72 | 240 | 85 |
| **合计** | **4934** | **6862** | **1916** | **1105** | **2432** | **1409** |

**四类全库占比（train+valid+test，按框）：**

| 类别 | 中文 | 框数 | 占比 |
|------|------|-----:|-----:|
| cigarettes_butts | 烟头 | 1916 | 27.9% |
| cig-pack | 烟盒 | 1105 | 16.1% |
| cigarette | 香烟 | 2432 | 35.4% |
| Lighter | 打火机 | 1409 | 20.5% |

**测试集四类（正式评测用，429 图 / 617 框）：** 烟头 220（35.7%）、烟盒 72（11.7%）、香烟 240（38.9%）、打火机 85（13.8%）。烟盒与打火机相对偏少，是易混评测上更吃紧的两类。

说明：早期 `multiclass_training_guide.md` 记录过仅三源合并时的规模（test 401/545）；后续补充 Lighter / aTarah / newtest 后，**以当前表为准**。  
**test 标注来自 Roboflow 独立导出 + 重映射**，不含 `newtest_*`，无「按预测改类」纠缠。

---

## 3. 训练

### 3.1 环境

| 项 | 值 |
|----|-----|
| Conda | `/data/zhangzhe/Yolo_Cigarettes_dectection/.conda/cig` |
| 框架 | Ultralytics **8.4.130**，PyTorch **2.5.1+cu121** |
| 基座 | YOLOv8m |

### 3.2 训练阶段（路线 A：迁移学习）

| 阶段 | 实验名 | 起点 | 要点 | 产出权重 |
|------|--------|------|------|----------|
| ① 烟头单类 | `cig_butts_1280` | `yolov8m.pt` | imgsz=1280，batch=32，早停 ~87 epoch | `runs/detect/cig_butts_1280/weights/best.pt` |
| ② 四类合并 | `cig_multiclass` | ① 的 best | nc=2→4 重建检测头，主干迁移 | `runs/detect/runs/cig_multiclass/weights/best.pt`（路径以实际为准） |
| ③ Lighter 微调 | `cig_multiclass_ft_lighter` | ② | 补打火机数据后微调 | `.../cig_multiclass_ft_lighter/weights/best.pt` |
| ④ 实拍难例微调 | `cig_multiclass_ft_newtest` | ③ | 加入 image_0828 GT，epochs=30，patience=10 | **`.../cig_multiclass_ft_newtest/weights/best.pt`（部署）** |

阶段 ④ 脚本要点（`train_finetune_newtest.py`）：

```text
data=merged_dataset/data.yaml
epochs=30, imgsz=1280, batch=32, patience=10
mosaic=1.0, close_mosaic=5, cache=ram
```

阶段 ② 典型超参：`epochs=200, imgsz=1280, batch=16~32, patience=40, mosaic=1.0`。

### 3.3 训练期参考指标（历史记录）

- 阶段 ① valid：烟头 mAP50 ≈ **0.89**，整体（含 others）≈ **0.68**
- 阶段 ② 结束时 valid：all mAP50 ≈ **0.93**（Lighter 样本极少时指标虚高，仅参考）
- 阶段 ④ 后以 **test 集实测**为准（见下一节），不以训练 valid 写简历主数字

---

## 4. 测试结果

评测权重：`cig_multiclass_ft_newtest/weights/best.pt`  
脚本：`eval_testset.py`、`eval_vlm_boxlog.py`、`eval_image_0828.py`

### 4.1 完整测试集 —— YOLO 检测（主指标：mAP）

**管道：YOLO + 规则后处理（无 VLM）**

| | Precision | Recall | mAP50 | mAP50-95 |
|--|----------:|-------:|------:|---------:|
| **all** | 0.884 | 0.852 | **0.857** | 0.554 |
| cigarettes_butts | 0.897 | 0.832 | 0.824 | 0.549 |
| cig-pack | 0.769 | 0.742 | 0.761 | 0.466 |
| cigarette | 0.927 | 0.846 | 0.867 | 0.556 |
| Lighter | 0.944 | 0.988 | 0.977 | 0.645 |

规模：429 图 / 617 GT 框。产物：`runs/detect/eval_testset/metrics_no_vlm.json`

### 4.2 完整测试集 —— +VLM 后的检测 mAP（负对照）

| | mAP50 | Precision | Recall | vlm_calls |
|--|------:|----------:|-------:|----------:|
| YOLO+规则 | **0.857** | 0.884 | 0.852 | 0 |
| +VLM | **0.818** | 0.857 | 0.838 | 196 |

VLM **不改框、只改类**；大集上几乎无「框对类错」，仲裁易误伤 → 全局 mAP 略降。  
**简历：mAP 只挂 YOLO；不要写「VLM 提升 mAP」。**

### 4.3 完整测试集 —— VLM 主指标：分类准确率 Acc

框级日志评测：`eval_vlm_boxlog.py` → `runs/detect/eval_vlm_boxlog/`

| 量 | 值 |
|----|-----|
| N（门控框） | **196** |
| M（门控 ∩ IoU≥0.5 对齐 GT） | **138** |
| Acc_pre（YOLO 类） | **100.0%** |
| Acc_post（+VLM 生效后） | **97.1%** |
| Δ | **−2.9 pp** |
| 改对 / 误伤 / 净收益 | **0 / 4 / −4** |
| Δ 95% CI（按图 bootstrap，B=2000） | **[−6.0%, −0.6%]**（上界&lt;0，下降显著） |

误伤方向（4 处）：butts→pack ×3，butts→Lighter ×1。  
解读：门控捞到的「易混框」在分布内测试集上类别已几乎全对，VLM 无可纠正空间。

### 4.4 个案集 image_0828（仅机制附录）

| 项 | 值 |
|----|-----|
| 规模 | 5 图 / 8 框（pack×4，Lighter×4） |
| 无 VLM mAP50 | 0.620 |
| +VLM mAP50 | 0.745 |
| 实质变化 | 1 处 butts(0.90)→Lighter(0.95)；净 +1 类纠正 |

GT 按预测框改类，N 极小 → **不作正式增益结论**，只证明「定位对、类错时可被 VLM 纠回」的机制。

### 4.5 指标怎么写（约定）

| 模块 | 写什么 | 不写什么 |
|------|--------|----------|
| YOLO | 测试集 mAP50 ≈ **0.86** | — |
| VLM | Acc_pre / Acc_post + 改对/误伤 + CI | 「VLM 提升 mAP」；未跑真数的 64%→88% |

---

## 5. 推理（独立章节）

### 5.1 环境与权重

| 项 | 路径 / 命令 |
|----|-------------|
| YOLO 环境 | `conda activate /data/zhangzhe/Yolo_Cigarettes_dectection/.conda/cig` |
| 检测权重 | `runs/detect/cig_multiclass_ft_newtest/weights/best.pt` |
| 入口脚本 | `predict_multiclass.py` |
| VLM 环境 | `conda activate qwen3vl`（仅启动服务时需要） |
| VLM 权重 | `/data/zhangzhe/models/Qwen3-VL-32B-Instruct-FP8` |
| VLM API | `http://127.0.0.1:8001/v1`，served name `qwen3-vl-idc` |

### 5.2 启动 VLM 服务（需要 `--vlm` 时）

```bash
conda activate qwen3vl
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

CUDA_VISIBLE_DEVICES=4 vllm serve /data/zhangzhe/models/Qwen3-VL-32B-Instruct-FP8 \
  --served-model-name qwen3-vl-idc \
  --host 0.0.0.0 --port 8001 \
  --tensor-parallel-size 1 --dtype auto \
  --gpu-memory-utilization 0.90 \
  --max-model-len 16384 --trust-remote-code
```

健康检查：`curl http://127.0.0.1:8001/v1/models`、`/health`。  
机柜巡检单图 JSON（预热后端到端）约 **0.4–0.5 s**（详见 `/data/zhangzhe/docs/qwen3-vl-start-to-infer.md`）。

### 5.3 YOLO 推理命令

**仅 YOLO + 规则：**

```bash
conda activate /data/zhangzhe/Yolo_Cigarettes_dectection/.conda/cig
cd /data/zhangzhe/Yolo_Cigarettes_dectection

CUDA_VISIBLE_DEVICES=5 python predict_multiclass.py \
  --source /path/to/images \
  --name my_run \
  --device 0 \
  --conf 0.25 \
  --imgsz 1280
```

**YOLO + VLM 仲裁：**

```bash
CUDA_VISIBLE_DEVICES=5 python predict_multiclass.py \
  --source /path/to/images \
  --name my_run_vlm \
  --device 0 \
  --vlm \
  --vlm-api http://127.0.0.1:8001/v1 \
  --vlm-model qwen3-vl-idc \
  --vlm-min-conf 0.5
```

### 5.4 推理默认参数

| 参数 | 默认 | 含义 |
|------|------|------|
| `--model` | `cig_multiclass_ft_newtest/.../best.pt` | 检测权重 |
| `--imgsz` | 1280 | 推理分辨率 |
| `--conf` | 0.25 | YOLO 置信度阈值 |
| `--iou-thresh` | 0.3 | 规则后处理 IoU（与 lighter 重叠则丢 butts/pack） |
| `--device` | 0 | 逻辑 GPU（配合 `CUDA_VISIBLE_DEVICES`） |
| `--project` / `--name` | `runs/detect/New_test` / `predict_vlm` | 输出目录 |
| `--vlm` | 关 | 开启二次仲裁 |
| `--vlm-min-conf` | **0.5** | VLM 返回置信度低于此则不改类 |
| `--no-vis-postprocess` | 关 | 关闭中文加粗百分比可视化 |

### 5.5 推理流水线（逐步）

1. `YOLO.predict` → 原始框  
2. `apply_rule_postprocess`：lighter 与 butts/pack IoU≥0.3 时丢弃 butts/pack  
3. 若 `--vlm`：`needs_vlm_arbitration` 选出索引 → 裁 ROI → `QwenVlmClient.classify_roi` → `should_apply_vlm_update` 决定是否改 `cls/conf`  
4. 再跑一遍规则后处理  
5. 写可视化 JPG + `labels/*.txt`（YOLO 归一化格式）

### 5.6 输出结构

```
runs/detect/New_test/<name>/
├── *.jpg           # 默认：中文类名 + 百分比置信度（可加粗）
└── labels/*.txt    # class xc yc w h
```

### 5.7 延迟量级（参考）

| 模式 | 量级 |
|------|------|
| 仅 YOLO（单图，H800） | 约数十 ms 量级（含后处理；首张更慢） |
| +VLM（按门控框次数） | 每调用一次 ROI 分类约秒级；全测试集 196 次调用总墙钟约 **177 s**（429 图） |
| 机柜 JSON 巡检（整图 VLM） | 预热后约 **0.4–0.5 s/图** |

---

## 6. VLM 二次仲裁 —— 完整细节

### 6.1 原理

YOLO 负责**定位**；在易混场景（打火机↔烟盒、烟头↔打火机、低置信扁框等）上，用视觉语言模型对**裁剪 ROI**做语义分类，**只替换类别（及可选抬高 conf），不移动框**。

设计动机：

- 复用已训检测器的框，避免端到端 grounding 的延迟与稳定性成本；  
- **条件门控**而非全量送 VLM，控制算力与误伤面；  
- 大集负对照证明：无类错可纠时上 VLM 会降 mAP / Acc，因此必须门控。

代码：`vlm_gate.py`、`vlm_arbitrate.py`、`predict_multiclass.py`。

### 6.2 总流程

整条链路按「先检出、再筛选、再仲裁、最后落稳」四步执行。首先用 YOLOv8 对整图做多类检测，得到每个框的坐标、类别与置信度，并做一轮规则后处理（例如 lighter 与 butts/pack 严重重叠时优先保留打火机）。接着并不把所有框都交给 VLM，而是由门控函数 `needs_vlm_arbitration`（默认低置信阈值 0.65、扁框宽高比阈值 0.85、重叠 IoU 0.3）筛出真正易混的候选集合 \(G\)——典型包括低置信或偏扁的烟盒/烟头、低置信打火机，以及 pack↔lighter、butts↔lighter 等重叠共现框。对 \(G\) 中每一个框，按框外扩 15% 裁出 ROI，缩放到最长边不超过 512 后编码为 JPEG，连同「YOLO 初判类别与置信度」一起写入 Prompt，调用本地 vLLM 上的 Qwen3-VL（temperature=0.1）做 ROI 分类，解析返回的 JSON 得到 VLM 类别与自报置信度。最后由 `should_apply_vlm_update` 做保守采纳：VLM 置信度须 ≥0.5，pack↔lighter 方向翻转还须 ≥0.75，且在框内多物体共现等情况下宁可保持 YOLO 原类；仅当通过时才把该框类别改为 VLM 结果，并将置信度更新为 `max(yolo_conf, vlm_conf)`，框坐标始终不变。改类结束后再跑一遍规则后处理，写出可视化与标签。一句话概括：**YOLO 出框 → 门控只送易混框 → VLM 只判类 → 保守门槛决定是否改类 → 规则再收敛。**

### 6.3 门控条件（`needs_vlm_arbitration`）

默认超参：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `low_conf` | **0.65** | 低于此置信度进入候选 |
| `tall_aspect` | **0.85** | 宽高比 = w/h；**小于**此值视为偏扁，pack/butts 进入候选 |
| `overlap_iou` | **0.3** | 类间/同类重叠达此 IoU 则双方进候选 |

具体规则：

1. **cig-pack**：`conf < 0.65` **或** `aspect < 0.85`  
2. **cigarettes_butts**：同上  
3. **Lighter**：仅 `conf < 0.65`  
4. **pack 与 lighter** IoU≥0.3 → 两框都进  
5. **butts 与 lighter** IoU≥0.3 → 两框都进  
6. **两个 pack** 互相 IoU≥0.3 → 两框都进  

`cigarette`（整支烟）默认不因低置信单独进门控（除非被上述重叠规则带入）。

全测试集一次评估：`vlm_calls = 196`（与框级日志 N_gated 一致）。

### 6.4 置信度阈值汇总

| 环节 | 阈值 | 作用 |
|------|------|------|
| YOLO 检出 | `conf=0.25` | 低于此不产出框 |
| 门控 low_conf | **0.65** | 低于此（或扁框）送 VLM |
| VLM 采纳下限 | **`vlm_min_conf=0.5`** | VLM 自报 confidence &lt; 0.5 → 不改类 |
| pack↔lighter 翻转 | **0.75** | 在 `should_apply_vlm_update` 中，pack→lighter 或 lighter→pack 还需 VLM conf≥0.75，且不能因 multi-object / 对方仍可见而拒绝 |
| 规则后处理 IoU | 0.3 | lighter 优先抑制重叠 butts/pack |
| ROI pad | 0.15 | 框外扩 15% 再裁切 |
| ROI 最长边 | 512 | 送 VLM 前缩放 |
| 采样 | temperature **0.1** | 偏确定性 |
| max_tokens | **320** | JSON 足够 |

### 6.5 更新是否生效（`should_apply_vlm_update`）

返回 `(False, reason)` 的常见原因：

| 条件 | skip 原因（摘要） |
|------|-------------------|
| `class_id is None` 或 conf &lt; 0.5 | `vlm conf too low` |
| 与 YOLO 同类 | `same class` |
| 单框且 multi-object、pack+lighter 共现 | `merged box: ... keep yolo` |
| YOLO=pack，VLM=lighter，但 crop 仍见 pack / multi / conf&lt;0.75 | 拒绝翻转 |
| YOLO=lighter，VLM=pack，对称条件 | 拒绝翻转 |

仅当返回 `apply` 时才写回类别。

### 6.6 Prompt（完整）

推理时若传入 YOLO 初判，使用 **`PROMPT_WITH_YOLO`**（默认路径）；否则用 `PROMPT`。

#### 6.6.1 带 YOLO 初判（实际默认）

```
你是物体识别助手。图中是从监控/巡检照片裁剪出的候选区域（可能含少量背景）。
YOLO 初判类别：{yolo_class}（置信度 {yolo_conf:.2f}）。请复核该裁剪内**最主要、最居中**的物体，只输出 JSON：

{
  "class": "Lighter" | "cig-pack" | "cigarette" | "cigarettes_butts" | "other",
  "confidence": 0.0到1.0的数字,
  "multiple_objects": true或false,
  "objects_present": ["cig-pack", "Lighter"] 或 [],
  "reason": "一句话说明"
}

判断要点：
- Lighter：一次性塑料/金属打火机，有点火按钮/金属罩，细长，常见绿/红/透明
- cig-pack：长方形纸盒或金属箔烟盒，有品牌文字、健康警示；**绿色包装仍是烟盒，不是打火机**
- cigarettes_butts：短小的滤嘴烟头残留
- cigarette：完整未点燃的单根香烟
- other：以上都不是或无法判断

【特殊情况 — 务必注意】
1. 打火机在烟盒上面/旁边/贴靠：框内可能有两个物体 → multiple_objects=true，objects_present 列全
2. 只有一个大框但明显含烟盒+打火机 → multiple_objects=true；不要整框统一判成 Lighter
3. 两个绿色物体：烟盒=扁平长方纸盒；打火机=更细长、有金属点火结构
4. YOLO 初判 cig-pack 且能看到纸盒/品牌/警示语 → 倾向保持 cig-pack，除非裁剪内明确只有打火机
5. YOLO 初判 Lighter 且能看到点火按钮/金属罩 → 倾向保持 Lighter
6. 只有非常确定裁剪内仅有单一物体时，才设 multiple_objects=false

输出要求：严格 JSON。
```

#### 6.6.2 无 YOLO 初判（备用 `PROMPT`）

结构相同，去掉「YOLO 初判类别…」一句；特殊情况条款更细（含「面积最大或最居中」「绿色烟盒勿判打火机」等）。全文见源码 `vlm_arbitrate.py` 中 `PROMPT` 常量。

### 6.7 API 请求形态

- Endpoint：`{api_base}/chat/completions`（默认 `http://127.0.0.1:8001/v1/chat/completions`）  
- `model`：`qwen3-vl-idc`  
- `messages`：user 内容 = `image_url`（data URL jpeg base64）+ text prompt  
- 响应：从 content 中正则抽取 `{...}` 再 `json.loads`  
- 类别映射：`NAME_TO_ID`；`other` → 不更新检测类

### 6.8 与「机柜状态 JSON 巡检」的区别

| | 异常目标二次仲裁 | 机柜状态结构化识别 |
|--|------------------|-------------------|
| 输入 | YOLO 裁出的 **ROI** | **整图** |
| Prompt | 四类物体 JSON | 柜门/指示灯/温湿度等（`ask_vl.py` 按文件名路由） |
| 是否改检测框 | 只改类 | 不产出检测框 |
| 典型延迟 | 按门控次数累计 | ~0.4–0.5 s/图 |

二者共用同一套 vLLM 上的 Qwen3-VL-32B-FP8，任务与 prompt 不同。

---

## 7. 关键文件索引

| 路径 | 说明 |
|------|------|
| `merge_datasets.py` | 多源合并 |
| `train_multiclass.py` / `train_finetune_*.py` | 训练入口 |
| `predict_multiclass.py` | 推理入口 |
| `postprocess.py` | IoU 规则 |
| `vlm_gate.py` | 门控 |
| `vlm_arbitrate.py` | Prompt、客户端、采纳逻辑 |
| `eval_testset.py` | 测试集 mAP ±VLM |
| `eval_vlm_boxlog.py` | 框级日志 + Acc + bootstrap |
| `eval_image_0828.py` | 5 张难例个案 |
| `doc/multiclass_training_guide.md` | 早期合并训练说明 |
| `doc/机房巡检-VLM感知模块.md` | 面试深挖备战（含未跑参考值） |
| `/data/zhangzhe/docs/cig-yolo-sam3-vlm-interface.md` | 接口/启动手册 |
| `/data/zhangzhe/docs/cig-vlm-boxlog-eval.md` | Acc 评测报告 |

---

## 8. 简历可用短述（基于本文真实数字）

> 基于 YOLOv8 的烟头/烟盒/香烟/打火机多类检测（合并测试集 429 图 / 617 框，mAP@0.5≈0.86）；本地部署 Qwen3-VL-32B-Instruct（FP8）对门控易混框做类别二次仲裁（不改定位）。以门控且 IoU≥0.5 对齐框上的分类准确率评估 VLM：N=196、M=138，Acc 100%→97.1%（按图 bootstrap Δ 95%CI [−6.0%,−0.6%]），完整测试集 mAP 0.857→0.818，说明分布内无可纠正类错时仲裁会误伤，故采用条件门控而非全量调用。

---

*文档生成日期：2026-09-06 · 路径：`doc/机房巡检-数据集训练测试与VLM仲裁.md`*
