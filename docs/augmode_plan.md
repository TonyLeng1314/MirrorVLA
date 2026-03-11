# Aug-Mode 条件化方案：实现步骤梳理

## 目标概述

1. **去掉**主模型的历史帧处理。
2. **新增**一个专门预测当前 aug mode（0/1/2/3）的模块。
3. **主模型训练**：用 GT aug_mode 作为 condition，只预测动作（无历史帧）。
4. **Aug-mode 预测器**：用定制数据单独训练；数据来自各 Libero 子任务初始状态下的「体操」轨迹（固定 y 轴 +0.2 等）。
5. **测试**：episode 开始时执行一次与训练对齐的体操 → 预测一次 aug_mode → 之后整局都以该 aug_mode 为 condition 做动作预测。

---

## 一、去掉模型的历史帧处理

### 1.1 训练脚本 `vla-scripts/finetune.py`

- **配置**：保证默认或显式 `history_frames=0`，不再使用 `use_history_hidden_states` / `history_compression_rate` / `normalize_history_actions_sign_only`。
- **前向**：删除「对 `pixel_values_history` 做多帧 VLA forward、拼 `history_multi_layer_hidden_states`」的整段逻辑（约 394–498 行附近）。
- **predict_kwargs**：不再传入 `actions_hidden_states_history`、`actions_history`。
- **build dataloader**：传 `history_frames=0`（以及 `history_compression_rate=1`），与无历史配置一致。

### 1.2 数据集 `prismatic/vla/datasets/datasets.py`

- **RLDSBatchTransform**：保留 `history_frames` 参数以兼容接口，但当 `history_frames=0` 时不再构造 `pixel_values_history`、`actions_history`（现有逻辑已支持，只需确保调用处传 0）。
- 若希望彻底清理，可后续再删 history 相关字段与分支。

### 1.3 数据处理 `prismatic/util/data_utils.py`

- Collate 时若仍有 `pixel_values_history` / `actions_history`，在 `history_frames=0` 下应为 `None`，无需改逻辑也可；若要精简，可不再在 output 中包含这两项当其为 None 时。

### 1.4 动作头 `prismatic/models/action_heads.py`

- **L1RegressionActionHead**：构造时传 `use_history_version=False`（或从 cfg 读，与 `history_frames==0` 一致）。
- **MLPResNet / MLPResNetBlock_Pro_History**：保留类与接口，仅通过 `use_history_version=False` 不再走 history 分支即可，无需删文件。

### 1.5 推理与评测

- **openvla_utils.py**：`get_vla_action` 中不再构建 `pixel_values_history`、`actions_history`（即当 `history_frames=0` 时不再传 history 相关参数）。
- **run_libero_eval_quadrant.py**：不再维护 `history_observations` / `history_actions_deque`，不传 `history_images` / `history_actions`。
- **submit/train_spatial.slurm**：去掉 `--history_frames`、`--use_history_hidden_states`、`--normalize_history_actions_sign_only`、`--history_compression_rate` 等与历史相关的参数。

---

## 二、主模型：将 aug_mode 作为 condition 接入

### 2.1 数据侧（已有基础）

- **datasets.py**：`enable_four_quadrant_input` 时已随机采样 `aug_mode in {0,1,2,3}` 并做图像/动作变换，且把 `aug_mode` 放进 return_dict。
- **data_utils.py**：已在 collate 时 stack `aug_mode` 为 `[B]` 的 long tensor。
- 训练时主模型只使用「当前帧图像 + 当前 task + GT aug_mode」，不再使用历史帧。

### 2.2 动作头侧 `prismatic/models/action_heads.py`

- **L1RegressionActionHead**：
  - 新增可选参数，例如 `use_aug_mode_condition: bool = False`，以及 `aug_mode_embed_dim`（或直接用 `llm_dim`）。
  - 新增 `aug_mode_embedding`：`nn.Embedding(4, embed_dim)` 或小型 MLP(1 → embed_dim)，将 0/1/2/3 映射为向量。
  - 在 `predict_action` 中新增参数 `aug_mode=None`（LongTensor [B]）。
  - 若 `aug_mode is not None`：对 `aug_mode` 做 embedding，得到 `[B, 1, D]`，与现有的 `p`（proprio_features）拼接或单独作为一路 condition。
  - 当前 `p` 来自 `proprio_projector(proprio)`；若项目里存在「无 proprio」的配置，需要兼容 `proprio is None`（例如用零向量或仅用 aug_mode）；若始终有 proprio，可把 `p = concat(proprio_features, aug_mode_embedding)` 或 `p = proprio_features + aug_mode_embedding`（维度需一致）。
- **MLPResNetBlock_Pro / MLPResNetBlock_Pro_History**：已有 `p` 的 cross-attention 条件，只需保证传入的 `p` 已包含 aug_mode 信息即可，无需改 block 内部。

### 2.3 训练脚本 `vla-scripts/finetune.py`

- 从 `batch["aug_mode"]` 读取 GT aug_mode（需保证 dataset 与 collate 在 `enable_four_quadrant_input` 时提供）。
- 在构建 `predict_kwargs` 时增加 `aug_mode=batch["aug_mode"]`（以及若 action head 需要 `aug_mode_projector` 也一并传入，若用 embedding 则不需要 projector）。
- 若使用「仅 aug_mode、无 proprio」配置，需在 `predict_action` 内处理 `proprio is None` 时仅用 aug_mode 生成 `p`。

### 2.4 配置

- 在 finetune 的配置中增加，例如：`use_aug_mode_condition: bool = True`、`enable_four_quadrant_input: bool = True`，并保证 dataloader 与 action head 的开关一致。

---

## 三、Aug-Mode 预测器模块（新网络）

### 3.1 输入输出定义

- **输入**（二选一或两者都支持）：
  - 方案 A：体操**前后两帧图像**（before / after 固定步数 y 轴 +0.2）。
  - 方案 B：**光流**（由前后图像计算）+ 可选 before 图。
- 再加上「这一小段动作的 **action sign**」（例如 [0, +1, 0, 0, 0, 0, 0] 表示 y 轴正方向）。
- **输出**：4 类 logits（对应 aug_mode 0/1/2/3），或 one-hot + CE loss。

### 3.2 网络结构建议

- **图像版**：小型 CNN（或复用现有 vision encoder 的几层）对「before + after」拼接或分别编码再 concat，再与 action sign 的 embedding 拼接，过 MLP → 4 维 logits。
- **光流版**：光流作为 2 通道输入进 CNN，其余同上。
- 单独文件，例如 `prismatic/models/aug_mode_predictor.py`，便于单独训练和加载。

### 3.3 与主模型的关系

- 训练时**不**更新主 VLA，只更新 aug_mode 预测器。
- 推理时先跑 aug_mode 预测器，再把预测的 mode 作为 condition 传给主模型。

---

## 四、Aug-Mode 定制数据生成

### 4.1 数据内容

- 对**每个 Libero 子任务**（或你选定的子集）：
  - 将环境 **reset 到该任务的初始状态**（每个任务可采样多组初始状态，例如不同 seed）。
  - 执行**固定步数**的「体操」动作：仅某一轴（先验为 y 轴）每步 +0.2，其余维度为 0（或与训练时约定一致）。
  - **记录**：
    - 体操**前**的图像（主视角 + 若有 wrist 也一并）。
    - 体操**后**的图像。
    - 可选：中间步的图像用于光流（例如前→后直接算光流）。
  - 计算**图像差**或**光流**（与 3.1 的输入方案一致）。
  - 记录这段**动作的 sign**（例如每步 action 的 sign，再 aggregate 成一段的 sign 向量）。
  - **Label**：该初始状态对应的**真实 aug_mode**。  
    真实 aug_mode 需要定义方式，例如：  
    - 由场景/机器人朝向等先验规则确定；或  
    - 用「已知 aug_mode 的主模型」在同样初始状态下执行，看哪个 mode 成功；或  
    - 人工标注少量再泛化。

### 4.2 实现方式

- 写一个**离线数据生成脚本**（如 `scripts/generate_augmode_probe_data.py`）：
  - 遍历任务列表，对每个任务 reset → 执行固定步数 y+0.2 → 存图/光流 + action sign + 标签。
  - 保存为 PyTorch Dataset 可读格式（如 `.pt`、`.npz` 或 TFRecord），并写一个 `AugModeProbeDataset` 读取。
- 若使用光流，需在脚本中调用光流算法（如 OpenCV 或现成光流库），与训练时的输入格式一致。

### 4.3 还原到初始状态

- 每次生成一条样本后，**再次 reset** 该任务，再采样下一个初始状态或下一个任务，保证「体操」总是在初始状态上执行，避免累积误差。

---

## 五、Aug-Mode 预测器的训练流程

### 5.1 数据加载

- 使用上述 `AugModeProbeDataset`，DataLoader 输出：
  - `before_image`, `after_image`（或 `optical_flow`）、`action_sign`、`aug_mode`（0/1/2/3）。

### 5.2 训练脚本

- **方案 A**：单独脚本，例如 `vla-scripts/train_augmode_predictor.py`，只实例化 aug_mode 预测器，用 CE loss，训练若干 epoch。
- **方案 B**：在现有 `finetune.py` 中增加一个「phase」或单独入口，仅训练 aug_mode 预测器（冻结 VLA 和 action head），数据来自 `AugModeProbeDataset`。
- 保存 checkpoint：仅保存 aug_mode 预测器参数，便于评测时加载。

### 5.3 与主模型训练顺序

- 建议：**先**用现有数据 + GT aug_mode 训练主模型（动作预测），**再**用定制数据训练 aug_mode 预测器；或两者解耦，主模型训练不依赖预测器。

---

## 六、测试 / 评测流程

### 6.1 加载

- 加载主模型（VLA + action head）与 **aug_mode 预测器** 的 checkpoint。
- 配置：`history_frames=0`，启用 aug_mode condition（与训练一致）。

### 6.2 Episode 开始时：执行体操并预测 aug_mode

- 在 `run_libero_eval_quadrant.py` 中，在「正式执行任务步」之前：
  - 记录**当前观测**为「before」图像。
  - 执行**与数据生成时完全一致**的固定步数、固定动作（y 轴 +0.2），得到「after」图像（及可选光流）。
  - 将 before/after（或光流）+ 这段动作的 sign 送入 **aug_mode 预测器**，得到 `aug_mode_pred`（0/1/2/3）。
  - 可选：若环境允许，执行完体操后 **reset 回初始状态**再开始正式任务，这样与「在初始状态做体操」的设定更一致；否则在「体操后状态」直接开始任务也可，需与训练时的语义一致。

### 6.3 整局动作预测

- 在 `get_action`（或 openvla_utils 中封装）中，每次调用时传入**本 episode 固定的** `aug_mode=aug_mode_pred`，不再使用历史帧。
- 主模型用「当前图像 + task + aug_mode_pred」预测动作，直到 episode 结束。

---

## 七、配置与兼容性小结

| 项目         | 说明 |
|--------------|------|
| `history_frames` | 固定为 0，移除所有 history 相关参数与逻辑。 |
| `enable_four_quadrant_input` | 主模型训练时 True，以产生并利用 GT aug_mode。 |
| `use_aug_mode_condition` | 主 action head 是否接受 aug_mode 作为 condition。 |
| Aug-mode 数据 | 仅用于训练 aug_mode 预测器，与主 RLDS 数据分离。 |
| 体操步数 / 轴 | 与数据生成脚本、评测脚本**完全一致**（例如 y 轴、步数 K、每步 +0.2）。 |

---

## 八、建议实施顺序

1. **阶段 1**：去掉历史帧（第一节），跑通「无历史、无 aug_mode」的 baseline，确保不报错。
2. **阶段 2**：在主模型与 action head 中接入 aug_mode 作为 condition，用 GT aug_mode 训练主模型（第二节），验证带 aug_mode 条件的动作预测正常。
3. **阶段 3**：实现 aug_mode 预测器网络与定制数据生成（第三、四节），并训练预测器（第五节）。
4. **阶段 4**：在评测中接入「体操 + 预测 aug_mode + 整局 condition」（第六节），做端到端评测。

按上述顺序可以最小化一次改动的范围，每步都可单独验证。若你希望，我可以从「阶段 1」或「阶段 2」开始，按文件逐处给出具体 diff 级别的修改建议（包括 `predict_action` 里对 `proprio is None` 的兼容）。
