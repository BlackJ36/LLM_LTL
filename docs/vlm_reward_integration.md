# VLM 奖励指导集成方案

## 概述

本文档描述了将 Vision-Language Model (VLM) 集成到 MAPLE 层次化强化学习框架中的方案。VLM 作为额外的奖励信号源，指导动作基元的选择。

## 设计目标

- **方案选择**: VLM 作为奖励/指导信号（方案3），而非替换 RL 策略
- **VLM 模型**: Qwen3-VL-8B-Instruct，本地部署于 `http://172.19.1.40:8001`
- **评估方式**: 二元判断 + 进度评估

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Training Loop                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   observation ──→ PAMDPPolicy ──→ action (primitive + params)
│        │                              │                     │
│        │                              ▼                     │
│        │         ┌──────────────────────────────────┐      │
│        │         │     VLMRewardWrapper             │      │
│        │         │  ┌────────────────────────────┐  │      │
│        └────────→│  │ 1. Render image            │  │      │
│                  │  │ 2. Get primitive name      │  │      │
│                  │  │ 3. Call VLM API (async)    │  │      │
│                  │  │ 4. Compute VLM reward      │  │      │
│                  │  └────────────────────────────┘  │      │
│                  │                                   │      │
│                  │  reward_total = env_reward        │      │
│                  │              + scale * vlm_reward │      │
│                  └──────────────────────────────────┘      │
│                              │                              │
│                              ▼                              │
│                      SAC Trainer                            │
└─────────────────────────────────────────────────────────────┘
```

## 文件结构

```
llm_ltl/
├── __init__.py
└── vlm/
    ├── __init__.py
    ├── vlm_client.py           # Qwen3-VL API 客户端
    ├── vlm_reward_wrapper.py   # 环境包装器 (同步/异步)
    ├── vlm_maple_launcher.py   # MAPLE 训练集成
    └── test_vlm_reward.py      # 测试脚本

experiments/
└── run_vlm_maple.py            # 运行训练脚本
```

## 核心组件

### 1. VLM Client (`vlm_client.py`)

与 Qwen3-VL API 通信的客户端。

**主要方法:**
- `evaluate_action()`: 二元判断动作选择是否合理
- `evaluate_progress()`: 评估任务进度 (0-1)
- `get_combined_reward()`: 综合计算 VLM 奖励

**奖励模式 (`reward_mode`):**
| 模式 | VLM 认为合理 | VLM 认为不合理 | 适用场景 |
|------|-------------|---------------|---------|
| `bonus_only` | +reward | 0 | 稠密环境奖励（推荐） |
| `symmetric` | +reward | -reward | 稀疏环境奖励 |
| `penalty_only` | 0 | -reward | 需要约束探索 |

### 2. VLM Reward Wrapper (`vlm_reward_wrapper.py`)

包装 robosuite 环境，添加 VLM 奖励。

**两个版本:**
- `VLMRewardWrapper`: 同步版本，每步阻塞等待 VLM
- `VLMRewardWrapperAsync`: 异步版本，后台线程评估（推荐）

**关键参数:**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `vlm_reward_scale` | 0.5 | VLM 奖励权重 |
| `eval_frequency` | 1 | 评估频率（1=每步） |
| `reward_mode` | `bonus_only` | 奖励计算模式 |
| `binary_weight` | 0.5 | 二元判断权重 |
| `progress_weight` | 0.5 | 进度评估权重 |

### 3. MAPLE Launcher (`vlm_maple_launcher.py`)

扩展 MAPLE 训练启动器，支持 VLM 奖励。

## 与稠密环境奖励的兼容性

Robosuite 使用稠密奖励 (`reward_shaping=True`)：
- Reaching: 0 ~ 1.0 (距离)
- Grasping: 0 | 0.25
- Lifting: 0 | 1.0

**问题**: VLM 奖励可能与环境奖励冲突

**解决方案**: 使用 `bonus_only` 模式
- VLM 只给正向奖励，不惩罚
- 环境奖励负责"执行效果"
- VLM 负责"决策加分"

## 使用方法

### 测试 VLM 连接

```bash
uv run python -m llm_ltl.vlm.test_vlm_reward --test-vlm-only
```

### 运行训练

```bash
# Debug 模式（快速测试，5 epochs）
uv run python experiments/run_vlm_maple.py --task Lift --num-envs 3 --debug

# 带 VLM 奖励（默认配置，3 并行环境）
uv run python experiments/run_vlm_maple.py --task Lift --num-envs 3

# 自定义参数
uv run python experiments/run_vlm_maple.py \
    --task Lift \
    --num-envs 3 \
    --vlm-scale 0.5 \
    --vlm-freq 1 \
    --vlm-mode bonus_only \
    --epochs 500

# 基线对比（无 VLM）
uv run python experiments/run_vlm_maple.py --task Lift --num-envs 3 --no-vlm
```

### TensorBoard 可视化

```bash
# 查看实验日志
tensorboard --logdir data/vlm-maple-Lift/

# 或查看所有实验
tensorboard --logdir data/
```

日志目录结构：
```
data/
└── vlm-maple-Lift/
    └── 01-29-vlm0.5/
        └── 01-29-vlm0.5_2026_01_29_xxx--s-0/
            ├── debug.log
            ├── progress.csv
            ├── variant.json
            └── events.out.tfevents.*  # TensorBoard 日志
```

### 配置示例

```python
vlm_variant = {
    'enabled': True,
    'task_description': 'Pick up the cube and lift it above the table',
    'api_base': 'http://172.19.1.40:8001',
    'model_name': 'Qwen/Qwen3-VL-8B-Instruct',
    'vlm_reward_scale': 0.5,
    'eval_frequency': 1,
    'use_async': True,
    'binary_weight': 0.5,
    'progress_weight': 0.5,
    'camera_name': 'frontview',
    'camera_height': 256,
    'camera_width': 256,
    'reward_mode': 'bonus_only',
}
```

## VLM Prompt 设计

### 二元判断 Prompt

```
You are evaluating a robot's action selection for a manipulation task.

Task: {task_description}

Available action primitives: {primitives}

The robot selected: "{selected_primitive}"

Based on the current scene shown in the image, is this action selection
reasonable for making progress on the task?

Respond with ONLY a JSON object:
{"reasonable": true/false, "confidence": 0.0-1.0, "reason": "..."}
```

### 进度评估 Prompt

```
You are evaluating task progress for a robot manipulation task.

Task: {task_description}

Based on the current scene shown in the image, estimate the progress
toward completing the task.

Respond with ONLY a JSON object:
{"progress": 0.0-1.0, "stage": "...", "next_step": "..."}
```

## 性能考虑

### 异步评估

- VLM 推理延迟约 0.5-2 秒
- 异步模式：VLM 奖励有 1 步延迟，但不阻塞训练
- 同步模式：严格每步评估，但训练速度受 VLM 限制

### 多相机支持

Robosuite 支持多相机：
- `frontview`: 正前方视角
- `agentview`: 机器人视角
- `sideview`: 侧视图
- `birdview`: 俯视图

## 未来扩展

1. **多相机融合**: 使用多视角图像提高 VLM 判断准确性
2. **VLM 蒸馏**: 将 VLM 知识蒸馏到轻量级网络
3. **自适应权重**: 根据训练阶段动态调整 VLM 奖励权重
4. **任务规划**: 使用 VLM 进行高层任务分解

## 参考

- MAPLE: MAnipulation Primitive-augmented LEarning
- Robosuite: A Modular Simulation Framework for Robot Learning
- Qwen-VL: A Versatile Vision-Language Model
