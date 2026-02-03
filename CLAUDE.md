# LLM_LTL 项目设计文档

## 项目概述

使用VLM (Vision Language Model) 辅助强化学习训练机器人操作任务 (Lift, Stack等)。

---

## 1. Replay Buffer 设计

### 1.1 存储粒度
- **按 transition (step) 存储**，不是按 episode
- 一个 transition = `(s, a, r, s', done, env_info)`
- env_info 包含 VLM 评分信息

### 1.2 环形缓冲区 (Ring Buffer)
```python
self._top = (self._top + 1) % self._max_replay_buffer_size
```
- 固定大小上限 (默认 100,000)
- 满后 FIFO 覆盖最旧样本
- **问题**: 按时间覆盖，而非按价值覆盖

### 1.3 已知问题
| 问题 | 影响 |
|------|------|
| 时序关系丢失 | 无法学习动作序列的因果关系 |
| 信用分配困难 | 关键决策步骤可能优先级低 |
| Episode边界模糊 | terminal 状态处理需注意 |
| VLM评分陈旧 | 存入时计算，不随policy更新 |

---

## 2. VLM 奖励设计 (B+C+D 方案)

### 2.1 VLM 评估信号
```python
action_score: float  # 动作合理性 (0-1)
progress: float      # 任务完成进度 (0-1)
```

### 2.2 奖励计算链

```
原始VLM奖励:
  vlm_raw = action_weight × action_score + progress_weight × progress_reward
          = 0.2 × action_score + 0.8 × progress_reward

奖励分摊 (Reward Spreading):
  vlm_spread = vlm_raw / eval_frequency
             = vlm_raw / 5  (每5步评估一次，奖励分摊到5步)

缩放:
  vlm_scaled = vlm_spread × vlm_reward_scale × curriculum_weight
             = vlm_spread × 2.0 × (1.0→0.1)

最终奖励:
  total_reward = env_reward + vlm_scaled
```

### 2.3 势能奖励 (Potential-Based Shaping) - 方案B

```python
if use_potential_shaping:
    # Φ(s) = progress
    progress_reward = γ × current_progress - prev_progress
    # γ = 0.99
else:
    progress_reward = max(0, current_progress - prev_progress)
```

**特点**:
- 理论保证不改变最优策略
- Progress 下降时给负奖励
- **问题**: Progress 稳定时奖励为负，可能抑制学习

### 2.4 课程学习 (Curriculum) - 方案D

```python
def _get_curriculum_weight(self) -> float:
    progress = current_epoch / total_epochs
    if curriculum_mode == "linear":
        return start + (end - start) × progress  # 1.0 → 0.1
```

**目的**: 训练初期依赖VLM引导，后期逐渐依赖环境奖励

### 2.5 当前配置
```python
vlm_reward_scale = 2.0        # 原始0.5，增大4倍
eval_frequency = 5            # 每5步评估一次VLM
action_weight = 0.2           # action_score 权重
progress_weight = 0.8         # progress 权重
use_potential_shaping = True  # 使用势能奖励
curriculum_mode = "linear"    # 线性衰减
curriculum_start = 1.0
curriculum_end = 0.1
```

---

## 3. 优先级采样设计

### 3.1 Hybrid 模式 (VLM + TD-error)

```python
priority = vlm_priority^(α×vlm_weight) × td_priority^(α×td_weight)

其中:
- vlm_priority = vlm_score + ε
- td_priority = |td_error| + ε
- α = 0.6 (优先级指数)
- vlm_weight = 0.5
- td_weight = 0.5
```

### 3.2 VLM 优先级计算

```python
# Progress 非线性加权
if progress < 0.1:
    progress_factor = progress  # 低优先级
elif progress < 0.8:
    progress_factor = progress + 0.3  # "差点成功"加成
else:
    progress_factor = progress  # 成功轨迹

vlm_score = action_score_weight × action_score + progress_weight × progress_factor
```

**设计意图**: 中等进度 (0.1-0.8) 的轨迹最有学习价值

### 3.3 采样方式
- **分段优先级采样** (Stratified Prioritized Sampling)
- 不是连续采样，每个 batch 内样本不相邻
- **问题**: 打乱时序关系，难以学习动作序列

### 3.4 重要性采样权重 (IS Weights)
```python
weight = (N × sampling_prob)^(-β)
weight = clip(weight, 0, max_is_weight)  # max_is_weight = 10.0
β: 0.4 → 1.0 (退火)
```

### 3.5 当前配置
```python
priority_mode = "hybrid"
alpha = 0.6
beta_start = 0.4
beta_end = 1.0
vlm_weight = 0.5
td_weight = 0.5
max_is_weight = 10.0
max_td_error = 20.0
min_progress_threshold = 0.0  # 不过滤
```

---

## 4. 训练流程

### 4.1 每个 Epoch
```
1. 探索 (Exploration)
   - 收集 num_expl_steps_per_train_loop 步 (默认3000)
   - 4个并行环境
   - 每步存入 buffer

2. 训练 (Training)
   - 重复 num_trains_per_train_loop 次 (默认1000)
   - 每次: 按优先级采样 batch_size=256 个 transition
   - 计算 loss, 更新网络
   - 更新 TD-error, 刷新优先级

3. 评估 (Evaluation)
   - 运行 num_eval_steps_per_epoch 步
   - 记录 success rate 等指标
```

### 4.2 样本复用率
```
每 epoch 采样: 1000 × 256 = 256,000 个样本
每 epoch 新增: ~875 个 transition (incomplete paths 被丢弃)
复用率: 256,000 / 875 ≈ 293 倍
```

---

## 5. 早停机制 (Early Stopping)

```python
early_stop_success_threshold = 0.95  # 成功率阈值
early_stop_patience = 3              # 连续达标次数
```

当连续 3 次评估 success_rate >= 95% 时停止训练。

---

## 6. 待改进方向

### 6.1 采样策略
- [ ] N-step returns: 采样连续 n 步
- [ ] Trajectory-level 采样: 保持 episode 完整性
- [ ] **Hindsight Experience Replay (HER)**: 解决稀疏奖励
  - HER与PER可以并行使用 (HER决定存什么，PER决定采什么)
  - 适配方案: 子目标HER或VLM+HER
  - VLM判断"实际完成了什么"作为hindsight goal

### 6.2 优先级更新
- [ ] 定期重新评估 VLM 分数 (解决陈旧问题)
- [ ] 按价值删除而非 FIFO
- [ ] **TD-error与VLM数量级平衡**:
  - 当前: VLM(0-1) vs TD(0-20)，TD贡献略大
  - 方案: 归一化TD或调整权重(vlm_weight=0.7)

### 6.3 VLM 奖励
- [ ] 考虑关闭势能奖励 (progress 稳定时负奖励问题)
- [ ] 软动作评分 (方案C): VLM 为所有 primitive 打分
- **注意**: 势能函数只影响reward，不影响采样优先级
  - 优先级使用 `vlm_progress` (原始进度 0-1)
  - 奖励使用 `vlm_progress_reward` (势能差，可为负)

### 6.4 Episode 长度优化
**问题**: 每epoch只有约24个episode (3000步 / 136步平均长度)

**方案A: 缩短 max_path_length**
```python
max_path_length = 120  # 从150减到120
# 预计增加到 ~30 episodes/epoch
```

**方案B: 动态截断 (推荐)**
```python
# 连续N步progress没有增长就提前结束
if steps_without_progress > 30:
    done = True
```
- 成功轨迹不受影响
- 失败/卡住的轨迹提前结束
- 减少无效样本

### 6.5 Flow Matching Loss (待探索)

**潜在应用场景:**

1. **动作流 (Action Flow)** - 替代高斯策略
```python
# 从噪声 z 流向目标动作 a*
loss_fm = ||v_θ(a_t, t) - (a* - a_t)||²
# 好处: 更好的多模态动作分布建模
```

2. **轨迹流 (Trajectory Flow)** - 生成连贯轨迹
```python
# 学习生成完整轨迹，保持时序连贯性
trajectory = [s0, a0, s1, a1, ..., sT, aT]
loss_fm = ||v_θ(τ_t, t) - (τ* - τ_t)||²
# 好处: 解决"采样打乱时序"问题
```

3. **状态预测 (Dynamics Flow)** - World Model
```python
loss_fm = ||v_θ(s'_t, t, s, a) - (s'* - s'_t)||²
# 好处: 更好的规划能力
```

**与SAC整合:**
```python
# 当前: π(a|s) = Gaussian(μ(s), σ(s))
# 改进: π(a|s) = FlowModel(z → a | s)
# Loss: SAC_loss + λ × flow_matching_loss
```

---

## 9. 当前实验状态 (2026-02-02)

### 正在运行: Stack VLM 2x
```bash
# 命令
nohup uv run python experiments/run_vlm_maple.py \
  --task Stack --curriculum-mode linear --vlm-priority \
  --num-envs 4 --min-steps 10000 --replay-buffer-size 100000 \
  --epochs 500 --vlm-no-warmup --vlm-freq 5 --vlm-scale 2.0 \
  --vlm-priority-max-td-error 20.0 \
  --early-stop-threshold 0.95 --early-stop-patience 3 \
  --label "stack-vlm2x-v2"
```

### TensorBoard
- Port 6010: `http://zjj:6010/`
- 目录: `data/vlm-maple-Stack`

### 监控
- 日志: `/tmp/vlm_stack.log`
- 监控脚本: 每20分钟输出到 `/tmp/stack_monitor.log`

### 观察到的问题
1. **vlm_reward_scaled 很小** (~0.036) - 因为势能奖励在progress稳定时为负
2. **Success rate 仍为 0%** - Stack任务较难，需要更多epoch
3. **num_paths 固定约24** - 受限于探索步数和episode长度

### 实验对比问题 (02-01 vs 01-31)

**现象**: `02-01-stack-vlm2x-v2` 和 `01-31-td-only-v1` 的训练曲线很相似

**原因分析**:

| 因素 | 01-31 (旧) | 02-01 (新) | 影响 |
|------|-----------|-----------|------|
| seed | 0 | 0 | 初始探索相同 |
| buffer_size | **1,000,000** | **100,000** | 旧的10倍大 |
| priority_mode | td_only | hybrid | β=1后差异被校正 |
| beta (当前) | 1.0 | 1.0 | IS权重完全校正 |

**为什么曲线相似**:
1. **Seed=0** → 网络初始化、探索轨迹相同
2. **β=1.0** → IS权重校正后，priority采样效果≈均匀采样
3. **Buffer未满** → epoch 108时新实验88K/100K，旧实验88K/1M，内容几乎相同
4. **任务难度主导** → Stack本身很难，算法差异被"淹没"

**预计差异显现时机**:
- 新实验 epoch ~115 后 buffer 开始覆盖
- 届时新旧实验的 buffer 内容会不同

**验证方法**:
1. 用不同 seed 对比
2. 降低 beta_end (如0.6) 保留采样偏差
3. 增大 vlm_weight (如0.8) 强化VLM影响

---

## 10. Sim2Real 迁移规划

### 10.1 问题：Sim-specific 奖励无法迁移
- `aff_success`: 依赖精确物体位置 → 真实世界无法获取
- `grasped`: 依赖接触检测 → 需要传感器
- `contact_force`: 依赖物理仿真 → 真实世界不精确

### 10.2 解决方案：用 VLM 替代环境奖励

| Sim奖励 | VLM替代方案 | Prompt示例 |
|---------|------------|-----------|
| `aff_success` | `vlm_affordance` | "Is gripper positioned correctly to grasp?" |
| `grasped` | `vlm_grasp_state` | "Is the robot holding an object?" |
| `progress` | `vlm_progress` | "Task completion percentage?" (已实现) |
| `success` | `vlm_task_success` | "Is the task completed successfully?" |

### 10.3 迁移路径
```
Phase 1 (Sim): env_reward + vlm_reward
Phase 2 (迁移): 逐步用VLM替代env_reward
Phase 3 (Real): 纯VLM奖励，不依赖simulator state
```

### 10.4 待实现的 VLM 评估方法
```python
# vlm_client.py 需要新增:
def evaluate_affordance(image, skill_name) -> float
def evaluate_grasp_state(image) -> float
def evaluate_contact(image, image_prev) -> float
```

---

## 11. LTL RewardMachine 设计

### 11.1 核心概念

基于论文:
- **IJCAI 2019**: "LTL and Beyond: Formal Languages for Reward Function Specification"
- **TRAPs (IEEE Cybernetics 2024)**: "Task-Driven RL with Action Primitives"

```python
# Reward Machine = Finite State Automaton + Reward Function
RM = (States, Events, Transitions, Rewards, Terminal)
```

### 11.2 势函数与距离计算

```python
# BFS 计算每个状态到终态的最短距离
distance[state] = BFS(state -> terminal_states)

# 势函数: Φ(q) = -dist(q, F) / max_dist
# Φ ∈ [-1, 0], 终态=0, 初态=-1

# Potential-based Shaping (保证不改变最优策略)
R_shaping = γ × Φ(q') - Φ(q)
```

### 11.3 奖励模式

| 模式 | 公式 | 适用场景 |
|------|------|----------|
| **sparse** | R = 1 if terminal else 0 | 简单任务、baseline |
| **distance** | R = -dist(q, F)/max_dist | 连续引导 |
| **progression** | R = transition_reward + terminal_bonus | 离散里程碑 |
| **hybrid** | R = transition + scale×(γΦ'-Φ) + terminal | 推荐 |

### 11.4 Stack 任务示例

```
LTL: ◇(grasped ∧ ◇(lifted ∧ ◇(aligned ∧ ◇stacked)))

状态机:
u0 (init) --[cubeA_grasped]--> u1 (grasped) --[cubeA_lifted]--> u2 (lifted)
    |                                                               |
    dist=4                                                       dist=2
    Φ=-1.0                                                       Φ=-0.5
                                                                    |
                                                            [cubes_aligned]
                                                                    v
u4 (terminal) <--[stacked]-- u3 (aligned)
    dist=0                      dist=1
    Φ=0.0                       Φ=-0.25
```

### 11.5 当前配置

```python
reward_mode = "hybrid"
gamma = 0.99
use_potential_shaping = True
potential_scale = 0.1           # 势能奖励缩放
terminal_reward = 1.0           # 终态额外奖励
```

### 11.6 关键文件

| 文件 | 功能 |
|------|------|
| `llm_ltl/reward_machines/reward_machine.py` | RM核心类、距离计算、势函数 |
| `llm_ltl/reward_machines/rm_factory.py` | 8个任务的RM工厂 |
| `llm_ltl/reward_machines/propositions/*.py` | 8个任务的命题函数 |
| `llm_ltl/envs/rm_wrapper.py` | 环境包装器 |
| `experiments/run_rm_maple.py` | RM训练入口 |

---

## 12. LTL RM vs VLM 设计对比

### 12.1 核心差异

| 维度 | LTL RewardMachine | VLM Reward |
|------|-------------------|------------|
| **信号来源** | 形式化规则 (代码定义) | 视觉模型推理 (神经网络) |
| **可解释性** | ✅ 高 (状态机可视化) | ❌ 低 (黑盒) |
| **延迟** | ✅ 毫秒级 (本地计算) | ❌ 百毫秒级 (API调用) |
| **泛化性** | ❌ 任务特定 | ✅ 可迁移 (换任务改prompt) |
| **Sim2Real** | ⚠️ 需要真实世界命题 | ✅ 天然迁移 (视觉输入) |

### 12.2 奖励信号对比

| 特性 | LTL RM | VLM |
|------|--------|-----|
| **粒度** | 离散 (状态转移时) | 连续 (每步/每N步) |
| **范围** | 稀疏 (只在转移时非零) | 稠密 (每步都有信号) |
| **progress** | 基于距离: 1 - dist/max_dist | VLM评估: 0-1连续值 |
| **势函数** | Φ(q) = -dist/max_dist | Φ(s) = progress |

### 12.3 势函数设计对比

```python
# LTL RM 势函数
Φ_rm(q) = -distance(q, terminal) / max_distance
R_rm = γ × Φ_rm(q') - Φ_rm(q)
# 特点: 基于状态机结构，离散、精确

# VLM 势函数
Φ_vlm(s) = vlm_progress(image)
R_vlm = γ × Φ_vlm(s') - Φ_vlm(s)
# 特点: 基于视觉感知，连续、可能噪声
```

### 12.4 互补关系

```
┌─────────────────────────────────────────────────────────┐
│                    Total Reward                          │
│  = env_reward                                            │
│  + rm_scale × R_rm      (结构化里程碑)                    │
│  + vlm_scale × R_vlm    (连续进度评估)                    │
└─────────────────────────────────────────────────────────┘

LTL RM: 提供"骨架"奖励 (离散子目标)
VLM:    提供"肌肉"奖励 (连续进度引导)
```

### 12.5 推荐组合策略

| 场景 | 配置 |
|------|------|
| **Sim训练 (快速)** | RM only, mode=hybrid |
| **Sim训练 (精细)** | RM + VLM, rm_weight=1.0, vlm_weight=0.5 |
| **Sim2Real迁移** | 逐步从RM过渡到VLM |
| **Real部署** | VLM only (不依赖simulator state) |

### 12.6 命题 vs VLM 评估

| 命题 (RM) | VLM等价评估 | 迁移难度 |
|-----------|------------|----------|
| `cube_grasped` | "Is the robot holding the cube?" | 中 |
| `cube_lifted` | "Is the cube above the table?" | 易 |
| `cubes_aligned` | "Is red cube above green cube?" | 易 |
| `stacked` | "Are the cubes stacked?" | 易 |

**结论**: 大多数RM命题可以转换为VLM prompt，实现Sim2Real迁移。

---

## 13. "成功后回退"问题与状态验证

### 13.1 问题发现 (2026-02-02)

在 Stack VLM2x-v2 训练中观察到：

```
Num Rollout Success: 90% (曾经成功)
final/success Mean:  55% (最终成功)
差值:                35% → "成功后又失败了"
```

**问题原因**:
1. Agent 把 cubeA 堆到 cubeB 上 → `success=True`
2. Agent 继续执行动作，撞倒 cube → `success=False`
3. Rollout 结束时，任务实际失败

### 13.2 指标定义差异

| 指标 | 计算方式 | 含义 |
|------|----------|------|
| `Num Rollout Success` | `any(info['success'] for info in rollout)` | 过程中曾经成功过 |
| `final/success Mean` | `rollout[-1]['success']` | 最终时刻是否成功 |

```python
# eval_util.py 中的计算
def get_num_rollout_success(paths):
    num_success = 0
    for path in paths:
        if any([info.get('success', False) for info in path['env_infos']]):
            num_success += 1  # 只要有一步成功就算
    return num_success
```

### 13.3 状态验证机制 (State Validation)

为解决此问题，在 LTL RM 中实现了状态验证：

```python
# reward_machine.py
class RewardMachine:
    def __init__(self, ..., state_validators=None):
        # state_validators = {state: (required_events, fallback_state, penalty)}
        self.state_validators = state_validators or {}

    def _validate_current_state(self, events):
        """如果当前状态的必需事件不满足，回退到fallback状态"""
        if self.current_state in self.state_validators:
            required, fallback, penalty = self.state_validators[self.current_state]
            if not required.issubset(events):
                self.current_state = fallback
                return True, penalty
        return False, 0.0
```

### 13.4 Stack 任务的状态验证配置

```python
state_validators = {
    # 状态: (必需事件, 回退目标, 惩罚)
    'u1': ({'cubeA_grasped'}, 'u0', 0.1),    # 抓取后必须保持
    'u2': ({'cubeA_grasped'}, 'u0', 0.15),   # 抬起后必须保持
    'u3': ({'cubeA_grasped'}, 'u0', 0.2),    # 对齐后必须保持
    # u4 (stacked) 是终态，完成后不再验证
}
```

### 13.5 回退时的惩罚机制

当检测到回退时，有两重惩罚：

1. **势函数惩罚** (自动):
   ```python
   # Φ 从高变低 → shaping 为负
   shaping = γ × Φ(new) - Φ(old)
   # 例: u2→u0, Φ: -0.5 → -1.0
   # shaping = 0.99 × (-1.0) - (-0.5) = -0.49
   ```

2. **显式回退惩罚** (可配置):
   ```python
   if regressed:
       reward -= regression_penalty  # 额外惩罚
   ```

### 13.6 预期效果

使用状态验证后，Agent 应该学会：
1. **成功后保持稳定**，不做多余动作
2. **避免回退行为**，因为回退会被惩罚
3. **final/success 更接近 Num Rollout Success**

### 13.7 验证实验

待运行: 对比有无状态验证的训练结果

```bash
# 无状态验证 (baseline)
python experiments/run_rm_maple.py --task stack --no-state-validation

# 有状态验证
python experiments/run_rm_maple.py --task stack --state-validation
```

**评估指标**:
- `final/success` / `Num Rollout Success` 比例
- 回退次数统计 (`rm/total_regressions`)

---

## 7. 关键文件

| 文件 | 功能 |
|------|------|
| `llm_ltl/vlm/vlm_reward_wrapper.py` | VLM奖励计算、势能奖励、课程学习 |
| `llm_ltl/vlm/vlm_client.py` | VLM API 调用 |
| `maple/maple/data_management/vlm_prioritized_replay_buffer.py` | 优先级采样 buffer |
| `maple/maple/torch/vlm_batch_rl_algorithm.py` | 训练循环、早停 |
| `maple/maple/torch/sac/sac_hybrid_vlm.py` | SAC训练器、IS权重 |
| `experiments/run_vlm_maple.py` | 实验入口、CLI参数 |

---

## 8. TensorBoard 关键指标

| 指标 | 含义 |
|------|------|
| `eval/env_infos/success Mean` | 评估成功率 (主要指标) |
| `replay_buffer/size` | Buffer 实际大小 (有上限) |
| `expl/num steps total` | 累计探索步数 (无上限) |
| `vlm_reward_scaled Mean` | VLM奖励缩放后均值 |
| `vlm_progress Mean` | VLM进度评估均值 |
| `trainer/Q Predictions Mean` | Q值预测 |
| `trainer/Policy Loss` | 策略损失 |
