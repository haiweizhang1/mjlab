# 旧版与新版 Walk → Teacher → Student 训练对比

更新日期：2026-08-30  
用途：冻结两条训练谱系、说明实际差异，并避免用不可比的训练日志判断 Student 好坏。

## 1. 结论先行

旧版和新版不是“只增大了网络”的单变量对照。新版同时改变了 Walk 基线、Teacher
网络、对称损失、奖励、物理域随机化、视觉时延、Student 辅助损失和 rollout 分布。

- 旧版当前最可靠的闭环 Student 参考仍是 `CalibratedVisualDR model_3500.pt` 和后续
  `constrained model_9999.pt`。旧版早期效果好，主要因为训练分布更容易，并且很早就让
  Student 自己闭环 rollout。
- 新版 Teacher 更强、更接近 Klavier/真机扰动，但其控制 MLP 从 `512/256/128` 增大到
  `1024/512/256`，Student 的拟合任务明显更难。
- 新版 E1–E4 因子实验目前是“冻结 MLP + 100% Teacher rollout”的第一阶段消融。
  其中 reward 和 episode length 主要描述 Teacher 驱动的轨迹，不能证明 Student 闭环好。
- 比较 Student 时应以相同相机、相同命令、相同推力、相同随机种子的固定闭环 sim2sim
  成功率为主，不能只比较 behavior/latent loss 或训练 reward。

## 2. 两条完整谱系

| 阶段 | 旧版谱系 | 新版谱系 |
|---|---|---|
| Walk | `football_pretrain/.../2026-07-23_18-17-07/model_16000.pt` | `g1_velocity_walk_klavier_replica/.../2026-08-24_11-38-37_.../model_20000.pt` |
| Teacher | `g1_velocity_football/.../2026-08-14_11-44-01_B1_A1R0_.../model_49999.pt` | `g1_velocity_football_klavier_ball_temporal/.../2026-08-27_08-25-54_.../model_47000.pt` |
| Student 起步 | CalibratedVisualDR，冻结 MLP，Teacher rollout | Klavier Teacher47000，从零训练，先冻结 MLP 或直接 constrained |
| Student 过渡 | MountRange VisualDR，`model_4000.pt` | SyncDelay02、MountRange025、latent 0.1 |
| Student 候选 | StrongVisualDR，末层 MLP，mixed30，`model_9999.pt` | Visibility BCE0.2 + 末层 MLP + mixed30，`model_9999.pt`；另有 E1–E4 冻结 MLP 因子实验 |

## 3. Walk 对比

| 项目 | 旧 Walk | 新 Walk |
|---|---:|---:|
| checkpoint | `model_16000.pt` | `model_20000.pt` |
| Actor/Critic MLP | 512 / 256 / 128 | 1024 / 512 / 256 |
| Actor 输入 | 当前本体观测 | 当前本体观测，相同类别 |
| PPO steps/env | 24 | 24，相同 |
| PPO epochs / mini-batches | 5 / 4 | 5 / 4，相同 |
| PPO learning rate | 1e-3 adaptive | 1e-3 adaptive，相同 |
| gamma / lambda / clip | 0.99 / 0.95 / 0.2 | 相同 |
| entropy | 0.01 | 0.005 |
| 对称性损失 | 无 | mirror loss = 1.0 |
| 训练目标 | 旧足球任务的行走初始化 | Klavier replica 行走初始化 |

新版 Teacher 的大 MLP 不是在 Teacher 阶段突然扩大的，而是从新版 Walk
`model_20000.pt` 继承。因此 Student 也必须采用 `1024/512/256` 才能严格载入其控制部分。

## 4. Teacher 对比

### 4.1 网络、观测和训练

| 项目 | 旧 Teacher | 新 Teacher |
|---|---|---|
| 控制 MLP | 512 / 256 / 128 | 1024 / 512 / 256 |
| 球历史 | 10 帧 × 7 维 | 相同 |
| 球历史编码器 | Temporal CNN 64 / 64 / 64 | 相同 |
| CNN 卷积 | kernel 3，causal，dilation 1/2/4，取最后时刻 | 相同 |
| 本体历史 | 5 帧，共 490 维 | 同类输入；网络容量增大 |
| ball feature 延迟 | 每环境随机 0–2 步 | 0–2 步，并显式与 Student depth 延迟同步 |
| PPO | 24 steps，5 epochs，4 minibatches，lr 1e-3 | 相同 |
| entropy | 0.01 | 0.01，相同 |
| 对称性损失 | 无 | mirror loss = 1.0 |
| 初始化 | 旧 Walk `model_16000.pt` | 新 Walk `model_20000.pt` |
| 最终采用 checkpoint | `model_49999.pt` | `model_47000.pt` |

### 4.2 奖励项

除下表最后两项外，旧、新 Teacher 的奖励名称和权重相同。

| 奖励项 | 旧 Teacher | 新 Teacher |
|---|---:|---:|
| is_terminated | -200 | 相同 |
| joint_torques_l2 | -1e-5 | 相同 |
| joint_acc_l2 | -1e-7 | 相同 |
| track_ball_lin_vel_xy_exp | 1.0 | 相同 |
| track_linear_velocity | 1.0 | 相同 |
| track_angular_velocity | 2.0 | 相同 |
| track_ball_relative_vel_xy_exp | 0 | 相同 |
| track_ball_relative_pos_xy_exp | 0 | 相同 |
| ball_outside_control_zone | 0 | 相同 |
| upright | 1.0 | 相同 |
| pose | 1.0 | 相同 |
| body_ang_vel | -0.05 | 相同 |
| angular_momentum | -0.02 | 相同 |
| dof_pos_limits | -1.0 | 相同 |
| action_rate_l2 | -0.2 | 相同 |
| air_time | 0 | 相同 |
| foot_clearance | -2.0 | 相同 |
| foot_swing_height | -0.25 | 相同 |
| foot_slip | -0.1 | 相同 |
| soft_landing | -1e-5 | 相同 |
| self_collisions | -1.0 | 相同 |
| ball_forbidden_contacts | -2.0 | 相同 |
| ball_front_control | 0.5 | 相同 |
| command_velocity_envelope | -1.0 | 删除，等效 0 |
| action_acc_l2 | -0.1 | 删除，等效 0 |

注意：`joint_acc_l2` 和 `action_acc_l2` 不是同一项。新版仍保留极小权重的
`joint_acc_l2=-1e-7`，删除的是 `action_acc_l2=-0.1`。

### 4.3 物理随机化和推力

| 项目 | 旧 Teacher | 新 Teacher |
|---|---|---|
| 脚底摩擦、球摩擦 | 有 | 相同类别 |
| base COM | 有 | 有 |
| encoder bias | 无 | 新增 |
| base mass | 无 | 新增 |
| joint default position | 无 | 新增 |
| joint friction / armature | 无 | 新增 |
| actuator gains | 无 | 新增 |
| 基础 interval push | 有 | 有，初始 x±0.5、y±0.3、z±0.2、roll/pitch±0.1、yaw±0.2 |
| 推力课程 | 较旧配置 | 新版 Student 按 Teacher 保存配置复现，最大 x±1.5、y±1.0、z±0.5、roll/pitch±0.8、yaw±1.57 |

所以“新版 Student 效果差”不能直接归因于推力过大，但新版确实同时面对更强物理 DR。
E1/E2 关闭的是推力课程，不是把 interval push 完全删除；基础推力仍存在。

## 5. Student 对比

### 5.1 共同结构

- 输入均为机器人本体观测 + 深度时序。
- Depth CNN 均为 `16 → 32 → 64`，输出 latent 64。
- 动作维度均为 29。
- 主行为损失均使用 Huber，CNN 学习率均为 `3e-4`。
- Teacher 球历史编码器均为 causal Temporal CNN `64/64/64`。

### 5.2 最终候选配置差异

| 项目 | 旧 constrained Student | 新 visibility constrained Student |
|---|---|---|
| Teacher | old `model_49999.pt` | Klavier `model_47000.pt` |
| 控制 MLP | 512 / 256 / 128 | 1024 / 512 / 256 |
| 训练起点 | 从冻结 MLP 的 `model_4000.pt` 续训 | 从零构建 Student，并载入 Teacher 控制部分 |
| 可训练控制层 | 仅 MLP 最后一层 | 仅 MLP 最后一层 |
| MLP learning rate | 1e-5 | 相同 |
| latent loss | 0.1 | 相同 |
| MLP anchor | 0.001 | 相同 |
| 可见性标签/BCE | 无 | 有，BCE 系数 0.2 |
| rollout | mixed，Student 最终 30% | 相同 |
| rollout ramp | 2000 updates | 相同 |
| depth/Teacher 延迟 | 旧前期 noDelay；最终方案没有新版的显式同步约束 | 同一每环境延迟状态，随机 0–2 步同步 |
| 相机安装随机化 | MountRange/StrongVisualDR，最终 alpha 0.35 | alpha 0–0.25；x±3 cm、z±1 cm、pitch±3°、FOV 40.5–44.5°、crop±1 px |
| 有球/无球监督 | 无独立显式头 | 由深度图几何可见性生成标签，训练 visibility head |

### 5.3 旧版为何较早表现好

旧 `CalibratedVisualDR model_3500.pt` 的保存配置不是纯 Teacher rollout：warmup 1000、
ramp 3000、最终 Student probability 1.0。在 iteration 3500 时 Student rollout 已约
`0.8337`，因此它从早期就在优化“自己动作造成的状态分布”，闭环 sim2sim 更容易看起来好。

旧版后续 `constrained model_9999.pt` 训练末值约为：behavior `0.01066`、latent
`0.08027`、reward `71.47`、episode length `970.19`。这些值来自 30% Student mixed
rollout，仍不能与纯 Teacher rollout 严格横比。

新版 E1 `model_12000.pt` 的代表性日志约为：behavior `0.01287`、latent `0.08910`、
Student rollout `0`、reward `84.96`、episode length `1000`。高 reward 主要说明
Teacher 驱动的采样轨迹稳定，不代表 Student 闭环稳定。冻结 MLP 时，CNN 只能学习把深度
映射到 Teacher 原有 latent；一旦视觉误差导致动作偏离，训练数据不覆盖 Student 自己进入的
错误状态，闭环误差会累积。

因此，新版效果差既可能有训练轮数因素，但核心不是简单“轮数不够”，而是：

1. 大 MLP Teacher 的目标函数更复杂，视觉 latent 更难拟合；
2. 同步 0–2 步延迟、相机随机化和更强物理 DR 提高了任务难度；
3. 纯 Teacher rollout 存在分布偏移；
4. behavior/latent loss 数值接近，并不保证动作闭环误差不累积；
5. 新旧 play 若相机位姿、推力开关或命令不同，肉眼效果不能作为严格对照。

## 6. 新版 E1–E4 因子实验定义

四组均为：Teacher47000、4096 env/GPU、30k iterations、每 1000 保存、控制 MLP 完全
冻结、100% Teacher rollout、Huber + latent 0.1。

| 实验 | 推力课程 | Student mirror loss |
|---|---|---:|
| E1 | 关闭；保留基础 interval push | 0 |
| E2 | 关闭；保留基础 interval push | 1.0 |
| E3 | 开启 | 0 |
| E4 | 开启 | 1.0 |

这四组只回答“推力课程和 Student 对称损失是否改善冻结阶段拟合”，不能替代第二阶段
mixed rollout。选出第一阶段 checkpoint 后，仍需开放 MLP 最后一层 `1e-5`，将 Student
rollout 逐渐提高到 30%，并进行统一闭环评估。

## 7. Checkpoint 身份校验

| Checkpoint | SHA-256 |
|---|---|
| 旧 Walk `model_16000.pt` | `669769b2d4a09dbae1c01a79d689f5d612d83b66e66898605e0baa5a8a1e4174` |
| 旧 Teacher `model_49999.pt` | `b9eff9ad3bbcc043393dd3f1c259a013878f4e85aed2667fb5633290641c5930` |
| 旧 Student `model_9999.pt` | `34fa57c058005826ea1610eeedf1092c58581242cbb79a52ad29acf99f9d97d5` |
| 新 Walk `model_20000.pt` | `a0eb99185d62294eae8ac470a3bde58f2ebe96bc288ab5d6a019a34ef317e391` |
| 新 Teacher `model_47000.pt` | `c666d516f7d6dc81bc6f6b8f27399b9865fe5eb7612413f4ca4ef4270ce53fc0` |
| 新 Student `model_9999.pt` | `c19d3bdcdf59717e669fc5aa029353cde75d18a2682c13168455a521d992009c` |

## 8. 后续公平比较规则

每个候选至少固定以下条件：相同命令序列、相同相机位姿、相同球初始状态、相同基础推力、
相同随机种子和相同评估回合数。分别报告：

- 无推力与有推力的闭环存活率；
- 球失控率、跌倒率、平均连续控球时间；
- 速度跟踪误差和球相对位置误差；
- 可见球、遮挡/丢球、重新见球三个阶段的恢复成功率；
- behavior、latent、visibility loss 仅作为诊断量，不作为最终排名依据。

建议至少比较四个 checkpoint：旧 calibrated `model_3500.pt`、旧 constrained
`model_9999.pt`、新版 visibility constrained `model_9999.pt`、E1–E4 中固定评估最好的
冻结阶段 checkpoint。只有在相同闭环条件下，新旧效果差异才可归因到训练方案。

## 9. 配置证据来源

- 每个 run 下的 `params/agent.yaml` 与 `params/env.yaml`；这是本文件的首要依据。
- `scripts/run_longdropout10_isaac_actor_dr_from_walk_to50k_seed42.sh`
- `scripts/run_football_klavier_scheme_a_longdrop10_from_walk20000_seed42_50k.sh`
- `scripts/run_depth_student_klavier_visibility_from_zero_stage2_10k_seed42.sh`
- `scripts/run_depth_student_frozen_factorial_seed42.sh`
- `scripts/launch_depth_student_factorial_4gpu.sh`
- [旧 Teacher 基线记录](实验记录/DT-TEACHER-BASE-0-20260814-B1-A1R0-LongDropout10教师基线.md)

