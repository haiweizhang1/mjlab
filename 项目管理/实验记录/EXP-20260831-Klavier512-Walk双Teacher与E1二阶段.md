# Klavier512 Walk、双 Teacher 与 E1 二阶段（2026-08-31）

## 目标与固定原则

本轮新增两条互不混用的训练链：

1. 从零训练 `512/256/128` Walk 到 20k，再分别初始化两个
   `512/256/128` 足球 Teacher。
2. 从现有 E1 Depth Student `model_27000.pt` 恢复，进入仅开放控制 MLP
   最后一层的二阶段，并额外训练 30k。

除本文列出的变量外，继续使用当前 Klavier G1 XML、奖励、物理域随机化、历史长度、
BallCNN 和深度视觉配置。

## Walk 20k

- Task：`Mjlab-Velocity-Walk-KlavierReplica-Legacy512-LegacyPush-Flat-Unitree-G1`
- Actor/Critic MLP：`512/256/128`
- 对称损失：关闭
- 推力课程：关闭
- 固定旧版推力：每 `5--6 s` 一次；`x±0.5`、`y±0.3`、`z±0.2`、
  `roll/pitch±0.1`、`yaw±0.2`
- 训练：`20_001` iterations，每 1000 保存，目标 checkpoint 为
  `model_20000.pt`

启动：

```bash
bash scripts/run_klavier_legacy512_walk_seed42_20k.sh
```

## 双足球 Teacher

共同设置：

- MLP：`512/256/128`
- 从上述 Walk `model_20000.pt` 转移初始化
- 无对称损失
- 无推力课程；保留与 Walk 相同的固定旧版推力
- 无球观测延迟、长丢失、固定偏置或额外独立噪声
- 训练 50k，每 1000 保存

唯一变量：

| 版本 | Task | 训练时球位置噪声 |
|---|---|---|
| Noise0 | `Mjlab-Velocity-Football-KlavierReplica-Legacy512-NoPushCurr-BallNoise0-BallTemporal-Flat-Unitree-G1` | 0 |
| Noise5cm | `Mjlab-Velocity-Football-KlavierReplica-Legacy512-NoPushCurr-BallNoise5cm-BallTemporal-Flat-Unitree-G1` | XY 每控制步共享均匀噪声 `[-0.05, 0.05] m` |

5 cm 噪声同时作用于球 XY、球到左脚和球到右脚的向量，因此三者保持几何一致。
播放/评估配置关闭该噪声。

Walk 训练完成后启动：

```bash
bash scripts/run_klavier_legacy512_teacher_seed42_50k.sh noise0
bash scripts/run_klavier_legacy512_teacher_seed42_50k.sh noise5cm
```

脚本默认自动选择最新的 Walk `model_20000.pt`；也可显式指定：

```bash
WALK_CHECKPOINT=/absolute/path/model_20000.pt \
  bash scripts/run_klavier_legacy512_teacher_seed42_50k.sh noise0
```

## E1 二阶段额外 30k

- 源实验：`2026-08-30_03-42-30_E1_PushCurrOff_FrozenMLP_NoSym_Mixed030_Teacher47000_seed42_30000iter_wandb`
- 默认源 checkpoint：`model_27000.pt`
- 环境：E1 PushCurrOff，保持无对称损失
- Rollout：Mixed，Student 比例上限 30%
- CNN 学习率：`3e-4`
- 控制 MLP：仅最后一层开放，学习率 `1e-5`
- 损失：Huber + `0.1 latent` + `0.001 MLP anchor`
- 续训长度：额外 30k，每 1000 保存；从 27k 恢复时最终强制保存的
  checkpoint 预计为 `model_56999.pt`

服务器启动：

```bash
bash scripts/run_depth_student_e1_stage2_add30k_seed42.sh
```

服务器日志目录若与默认值不同，可覆盖源实验：

```bash
SOURCE_RUN=<E1目录名> SOURCE_CHECKPOINT=model_27000.pt \
  bash scripts/run_depth_student_e1_stage2_add30k_seed42.sh
```

注意：该脚本使用恢复训练，不能改成 `--pretrained-checkpoint`，否则会丢失 E1
Student、Teacher 和已训练迭代状态的完整继承关系。

## Noise0 Teacher 50k续训与Legacy512一阶段蒸馏

从 Noise0 Teacher `model_50000.pt` 分出两条可并行运行的链路：

1. PPO使用 `resume` 保留优化器与课程状态，再训练30k，最终强制保存预计为
   `model_79999.pt`。
2. 固定同一个 `model_50000.pt`，从0训练Depth Student第一阶段10k。

Legacy512第一阶段严格匹配Teacher：MLP `512/256/128`、Noise0、无球观测延迟、
无推力课程、固定旧推力、无对称损失。Teacher rollout为100%，控制MLP完全冻结，
只训练深度CNN；损失为Huber动作损失加`0.1` latent loss。

```bash
bash scripts/resume_klavier_legacy512_noise0_teacher50k_to80k.sh
bash scripts/run_depth_student_legacy512_noise0_stage1_10k.sh
```

## 旧版标准G1 XML Walk单变量对照

Task：`Mjlab-Velocity-Walk-LegacyXML-Legacy512-LegacyPush-Flat-Unitree-G1`。
它相对Klavier Legacy512 Walk只将机器人模型从`get_klavier_spec`替换为旧版标准
`get_spec`；MLP `512/256/128`、固定旧推力、无推力课程、无对称损失、seed42、
20k训练和每1000保存均保持相同。

```bash
bash scripts/run_legacy_xml_legacy512_walk_seed42_20k.sh
```
