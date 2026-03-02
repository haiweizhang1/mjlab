# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal  # 导入正态分布，用于动作采样

from rsl_rl.networks import MLP, EmpiricalNormalization  # 自定义MLP和经验归一化层

class ActorModule(nn.Module):
    def __init__(self,num_actor_obs,num_actions ):
        super().__init__()
        actor_hidden_dims=[256, 256, 256]
        activation ="elu"
        self.backbone = MLP(num_actor_obs, actor_hidden_dims[-1], actor_hidden_dims, activation)
        # self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        last_dim = actor_hidden_dims[-1]
        self.action_head = nn.Linear(last_dim, num_actions)
        self.kp_head = nn.Linear(last_dim, num_actions)
        self.kd_head = nn.Linear(last_dim, num_actions)
        # 初始权重微调（建议）：让 KP/KD 初始输出相对稳定
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.kp_head.weight)
        nn.init.zeros_(self.kd_head.weight)


    def forward(self, x):
        features = self.backbone(x)
        actions = self.action_head(features)
        kps = self.kp_head(features)
        kds = self.kd_head(features)
        # 注意：导出 ONNX 时通常需要返回单个 Tensor
        # 建议合并输出：[Batch, 3 * num_actions]
        return torch.cat([actions, kps, kds], dim=-1)


class ActorCritic(nn.Module):
    """
    扩展版Actor-Critic网络（RSL-RL适配）
    核心修改：Actor除输出动作外，额外输出kp（比例增益）和kd（微分增益）
    特性：
    1. 仅对动作添加探索噪声，kp/kd为确定性输出
    2. 保持原有的观测分组、归一化、状态依赖/独立标准差逻辑
    3. 输出格式：(actions, kp, kd) 三元组
    """
    # 标记是否为循环网络（MLP结构，非循环）
    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        # actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        state_dependent_std=False,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        # 核心参数保存（新增）
        self.num_actions = num_actions
        self.num_kp = num_actions
        self.num_kd = num_actions
        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        self.state_dependent_std = state_dependent_std
        self.noise_std_type = noise_std_type

        # =========================================================
        # 1. Actor 重构：共享特征提取 + 三个独立输出头
        # =========================================================
        # 共享 Backbone：提取环境通用的特征
        # 最后一层维度设为 actor_hidden_dims[-1]
        # self.actor_backbone = MLP(num_actor_obs, actor_hidden_dims[-1], actor_hidden_dims[:-1], activation)
        # last_dim = actor_hidden_dims[-1]
        # self.action_head = nn.Linear(last_dim, num_actions)
        # self.kp_head = nn.Linear(last_dim, num_actions)
        # self.kd_head = nn.Linear(last_dim, num_actions)
        #
        # # 初始权重微调（建议）：让 KP/KD 初始输出相对稳定
        # nn.init.zeros_(self.action_head.weight)
        # nn.init.zeros_(self.kp_head.weight)
        # nn.init.zeros_(self.kd_head.weight)

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        # self.actor = ActorModule(
        #     self.actor_backbone,
        # self.action_head,
        # self.kp_head,
        # self.kd_head,
        # )
        self.actor = ActorModule(num_actor_obs,num_actions)
        print(f"Actor MLP: {self.actor}")
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()
        print(f"Critic MLP: {self.critic}")

        # 2. 标准差逻辑（适配三个分支）
        if self.state_dependent_std:
            # 状态依赖：为每个分支定义独立的 Std Head
            pass
            # self.action_std_head = nn.Linear(last_dim, num_actions)
            # self.kp_std_head = nn.Linear(last_dim, num_actions)
            # self.kd_std_head = nn.Linear(last_dim, num_actions)
        else:
            # 状态独立：参数维度依然是 3 * num_actions
            total_dim = num_actions * 3
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(total_dim))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(total_dim) + 1e-7))
        self.action_distribution = None
        self.kp_distribution = None
        self.kd_distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        pass

    # def forward(self):
    #     raise NotImplementedError

    def forward(self, x):
        # 导出工具会调用这个方法
        return self.actor(x)
    # 新增：KP/KD的均值/标准差属性
    @property
    def kp_mean(self):
        return self.kp_distribution.mean

    @property
    def kp_std(self):
        return self.kp_distribution.stddev

    @property
    def kd_mean(self):
        return self.kd_distribution.mean

    @property
    def kd_std(self):
        return self.kd_distribution.stddev

    # 原有动作属性保留
    @property
    def action_mean(self):
        return self.action_distribution.mean

    @property
    def action_std(self):
        return self.action_distribution.stddev

    # 扩展熵计算：包含动作+KP+KD的总熵（新增）
    @property
    def entropy(self):
        action_ent = self.action_distribution.entropy().sum(dim=-1) if self.action_distribution is not None else 0
        kp_ent = self.kp_distribution.entropy().sum(dim=-1) if self.kp_distribution is not None else 0
        kd_ent = self.kd_distribution.entropy().sum(dim=-1) if self.kd_distribution is not None else 0
        return 0.5*action_ent + 0.25*kp_ent + 0.25*kd_ent

    def update_distribution(self, obs):
        """核心逻辑修改：从不同头提取数据并构建分布"""
        # 前向传播经过共享层
        features = self.actor.backbone(obs)
        # 1. 提取均值 (Mean)
        action_mean = self.actor.action_head(features)
        kp_mean = self.actor.kp_head(features)
        kd_mean = self.actor.kd_head(features)

        # 2. 提取标准差 (Std)
        if self.state_dependent_std:
            a_std_raw = self.action_std_head(features)
            kp_std_raw = self.kp_std_head(features)
            kd_std_raw = self.kd_std_head(features)

            if self.noise_std_type == "log":
                a_std = torch.exp(torch.clamp(a_std_raw, min=-5.0, max=0.5))
                kp_std = torch.exp(torch.clamp(kp_std_raw, min=-5.0, max=0.5))
                kd_std = torch.exp(torch.clamp(kd_std_raw, min=-5.0, max=0.5))
            else:
                a_std = torch.clamp(a_std_raw, min=0.01, max=2.0)
                kp_std = torch.clamp(kp_std_raw, min=0.01, max=2.0)
                kd_std = torch.clamp(kd_std_raw, min=0.01, max=2.0)
        else:
            # 状态独立标准差切片
            if self.noise_std_type == "scalar":
                std_all = self.std
            elif self.noise_std_type == "log":
                std_all = torch.exp(torch.clamp(self.log_std, min=-5.0, max=0.5))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
            a_std = std_all[:self.num_actions]
            kp_std = std_all[self.num_actions : 2*self.num_actions]
            kd_std = std_all[2*self.num_actions :]

        # 3. 防御性检查（针对 NaN）
        if not torch.isfinite(action_mean).all():
            print("Warning: Action Mean contains NaN!")

        # 4. 构建分布
        self.action_distribution = Normal(action_mean, a_std + 1e-6)
        self.kp_distribution = Normal(kp_mean, kp_std + 1e-6)
        self.kd_distribution = Normal(kd_mean, kd_std + 1e-6)

    def act(self, obs, **kwargs):
        """修改：返回 动作+KP+KD 的采样值"""
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self.update_distribution(obs)

        actions = self.action_distribution.sample()
        kp = self.kp_distribution.sample()
        kd = self.kd_distribution.sample()

        # 执行物理安全截断
        return actions, kp,kd,

    def act_inference(self, obs):
        """修改：推理模式（无噪声），返回动作+KP+KD的均值"""
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        features = self.actor.backbone(obs)

        action_mean = torch.tanh(self.actor.action_head(features))
        kp_mean = torch.clamp(self.actor.kp_head(features), -1.0, 1.0)
        kd_mean = torch.clamp(self.actor.kd_head(features), -0.5, 0.5)

        return action_mean, kp_mean, kd_mean

    def evaluate(self, obs, **kwargs):
        """保持不变：Critic评估价值"""
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return self.critic(obs)

    def get_actor_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions):
        """原有：仅计算动作的对数概率"""
        return self.action_distribution.log_prob(actions).sum(dim=-1)

    def get_kp_log_prob(self, kp):
        """新增：计算KP的对数概率"""
        return self.kp_distribution.log_prob(kp).sum(dim=-1)

    def get_kd_log_prob(self, kd):
        """新增：计算KD的对数概率"""
        return self.kd_distribution.log_prob(kd).sum(dim=-1)

    def get_total_log_prob(self, actions, kp, kd):
        """新增：计算动作+KP+KD的总对数概率（用于策略梯度）"""
        action_log_prob = self.get_actions_log_prob(actions)
        kp_log_prob = self.get_kp_log_prob(kp)
        kd_log_prob = self.get_kd_log_prob(kd)
        return action_log_prob + kp_log_prob + kd_log_prob

    def update_normalization(self, obs):
        """保持不变：更新归一化器"""
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict, strict=True):
        """保持不变：加载参数"""
        super().load_state_dict(state_dict, strict=strict)
        return True  # training resumes