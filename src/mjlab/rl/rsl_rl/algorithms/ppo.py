# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain

from mjlab.rl.rsl_rl.modules import ActorCritic
from mjlab.rl.rsl_rl.modules.rnd import RandomNetworkDistillation
from mjlab.rl.rsl_rl.storage import RolloutStorage
from mjlab.rl.rsl_rl.utils import string_to_callable


class PPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    policy: ActorCritic
    """The actor critic module."""

    def __init__(
        self,
        policy,
        num_learning_epochs=5,
        num_mini_batches=4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        learning_rate=0.001,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="adaptive",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        # 新增：KP/KD相关参数（和动作维度一致）
        num_kp: int = 29,
        num_kd: int = 29,
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # 新增：保存KP/KD维度
        self.num_kp = num_kp
        self.num_kd = num_kd

        # RND components
        if rnd_cfg is not None:
            # Extract parameters used in ppo
            rnd_lr = rnd_cfg.pop("learning_rate", 1e-3)
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=rnd_lr)
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # Print that we are not using symmetry
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            # If function is a string then resolve it to a function
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if symmetry_cfg["use_data_augmentation"] and not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    "Data augmentation enabled but the function is not callable:"
                    f" {symmetry_cfg['data_augmentation_func']}"
                )
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # PPO components
        self.policy = policy
        self.policy.to(self.device)
        # Create optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        # Create rollout storage
        self.storage: RolloutStorage = None  # type: ignore
        # 新增：扩展Transition，支持KP/KD
        self.transition = RolloutStorage.Transition()
        # 为Transition新增KP/KD字段（兼容原有逻辑）
        if not hasattr(self.transition, 'kps'):
            self.transition.kps = None
        if not hasattr(self.transition, 'kds'):
            self.transition.kds = None
        if not hasattr(self.transition, 'kps_log_prob'):
            self.transition.kps_log_prob = None
        if not hasattr(self.transition, 'kds_log_prob'):
            self.transition.kds_log_prob = None

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        # 新增：扩展actions_shape，包含KP/KD维度
        # 假设actions_shape是动作维度，KP/KD维度和动作一致
        # total_policy_shape = (actions_shape[0] + self.num_kp + self.num_kd,) if isinstance(actions_shape, tuple) else actions_shape + self.num_kp + self.num_kd
        # create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,  # 保留原有动作维度（兼容存储）
            self.device,
        )


    def act(self, obs):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # compute the actions and values（修改：接收动作+KP+KD）
        actions, kps, kds = self.policy.act(obs)  # 解包Actor返回的三元组

        # 记录动作相关（原有逻辑）
        self.transition.actions = actions.detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # 新增：记录KP/KD相关
        self.transition.kps = kps.detach() if kps is not None else None
        self.transition.kds = kds.detach() if kds is not None else None
        # 计算并记录KP/KD的对数概率
        self.transition.kps_log_prob = self.policy.get_kp_log_prob(self.transition.kps).detach()
        self.transition.kds_log_prob = self.policy.get_kd_log_prob(self.transition.kds).detach()
        self.transition.kp_mean = self.policy.kp_mean.detach()
        self.transition.kp_sigma = self.policy.kp_std.detach()
        self.transition.kd_mean = self.policy.kd_mean.detach()
        self.transition.kd_sigma = self.policy.kd_std.detach()

        # need to record obs before env.step()
        self.transition.observations = obs
        return self.transition.actions, self.transition.kps, self.transition.kds

    def process_env_step(self, obs, rewards, dones, extras):
        # update the normalizers
        self.policy.update_normalization(obs)#####？
        if self.rnd:
            self.rnd.update_normalization(obs)

        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Compute the intrinsic rewards
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # record the transition（修改：包含KP/KD）
        self.storage.add_transitions(self.transition)

        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs):
        # compute value for the last step
        last_values = self.policy.evaluate(obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # -- RND loss
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        # generator for mini batches
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            # 扩展mini batch生成器，包含KP/KD
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches,
                self.num_learning_epochs,
                # 新增：指定需要返回的KP/KD字段

            )

        # iterate over batches
        for (obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch,old_actions_log_prob_batch, old_mu_batch, old_sigma_batch,kps_batch, kds_batch, old_kps_log_prob_batch, old_kds_log_prob_batch, old_kp_mu_batch, old_kp_sigma_batch, old_kd_mu_batch, old_kd_sigma_batch, hid_states_batch,masks_batch,) in generator:

            # number of augmentations per sample
            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1
            # original batch size
            # we assume policy group is always there and needs augmentation
            original_batch_size = obs_batch.batch_size[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                num_aug = int(obs_batch.batch_size[0] / original_batch_size) if hasattr(obs_batch, 'batch_size') else int(obs_batch.shape[0] / original_batch_size)
                # repeat the rest of the batch
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)
                # 新增：重复KP/KD批次
                kps_batch = kps_batch.repeat(num_aug, 1)
                kds_batch = kds_batch.repeat(num_aug, 1)
                old_kps_log_prob_batch = old_kps_log_prob_batch.repeat(num_aug, 1)
                old_kds_log_prob_batch = old_kds_log_prob_batch.repeat(num_aug, 1)
            # Recompute actions log prob and entropy for current batch
            # -- actor: 重新计算动作+KP+KD的分布
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0] if hid_states_batch else None)

            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            kps_log_prob_batch = self.policy.get_kp_log_prob(kps_batch)
            kds_log_prob_batch = self.policy.get_kd_log_prob(kds_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1] if hid_states_batch else None)
            # -- entropy（已包含动作+KP+KD的总熵）

            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            kp_mu_batch = self.policy.kp_mean
            kp_sigma_batch = self.policy.kp_std
            kd_mu_batch = self.policy.kd_mean
            kd_sigma_batch = self.policy.kd_std



            entropy_batch = self.policy.entropy[:original_batch_size]
            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl_a = torch.sum(torch.log(sigma_batch / old_sigma_batch + 1e-5) +
                                     (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) /
                                     (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    # KP/KD KL (逻辑一致)
                    kl_kp = torch.sum(torch.log(kp_sigma_batch / old_kp_sigma_batch + 1e-5) +
                                      (torch.square(old_kp_sigma_batch) + torch.square(old_kp_mu_batch - kp_mu_batch)) /
                                      (2.0 * torch.square(kp_sigma_batch)) - 0.5, axis=-1)
                    kl_kd = torch.sum(torch.log(kd_sigma_batch / old_kd_sigma_batch + 1e-5) +
                                      (torch.square(old_kd_sigma_batch) + torch.square(old_kd_mu_batch - kd_mu_batch)) /
                                      (2.0 * torch.square(kd_sigma_batch)) - 0.5, axis=-1)

                    # 总 KL 均值 (0.5/0.25/0.25 权重分配)
                    kl_mean = torch.mean(0.5 * kl_a + 0.25 * kl_kp + 0.25 * kl_kd)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate
                    # Perform this adaptation only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss（修改：使用总对数概率）
            # 原有：仅动作的比率
            # ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            # 新增：总对数概率的比率（动作+KP+KD）
            # old_total_log_prob_batch = 0.5*old_actions_log_prob_batch
            # old_total_log_prob_batch += 0.25*old_kps_log_prob_batch
            # old_total_log_prob_batch += 0.25*old_kds_log_prob_batch

            ratio_actions = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            ratio_kps = torch.exp(kps_log_prob_batch - torch.squeeze(old_kps_log_prob_batch))
            ratio_kds = torch.exp(kds_log_prob_batch - torch.squeeze(old_kds_log_prob_batch))
            # ratio = torch.exp(total_log_prob_batch - torch.squeeze(old_total_log_prob_batch))
            surrogate_action = -torch.squeeze(advantages_batch) * ratio_actions
            surrogate_clipped_action = -torch.squeeze(advantages_batch) * torch.clamp( ratio_actions, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss_a = torch.max(surrogate_action, surrogate_clipped_action).mean()

            surrogate_kp = -torch.squeeze(advantages_batch) * ratio_kps
            surrogate_clipped_kp = -torch.squeeze(advantages_batch) * torch.clamp(ratio_kps,  1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss_kp = torch.max(surrogate_kp, surrogate_clipped_kp).mean()

            surrogate_kd = -torch.squeeze(advantages_batch) * ratio_kds
            surrogate_clipped_kd = -torch.squeeze(advantages_batch) * torch.clamp(ratio_kds,1.0 - self.clip_param,1.0 + self.clip_param)
            surrogate_loss_kd = torch.max(surrogate_kd, surrogate_clipped_kd).mean()

            surrogate_loss =  0.5* surrogate_loss_a +0.25*surrogate_loss_kp +0.25*surrogate_loss_kd

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # 总损失（包含动作+KP+KD的熵）
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Symmetry loss
            if self.symmetry:
                # obtain the symmetric actions
                # if we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    # compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # 注意：act_inference现在返回 (action_mean, kp_mean, kd_mean)
                mean_actions_batch, mean_kps_batch, mean_kds_batch = self.policy.act_inference(obs_batch.detach().clone())

                # 仅对动作计算对称损失（KP/KD无需对称，物理意义不支持）
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                # compute the loss (we skip the first augmentation as it is the original one)
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # Random Network Distillation loss（原有逻辑）
            if self.rnd:
                # extract the rnd_state
                # TODO: Check if we still need torch no grad. It is just an affine transformation.
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                # predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients
            # -- For PPO
            self.optimizer.zero_grad()
            loss.backward()
            # -- For RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients
            # -- For PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # -- Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        # -- For RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # -- Clear the storage
        self.storage.clear()

        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel
