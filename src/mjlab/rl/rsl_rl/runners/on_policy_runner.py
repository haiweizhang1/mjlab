# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
import warnings
from collections import deque

from mjlab.rl import rsl_rl
from mjlab.rl.rsl_rl.algorithms import PPO
from mjlab.rl.rsl_rl.env import VecEnv
from mjlab.rl.rsl_rl.modules import ActorCritic, ActorCriticRecurrent, resolve_rnd_config, resolve_symmetry_config
from mjlab.rl.rsl_rl.utils import resolve_obs_groups, store_code_state


class OnPolicyRunner:
    """
    适配 kp/kd 输出的 On-Policy 训练器（基于RSL-RL框架）
    核心扩展：
    1. 支持Actor输出 (actions, kp, kd) 三元组
    2. 将kp/kd传递到环境step中，用于机器人控制器
    3. 新增kp/kd日志监控，实时跟踪输出范围
    4. 保持原框架的多GPU、RND、日志等所有功能
    """

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        # 保存配置参数
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]  # PPO算法配置
        self.policy_cfg = train_cfg["policy"]  # 策略网络配置
        self.device = device  # 训练设备（cpu/cuda）
        self.env = env  # 向量环境

        # 配置多GPU训练（原逻辑，无修改）
        self._configure_multi_gpu()

        # 训练核心参数
        self.num_steps_per_env = self.cfg["num_steps_per_env"]  # 每个环境的采样步数
        self.save_interval = self.cfg["save_interval"]  # 模型保存间隔

        # 从环境获取观测，解析观测分组（policy/critic/rnd）
        obs = self.env.get_observations()
        default_sets = ["critic"]
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], default_sets)

        # 构建PPO算法（包含ActorCritic网络）
        self.alg = self._construct_algorithm(obs)

        # 分布式训练时，仅主进程（rank 0）记录日志
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        # 日志相关初始化
        self.log_dir = log_dir  # 日志保存目录
        self.writer = None  # 日志写入器（tensorboard/wandb/neptune）
        self.tot_timesteps = 0  # 总训练步数
        self.tot_time = 0  # 总训练时间
        self.current_learning_iteration = 0  # 当前训练迭代数
        self.git_status_repos = [rsl_rl.__file__]  # 记录代码版本的仓库路径

        # ===================== 新增：kp/kd 日志缓存 =====================？
        # 用于存储最近100个episode的kp/kd均值，监控输出是否在合理范围
        self.kp_buffer = deque(maxlen=100)
        self.kd_buffer = deque(maxlen=100)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        """
        核心训练函数：执行多轮PPO训练迭代
        Args:
            num_learning_iterations: 训练迭代数
            init_at_random_ep_len: 是否随机初始化episode长度（用于探索）
        """
        # 初始化日志写入器（tensorboard/wandb等）
        self._prepare_logging_writer()

        # 随机初始化episode长度（增加探索性）
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # 获取初始观测并移到训练设备
        obs = self.env.get_observations().to(self.device)
        self.train_mode()  # 切换到训练模式（启用dropout等）

        # ===================== 训练数据缓存初始化 =====================
        ep_infos = []  # 存储episode级别的信息（奖励、长度等）
        rewbuffer = deque(maxlen=100)  # 最近100个episode的奖励缓存
        lenbuffer = deque(maxlen=100)  # 最近100个episode的长度缓存
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)  # 当前累计奖励
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)  # 当前episode长度

        # RND（随机网络蒸馏）相关缓存（如果启用RND）
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)  # 外在奖励缓存
            irewbuffer = deque(maxlen=100)  # 内在奖励缓存
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # 分布式训练时，同步所有进程的参数
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # ===================== 开始训练迭代 =====================
        start_iter = self.current_learning_iteration  # 起始迭代数（断点续训用）
        tot_iter = start_iter + num_learning_iterations  # 总迭代数
        for it in range(start_iter, tot_iter):
            start = time.time()  # 记录迭代开始时间

            # ===================== 1. 环境采样（Rollout） =====================
            with torch.inference_mode():  # 推理模式，禁用梯度计算
                for _ in range(self.num_steps_per_env):
                    actions, kps, kds = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(
                        actions.to(self.env.device),  # 动作移到环境设备
                        kps=kps.to(self.env.device),  # kp移到环境设备
                        kds=kds.to(self.env.device)  # kd移到环境设备
                    )

                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))

                    # -------------------- 新增：记录kp/kd用于日志 --------------------
                    # 找出完成的episode，记录其kp/kd均值
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    if len(new_ids) > 0:
                        # 取完成episode的kp/kd，计算每个env的均值并加入缓存
                        self.kp_buffer.extend(kps[new_ids].mean(dim=-1).cpu().numpy().tolist())
                        self.kd_buffer.extend(kds[new_ids].mean(dim=-1).cpu().numpy().tolist())

                    # 处理环境step，将数据存入PPO的经验缓存
                    self.alg.process_env_step(obs, rewards, dones, extras)

                    # 提取内在奖励（仅RND模式下有效）
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    # -------------------- 训练数据缓存更新 --------------------
                    if self.log_dir is not None:
                        # 收集episode信息（奖励、长度等）
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])

                        # 更新奖励累计值
                        if self.alg.rnd:
                            cur_ereward_sum += rewards  # 外在奖励
                            cur_ireward_sum += intrinsic_rewards  # 内在奖励
                            cur_reward_sum += rewards + intrinsic_rewards  # 总奖励
                        else:
                            cur_reward_sum += rewards  # 普通奖励

                        # 更新当前episode长度
                        cur_episode_length += 1

                        # 重置完成episode的累计值
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        if len(new_ids) > 0:
                            # 普通奖励和长度缓存
                            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                            cur_reward_sum[new_ids] = 0
                            cur_episode_length[new_ids] = 0

                            # RND奖励缓存（如果启用）
                            if self.alg.rnd:
                                erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                                irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                                cur_ereward_sum[new_ids] = 0
                                cur_ireward_sum[new_ids] = 0

                # 记录采样阶段耗时
                stop = time.time()
                collection_time = stop - start
                start = stop

                # 计算回报（GAE/优势函数）
                self.alg.compute_returns(obs)

            # ===================== 2. 策略更新 =====================
            loss_dict = self.alg.update()  # 执行PPO更新，返回损失字典

            # 记录策略更新耗时
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it  # 更新当前迭代数

            # ===================== 3. 日志记录与模型保存 =====================
            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())  # 记录本轮迭代的日志
                # 按间隔保存模型
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # 清空episode信息缓存
            ep_infos.clear()

            # 保存代码版本（仅第一轮迭代执行）
            if it == start_iter and not self.disable_logs:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # 将代码版本文件上传到wandb/neptune（如果启用）
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # 训练结束后保存最终模型
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        """
        日志记录函数：记录训练过程中的关键指标
        Args:
            locs: 当前迭代的局部变量字典（包含loss、rewbuffer等）
            width: 终端日志打印宽度
            pad: 终端日志对齐填充长度
        """
        # 计算本轮采样的总步数（所有环境 × 每环境步数 × GPU数）
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # 更新总步数和总时间
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]  # 本轮迭代总耗时

        # ===================== 1. Episode级别信息日志 =====================
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # 处理标量和0维张量的episode信息
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)  # 计算该指标的均值
                # 写入日志（tensorboard/wandb）
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        # ===================== 2. 策略相关日志 =====================
        mean_std = self.alg.policy.action_std.mean()  # 动作噪声标准差均值
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))  # 每秒采样步数

        # 损失函数日志
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])  # 学习率

        # 动作噪声标准差日志
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # ===================== 新增：kp/kd 均值日志 =====================
        # if len(self.kp_buffer) > 0 and len(self.kd_buffer)

        # ===================== 3. 性能相关日志 =====================
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])  # 每秒总步数
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])  # 采样耗时
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])  # 学习耗时

        # ===================== 4. 训练核心指标日志 =====================
        if len(locs["rewbuffer"]) > 0:
            # RND模式下的内外在奖励日志
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])

            # 普通奖励和episode长度日志
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])

            # 按时间维度记录（wandb不支持非整数x轴，跳过）
            if self.logger_type != "wandb":
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        # ===================== 终端日志格式化输出 =====================
        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            # 损失函数日志
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            # RND奖励日志
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                log_string += (
                    f"""{'Mean extrinsic reward:':>{pad}} {statistics.mean(locs['erewbuffer']):.2f}\n"""
                    f"""{'Mean intrinsic reward:':>{pad}} {statistics.mean(locs['irewbuffer']):.2f}\n"""
                )
            # 普通奖励日志
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            # Episode长度日志
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            # 损失函数日志
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        # 拼接episode信息和kp/kd日志
        log_string += ep_string
        # 总统计信息
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime(
                "%H:%M:%S",
                time.gmtime(
                    self.tot_time / (locs['it'] - locs['start_iter'] + 1)
                    * (locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])
                )
            )}\n"""
        )
        print(log_string)  # 打印到终端

    def save(self, path: str, infos=None):
        """
        保存模型和优化器状态
        Args:
            path: 保存路径
            infos: 额外保存的信息（如训练配置、环境参数等）
        """
        # 构建保存字典
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),  # 策略网络参数
            "optimizer_state_dict": self.alg.optimizer.state_dict(),  # 优化器参数
            "iter": self.current_learning_iteration,  # 当前迭代数
            "infos": infos,  # 额外信息
        }
        # 保存RND网络参数（如果启用）
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()

        # 保存到文件
        torch.save(saved_dict, path)

        # 上传模型到wandb/neptune（如果启用）
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None):
        """
        加载模型和优化器状态（断点续训）
        Args:
            path: 模型路径
            load_optimizer: 是否加载优化器状态
            map_location: 设备映射（如cpu/cuda:0）
        Returns:
            infos: 保存的额外信息
        """
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)

        # 加载策略网络参数
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])

        # 加载RND网络参数（如果启用）
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])

        # 加载优化器参数（如果需要）
        if load_optimizer and resumed_training:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])

        # 恢复当前迭代数
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]

        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        """
        获取推理用策略函数（实机部署时使用）
        Args:
            device: 推理设备（如cpu/cuda）
        Returns:
            inference_fn: 推理函数，输入obs，输出(actions, kp, kd)
        """
        self.eval_mode()  # 切换到评估模式（禁用dropout等）
        if device is not None:
            self.alg.policy.to(device)  # 将模型移到推理设备

        # ===================== 核心修改3：推理函数适配kp/kd =====================
        # 包装act_inference，确保返回(actions, kp, kd)三元组
        def inference_fn(obs):
            actions, kps, kds = self.alg.policy.act_inference(obs)
            return actions, kps, kds

        return inference_fn

    def train_mode(self):
        """切换到训练模式"""
        self.alg.policy.train()  # 策略网络训练模式
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.train()  # RND网络训练模式

    def eval_mode(self):
        """切换到评估模式"""
        self.alg.policy.eval()  # 策略网络评估模式
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.eval()  # RND网络评估模式

    def add_git_repo_to_log(self, repo_file_path):
        """添加需要记录版本的代码仓库路径"""
        self.git_status_repos.append(repo_file_path)

    """
    以下为辅助函数（原逻辑，无核心修改）
    """

    def _configure_multi_gpu(self):
        """配置多GPU分布式训练"""
        # 获取GPU数量和进程rank
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # 单GPU/CPU训练，直接返回
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # 分布式训练，获取local/global rank
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # 构建多GPU配置字典
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,
            "local_rank": self.gpu_local_rank,
            "world_size": self.gpu_world_size,
        }

        # 验证设备配置
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # 初始化分布式进程组
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        torch.cuda.set_device(self.gpu_local_rank)  # 设置当前进程的GPU

    def _construct_algorithm(self, obs) -> PPO:
        """构建PPO算法（包含ActorCritic网络）"""
        # 解析RND配置
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
        # 解析对称配置
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        # 兼容旧版归一化配置（弃用警告）
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        # 初始化ActorCritic网络
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))  # 动态加载网络类（ActorCritic/ActorCriticRecurrent）
        actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # 初始化PPO算法
        alg_class = eval(self.alg_cfg.pop("class_name"))  # 动态加载算法类（PPO）
        alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)

        # 初始化经验缓存
        alg.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )

        return alg

    def _prepare_logging_writer(self):
        """初始化日志写入器（tensorboard/wandb/neptune）"""
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            self.logger_type = self.cfg.get("logger", "tensorboard").lower()

            # 根据配置加载不同的日志写入器
            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter
                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter
                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")