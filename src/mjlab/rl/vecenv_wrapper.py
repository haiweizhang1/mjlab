import torch
from mjlab.rl.rsl_rl.env import VecEnv
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.utils.spaces import Space


class RslRlVecEnvWrapper(VecEnv):
  def __init__(
    self,
    env: ManagerBasedRlEnv,
    clip_actions: float | None = None,
  ):
    self.env = env
    self.clip_actions = clip_actions

    self.num_envs = self.unwrapped.num_envs
    self.device = torch.device(self.unwrapped.device)
    self.max_episode_length = self.unwrapped.max_episode_length
    self.num_actions = self.unwrapped.action_manager.total_action_dim
    self._modify_action_space()

    # Reset at the start since rsl_rl does not call reset.
    self.env.reset()

  @property
  def cfg(self) -> ManagerBasedRlEnvCfg:
    return self.unwrapped.cfg

  @property
  def render_mode(self) -> str | None:
    return self.env.render_mode

  @property
  def observation_space(self) -> Space:
    return self.env.observation_space

  @property
  def action_space(self) -> Space:
    return self.env.action_space

  @classmethod
  def class_name(cls) -> str:
    return cls.__name__

  @property
  def unwrapped(self) -> ManagerBasedRlEnv:
    return self.env

  # Properties.

  @property
  def episode_length_buf(self) -> torch.Tensor:
    return self.unwrapped.episode_length_buf

  @episode_length_buf.setter
  def episode_length_buf(self, value: torch.Tensor) -> None:  # pyright: ignore[reportIncompatibleVariableOverride]
    self.unwrapped.episode_length_buf = value

  def seed(self, seed: int = -1) -> int:
    return self.unwrapped.seed(seed)

  def get_observations(self) -> TensorDict:
    obs_dict = self.unwrapped.observation_manager.compute()
    return TensorDict(obs_dict, batch_size=[self.num_envs])

  def reset(self) -> tuple[TensorDict, dict]:
    obs_dict, extras = self.env.reset()
    return TensorDict(obs_dict, batch_size=[self.num_envs]), extras
####################原
  # def step(
  #   self, actions: torch.Tensor
  # ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
  #   if self.clip_actions is not None:
  #     actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
  #   obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
  #   term_or_trunc = terminated | truncated
  #   assert isinstance(rew, torch.Tensor)
  #   assert isinstance(term_or_trunc, torch.Tensor)
  #   dones = term_or_trunc.to(dtype=torch.long)
  #   if not self.cfg.is_finite_horizon:
  #     extras["time_outs"] = truncated
  #   return (
  #     TensorDict(obs_dict, batch_size=[self.num_envs]),
  #     rew,
  #     dones,
  #     extras,
  #   )
  def step(
        self, actions: torch.Tensor, kps: torch.Tensor = None, kds: torch.Tensor = None
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    if self.clip_actions is not None:
      actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)####cfg   kp kd clip
      # ===================== 2. 设备对齐：确保张量与环境设备一致 =====================
    actions = actions.to(self.device)
    # step_kwargs = {}
    if kps is not None:
      kps = kps.to(self.device)
    if kds is not None:
      kds = kds.to(self.device)
    # ===================== 3. 调用底层环境step：执行动作并获取反馈 =====================
    # 底层env.step返回格式：(obs_dict, rew, terminated, truncated, extras)
    ####################################################################
    # 从环境中获取机器人实体
    robot = self.env.scene.entities["robot"]

    ##################################
    # 获取所有 PD 执行器（顺序和之前一致：5020/7520_14/7520_22/4010/WAIST/ANKLE）
    actuator0 = robot.actuators[0]
    actuator0.set_gains(env_ids=slice(None), kp=kps[:,0:10], kd=kds[:,0:10])
    actuator1 = robot.actuators[1]
    actuator1.set_gains(env_ids=slice(None), kp=kps[:,10:15], kd=kds[:,10:15])
    actuator2 = robot.actuators[2]
    actuator2.set_gains(env_ids=slice(None), kp=kps[:,15:19], kd=kds[:,15:19])
    actuator3 = robot.actuators[3]
    actuator3.set_gains(env_ids=slice(None), kp=kps[:,19:23], kd=kds[:,19:23])
    actuator4 = robot.actuators[4]
    actuator4.set_gains(env_ids=slice(None), kp=kps[:, 23:25], kd=kds[:, 23:25])
    actuator5 = robot.actuators[5]
    actuator5.set_gains(env_ids=slice(None), kp=kps[:, 25:29], kd=kds[:, 25:29])


    obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
    # ===================== 4. 结束标志处理：合并terminated和truncated =====================
    # terminated: 任务完成/失败（如机器人摔倒）；truncated: 超时（episode达到最大长度）
    term_or_trunc = terminated | truncated
    # 类型断言：确保后续计算的张量类型正确
    assert isinstance(rew, torch.Tensor), "Reward must be a torch.Tensor (shape: [num_envs,])"
    assert isinstance(term_or_trunc, torch.Tensor), "Termination flag must be a torch.Tensor (shape: [num_envs,])"
    # 转换为long类型（RSL-RL框架标准格式）
    dones = term_or_trunc.to(dtype=torch.long)
    # ===================== 5. 补充超时信息：仅无限视界配置下 =====================
    if not self.cfg.is_finite_horizon:
      extras["time_outs"] = truncated
    # ===================== 6. 封装观测：转换为TensorDict（框架标准格式） =====================
    observations = TensorDict(obs_dict, batch_size=[self.num_envs])
    # ===================== 7. 返回标准化结果 =====================
    return observations, rew, dones, extras

  def close(self) -> None:
    return self.env.close()

  # Private methods.

  def _modify_action_space(self) -> None:
    if self.clip_actions is None:
      return

    from mjlab.utils.spaces import Box, batch_space

    self.unwrapped.single_action_space = Box(
      shape=(self.num_actions,), low=-self.clip_actions, high=self.clip_actions
    )
    self.unwrapped.action_space = batch_space(
      self.unwrapped.single_action_space, self.num_envs
    )
