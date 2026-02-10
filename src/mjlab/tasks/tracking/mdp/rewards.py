from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_error_magnitude

from .commands import MotionCommand
from typing import Dict

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _get_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_pos_relative_w[:, body_indexes]
      - command.robot_body_pos_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = (
    quat_error_magnitude(
      command.body_quat_relative_w[:, body_indexes],
      command.robot_body_quat_w[:, body_indexes],
    )
    ** 2
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_lin_vel_w[:, body_indexes]
      - command.robot_body_lin_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_ang_vel_w[:, body_indexes]
      - command.robot_body_ang_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def self_collision_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Cost that returns the number of self-collisions detected by a sensor."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return sensor.data.found.squeeze(-1)




def zmp(env: ManagerBasedRlEnv, force_threshold: float = 5.0) -> Dict[str, torch.Tensor]:
  # --- 1. 资源提取 (假设数据已在正确的 device 上) ---
  sensor = env.scene["feet_contact"]
  data = sensor.data
  robot = env.scene["robot"]
  device = env.device
  B = env.num_envs
  # 直接使用原始引用，减少 .to(device) 的额外开销（Isaac Lab 默认在 GPU）
  pos_w = data.pos[..., :2]  # [B, 14, 2]
  forces_3d = data.force  # [B, 14, 3]
  torques_3d = data.torque  # [B, 14, 3]
  com_pos_w = robot.data.root_com_pos_w[:, :2]
  # --- 2. 动态接触掩码 ---
  fz = forces_3d[..., 2]
  contact_mask = (fz > force_threshold).float()
  total_fz = torch.sum(fz * contact_mask, dim=1, keepdim=True)
  total_fz_safe = total_fz + 1e-6  # 提升稳定性
  is_contact = total_fz.squeeze(-1) > 1e-4
  # --- 3. 高效 ZMP 计算 (手写叉乘替代 torch.cross) ---
  # ZMP_x = Σ(pos_x * fz - (pos_z - pz_avg) * fx - torque_y) / Σfz
  # 这里为了极致简洁，我们直接利用你之前的力矩平衡公式
  f_v = forces_3d * contact_mask.unsqueeze(-1)
  t_v = torques_3d * contact_mask.unsqueeze(-1)
  # 手写 2D 叉乘：(r.x * f.y - r.y * f.x) 对应力矩的 Z 分量
  # 但 ZMP 需要的是水平力矩平衡，我们直接计算 num_x, num_y
  # 注意：pos_z 相对偏移通常较小，但在地形不平时很重要
  pz_rel = data.pos[..., 2] - torch.mean(data.pos[..., 2], dim=1, keepdim=True)
  num_x = torch.sum((pos_w[..., 0] * fz - pz_rel * f_v[..., 0] - t_v[..., 1]) * contact_mask, dim=1)
  num_y = torch.sum((pos_w[..., 1] * fz - pz_rel * f_v[..., 1] + t_v[..., 0]) * contact_mask, dim=1)
  zmp_xy = torch.stack([num_x, num_y], dim=-1) / total_fz_safe
  zmp_xy = torch.where(is_contact.unsqueeze(-1), zmp_xy, com_pos_w)
  # --- 4. 无点序暴力凸包 (极致优化广播) ---
  p1 = pos_w.unsqueeze(2)  # [B, 14, 1, 2]
  p2 = pos_w.unsqueeze(1)  # [B, 1, 14, 2]
  edge_vec = p2 - p1  # [B, 14, 14, 2]
  # 边掩码：(i!=j) & active_i & active_j
  edge_active = contact_mask.unsqueeze(2) * contact_mask.unsqueeze(1)
  edge_active.diagonal(dim1=1, dim2=2).fill_(0)
  # 4.1 批量叉乘判定 Hull Edge
  # 计算 (p2-p1) x (pk-p1)
  # 使用广播直接生成 [B, 14, 14, 14]
  rel_pk = pos_w.view(B, 1, 1, 14, 2) - p1.unsqueeze(3)
  # 手写 2D 叉乘替代 torch.cross 以提速 3-5 倍
  cross_all = edge_vec[..., 0:1] * rel_pk[..., 1] - edge_vec[..., 1:2] * rel_pk[..., 0]
  cross_all = cross_all * contact_mask.view(B, 1, 1, 14)
  # 4.2 判定边界
  is_hull_edge = ((torch.all(cross_all >= -1e-5, dim=3) |
                   torch.all(cross_all <= 1e-5, dim=3)) & (edge_active > 0.5))
  # 4.3 计算符号距离
  zmp_rel = zmp_xy.view(B, 1, 1, 2) - p1
  zmp_cross = edge_vec[..., 0] * zmp_rel[..., 1] - edge_vec[..., 1] * zmp_rel[..., 0]
  # 统一符号：正代表凸包内，负代表凸包外
  hull_sign = torch.sign(torch.sum(cross_all, dim=3))
  dist = (zmp_cross / (torch.norm(edge_vec, dim=-1) + 1e-8)) * hull_sign
  # 4.4 提取最小余量
  valid_dist = torch.where(is_hull_edge, dist, torch.tensor(1e6, device=device))
  stability_margin, _ = torch.min(valid_dist.view(B, -1), dim=1)
  # --- 5. 状态融合与兜底 ---
  active_num = torch.sum(contact_mask, dim=1)
  has_hull = active_num >= 3
  stability_margin = torch.where(has_hull, stability_margin,
                                 torch.where(is_contact, torch.tensor(-0.05, device=device),
                                             torch.tensor(-0.1, device=device)))

  return {
    "zmp_xy": zmp_xy,
    "stability_margin": stability_margin,
    "is_contact": is_contact,
    "active_l": torch.any(contact_mask[:, :7] > 0.5, dim=1),
    "active_r": torch.any(contact_mask[:, 7:] > 0.5, dim=1),
    "total_fz": total_fz.squeeze(-1)
  }


def zmp_reward(env: ManagerBasedRlEnv, threshold: float = 0.02) -> torch.Tensor:
  """
  基于支撑余量的稳定性奖励。
  :param threshold: 容忍边界值 (m)，ZMP 离边缘小于此值时开始惩罚。
  """
  metrics = zmp(env)
  margin = metrics["stability_margin"]  # [B]
  # 方案：指数奖励（ZMP 越靠近中心奖励越高，出界后变为负惩罚）
  # 当 margin > 0 (在内): reward 为 [0.5, 1.0]
  # 当 margin < 0 (在外): reward 迅速变为负数
  # 平滑的指数映射
  reward_inside = torch.exp(margin * 10.0)  # margin=0.05(中心)时, 奖励较大
  # 如果出界，给予固定的阶梯惩罚加上距离相关的线性惩罚
  penalty_outside = -1.0 + margin * 5.0  # margin 为负，所以这是在加大惩罚
  reward = torch.where(margin > 0, reward_inside, penalty_outside)
  # 过滤掉完全没接触的情况（腾空期不计入此奖励，或给个中性值）
  return torch.where(metrics["is_contact"], reward, torch.zeros_like(reward))