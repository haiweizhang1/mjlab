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




# def zmp(env: ManagerBasedRlEnv, force_threshold: float = 5.0) -> Dict[str, torch.Tensor]:
#   # --- 1. 资源提取 (假设数据已在正确的 device 上) ---
#   sensor = env.scene["feet_contact"]
#   data = sensor.data
#   robot = env.scene["robot"]
#   device = env.device
#   B = env.num_envs
#   # 直接使用原始引用，减少 .to(device) 的额外开销（Isaac Lab 默认在 GPU）
#   pos_w = data.pos[..., :2]  # [B, 14, 2]
#   forces_3d = data.force  # [B, 14, 3]
#   torques_3d = data.torque  # [B, 14, 3]
#   com_pos_w = robot.data.root_com_pos_w[:, :2]
#   # --- 2. 动态接触掩码 ---
#   fz = forces_3d[..., 2]
#   contact_mask = (fz > force_threshold).float()
#   total_fz = torch.sum(fz * contact_mask, dim=1, keepdim=True)
#   total_fz_safe = total_fz + 1e-6  # 提升稳定性
#   is_contact = total_fz.squeeze(-1) > 1e-4
#   # --- 3. 高效 ZMP 计算 (手写叉乘替代 torch.cross) ---
#   # ZMP_x = Σ(pos_x * fz - (pos_z - pz_avg) * fx - torque_y) / Σfz
#   # 这里为了极致简洁，我们直接利用你之前的力矩平衡公式
#   f_v = forces_3d * contact_mask.unsqueeze(-1)
#   t_v = torques_3d * contact_mask.unsqueeze(-1)
#   # 手写 2D 叉乘：(r.x * f.y - r.y * f.x) 对应力矩的 Z 分量
#   # 但 ZMP 需要的是水平力矩平衡，我们直接计算 num_x, num_y
#   # 注意：pos_z 相对偏移通常较小，但在地形不平时很重要
#   pz_rel = data.pos[..., 2] - torch.mean(data.pos[..., 2], dim=1, keepdim=True)
#   num_x = torch.sum((pos_w[..., 0] * fz - pz_rel * f_v[..., 0] - t_v[..., 1]) * contact_mask, dim=1)
#   num_y = torch.sum((pos_w[..., 1] * fz - pz_rel * f_v[..., 1] + t_v[..., 0]) * contact_mask, dim=1)
#   zmp_xy = torch.stack([num_x, num_y], dim=-1) / total_fz_safe
#   zmp_xy = torch.where(is_contact.unsqueeze(-1), zmp_xy, com_pos_w)
#   # --- 4. 无点序暴力凸包 (极致优化广播) ---
#   p1 = pos_w.unsqueeze(2)  # [B, 14, 1, 2]
#   p2 = pos_w.unsqueeze(1)  # [B, 1, 14, 2]
#   edge_vec = p2 - p1  # [B, 14, 14, 2]
#   # 边掩码：(i!=j) & active_i & active_j
#   edge_active = contact_mask.unsqueeze(2) * contact_mask.unsqueeze(1)
#   edge_active.diagonal(dim1=1, dim2=2).fill_(0)
#   # 4.1 批量叉乘判定 Hull Edge
#   # 计算 (p2-p1) x (pk-p1)
#   # 使用广播直接生成 [B, 14, 14, 14]
#   rel_pk = pos_w.view(B, 1, 1, 14, 2) - p1.unsqueeze(3)
#   # 手写 2D 叉乘替代 torch.cross 以提速 3-5 倍
#   cross_all = edge_vec[..., 0:1] * rel_pk[..., 1] - edge_vec[..., 1:2] * rel_pk[..., 0]
#   cross_all = cross_all * contact_mask.view(B, 1, 1, 14)
#   # 4.2 判定边界
#   is_hull_edge = ((torch.all(cross_all >= -1e-5, dim=3) |
#                    torch.all(cross_all <= 1e-5, dim=3)) & (edge_active > 0.5))
#   # 4.3 计算符号距离
#   zmp_rel = zmp_xy.view(B, 1, 1, 2) - p1
#   zmp_cross = edge_vec[..., 0] * zmp_rel[..., 1] - edge_vec[..., 1] * zmp_rel[..., 0]
#   # 统一符号：正代表凸包内，负代表凸包外
#   hull_sign = torch.sign(torch.sum(cross_all, dim=3))
#   dist = (zmp_cross / (torch.norm(edge_vec, dim=-1) + 1e-8)) * hull_sign
#   # 4.4 提取最小余量
#   valid_dist = torch.where(is_hull_edge, dist, torch.tensor(1e6, device=device))
#   stability_margin, _ = torch.min(valid_dist.view(B, -1), dim=1)
#   # --- 5. 状态融合与兜底 ---
#   active_num = torch.sum(contact_mask, dim=1)
#   has_hull = active_num >= 3
#   stability_margin = torch.where(has_hull, stability_margin,
#                                  torch.where(is_contact, torch.tensor(-0.05, device=device),
#                                              torch.tensor(-0.1, device=device)))
#
#   return {
#     "zmp_xy": zmp_xy,
#     "stability_margin": stability_margin,
#     "is_contact": is_contact,
#     "active_l": torch.any(contact_mask[:, :7] > 0.5, dim=1),
#     "active_r": torch.any(contact_mask[:, 7:] > 0.5, dim=1),
#     "total_fz": total_fz.squeeze(-1)
#   }
#
#
# def zmp_reward(env: ManagerBasedRlEnv, threshold: float = 0.02) -> torch.Tensor:
#   """
#   基于支撑余量的稳定性奖励。
#   :param threshold: 容忍边界值 (m)，ZMP 离边缘小于此值时开始惩罚。
#   """
#   metrics = zmp(env)
#   margin = metrics["stability_margin"]  # [B]
#   # 方案：指数奖励（ZMP 越靠近中心奖励越高，出界后变为负惩罚）
#   # 当 margin > 0 (在内): reward 为 [0.5, 1.0]
#   # 当 margin < 0 (在外): reward 迅速变为负数
#   # 平滑的指数映射
#   reward_inside = torch.exp(margin * 10.0)  # margin=0.05(中心)时, 奖励较大
#   # 如果出界，给予固定的阶梯惩罚加上距离相关的线性惩罚
#   penalty_outside = -1.0 + margin * 5.0  # margin 为负，所以这是在加大惩罚
#   reward = torch.where(margin > 0, reward_inside, penalty_outside)
#   # 过滤掉完全没接触的情况（腾空期不计入此奖励，或给个中性值）
#   return torch.where(metrics["is_contact"], reward, torch.zeros_like(reward))



def compute_zmp_mjlab(env: ManagerBasedRlEnv, force_threshold: float = 5.0):
    """适配 mjlab 的 ZMP 计算逻辑"""
    # 提取传感器数据
    # mjlab 的接触传感器通常返回 force, pos, torque 等 Tensor
    sensor = env.scene["feet_contact"]
    robot = env.scene["robot"]
    B = env.num_envs
    device = env.device

    # 数据提取 (注意：mjlab/IsaacLab 的 force 维度通常是 [B, num_sensors, 3])
    forces_z = sensor.data.force[..., 2]  # [B, num_feet]
    pos_w = sensor.data.pos[..., :2]  # [B, num_feet, 2]
    torque_x = sensor.data.torque[..., 0]
    torque_y = sensor.data.torque[..., 1]
    robot_com = robot.data.root_com_pose_w[:, :2]  # [B, 2]

    # 1. 接触掩码处理 (Bool型用于逻辑，Float型用于计算)
    mask_bool = (forces_z > force_threshold)
    mask = mask_bool.float()

    # 假设前 N/2 是左脚，后 N/2 是右脚 (请根据你的 MJCF 修改索引)
    num_sensors = forces_z.shape[1]
    mid = num_sensors // 2
    mask_l, mask_r = mask[:, :mid], mask[:, mid:]

    has_contact_l = (torch.sum(mask_l, dim=1) > 0.5)
    has_contact_r = (torch.sum(mask_r, dim=1) > 0.5)

    # 2. ZMP 核心计算 (添加 eps 防止除以 0)
    eps = 1e-6
    total_fz = torch.clamp(torch.sum(forces_z * mask, dim=1, keepdim=True), min=eps)
    total_torque_x = torch.sum(torque_x * mask, dim=1, keepdim=True)
    total_torque_y = torch.sum(torque_y * mask, dim=1, keepdim=True)

    # 加权平均位置
    sum_mask = torch.clamp(torch.sum(mask, dim=1, keepdim=True), min=eps)
    pos_weighted = torch.sum(pos_w * mask.unsqueeze(-1), dim=1) / sum_mask

    # ZMP 公式: x_zmp = x_p - tau_y/Fz, y_zmp = y_p + tau_x/Fz
    zmp_offset = torch.cat([-total_torque_y / total_fz, total_torque_x / total_fz], dim=-1)
    zmp_xy_raw = pos_weighted + zmp_offset

    # 3. 计算支撑区域 (AABB 简化版)
    mask_expand = mask_bool.unsqueeze(-1).expand(-1, -1, 2)
    # 计算接触点的外接矩形
    pos_inf = torch.full_like(pos_w, float('inf'))
    min_xy = torch.min(torch.where(mask_expand, pos_w, pos_inf), dim=1).values
    max_xy = torch.max(torch.where(mask_expand, pos_w, -pos_inf), dim=1).values

    # 支撑中心
    support_center = (min_xy + max_xy) / 2.0

    # 4. 判断 ZMP 是否在支撑域内
    zmp_inside = (zmp_xy_raw[:, 0] >= min_xy[:, 0]) & (zmp_xy_raw[:, 0] <= max_xy[:, 0]) & \
                 (zmp_xy_raw[:, 1] >= min_xy[:, 1]) & (zmp_xy_raw[:, 1] <= max_xy[:, 1])

    # 如果全离地，强制判定为不在域内
    any_contact = (torch.sum(mask, dim=1) > 0.5)
    zmp_inside = zmp_inside & any_contact

    return {
        "zmp_inside": zmp_inside,
        "zmp_to_center_dist": torch.norm(zmp_xy_raw - support_center, p=2, dim=-1),
        "any_contact": any_contact
    }

# 惩罚/奖励函数无需修改
def zmp_stability_reward(env: ManagerBasedRlEnv,  max_dist: float = 0.1) -> torch.Tensor:
    """
    ZMP 稳定性奖励：
    1. 在支撑域内：给予基于到中心距离的奖励
    2. 在支撑域外：奖励为 0 或负值
    """
    data = compute_zmp_mjlab(env)

    # 距离惩罚项 (Exponential 形式通常比 Linear 效果更好)
    reward = torch.exp(-data["zmp_to_center_dist"] / max_dist)

    # 只有在有接触且在域内时才给奖励
    reward = torch.where(data["zmp_inside"], reward, torch.zeros_like(reward))

    return reward


def zmp_outside_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    """ZMP 越界惩罚"""
    data = compute_zmp_mjlab(env)
    # 如果有接触但 ZMP 跑出去了，给惩罚
    penalty = (data["any_contact"] & ~data["zmp_inside"]).float()
    return penalty