from __future__ import annotations

from typing import TYPE_CHECKING, cast
from typing import Dict, Tuple
import torch

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  subtract_frame_transforms,
)

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  pos, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )

  return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  _, ori = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )
  mat = matrix_from_quat(ori)
  return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  pos_b, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )

  return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  _, ori_b = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(mat.shape[0], -1)
# def zmp(env: ManagerBasedRlEnv) -> torch.Tensor:
#   # command = cast(MotionCommand, env.command_manager.get_term(command_name))
#   """计算机器人当前的 ZMP (Zero Moment Point)"""
#
#   # 1. 这里的 sensor_name 必须对应你在机器人 config 中实例化的 ContactSensor 名字
#   # 假设你之前定义的 ContactSensor 叫 "foot_contact"
#   sensor_name = "feet_contact"
#   sensor: ContactSensor = env.scene[sensor_name]
#   assert sensor.data.found is not None
#   data = sensor.data  # 获取 ContactData 对象
#   if data.force is None or data.pos is None:
#     return torch.zeros((env.num_envs, 2), device=env.device)
#   forces = data.force
#   positions = data.pos
#   found = data.found  # [B, N], 值为接触点的数量
#   # 3. 构造掩码，只保留真正发生碰撞的点 (found > 0)
#   contact_mask = (found > 0).unsqueeze(-1).float()
#   # 提取垂直分量 fz [B, N, 1]
#   fz = forces[..., 2:3] * contact_mask
#   # 4. 计算总垂直支撑力
#   total_fz = torch.sum(fz, dim=1)  # [B, 1]
#   # 5. 加权平均计算 X 和 Y 坐标
#   # 公式: ZMP = sum(pos_i * fz_i) / sum(fz_i)
#   # 使用 epsilon 防止除以零（当机器人腾空时）
#   epsilon = 1e-6
#   zmp_xy = torch.sum(positions[..., :2] * fz, dim=1) / (total_fz + epsilon)
#   # 6. 处理腾空状态：如果完全没接触，将 ZMP 设为质心投影
#   # 你可以从内置传感器获取质心位置
#   no_contact = (total_fz < 0.1).flatten()
#   if no_contact.any():
#
#     com_pos = env.scene["robot"].data.body_com_pos_w[:,0,].squeeze()
#     zmp_xy[no_contact] = com_pos[no_contact, :2]
#   return  zmp_xy # 返回 [B, 2] (x, y)
#   import torch


import torch
from typing import Dict


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

  # return {
  #   "zmp_xy": zmp_xy,
  #   "stability_margin": stability_margin,
  #   "is_contact": is_contact,
  #   "active_l": torch.any(contact_mask[:, :7] > 0.5, dim=1),
  #   "active_r": torch.any(contact_mask[:, 7:] > 0.5, dim=1),
  #   "total_fz": total_fz.squeeze(-1)
  # }
  return  zmp_xy
