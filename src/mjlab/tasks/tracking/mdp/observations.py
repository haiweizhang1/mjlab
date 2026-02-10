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


def zmp_dynamic_full(env: ManagerBasedRlEnv) -> torch.Tensor:
  # --- 1. 资源提取 ---
  robot = env.scene["robot"]
  B = env.num_envs
  device = env.device

  # 获取质心 (CoM) 的世界坐标系状态
  # root_com_pos_w: [B, 3]
  # root_com_lin_vel_w: [B, 3] -> 用于计算加速度
  com_pos = robot.data.root_com_pos_w
  com_vel = robot.data.root_com_lin_vel_w

  # 模拟加速度 (或者直接从 robot.data 中获取，如果 Isaac 版本支持)
  # 在 RL 步进中，通常用 (vel - prev_vel) / dt
  dt = env.physics_dt
  com_acc = (com_vel - getattr(env, "_prev_com_vel", com_vel)) / dt
  env._prev_com_vel = com_vel.clone()  # 缓存用于下一帧计算

  # --- 2. 提取角动量导数 (L_dot) ---
  # 这是处理“肢体甩动”和“空翻”的核心项
  # h_dot = dL/dt. 在 Isaac Lab 中可以通过 center_of_mass_ang_mom 的差分获得
  ang_mom = robot.data.root_com_ang_mom_w  # [B, 3]
  ang_mom_dot = (ang_mom - getattr(env, "_prev_ang_mom", ang_mom)) / dt
  env._prev_ang_mom = ang_mom.clone()

  # --- 3. 完整动力学计算 ---
  # 公式: P_zmp = P_com - (z_com / (z_acc + g)) * x_acc - (L_dot / (m * (z_acc + g)))
  g = 9.81
  m = robot.data.default_mass.unsqueeze(-1)  # [B, 1]

  # 垂直分母: m * (z_acc + g)
  # 这代表了有效支撑力。如果为0，说明机器人处于完全失重状态
  denom = m * (com_acc[:, 2:3] + g)
  denom_safe = torch.where(denom.abs() < 1e-2, torch.sign(denom) * 1e-2, denom)

  z_ref = com_pos[:, 2:3]  # 以质心高度作为参考

  # ZMP X 分量 (对应力矩 Y 的平衡)
  # x_zmp = x_com - (z_com * m * x_acc + L_dot_y) / denom
  zmp_x = com_pos[:, 0:1] - (z_ref * m * com_acc[:, 0:1] + ang_mom_dot[:, 1:2]) / denom_safe

  # ZMP Y 分量 (对应力矩 X 的平衡)
  # y_zmp = y_com - (z_com * m * y_acc - L_dot_x) / denom
  zmp_y = com_pos[:, 1:2] - (z_ref * m * com_acc[:, 1:2] - ang_mom_dot[:, 0:1]) / denom_safe

  # --- 4. 极端动态修正 ---
  # 如果机器人处于空中（没有脚触地），ZMP 理论上无定义。
  # 我们检测总垂直力（如果有传感器）或直接用加速度判定。
  feet_fz = env.scene["feet_contact"].data.force[..., 2]
  is_airborne = torch.sum(feet_fz, dim=1, keepdim=True) < 1.0  # 阈值 1N

  zmp_xy = torch.cat([zmp_x, zmp_y], dim=-1)

  # 腾空时回归 CoM 投影
  zmp_xy = torch.where(is_airborne, com_pos[:, :2], zmp_xy)

  return zmp_xy

