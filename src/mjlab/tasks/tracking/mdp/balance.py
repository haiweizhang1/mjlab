from __future__ import annotations

from typing import TYPE_CHECKING, cast
from typing import Dict, Tuple
import torch

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  subtract_frame_transforms, default_orientation,
)

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv



#############center of mass#####################
def center_of_mass(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  
  pass


############zmp###############################
import torch


def compute_zmp_and_margin(env, force_threshold: float = 5.0):
  sensor = env.scene["feet_contact"]
  robot = env.scene["robot"]
  device = env.device
  B = env.num_envs

  # 1. 提取力与位置 (假设 0-6 为左脚, 7-13 为右脚)
  forces_z = sensor.data.force[..., 2]
  pos_w = sensor.data.pos[..., :2]
  mask = (forces_z > force_threshold).float()

  # 2. 状态判定
  contact_l = (torch.sum(mask[:, :7], dim=1) > 0)
  contact_r = (torch.sum(mask[:, 7:], dim=1) > 0)

  # 腾空标志: 0, 单脚: 1, 双脚: 2
  contact_state = contact_l.float() + contact_r.float()

  # 3. ZMP 计算 (保持原逻辑，优化求和)
  total_fz = torch.sum(forces_z * mask, dim=1, keepdim=True) + 1e-6
  zmp_xy = torch.sum(pos_w * forces_z.unsqueeze(-1) * mask.unsqueeze(-1), dim=1) / total_fz

  # 4. 支撑域与余量 (Stability Margin) 计算
  # 技巧：不再计算动态凸包，而是提取活跃脚的“边界点”
  # 假设每只脚通过传感器能拿到 4 个角点坐标（或通过机器人运动学计算）
  # 这里我们简化为直接提取当前活跃的接触点作为支撑点

  margin = torch.full((B,), -0.1, device=device)  # 默认值（腾空）

  # --- 逻辑分支优化 ---

  # CASE 1: 单脚支撑 (使用 AABB 或简单的距离判定)
  single_mask = (contact_state == 1)
  if single_mask.any():
    # 提取活跃脚的所有接触点
    # 简化：对于单脚，margin 可以近似为 ZMP 到该脚接触点质心的距离限制
    # 或者计算 ZMP 到该脚所有活跃接触点连线的最小距离
    pass

    # CASE 2: 双脚支撑 (核心挑战)
  # 改进方案：对于双足，直接取左右脚的最外侧 4 个关键点（脚尖、脚跟的内外侧）
  # 这样 N 永久等于 8，计算开销恒定
  double_mask = (contact_state == 2)
  if double_mask.any():
    # 提取 8 个关键点 [B_double, 8, 2]
    # 使用固定的 8 点 Graham Scan 或直接利用双足间距计算
    # 针对舞蹈动作，重心通常在两脚连线附近
    pass

  # --- 极致简化版：SDF 投影法 ---
  # 将 ZMP 投影到“左脚中心”和“右脚中心”的连线上
  # 这是高动态任务中最有效的简化观测
  foot_l_com = torch.sum(pos_w[:, :7] * mask[:, :7].unsqueeze(-1), dim=1) / (
            torch.sum(mask[:, :7], dim=1, keepdim=True) + 1e-6)
  foot_r_com = torch.sum(pos_w[:, 7:] * mask[:, 7:].unsqueeze(-1), dim=1) / (
            torch.sum(mask[:, 7:], dim=1, keepdim=True) + 1e-6)

  # 计算 ZMP 到双脚连线的垂直距离和投影位置
  line_vec = foot_r_com - foot_l_com
  line_len_sq = torch.sum(line_vec ** 2, dim=-1) + 1e-6
  relative_zmp = zmp_xy - foot_l_com
  projection = torch.sum(relative_zmp * line_vec, dim=-1) / line_len_sq

  # 此时 projection 在 [0, 1] 之间代表在双脚之间
  # 这是一个非常强的平衡特征，适合作为模仿学习的 Observation

  return {
    "zmp_xy": zmp_xy,
    "contact_state": contact_state,  # 0:空, 1:单, 2:双
    "zmp_projection": projection,  # 平衡比例：0(全在左), 1(全在右)
    "foot_dist": torch.sqrt(line_len_sq)
  }

import torch
#基础力版（仅接触力加权，无任何修正）
def compute_zmp_force_only(env:ManagerBasedRlEnv, force_threshold: float = 5.0):
  """
  ZMP版本1：仅接触力加权平均（最简化，无力矩/动量）
  适用场景：低速/静态动作，仅反映重心偏向
  """
  sensor = env.scene["feet_contact"]
  robot = env.scene["robot"]
  device = env.device
  B = env.num_envs

  # 1. 基础数据提取
  forces_z = sensor.data.force[..., 2]  # [B, 14] 垂直接触力
  pos_w = sensor.data.pos[..., :2]      # [B, 14, 2] 接触点xy坐标
  mask = (forces_z > force_threshold).float().clamp(min=1e-6)  # [B, 14]

  # 2. 接触状态判定
  contact_l = (torch.sum(mask[:, :7], dim=1) > 1e-3)
  contact_r = (torch.sum(mask[:, 7:], dim=1) > 1e-3)
  contact_state = contact_l.float() + contact_r.float()

  # 3. 基础力版ZMP（仅力加权）
  total_fz = torch.sum(forces_z * mask, dim=1, keepdim=True) + 1e-6  # [B, 1]
  zmp_xy = torch.sum(pos_w * (forces_z * mask).unsqueeze(-1), dim=1) / total_fz  # [B, 2]

  # 4. 核心特征（重心偏向）
  sum_mask_l = torch.sum(mask[:, :7], dim=1, keepdim=True) + 1e-6
  foot_l_com = torch.sum(pos_w[:, :7] * mask[:, :7].unsqueeze(-1), dim=1) / sum_mask_l
  sum_mask_r = torch.sum(mask[:, 7:], dim=1, keepdim=True) + 1e-6
  foot_r_com = torch.sum(pos_w[:, 7:] * mask[:, 7:].unsqueeze(-1), dim=1) / sum_mask_r

  line_vec = foot_r_com - foot_l_com
  line_len_sq = torch.sum(line_vec ** 2, dim=-1) + 1e-6
  relative_zmp = zmp_xy - foot_l_com
  projection = torch.sum(relative_zmp * line_vec, dim=-1) / line_len_sq
  projection = torch.clamp(projection, -0.2, 1.2)

  return {
    "zmp_xy": zmp_xy,
    "contact_state": contact_state,
    "zmp_projection": projection,
    "foot_dist": torch.sqrt(line_len_sq),
    "version": "force_only"
  }
#力矩版（加入地面反作用力矩修正）import torch
import torch




















def compute_zmp_with_quad_and_reward(env, force_threshold: float = 5.0):
    """
    ZMP版本2 + 支撑四边形 + ZMP与支撑中心距离
    支撑中心距离可直接用于奖励计算
    """
    sensor = env.scene["feet_contact"]
    robot = env.scene["robot"]
    B = env.num_envs

    # 1. 基础数据
    forces_z = sensor.data.force[..., 2]  # [B, 14]
    pos_w = sensor.data.pos[..., :2]      # [B, 14, 2]
    torque_x = sensor.data.torque[..., 0]
    torque_y = sensor.data.torque[..., 1]

    # 2. 接触掩码
    mask = (forces_z > force_threshold).float()  # [B, 14]

    # 3. 接触状态
    contact_l = (torch.sum(mask[:, :7], dim=1) > 1e-3)
    contact_r = (torch.sum(mask[:, 7:], dim=1) > 1e-3)

    mask_double = (contact_l & contact_r).float()
    mask_single_l = (contact_l & ~contact_r).float()
    mask_single_r = (~contact_l & contact_r).float()
    mask_none = (~contact_l & ~contact_r).float()

    contact_state = mask_none*0 + mask_single_l*1 + mask_single_r*2 + mask_double*3

    # 4. 总力/总力矩
    total_fz = torch.sum(forces_z * mask, dim=1, keepdim=True) + 1e-6
    total_torque_x = torch.sum(torque_x * mask, dim=1, keepdim=True)
    total_torque_y = torch.sum(torque_y * mask, dim=1, keepdim=True)

    # 5. 力加权ZMP（仅力）
    sum_mask = torch.sum(mask, dim=1, keepdim=True) + 1e-6
    zmp_xy_force_only = torch.sum(pos_w * mask.unsqueeze(-1), dim=1) / sum_mask

    # 6. 完整ZMP修正
    zmp_x = zmp_xy_force_only[..., 0:1] - (total_torque_y / total_fz)
    zmp_y = zmp_xy_force_only[..., 1:2] + (total_torque_x / total_fz)
    zmp_xy_raw = torch.cat([zmp_x, zmp_y], dim=-1)

    # 7. 脚中心
    sum_mask_l = torch.sum(mask[:, :7], dim=1, keepdim=True) + 1e-6
    foot_l_com = torch.sum(pos_w[:, :7] * mask[:, :7].unsqueeze(-1), dim=1) / sum_mask_l
    sum_mask_r = torch.sum(mask[:, 7:], dim=1, keepdim=True) + 1e-6
    foot_r_com = torch.sum(pos_w[:, 7:] * mask[:, 7:].unsqueeze(-1), dim=1) / sum_mask_r

    robot_com = robot.root_pos[:, :2]

    # 8. ZMP最终值按接触状态限制
    zmp_xy = (
        mask_none.unsqueeze(-1) * robot_com +
        mask_single_l.unsqueeze(-1) * torch.clamp(zmp_xy_raw, foot_l_com - 0.05, foot_l_com + 0.05) +
        mask_single_r.unsqueeze(-1) * torch.clamp(zmp_xy_raw, foot_r_com - 0.05, foot_r_com + 0.05) +
        mask_double.unsqueeze(-1) * zmp_xy_raw
    )

    # 9. 支撑四边形 + 支撑中心 + ZMP到中心距离
    support_quads = []
    support_centers = []
    zmp_inside = []
    zmp_to_center_dist = []

    for b in range(B):
        points = pos_w[b][mask[b] > 0]  # [N,2]
        if points.shape[0] > 0:
            # 四边形顶点
            min_xy = torch.min(points, dim=0).values
            max_xy = torch.max(points, dim=0).values
            quad = torch.stack([
                min_xy,                                      # 左下
                torch.tensor([min_xy[0], max_xy[1]], device=points.device),  # 左上
                max_xy,                                      # 右上
                torch.tensor([max_xy[0], min_xy[1]], device=points.device)   # 右下
            ], dim=0)
            support_quads.append(quad)

            # 支撑中心
            center = (min_xy + max_xy) / 2.0
            support_centers.append(center)

            # ZMP是否在四边形内
            inside = ((zmp_xy[b,0] >= min_xy[0]) & (zmp_xy[b,0] <= max_xy[0]) &
                      (zmp_xy[b,1] >= min_xy[1]) & (zmp_xy[b,1] <= max_xy[1]))
            zmp_inside.append(inside.item())

            # ZMP到支撑中心距离
            dist = torch.norm(zmp_xy[b] - center, p=2)
            zmp_to_center_dist.append(dist.item())
        else:
            support_quads.append(torch.zeros((4,2), device=pos_w.device))
            support_centers.append(torch.zeros(2, device=pos_w.device))
            zmp_inside.append(False)
            zmp_to_center_dist.append(0.0)

    return {
        "zmp_xy": zmp_xy,                   # 最终ZMP
        "contact_state": contact_state,     # 接触状态
        "support_quad": support_quads,      # 四边形顶点
        "support_center": support_centers,  # 支撑中心
        "zmp_inside": zmp_inside,           # ZMP是否在支撑域
        "zmp_to_center_dist": zmp_to_center_dist  # ZMP到中心距离，可作奖励
    }

import torch

# ---------------------------
# 约束惩罚函数
def compute_zmp_constraint_penalty(constraints, penalty_outside: float = 1.0):
    """
    计算ZMP约束惩罚
    - ZMP在支撑域外时返回惩罚值 penalty_outside
    - ZMP在支撑域内返回0
    """
    zmp_inside = constraints["zmp_inside"]
    penalty = [0.0 if inside else penalty_outside for inside in zmp_inside]
    return penalty

# ---------------------------
# 距离奖励函数
def compute_zmp_distance_reward(constraints, max_dist: float = 0.1):
    """
    根据ZMP与支撑中心距离计算奖励
    - 仅在ZMP在支撑域内时计算
    - 越靠近中心奖励越高
    - 距离超过max_dist奖励为0
    """
    zmp_xy = constraints["zmp_xy"]
    support_center = constraints["support_center"]
    zmp_inside = constraints["zmp_inside"]

    reward = []
    for b in range(zmp_xy.shape[0]):
        if zmp_inside[b]:
            center = support_center[b]
            dist = torch.norm(zmp_xy[b]-center, p=2).item()
            r = max(0.0, 1.0 - dist/max_dist)
            reward.append(r)
        else:
            reward.append(0.0)  # 不在支撑域，不给奖励
    return reward



























































#角动量扩展版（力矩 + 角动量变化率）
import torch

def compute_zmp_with_angular_momentum(env, force_threshold: float = 5.0):
  """
  ZMP版本3：力矩 + 角动量变化率（扩展ZMP，无平动动量）
  适用场景：含旋转的动态动作（如舞蹈转身），修正角动量影响
  """
  sensor = env.scene["feet_contact"]
  robot = env.scene["robot"]
  device = env.device
  B = env.num_envs

  # 1. 基础数据提取（力/力矩）
  forces_z = sensor.data.force[..., 2]  # [B, 14]
  pos_w = sensor.data.pos[..., :2]      # [B, 14, 2]
  torque_x = sensor.data.torque[..., 0] # [B, 14]
  torque_y = sensor.data.torque[..., 1] # [B, 14]
  mask = (forces_z > force_threshold).float().clamp(min=1e-6)  # [B, 14]

  # 2. 机器人质量 + 角动量数据
  if hasattr(robot.data, "default_mass"):
    robot_mass = torch.sum(robot.data.default_mass, dim=1, keepdim=True).to(device)  # [B, 1]
  else:
    robot_mass = torch.ones((B, 1), device=device) * 10.0  # 默认10kg
  ang_acc = robot.data.root_ang_acc_w   # [B, 3] 角加速度（角动量变化率）

  # 3. 接触状态判定
  contact_l = (torch.sum(mask[:, :7], dim=1) > 1e-3)
  contact_r = (torch.sum(mask[:, 7:], dim=1) > 1e-3)
  contact_state = contact_l.float() + contact_r.float()

  # 4. 角动量扩展版ZMP
  total_fz = torch.sum(forces_z * mask, dim=1, keepdim=True) + 1e-6  # [B, 1]
  # 基础力矩项
  zmp_x_base = torch.sum((forces_z * mask) * pos_w[..., 0] - torque_y * mask, dim=1, keepdim=True)
  zmp_y_base = torch.sum((forces_z * mask) * pos_w[..., 1] + torque_x * mask, dim=1, keepdim=True)
  # 角动量修正项（0.1为转动惯量经验系数）
  angular_momentum_x = robot_mass * 0.1 * ang_acc[:, 0:1]  # [B, 1]
  angular_momentum_y = robot_mass * 0.1 * ang_acc[:, 1:2]  # [B, 1]
  # 最终ZMP（融入角动量）
  zmp_x = (zmp_x_base - angular_momentum_y) / total_fz
  zmp_y = (zmp_y_base + angular_momentum_x) / total_fz
  zmp_xy = torch.cat([zmp_x, zmp_y], dim=-1)  # [B, 2]

  # 5. 核心特征（重心偏向）
  sum_mask_l = torch.sum(mask[:, :7], dim=1, keepdim=True) + 1e-6
  foot_l_com = torch.sum(pos_w[:, :7] * mask[:, :7].unsqueeze(-1), dim=1) / sum_mask_l
  sum_mask_r = torch.sum(mask[:, 7:], dim=1, keepdim=True) + 1e-6
  foot_r_com = torch.sum(pos_w[:, 7:] * mask[:, 7:].unsqueeze(-1), dim=1) / sum_mask_r

  line_vec = foot_r_com - foot_l_com
  line_len_sq = torch.sum(line_vec ** 2, dim=-1) + 1e-6
  relative_zmp = zmp_xy - foot_l_com
  projection = torch.sum(relative_zmp * line_vec, dim=-1) / line_len_sq
  projection = torch.clamp(projection, -0.2, 1.2)

  return {
    "zmp_xy": zmp_xy,
    "contact_state": contact_state,
    "zmp_projection": projection,
    "foot_dist": torch.sqrt(line_len_sq),
    "version": "with_angular_momentum"
  }
#完整版（力矩 + 角动量 + 平动动量

import torch


def compute_zmp_full(env:ManagerBasedRlEnv, force_threshold: float = 5.0):
  """
  ZMP版本4：力矩 + 角动量 + 平动动量（终极扩展ZMP）
  适用场景：高动态舞蹈动作（跨步/转身/急停），最贴合实际平衡状态
  """
  sensor = env.scene["feet_contact"]
  robot = env.scene["robot"]
  device = env.device
  B = env.num_envs

  # 1. 基础数据提取（力/力矩）
  forces_z = sensor.data.force[..., 2]  # [B, 14]
  pos_w = sensor.data.pos[..., :2]      # [B, 14, 2]
  torque_x = sensor.data.torque[..., 0] # [B, 14]
  torque_y = sensor.data.torque[..., 1] # [B, 14]
  mask = (forces_z > force_threshold).float().clamp(min=1e-6)  # [B, 14]

  # 2. 机器人动量/质量数据（核心：平动+角动量）
  # 2.1 总质量
  if hasattr(robot.data, "default_mass"):
    robot_mass = torch.sum(robot.data.default_mass, dim=1, keepdim=True).to(device)  # [B, 1]
  else:
    robot_mass = torch.ones((B, 1), device=device) * 10.0  # 默认10kg
  # 2.2 质心状态（平动动量相关）
  com_pos = robot.data.root_com_pos_w  # [B, 3] 质心位置
  com_acc = robot.data.root_lin_acc_w  # [B, 3] 质心加速度（平动动量变化率）
  # 2.3 角动量状态
  ang_acc = robot.data.root_ang_acc_w   # [B, 3] 角加速度（角动量变化率）

  # 3. 接触状态判定
  contact_l = (torch.sum(mask[:, :7], dim=1) > 1e-3)
  contact_r = (torch.sum(mask[:, 7:], dim=1) > 1e-3)
  contact_state = contact_l.float() + contact_r.float()

  # 4. 完整版ZMP（力矩+角动量+平动动量）
  total_fz = torch.sum(forces_z * mask, dim=1, keepdim=True) + 1e-6  # [B, 1]
  # 4.1 基础力矩项
  zmp_x_base = torch.sum((forces_z * mask) * pos_w[..., 0] - torque_y * mask, dim=1, keepdim=True)
  zmp_y_base = torch.sum((forces_z * mask) * pos_w[..., 1] + torque_x * mask, dim=1, keepdim=True)
  # 4.2 角动量修正项
  angular_momentum_x = robot_mass * 0.1 * ang_acc[:, 0:1]  # [B, 1]
  angular_momentum_y = robot_mass * 0.1 * ang_acc[:, 1:2]  # [B, 1]
  # 4.3 平动动量修正项（核心：质心高度×平动加速度）
  z_com = torch.clamp(com_pos[:, 2:3], min=0.1)  # 质心z高度（避免除零）
  linear_momentum_x = robot_mass * z_com * com_acc[:, 0:1]  # [B, 1] 平动动量x修正
  linear_momentum_y = robot_mass * z_com * com_acc[:, 1:2]  # [B, 1] 平动动量y修正
  # 4.4 最终ZMP（融合所有修正）
  zmp_x = (zmp_x_base - angular_momentum_y + linear_momentum_x) / total_fz
  zmp_y = (zmp_y_base + angular_momentum_x + linear_momentum_y) / total_fz
  zmp_xy = torch.cat([zmp_x, zmp_y], dim=-1)  # [B, 2]

  # 5. 核心特征（重心偏向）
  sum_mask_l = torch.sum(mask[:, :7], dim=1, keepdim=True) + 1e-6
  foot_l_com = torch.sum(pos_w[:, :7] * mask[:, :7].unsqueeze(-1), dim=1) / sum_mask_l
  sum_mask_r = torch.sum(mask[:, 7:], dim=1, keepdim=True) + 1e-6
  foot_r_com = torch.sum(pos_w[:, 7:] * mask[:, 7:].unsqueeze(-1), dim=1) / sum_mask_r

  line_vec = foot_r_com - foot_l_com
  line_len_sq = torch.sum(line_vec ** 2, dim=-1) + 1e-6
  relative_zmp = zmp_xy - foot_l_com
  projection = torch.sum(relative_zmp * line_vec, dim=-1) / line_len_sq
  projection = torch.clamp(projection, -0.2, 1.2)

  return {
    "zmp_xy": zmp_xy,
    "contact_state": contact_state,
    "zmp_projection": projection,
    "foot_dist": torch.sqrt(line_len_sq),
    "linear_momentum": torch.norm(robot_mass * com_acc, dim=-1),  # 平动动量大小
    "angular_momentum": torch.norm(robot_mass * 0.1 * ang_acc, dim=-1),  # 角动量大小
    "version": "full_momentum"
  }

def com_momentum(env: ManagerBasedRlEnv):
  robot_mass = torch.sum(robot.data.default_mass, dim=1, keepdim=True).to(device)
  com_pos = robot.data.root_com_pos_w
  line_momentum = robot_mass * robot.data.subtree_linvel[body_id].copy()## body_id   根
  body_id = robot.data.root_body_id
  angular_momentum = robot.data.subtree_angmom[body_id].copy()

  return {
    "com_pos": com_pos,  # 3阶向量
    "linear_mom": line_momentum,  # 3阶向量 (kg*m/s)
    "angular_mom": angular_momentum,  # 3阶向量 (kg*m^2/s)
  }