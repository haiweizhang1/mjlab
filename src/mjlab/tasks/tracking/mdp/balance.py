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


#############center of mass#####################
def center_of_mass(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  
  pass


############zmp###############################
import torch
from typing import cast


# 假设这些类已经在你的环境中定义
# from omni.isaac.lab.envs import ManagerBasedRlEnv
# from your_project.commands import MotionCommand

def zmp(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
    """
    计算机器人的零力矩点 (ZMP)。

    公式:
        ZMP_x = sum(x_i * f_z_i - tau_y_i) / sum(f_z_i)
        ZMP_y = sum(y_i * f_z_i + tau_x_i) / sum(f_z_i)

    Args:
        env: 强化学习环境实例
        command_name: 命令名称

    Returns:
        zmp_pos: [Batch, 2] ZMP在世界坐标系下的 (x, y) 位置
    """
    # --- 0. 获取环境对象 ---
    # 根据你的 snippet，假设已经正确获取了 sensor 和 robot
    sensor = env.scene["feet_contact"]
    # --- 1. 基础数据提取 ---
    # sensor.data.force: [Batch, Num_Sensors, 3]
    # sensor.data.pos:   [Batch, Num_Sensors, 3] (世界坐标系)
    # sensor.data.torque:[Batch, Num_Sensors, 3]
    forces_z = sensor.data.force[..., 2]  # [B, 14], 垂直方向力 (f_z)
    pos_x = sensor.data.pos[..., 0]  # [B, 14], 接触点 x 坐标 (x_i)
    pos_y = sensor.data.pos[..., 1]  # [B, 14], 接触点 y 坐标 (y_i)

    torque_x = sensor.data.torque[..., 0]  # [B, 14], 绕 x 轴力矩 (tau_x)
    torque_y = sensor.data.torque[..., 1]  # [B, 14], 绕 y 轴力矩 (tau_y)
    # --- 2. 创建 Mask (重要) ---
    # 只有接触地面的点才贡献 ZMP。通常用一个小的力阈值来过滤噪声。
    CONTACT_FORCE_THRESHOLD = 1.0  # 根据仿真器单位调整，通常是 1N
    mask = (forces_z > CONTACT_FORCE_THRESHOLD).float()  # [B, 14]
    # --- 3. 计算分母：总垂直力 ---
    # sum(f_z_i)
    # 加上 1e-6 防止除以零（当机器人完全腾空时）
    total_fz = torch.sum(forces_z * mask, dim=1, keepdim=True) + 1e-6  # [B, 1]
    # --- 4. 计算分子：力矩平衡项 ---
    # 分子 X: sum(x_i * f_z_i - tau_y_i)
    # 注意: 绕Y轴的正力矩会倾向于将ZMP推向负X方向，所以是减号
    numerator_x = torch.sum((pos_x * forces_z - torque_y) * mask, dim=1, keepdim=True)
    # 分子 Y: sum(y_i * f_z_i + tau_x_i)
    # 注意: 绕X轴的正力矩会倾向于将ZMP推向正Y方向，所以是加号
    numerator_y = torch.sum((pos_y * forces_z + torque_x) * mask, dim=1, keepdim=True)
    # --- 5. 计算 ZMP 坐标 ---
    zmp_x = numerator_x / total_fz
    zmp_y = numerator_y / total_fz
    # 拼接结果 [B, 2]
    zmp_w = torch.cat([zmp_x, zmp_y], dim=1)
    # --- 6. (可选) 特殊情况处理：完全腾空 ---
    # 如果 total_fz 非常小（腾空），ZMP 计算无意义。
    # 通常将其设置为机器人当前的水平位置，或者 mask 掉不参与 loss 计算。
    # 这里简单处理：如果受力太小，就用此时机器人基座的 (x,y) 代替，避免数值爆炸。
    is_flying = total_fz.squeeze(-1) < 2.0 * CONTACT_FORCE_THRESHOLD
    if is_flying.any():
        base_pos = env.scene["robot"].data.root_pos_w[:, :2]  # 获取基座位置
        zmp_w[is_flying] = base_pos[is_flying]
    return zmp_w

#########dcm##############################################

def dcm(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  sensor = env.scene["feet_contact"]
  robot = env.scene["robot"]
  device = env.device
##################支撑域
"""输入，接触点坐标shape (4096,0-12,2)  
   输出，凸包
"""

def support_polygons(contact_points_batch, mask=None):
    """
    计算双足机器人支撑域（凸包）。

    参数:
        contact_points_batch: np.array, shape (Batch, Max_Points, 2)
                              例如 (4096, 12, 2)。
        mask: np.array, shape (Batch, Max_Points), 可选
              布尔值，True表示该点是有效接触点。如果为None，
              则假设所有非NaN/非无穷大的点都有效。

    返回:
        polygons: list of np.array
                  由凸包顶点组成的列表。每个元素的形状为 (N_vertices, 2)。
                  注意：由于凸包顶点数不固定，无法返回规则Tensor。
    """
    batch_size = contact_points_batch.shape[0]
    support_polygons = []

    for i in range(batch_size):
        points = contact_points_batch[i]  # shape (12, 2)

        # 1. 数据清洗/筛选有效点
        if mask is not None:
            valid_points = points[mask[i] > 0]
        else:
            # 示例：去除 NaN 或 极值，具体取决于你的数据预处理
            # 这里假设不做过滤，或者假设数据已经清洗过
            valid_points = points[~np.isnan(points).any(axis=1)]

        # 去重 (防止这就地打转)
        valid_points = np.unique(valid_points, axis=0)

        num_points = valid_points.shape[0]

        # 2. 计算凸包
        # 情况 A: 点数不足以构成多边形 (0, 1, 2 个点)
        if num_points < 3:
            # 支撑域就是点本身或线段
            support_polygons.append(valid_points)
            continue

        # 情况 B: 3个及以上点，计算 ConvexHull
        try:
            hull = ConvexHull(valid_points)
            # hull.vertices 包含顶点索引，我们需要按逆时针顺序排列的坐标
            hull_points = valid_points[hull.vertices]
            support_polygons.append(hull_points)
        except QhullError:
            # 极少数情况（如所有点共线），降级为直接返回点集
            support_polygons.append(valid_points)

    return support_polygons


# --- 使用示例 ---
# 模拟输入: 4096帧, 最大12个点, 2D坐标
input_data = np.random.rand(4096, 12, 2)

# 假设前6个点是左脚，后6个点是右脚，模拟随机接触
# 这里生成一个随机mask作为演示
input_mask = np.random.randint(0, 2, (4096, 12))

results = compute_support_polygons(input_data, mask=input_mask)

# 查看第一帧的支撑域顶点
print(f"第一帧支撑域顶点数量: {len(results[0])}")
print(results[0])




















































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