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

def com_linear_mom(env: ManagerBasedRlEnv):
  robot = env.scene["robot"]

  device =env.device
  robot_mass = torch.sum(robot.data.model.body_mass, dim=1, keepdim=True).to(device)
  root_body_id = robot.indexing.root_body_id
  line_momentum = robot_mass * env.sim.data.subtree_linvel[:,root_body_id].to(device)####id
  return line_momentum
def com_angular_mom(env: ManagerBasedRlEnv):
  device =env.device
  # robot_mass = torch.sum(robot.data.default_mass, dim=1, keepdim=True).to(device)
  robot = env.scene["robot"]
  # device =env.device
  root_body_id = robot.indexing.root_body_id
  angular_momentum = env.sim.data.subtree_angmom[:,root_body_id].to(device)
  return angular_momentum

