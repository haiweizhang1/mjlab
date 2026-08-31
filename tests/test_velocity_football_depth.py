"""Configuration and observation tests for the two active depth tasks."""

from types import SimpleNamespace
from typing import Any, cast

import mujoco
import pytest
import torch

from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity_football_depth import (
  DEPTH_BASELINE_TASK_ID,
  DEPTH_CALIBRATED_LEGACY_TASK_ID,
  DEPTH_CANDIDATE_TASK_ID,
  DEPTH_KLAVIER_E1_STAGE2_TASK_ID,
  DEPTH_KLAVIER_FACTORIAL_PUSH_OFF_NO_SYM_TASK_ID,
  DEPTH_KLAVIER_FACTORIAL_PUSH_OFF_SYM_TASK_ID,
  DEPTH_KLAVIER_FACTORIAL_PUSH_ON_NO_SYM_TASK_ID,
  DEPTH_KLAVIER_FACTORIAL_PUSH_ON_SYM_TASK_ID,
  DEPTH_KLAVIER_STAGE1_TASK_ID,
  DEPTH_KLAVIER_STAGE2_TASK_ID,
  DEPTH_KLAVIER_VISIBILITY_STAGE2_TASK_ID,
)
from mjlab.tasks.velocity_football_depth.env_cfg import (
  DEPTH_CAMERA_ROTATION_DR_RADIANS,
  DEPTH_HEIGHT,
  DEPTH_SENSOR_NAME,
  DEPTH_WIDTH,
)
from mjlab.tasks.velocity_football_depth.observations import (
  ball_visibility_from_camera_segmentation,
  normalized_camera_depth,
)
from mjlab.tasks.velocity_football_depth.runner import DepthTeacherDistillationRunner


def test_only_expected_depth_football_tasks_are_registered() -> None:
  task_ids = {
    task_id
    for task_id in list_tasks()
    if task_id.startswith("Mjlab-Velocity-Football-Depth-")
  }
  assert task_ids == {
    DEPTH_BASELINE_TASK_ID,
    DEPTH_CALIBRATED_LEGACY_TASK_ID,
    DEPTH_CANDIDATE_TASK_ID,
    DEPTH_KLAVIER_STAGE1_TASK_ID,
    DEPTH_KLAVIER_STAGE2_TASK_ID,
    DEPTH_KLAVIER_VISIBILITY_STAGE2_TASK_ID,
    DEPTH_KLAVIER_FACTORIAL_PUSH_OFF_NO_SYM_TASK_ID,
    DEPTH_KLAVIER_FACTORIAL_PUSH_OFF_SYM_TASK_ID,
    DEPTH_KLAVIER_FACTORIAL_PUSH_ON_NO_SYM_TASK_ID,
    DEPTH_KLAVIER_FACTORIAL_PUSH_ON_SYM_TASK_ID,
    DEPTH_KLAVIER_E1_STAGE2_TASK_ID,
  }


@pytest.mark.parametrize(
  "task_id",
  (
    DEPTH_BASELINE_TASK_ID,
    DEPTH_CALIBRATED_LEGACY_TASK_ID,
    DEPTH_CANDIDATE_TASK_ID,
    DEPTH_KLAVIER_STAGE1_TASK_ID,
    DEPTH_KLAVIER_STAGE2_TASK_ID,
    DEPTH_KLAVIER_VISIBILITY_STAGE2_TASK_ID,
    DEPTH_KLAVIER_FACTORIAL_PUSH_OFF_NO_SYM_TASK_ID,
    DEPTH_KLAVIER_FACTORIAL_PUSH_OFF_SYM_TASK_ID,
    DEPTH_KLAVIER_FACTORIAL_PUSH_ON_NO_SYM_TASK_ID,
    DEPTH_KLAVIER_FACTORIAL_PUSH_ON_SYM_TASK_ID,
    DEPTH_KLAVIER_E1_STAGE2_TASK_ID,
  ),
)
def test_active_depth_tasks_expose_temporal_teacher_contract(task_id: str) -> None:
  cfg = load_env_cfg(task_id)
  runner_cfg = cast(Any, load_rl_cfg(task_id))

  assert cfg.observations["actor"].history_length == 5
  assert cfg.observations["actor_history"].history_length == 10
  assert cfg.observations["depth"].history_length == 10
  assert cfg.observations["student_proprio"].history_length == 5
  assert runner_cfg.obs_groups["student"] == ("actor",)
  assert runner_cfg.obs_groups["teacher"] == ("actor", "actor_history")
  assert load_runner_cls(task_id) is DepthTeacherDistillationRunner


def test_frozen_mlp_depth_baseline_contract() -> None:
  cfg = load_env_cfg(DEPTH_BASELINE_TASK_ID)
  runner_cfg = cast(Any, load_rl_cfg(DEPTH_BASELINE_TASK_ID))
  event = cfg.events["randomize_depth_camera_extrinsics"]
  depth = cfg.observations["depth"].terms["image"]

  assert event.func.__name__ == "randomize_camera_between_uncertain_limits"
  assert event.params["alpha_range"] == (0.0, 0.25)
  assert event.params["fixed_lateral_position"] == pytest.approx(0.01753)
  assert event.params["lower_x_residual_range"] == (-0.03, 0.03)
  assert event.params["lower_z_residual_range"] == (-0.01, 0.01)
  assert event.params["lower_pitch_residual_range"] == pytest.approx(
    (-DEPTH_CAMERA_ROTATION_DR_RADIANS, DEPTH_CAMERA_ROTATION_DR_RADIANS)
  )
  assert depth.params["crop_shift_x_pixels"] == 1
  assert depth.params["crop_shift_y_pixels"] == 1
  assert depth.params["dropout_probability"] == 0.05
  assert (depth.delay_min_lag, depth.delay_max_lag) == (0, 0)
  assert runner_cfg.student.cnn_cfg["freeze_coordinate_actor"] is True
  assert runner_cfg.max_iterations == 10_000


def test_constrained_depth_candidate_contract() -> None:
  cfg = load_env_cfg(DEPTH_CANDIDATE_TASK_ID)
  runner_cfg = cast(Any, load_rl_cfg(DEPTH_CANDIDATE_TASK_ID))
  event = cfg.events["randomize_depth_camera_extrinsics"]

  assert event.params["alpha_range"] == (0.0, 0.35)
  assert event.params["fixed_lateral_position"] == pytest.approx(0.01753)
  assert event.params["lower_x_residual_range"] == (-0.035, 0.035)
  assert event.params["lower_z_residual_range"] == (-0.015, 0.015)
  assert event.params["lower_pitch_residual_range"] == pytest.approx(
    (-1.5 * DEPTH_CAMERA_ROTATION_DR_RADIANS, 1.5 * DEPTH_CAMERA_ROTATION_DR_RADIANS)
  )
  assert runner_cfg.algorithm.class_name.endswith("ConstrainedLatentDistillation")
  assert runner_cfg.algorithm.rollout_policy == "mixed"
  assert runner_cfg.algorithm.student_rollout_final_probability == pytest.approx(0.3)
  assert runner_cfg.student.cnn_cfg["freeze_coordinate_actor"] is False
  assert runner_cfg.student.cnn_cfg["train_mlp_last_layer_only"] is True


def test_klavier_stage_one_depth_contract() -> None:
  cfg = load_env_cfg(DEPTH_KLAVIER_STAGE1_TASK_ID)
  runner_cfg = cast(Any, load_rl_cfg(DEPTH_KLAVIER_STAGE1_TASK_ID))
  event = cfg.events["randomize_depth_camera_extrinsics"]
  push_max = cfg.curriculum["push_velocity_levels"].params["max_velocity_range"]

  assert tuple(cfg.observations["actor_history"].terms) == ("ball_features_b",)
  assert cfg.observations["depth"].history_length == 10
  ball_delay = cfg.observations["actor_history"].terms["ball_features_b"]
  depth_delay = cfg.observations["depth"].terms["image"]
  assert (ball_delay.delay_min_lag, ball_delay.delay_max_lag) == (0, 2)
  assert (depth_delay.delay_min_lag, depth_delay.delay_max_lag) == (0, 2)
  assert ball_delay.delay_shared_key == depth_delay.delay_shared_key
  assert ball_delay.delay_shared_key is not None
  assert event.params["alpha_range"] == (0.0, 0.25)
  assert push_max["x"] == (-1.5, 1.5)
  assert push_max["yaw"] == (-1.57, 1.57)
  assert runner_cfg.teacher.hidden_dims == (1024, 512, 256)
  assert runner_cfg.student.hidden_dims == (1024, 512, 256)
  assert runner_cfg.student.cnn_cfg["freeze_coordinate_actor"] is True
  assert runner_cfg.algorithm.class_name.endswith("FrozenLatentDistillation")
  assert runner_cfg.algorithm.rollout_policy == "teacher"
  assert runner_cfg.algorithm.latent_loss_coef == pytest.approx(0.1)


def test_klavier_stage_two_depth_contract() -> None:
  cfg = load_env_cfg(DEPTH_KLAVIER_STAGE2_TASK_ID)
  runner_cfg = cast(Any, load_rl_cfg(DEPTH_KLAVIER_STAGE2_TASK_ID))
  ball_delay = cfg.observations["actor_history"].terms["ball_features_b"]
  depth_delay = cfg.observations["depth"].terms["image"]

  assert ball_delay.delay_shared_key == depth_delay.delay_shared_key
  assert ball_delay.delay_shared_key is not None
  assert runner_cfg.teacher.hidden_dims == (1024, 512, 256)
  assert runner_cfg.student.hidden_dims == (1024, 512, 256)
  assert runner_cfg.student.cnn_cfg["freeze_coordinate_actor"] is False
  assert runner_cfg.student.cnn_cfg["train_mlp_last_layer_only"] is True
  assert runner_cfg.algorithm.class_name.endswith("ConstrainedLatentDistillation")
  assert runner_cfg.algorithm.rollout_policy == "mixed"
  assert runner_cfg.algorithm.student_rollout_ramp_updates == 2_000
  assert runner_cfg.algorithm.student_rollout_final_probability == pytest.approx(0.3)
  assert runner_cfg.algorithm.learning_rate == pytest.approx(3.0e-4)
  assert runner_cfg.algorithm.mlp_learning_rate == pytest.approx(1.0e-5)
  assert runner_cfg.algorithm.latent_loss_coef == pytest.approx(0.1)
  assert runner_cfg.algorithm.mlp_anchor_loss_coef == pytest.approx(1.0e-3)


def test_klavier_e1_stage_two_continuation_contract() -> None:
  cfg = load_env_cfg(DEPTH_KLAVIER_E1_STAGE2_TASK_ID)
  runner_cfg = cast(Any, load_rl_cfg(DEPTH_KLAVIER_E1_STAGE2_TASK_ID))

  assert "push_velocity_levels" not in cfg.curriculum
  assert runner_cfg.student.cnn_cfg["freeze_coordinate_actor"] is False
  assert runner_cfg.student.cnn_cfg["train_mlp_last_layer_only"] is True
  assert runner_cfg.algorithm.symmetry_cfg is None
  assert runner_cfg.algorithm.rollout_policy == "mixed"
  assert runner_cfg.algorithm.student_rollout_final_probability == pytest.approx(0.3)
  assert runner_cfg.algorithm.learning_rate == pytest.approx(3.0e-4)
  assert runner_cfg.algorithm.mlp_learning_rate == pytest.approx(1.0e-5)
  assert runner_cfg.algorithm.latent_loss_coef == pytest.approx(0.1)
  assert runner_cfg.algorithm.mlp_anchor_loss_coef == pytest.approx(1.0e-3)
  assert runner_cfg.max_iterations == 30_000
  assert runner_cfg.save_interval == 1_000


def test_klavier_visibility_stage_two_contract() -> None:
  cfg = load_env_cfg(DEPTH_KLAVIER_VISIBILITY_STAGE2_TASK_ID)
  play_cfg = load_env_cfg(DEPTH_KLAVIER_VISIBILITY_STAGE2_TASK_ID, play=True)
  runner_cfg = cast(Any, load_rl_cfg(DEPTH_KLAVIER_VISIBILITY_STAGE2_TASK_ID))
  camera = next(
    sensor for sensor in cfg.scene.sensors if sensor.name == DEPTH_SENSOR_NAME
  )
  play_camera = next(
    sensor for sensor in play_cfg.scene.sensors if sensor.name == DEPTH_SENSOR_NAME
  )
  depth_term = cfg.observations["depth"].terms["image"]
  target_group = cfg.observations["depth_visibility_target"]
  target_term = target_group.terms["visible"]

  assert camera.data_types == ("depth", "segmentation")
  assert play_camera.data_types == ("depth",)
  assert target_group.history_length == 10
  assert (target_term.delay_min_lag, target_term.delay_max_lag) == (0, 2)
  assert target_term.delay_shared_key == depth_term.delay_shared_key
  assert target_term.params["min_ball_pixels"] == 2
  assert runner_cfg.student.cnn_cfg["predict_visibility"] is True
  assert runner_cfg.algorithm.visibility_loss_coef == pytest.approx(0.2)
  assert runner_cfg.algorithm.visibility_target_group == "depth_visibility_target"


def test_depth_camera_matches_deployment_calibration() -> None:
  cfg = load_env_cfg(DEPTH_BASELINE_TASK_ID)
  camera = next(
    sensor for sensor in cfg.scene.sensors if sensor.name == DEPTH_SENSOR_NAME
  )

  assert isinstance(camera, CameraSensorCfg)
  assert camera.parent_body == "robot/torso_link"
  assert camera.pos == pytest.approx((0.1135993074, 0.01753, 0.3934754688))
  assert camera.quat == pytest.approx(
    (0.6980107470, 0.1130530720, -0.1130530720, -0.6980107470)
  )
  assert (camera.height, camera.width) == (DEPTH_HEIGHT, DEPTH_WIDTH)
  assert "depth" in camera.data_types


def test_normalized_camera_depth_sanitizes_invalid_pixels() -> None:
  raw = torch.tensor([[[[0.1], [0.6], [3.5], [float("nan")]]]])
  camera = SimpleNamespace(data=SimpleNamespace(depth=raw))
  env = SimpleNamespace(scene={DEPTH_SENSOR_NAME: camera})

  actual = normalized_camera_depth(
    env,  # type: ignore[arg-type]
    DEPTH_SENSOR_NAME,
    min_depth=0.2,
    max_depth=3.0,
  )

  assert actual.shape == (1, 1, 1, 4)
  torch.testing.assert_close(actual, torch.tensor([[[[1.0, 0.2, 1.0, 1.0]]]]))


def test_depth_ball_pixels_are_masked_only_in_sensor_hidden_envs() -> None:
  raw = torch.tensor(
    [
      [[[0.6], [0.9], [1.2]]],
      [[[0.6], [0.9], [1.2]]],
    ]
  )
  geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
  segmentation = torch.tensor(
    [
      [[[3, geom_type], [7, geom_type], [4, geom_type]]],
      [[[3, geom_type], [7, geom_type], [4, geom_type]]],
    ]
  )
  camera = SimpleNamespace(data=SimpleNamespace(depth=raw, segmentation=segmentation))
  ball = SimpleNamespace(indexing=SimpleNamespace(geom_ids=torch.tensor([7])))
  env = SimpleNamespace(scene={DEPTH_SENSOR_NAME: camera, "ball": ball})
  env._football_masked_ball_visual = {
    "episode_hidden": torch.tensor([False, False]),
    "synthetic_hidden": torch.tensor([True, False]),
  }

  actual = normalized_camera_depth(
    env,  # type: ignore[arg-type]
    DEPTH_SENSOR_NAME,
    min_depth=0.2,
    max_depth=3.0,
    mask_ball_when_sensor_hidden=True,
  )

  torch.testing.assert_close(
    actual,
    torch.tensor(
      [
        [[[0.2, 1.0, 0.4]]],
        [[[0.2, 0.3, 0.4]]],
      ]
    ),
  )


def test_segmentation_visibility_requires_two_policy_pixels() -> None:
  geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
  segmentation = torch.zeros(2, 4, 4, 2, dtype=torch.long)
  segmentation[..., 1] = geom_type
  segmentation[0, 0, 0, 0] = 7
  segmentation[0, 1, 1, 0] = 7
  segmentation[1, 0, 0, 0] = 7
  camera = SimpleNamespace(data=SimpleNamespace(segmentation=segmentation))
  ball = SimpleNamespace(indexing=SimpleNamespace(geom_ids=torch.tensor([7])))
  env = SimpleNamespace(scene={DEPTH_SENSOR_NAME: camera, "ball": ball})

  actual = ball_visibility_from_camera_segmentation(
    env,  # type: ignore[arg-type]
    DEPTH_SENSOR_NAME,
    output_size=(4, 4),
    min_ball_pixels=2,
  )

  torch.testing.assert_close(actual, torch.tensor([[1.0], [0.0]]))


@pytest.mark.parametrize(
  ("min_depth", "max_depth"),
  [(-0.1, 3.0), (0.2, 0.2), (1.0, 0.5)],
)
def test_normalized_camera_depth_rejects_invalid_ranges(
  min_depth: float,
  max_depth: float,
) -> None:
  env = SimpleNamespace(scene={})
  with pytest.raises(ValueError):
    normalized_camera_depth(
      env,  # type: ignore[arg-type]
      DEPTH_SENSOR_NAME,
      min_depth=min_depth,
      max_depth=max_depth,
    )
