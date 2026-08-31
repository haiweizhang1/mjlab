"""Registration contracts for the active coordinate-football tasks."""

from typing import Any, cast

import pytest

from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity_football.config.g1 import (
  BASE_TASK_ID,
  KLAVIER_LEGACY512_TEACHER_NOISE0_TASK_ID,
  KLAVIER_LEGACY512_TEACHER_NOISE5CM_TASK_ID,
  KLAVIER_LEGACY512_WALK_TASK_ID,
  KLAVIER_TEACHER_TASK_ID,
  KLAVIER_WALK_TASK_ID,
  TEACHER_BASELINE_TASK_ID,
)
from mjlab.tasks.velocity_football.rl import VelocityOnPolicyRunner


def test_only_expected_coordinate_football_tasks_are_registered() -> None:
  task_ids = {
    task_id
    for task_id in list_tasks()
    if task_id.startswith("Mjlab-Velocity-Football") and "-Depth-" not in task_id
  }
  assert task_ids == {
    BASE_TASK_ID,
    KLAVIER_LEGACY512_TEACHER_NOISE0_TASK_ID,
    KLAVIER_LEGACY512_TEACHER_NOISE5CM_TASK_ID,
    KLAVIER_TEACHER_TASK_ID,
    TEACHER_BASELINE_TASK_ID,
  }


@pytest.mark.parametrize(
  "task_id",
  (
    BASE_TASK_ID,
    KLAVIER_TEACHER_TASK_ID,
    KLAVIER_LEGACY512_TEACHER_NOISE0_TASK_ID,
    KLAVIER_LEGACY512_TEACHER_NOISE5CM_TASK_ID,
    TEACHER_BASELINE_TASK_ID,
  ),
)
def test_active_coordinate_tasks_load(task_id: str) -> None:
  training_cfg = load_env_cfg(task_id)
  play_cfg = load_env_cfg(task_id, play=True)

  assert training_cfg.scene.num_envs >= 1
  assert play_cfg.scene.num_envs == 1
  assert load_runner_cls(task_id) is VelocityOnPolicyRunner


def test_teacher_baseline_observation_and_policy_contract() -> None:
  cfg = load_env_cfg(TEACHER_BASELINE_TASK_ID)
  runner_cfg = cast(Any, load_rl_cfg(TEACHER_BASELINE_TASK_ID))

  actor = cfg.observations["actor"]
  actor_history = cfg.observations["actor_history"]
  assert actor.history_length == 5
  assert actor_history.history_length == 10
  assert tuple(actor_history.terms) == (
    "ball_pos_b",
    "ball_to_feet_vectors_b",
    "ball_visible_mask",
  )
  assert (
    sum(
      term.params.get("transition_dropout_probability", 0.0) > 0.0
      for term in actor_history.terms.values()
    )
    == 3
  )
  assert runner_cfg.actor.hidden_dims == (512, 256, 128)
  assert runner_cfg.critic.hidden_dims == (512, 256, 128)
  assert runner_cfg.obs_groups["actor"] == ("actor", "actor_history")
  assert runner_cfg.algorithm.entropy_coef == pytest.approx(0.01)


def test_base_task_remains_a_simple_smoke_baseline() -> None:
  cfg = load_env_cfg(BASE_TASK_ID)
  runner_cfg = cast(Any, load_rl_cfg(BASE_TASK_ID))

  assert "actor" in cfg.observations
  assert "critic" in cfg.observations
  assert runner_cfg.actor.hidden_dims == (512, 256, 128)


def test_klavier_walk_retraining_contract() -> None:
  cfg = load_env_cfg(KLAVIER_WALK_TASK_ID)
  runner_cfg = cast(Any, load_rl_cfg(KLAVIER_WALK_TASK_ID))

  assert cfg.scene.num_envs == 4096
  assert cfg.observations["actor"].history_length == 5
  assert runner_cfg.actor.hidden_dims == (512, 256, 128)
  assert runner_cfg.critic.hidden_dims == (512, 256, 128)
  assert runner_cfg.algorithm.entropy_coef == pytest.approx(0.005)
  assert runner_cfg.algorithm.symmetry_cfg is None
  assert runner_cfg.save_interval == 1000
  assert runner_cfg.max_iterations == 20_001


def test_klavier_teacher_walk_transfer_and_symmetry_contract() -> None:
  cfg = load_env_cfg(KLAVIER_TEACHER_TASK_ID)
  runner_cfg = cast(Any, load_rl_cfg(KLAVIER_TEACHER_TASK_ID))

  assert cfg.scene.num_envs == 4096
  assert cfg.observations["actor"].history_length == 5
  assert cfg.observations["actor_history"].history_length == 10
  assert tuple(cfg.observations["actor_history"].terms) == ("ball_features_b",)
  assert "command_velocity_envelope" not in cfg.rewards
  assert "action_acc_l2" not in cfg.rewards
  push_max = cfg.curriculum["push_velocity_levels"].params["max_velocity_range"]
  assert max(abs(bound) for bounds in push_max.values() for bound in bounds) == 1.0
  assert runner_cfg.actor.hidden_dims == (1024, 512, 256)
  assert runner_cfg.critic.hidden_dims == (1024, 512, 256)
  assert runner_cfg.obs_groups["actor"] == ("actor", "actor_history")
  assert runner_cfg.algorithm.entropy_coef == pytest.approx(0.01)
  assert runner_cfg.algorithm.symmetry_cfg == {
    "data_augmentation_func": (
      "mjlab.tasks.velocity_football.rl.klavier_symmetry:data_augmentation_func"
    ),
    "use_data_augmentation": False,
    "use_mirror_loss": True,
    "mirror_loss_coeff": 1.0,
  }
  assert runner_cfg.max_iterations == 50_001


def test_klavier_legacy512_walk_contract() -> None:
  cfg = load_env_cfg(KLAVIER_LEGACY512_WALK_TASK_ID)
  runner_cfg = cast(Any, load_rl_cfg(KLAVIER_LEGACY512_WALK_TASK_ID))
  push = cfg.events["push_robot"]

  assert "push_velocity_levels" not in cfg.curriculum
  assert push.interval_range_s == (5.0, 6.0)
  assert push.params["velocity_range"] == {
    "x": (-0.5, 0.5),
    "y": (-0.3, 0.3),
    "z": (-0.2, 0.2),
    "roll": (-0.1, 0.1),
    "pitch": (-0.1, 0.1),
    "yaw": (-0.2, 0.2),
  }
  assert runner_cfg.actor.hidden_dims == (512, 256, 128)
  assert runner_cfg.critic.hidden_dims == (512, 256, 128)
  assert runner_cfg.algorithm.symmetry_cfg is None
  assert runner_cfg.max_iterations == 20_001
  assert runner_cfg.save_interval == 1_000


@pytest.mark.parametrize(
  ("task_id", "expected_noise"),
  (
    (KLAVIER_LEGACY512_TEACHER_NOISE0_TASK_ID, 0.0),
    (KLAVIER_LEGACY512_TEACHER_NOISE5CM_TASK_ID, 0.05),
  ),
)
def test_klavier_legacy512_teacher_pair_contract(
  task_id: str, expected_noise: float
) -> None:
  cfg = load_env_cfg(task_id)
  play_cfg = load_env_cfg(task_id, play=True)
  runner_cfg = cast(Any, load_rl_cfg(task_id))
  push = cfg.events["push_robot"]
  ball = cfg.observations["actor_history"].terms["ball_features_b"]
  play_ball = play_cfg.observations["actor_history"].terms["ball_features_b"]

  assert "push_velocity_levels" not in cfg.curriculum
  assert push.interval_range_s == (5.0, 6.0)
  assert push.params["velocity_range"]["x"] == (-0.5, 0.5)
  assert ball.params["frame_noise_range"] == pytest.approx(expected_noise)
  assert play_ball.params["frame_noise_range"] == pytest.approx(0.0)
  assert ball.params["bias_range"] == pytest.approx(0.0)
  assert ball.params["dropout_probability"] == pytest.approx(0.0)
  assert (ball.delay_min_lag, ball.delay_max_lag) == (0, 0)
  assert ball.noise is None
  assert runner_cfg.actor.hidden_dims == (512, 256, 128)
  assert runner_cfg.critic.hidden_dims == (512, 256, 128)
  assert runner_cfg.algorithm.symmetry_cfg is None
  assert runner_cfg.max_iterations == 50_001
  assert runner_cfg.save_interval == 1_000
