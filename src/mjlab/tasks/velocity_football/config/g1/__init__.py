"""Active Unitree G1 velocity-football task registrations."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity_football.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_klavier_ball_temporal_flat_env_cfg,
  unitree_g1_klavier_legacy512_ball_temporal_flat_env_cfg,
  unitree_g1_long_dropout10_envelope30_legacy_curriculum_flat_env_cfg,
)
from .rl_cfg import (
  unitree_g1_factorial_ppo_runner_cfg,
  unitree_g1_klavier_ball_temporal_ppo_runner_cfg,
  unitree_g1_klavier_legacy512_ball_temporal_ppo_runner_cfg,
  unitree_g1_klavier_legacy512_walk_ppo_runner_cfg,
  unitree_g1_klavier_replica_ppo_runner_cfg,
  unitree_g1_legacy_xml_legacy512_walk_ppo_runner_cfg,
  unitree_g1_ppo_runner_cfg,
)
from .velocity_env_cfgs import (
  unitree_g1_klavier_legacy512_walk_flat_env_cfg,
  unitree_g1_klavier_replica_flat_env_cfg,
  unitree_g1_legacy_xml_legacy512_walk_flat_env_cfg,
)

BASE_TASK_ID = "Mjlab-Velocity-Football-Flat-Unitree-G1"
TEACHER_BASELINE_TASK_ID = (
  "Mjlab-Velocity-Football-A1R0-LongDropout10-Envelope30-"
  "LegacyCurriculum-Flat-Unitree-G1"
)
KLAVIER_WALK_TASK_ID = "Mjlab-Velocity-Walk-KlavierReplica-Flat-Unitree-G1"
KLAVIER_TEACHER_TASK_ID = (
  "Mjlab-Velocity-Football-KlavierReplica-BallTemporal-Flat-Unitree-G1"
)
KLAVIER_LEGACY512_WALK_TASK_ID = (
  "Mjlab-Velocity-Walk-KlavierReplica-Legacy512-LegacyPush-Flat-Unitree-G1"
)
LEGACY_XML_LEGACY512_WALK_TASK_ID = (
  "Mjlab-Velocity-Walk-LegacyXML-Legacy512-LegacyPush-Flat-Unitree-G1"
)
KLAVIER_LEGACY512_TEACHER_NOISE0_TASK_ID = (
  "Mjlab-Velocity-Football-KlavierReplica-Legacy512-NoPushCurr-"
  "BallNoise0-BallTemporal-Flat-Unitree-G1"
)
KLAVIER_LEGACY512_TEACHER_NOISE5CM_TASK_ID = (
  "Mjlab-Velocity-Football-KlavierReplica-Legacy512-NoPushCurr-"
  "BallNoise5cm-BallTemporal-Flat-Unitree-G1"
)
KLAVIER_LEGACY512_TEACHER_LEGACY_REWARDS_NOISE0_TASK_ID = (
  "Mjlab-Velocity-Football-KlavierReplica-Legacy512-NoPushCurr-"
  "LegacyRewards-BallNoise0-BallTemporal-Flat-Unitree-G1"
)
KLAVIER_LEGACY512_TEACHER_LEGACY_REWARDS_NOISE5CM_TASK_ID = (
  "Mjlab-Velocity-Football-KlavierReplica-Legacy512-NoPushCurr-"
  "LegacyRewards-BallNoise5cm-BallTemporal-Flat-Unitree-G1"
)


register_mjlab_task(
  task_id=BASE_TASK_ID,
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id=KLAVIER_WALK_TASK_ID,
  env_cfg=unitree_g1_klavier_replica_flat_env_cfg(),
  play_env_cfg=unitree_g1_klavier_replica_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_klavier_replica_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id=KLAVIER_TEACHER_TASK_ID,
  env_cfg=unitree_g1_klavier_ball_temporal_flat_env_cfg(),
  play_env_cfg=unitree_g1_klavier_ball_temporal_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_klavier_ball_temporal_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id=KLAVIER_LEGACY512_WALK_TASK_ID,
  env_cfg=unitree_g1_klavier_legacy512_walk_flat_env_cfg(),
  play_env_cfg=unitree_g1_klavier_legacy512_walk_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_klavier_legacy512_walk_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id=LEGACY_XML_LEGACY512_WALK_TASK_ID,
  env_cfg=unitree_g1_legacy_xml_legacy512_walk_flat_env_cfg(),
  play_env_cfg=unitree_g1_legacy_xml_legacy512_walk_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_legacy_xml_legacy512_walk_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)


def _register_legacy512_teacher(
  task_id: str,
  noise_meters: float,
  *,
  legacy_smoothness_rewards: bool = False,
) -> None:
  runner_cfg = unitree_g1_klavier_legacy512_ball_temporal_ppo_runner_cfg()
  noise_cm = int(round(noise_meters * 100.0))
  reward_label = "Envelope30_ActionAcc01_" if legacy_smoothness_rewards else ""
  runner_cfg.run_name = (
    f"KlavierLegacy512_NoPushCurr_{reward_label}BallNoise{noise_cm}cm_"
    "FromWalk20k_seed42_50k_wandb"
  )
  register_mjlab_task(
    task_id=task_id,
    env_cfg=unitree_g1_klavier_legacy512_ball_temporal_flat_env_cfg(
      ball_position_noise_meters=noise_meters,
      legacy_smoothness_rewards=legacy_smoothness_rewards,
    ),
    play_env_cfg=unitree_g1_klavier_legacy512_ball_temporal_flat_env_cfg(
      ball_position_noise_meters=noise_meters,
      legacy_smoothness_rewards=legacy_smoothness_rewards,
      play=True,
    ),
    rl_cfg=runner_cfg,
    runner_cls=VelocityOnPolicyRunner,
  )


_register_legacy512_teacher(KLAVIER_LEGACY512_TEACHER_NOISE0_TASK_ID, 0.0)
_register_legacy512_teacher(KLAVIER_LEGACY512_TEACHER_NOISE5CM_TASK_ID, 0.05)
_register_legacy512_teacher(
  KLAVIER_LEGACY512_TEACHER_LEGACY_REWARDS_NOISE0_TASK_ID,
  0.0,
  legacy_smoothness_rewards=True,
)
_register_legacy512_teacher(
  KLAVIER_LEGACY512_TEACHER_LEGACY_REWARDS_NOISE5CM_TASK_ID,
  0.05,
  legacy_smoothness_rewards=True,
)

register_mjlab_task(
  task_id=TEACHER_BASELINE_TASK_ID,
  env_cfg=unitree_g1_long_dropout10_envelope30_legacy_curriculum_flat_env_cfg(),
  play_env_cfg=(
    unitree_g1_long_dropout10_envelope30_legacy_curriculum_flat_env_cfg(play=True)
  ),
  rl_cfg=unitree_g1_factorial_ppo_runner_cfg(use_b1_history=True),
  runner_cls=VelocityOnPolicyRunner,
)
