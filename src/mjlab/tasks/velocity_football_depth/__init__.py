"""Active depth-football distillation task registrations."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import (
  unitree_g1_depth_klavier_mount_range_visual_dr_flat_env_cfg,
  unitree_g1_depth_klavier_no_push_curriculum_flat_env_cfg,
  unitree_g1_depth_klavier_visibility_supervised_flat_env_cfg,
  unitree_g1_depth_temporal_calibrated_visual_dr_flat_env_cfg,
  unitree_g1_depth_temporal_mount_range_strong_visual_dr_flat_env_cfg,
  unitree_g1_depth_temporal_mount_range_visual_dr_flat_env_cfg,
)
from .rl_cfg import (
  unitree_g1_depth_klavier_constrained_latent_runner_cfg,
  unitree_g1_depth_klavier_factorial_mixed_runner_cfg,
  unitree_g1_depth_klavier_frozen_latent_runner_cfg,
  unitree_g1_depth_klavier_visibility_constrained_runner_cfg,
  unitree_g1_depth_temporal_calibrated_frozen_mlp_runner_cfg,
  unitree_g1_depth_temporal_constrained_latent_runner_cfg,
)
from .runner import DepthTeacherDistillationRunner

DEPTH_BASELINE_TASK_ID = (
  "Mjlab-Velocity-Football-Depth-TemporalTeacher-MountRangeVisualDR-"
  "FrozenMLP-Distillation-Flat-Unitree-G1"
)
DEPTH_CALIBRATED_LEGACY_TASK_ID = (
  "Mjlab-Velocity-Football-Depth-TemporalTeacher-CalibratedVisualDR-"
  "FrozenMLP-Distillation-Flat-Unitree-G1"
)
DEPTH_CANDIDATE_TASK_ID = (
  "Mjlab-Velocity-Football-Depth-TemporalTeacher-MountRangeStrongVisualDR-"
  "ConstrainedMLP-Distillation-Flat-Unitree-G1"
)
DEPTH_KLAVIER_STAGE1_TASK_ID = (
  "Mjlab-Velocity-Football-Depth-KlavierTeacher-MountRangeVisualDR-"
  "FrozenMLP-LatentDistillation-Flat-Unitree-G1"
)
DEPTH_KLAVIER_STAGE2_TASK_ID = (
  "Mjlab-Velocity-Football-Depth-KlavierTeacher-MountRangeVisualDR-"
  "ConstrainedMLP-LatentDistillation-Flat-Unitree-G1"
)
DEPTH_KLAVIER_VISIBILITY_STAGE2_TASK_ID = (
  "Mjlab-Velocity-Football-Depth-KlavierTeacher-MountRangeVisualDR-"
  "ConstrainedMLP-LatentVisibilityDistillation-Flat-Unitree-G1"
)
DEPTH_KLAVIER_FACTORIAL_PUSH_OFF_NO_SYM_TASK_ID = (
  "Mjlab-Velocity-Football-Depth-KlavierTeacher-PushCurrOff-"
  "FrozenMLP-NoSym-LatentDistillation-Flat-Unitree-G1"
)
DEPTH_KLAVIER_FACTORIAL_PUSH_OFF_SYM_TASK_ID = (
  "Mjlab-Velocity-Football-Depth-KlavierTeacher-PushCurrOff-"
  "FrozenMLP-Sym-LatentDistillation-Flat-Unitree-G1"
)
DEPTH_KLAVIER_FACTORIAL_PUSH_ON_NO_SYM_TASK_ID = (
  "Mjlab-Velocity-Football-Depth-KlavierTeacher-PushCurrOn-"
  "FrozenMLP-NoSym-LatentDistillation-Flat-Unitree-G1"
)
DEPTH_KLAVIER_FACTORIAL_PUSH_ON_SYM_TASK_ID = (
  "Mjlab-Velocity-Football-Depth-KlavierTeacher-PushCurrOn-"
  "FrozenMLP-Sym-LatentDistillation-Flat-Unitree-G1"
)


register_mjlab_task(
  task_id=DEPTH_BASELINE_TASK_ID,
  env_cfg=unitree_g1_depth_temporal_mount_range_visual_dr_flat_env_cfg(),
  play_env_cfg=unitree_g1_depth_temporal_mount_range_visual_dr_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_depth_temporal_calibrated_frozen_mlp_runner_cfg(),
  runner_cls=DepthTeacherDistillationRunner,
)

register_mjlab_task(
  task_id=DEPTH_CALIBRATED_LEGACY_TASK_ID,
  env_cfg=unitree_g1_depth_temporal_calibrated_visual_dr_flat_env_cfg(),
  play_env_cfg=unitree_g1_depth_temporal_calibrated_visual_dr_flat_env_cfg(
    play=True
  ),
  rl_cfg=unitree_g1_depth_temporal_calibrated_frozen_mlp_runner_cfg(),
  runner_cls=DepthTeacherDistillationRunner,
)

register_mjlab_task(
  task_id=DEPTH_CANDIDATE_TASK_ID,
  env_cfg=unitree_g1_depth_temporal_mount_range_strong_visual_dr_flat_env_cfg(),
  play_env_cfg=unitree_g1_depth_temporal_mount_range_strong_visual_dr_flat_env_cfg(
    play=True
  ),
  rl_cfg=unitree_g1_depth_temporal_constrained_latent_runner_cfg(),
  runner_cls=DepthTeacherDistillationRunner,
)

register_mjlab_task(
  task_id=DEPTH_KLAVIER_STAGE1_TASK_ID,
  env_cfg=unitree_g1_depth_klavier_mount_range_visual_dr_flat_env_cfg(),
  play_env_cfg=unitree_g1_depth_klavier_mount_range_visual_dr_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_depth_klavier_frozen_latent_runner_cfg(),
  runner_cls=DepthTeacherDistillationRunner,
)

register_mjlab_task(
  task_id=DEPTH_KLAVIER_STAGE2_TASK_ID,
  env_cfg=unitree_g1_depth_klavier_mount_range_visual_dr_flat_env_cfg(),
  play_env_cfg=unitree_g1_depth_klavier_mount_range_visual_dr_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_depth_klavier_constrained_latent_runner_cfg(),
  runner_cls=DepthTeacherDistillationRunner,
)

register_mjlab_task(
  task_id=DEPTH_KLAVIER_VISIBILITY_STAGE2_TASK_ID,
  env_cfg=unitree_g1_depth_klavier_visibility_supervised_flat_env_cfg(),
  play_env_cfg=unitree_g1_depth_klavier_visibility_supervised_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_depth_klavier_visibility_constrained_runner_cfg(),
  runner_cls=DepthTeacherDistillationRunner,
)


def _register_frozen_factorial_task(
  task_id: str,
  *,
  push_curriculum: bool,
  mirror_loss: bool,
) -> None:
  env_factory = (
    unitree_g1_depth_klavier_mount_range_visual_dr_flat_env_cfg
    if push_curriculum
    else unitree_g1_depth_klavier_no_push_curriculum_flat_env_cfg
  )
  register_mjlab_task(
    task_id=task_id,
    env_cfg=env_factory(),
    play_env_cfg=env_factory(play=True),
    rl_cfg=unitree_g1_depth_klavier_factorial_mixed_runner_cfg(
      mirror_loss=mirror_loss
    ),
    runner_cls=DepthTeacherDistillationRunner,
  )


_register_frozen_factorial_task(
  DEPTH_KLAVIER_FACTORIAL_PUSH_OFF_NO_SYM_TASK_ID,
  push_curriculum=False,
  mirror_loss=False,
)
_register_frozen_factorial_task(
  DEPTH_KLAVIER_FACTORIAL_PUSH_OFF_SYM_TASK_ID,
  push_curriculum=False,
  mirror_loss=True,
)
_register_frozen_factorial_task(
  DEPTH_KLAVIER_FACTORIAL_PUSH_ON_NO_SYM_TASK_ID,
  push_curriculum=True,
  mirror_loss=False,
)
_register_frozen_factorial_task(
  DEPTH_KLAVIER_FACTORIAL_PUSH_ON_SYM_TASK_ID,
  push_curriculum=True,
  mirror_loss=True,
)
