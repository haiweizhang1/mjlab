#!/usr/bin/env bash
set -euo pipefail

repo_dir="${MJLAB_SOCCER_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
train_entrypoint="$repo_dir/.venv/bin/train"
teacher_run="${TEACHER_RUN:-2026-09-01_15-44-14_KlavierLegacy512_LegacyRewards_BallNoise5cm_FromZero_seed42_50k_save1k_wandb}"
teacher_checkpoint="${TEACHER_CHECKPOINT:-$repo_dir/logs/rsl_rl/g1_velocity_football_klavier_legacy512_ball_temporal/$teacher_run/model_25000.pt}"
num_envs="${NUM_ENVS:-4096}"

if [[ ! -f "$teacher_checkpoint" ]]; then
  echo "ERROR: missing LegacyRewards Teacher checkpoint: $teacher_checkpoint" >&2
  exit 1
fi
if [[ ! -x "$train_entrypoint" ]]; then
  echo "ERROR: missing project train entrypoint: $train_entrypoint" >&2
  exit 1
fi

cd "$repo_dir"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/mjlab-uv-cache}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/mjlab-warp-depth-legacy-rewards-stage1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mjlab-mpl-depth-legacy-rewards-stage1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/mjlab-xdg-depth-legacy-rewards-stage1}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

exec "$train_entrypoint" \
  Mjlab-Velocity-Football-Depth-KlavierLegacyRewardsNoise5Teacher-LegacyStage1DR-FrozenMLP-NoSym-ActionOnlyDistillation-Flat-Unitree-G1 \
  --pretrained-checkpoint "$teacher_checkpoint" \
  --env.scene.num-envs "$num_envs" \
  --agent.seed 42 \
  --agent.max-iterations 10000 \
  --agent.save-interval 1000 \
  --agent.logger wandb \
  --agent.wandb-project mjlab \
  --agent.upload-model False \
  --agent.run-name DepthStudent_KlavierLegacyRewardsNoise5Teacher25000_FrozenMLP_TeacherRollout_ActionHuberOnly_LegacyStage1DR_NoDelay_MountRange025_seed42_10k_wandb
