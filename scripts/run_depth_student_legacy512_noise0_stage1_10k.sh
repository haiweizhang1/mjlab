#!/usr/bin/env bash
set -euo pipefail

repo_dir="${MJLAB_SOCCER_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
train_entrypoint="$repo_dir/.venv/bin/train"
teacher_run="${TEACHER_RUN:-2026-08-31_09-59-44_KlavierLegacy512_NoPushCurr_BallNoise0_FromWalk20000_seed42_50k_save2k_wandb}"
teacher_checkpoint="${TEACHER_CHECKPOINT:-$repo_dir/logs/rsl_rl/g1_velocity_football_klavier_legacy512_ball_temporal/$teacher_run/model_50000.pt}"
num_envs="${NUM_ENVS:-4096}"

if [[ ! -f "$teacher_checkpoint" ]]; then
  echo "ERROR: missing frozen Teacher checkpoint: $teacher_checkpoint" >&2
  exit 1
fi
if [[ ! -x "$train_entrypoint" ]]; then
  echo "ERROR: missing project train entrypoint: $train_entrypoint" >&2
  exit 1
fi

cd "$repo_dir"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/mjlab-uv-cache}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/mjlab-warp-depth-legacy512-stage1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mjlab-mpl-depth-legacy512-stage1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/mjlab-xdg-depth-legacy512-stage1}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

exec "$train_entrypoint" \
  Mjlab-Velocity-Football-Depth-KlavierLegacy512Noise0Teacher-PushCurrOff-FrozenMLP-NoSym-LatentDistillation-Flat-Unitree-G1 \
  --pretrained-checkpoint "$teacher_checkpoint" \
  --env.scene.num-envs "$num_envs" \
  --agent.seed 42 \
  --agent.max-iterations 10000 \
  --agent.save-interval 1000 \
  --agent.logger wandb \
  --agent.wandb-project mjlab \
  --agent.upload-model False \
  --agent.run-name DepthStudent_KlavierLegacy512_Noise0Teacher50000_FrozenMLP_TeacherRollout_Latent01_NoDelay_MountRange025_seed42_10k_wandb
