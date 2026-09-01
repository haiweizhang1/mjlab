#!/usr/bin/env bash
set -euo pipefail

repo_dir="${MJLAB_SOCCER_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
train_entrypoint="$repo_dir/.venv/bin/train"
source_run="${SOURCE_RUN:-2026-08-31_09-59-44_KlavierLegacy512_NoPushCurr_BallNoise0_FromWalk20000_seed42_50k_save2k_wandb}"
source_checkpoint="${SOURCE_CHECKPOINT:-model_50000.pt}"
checkpoint="$repo_dir/logs/rsl_rl/g1_velocity_football_klavier_legacy512_ball_temporal/$source_run/$source_checkpoint"
num_envs="${NUM_ENVS:-4096}"

if [[ ! -f "$checkpoint" ]]; then
  echo "ERROR: missing Teacher checkpoint: $checkpoint" >&2
  exit 1
fi
if [[ ! -x "$train_entrypoint" ]]; then
  echo "ERROR: missing project train entrypoint: $train_entrypoint" >&2
  exit 1
fi

cd "$repo_dir"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/mjlab-uv-cache}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/mjlab-warp-teacher-noise0-80k}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mjlab-mpl-teacher-noise0-80k}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/mjlab-xdg-teacher-noise0-80k}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

exec "$train_entrypoint" \
  Mjlab-Velocity-Football-KlavierReplica-Legacy512-NoPushCurr-BallNoise0-BallTemporal-Flat-Unitree-G1 \
  --env.scene.num-envs "$num_envs" \
  --agent.seed 42 \
  --agent.resume True \
  --agent.load-run "$source_run" \
  --agent.load-checkpoint "$source_checkpoint" \
  --agent.max-iterations 30000 \
  --agent.save-interval 2000 \
  --agent.logger wandb \
  --agent.wandb-project mjlab \
  --agent.upload-model False \
  --agent.run-name KlavierLegacy512_NoPushCurr_BallNoise0_seed42_resume50k_to80k_save2k_wandb
