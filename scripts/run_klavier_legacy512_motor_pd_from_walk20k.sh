#!/usr/bin/env bash
set -euo pipefail

repo_dir="${MJLAB_SOCCER_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
train_entrypoint="$repo_dir/.venv/bin/train"
num_envs="${NUM_ENVS:-4096}"

task_id="Mjlab-Velocity-Football-KlavierReplica-Legacy512-MotorPD-IdealPd-NoPushCurr-LegacyRewards-BallNoise0-BallTemporal-Flat-Unitree-G1"
run_name="KlavierLegacy512_MotorPD_IdealPd_Envelope30_ActionAcc01_BallNoise0cm_FromWalk20k_seed42_30k_wandb"

default_walk_checkpoint="$repo_dir/../log_old/logs/rsl_rl/g1_velocity_football_pretrain/2026-07-22_10-38-13/model_20000.pt"
walk_checkpoint="${WALK_CHECKPOINT:-$default_walk_checkpoint}"

if [[ ! -f "$walk_checkpoint" ]]; then
  echo "ERROR: missing Legacy512 Walk checkpoint: $walk_checkpoint" >&2
  echo "Set WALK_CHECKPOINT=/absolute/path/to/model_20000.pt" >&2
  exit 1
fi
if [[ ! -x "$train_entrypoint" ]]; then
  echo "ERROR: missing project train entrypoint: $train_entrypoint" >&2
  exit 1
fi

cd "$repo_dir"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/mjlab-uv-cache}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/mjlab-warp-motorpd-teacher}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mjlab-mpl-motorpd-teacher}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/mjlab-xdg-motorpd-teacher}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

exec "$train_entrypoint" \
  "$task_id" \
  --pretrained-checkpoint "$walk_checkpoint" \
  --env.scene.num-envs "$num_envs" \
  --agent.seed 42 \
  --agent.max-iterations 30000 \
  --agent.save-interval 1000 \
  --agent.logger wandb \
  --agent.wandb-project mjlab \
  --agent.upload-model False \
  --agent.run-name "$run_name"
