#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {noise0|noise5cm}" >&2
  exit 2
fi

variant="$1"
repo_dir="${MJLAB_SOCCER_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
train_entrypoint="$repo_dir/.venv/bin/train"
num_envs="${NUM_ENVS:-4096}"

case "$variant" in
  noise0)
    task_id="Mjlab-Velocity-Football-KlavierReplica-Legacy512-NoPushCurr-BallNoise0-BallTemporal-Flat-Unitree-G1"
    run_name="KlavierLegacy512_NoPushCurr_BallNoise0cm_FromWalk20k_seed42_50k_wandb"
    ;;
  noise5cm)
    task_id="Mjlab-Velocity-Football-KlavierReplica-Legacy512-NoPushCurr-BallNoise5cm-BallTemporal-Flat-Unitree-G1"
    run_name="KlavierLegacy512_NoPushCurr_BallNoise5cm_FromWalk20k_seed42_50k_wandb"
    ;;
  *)
    echo "ERROR: unknown variant '$variant'; expected noise0 or noise5cm" >&2
    exit 2
    ;;
esac

walk_checkpoint="${WALK_CHECKPOINT:-}"
if [[ -z "$walk_checkpoint" ]]; then
  shopt -s nullglob
  candidates=(
    "$repo_dir/logs/rsl_rl/g1_velocity_walk_klavier_legacy512"/*/model_20000.pt
  )
  shopt -u nullglob
  if [[ ${#candidates[@]} -eq 0 ]]; then
    echo "ERROR: no Walk model_20000.pt found; set WALK_CHECKPOINT explicitly." >&2
    exit 1
  fi
  walk_checkpoint="${candidates[-1]}"
fi

if [[ ! -f "$walk_checkpoint" ]]; then
  echo "ERROR: missing Walk checkpoint: $walk_checkpoint" >&2
  exit 1
fi
if [[ ! -x "$train_entrypoint" ]]; then
  echo "ERROR: missing project train entrypoint: $train_entrypoint" >&2
  exit 1
fi

cd "$repo_dir"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/mjlab-uv-cache}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/mjlab-warp-teacher512-$variant}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mjlab-mpl-teacher512-$variant}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/mjlab-xdg-teacher512-$variant}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

exec "$train_entrypoint" \
  "$task_id" \
  --pretrained-checkpoint "$walk_checkpoint" \
  --env.scene.num-envs "$num_envs" \
  --agent.seed 42 \
  --agent.max-iterations 50001 \
  --agent.save-interval 1000 \
  --agent.logger wandb \
  --agent.wandb-project mjlab \
  --agent.upload-model False \
  --agent.run-name "$run_name"
