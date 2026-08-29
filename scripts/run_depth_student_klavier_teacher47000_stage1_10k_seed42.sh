#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/ut/football_project/mjlab_soccer"
teacher_checkpoint="$repo_dir/logs/rsl_rl/g1_velocity_football_klavier_ball_temporal/2026-08-27_08-25-54_schemeA_zeroBallNoise_syncDelay02_entropy001_symmetry_seed42_50k/model_47000.pt"
run_name="DepthStudent_KlavierTeacher47000_FrozenMLP_Latent01_SyncDelay02_MountRange025_seed42_10k_wandb"

if [[ ! -f "$teacher_checkpoint" ]]; then
  echo "ERROR: missing Klavier Teacher: $teacher_checkpoint" >&2
  exit 1
fi

cd "$repo_dir"
export UV_CACHE_DIR=/tmp/mjlab-uv-cache
export WARP_CACHE_PATH=/tmp/mjlab-warp
export MPLCONFIGDIR=/tmp/mjlab-mpl
export XDG_CACHE_HOME=/tmp/mjlab-xdg
export PYTORCH_ALLOC_CONF=expandable_segments:True

exec uv run train \
  Mjlab-Velocity-Football-Depth-KlavierTeacher-MountRangeVisualDR-FrozenMLP-LatentDistillation-Flat-Unitree-G1 \
  --pretrained-checkpoint "$teacher_checkpoint" \
  --env.scene.num-envs 4096 \
  --agent.seed 42 \
  --agent.algorithm.rollout-policy teacher \
  --agent.algorithm.latent-loss-coef 0.1 \
  --agent.max-iterations 10000 \
  --agent.save-interval 500 \
  --agent.logger wandb \
  --agent.wandb-project mjlab \
  --agent.upload-model False \
  --agent.run-name "$run_name"
