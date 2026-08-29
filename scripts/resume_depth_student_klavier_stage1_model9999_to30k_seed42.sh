#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/ut/football_project/mjlab_soccer"
source_run="2026-08-28_15-05-50_DepthStudent_KlavierTeacher47000_FrozenMLP_Latent01_SyncDelay02_MountRange025_seed42_10k_wandb"
source_checkpoint="model_9999.pt"
checkpoint="$repo_dir/logs/rsl_rl/g1_velocity_football_depth_temporal_distillation/$source_run/$source_checkpoint"
run_name="DepthStudent_KlavierTeacher47000_FrozenMLP_Latent01_SyncDelay02_MountRange025_seed42_resume10k_to30k_wandb"

if [[ ! -f "$checkpoint" ]]; then
  echo "ERROR: missing Student checkpoint: $checkpoint" >&2
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
  --env.scene.num-envs 4096 \
  --agent.seed 42 \
  --agent.resume True \
  --agent.load-run "$source_run" \
  --agent.load-checkpoint "$source_checkpoint" \
  --agent.algorithm.rollout-policy teacher \
  --agent.algorithm.latent-loss-coef 0.1 \
  --agent.max-iterations 20000 \
  --agent.save-interval 500 \
  --agent.logger wandb \
  --agent.wandb-project mjlab \
  --agent.upload-model False \
  --agent.run-name "$run_name"
