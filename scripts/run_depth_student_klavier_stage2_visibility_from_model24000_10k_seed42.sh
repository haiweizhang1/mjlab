#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/ut/football_project/mjlab_soccer"
source_run="2026-08-28_23-14-16_DepthStudent_KlavierTeacher47000_FrozenMLP_Latent01_SyncDelay02_MountRange025_seed42_resume10k_to30k_wandb"
source_checkpoint="model_24000.pt"
checkpoint="$repo_dir/logs/rsl_rl/g1_velocity_football_depth_temporal_distillation/$source_run/$source_checkpoint"
run_name="DepthStudent_KlavierTeacher47000_ConstrainedLastMLP_Latent01_VisibilityBCE02_Mixed030_SyncDelay02_MountRange025_seed42_from24000_stage2_10k_wandb"

if [[ ! -f "$checkpoint" ]]; then
  echo "ERROR: missing Stage-one Student checkpoint: $checkpoint" >&2
  exit 1
fi

cd "$repo_dir"
export UV_CACHE_DIR=/tmp/mjlab-uv-cache
export WARP_CACHE_PATH=/tmp/mjlab-warp
export MPLCONFIGDIR=/tmp/mjlab-mpl
export XDG_CACHE_HOME=/tmp/mjlab-xdg
export PYTORCH_ALLOC_CONF=expandable_segments:True

exec uv run train \
  Mjlab-Velocity-Football-Depth-KlavierTeacher-MountRangeVisualDR-ConstrainedMLP-LatentVisibilityDistillation-Flat-Unitree-G1 \
  --env.scene.num-envs 4096 \
  --agent.seed 42 \
  --agent.resume True \
  --agent.load-run "$source_run" \
  --agent.load-checkpoint "$source_checkpoint" \
  --agent.algorithm.rollout-policy mixed \
  --agent.algorithm.student-rollout-warmup-updates 0 \
  --agent.algorithm.student-rollout-ramp-updates 2000 \
  --agent.algorithm.student-rollout-final-probability 0.3 \
  --agent.algorithm.learning-rate 0.0003 \
  --agent.algorithm.mlp-learning-rate 0.00001 \
  --agent.algorithm.latent-loss-coef 0.1 \
  --agent.algorithm.mlp-anchor-loss-coef 0.001 \
  --agent.algorithm.visibility-loss-coef 0.2 \
  --agent.algorithm.visibility-target-group depth_visibility_target \
  --agent.max-iterations 10000 \
  --agent.save-interval 500 \
  --agent.logger wandb \
  --agent.wandb-project mjlab \
  --agent.upload-model False \
  --agent.run-name "$run_name"
