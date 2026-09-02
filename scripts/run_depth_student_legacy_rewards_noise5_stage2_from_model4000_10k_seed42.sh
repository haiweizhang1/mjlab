#!/usr/bin/env bash
set -euo pipefail

repo_dir="${MJLAB_SOCCER_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
train_entrypoint="$repo_dir/.venv/bin/train"
experiment_root="$repo_dir/logs/rsl_rl/g1_velocity_football_depth_temporal_distillation"
source_run="${SOURCE_RUN:-2026-09-02_02-40-41_DepthStudent_KlavierLegacyRewardsNoise5Teacher25000_FrozenMLP_TeacherRollout_ActionHuberOnly_LegacyStage1DR_NoDelay_MountRange025_seed42_10k_wandb}"
source_checkpoint="${SOURCE_CHECKPOINT:-model_4000.pt}"
checkpoint="$experiment_root/$source_run/$source_checkpoint"
num_envs="${NUM_ENVS:-4096}"
max_iterations="${MAX_ITERATIONS:-10000}"
run_name="${RUN_NAME:-DepthStudent_KlavierLegacyRewardsNoise5Teacher25000_ConstrainedLastMLP_Mixed030_Latent01_MLPAnchor001_LegacyStage1DR_NoDelay_seed42_from4000_stage2_10k_wandb}"

if [[ ! -f "$checkpoint" ]]; then
  echo "ERROR: missing Stage 1 checkpoint: $checkpoint" >&2
  echo "Set MJLAB_SOCCER_DIR, SOURCE_RUN, and SOURCE_CHECKPOINT if the source differs." >&2
  exit 1
fi
if [[ ! -x "$train_entrypoint" ]]; then
  echo "ERROR: missing project train entrypoint: $train_entrypoint" >&2
  exit 1
fi

cd "$repo_dir"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/mjlab-uv-cache}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/mjlab-warp-depth-legacy-rewards-stage2}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mjlab-mpl-depth-legacy-rewards-stage2}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/mjlab-xdg-depth-legacy-rewards-stage2}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

exec "$train_entrypoint" \
  Mjlab-Velocity-Football-Depth-KlavierLegacyRewardsNoise5Teacher-LegacyStage1DR-ConstrainedLastMLP-NoSym-LatentDistillation-Flat-Unitree-G1 \
  --env.scene.num-envs "$num_envs" \
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
  --agent.max-iterations "$max_iterations" \
  --agent.save-interval 1000 \
  --agent.logger wandb \
  --agent.wandb-project mjlab \
  --agent.upload-model False \
  --agent.run-name "$run_name"
