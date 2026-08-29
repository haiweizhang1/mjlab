#!/usr/bin/env bash
set -euo pipefail

cd /home/ut/football_project/mjlab_soccer

export UV_CACHE_DIR=/tmp/uv-cache
export WARP_CACHE_PATH=/tmp/warp-cache
export MPLCONFIGDIR=/tmp/matplotlib-cache
export XDG_CACHE_HOME=/tmp/xdg-cache

exec env -u VIRTUAL_ENV uv run train \
  Mjlab-Velocity-Walk-KlavierReplica-Flat-Unitree-G1 \
  --env.scene.num-envs 4096 \
  --agent.seed 42 \
  --agent.max-iterations 20001 \
  --agent.save-interval 1000 \
  --agent.logger wandb \
  --agent.wandb-project mjlab \
  --agent.upload-model False \
  --agent.run-name unitree_g1_flat_mlp512_noMirrorLoss_seed42_20k_wandb
