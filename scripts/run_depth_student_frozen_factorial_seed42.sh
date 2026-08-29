#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {e1|e2|e3|e4}" >&2
  exit 2
fi

variant="$1"
repo_dir="${MJLAB_SOCCER_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
teacher_checkpoint="$repo_dir/checkpoints/football_2026-08-29/teacher_klavier_model_47000.pt"

case "$variant" in
  e1)
    task_id="Mjlab-Velocity-Football-Depth-KlavierTeacher-PushCurrOff-FrozenMLP-NoSym-LatentDistillation-Flat-Unitree-G1"
    run_name="E1_PushCurrOff_FrozenMLP_NoSym_Teacher47000_seed42_10k_wandb"
    ;;
  e2)
    task_id="Mjlab-Velocity-Football-Depth-KlavierTeacher-PushCurrOff-FrozenMLP-Sym-LatentDistillation-Flat-Unitree-G1"
    run_name="E2_PushCurrOff_FrozenMLP_Sym1_Teacher47000_seed42_10k_wandb"
    ;;
  e3)
    task_id="Mjlab-Velocity-Football-Depth-KlavierTeacher-PushCurrOn-FrozenMLP-NoSym-LatentDistillation-Flat-Unitree-G1"
    run_name="E3_PushCurrOn_FrozenMLP_NoSym_Teacher47000_seed42_10k_wandb"
    ;;
  e4)
    task_id="Mjlab-Velocity-Football-Depth-KlavierTeacher-PushCurrOn-FrozenMLP-Sym-LatentDistillation-Flat-Unitree-G1"
    run_name="E4_PushCurrOn_FrozenMLP_Sym1_Teacher47000_seed42_10k_wandb"
    ;;
  *)
    echo "ERROR: unknown variant '$variant'; expected e1, e2, e3, or e4" >&2
    exit 2
    ;;
esac

if [[ ! -f "$teacher_checkpoint" ]]; then
  echo "ERROR: missing Teacher checkpoint: $teacher_checkpoint" >&2
  exit 1
fi

num_envs="${NUM_ENVS:-4096}"
cd "$repo_dir"
export UV_CACHE_DIR="/tmp/mjlab-uv-cache-$variant"
export WARP_CACHE_PATH="/tmp/mjlab-warp-$variant"
export MPLCONFIGDIR="/tmp/mjlab-mpl-$variant"
export XDG_CACHE_HOME="/tmp/mjlab-xdg-$variant"
export PYTORCH_ALLOC_CONF=expandable_segments:True

exec uv run train \
  "$task_id" \
  --pretrained-checkpoint "$teacher_checkpoint" \
  --env.scene.num-envs "$num_envs" \
  --agent.seed 42 \
  --agent.algorithm.rollout-policy teacher \
  --agent.algorithm.latent-loss-coef 0.1 \
  --agent.max-iterations 10000 \
  --agent.save-interval 500 \
  --agent.logger wandb \
  --agent.wandb-project mjlab \
  --agent.upload-model False \
  --agent.run-name "$run_name"
