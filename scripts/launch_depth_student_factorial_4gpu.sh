#!/usr/bin/env bash
set -euo pipefail

repo_dir="${MJLAB_SOCCER_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
runner="$repo_dir/scripts/run_depth_student_frozen_factorial_seed42.sh"
num_envs="${NUM_ENVS:-4096}"
gpu_ids_raw="${GPU_IDS:-0 1 2 3}"
read -r -a gpu_ids <<< "$gpu_ids_raw"
variants=(e1 e2 e3 e4)
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/mjlab-uv-cache}"
launch_stamp="$(date +%Y%m%d_%H%M%S)"
launch_log_dir="$repo_dir/logs/launch/depth_factorial_$launch_stamp"

if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: tmux is required for the four-job launcher" >&2
  exit 1
fi
if [[ ${#gpu_ids[@]} -ne 4 ]]; then
  echo "ERROR: GPU_IDS must contain exactly four GPU indices" >&2
  exit 2
fi

echo "Preparing the shared uv environment once before launching four jobs..."
cd "$repo_dir"
uv sync --locked
mkdir -p "$launch_log_dir"

for index in "${!variants[@]}"; do
  variant="${variants[$index]}"
  session="depth_${variant}"
  gpu="${gpu_ids[$index]}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "ERROR: tmux session '$session' already exists" >&2
    exit 1
  fi
  log_file="$launch_log_dir/$variant.log"
  tmux new-session -d -s "$session" -c "$repo_dir"
  tmux send-keys -t "$session" \
    "CUDA_VISIBLE_DEVICES='$gpu' NUM_ENVS='$num_envs' bash '$runner' '$variant' 2>&1 | tee '$log_file'" \
    C-m
  echo "Started $variant in tmux session $session on GPU $gpu ($num_envs envs); log: $log_file"
done

echo "Use 'tmux ls' to list jobs and 'tmux attach -t depth_e1' to inspect one."
echo "Persistent launch logs: $launch_log_dir"
