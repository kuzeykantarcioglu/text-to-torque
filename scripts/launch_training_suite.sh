#!/usr/bin/env bash
set -euo pipefail

PROMPT_FILE="${1:-prompts/prompt_suite.csv}"
MAX_ITERATIONS="${MAX_ITERATIONS:-2000}"
NUM_ENVS="${NUM_ENVS:-1024}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
TRAIN_VIDEO_INTERVAL="${TRAIN_VIDEO_INTERVAL:-50000}"
TRAIN_VIDEO_LENGTH="${TRAIN_VIDEO_LENGTH:-120}"
ARTIFACT_SYNC_INTERVAL_S="${ARTIFACT_SYNC_INTERVAL_S:-60}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-50}"
OUTPUT_FPS="${OUTPUT_FPS:-50}"
PROMPT_IDS="${PROMPT_IDS:-walk_forward,wave_right,tap_head,squat_stand}"
TERMINATION_MODE="${TERMINATION_MODE:-loose}"
RUN_LABEL="${RUN_LABEL:-${TERMINATION_MODE}}"
LAUNCH_LOG_DIR="${LAUNCH_LOG_DIR:-logs/modal_launches}"

if ! command -v modal >/dev/null 2>&1; then
  echo "modal is not on PATH. Activate the text-to-torque environment first." >&2
  exit 1
fi

case "$TERMINATION_MODE" in
  loose)
    TERMINATION_FLAG="--disable-terminations"
    ;;
  strict)
    TERMINATION_FLAG="--no-disable-terminations"
    ;;
  *)
    echo "TERMINATION_MODE must be loose or strict, got: ${TERMINATION_MODE}" >&2
    exit 1
    ;;
esac

IFS=, read -ra SELECTED_IDS <<< "$PROMPT_IDS"
mkdir -p "$LAUNCH_LOG_DIR"

is_selected() {
  local candidate="$1"
  for selected in "${SELECTED_IDS[@]}"; do
    if [[ "$candidate" == "$selected" ]]; then
      return 0
    fi
  done
  return 1
}

while IFS=, read -r prompt_id difficulty prompt duration seed notes; do
  if ! is_selected "$prompt_id"; then
    continue
  fi

  echo "Launching ${TERMINATION_MODE} training for ${prompt_id}: ${prompt}"
  nohup modal run --detach modal_kimolab.py \
      --prompt "$prompt" \
      --run-label "$RUN_LABEL" \
      --duration "$duration" \
      --seed "$seed" \
      --diffusion-steps "$DIFFUSION_STEPS" \
      --output-fps "$OUTPUT_FPS" \
      --no-render-reference \
      --train \
      --num-envs "$NUM_ENVS" \
      --max-iterations "$MAX_ITERATIONS" \
      --save-interval "$SAVE_INTERVAL" \
      "$TERMINATION_FLAG" \
      --record-train-video \
      --train-video-interval "$TRAIN_VIDEO_INTERVAL" \
      --train-video-length "$TRAIN_VIDEO_LENGTH" \
      --artifact-sync-interval-s "$ARTIFACT_SYNC_INTERVAL_S" \
      --no-spawn \
    > "${LAUNCH_LOG_DIR}/train_${TERMINATION_MODE}_${prompt_id}.log" 2>&1 &
done < <(tail -n +2 "$PROMPT_FILE")

echo "Queued training jobs. Logs are in ${LAUNCH_LOG_DIR}/train_${TERMINATION_MODE}_<prompt_id>.log"
