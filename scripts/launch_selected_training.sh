#!/usr/bin/env bash
set -euo pipefail

SELECTED_FILE="${1:-results/selected_references.csv}"
SELECTIONS="${SELECTIONS:-best,worst}"
MOTION_IDS="${MOTION_IDS:-squat_stand,jump,roll}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1200}"
NUM_ENVS="${NUM_ENVS:-1024}"
SAVE_INTERVAL="${SAVE_INTERVAL:-400}"
TRAIN_VIDEO_INTERVAL="${TRAIN_VIDEO_INTERVAL:-10000}"
TRAIN_VIDEO_LENGTH="${TRAIN_VIDEO_LENGTH:-250}"
ARTIFACT_SYNC_INTERVAL_S="${ARTIFACT_SYNC_INTERVAL_S:-60}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-50}"
OUTPUT_FPS="${OUTPUT_FPS:-50}"
TERMINATION_MODE="${TERMINATION_MODE:-strict}"
REFERENCE_TIME_SCALE="${REFERENCE_TIME_SCALE:-1.0}"
LABEL_PREFIX="${LABEL_PREFIX:-bestofn}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
LAUNCH_LOG_DIR="${LAUNCH_LOG_DIR:-logs/modal_launches/selected_training}"

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

mkdir -p "$LAUNCH_LOG_DIR"
IFS=, read -ra SELECTED_SELECTIONS <<< "$SELECTIONS"
IFS=, read -ra SELECTED_MOTIONS <<< "$MOTION_IDS"

contains() {
  local candidate="$1"
  shift
  for value in "$@"; do
    if [[ "$candidate" == "$value" ]]; then
      return 0
    fi
  done
  return 1
}

active_jobs() {
  jobs -pr | wc -l | tr -d ' '
}

while IFS=, read -r motion_id difficulty prompt duration seed selection source_run_id; do
  if [[ -z "${motion_id}" || "${motion_id}" == \#* ]]; then
    continue
  fi
  if ! contains "$motion_id" "${SELECTED_MOTIONS[@]}"; then
    continue
  fi
  if ! contains "$selection" "${SELECTED_SELECTIONS[@]}"; then
    continue
  fi

  while [[ "$(active_jobs)" -ge "$MAX_PARALLEL" ]]; do
    sleep 5
  done

  scale_label="${REFERENCE_TIME_SCALE//./p}"
  if [[ "$LABEL_PREFIX" == "temporal" ]]; then
    label="temporal-${motion_id}-${selection}-scale-${scale_label}-${TERMINATION_MODE}"
  else
    label="${LABEL_PREFIX}-${motion_id}-${selection}-${TERMINATION_MODE}"
    if [[ "$REFERENCE_TIME_SCALE" != "1.0" ]]; then
      label="${label}-scale-${scale_label}"
    fi
  fi

  echo "Launching ${label}: ${prompt} seed=${seed} source=${source_run_id}"
  modal run --detach modal_kimolab.py \
      --prompt "$prompt" \
      --run-label "$label" \
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
      --reference-time-scale "$REFERENCE_TIME_SCALE" \
      --record-train-video \
      --train-video-interval "$TRAIN_VIDEO_INTERVAL" \
      --train-video-length "$TRAIN_VIDEO_LENGTH" \
      --artifact-sync-interval-s "$ARTIFACT_SYNC_INTERVAL_S" \
      --no-spawn \
    > "${LAUNCH_LOG_DIR}/train_${label}.log" 2>&1 &
done < <(tail -n +2 "$SELECTED_FILE")

wait
echo "Finished selected training jobs. Logs are in ${LAUNCH_LOG_DIR}/train_<label>.log"
