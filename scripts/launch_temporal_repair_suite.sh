#!/usr/bin/env bash
set -euo pipefail

MAX_ITERATIONS="${MAX_ITERATIONS:-1200}"
NUM_ENVS="${NUM_ENVS:-1024}"
SAVE_INTERVAL="${SAVE_INTERVAL:-400}"
TRAIN_VIDEO_INTERVAL="${TRAIN_VIDEO_INTERVAL:-10000}"
ARTIFACT_SYNC_INTERVAL_S="${ARTIFACT_SYNC_INTERVAL_S:-60}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-50}"
OUTPUT_FPS="${OUTPUT_FPS:-50}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
LAUNCH_LOG_DIR="${LAUNCH_LOG_DIR:-logs/modal_launches/temporal_repair}"
MOTION_IDS="${MOTION_IDS:-squat_stand,jump,roll}"
TIME_SCALES="${TIME_SCALES:-1.5,2.0}"

if ! command -v modal >/dev/null 2>&1; then
  echo "modal is not on PATH. Activate the text-to-torque environment first." >&2
  exit 1
fi

mkdir -p "$LAUNCH_LOG_DIR"
IFS=, read -ra SELECTED_MOTIONS <<< "$MOTION_IDS"
IFS=, read -ra SELECTED_SCALES <<< "$TIME_SCALES"

active_jobs() {
  jobs -pr | wc -l | tr -d ' '
}

prompt_for_motion() {
  case "$1" in
    squat_stand) echo "A person squats down and stands up" ;;
    jump) echo "A person jumps" ;;
    roll) echo "A person rolls forward on the ground" ;;
    turn_walk) echo "A person turns around and walks forward" ;;
    *) echo "Unknown motion id: $1" >&2; return 1 ;;
  esac
}

video_length_for_scale() {
  python3 - "$1" <<'PY'
import sys
scale = float(sys.argv[1])
print(int(round((4.0 * scale + 1.0) * 50)))
PY
}

for motion_id in "${SELECTED_MOTIONS[@]}"; do
  prompt="$(prompt_for_motion "$motion_id")"
  for scale in "${SELECTED_SCALES[@]}"; do
    while [[ "$(active_jobs)" -ge "$MAX_PARALLEL" ]]; do
      sleep 5
    done

    scale_label="${scale//./p}"
    label="temporal-${motion_id}-scale-${scale_label}-strict"
    video_length="$(video_length_for_scale "$scale")"

    echo "Launching ${label}: ${prompt}"
    modal run --detach modal_kimolab.py \
        --prompt "$prompt" \
        --run-label "$label" \
        --duration 4.0 \
        --seed 0 \
        --diffusion-steps "$DIFFUSION_STEPS" \
        --output-fps "$OUTPUT_FPS" \
        --no-render-reference \
        --train \
        --num-envs "$NUM_ENVS" \
        --max-iterations "$MAX_ITERATIONS" \
        --save-interval "$SAVE_INTERVAL" \
        --no-disable-terminations \
        --reference-time-scale "$scale" \
        --record-train-video \
        --train-video-interval "$TRAIN_VIDEO_INTERVAL" \
        --train-video-length "$video_length" \
        --artifact-sync-interval-s "$ARTIFACT_SYNC_INTERVAL_S" \
        --no-spawn \
      > "${LAUNCH_LOG_DIR}/train_${label}.log" 2>&1 &
  done
done

wait
echo "Finished temporal repair jobs. Logs are in ${LAUNCH_LOG_DIR}/train_<label>.log"
