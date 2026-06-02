#!/usr/bin/env bash
set -euo pipefail

PROMPT_FILE="${1:-prompts/seed_sweep_suite.csv}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-50}"
OUTPUT_FPS="${OUTPUT_FPS:-50}"
RUN_LABEL="${RUN_LABEL:-seed-sweep}"
PROMPT_IDS="${PROMPT_IDS:-all}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
LAUNCH_LOG_DIR="${LAUNCH_LOG_DIR:-logs/modal_launches/seed_sweep}"
RENDER_REFERENCE="${RENDER_REFERENCE:-true}"

if ! command -v modal >/dev/null 2>&1; then
  echo "modal is not on PATH. Activate the text-to-torque environment first." >&2
  exit 1
fi

mkdir -p "$LAUNCH_LOG_DIR"
IFS=, read -ra SELECTED_IDS <<< "$PROMPT_IDS"

is_selected() {
  local candidate="$1"
  if [[ "$PROMPT_IDS" == "all" ]]; then
    return 0
  fi
  for selected in "${SELECTED_IDS[@]}"; do
    if [[ "$candidate" == "$selected" ]]; then
      return 0
    fi
  done
  return 1
}

active_jobs() {
  jobs -pr | wc -l | tr -d ' '
}

render_flag() {
  if [[ "$RENDER_REFERENCE" == "true" || "$RENDER_REFERENCE" == "1" ]]; then
    echo "--render-reference"
  else
    echo "--no-render-reference"
  fi
}

while IFS=, read -r prompt_id difficulty prompt duration seed notes; do
  if [[ -z "${prompt_id}" || "${prompt_id}" == \#* ]]; then
    continue
  fi
  if ! is_selected "$prompt_id"; then
    continue
  fi

  while [[ "$(active_jobs)" -ge "$MAX_PARALLEL" ]]; do
    sleep 5
  done

  echo "Launching reference seed sweep ${prompt_id}: ${prompt} seed=${seed}"
  modal run --detach modal_kimolab.py \
      --prompt "$prompt" \
      --run-label "$RUN_LABEL" \
      --duration "$duration" \
      --seed "$seed" \
      --diffusion-steps "$DIFFUSION_STEPS" \
      --output-fps "$OUTPUT_FPS" \
      "$(render_flag)" \
      --no-train \
      --no-spawn \
    > "${LAUNCH_LOG_DIR}/reference_${prompt_id}_seed${seed}.log" 2>&1 &
done < <(tail -n +2 "$PROMPT_FILE")

wait
echo "Finished seed sweep jobs. Logs are in ${LAUNCH_LOG_DIR}/reference_<prompt_id>_seed<seed>.log"
