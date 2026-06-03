#!/usr/bin/env bash
set -euo pipefail

PROMPT_FILE="${1:-prompts/prompt_suite.csv}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-50}"
OUTPUT_FPS="${OUTPUT_FPS:-50}"
LAUNCH_LOG_DIR="${LAUNCH_LOG_DIR:-logs/modal_launches}"
PROMPT_IDS="${PROMPT_IDS:-all}"
RUN_LABEL="${RUN_LABEL:-reference}"

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

while IFS=, read -r prompt_id difficulty prompt duration seed notes; do
  if [[ -z "${prompt_id}" || "${prompt_id}" == \#* ]]; then
    continue
  fi
  if ! is_selected "$prompt_id"; then
    continue
  fi

  echo "Launching ${prompt_id}: ${prompt}"
  nohup modal run --detach modal_kimolab.py \
      --prompt "$prompt" \
      --run-label "$RUN_LABEL" \
      --duration "$duration" \
      --seed "$seed" \
      --diffusion-steps "$DIFFUSION_STEPS" \
      --output-fps "$OUTPUT_FPS" \
      --render-reference \
      --no-train \
      --no-spawn \
    > "${LAUNCH_LOG_DIR}/reference_${prompt_id}.log" 2>&1 &
done < <(tail -n +2 "$PROMPT_FILE")

echo "Queued reference jobs. Logs are in ${LAUNCH_LOG_DIR}/reference_<prompt_id>.log"
