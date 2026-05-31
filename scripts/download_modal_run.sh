#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:?Usage: scripts/download_modal_run.sh <run_id> [destination_root]}"
DEST_ROOT="${2:-motions/from_modal}"
DEST_DIR="${DEST_ROOT}/${RUN_ID}"

if ! command -v modal >/dev/null 2>&1; then
  echo "modal is not on PATH. Activate the text-to-torque environment first." >&2
  exit 1
fi

mkdir -p "$DEST_DIR"

for file in metadata.json motion.csv motion.npz reference_motion.mp4; do
  if modal volume get --force text-to-torque-results "kimolab/${RUN_ID}/${file}" "${DEST_DIR}/${file}"; then
    echo "Downloaded ${file}"
  else
    echo "Skipped missing ${file}"
  fi
done

if modal volume ls text-to-torque-results "kimolab/${RUN_ID}/logs" >/dev/null 2>&1; then
  rm -rf "${DEST_DIR}/logs"
  modal volume get text-to-torque-results "kimolab/${RUN_ID}/logs" "${DEST_DIR}"
  echo "Downloaded logs"
fi
