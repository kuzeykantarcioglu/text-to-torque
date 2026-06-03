#!/usr/bin/env bash
set -euo pipefail

PATTERN="${1:?Usage: scripts/download_modal_matching.sh <run_id_regex> [destination_root]}"
DEST_ROOT="${2:-motions/from_modal}"
VOLUME="${VOLUME:-text-to-torque-results}"

if ! command -v modal >/dev/null 2>&1; then
  echo "modal is not on PATH. Activate the text-to-torque environment first." >&2
  exit 1
fi

RUN_LIST="$(mktemp)"
modal volume ls "$VOLUME" kimolab \
  | sed 's#^kimolab/##' \
  | grep -E "$PATTERN" \
  | sort > "$RUN_LIST" || true

if [[ ! -s "$RUN_LIST" ]]; then
  rm -f "$RUN_LIST"
  echo "No Modal runs matched regex: ${PATTERN}" >&2
  exit 1
fi

count=0
while IFS= read -r run_id; do
  echo "Downloading ${run_id}"
  scripts/download_modal_run.sh "$run_id" "$DEST_ROOT"
  count=$((count + 1))
done < "$RUN_LIST"
rm -f "$RUN_LIST"

echo "Downloaded ${count} runs matching ${PATTERN}"
