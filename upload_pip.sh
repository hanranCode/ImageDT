#!/usr/bin/env bash
set -euo pipefail
# Build and validate a fresh release. Upload is an explicit, separate choice.
python_cmd="${PYTHON:-python3}"
output_dir="$(mktemp -d "${TMPDIR:-/tmp}/imagedt-release.XXXXXX")"
"$python_cmd" -m build --outdir "$output_dir"
"$python_cmd" -m twine check "$output_dir"/*
printf 'Validated artifacts: %s\n' "$output_dir"
if [[ "${1:-}" == "--upload" ]]; then
  "$python_cmd" -m twine upload "$output_dir"/*
else
  printf 'Ready for review. To publish explicitly, run: PYTHON=%s bash upload_pip.sh --upload\n' "$python_cmd"
fi
