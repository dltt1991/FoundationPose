#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
exec "$project_dir/.venv/bin/python" "$project_dir/run_portable.py" \
  --source realsense \
  --mesh "$project_dir/../SR300_step_target_M4.stl" \
  --mesh-scale 0.001 --mesh-color 255 255 255 \
  --device mps --max-frames 0 --display "$@"
