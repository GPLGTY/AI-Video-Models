#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMFY="${H3L_COMFY_ROOT:-${ROOT}/third_party/ComfyUI}"
ENV_NAME="${H3L_CONDA_ENV:-h3lite-te}"
PORT="${H3L_COMFY_PORT:-8962}"
[[ -d "$COMFY" ]] || { echo "ComfyUI is missing; run scripts/install_deps.sh" >&2; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$ROOT/logs" "$ROOT/outputs" "$COMFY/input" "$COMFY/output"
cd "$COMFY"
exec python main.py --listen "${H3L_COMFY_HOST:-127.0.0.1}" --port "$PORT" --disable-auto-launch
