#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${H3L_CONDA_ENV:-h3lite-te}"
export H3L_COMFY_BASE="${H3L_COMFY_BASE:-http://127.0.0.1:${H3L_COMFY_PORT:-8962}}"
export H3L_USE_TESPEED="${H3L_USE_TESPEED:-1}"
export H3L_FL2VA_PROMPT_REWRITE="${H3L_FL2VA_PROMPT_REWRITE:-0}"
cd "$ROOT"
exec python gateway_h3lite_fastapi.py --host "${H3L_GATEWAY_HOST:-127.0.0.1}" --port "${GATEWAY_H3L_PORT:-8960}"
