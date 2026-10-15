#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
python -m pip install -U huggingface_hub modelscope
bash "$ROOT/scripts/download_lite_weights_upstream.sh"
