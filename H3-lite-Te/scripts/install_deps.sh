#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMFY="${H3L_COMFY_ROOT:-${ROOT}/third_party/ComfyUI}"
ENV_NAME="${H3L_CONDA_ENV:-h3lite-te}"
if ! command -v conda >/dev/null 2>&1; then echo "conda is required; install Miniconda/Miniforge first" >&2; exit 1; fi
if [[ ! -d "$COMFY" ]]; then
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY"
fi
if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then conda create -y -n "$ENV_NAME" python=3.10; fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
python -m pip install --upgrade pip
# Install PyTorch matching the installed NVIDIA driver/CUDA. Set TORCH_INDEX_URL if needed.
if [[ -n "${TORCH_INDEX_URL:-}" ]]; then python -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"; else python -m pip install torch torchvision; fi
python -m pip install -r "$COMFY/requirements.txt"
python -m pip install -r "$ROOT/requirements-gateway.txt"
echo "Dependencies installed. Review the H3 ComfyUI/H3-lite-Te node installation docs before inference."
