#!/usr/bin/env bash
# Symlink lite weights into ComfyUI-FlashVSR models dirs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMFY="${COMFYUI_ROOT:-${ROOT}/third_party/ComfyUI}"
SRC="${ROOT}/models"

if [[ ! -d "$COMFY" ]]; then
  echo "[lite] ComfyUI root missing: $COMFY" >&2
  exit 1
fi

link_dir() {
  local sub="$1"
  mkdir -p "${COMFY}/models/${sub}" "${SRC}/${sub}"
  shopt -s nullglob
  for f in "${SRC}/${sub}/"*; do
    [[ -e "$f" ]] || continue
    local name base
    name="$(basename "$f")"
    # resolve to real file if symlink
    if [[ -L "$f" ]]; then
      base="$(readlink -f "$f")"
    else
      base="$(realpath "$f")"
    fi
    ln -sfn "$base" "${COMFY}/models/${sub}/${name}"
    echo "[lite] link ${COMFY}/models/${sub}/${name} -> ${base}"
  done
}

link_dir diffusion_models
link_dir text_encoders
link_dir vae
link_dir unet
link_dir loras

echo "[lite] ComfyUI links ready under ${COMFY}/models"
