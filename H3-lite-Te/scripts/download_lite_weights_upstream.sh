#!/usr/bin/env bash
# Download MiniMax-H3 lite weights: Comfy quantized + GGUF + Turbo LoRA.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="${ROOT}/models"
LOG_DIR="${ROOT}/logs"
mkdir -p "$LOG_DIR" \
  "${MODELS}/diffusion_models" \
  "${MODELS}/text_encoders" \
  "${MODELS}/vae" \
  "${MODELS}/unet" \
  "${MODELS}/loras" \
  "${ROOT}/third_party"

LOG="${LOG_DIR}/download_lite_weights.log"
exec > >(tee -a "$LOG") 2>&1

echo "[lite] $(date -Is) start"
echo "[lite] models -> ${MODELS}"

place_file() {
  # place_file <src_glob_or_path> <dest_dir>
  local src="$1"
  local dest_dir="$2"
  local f
  # shellcheck disable=SC2086
  for f in $src; do
    if [[ -f "$f" ]]; then
      local name
      name="$(basename "$f")"
      if [[ -f "${dest_dir}/${name}" ]]; then
        echo "[lite] exists ${dest_dir}/${name}"
      else
        echo "[lite] copy ${f} -> ${dest_dir}/${name}"
        cp -n "$f" "${dest_dir}/${name}"
      fi
      return 0
    fi
  done
  echo "[lite] WARN not found: $src"
  return 1
}

# ---------- Comfy-Org (ModelScope) ----------
COMFY_DIR="${MODELS}/_tmp_comfy_org"
mkdir -p "$COMFY_DIR"
echo "[lite] downloading Comfy-Org/MiniMax-H3 (selected light files)..."
modelscope download \
  --model "Comfy-Org/MiniMax-H3" \
  --local_dir "$COMFY_DIR" \
  --include \
    "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" \
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" \
    "vae/minimax_h3_audio_vae_fp32.safetensors" \
    "vae/minimax_h3_video_vae_fp16.safetensors" \
    "README.md"

place_file "${COMFY_DIR}/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" "${MODELS}/diffusion_models" || \
  place_file "${COMFY_DIR}/minimax_h3_fl2va_pruned_int8_convrot.safetensors" "${MODELS}/diffusion_models"

place_file "${COMFY_DIR}/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" "${MODELS}/text_encoders" || \
  place_file "${COMFY_DIR}/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" "${MODELS}/text_encoders"

place_file "${COMFY_DIR}/vae/minimax_h3_audio_vae_fp32.safetensors" "${MODELS}/vae" || \
  place_file "${COMFY_DIR}/minimax_h3_audio_vae_fp32.safetensors" "${MODELS}/vae"

place_file "${COMFY_DIR}/vae/minimax_h3_video_vae_fp16.safetensors" "${MODELS}/vae" || \
  place_file "${COMFY_DIR}/minimax_h3_video_vae_fp16.safetensors" "${MODELS}/vae"

# ---------- GGUF (ModelScope) ----------
GGUF_DIR="${MODELS}/_tmp_gguf"
mkdir -p "$GGUF_DIR"
echo "[lite] downloading realrebelai/MiniMax-H3_GGUFs (Q4_K_M)..."
modelscope download \
  --model "realrebelai/MiniMax-H3_GGUFs" \
  --local_dir "$GGUF_DIR" \
  --include \
    "MiniMax-H3-FL2VA-Q4_K_M.gguf" \
    "qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf" \
    "README.md" \
    "FL2V_(WORKFLOW).json"

place_file "${GGUF_DIR}/MiniMax-H3-FL2VA-Q4_K_M.gguf" "${MODELS}/unet"
place_file "${GGUF_DIR}/qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf" "${MODELS}/text_encoders"
if [[ -f "${GGUF_DIR}/FL2V_(WORKFLOW).json" ]]; then
  cp -n "${GGUF_DIR}/FL2V_(WORKFLOW).json" "${ROOT}/docs/" || true
fi

# ---------- Turbo LoRA (Hugging Face mirror) ----------
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
TURBO_DIR="${MODELS}/_tmp_turbo"
mkdir -p "$TURBO_DIR"
echo "[lite] downloading lightx2v/Minimax-h3-Turbo ..."
hf download "lightx2v/Minimax-h3-Turbo" \
  --local-dir "$TURBO_DIR" \
  "minimax_h3_fl2v_turbo_4step_v0.1.safetensors" "README.md"

place_file "${TURBO_DIR}/minimax_h3_fl2v_turbo_4step_v0.1.safetensors" "${MODELS}/loras"

# ---------- ModelTC Turbo code ----------
TURBO_CODE="${ROOT}/third_party/Minimax-H3-Turbo"
if [[ ! -d "${TURBO_CODE}/.git" ]]; then
  echo "[lite] cloning ModelTC/Minimax-H3-Turbo ..."
  if ! git clone --depth 1 https://github.com/ModelTC/Minimax-H3-Turbo.git "$TURBO_CODE"; then
    git clone --depth 1 https://ghproxy.net/https://github.com/ModelTC/Minimax-H3-Turbo.git "$TURBO_CODE" || \
      echo "[lite] WARN: clone Turbo code failed; LoRA weights still usable in ComfyUI"
  fi
else
  echo "[lite] Turbo code already present"
fi

echo "[lite] $(date -Is) download pass finished"
du -sh "${MODELS}/diffusion_models" "${MODELS}/text_encoders" "${MODELS}/vae" "${MODELS}/unet" "${MODELS}/loras" 2>/dev/null || true
