#!/usr/bin/env bash
# Verify expected lite weight files exist and report sizes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
M="${ROOT}/models"
fail=0

need() {
  local path="$1"
  local min_bytes="${2:-1}"
  if [[ ! -e "$path" ]]; then
    echo "MISSING  $path"
    fail=1
    return
  fi
  local sz
  sz=$(stat -c%s "$path" 2>/dev/null || echo 0)
  if (( sz < min_bytes )); then
    echo "TOO_SMALL (${sz}) $path"
    fail=1
    return
  fi
  printf 'OK  %10s  %s\n' "$(numfmt --to=iec --suffix=B "$sz" 2>/dev/null || echo "${sz}B")" "$path"
}

echo "=== Comfy-Org light ==="
need "${M}/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" $((15 * 1024 * 1024 * 1024))
need "${M}/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors" $((15 * 1024 * 1024 * 1024))
need "${M}/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" $((10 * 1024 * 1024 * 1024))
need "${M}/vae/minimax_h3_audio_vae_fp32.safetensors" $((100 * 1024 * 1024))
need "${M}/vae/minimax_h3_video_vae_fp16.safetensors" $((1 * 1024 * 1024 * 1024))

echo "=== GGUF ==="
need "${M}/unet/MiniMax-H3-FL2VA-Q4_K_M.gguf" $((10 * 1024 * 1024 * 1024))
need "${M}/text_encoders/qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf" $((8 * 1024 * 1024 * 1024))

echo "=== Turbo LoRA ==="
need "${M}/loras/minimax_h3_fl2v_turbo_4step_v0.1.safetensors" $((500 * 1024 * 1024))

echo "=== Turbo code ==="
if [[ -d "${ROOT}/third_party/Minimax-H3-Turbo" ]]; then
  echo "OK  third_party/Minimax-H3-Turbo"
else
  echo "MISSING  third_party/Minimax-H3-Turbo"
  fail=1
fi

echo "=== totals ==="
du -sh "${M}/diffusion_models" "${M}/text_encoders" "${M}/vae" "${M}/unet" "${M}/loras" 2>/dev/null || true

if (( fail )); then
  echo "[lite] verify FAILED"
  exit 1
fi
echo "[lite] verify OK"
