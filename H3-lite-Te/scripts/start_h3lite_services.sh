#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/logs"
for p in "${H3L_COMFY_PORT:-8962}" "${GATEWAY_H3L_PORT:-8960}"; do
  if (echo >/dev/tcp/127.0.0.1/"$p") >/dev/null 2>&1; then echo "Port $p is already occupied; stop its owner before starting." >&2; exit 1; fi
done
nohup bash "$ROOT/scripts/start_comfyui.sh" >>"$ROOT/logs/comfyui.log" 2>&1 & echo $! >"$ROOT/logs/comfyui.pid"
nohup bash "$ROOT/scripts/start_gateway.sh" >>"$ROOT/logs/gateway.log" 2>&1 & echo $! >"$ROOT/logs/gateway.pid"
echo "Started ComfyUI and gateway. Check logs under $ROOT/logs; health: http://127.0.0.1:${GATEWAY_H3L_PORT:-8960}/health"
