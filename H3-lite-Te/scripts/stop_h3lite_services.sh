#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for name in gateway comfyui; do
  f="$ROOT/logs/$name.pid"
  if [[ -f "$f" ]]; then pid="$(<"$f")"; if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then kill "$pid"; fi; rm -f "$f"; fi
done
