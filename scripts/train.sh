#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

if [ ! -d .venv ]; then
  echo "[train] Please run scripts/setup_env.sh first to create .venv"
  exit 1
fi
source .venv/bin/activate

CSV_PATH=${1:-"data/prices.csv"}
WINDOW=${WINDOW_SIZE:-252}
STEP=${STEP_SIZE:-21}
DEVICE=${DEVICE:-cpu}

python -m code.train_dynamic --csv "$CSV_PATH" --window_size "$WINDOW" --step_size "$STEP" --device "$DEVICE" | cat
