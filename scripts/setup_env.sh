#!/usr/bin/env bash
set -euo pipefail

# Detect package manager
detect_pm() {
  if command -v apt-get >/dev/null 2>&1; then echo apt; return; fi
  if command -v yum >/dev/null 2>&1; then echo yum; return; fi
  if command -v dnf >/dev/null 2>&1; then echo dnf; return; fi
  if command -v pacman >/dev/null 2>&1; then echo pacman; return; fi
  echo none
}

PM=$(detect_pm)

echo "[setup] Package manager: ${PM}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[setup] Installing Python 3 and pip..."
  case "$PM" in
    apt)
      sudo apt-get update -y
      sudo apt-get install -y python3 python3-venv python3-pip build-essential
      ;;
    yum)
      sudo yum install -y python3 python3-virtualenv python3-pip gcc gcc-c++ make
      ;;
    dnf)
      sudo dnf install -y python3 python3-virtualenv python3-pip gcc gcc-c++ make
      ;;
    pacman)
      sudo pacman -Syu --noconfirm python python-pip base-devel
      ;;
    *)
      echo "[setup] Please install Python 3.10+ and pip manually."
      ;;
  esac
fi

PY=${PYTHON:-python3}
PIP=${PIP:-pip3}

if [ ! -d ".venv" ]; then
  echo "[setup] Creating virtual environment .venv"
  ${PY} -m venv .venv
fi

source .venv/bin/activate

echo "[setup] Upgrading pip/setuptools/wheel"
pip install --upgrade pip setuptools wheel

REQ_FILE="requirements.txt"
if [ -f "$REQ_FILE" ] && [ -s "$REQ_FILE" ]; then
  echo "[setup] Installing Python dependencies from $REQ_FILE"
  pip install -r requirements.txt
else
  echo "[setup] Installing base dependencies"
  pip install numpy pandas scikit-learn torch --index-url https://download.pytorch.org/whl/cpu
fi

echo "[setup] Installing torch_geometric and companions"
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.4.0+cpu.html || true

echo "[setup] Installing neural GC packages (optional)"
pip install git+https://github.com/Quadflor/Neural-GC.git@master || true
pip install mlcausality || true

echo "[setup] Done. Activate with: source .venv/bin/activate"

