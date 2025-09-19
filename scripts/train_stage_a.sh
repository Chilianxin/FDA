#!/usr/bin/env bash
set -euo pipefail

python -m fda.training.train_dl "$@" | cat

