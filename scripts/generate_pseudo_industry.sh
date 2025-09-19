#!/usr/bin/env bash
set -euo pipefail

INPUT_CSV=${1:-data/hs300.csv}
OUTDIR=${2:-outputs/pseudo_industry}

python -m code.cli_pseudo_industry --input "$INPUT_CSV" --outdir "$OUTDIR" --window 120 --topk 15 --resolution 1.0 --min_size 5 | cat

