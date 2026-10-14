#!/usr/bin/env bash
# Usage: scripts/train.sh configs/smoke.yaml data/sample_bin
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/gpt2_small.yaml}"
DATA_DIR="${2:-data/sample_bin}"
python -m torchllm.train --config "$CONFIG" --data-dir "$DATA_DIR"
