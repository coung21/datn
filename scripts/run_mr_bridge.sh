#!/bin/bash
# Pipeline flow matching MR -> CT trên server (chạy trong tmux của WSL, từ thư mục repo):
#   1) train (configs/exp/mr_bridge.yaml)  2) đánh giá test NFE 1 2 4
#   bash scripts/run_mr_bridge.sh 2>&1 | tee outputs/mr_bridge_pipeline.log
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/mnt/c/Users/user/miniconda3/envs/datn/python.exe}

echo "== [1/2] train $(date)"
$PY -u -X utf8 -m src.train --exp configs/exp/mr_bridge.yaml
RUN=$(ls -td outputs/runs/mr_bridge_* | head -1)
echo "run: $RUN"

echo "== [2/2] đánh giá test $(date)"
$PY -u -X utf8 -m src.eval.evaluate --run "$RUN" --ckpt best --split test --nfe 1 2 4

echo "MR_BRIDGE_PIPELINE_DONE $RUN $(date)"
