#!/bin/bash
# Pipeline đối chứng regression L2 trên server (chạy trong tmux của WSL, từ thư mục repo):
#   1) train (configs/exp/regression_l2.yaml)  2) đánh giá test
#   bash scripts/run_regression_l2.sh 2>&1 | tee outputs/regression_l2_pipeline.log
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/mnt/c/Users/user/miniconda3/envs/datn/python.exe}

echo "== [1/2] train $(date)"
$PY -u -X utf8 -m src.train --exp configs/exp/regression_l2.yaml
RUN=$(ls -td outputs/runs/regression_l2_* | head -1)
echo "run: $RUN"

echo "== [2/2] đánh giá test $(date)"
$PY -u -X utf8 -m src.eval.evaluate --run "$RUN" --ckpt best --split test --nfe 1

echo "REGRESSION_L2_PIPELINE_DONE $RUN $(date)"
