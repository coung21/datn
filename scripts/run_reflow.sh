#!/bin/bash
# Pipeline conditional reflow trên server (chạy trong tmux của WSL, từ thư mục repo):
#   1) sinh cặp (teacher Euler 2 bước, noise seed 1) cho tập train
#   2) train student (configs/exp/reflow.yaml)
#   3) đánh giá test với NFE 1 2 4
# Chạy lại sau khi bị ngắt: bước 1 bỏ qua BN đã xong; nếu bước 2 dở, dùng --resume thủ công.
#   bash scripts/run_reflow.sh 2>&1 | tee outputs/reflow_pipeline.log
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/mnt/c/Users/user/miniconda3/envs/datn/python.exe}
TEACHER=outputs/runs/flow_matching_20261018-094349

echo "== [1/3] sinh cặp reflow $(date)"
$PY -u -X utf8 scripts/generate_reflow.py --teacher "$TEACHER" --nfe 2 --seed 1

echo "== [2/3] train student $(date)"
$PY -u -X utf8 -m src.train --exp configs/exp/reflow.yaml
RUN=$(ls -td outputs/runs/reflow_* | head -1)
echo "run: $RUN"

echo "== [3/3] đánh giá test $(date)"
$PY -u -X utf8 -m src.eval.evaluate --run "$RUN" --ckpt best --split test --nfe 1 2 4

echo "REFLOW_PIPELINE_DONE $RUN $(date)"
