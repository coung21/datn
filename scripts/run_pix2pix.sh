#!/bin/bash
# Pipeline pix2pix (G = UNet 2.5D + BN, D = PatchGAN) trên server (chạy trong tmux của WSL, từ thư mục repo):
#   1) train (configs/exp/pix2pix.yaml)  2) đánh giá test, lưu nii
#   bash scripts/run_pix2pix.sh 2>&1 | tee outputs/pix2pix_pipeline.log
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/mnt/c/Users/user/miniconda3/envs/datn/python.exe}

echo "== [1/2] train $(date)"
$PY -u -X utf8 -m src.train --exp configs/exp/pix2pix.yaml "$@"
RUN=$(ls -td outputs/runs/pix2pix_* | head -1)
echo "run: $RUN"

echo "== [2/2] đánh giá test $(date)"
$PY -u -X utf8 -m src.eval.evaluate --run "$RUN" --ckpt best --split test --save-nii

echo "PIX2PIX_PIPELINE_DONE $RUN $(date)"
