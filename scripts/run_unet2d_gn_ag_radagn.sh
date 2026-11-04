#!/bin/bash
# Pipeline UNet 2.5D + GN + AG + AdaGN theo vùng trên server (chạy trong tmux của WSL, từ thư mục repo):
#   1) train (configs/exp/unet2d_gn_ag_radagn.yaml)  2) đánh giá test, lưu nii
#   bash scripts/run_unet2d_gn_ag_radagn.sh 2>&1 | tee outputs/unet2d_gn_ag_radagn_pipeline.log
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/mnt/c/Users/user/miniconda3/envs/datn/python.exe}

echo "== [1/2] train $(date)"
$PY -u -X utf8 -m src.train --exp configs/exp/unet2d_gn_ag_radagn.yaml "$@"
RUN=$(ls -td outputs/runs/unet2d_gn_ag_radagn_* | head -1)
echo "run: $RUN"

echo "== [2/2] đánh giá test $(date)"
$PY -u -X utf8 -m src.eval.evaluate --run "$RUN" --ckpt best --split test --save-nii

echo "UNET2D_GN_AG_RADAGN_PIPELINE_DONE $RUN $(date)"
