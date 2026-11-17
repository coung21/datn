#!/bin/bash
# Pipeline UNet 2.5D + GN + AG + multi-view input + L1 trọng số xương trên server (chạy trong tmux của WSL, từ thư mục repo):
#   1) train (configs/exp/unet2d_gn_ag_mv_bw2.yaml)  2) đánh giá test
#   bash scripts/run_unet2d_gn_ag_mv_bw2.sh 2>&1 | tee outputs/unet2d_gn_ag_mv_bw2_pipeline.log
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/mnt/c/Users/user/miniconda3/envs/datn/python.exe}

echo "== [1/2] train $(date)"
$PY -u -X utf8 -m src.train --exp configs/exp/unet2d_gn_ag_mv_bw2.yaml "$@"
RUN=$(ls -td outputs/runs/unet2d_gn_ag_mv_bw2_* | head -1)
echo "run: $RUN"

echo "== [2/2] đánh giá test $(date)"
$PY -u -X utf8 -m src.eval.evaluate --run "$RUN" --ckpt best --split test

echo "UNET2D_GN_AG_MV_BW2_PIPELINE_DONE $RUN $(date)"
