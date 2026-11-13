#!/bin/bash
# GN + AG với L1 có trọng số ở xương (bone_weight 2 và 5) trên server (chạy trong tmux của WSL, từ thư mục repo).
# Mỗi run: train rồi đánh giá test ở checkpoint best.
#   bash scripts/run_bone_weight.sh 2>&1 | tee outputs/bone_weight_pipeline.log
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/mnt/c/Users/user/miniconda3/envs/datn/python.exe}
for L in ${@:-2 5}; do
  NAME=unet2d_gn_ag_bw$L
  echo "== [$NAME] train $(date)"
  if ! $PY -u -X utf8 -m src.train --exp configs/exp/$NAME.yaml; then echo "RUN_FAILED $NAME (train) $(date)"; continue; fi
  RUN=$(ls -td outputs/runs/${NAME}_2* | head -1)
  echo "== [$NAME] đánh giá test $(date)"
  if ! $PY -u -X utf8 -m src.eval.evaluate --run "$RUN" --ckpt best --split test; then echo "RUN_FAILED $NAME (eval) $(date)"; continue; fi
  echo "RUN_DONE $NAME $RUN $(date)"
done
echo "BONE_WEIGHT_DONE $(date)"
