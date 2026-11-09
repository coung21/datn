#!/bin/bash
# Chạy lại 1 cấu hình với nhiều seed huấn luyện (cùng split), mỗi seed: train rồi đánh giá test ở checkpoint best.
#   bash scripts/run_seeds.sh <exp.yaml> <tên run> <seed...>
#   vd: bash scripts/run_seeds.sh configs/exp/unet2d_gn_ag.yaml unet2d_gn_ag 1 2 3 4
# Run lưu ở outputs/runs/<tên run>_s<seed>_<thời gian>. Seed lỗi được ghi log, các seed sau vẫn chạy.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/mnt/c/Users/user/miniconda3/envs/datn/python.exe}
EXP=$1; NAME=$2; shift 2

for S in "$@"; do
  echo "== [$NAME seed $S] train $(date)"
  if ! $PY -u -X utf8 -m src.train --exp "$EXP" --set seed=$S run_name=${NAME}_s$S; then
    echo "SEED_FAILED $NAME seed $S (train) $(date)"; continue
  fi
  RUN=$(ls -td outputs/runs/${NAME}_s${S}_* | head -1)
  echo "run: $RUN"
  echo "== [$NAME seed $S] đánh giá test $(date)"
  if ! $PY -u -X utf8 -m src.eval.evaluate --run "$RUN" --ckpt best --split test; then
    echo "SEED_FAILED $NAME seed $S (eval) $(date)"; continue
  fi
  echo "SEED_DONE $NAME seed $S $RUN $(date)"
done
echo "SEEDS_PIPELINE_DONE $NAME $(date)"
