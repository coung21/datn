#!/bin/bash
# Ablation theo mặt cắt của nhánh multi-view trên server (chạy trong tmux của WSL, từ thư mục repo), seed 42:
# chỉ axial, bỏ axial, bỏ coronal, bỏ sagittal. Mỗi run: train rồi đánh giá test ở checkpoint best.
#   bash scripts/run_plane_ablation.sh 2>&1 | tee outputs/plane_ablation_pipeline.log
set -uo pipefail
cd "$(dirname "$0")/.."
for V in ${@:-ax noax nocor nosag}; do
  bash scripts/run_seeds.sh configs/exp/unet2d_gn_ag_mv_$V.yaml unet2d_gn_ag_mv_$V 42
done
echo "PLANE_ABLATION_DONE $(date)"
