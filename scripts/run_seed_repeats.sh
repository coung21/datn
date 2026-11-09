#!/bin/bash
# Hàng đợi lặp seed cho nhận xét của GVHD: GN + AG rồi GN + AG + multi-view, seed 1-4 (seed 42 đã có).
#   bash scripts/run_seed_repeats.sh 2>&1 | tee outputs/seed_repeats_pipeline.log
cd "$(dirname "$0")/.."
bash scripts/run_seeds.sh configs/exp/unet2d_gn_ag.yaml unet2d_gn_ag 1 2 3 4
bash scripts/run_seeds.sh configs/exp/unet2d_gn_ag_mv.yaml unet2d_gn_ag_mv 1 2 3 4
echo "SEED_REPEATS_DONE $(date)"
