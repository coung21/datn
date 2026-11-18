#!/bin/bash
# Lặp seed cho multi-view phân cấp: MV pyramid rồi MV pyramid + bw2, seed 1-2 (seed 42 đã có).
#   EXTRA_SET=logging.wandb.mode=offline bash scripts/run_seed_mvp.sh 2>&1 | tee outputs/seed_mvp_pipeline.log
cd "$(dirname "$0")/.."
bash scripts/run_seeds.sh configs/exp/unet2d_gn_ag_mvp.yaml unet2d_gn_ag_mvp 1 2
bash scripts/run_seeds.sh configs/exp/unet2d_gn_ag_mvp_bw2.yaml unet2d_gn_ag_mvp_bw2 1 2
echo "SEED_MVP_DONE $(date)"
