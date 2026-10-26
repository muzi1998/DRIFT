#!/usr/bin/env bash
# =============================================================================
# Main experiment: TAOBAO-MM
#
# Trains both encoder-prefix variants of the proposed method.
#
# Three differences from KuaiRec / ML-20M:
#   * codebook [64, 128, 128] instead of [256, 256, 256], which also changes the
#     derived SID-tier dimension and BOS token id;
#   * the longest histories (max_len = 1000, r = 50, K = 20), so the batch size is
#     reduced accordingly;
#   * no usable interaction timestamps, hence --use_timestamps 0 and no "_ts"
#     dataset suffix. The temporal terms then reduce to a constant and contribute
#     no gradient.
#
# Prerequisite: the processed dataset must exist at
#   <data_root>/TAOBAO_MM/t5_data_64_128_128_seqlen1000_align/glass_data_64_128_128/
# (see preprocess/ and README.md).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}

COMMON="--data_root TAOBAO_MM/t5_data_64_128_128_seqlen1000_align \
  --codebook 64,128,128 \
  --use_timestamps 0 \
  --max_len 1000 \
  --recent_window_r 50 \
  --K 20 --K_min 1 --K_max 20 \
  --batch_size 128 --infer_size 128 \
  --num_epochs 100 --early_stop 50 \
  --beam_size 20 --lr 1e-4 --weight_decay 0.01 --seed 2025 \
  --lambda_rec 0.5 --lambda_subrec 0.5 --lambda_sub 0.05 \
  --rec_tau 0.07 --rec_item_repr first \
  --sub_lam_t 0.5 --sub_lam_c 0.5"

echo "=== [TAOBAO-MM] variant A: prefix [SID-tier ; recent] ==="
$PY train_drift_sidtier.py $COMMON --out_dir outputs/taobaomm_sidtier

echo "=== [TAOBAO-MM] variant B: prefix [recent] ==="
$PY train_drift_recent.py  $COMMON \
  --dataset_dir TAOBAO_MM/t5_data_64_128_128_seqlen1000_align \
  --out_dir outputs/taobaomm_recent

echo "Done. Metrics were written to outputs/taobaomm_*/*_best.json"
