#!/usr/bin/env bash
# =============================================================================
# Main experiment: ML-20M
#
# Trains both encoder-prefix variants of the proposed method.
# ML-20M uses a longer model context (max_len = 200) and K = 10 sub-sequences.
#
# Prerequisite: the processed dataset must exist at
#   <data_root>/ml20m_processed_full_seqlen200/glass_data_256_256_256_ts/
# (see preprocess/ and README.md).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}

COMMON="--data_root ml20m_processed_full_seqlen200 \
  --codebook 256,256,256 \
  --ts_suffix \
  --use_timestamps 1 \
  --max_len 200 \
  --recent_window_r 10 \
  --K 10 --K_min 1 --K_max 10 \
  --batch_size 2048 --infer_size 128 \
  --num_epochs 100 --early_stop 50 \
  --beam_size 20 --lr 1e-4 --weight_decay 0.01 --seed 2025 \
  --lambda_rec 0.5 --lambda_subrec 0.5 --lambda_sub 0.05 \
  --rec_tau 0.07 --rec_item_repr first \
  --sub_lam_t 0.5 --sub_lam_c 0.5"

echo "=== [ML-20M] variant A: prefix [SID-tier ; recent] ==="
$PY train_drift_sidtier.py $COMMON --out_dir outputs/ml20m_sidtier

echo "=== [ML-20M] variant B: prefix [recent] ==="
$PY train_drift_recent.py  $COMMON --timestamp_unit seconds --out_dir outputs/ml20m_recent

echo "Done. Metrics were written to outputs/ml20m_*/*_best.json"
