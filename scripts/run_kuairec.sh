#!/usr/bin/env bash
# =============================================================================
# Main experiment: KuaiRec
#
# Trains both encoder-prefix variants of the proposed method:
#   A) train_drift_sidtier.py -- prefix [SID-tier ; recent window]
#   B) train_drift_recent.py  -- prefix [recent window]
#
# Prerequisite: the processed dataset must exist at
#   <data_root>/kuairec_processed/glass_data_256_256_256_ts/
# (see preprocess/ and README.md).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}

COMMON="--data_root kuairec_processed \
  --codebook 256,256,256 \
  --ts_suffix \
  --use_timestamps 1 \
  --max_len 1000 \
  --recent_window_r 50 \
  --K 20 --K_min 1 --K_max 20 \
  --batch_size 512 --infer_size 128 \
  --num_epochs 100 --early_stop 50 \
  --beam_size 20 --lr 1e-4 --weight_decay 0.01 --seed 2025 \
  --lambda_rec 0.5 --lambda_subrec 0.5 --lambda_sub 0.05 \
  --rec_tau 0.07 --rec_item_repr first \
  --sub_lam_t 0.5 --sub_lam_c 0.5"

echo "=== [KuaiRec] variant A: prefix [SID-tier ; recent] ==="
$PY train_drift_sidtier.py $COMMON --out_dir outputs/kuairec_sidtier

echo "=== [KuaiRec] variant B: prefix [recent] ==="
$PY train_drift_recent.py  $COMMON --timestamp_unit seconds --out_dir outputs/kuairec_recent

echo "Done. Metrics were written to outputs/kuairec_*/*_best.json"
