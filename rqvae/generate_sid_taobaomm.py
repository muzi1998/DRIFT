"""Export the trained RQ-VAE semantic IDs (TAOBAO-MM).

Runs the trained quantizer over every item embedding and writes the resulting
three-level semantic ID codes to a ``.npy`` file consumed by
``preprocess/step4_build_taobaomm.py``.

Two details matter for reproducibility:

* Codes are produced with ``use_sk=False``, i.e. the plain nearest-code assignment
  without the Sinkhorn balancing that is used during training. The exported SIDs
  are therefore the model's unmodified predictions.
* A fourth all-zero column is appended instead of a de-duplicating counter, so
  colliding items keep identical SIDs rather than being artificially separated.
  The collision statistics printed below quantify how often this happens.
"""
import collections
import json
import logging
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import torch
import os
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

from rqvaedatasets import EmbDataset
from models.rqvae import RQVAE

# ================= Configuration =================
# 1) Paths (relative to the repository; override with DRIFT_DATA_ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.environ.get("DRIFT_DATA_ROOT") or os.path.join(_HERE, os.pardir, "data")

MICRO_DIR = os.path.join(_DATA_ROOT, "TAOBAO_MM", "micro")
FILTER_IDS_PATH = os.path.join(MICRO_DIR, "filtered_item_ids.txt")

# Item embedding table covering all items.
NEW_ITEM_EMB_PARQUET = os.path.join(_DATA_ROOT, "TAOBAO_MM", "rqvae_data",
                                    "item_emb.parquet")

# 2) Column names
COL_ITEM_ID = "item_id"     # item id column
COL_EMB = "embedding"       # embedding column (list/array per row)

# 3) Model
CODEBOOK = [64, 128, 128]
DATASET_NAME = "TAOBAO_MM"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Checkpoints are stored in timestamped sub-directories; pick the most recent one.
BASE_CKPT_DIR = os.path.join(_HERE, "ckpt",
                             f"taobao_{CODEBOOK[0]}_{CODEBOOK[1]}_{CODEBOOK[2]}")
def get_latest_ckpt(base_dir):
    if not os.path.exists(base_dir): return None
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not subdirs: return None
    # Directory names are timestamps, so a reverse string sort yields the latest.
    subdirs.sort(reverse=True)
    return os.path.join(base_dir, subdirs[0], "best_collision_model.pth")

ckpt_path = get_latest_ckpt(BASE_CKPT_DIR)
OUTPUT_FILE = os.path.join(
    _DATA_ROOT, DATASET_NAME,
    f"{DATASET_NAME}_t5_rqvae_{CODEBOOK[0]}_{CODEBOOK[1]}_{CODEBOOK[2]}.npy")
# ==================================================

if not ckpt_path:
    raise FileNotFoundError(f"[error] no checkpoint found under: {BASE_CKPT_DIR}")

print(f"[info] loading checkpoint from: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
args_model = ckpt["args"]
state_dict = ckpt["state_dict"]

# ===== Load all item embeddings and flag the target-domain subset =====
data_source = None
is_target_mask = None

# 1) Optional list of target-domain item ids, used for the collision report.
target_ids_set = set()
if os.path.exists(FILTER_IDS_PATH):
    with open(FILTER_IDS_PATH, 'r') as f:
        target_ids_set = set([line.strip() for line in f])
    print(f"[info] loaded target-domain id list with {len(target_ids_set)} ids")
else:
    print(f"[warn] filter file missing, skipping the target-domain collision "
          f"analysis: {FILTER_IDS_PATH}")

# 2) Load the full embedding table.
if os.path.exists(NEW_ITEM_EMB_PARQUET):
    print(f"[info] reading embeddings: {NEW_ITEM_EMB_PARQUET}")
    df = pd.read_parquet(NEW_ITEM_EMB_PARQUET)
    df[COL_ITEM_ID] = df[COL_ITEM_ID].astype(str)
    
    # Flag rows belonging to the target domain.
    is_target_mask = df[COL_ITEM_ID].isin(target_ids_set).values
    print(f"   - total items: {len(df)}")
    print(f"   - target-domain items: {is_target_mask.sum()}")

    # Extract the embedding matrix.
    if COL_EMB in df.columns:
        emb_matrix = np.stack(df[COL_EMB].values)
    else:
        feat_cols = [c for c in df.columns if c != COL_ITEM_ID]
        emb_matrix = df[feat_cols].values.astype(np.float32)

    data_source = TensorDataset(torch.from_numpy(emb_matrix))
    data_source.dim = emb_matrix.shape[1]
else:
    print(f"[warn] embedding parquet missing; falling back to the path recorded "
          f"in the checkpoint: {args_model.data_path}")
    data_source = EmbDataset(args_model.data_path)

# ===== Rebuild the model from the checkpoint and run inference =====
model = RQVAE(in_dim=data_source.dim,
              num_emb_list=args_model.num_emb_list,
              e_dim=args_model.e_dim,
              layers=args_model.layers,
              dropout_prob=args_model.dropout_prob,
              bn=args_model.bn,
              loss_type=args_model.loss_type,
              quant_loss_weight=args_model.quant_loss_weight,
              kmeans_init=args_model.kmeans_init,
              kmeans_iters=args_model.kmeans_iters,
              sk_epsilons=args_model.sk_epsilons,
              sk_iters=args_model.sk_iters)

model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

data_loader = DataLoader(data_source, batch_size=4096, shuffle=False, num_workers=4)

all_indices_raw = []
all_indices_str = []

print(f"\n[info] encoding all items (device: {DEVICE}) ...")
with torch.no_grad():
    for d in tqdm(data_loader):
        if isinstance(d, list): d = d[0]
        d = d.to(DEVICE)
        
        # Plain nearest-code assignment; no Sinkhorn de-duplication.
        indices = model.get_indices(d, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        
        for index in indices:
            all_indices_raw.append(index.tolist())
            # Keep a string form to make collision counting straightforward.
            all_indices_str.append("-".join([str(int(x)) for x in index]))

all_indices_str = np.array(all_indices_str)

# ===== Collision analysis =====
print("\n" + "=" * 40)
print("SID collision report")
print("=" * 40)

if is_target_mask is not None:
    target_sids = all_indices_str[is_target_mask]
    num_target = len(target_sids)
    num_unique_target = len(set(target_sids))
    collision_cnt = num_target - num_unique_target
    
    print("[target domain]")
    print(f"1. items:            {num_target}")
    print(f"2. unique SIDs:      {num_unique_target}")
    print(f"3. collisions:       {collision_cnt}")
    print(f"4. collision rate:   {collision_cnt / num_target:.4%}")

    # Largest number of items sharing a single SID.
    if num_target > 0:
        counts = collections.Counter(target_sids)
        print(f"5. max collision depth: {max(counts.values())}")
else:
    print("[warn] no id matching information, skipping the target-domain analysis.")

# Whole-catalogue reference numbers.
num_total = len(all_indices_str)
num_unique_total = len(set(all_indices_str))
print("\n[all items]")
print(f"1. items:          {num_total}")
print(f"2. collision rate: {(num_total - num_unique_total) / num_total:.4%}")

# ===== Save the exported codes =====
codes_array = np.array(all_indices_raw)

# Append a constant zero column instead of a de-duplicating counter, so that
# colliding items keep identical SIDs.
padding = np.zeros((codes_array.shape[0], 1), dtype=int)
codes_array_save = np.hstack((codes_array, padding))

print("\n" + "-" * 30)
print(f"[done] saving SID codes to: {OUTPUT_FILE}")
print(f"Final shape: {codes_array_save.shape}")
print(f"Sample (the last column is always 0):\n{codes_array_save[:3]}")
np.save(OUTPUT_FILE, codes_array_save)
print("-" * 30)