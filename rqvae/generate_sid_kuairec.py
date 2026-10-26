"""End-to-end semantic ID (SID) generation for KuaiRec.

Trains the RQ-VAE quantizer and exports the resulting three-level semantic IDs in
a single run:

    [1] Load the item text embeddings from parquet.
    [2] Train the RQ-VAE, codebook sizes [256, 256, 256].
    [3] Encode every item with the best checkpoint.
    [4] Save the SID matrix (.npy) plus an item_id -> code mapping (.csv).

Codes are produced with ``use_sk=False``, i.e. the plain nearest-code assignment
without the Sinkhorn balancing used during training, and a fourth all-zero column
is appended instead of a de-duplicating counter, so colliding items keep identical
SIDs. The collision statistics logged in stage 3 quantify how often this happens.

Use ``--skip_train`` to reuse an existing checkpoint, or ``--force_retrain`` to
retrain even when one is present. Default paths are relative to the repository and
can be overridden on the command line or via ``DRIFT_DATA_ROOT``.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
os.environ.setdefault("OMP_NUM_THREADS", "32")

import argparse
import collections
import logging
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.rqvae import RQVAE
from trainer import Trainer

# ================= Top-level configuration =================
CODEBOOK = [256, 256, 256]
DATASET_NAME = "kuairec"

# Expected column names in the embedding parquet.
COL_ITEM_ID = "item_id"
COL_EMB = "embedding"

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.environ.get("DRIFT_DATA_ROOT") or os.path.join(_HERE, os.pardir, "data")


# ================= Argument parsing =================
def parse_args():
    parser = argparse.ArgumentParser(description="KuaiRec Semantic-ID End-to-End Generation")

    # ---------- Training hyper-parameters ----------
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--learner', type=str, default='adamw')
    parser.add_argument('--lr_scheduler_type', type=str, default='linear')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--save_limit', type=int, default=3)
    parser.add_argument('--eval_step', type=int, default=50)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=20384)

    # ---------- Model architecture ----------
    parser.add_argument('--num_emb_list', type=int, nargs='+', default=CODEBOOK,
                        help='codebook sizes of each residual layer, default [256,256,256]')
    parser.add_argument('--e_dim', type=int, default=32)
    parser.add_argument('--layers', type=int, nargs='+', default=[512, 256, 128, 64])
    parser.add_argument('--dropout_prob', type=float, default=0.0)
    parser.add_argument('--bn', type=bool, default=False)

    # ---------- RQ-VAE specific ----------
    parser.add_argument('--quant_loss_weight', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.25)
    parser.add_argument('--loss_type', type=str, default='mse')
    parser.add_argument('--kmeans_init', type=bool, default=True)
    parser.add_argument('--kmeans_iters', type=int, default=100)
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.003])
    parser.add_argument('--sk_iters', type=int, default=50)

    # ---------- Paths and device ----------
    parser.add_argument('--dataset', type=str, default=DATASET_NAME,
                        choices=['taobao-mm', 'kuairec'])
    parser.add_argument('--data_path', type=str,
                        default=os.path.join(_DATA_ROOT, 'kuairec_processed',
                                             'item_embeddings.parquet'),
                        help='item text embedding parquet')
    parser.add_argument('--ckpt_dir', type=str,
                        default=os.path.join(
                            _HERE, 'ckpt',
                            f'{DATASET_NAME}_{CODEBOOK[0]}_{CODEBOOK[1]}_{CODEBOOK[2]}'))
    parser.add_argument('--output_file', type=str,
                        default=os.path.join(
                            _DATA_ROOT, 'kuairec_processed',
                            f'{DATASET_NAME}_rqvae_'
                            f'{CODEBOOK[0]}_{CODEBOOK[1]}_{CODEBOOK[2]}.npy'),
                        help='output .npy path for the semantic IDs')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=2024)

    # ---------- Control flags ----------
    parser.add_argument('--skip_train', action='store_true',
                        help='reuse an existing checkpoint and only run inference')
    parser.add_argument('--force_retrain', action='store_true',
                        help='retrain even if a checkpoint already exists')

    return parser.parse_args()


# ================= Utilities =================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(save_dir, "generate_code_kuairec.log"))
        ]
    )


class GPUDataLoader:
    """Minimal loader for an embedding table that stays resident on the GPU."""

    def __init__(self, data_tensor, batch_size, shuffle=True):
        self.data = data_tensor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = data_tensor.device

    def __iter__(self):
        n = self.data.shape[0]
        if self.shuffle:
            indices = torch.randperm(n, device=self.device)
        else:
            indices = torch.arange(n, device=self.device)
        for i in range(0, n, self.batch_size):
            yield self.data[indices[i: i + self.batch_size]]

    def __len__(self):
        return (self.data.shape[0] + self.batch_size - 1) // self.batch_size


def load_item_embeddings(data_path):
    """Read the item text embeddings from parquet.

    Returns ``(item_ids: np.ndarray[str], emb_matrix: np.ndarray[float32])``.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"[error] data file not found: {data_path}")

    df = pd.read_parquet(data_path)

    # embedding
    if COL_EMB in df.columns:
        emb_matrix = np.stack(df[COL_EMB].values).astype(np.float32)
    else:
        feat_cols = [c for c in df.columns if c != COL_ITEM_ID]
        emb_matrix = df[feat_cols].select_dtypes(include=[np.number]).values.astype(np.float32)

    # Item ids, keeping the original row order so they stay aligned with the codes.
    if COL_ITEM_ID in df.columns:
        item_ids = df[COL_ITEM_ID].astype(str).values
    else:
        item_ids = np.array([str(i) for i in range(len(df))])

    # Basic sanity checks.
    assert emb_matrix.ndim == 2, f"embedding must be 2D, got shape={emb_matrix.shape}"
    assert not np.isnan(emb_matrix).any(), "embedding contains NaN"
    assert len(item_ids) == emb_matrix.shape[0]

    return item_ids, emb_matrix


def get_latest_ckpt(base_dir, fname="best_collision_model.pth"):
    """Return the newest checkpoint; sub-directories are timestamped names."""
    if not os.path.exists(base_dir):
        return None
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not subdirs:
        return None
    subdirs.sort(reverse=True)
    candidate = os.path.join(base_dir, subdirs[0], fname)
    return candidate if os.path.exists(candidate) else None


# ================= Training stage =================
def train_rqvae(args, emb_matrix, logger):
    logger.info("========== [Stage 2/4] training the RQ-VAE ==========")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_tensor = torch.from_numpy(emb_matrix).float().to(device)
    train_loader = GPUDataLoader(data_tensor, args.batch_size, shuffle=True)

    input_dim = data_tensor.shape[1]
    model = RQVAE(
        in_dim=input_dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        beta=args.beta,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
    )

    trainer = Trainer(args, model, data_num=len(train_loader))
    logger.info("Starting Training ...")
    best_loss, best_coll = trainer.fit(train_loader)
    logger.info(f"[done] training finished. best loss: {best_loss}, "
                f"best collision rate: {best_coll}")

    # Release the training-stage GPU memory before inference.
    del data_tensor, model, trainer
    torch.cuda.empty_cache()


# ================= Inference stage =================
def generate_semantic_ids(args, item_ids, emb_matrix, logger):
    logger.info("========== [Stage 3/4] encoding items into semantic IDs ==========")

    ckpt_path = get_latest_ckpt(args.ckpt_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"[error] no usable checkpoint under {args.ckpt_dir}")
    logger.info(f"[info] loading checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
    args_model = ckpt["args"]
    state_dict = ckpt["state_dict"]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = RQVAE(
        in_dim=emb_matrix.shape[1],
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
        sk_iters=args_model.sk_iters,
    )
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    data_loader = DataLoader(
        TensorDataset(torch.from_numpy(emb_matrix)),
        batch_size=4096, shuffle=False, num_workers=4,
    )

    all_indices_raw, all_indices_str = [], []
    logger.info(f"[info] encoding all items (device: {device}, N={len(emb_matrix)}) ...")
    with torch.no_grad():
        for d in tqdm(data_loader):
            if isinstance(d, (list, tuple)):
                d = d[0]
            d = d.to(device)
            indices = model.get_indices(d, use_sk=False)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for idx in indices:
                all_indices_raw.append(idx.tolist())
                all_indices_str.append("-".join([str(int(x)) for x in idx]))

    # -------- Collision statistics --------
    n_total = len(all_indices_str)
    n_unique = len(set(all_indices_str))
    collision_cnt = n_total - n_unique
    logger.info("=" * 48)
    logger.info("SID collision report (all items)")
    logger.info(f"  items:               {n_total}")
    logger.info(f"  unique SIDs:         {n_unique}")
    logger.info(f"  collisions:          {collision_cnt}")
    logger.info(f"  collision rate:      {collision_cnt / max(n_total, 1):.4%}")
    if n_total:
        counts = collections.Counter(all_indices_str)
        logger.info(f"  max collision depth: {max(counts.values())}")
    logger.info("=" * 48)

    return np.array(all_indices_raw, dtype=np.int64), np.array(all_indices_str)


# ================= Saving stage =================
def save_results(args, item_ids, codes_array, codes_str, logger):
    logger.info("========== [Stage 4/4] saving the semantic IDs ==========")

    # Append a constant zero column instead of a de-duplicating counter, so that
    # colliding items keep identical SIDs downstream.
    padding = np.zeros((codes_array.shape[0], 1), dtype=np.int64)
    codes_array_save = np.hstack((codes_array, padding))

    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.save(args.output_file, codes_array_save)
    logger.info(f"[done] semantic IDs -> {args.output_file}")
    logger.info(f"   shape = {codes_array_save.shape} (last dim always 0)")
    logger.info(f"   sample:\n{codes_array_save[:3]}")

    # item_id -> code mapping, for inspection and downstream joins.
    mapping_path = args.output_file.replace(".npy", "_id2code.csv")
    pd.DataFrame({
        COL_ITEM_ID: item_ids,
        "code": codes_str,
    }).to_csv(mapping_path, index=False)
    logger.info(f"[done] id->code map -> {mapping_path}")


# ================= Main =================
if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    setup_logger(args.ckpt_dir)
    logger = logging.getLogger()

    logger.info("=" * 60)
    logger.info(f"[Config] Dataset     : {args.dataset}")
    logger.info(f"[Config] CODEBOOK    : {args.num_emb_list}")
    logger.info(f"[Config] Device      : {args.device}")
    logger.info(f"[Config] Data path   : {args.data_path}")
    logger.info(f"[Config] Ckpt dir    : {args.ckpt_dir}")
    logger.info(f"[Config] Output file : {args.output_file}")
    logger.info("=" * 60)

    # ---- Stage 1: load the item embeddings ----
    logger.info("========== [Stage 1/4] loading item embeddings ==========")
    item_ids, emb_matrix = load_item_embeddings(args.data_path)
    logger.info(f"[done] data loaded. shape={emb_matrix.shape}, #items={len(item_ids)}")

    # ---- Stage 2: train the RQ-VAE ----
    existing_ckpt = get_latest_ckpt(args.ckpt_dir)
    if args.force_retrain or (existing_ckpt is None and not args.skip_train):
        train_rqvae(args, emb_matrix, logger)
    elif args.skip_train and existing_ckpt is None:
        raise RuntimeError("--skip_train was given but no checkpoint was found, "
                           "so inference alone is not possible.")
    else:
        logger.info(f"[info] reusing the existing checkpoint, skipping training: "
                    f"{existing_ckpt}")

    # ---- Stage 3: encode items into semantic IDs ----
    codes_array, codes_str = generate_semantic_ids(args, item_ids, emb_matrix, logger)

    # ---- Stage 4: save ----
    save_results(args, item_ids, codes_array, codes_str, logger)

    logger.info("All done.")
