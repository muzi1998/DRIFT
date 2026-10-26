"""RQ-VAE training entry (TAOBAO-MM).

Learns the residual-quantized semantic IDs (SIDs) used by the generative
recommender. Item text/multimodal embeddings are quantized into three levels with
codebook sizes [64, 128, 128].

* Input:  ``<data_dir>/rqvae_data/item_emb.parquet`` (an ``embedding`` column, or
          otherwise all numeric columns).
* Output: checkpoints under ``<ckpt_dir>/``; the SID codes themselves are exported
          afterwards by ``generate_sid_taobaomm.py``.

Default paths are relative to the repository; override them on the command line or
via ``DRIFT_DATA_ROOT``.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
os.environ.setdefault("OMP_NUM_THREADS", "32")

import argparse
import logging
import random
import sys

import numpy as np
import pandas as pd
import torch

from trainer import Trainer
from models.rqvae import RQVAE

# ================= Top-level configuration =================
# Defined at module level so that the argparse defaults below can reference it.
CODEBOOK = [64, 128, 128]
DATASET_NAME = "taobao-mm"

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.environ.get("DRIFT_DATA_ROOT") or os.path.join(_HERE, os.pardir, "data")


def parse_args():
    parser = argparse.ArgumentParser(description="RQ-VAE training (TAOBAO-MM)")

    # ================= Core training =================
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--learner', type=str, default='adamw', help='optimizer: adam, sgd, adamw')
    parser.add_argument('--lr_scheduler_type', type=str, default='linear', help='scheduler type')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='warmup epochs')
    parser.add_argument('--save_limit', type=int, default=3, help='max checkpoints to keep')
    parser.add_argument('--eval_step', type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=20384)

    # ================= Model architecture =================
    parser.add_argument('--num_emb_list', type=int, nargs='+', default=CODEBOOK,
                        help='per-level codebook sizes')
    parser.add_argument('--e_dim', type=int, default=32, help='code embedding dim')
    parser.add_argument('--layers', type=int, nargs='+', default=[512, 256, 128, 64],
                        help='encoder MLP widths (the decoder mirrors them)')
    parser.add_argument("--dropout_prob", type=float, default=0.0)
    parser.add_argument("--bn", type=bool, default=False)

    # ================= RQ-VAE specific =================
    parser.add_argument('--quant_loss_weight', type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.25, help='commitment loss weight')
    parser.add_argument("--loss_type", type=str, default="mse")
    parser.add_argument("--kmeans_init", type=bool, default=True,
                        help='initialize codebooks with k-means')
    parser.add_argument("--kmeans_iters", type=int, default=100)
    # Sinkhorn regularization is applied to the last level only, which balances
    # its code usage and reduces SID collisions.
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.003])
    parser.add_argument("--sk_iters", type=int, default=50)

    # ================= Paths =================
    parser.add_argument("--dataset", type=str, default=DATASET_NAME)
    parser.add_argument("--data_path", type=str,
                        default=os.path.join(_DATA_ROOT, "TAOBAO_MM", "rqvae_data",
                                             "item_emb.parquet"),
                        help='item embedding parquet')
    parser.add_argument("--ckpt_dir", type=str,
                        default=os.path.join(
                            _HERE, "ckpt",
                            f"taobao_{CODEBOOK[0]}_{CODEBOOK[1]}_{CODEBOOK[2]}"),
                        help='checkpoint directory')
    parser.add_argument("--device", type=str, default="cuda")

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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(save_dir, "train.log"))
        ]
    )


class GPUDataLoader:
    """Minimal loader for a dataset that fits entirely in GPU memory.

    The item embedding table is small enough to be resident on the device, so we
    avoid ``torch.utils.data.DataLoader`` (whose worker processes do not handle
    GPU tensors well) and simply index a permutation on-device.
    """

    def __init__(self, data_tensor, batch_size, shuffle=True):
        self.data = data_tensor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = data_tensor.device

    def __iter__(self):
        n = self.data.shape[0]
        # Build the index permutation directly on the device.
        if self.shuffle:
            indices = torch.randperm(n, device=self.device)
        else:
            indices = torch.arange(n, device=self.device)

        for i in range(0, n, self.batch_size):
            batch_indices = indices[i: i + self.batch_size]
            yield self.data[batch_indices]

    def __len__(self):
        return (self.data.shape[0] + self.batch_size - 1) // self.batch_size


if __name__ == '__main__':
    # 1) Setup
    args = parse_args()
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    setup_logger(args.ckpt_dir)

    logger = logging.getLogger()
    logger.info(f"Using device: {args.device}")

    # 2) Load the item embeddings onto the device
    logger.info(f"Loading data from {args.data_path} ...")
    df = pd.read_parquet(args.data_path)

    # The embedding column may hold arrays/lists; otherwise use all numeric columns.
    if 'embedding' in df.columns:
        data_numpy = np.stack(df['embedding'].values)
    else:
        data_numpy = df.select_dtypes(include=[np.number]).values

    data_tensor = torch.from_numpy(data_numpy).float().to(args.device)
    logger.info(f"Data shape: {data_tensor.shape}")

    # 3) Wrap into a batch iterator
    train_loader = GPUDataLoader(data_tensor, args.batch_size, shuffle=True)

    # 4) Build the model; input dim is inferred from the embeddings
    input_dim = data_tensor.shape[1]

    model = RQVAE(in_dim=input_dim,
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
                  sk_iters=args.sk_iters)

    # 5) Train
    trainer = Trainer(args, model, data_num=len(train_loader))

    logger.info("Starting training ...")
    best_loss, best_coll = trainer.fit(train_loader)
    logger.info(f"Training finished. Best loss: {best_loss}, best collision rate: {best_coll}")
