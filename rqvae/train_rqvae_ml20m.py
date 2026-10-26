"""RQ-VAE training entry (ML-20M).

Learns the residual-quantized semantic IDs used by the generative recommender.

* Input: ``<data_dir>/item_embeddings.parquet`` by default; alternatively an
  ``item_text_embeddings.npy`` + ``item_text_embedding_ids.json`` pair.
* Codebook: [256, 256, 256] (three levels).
* Checkpoints: ``<ckpt_dir>/<timestamp>/``.

All default paths are relative to the repository; override them on the command
line or via ``DRIFT_DATA_ROOT``.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
os.environ.setdefault("OMP_NUM_THREADS", "32")

import torch  # noqa: E402

import argparse
import json
import logging
import random
import shutil
import sys

import pandas as pd
import numpy as np
import torch
from trainer import Trainer
from models.rqvae import RQVAE

# ================= Top-level configuration =================
CODEBOOK = [256, 256, 256]
DATASET_NAME = "ml20m"

# Repository-relative default paths.
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.environ.get("DRIFT_DATA_ROOT") or os.path.join(_HERE, os.pardir, "data")


def parse_args():
    parser = argparse.ArgumentParser(description="High-Performance RQVAE Training (ML-20M)")

    # ---------- Core training ----------
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--learner', type=str, default='adamw')
    parser.add_argument('--lr_scheduler_type', type=str, default='linear')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--save_limit', type=int, default=3)
    parser.add_argument('--eval_step', type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=20384)

    # ---------- Model architecture ----------
    parser.add_argument('--num_emb_list', type=int, nargs='+', default=CODEBOOK,
                        help='codebook sizes, default [256,256,256]')
    parser.add_argument('--e_dim', type=int, default=32)
    parser.add_argument('--layers', type=int, nargs='+', default=[512, 256, 128, 64])
    parser.add_argument("--dropout_prob", type=float, default=0.0)
    parser.add_argument("--bn", type=bool, default=False)

    # ---------- RQ-VAE specific ----------
    parser.add_argument('--quant_loss_weight', type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--loss_type", type=str, default="mse")
    parser.add_argument("--kmeans_init", type=bool, default=True)
    parser.add_argument("--kmeans_iters", type=int, default=100)
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.003])
    parser.add_argument("--sk_iters", type=int, default=50)

    # ---------- Data and paths ----------
    parser.add_argument("--dataset", type=str, default=DATASET_NAME)
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(_DATA_ROOT, "ml20m_processed"),
                        help="directory holding the item embeddings")
    parser.add_argument("--embedding_path", type=str, default="",
                        help="explicit .npy path (optional)")
    parser.add_argument("--id_path", type=str, default="",
                        help="explicit ids .json path (optional)")
    parser.add_argument("--data_path", type=str, default="",
                        help="explicit parquet path (optional; takes precedence "
                             "over data_dir)")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(_HERE, "outputs", DATASET_NAME),
                        help="directory for the exported SID codes")
    parser.add_argument("--ckpt_dir", type=str,
                        default=os.path.join(
                            _HERE, "ckpt",
                            f"{DATASET_NAME}_{CODEBOOK[0]}_{CODEBOOK[1]}_{CODEBOOK[2]}"),
                        help="checkpoint root directory")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--num_quantizers", type=int, default=3)
    parser.add_argument("--embedding_dim", type=int, default=0,
                        help="0 = infer from the input embeddings")
    parser.add_argument("--hidden_dim", type=int, default=0, help="reserved")
    parser.add_argument("--debug", action="store_true",
                        help="debug mode: epochs=1, batch_size=256, eval_step=1")

    return parser.parse_args()


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
            logging.FileHandler(os.path.join(save_dir, "train.log")),
        ],
        force=True,
    )


class GPUDataLoader:
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


def load_embeddings(args, logger):
    """Load the item embeddings, trying the following sources in order:

    1) ``args.data_path`` (parquet)
    2) ``args.embedding_path`` + ``args.id_path`` (npy + json)
    3) ``args.data_dir/item_embeddings.parquet``
    4) ``args.data_dir/item_text_embeddings.npy`` +
       ``item_text_embedding_ids.json``

    Returns ``(item_ids: np.ndarray[int64], emb: np.ndarray[float32, (N, D)])``.
    """
    # 1) parquet
    candidates_pq = []
    if args.data_path:
        candidates_pq.append(args.data_path)
    candidates_pq.append(os.path.join(args.data_dir, "item_embeddings.parquet"))

    for pq in candidates_pq:
        if os.path.exists(pq):
            logger.info(f"[info] reading parquet: {pq}")
            try:
                df = pd.read_parquet(pq)
            except Exception as _e:
                logger.warning(f"[warn] failed to read parquet: {_e}; "
                               f"falling back to npy")
                df = None
            if df is None:
                emb = None
            elif "embedding" in df.columns:
                first = df["embedding"].iloc[0]
                if isinstance(first, (bytes, bytearray)):
                    import ast
                    parsed = [np.asarray(ast.literal_eval(v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v), dtype=np.float32) for v in df["embedding"].values]
                    emb = np.stack(parsed).astype(np.float32)
                elif isinstance(first, str):
                    import ast
                    parsed = [np.asarray(ast.literal_eval(v), dtype=np.float32) for v in df["embedding"].values]
                    emb = np.stack(parsed).astype(np.float32)
                else:
                    emb = np.stack(df["embedding"].values).astype(np.float32)
            else:
                emb = df.select_dtypes(include=[np.number]).values.astype(np.float32)
            if emb is None or df is None:
                continue
            if "video_id" in df.columns:
                vids = df["video_id"].astype(np.int64).values
            elif "item_id" in df.columns:
                vids = df["item_id"].astype(np.int64).values
            else:
                vids = np.arange(len(df), dtype=np.int64)
            return vids, emb

    # 2/4) npy
    npy_path = args.embedding_path or os.path.join(args.data_dir, "item_text_embeddings.npy")
    id_path = args.id_path or os.path.join(args.data_dir, "item_text_embedding_ids.json")
    if os.path.exists(npy_path):
        logger.info(f"[info] reading npy: {npy_path}")
        emb = np.load(npy_path).astype(np.float32)
        if os.path.exists(id_path):
            with open(id_path) as fp:
                vids = np.array(json.load(fp), dtype=np.int64)
        else:
            vids = np.arange(emb.shape[0], dtype=np.int64)
        return vids, emb

    raise FileNotFoundError(
        f"no usable item embedding source found: {candidates_pq} / {npy_path}"
    )


def copy_final_artifacts(args, ckpt_run_dir, logger):
    """Copy the best/last checkpoints and train.log into output_dir/checkpoints/."""
    out_ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(out_ckpt_dir, exist_ok=True)

    src_best = os.path.join(ckpt_run_dir, "best_collision_model.pth")
    if not os.path.exists(src_best):
        src_best = os.path.join(ckpt_run_dir, "best_loss_model.pth")
    if os.path.exists(src_best):
        dst_best = os.path.join(out_ckpt_dir, "best_model.pt")
        shutil.copy2(src_best, dst_best)
        logger.info(f"💾 best checkpoint -> {dst_best}")

    epoch_ckpts = [
        f for f in os.listdir(ckpt_run_dir)
        if f.startswith("epoch_") and f.endswith(".pth")
    ]
    if epoch_ckpts:
        epoch_ckpts.sort(key=lambda n: int(n.split("_")[1]))
        src_last = os.path.join(ckpt_run_dir, epoch_ckpts[-1])
        dst_last = os.path.join(out_ckpt_dir, "last_model.pt")
        shutil.copy2(src_last, dst_last)
        logger.info(f"💾 last checkpoint -> {dst_last}")
    elif os.path.exists(src_best):
        dst_last = os.path.join(out_ckpt_dir, "last_model.pt")
        shutil.copy2(src_best, dst_last)
        logger.info(f"💾 last checkpoint (fallback=best) -> {dst_last}")

    src_log = os.path.join(ckpt_run_dir, "train.log")
    if os.path.exists(src_log):
        shutil.copy2(src_log, os.path.join(args.output_dir, "train.log"))

    cfg = {k: (v if not isinstance(v, (np.integer, np.floating)) else float(v))
           for k, v in vars(args).items()}
    with open(os.path.join(args.output_dir, "config.json"), "w") as fp:
        json.dump(cfg, fp, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    args = parse_args()

    if not any(args.num_emb_list) or len(args.num_emb_list) != args.num_quantizers or \
            any(x != args.codebook_size for x in args.num_emb_list):
        if args.codebook_size != 256 or args.num_quantizers != 3:
            args.num_emb_list = [args.codebook_size] * args.num_quantizers

    if args.debug:
        args.epochs = 1
        args.batch_size = min(args.batch_size, 256)
        args.eval_step = 1
        args.warmup_epochs = 0

    set_seed(args.seed)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logger(args.output_dir)

    logger = logging.getLogger()
    logger.info(f"[Config] Dataset  : {args.dataset}")
    logger.info(f"[Config] CODEBOOK : {args.num_emb_list}")
    logger.info(f"[Config] Device   : {args.device}")
    logger.info(f"[Config] Data dir : {args.data_dir}")
    logger.info(f"[Config] Ckpt dir : {args.ckpt_dir}")
    logger.info(f"[Config] Output   : {args.output_dir}")
    logger.info(f"[Config] Debug    : {args.debug}")

    # 1) Load the item embeddings
    video_ids, data_numpy = load_embeddings(args, logger)
    logger.info(f"Data shape: {data_numpy.shape}  #items={len(video_ids)}")

    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("[warn] CUDA unavailable, falling back to CPU")
    args.device = device

    data_tensor = torch.from_numpy(data_numpy).float().to(device)

    # 2) Batch iterator
    train_loader = GPUDataLoader(data_tensor, args.batch_size, shuffle=True)

    # 3) Model
    model = RQVAE(
        in_dim=data_tensor.shape[1],
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

    # 4. Trainer
    trainer = Trainer(args, model, data_num=len(train_loader))
    logger.info("Starting Training ...")
    best_loss, best_coll = trainer.fit(train_loader)
    logger.info(f"✅ Training Finished. Best Loss: {best_loss}, Best Collision: {best_coll}")

    # 5) Copy the final artifacts
    copy_final_artifacts(args, trainer.ckpt_dir, logger)
    logger.info("🎉 All done.")
