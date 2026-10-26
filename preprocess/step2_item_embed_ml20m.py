#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_ml20m_embed.py
========================================================================
Generate a text embedding for every unique item in the ML-20M dataset using a
sentence-embedding model.

Workflow:
    1. Read the item text fields from item_metadata.parquet and deduplicate by
       item_id;
    2. Clean / normalize the text;
    3. Encode in batches with the text-embedding model, using CLS pooling plus
       L2 normalization;
    4. Save both a numpy (.npy) and a parquet copy for downstream convenience.

Reference: generate_kuairec_embed.py
"""
from __future__ import annotations

import torch  # noqa: E402  pylint: disable=wrong-import-order
from torch.utils.data import DataLoader, Dataset  # noqa: E402

import argparse
import gc
import os
import re
import sys
import time
import unicodedata
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, BertModel, BertConfig


# ---------------------------------------------------------------------------
# Default paths (relative to the repository) and hyper-parameters
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.environ.get("DRIFT_DATA_ROOT") or os.path.join(_HERE, os.pardir, "data")
_PROCESSED = os.path.join(_DATA_ROOT, "ml20m_processed")

DEFAULT_INPUT_PARQUET = os.path.join(_PROCESSED, "item_metadata.parquet")
DEFAULT_OUTPUT_DIR = _PROCESSED
# Any sentence-embedding checkpoint loadable by transformers; hidden size 768.
DEFAULT_MODEL_DIR = os.environ.get("TEXT_EMBED_MODEL_DIR", "")

ID_COLUMN = "item_id"
TEXT_COLUMN = "item_text"

MODEL_MAX_LEN = 512
DEFAULT_BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Small helper: concise logging
# ---------------------------------------------------------------------------
def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1) Data loading
# ---------------------------------------------------------------------------
def load_item_text_frame(
    parquet_path: str = DEFAULT_INPUT_PARQUET,
    id_col: str = ID_COLUMN,
    text_col: str = TEXT_COLUMN,
) -> pd.DataFrame:
    """Load the item-level DataFrame."""
    usecols = [id_col, text_col]

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Input file not found: {parquet_path}")

    _log(f"Reading parquet: {parquet_path}")
    for engine in ("fastparquet", "pyarrow"):
        try:
            df = pd.read_parquet(parquet_path, columns=usecols, engine=engine)
            _log(f"  [done] successfully read with engine={engine}")
            break
        except Exception as e:
            _log(f"  [warn] engine={engine} failed to read ({type(e).__name__}): {e}")
            df = None

    if df is None:
        raise RuntimeError(f"Unable to read the parquet file: {parquet_path}")

    # Deduplicate by item_id
    df = df.drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True)
    df[id_col] = pd.to_numeric(df[id_col], errors="coerce")
    df = df.dropna(subset=[id_col]).reset_index(drop=True)
    df[id_col] = df[id_col].astype(np.int64)
    _log(f"  [done] items after deduplication: {len(df):,}")

    return df


# ---------------------------------------------------------------------------
# 2) Text preprocessing
# ---------------------------------------------------------------------------
_WHITESPACE = re.compile(r"\s+")
_INVISIBLE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")


def _clean_text(value: object) -> str:
    """Clean a single field value."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan" or text == "[]":
        return ""

    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if inner and not any(ch in inner for ch in "{}\""):
            text = inner.replace(",", " ")

    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE.sub("", text)
    text = "".join(ch for ch in text if ch >= " " or ch == "\n")
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def build_item_text(row_text: str) -> str:
    """Clean the text and build the natural-language description."""
    cleaned = _clean_text(row_text)
    if not cleaned:
        return "Unknown movie"
    return cleaned


# ---------------------------------------------------------------------------
# 3) Model inference
# ---------------------------------------------------------------------------
class _TextDataset(Dataset):
    """Memory-friendly text Dataset."""

    def __init__(self, texts: Sequence[str]):
        self._texts = texts

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, idx: int) -> str:
        return self._texts[idx]


def _select_device(force_cpu: bool = False) -> torch.device:
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def _load_model(model_dir: str, device: torch.device):
    _log(f"Loading the text-embedding model: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # AutoModel.from_pretrained in transformers 4.45.2 mishandles the metadata of
    # pytorch_model.bin, so we load manually instead:
    # BertConfig + BertModel + state_dict.
    config = BertConfig.from_pretrained(model_dir)
    model = BertModel(config)
    state_dict = torch.load(
        os.path.join(model_dir, "pytorch_model.bin"),
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(state_dict, strict=False)
    del state_dict
    gc.collect()

    model.eval()
    model.to(device)
    if device.type == "cuda":
        model.half()
    _log(
        f"  [done] model loaded onto {device}"
        + (", fp16" if device.type == "cuda" else ", fp32")
        + f", hidden_size={model.config.hidden_size}"
    )
    return tokenizer, model


@torch.no_grad()
def encode_texts(
    texts: Sequence[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_len: int = MODEL_MAX_LEN,
    show_progress: bool = True,
) -> np.ndarray:
    """Encode in batches: CLS pooling + L2 normalization."""
    hidden_size = model.config.hidden_size
    total = len(texts)
    embeddings = np.empty((total, hidden_size), dtype=np.float32)

    loader = DataLoader(
        _TextDataset(texts),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=list,
    )

    iter_obj: Iterable = loader
    if show_progress:
        iter_obj = tqdm(
            loader,
            total=(total + batch_size - 1) // batch_size,
            desc="Embedding",
            unit="batch",
        )

    offset = 0
    for batch_texts in iter_obj:
        enc = tokenizer(
            list(batch_texts),
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}

        outputs = model(**enc)
        cls = outputs.last_hidden_state[:, 0]
        cls = torch.nn.functional.normalize(cls.float(), p=2, dim=1)

        bsz = cls.size(0)
        embeddings[offset: offset + bsz] = cls.detach().cpu().numpy()
        offset += bsz

    assert offset == total, f"the number of encoded items {offset} does not match the input {total}"
    return embeddings


# ---------------------------------------------------------------------------
# 4) Saving the results
# ---------------------------------------------------------------------------
def save_embeddings(
    out_dir: str,
    item_ids: np.ndarray,
    embeddings: np.ndarray,
    texts: Sequence[str] | None = None,
    name: str = "item_embeddings",
) -> None:
    """Save the results into out_dir:
    - <name>.npy: the embedding matrix (N, D) float32
    - <name>_ids.json: the corresponding list of item_id values
    - <name>.parquet: a DataFrame (video_id + embedding + optional item_text)
    """
    import json
    os.makedirs(out_dir, exist_ok=True)

    npy_path = os.path.join(out_dir, f"{name}.npy")
    id_path = os.path.join(out_dir, f"{name}_ids.json")
    pq_path = os.path.join(out_dir, f"{name}.parquet")

    _log(f"Saving numpy: {npy_path}, shape={embeddings.shape}, dtype={embeddings.dtype}")
    np.save(npy_path, embeddings)

    _log(f"Saving ids: {id_path}")
    with open(id_path, "w") as f:
        json.dump(item_ids.astype(np.int64).tolist(), f)

    # parquet: keep the same layout as kuairec_embed, with columns
    # video_id + embedding
    data = {
        "video_id": item_ids.astype(np.int64),  # keep the column name the downstream code expects
        "embedding": [row.tolist() for row in embeddings],
    }
    if texts is not None:
        data["item_text"] = list(texts)

    df_out = pd.DataFrame(data)

    _log(f"Saving parquet: {pq_path}, rows={len(df_out)}")
    wrote = False
    for engine in ("fastparquet", "pyarrow"):
        try:
            df_out.to_parquet(pq_path, index=False, engine=engine)
            _log(f"  [done] parquet written via {engine}")
            wrote = True
            break
        except Exception as e:
            _log(f"  [warn] engine={engine} failed to write parquet ({type(e).__name__}): {e}")

    if not wrote:
        csv_path = os.path.join(out_dir, f"{name}.csv")
        _log(f"  [warn] all parquet writes failed, falling back to CSV: {csv_path}")
        df_out.to_csv(csv_path, index=False)

    _log("  [done] saving complete")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate ML-20M item text embeddings")
    p.add_argument("--input-parquet", default=DEFAULT_INPUT_PARQUET,
                   help="path to the input item_metadata.parquet")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help="output directory")
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                   help="text-embedding model directory")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                   help="encoding batch size")
    p.add_argument("--max-len", type=int, default=MODEL_MAX_LEN,
                   help="tokenizer maximum length")
    p.add_argument("--cpu", action="store_true", help="force the use of CPU")
    p.add_argument("--limit", type=int, default=None,
                   help="for debugging: encode only the first N items")
    p.add_argument("--name", default="item_embeddings",
                   help="output file name prefix")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    t0 = time.time()

    _log("=" * 72)
    _log("ML-20M item text embedding generation")
    _log("=" * 72)

    # Step 1: load and deduplicate
    df = load_item_text_frame(
        parquet_path=args.input_parquet,
    )

    # Step 2: text cleaning
    _log("Cleaning the text ...")
    df["item_text_clean"] = df[TEXT_COLUMN].apply(build_item_text)

    if args.limit is not None:
        df = df.head(args.limit).reset_index(drop=True)
        _log(f"DEBUG mode: processing only the first {len(df)} entries")

    item_ids = df[ID_COLUMN].to_numpy(dtype=np.int64)
    texts = df["item_text_clean"].tolist()
    del df
    gc.collect()

    # Step 3: model inference
    device = _select_device(force_cpu=args.cpu)
    tokenizer, model = _load_model(args.model_dir, device)

    _log(f"Starting encoding: {len(texts):,} texts, batch_size={args.batch_size}, max_len={args.max_len}")
    embeddings = encode_texts(
        texts,
        tokenizer,
        model,
        device,
        batch_size=args.batch_size,
        max_len=args.max_len,
    )
    _log(f"  [done] encoding complete: shape={embeddings.shape}")

    # Release the model's GPU memory
    del model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # Step 4: save
    save_embeddings(
        out_dir=args.output_dir,
        item_ids=item_ids,
        embeddings=embeddings,
        texts=texts,
        name=args.name,
    )

    _log(f"[done] all finished in {time.time() - t0:.1f} seconds")


if __name__ == "__main__":
    main()
