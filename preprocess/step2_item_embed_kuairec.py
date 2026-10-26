#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_kuairec_embed.py
========================================================================
Generate a text embedding for every unique item (video_id) in the KuaiRec
dataset using a Chinese sentence-embedding model.

Workflow:
    1. Read the item text fields from merged_all.(parquet|csv) and deduplicate
       by video_id;
    2. Clean / normalize / concatenate the text;
    3. Encode in batches with the text-embedding model, using CLS pooling plus
       L2 normalization;
    4. Save both a numpy (.npy) and a parquet copy for downstream convenience.

Usage:
    python preprocess/step2_item_embed_kuairec.py \
        --input <processed_dir>/merged_all.parquet \
        --output_dir <processed_dir> \
        --model_dir <path/to/text-embedding-model>
"""
from __future__ import annotations

# torch must be imported before pandas/numpy on some systems with an older
# GLIBC/GLIBCXX: the libgomp that pandas preloads can otherwise crash torch when
# it loads its CUDA libraries.
import torch  # noqa: E402  pylint: disable=wrong-import-order
from torch.utils.data import DataLoader, Dataset  # noqa: E402

import argparse
import gc
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


# ---------------------------------------------------------------------------
# Default paths (relative to the repository) and hyper-parameters
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.environ.get("DRIFT_DATA_ROOT") or os.path.join(_HERE, os.pardir, "data")
_PROCESSED = os.path.join(_DATA_ROOT, "kuairec_processed")

DEFAULT_INPUT_PARQUET = os.path.join(_PROCESSED, "merged_all.parquet")
DEFAULT_INPUT_CSV = os.path.join(_PROCESSED, "merged_all.csv")
DEFAULT_OUTPUT_DIR = _PROCESSED
# Any sentence-embedding checkpoint loadable by transformers.AutoModel; we used a
# publicly available Chinese text-embedding model of hidden size 768.
DEFAULT_MODEL_DIR = os.environ.get("TEXT_EMBED_MODEL_DIR", "")

# Text fields used to encode each item (concatenated into a single string in
# this priority order)
TEXT_FIELDS: List[str] = [
    "caption",
    "topic_tag",
    "first_level_category_name",
    "second_level_category_name",
]

ID_COLUMN = "video_id"

# This model has hidden_size = 768 and max_position = 512; 2 slots are reserved
# for [CLS]/[SEP]
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
    csv_path: str = DEFAULT_INPUT_CSV,
    id_col: str = ID_COLUMN,
    text_fields: Sequence[str] = TEXT_FIELDS,
) -> pd.DataFrame:
    """Load the item-level DataFrame: deduplicate by `id_col` and keep only
    id_col plus the text fields.

    Parquet is tried first (smaller and faster to read); on failure we fall back
    to CSV.
    """
    usecols = [id_col] + [c for c in text_fields]

    df: pd.DataFrame | None = None
    if os.path.exists(parquet_path):
        _log(f"Reading parquet: {parquet_path}")
        # Note: on some systems pyarrow segfaults outright because of a GLIBC
        # version mismatch (it does not raise a Python exception), so we try
        # fastparquet first (pure Python, better compatibility), then pyarrow,
        # and finally fall back to CSV.
        for engine in ("fastparquet", "pyarrow"):
            try:
                df = pd.read_parquet(parquet_path, columns=usecols, engine=engine)
                _log(f"  [done] successfully read with engine={engine}")
                break
            except Exception as e:
                _log(f"  [warn] engine={engine} failed to read ({type(e).__name__}): {e}")
                df = None

    if df is None:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Neither a readable parquet nor a CSV could be found: {parquet_path} / {csv_path}"
            )
        _log(f"Reading CSV: {csv_path}")
        df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)

    _log(f"  interaction-level data shape: {df.shape}")

    # Deduplicate by id (the same video_id appears many times in the interaction
    # data, always with identical text fields).
    df = df.drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True)
    df[id_col] = pd.to_numeric(df[id_col], errors="coerce")
    df = df.dropna(subset=[id_col]).reset_index(drop=True)
    df[id_col] = df[id_col].astype(np.int64)
    _log(f"  [done] items after deduplication: {len(df):,}")

    return df


# ---------------------------------------------------------------------------
# 2) Text preprocessing and concatenation
# ---------------------------------------------------------------------------
_WHITESPACE = re.compile(r"\s+")
_INVISIBLE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")


def _clean_text(value: object) -> str:
    """Clean a single field value:
    - normalize NaN / None / "nan" / "[]" to the empty string
    - apply Unicode NFKC normalization and strip zero-width characters
    - remove control characters and collapse redundant whitespace
    - for a topic_tag shaped like "[a,b,c]", drop the surrounding brackets and
      replace the commas with spaces
    """
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
    # Remove control characters
    text = "".join(ch for ch in text if ch >= " " or ch == "\n")
    text = _WHITESPACE.sub(" ", text).strip()
    return text


_FIELD_PREFIX = {
    "caption": "标题",
    "topic_tag": "标签",
    "first_level_category_name": "一级类目",
    "second_level_category_name": "二级类目",
}


def build_item_text(row: pd.Series, text_fields: Sequence[str] = TEXT_FIELDS) -> str:
    """Concatenate several fields into one natural-language description that a
    general-purpose embedding model can interpret.

    The result looks like "<title>: xxx . <tag>: a b c . <level-1 category>: yyy
    . <level-2 category>: zzz". The field prefixes are deliberately kept in
    Chinese because the encoder is a Chinese text-embedding model and the
    prefixes are part of its input, not log output. Missing fields are skipped
    entirely, so no empty prefix is left behind.
    """
    parts: List[str] = []
    for field in text_fields:
        cleaned = _clean_text(row.get(field))
        if not cleaned:
            continue
        prefix = _FIELD_PREFIX.get(field, field)
        parts.append(f"{prefix}: {cleaned}")
    if not parts:
        return "无描述"
    return " 。 ".join(parts)


def attach_item_texts(df: pd.DataFrame, text_fields: Sequence[str] = TEXT_FIELDS) -> pd.DataFrame:
    """Add an `item_text` column to the DataFrame."""
    _log("Concatenating the text fields -> item_text ...")
    df = df.copy()
    df["item_text"] = df.apply(lambda r: build_item_text(r, text_fields), axis=1)

    empties = int((df["item_text"] == "无描述").sum())
    avg_len = float(df["item_text"].str.len().mean())
    max_len = int(df["item_text"].str.len().max())
    _log(
        f"  [done] text generated: {len(df):,} entries in total, {empties} with an empty description, "
        f"mean character count {avg_len:.1f}, longest {max_len}"
    )
    return df


# ---------------------------------------------------------------------------
# 3) Model inference
# ---------------------------------------------------------------------------
class _TextDataset(Dataset):
    """Memory-friendly text Dataset: holds a reference rather than a copy."""

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
    model = AutoModel.from_pretrained(model_dir)
    model.eval()
    model.to(device)
    # Use fp16 at inference time to reduce GPU memory usage (CUDA only).
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
    """Encode in batches: CLS pooling + L2 normalization, returning a float32
    array of shape (N, hidden_size)."""
    hidden_size = model.config.hidden_size
    total = len(texts)
    embeddings = np.empty((total, hidden_size), dtype=np.float32)

    loader = DataLoader(
        _TextDataset(texts),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=list,  # return the raw list of strings
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

        # As recommended by the model authors, take the [CLS] position of
        # last_hidden_state as the sentence vector.
        cls = outputs.last_hidden_state[:, 0]
        cls = torch.nn.functional.normalize(cls.float(), p=2, dim=1)

        bsz = cls.size(0)
        embeddings[offset : offset + bsz] = cls.detach().cpu().numpy()
        offset += bsz

    assert offset == total, f"the number of encoded items {offset} does not match the input {total}"
    return embeddings


# ---------------------------------------------------------------------------
# 4) Saving the results
# ---------------------------------------------------------------------------
def save_embeddings(
    out_dir: str,
    video_ids: np.ndarray,
    embeddings: np.ndarray,
    texts: Sequence[str] | None = None,
    name: str = "item_embeddings",
) -> None:
    """Save the results into out_dir:
    - <name>.npy: the embedding matrix (N, D) float32
    - <name>_ids.npy: the corresponding video_id values (N,) int64
    - <name>.parquet: a DataFrame (video_id + embedding + optional item_text)
    """
    os.makedirs(out_dir, exist_ok=True)

    npy_path = os.path.join(out_dir, f"{name}.npy")
    id_path = os.path.join(out_dir, f"{name}_ids.npy")
    pq_path = os.path.join(out_dir, f"{name}.parquet")

    _log(f"Saving numpy: {npy_path}, shape={embeddings.shape}, dtype={embeddings.dtype}")
    np.save(npy_path, embeddings)
    np.save(id_path, video_ids.astype(np.int64))

    data = {
        "video_id": video_ids.astype(np.int64),
        "embedding": [row.tolist() for row in embeddings],
    }
    if texts is not None:
        data["item_text"] = list(texts)

    df_out = pd.DataFrame(data)

    _log(f"Saving parquet: {pq_path}, rows={len(df_out)}")
    wrote = False
    # As on the read path, prefer fastparquet to avoid a pyarrow segfault on
    # some systems.
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
    p = argparse.ArgumentParser(description="Generate KuaiRec item text embeddings")
    p.add_argument("--input-parquet", default=DEFAULT_INPUT_PARQUET,
                   help="input parquet path (preferred)")
    p.add_argument("--input-csv", default=DEFAULT_INPUT_CSV,
                   help="input CSV path (fallback when parquet is unavailable)")
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
    _log("KuaiRec item text embedding generation")
    _log("=" * 72)

    # Step 1: load and deduplicate
    df = load_item_text_frame(
        parquet_path=args.input_parquet,
        csv_path=args.input_csv,
    )

    # Step 2: text concatenation
    df = attach_item_texts(df)

    if args.limit is not None:
        df = df.head(args.limit).reset_index(drop=True)
        _log(f"DEBUG mode: processing only the first {len(df)} entries")

    video_ids = df[ID_COLUMN].to_numpy(dtype=np.int64)
    texts = df["item_text"].tolist()
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
        video_ids=video_ids,
        embeddings=embeddings,
        texts=texts,
        name=args.name,
    )

    _log(f"[done] all finished in {time.time() - t0:.1f} seconds")


if __name__ == "__main__":
    main()