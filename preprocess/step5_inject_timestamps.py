#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 5 - inject interaction timestamps into the processed parquet files.

The adaptive partition module can use real interaction timestamps for its temporal
terms (``log(1 + dt)`` in the shift descriptor, and the temporal-continuity
statistic in the structure loss). This script reads the timestamps from the raw
dataset, aligns them with the ``history`` column of the processed parquet files
produced by ``step4_build_*.py``, and writes a *new* directory with an extra
``timestamps`` column. Existing files are never modified or overwritten.

Pipeline::

    raw dataset (interaction timestamps)
      -> align one timestamp per history item
         -> new parquet with a `timestamps` column, in <dataset_dir>_ts
            -> model/timestamped_data.py exposes it as batch["timestamps"]
               -> SoftPartition(history_ts_long=...) enables time-aware boundaries

Output format
-------------
``timestamps``: list of Unix timestamps in **seconds**, one per history *item*
(not per SID token), left-padded with 0 to match the item padding convention of
``model/dataset.py``.

Datasets
--------
* KuaiRec — the raw sequence file stores comma-separated millisecond timestamps in
  ``sequence_timestamps``.
* ML-20M  — ``ratings.csv`` stores second-level Unix timestamps; sequences are
  rebuilt by sorting each user's interactions by timestamp.

TAOBAO-MM does not ship usable interaction timestamps, so no ``_ts`` directory is
built for it; training is launched with ``--use_timestamps 0``, in which case the
temporal terms degenerate to a constant and contribute no gradient.

Usage::

    python preprocess/step5_inject_timestamps.py --dataset kuairec \\
        --raw <path/to/kuairec_sequences.csv> \\
        --dataset_dir ../data/kuairec_processed/glass_data_256_256_256 \\
        --max_len 1000

    python preprocess/step5_inject_timestamps.py --dataset ml20m \\
        --raw <path/to/ml-20m/ratings.csv> \\
        --dataset_dir ../data/ml20m_processed_full_seqlen200/glass_data_256_256_256 \\
        --max_len 200
"""

import argparse
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.environ.get("DRIFT_DATA_ROOT") or os.path.join(_HERE, os.pardir, "data")


# ============================================================
# Helpers
# ============================================================
def parse_timestamp_list(ts_str, delimiter=","):
    """Parse a delimited timestamp string (or array-like) into int64."""
    if isinstance(ts_str, str):
        parts = ts_str.strip().split(delimiter)
        return np.array([int(float(p.strip())) for p in parts if p.strip()],
                        dtype=np.int64)
    if isinstance(ts_str, (list, np.ndarray)):
        return np.array(ts_str, dtype=np.int64)
    return np.array([], dtype=np.int64)


def normalize_to_seconds(ts_array):
    """Convert millisecond timestamps to seconds when needed.

    Heuristic: values above 1e12 are treated as milliseconds.
    """
    if len(ts_array) > 0 and ts_array.max() > 1e12:
        return ts_array // 1000
    return ts_array


def pad_or_truncate_ts(ts_list, max_len, pad_value=0):
    """Left-pad / left-truncate, matching ``model/dataset.py::pad_or_truncate``."""
    if len(ts_list) > max_len:
        return list(ts_list[-max_len:])
    return [pad_value] * (max_len - len(ts_list)) + list(ts_list)


def _write_with_timestamps(df, all_timestamps, out_path, matched, unmatched):
    df_ts = df.copy()
    df_ts["timestamps"] = all_timestamps
    df_ts.to_parquet(out_path, index=False)
    print(f"   [done] {out_path} (matched={matched}, unmatched={unmatched})")


# ============================================================
# KuaiRec
# ============================================================
def build_kuairec_timestamps(dataset_dir, raw_csv, output_dir, max_len=1000):
    """Align KuaiRec timestamps with the processed ``history`` sequences.

    Because the processed files store SID tokens rather than raw item ids, the
    alignment is positional: each user's last ``n_items`` timestamps are matched to
    the tail of the history, consistent with the left-padding convention.
    """
    print(f"[KuaiRec] reading raw sequences: {raw_csv}")
    raw_df = pd.read_csv(raw_csv)
    print(f"   {len(raw_df)} rows, columns: {list(raw_df.columns)}")

    # user_id -> full timestamp sequence (seconds, ascending)
    user_ts_seq = {}
    for row in raw_df.itertuples(index=False):
        uid = int(row.user_id)
        ts = normalize_to_seconds(parse_timestamp_list(row.sequence_timestamps))
        user_ts_seq[uid] = ts
    print(f"   built timestamp sequences for {len(user_ts_seq)} users")

    os.makedirs(output_dir, exist_ok=True)
    for fname in sorted(os.listdir(dataset_dir)):
        if not fname.endswith(".parquet"):
            continue
        src_path = os.path.join(dataset_dir, fname)
        out_path = os.path.join(output_dir, fname)
        if os.path.exists(out_path):
            print(f"   [skip] already exists: {out_path}")
            continue

        print(f"   processing: {fname}")
        df = pd.read_parquet(src_path)

        all_timestamps, matched, unmatched = [], 0, 0
        for row in df.itertuples(index=False):
            user_id = int(row.user)
            history = list(row.history) if row.history is not None else []
            # history holds [sid1, sid2, sid3] per item -> 3 tokens per item.
            n_items = len(history) // 3

            ts = user_ts_seq.get(user_id)
            if ts is not None and len(ts) > 0:
                item_ts = ts[-n_items:] if len(ts) >= n_items else ts
                all_timestamps.append(pad_or_truncate_ts(item_ts, max_len, 0))
                matched += 1
            else:
                all_timestamps.append([0] * max_len)
                unmatched += 1

        _write_with_timestamps(df, all_timestamps, out_path, matched, unmatched)

    print(f"[KuaiRec] finished; output directory: {output_dir}")


# ============================================================
# ML-20M
# ============================================================
def build_ml20m_timestamps(dataset_dir, raw_csv, output_dir, max_len=200):
    """Align ML-20M timestamps with the processed ``history`` sequences.

    ``ratings.csv`` columns: ``userId, movieId, rating, timestamp``. Interactions
    are grouped by user and sorted by timestamp, then aligned positionally to the
    tail of each processed history.
    """
    print(f"[ML-20M] reading ratings: {raw_csv}")
    ratings = pd.read_csv(raw_csv)
    ratings = ratings.sort_values(["userId", "timestamp"])

    user_ts = {}
    for uid, group in ratings.groupby("userId"):
        user_ts[int(uid)] = group["timestamp"].values.astype(np.int64)
    print(f"   built timestamp sequences for {len(user_ts)} users")

    os.makedirs(output_dir, exist_ok=True)
    for fname in sorted(os.listdir(dataset_dir)):
        if not fname.endswith(".parquet"):
            continue
        src_path = os.path.join(dataset_dir, fname)
        out_path = os.path.join(output_dir, fname)
        if os.path.exists(out_path):
            print(f"   [skip] already exists: {out_path}")
            continue

        print(f"   processing: {fname}")
        df = pd.read_parquet(src_path)

        all_timestamps, matched, unmatched = [], 0, 0
        for row in df.itertuples(index=False):
            user_id = int(row.user)
            history = list(row.history) if row.history is not None else []
            n_items = len(history) // 3

            ts_seq = user_ts.get(user_id)
            if ts_seq is not None and len(ts_seq) > 0:
                item_ts = ts_seq[-n_items:] if len(ts_seq) >= n_items else ts_seq
                all_timestamps.append(pad_or_truncate_ts(item_ts, max_len, 0))
                matched += 1
            else:
                all_timestamps.append([0] * max_len)
                unmatched += 1

        _write_with_timestamps(df, all_timestamps, out_path, matched, unmatched)

    print(f"[ML-20M] finished; output directory: {output_dir}")


# ============================================================
# Entry point
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="Inject interaction timestamps into processed parquet files "
                    "(writes a new '<dataset_dir>_ts' directory).")
    ap.add_argument("--dataset", choices=["kuairec", "ml20m"], required=True)
    ap.add_argument("--raw", required=True,
                    help="raw sequence CSV (KuaiRec) or ratings.csv (ML-20M)")
    ap.add_argument("--dataset_dir", required=True,
                    help="processed dataset directory produced by step4_build_*.py")
    ap.add_argument("--out_dir", default="",
                    help="output directory; defaults to '<dataset_dir>_ts'")
    ap.add_argument("--max_len", type=int, default=0,
                    help="timestamp sequence length; defaults to 1000 (KuaiRec) "
                         "or 200 (ML-20M)")
    ap.add_argument("--dry_run", action="store_true",
                    help="only print the resolved paths")
    args = ap.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else dataset_dir + "_ts"
    max_len = args.max_len or (1000 if args.dataset == "kuairec" else 200)

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"dataset directory not found: {dataset_dir}")
    if not os.path.exists(args.raw):
        raise FileNotFoundError(f"raw file not found: {args.raw}")

    if args.dry_run:
        print(f"[dry run] dataset={args.dataset} raw={args.raw}\n"
              f"          dataset_dir={dataset_dir}\n"
              f"          out_dir={out_dir} max_len={max_len}")
        return

    if args.dataset == "kuairec":
        build_kuairec_timestamps(dataset_dir, args.raw, out_dir, max_len)
    else:
        build_ml20m_timestamps(dataset_dir, args.raw, out_dir, max_len)

    print("\nAll done.")


if __name__ == "__main__":
    main()
