# -*- coding: utf-8 -*-
"""Timestamp-aware Dataset / DataLoader wrappers.

The adaptive partition module can consume real interaction timestamps to build
time-aware boundaries. Timestamps are stored as an extra ``timestamps`` column in
the processed parquet files (see ``preprocess/step5_inject_timestamps.py``).

These wrappers subclass the base ``GenRecDataset`` / ``GenRecDataLoader`` so that
each batch carries an additional ``timestamps`` tensor of shape ``(B, L)``. When no
timestamp column is available the tensor is filled with zeros, in which case the
temporal terms in the structure loss degenerate to a constant (zero gradient),
i.e. they are effectively disabled.
"""

from __future__ import annotations

import os
import glob

import numpy as np
import pandas as pd
import torch


def pad_or_truncate_ts(ts_list, max_len, pad_value=0):
    """Left-pad (or left-truncate) a timestamp sequence to ``max_len``.

    Padding is applied on the left so that the most recent interactions stay
    aligned with the right end of the sequence, matching the item-token padding
    convention in ``model/dataset.py::pad_or_truncate``.
    """
    ts = list(ts_list)
    if len(ts) > max_len:
        return ts[-max_len:]
    return [pad_value] * (max_len - len(ts)) + ts


def _coerce_ts_row(raw, max_len):
    """Normalize one stored timestamp cell into a float32 array of ``max_len``."""
    if raw is None:
        return np.zeros(max_len, dtype=np.float32)
    if isinstance(raw, str):
        raw = [float(x) for x in raw.split(",") if x.strip()]
    padded = pad_or_truncate_ts(np.asarray(raw, dtype=np.float32), max_len, pad_value=0)
    return np.asarray(padded, dtype=np.float32)


def load_timestamps_into_dataset(dataset, parquet_path, max_len):
    """Attach timestamps read from ``parquet_path`` onto ``dataset``.

    Used when the training script already knows the exact parquet file (which may
    carry a sequence-length prefix), so filename globbing is unnecessary.
    """
    dataset.ts_max_len = max_len
    dataset._timestamps = None

    if not os.path.exists(parquet_path):
        print(f"   [warn] timestamp file not found: {parquet_path}; using zeros")
        return

    df = pd.read_parquet(parquet_path)
    if "timestamps" not in df.columns:
        print(f"   [warn] no 'timestamps' column in {parquet_path}; using zeros")
        return

    dataset._timestamps = [_coerce_ts_row(r, max_len) for r in df["timestamps"].tolist()]
    print(f"   [info] loaded {len(dataset._timestamps)} timestamp rows from {parquet_path}")


def make_timestamped_dataset_class(base_dataset_cls, base_dataloader_cls):
    """Build timestamp-aware subclasses of the given Dataset / DataLoader."""

    class TimestampedGenRecDataset(base_dataset_cls):
        """``GenRecDataset`` plus a per-sample ``timestamps`` field."""

        def __init__(self, dataset_path, mode, max_len, sim_max_len=30,
                     codebook_size=128, target_depth=3, PAD_TOKEN=0, ts_max_len=None,
                     autoload_ts=True):
            super().__init__(dataset_path=dataset_path, mode=mode, max_len=max_len,
                             sim_max_len=sim_max_len, codebook_size=codebook_size,
                             target_depth=target_depth, PAD_TOKEN=PAD_TOKEN)
            self.ts_max_len = ts_max_len or max_len
            self._timestamps = None

            # When the caller knows the exact split file it should pass
            # ``autoload_ts=False`` and call ``load_timestamps_into_dataset``
            # directly; the glob below cannot distinguish valid from test.
            if not autoload_ts:
                return

            if mode == "train":
                pattern = "train*sim_augmented.parquet"
            elif mode == "evaluation":
                pattern = "valid*with_history.parquet"
            else:
                pattern = "test*with_history.parquet"

            ts_dir = dataset_path
            if os.path.isfile(ts_dir):
                ts_dir = os.path.dirname(ts_dir)
            if not ts_dir.endswith("_ts"):
                ts_dir = os.path.join(os.path.dirname(ts_dir),
                                      os.path.basename(ts_dir) + "_ts")

            matches = sorted(glob.glob(os.path.join(ts_dir, pattern)))
            if matches:
                load_timestamps_into_dataset(self, matches[0], self.ts_max_len)

        def __getitem__(self, index):
            item = super().__getitem__(index)
            if self._timestamps is not None and index < len(self._timestamps):
                item["timestamps"] = self._timestamps[index]
            else:
                item["timestamps"] = np.zeros(self.ts_max_len, dtype=np.float32)
            return item

    class TimestampedGenRecDataLoader(base_dataloader_cls):
        """``GenRecDataLoader`` that additionally collates ``timestamps``."""

        def collate_fn(self, batch, pad_token=0):
            result = super().collate_fn(batch, pad_token=pad_token)
            ts = [item.get("timestamps", np.zeros(1, dtype=np.float32)) for item in batch]
            result["timestamps"] = torch.tensor(np.stack(ts), dtype=torch.float32)
            return result

    return TimestampedGenRecDataset, TimestampedGenRecDataLoader
