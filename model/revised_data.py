# -*- coding: utf-8 -*-
"""Minimal SID data pipeline for the revised DRIFT training entry.

Raw SID triples are offset before PAD tokens are added. Consequently a real
``[0, 0, 0]`` item remains valid. Only history, target and optional timestamps
are read; no retrieval or SID-tier features are constructed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _codebook_array(codebook):
    values = np.asarray(codebook)
    if values.shape != (3,) or values.dtype.kind not in "iu" or np.any(values <= 0):
        raise ValueError("codebook must contain three positive integer sizes")
    return values.astype(np.int64)


def sid_codes(value, codebook, *, sequence):
    """Validate raw three-level SID codes using each level's actual range."""
    sizes = _codebook_array(codebook)
    try:
        raw = np.asarray(value)
        if sequence and raw.size == 0:
            raw = raw.reshape(0, 3)
        expected = raw.ndim == 2 and raw.shape[1] == 3 if sequence else raw.shape == (3,)
        if not expected or raw.dtype.kind not in "iuf":
            raise ValueError("Expected numeric raw SID triples")
        if not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw >= sizes):
            raise ValueError("SID code outside its level's codebook range")
        codes = raw.astype(np.int64)
        if not np.all(raw == codes):
            raise ValueError("SID codes must be integers")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Expected integer raw SID triples within the codebook ranges") from exc
    return codes


def sid_offsets(codebook):
    """Keep the original package's fixed-stride vocabulary convention."""
    sizes = _codebook_array(codebook)
    return np.arange(3, dtype=np.int64) * int(sizes.max()) + 1


def encode_sid_sequence(value, codebook, max_len):
    """Offset real items, keep the most recent items, and then left-pad."""
    if max_len < 1:
        raise ValueError("max_len must be positive")
    codes = sid_codes(value, codebook, sequence=True)[-max_len:]
    result = np.zeros((max_len, 3), dtype=np.int64)
    if len(codes):
        result[-len(codes):] = codes + sid_offsets(codebook)
    return result.reshape(-1)


def _trim_timestamp_padding(value, n_history, label):
    """Remove only verifiable zero padding before the real timestamp sequence."""
    try:
        values = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(label + ": timestamps must be finite numeric arrays") from exc
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError(label + ": timestamps must be finite one-dimensional arrays")
    extra = len(values) - n_history
    if extra < 0:
        raise ValueError(label + ": fewer timestamps than historical items")
    if extra:
        if np.any(values[:extra] != 0):
            raise ValueError(label + ": nonzero left timestamp padding")
        values = values[extra:]
    if np.any(np.diff(values) < 0):
        raise ValueError(label + ": timestamps must follow chronological history")
    return values, extra


def aligned_days(value, n_history, max_len, *, timestamp_unit="days", unix_value=None):
    """Center in float64, convert to days, truncate, and finally cast to float32.

    ``unix_value`` is an optional independent alignment check in Unix seconds.
    The original sequence length is used to trim timestamp padding before the
    same most-recent slice is taken from both items and timestamps.
    """
    if timestamp_unit not in ("days", "seconds"):
        raise ValueError("timestamp_unit must be days or seconds")
    values, removed = _trim_timestamp_padding(value, n_history, "timestamps")
    unix = None
    if unix_value is not None:
        unix, _ = _trim_timestamp_padding(unix_value, n_history, "Unix seconds")
    result = np.zeros(max_len, dtype=np.float32)
    if not n_history:
        return result, removed
    relative = values - values[0]
    if timestamp_unit == "seconds":
        relative = relative / 86400.0
    if unix is not None and not np.allclose(
            relative, (unix - unix[0]) / 86400.0, atol=2e-5, rtol=2e-5):
        raise ValueError("Timestamp day gaps differ from aligned Unix-second gaps")
    kept = min(n_history, max_len)
    with np.errstate(over="ignore", invalid="ignore"):
        centered = relative[-kept:].astype(np.float32)
    if not np.isfinite(centered).all():
        raise ValueError("Centered timestamps exceed the supported float32 day range")
    result[-kept:] = centered
    return result, removed


class RevisedDataset(Dataset):
    """Read one parquet split using an explicit, small column allowlist.

    ``timestamp_unit`` describes the ``timestamps`` column. A fallback
    ``timestamps_unix_s`` column always carries seconds. Returned timestamps are
    centered days in either case, matching ``DriftConfig.time_unit_scale=1``.
    """

    def __init__(self, parquet_path, codebook, max_len, use_timestamps=True,
                 timestamp_unit="days"):
        import pyarrow.parquet as pq

        self.codebook = tuple(int(x) for x in _codebook_array(codebook))
        if not isinstance(max_len, int) or max_len < 1:
            raise ValueError("max_len must be a positive integer")
        if timestamp_unit not in ("days", "seconds"):
            raise ValueError("timestamp_unit must be days or seconds")
        self.max_len = max_len
        self.source_name = Path(parquet_path).name
        self.use_timestamps = bool(use_timestamps)
        self.timestamp_unit = timestamp_unit
        parquet = pq.ParquetFile(parquet_path)
        available = set(parquet.schema_arrow.names)
        missing = {"history", "target"} - available
        if missing:
            raise ValueError(f"{self.source_name}: missing columns {sorted(missing)}")
        columns = ["history", "target"]
        timestamp_columns = [name for name in ("timestamps", "timestamps_unix_s")
                             if name in available]
        if self.use_timestamps:
            if not timestamp_columns:
                raise ValueError(
                    f"{self.source_name}: timestamps required; supply timestamps or "
                    "timestamps_unix_s, or set --use_timestamps 0")
            columns.extend(timestamp_columns)
        self.rows = []
        self.history_lengths = []
        removed_padding = 0
        offsets = sid_offsets(self.codebook)
        for record_batch in parquet.iter_batches(batch_size=128, columns=columns):
            for source in record_batch.to_pylist():
                try:
                    history = sid_codes(source["history"], self.codebook, sequence=True)
                    target = sid_codes(source["target"], self.codebook, sequence=False)
                    length = len(history)
                    if length == 0:
                        raise ValueError("History must contain at least one real item")
                    tokens = encode_sid_sequence(history, self.codebook, self.max_len)
                    times = np.zeros(self.max_len, dtype=np.float32)
                    if self.use_timestamps:
                        primary = source.get("timestamps")
                        unix = source.get("timestamps_unix_s")
                        unit = self.timestamp_unit
                        if primary is None:
                            if unix is None:
                                raise ValueError("Missing timestamp sequence for this row")
                            primary, unit = unix, "seconds"
                        times, removed = aligned_days(
                            primary, length, self.max_len, timestamp_unit=unit,
                            unix_value=unix)
                        removed_padding += removed
                    self.rows.append({
                        "history": torch.from_numpy(tokens),
                        "attention_mask": torch.from_numpy((tokens != 0).astype(np.float32)),
                        "target": torch.from_numpy(target + offsets),
                        "timestamps": torch.from_numpy(times),
                    })
                    self.history_lengths.append(min(length, self.max_len))
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f"{self.source_name} row {len(self.rows)}: {exc}") from exc
        if not self.rows:
            raise ValueError(f"{self.source_name}: empty source parquet")
        self.n_rows = len(self.rows)
        self.metadata = {
            "source_name": self.source_name,
            "rows": self.n_rows,
            "columns_read": columns,
            "history_min_items": min(self.history_lengths),
            "history_max_items": max(self.history_lengths),
            "zero_left_timestamp_padding_removed": removed_padding,
            "timestamp_output_unit": "days",
        }

    def __len__(self):
        return self.n_rows

    def __getitem__(self, index):
        return self.rows[index]
