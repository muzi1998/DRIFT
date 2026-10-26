# -*- coding: utf-8 -*-
"""Repository-relative path resolution.

All paths used by the training / preprocessing scripts are resolved relative to
this file's parent directory, so the code package can be placed anywhere without
editing absolute paths.

Layout assumed by default::

    <repo_root>/
      paths.py                 <- this file
      train_drift_sidtier.py   <- main entry (variant A)
      train_drift_recent.py    <- main entry (variant B)
      model/                   <- generative backbone + data pipeline
      drift_method/            <- preference-shift subsequence modules
      preprocess/              <- dataset preprocessing scripts
      rqvae/                   <- RQ-VAE semantic-ID quantizer
      data/                    <- (user-provided) processed datasets
      outputs/                 <- run artifacts (created on demand)

The ``data`` directory can be relocated by exporting the environment variable
``DRIFT_DATA_ROOT`` before launching a script.
"""

import os

# Absolute path of the repository root (the directory containing this file).
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Sub-packages that scripts add to ``sys.path``.
MODEL_DIR = os.path.join(REPO_ROOT, "model")
PREPROCESS_DIR = os.path.join(REPO_ROOT, "preprocess")

# Root that holds the processed datasets. Override with DRIFT_DATA_ROOT.
DATA_ROOT = os.environ.get("DRIFT_DATA_ROOT") or os.path.join(REPO_ROOT, "data")

# Default directory for checkpoints / metric JSON files.
OUTPUT_ROOT = os.environ.get("DRIFT_OUTPUT_ROOT") or os.path.join(REPO_ROOT, "outputs")


def data_path(*parts):
    """Join ``parts`` under the dataset root."""
    return os.path.join(DATA_ROOT, *parts)


def output_path(*parts):
    """Join ``parts`` under the output root."""
    return os.path.join(OUTPUT_ROOT, *parts)


def ensure_import_paths():
    """Make ``model/`` importable as a flat module namespace.

    ``model/glass_backbone.py`` imports ``dataset`` / ``dataloader`` as top-level
    modules, mirroring the original single-directory layout.
    """
    import sys

    for p in (REPO_ROOT, MODEL_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
