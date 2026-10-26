"""Portable paths for HSTU entries, anchored at the containing repository.

Relative CLI paths have the same meaning when launched from the repository or
this directory. ``data/`` and ``outputs/`` prefixes honor DRIFT_DATA_ROOT and
DRIFT_OUTPUT_ROOT respectively. Only repository-relative paths are serialized.
"""
from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
_PATH_FIELDS = (
    'kuairec_data_dir', 'drift_cache_dir', 'data_dir', 'cache_dir',
    'category_csv', 'ratings_csv', 'movies_csv', 'state_path', 'input_dir',
    'output_dir', 'source_root', 'stage0_project', 'stage0_state',
    'preflight_report', 'log_path', 'save_path', 'output_json',
)


def resolve_path(value):
    """Resolve a path for I/O without embedding a machine-specific location."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    parts = path.parts
    variable = {'data': 'DRIFT_DATA_ROOT', 'outputs': 'DRIFT_OUTPUT_ROOT'}
    if parts and parts[0] in variable and os.environ.get(variable[parts[0]]):
        root = Path(os.environ[variable[parts[0]]]).expanduser()
        if not root.is_absolute():
            root = REPO_ROOT / root
        return root.joinpath(*parts[1:])
    return REPO_ROOT / path


def resolve_args(args):
    """Normalize CLI path fields while keeping their original string/Path type."""
    for name in _PATH_FIELDS:
        value = getattr(args, name, None)
        if value is not None and str(value):
            resolved = resolve_path(value)
            setattr(args, name, resolved if isinstance(value, Path) else str(resolved))
    return args


def portable_path(value):
    """Represent a runtime location relative to the repository in reports."""
    path = Path(value)
    if path.is_absolute():
        try:
            return Path(os.path.relpath(path, REPO_ROOT)).as_posix()
        except ValueError:
            # Different Windows drives cannot have a relative path.
            return path.name
    return path.as_posix()


def portable_config(value):
    """Recursively strip absolute runtime locations from saved metadata."""
    if isinstance(value, dict):
        return {key: portable_config(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [portable_config(item) for item in value]
    if isinstance(value, Path) or (isinstance(value, str) and Path(value).is_absolute()):
        return portable_path(value)
    return value
