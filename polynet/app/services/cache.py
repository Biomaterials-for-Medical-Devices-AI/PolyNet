"""Bounded GUI-only caches for immutable on-disk data snapshots.

File metadata is checked on every call, including after CLI runs. Cached values
are copied by Streamlit; trained models and training operations are never cached.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st


def file_version(path):
    """Return a resolved filename and metadata fingerprint, raising if missing.

    Both modification and change timestamps are included so same-size rewrites
    and atomic replacements invalidate the cache without reading the whole file.
    """
    path = Path(path).resolve()
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size


@st.cache_data(show_spinner=False, max_entries=32)
def _read_csv(version, kwargs):
    return pd.read_csv(version[0], **kwargs)


def read_csv(path, **kwargs):
    """Read a local CSV snapshot with pandas options; return an independent copy.

    At most 32 path/version/options combinations are kept in the GUI data cache.
    Uploaded file objects are intentionally outside this path-based interface.
    """
    return _read_csv(file_version(path), kwargs)


@st.cache_data(show_spinner=False, max_entries=64)
def _read_json(version):
    with open(version[0]) as handle:
        return json.load(handle)


def read_json(path):
    """Read local JSON with file-aware invalidation and a 64-entry cache limit."""
    return _read_json(file_version(path))
