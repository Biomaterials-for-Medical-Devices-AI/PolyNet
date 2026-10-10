"""Bounded GUI-only caches for immutable on-disk data snapshots.

File metadata is checked on every call, including after CLI runs. Cached values
are copied by Streamlit; trained models and training operations are never cached.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st


def file_version(path):
    path = Path(path).resolve()
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size


@st.cache_data(show_spinner=False, max_entries=32)
def _read_csv(version, kwargs):
    return pd.read_csv(version[0], **kwargs)


def read_csv(path, **kwargs):
    return _read_csv(file_version(path), kwargs)


@st.cache_data(show_spinner=False, max_entries=64)
def _read_json(version):
    with open(version[0]) as handle:
        return json.load(handle)


def read_json(path):
    return _read_json(file_version(path))
