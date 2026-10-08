"""
polynet.applicability.reference
===============================
Training references of a trained experiment, read from its saved files.

The applicability domain is assessed at prediction time from what every
experiment already stores, so it also works for experiments trained before
it existed:

- the training dataset (``{experiment}/{data_name}``, written by the GUI
  and the CLI; the raw GNN CSV is used when it is missing);
- the sample IDs of every split (``split_indices.json``);
- the held-out test predictions (``ml_results/predictions.csv``);
- for the ``euclidean_model_inputs`` distance, the training descriptors of
  each representation and the scaler / feature selector of each split.

The reference of a model family is what its models were trained on: the
training set for GNNs (the validation set only selects the best epoch), and
the training set plus the validation set for TML models when
``tml_models.include_validation_in_training`` is true.

::

    from polynet.applicability.reference import load_split_ids, load_training_dataset
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from polynet.config.constants import ResultColumn
from polynet.config.io import load_options
from polynet.config.paths import (
    data_file_path,
    gnn_raw_data_file,
    ml_results_file_path,
    model_dir,
    representation_file,
    split_indices_path,
    train_tml_model_options_path,
)
from polynet.config.schemas import DataConfig, TrainTMLConfig

logger = logging.getLogger(__name__)

GNN = "GNN"
TML = "TML"


def load_training_dataset(experiment_path: Path, data_cfg: DataConfig) -> pd.DataFrame:
    """
    The experiment's training dataset, indexed by sample ID (as strings).

    Reads ``{experiment}/{data_name}`` (GUI: ``id_col`` as a column; CLI:
    the ID as the saved index), falling back to the raw GNN CSV.

    Raises
    ------
    FileNotFoundError
        If neither file exists.
    """
    candidates = [
        data_file_path(data_cfg.data_name, experiment_path),
        gnn_raw_data_file(data_cfg.data_name, experiment_path),
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            "The training dataset of the experiment was not found (looked for "
            f"{[str(p) for p in candidates]})."
        )
    df = pd.read_csv(path)
    if data_cfg.id_col and data_cfg.id_col in df.columns:
        df = df.set_index(data_cfg.id_col)
    elif df.columns[0].startswith("Unnamed"):
        df = df.set_index(df.columns[0])
    df.index = df.index.astype(str)
    return df


def load_split_ids(experiment_path: Path) -> dict[str, dict[str, list[str]]] | None:
    """
    Sample IDs of every split, keyed by split number (``"1"``, ``"2"``, …).

    Returns
    -------
    dict or None
        ``{split: {"train": [...], "val": [...], "test": [...]}}``, or
        ``None`` when the experiment has no ``split_indices.json``.
    """
    path = split_indices_path(experiment_path)
    if not path.exists():
        return None
    with open(path) as f:
        indices = json.load(f)
    return {
        str(i + 1): {
            "train": [str(s) for s in train],
            "val": [str(s) for s in (indices.get("val") or [[]] * len(indices["train"]))[i]],
            "test": [str(s) for s in indices["test"][i]],
        }
        for i, train in enumerate(indices["train"])
    }


def tml_trained_with_validation(experiment_path: Path) -> bool:
    """``tml_models.include_validation_in_training`` of the experiment (default true)."""
    path = train_tml_model_options_path(experiment_path)
    if not path.exists():
        return TrainTMLConfig.model_fields["include_validation_in_training"].default
    return load_options(path, TrainTMLConfig).include_validation_in_training


def reference_ids(split: dict[str, list[str]], family: str, experiment_path: Path) -> list[str]:
    """IDs a model family of one split was trained on (see module docstring)."""
    if family == TML and tml_trained_with_validation(experiment_path):
        return split["train"] + split["val"]
    return split["train"]


def load_training_predictions(experiment_path: Path) -> pd.DataFrame | None:
    """The experiment's ``ml_results/predictions.csv`` (IDs as strings), if present."""
    path = ml_results_file_path(experiment_path)
    if not path.exists():
        return None
    return pd.read_csv(path, dtype={ResultColumn.INDEX: str})


def load_training_descriptors(
    experiment_path: Path, representation: str, data_cfg: DataConfig, weights_cols: list[str] | None
) -> pd.DataFrame | None:
    """
    Training descriptors of a TML representation (e.g. ``"rdkit"``), indexed by sample ID.

    Read from ``representation/Descriptors/{representation}.csv`` and cleaned
    like the training data (structure, ratio and target columns dropped).
    ``None`` when the file does not exist.
    """
    from polynet.data.preprocessing import sanitise_df

    path = representation_file(f"{representation}.csv", experiment_path)
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    clean = sanitise_df(df, data_cfg.smiles_cols, data_cfg.target_variable_col, weights_cols)
    return clean.drop(columns=[data_cfg.target_variable_col], errors="ignore")


def load_feature_transformer(experiment_path: Path, representation: str, split: str):
    """The scaler / feature selector fitted on a split for a representation, if saved."""
    import joblib

    path = model_dir(experiment_path) / f"{representation}_{split}.pkl"
    return joblib.load(path) if path.exists() else None
