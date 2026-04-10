"""
polynet.models.persistence
===========================
Model and scaler serialisation helpers.

Provides load/save functions for GNN (.pt) and TML (.joblib) models,
plus experiment-level helpers that load all models or scalers from a
saved experiment directory.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
import torch
from torch import load, save

from polynet.config.enums import MolecularDescriptor
from polynet.config.paths import model_dir, representation_file
from polynet.config.schemas import DataConfig, RepresentationConfig
from polynet.data import sanitise_df

logger = logging.getLogger(__name__)

# Descriptors whose feature column names are determined by the featurizer
# implementation rather than the user config, so we read them from the saved
# CSV directly instead of trying to reconstruct them.
# PolyMetriX is included because its column naming depends on internal
# PolyMetriX class names (e.g. "num_diverse_sidechains", "numbackbonefeaturizer")
# and some featurizers (e.g. BondCounts) expand to multiple output columns —
# making reconstruction from config keys unreliable.
_AUTO_FEATURE_DESCRIPTORS: set[MolecularDescriptor] = {
    MolecularDescriptor.PolyBERT,
    MolecularDescriptor.Morgan,
    MolecularDescriptor.RDKitFP,
    MolecularDescriptor.PolyMetriX,
}


# ---------------------------------------------------------------------------
# Model persistence helpers
# ---------------------------------------------------------------------------


def save_tml_model(model, path):
    joblib.dump(model, path)


def load_tml_model(path):
    return joblib.load(path)


def save_gnn_model(model, path):
    """Save a GNN model to the specified path.

    Args:
        model: The GNN model to save.
        path: The path where the model will be saved.
    """
    save(model, path)


def load_gnn_model(path):
    """Load a GNN model from the specified path.

    Args:
        path: The path from which to load the model.

    Returns:
        The loaded GNN model.
    """
    return load(path, weights_only=False, map_location=torch.device("cpu"))


# ---------------------------------------------------------------------------
# DataFrame loading
# ---------------------------------------------------------------------------


def load_dataframes(
    representation_options: RepresentationConfig, data_options: DataConfig, experiment_path: Path
) -> dict[MolecularDescriptor, pd.DataFrame]:
    """Load and validate the per-descriptor CSVs saved by the featuriser.

    For each descriptor type in ``representation_options.molecular_descriptors``:

    1. Reads the corresponding CSV from the experiment's Descriptors directory.
    2. Sanitises the DataFrame (drops SMILES / weight columns, ensures target is last).
    3. Resolves the expected feature columns for that descriptor type.
    4. Validates that all expected features are present.

    Returns
    -------
    dict[MolecularDescriptor, pd.DataFrame]
        Mapping from descriptor type to its cleaned, validated DataFrame.

    Raises
    ------
    ValueError
        If expected feature columns are missing, or the target column is not last.
    """
    weights_cols = (
        list(representation_options.weights_col.values())
        if representation_options.weights_col
        else None
    )
    dataframe_dict = {}

    for representation, features in representation_options.molecular_descriptors.items():
        file_path = representation_file(
            experiment_path=experiment_path, file_name=f"{representation}.csv"
        )
        df = pd.read_csv(file_path, index_col=0)
        df = sanitise_df(
            df=df,
            smiles_cols=data_options.smiles_cols,
            target_variable_col=data_options.target_variable_col,
            weights_cols=weights_cols,
        )

        target = data_options.target_variable_col
        expected_features = _resolve_features(representation, features, list(df.columns), target)

        if expected_features is None:
            continue

        missing = set(expected_features) - set(df.columns)
        if missing:
            raise ValueError(
                f"[{representation}] Missing expected feature columns: {sorted(missing)}"
            )
        if list(df.columns)[-1] != target:
            raise ValueError(
                f"[{representation}] Target column '{target}' must be last. "
                f"Got: '{list(df.columns)[-1]}'"
            )

        dataframe_dict[representation] = df

    return dataframe_dict


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_features(
    representation: MolecularDescriptor, features, sanitised_cols: list[str], target_col: str
) -> list[str] | None:
    """Decide which feature column names to validate for a given descriptor type.

    Returns ``None`` to signal that this descriptor should be skipped.
    """
    if representation in _AUTO_FEATURE_DESCRIPTORS:
        # Column names are determined by the featurizer implementation, not the
        # config. Read them directly from the saved CSV.
        return [c for c in sanitised_cols if c != target_col]

    if not features:
        return None
    return list(features)


# ---------------------------------------------------------------------------
# Experiment loading helpers
# ---------------------------------------------------------------------------


def load_models_from_experiment(experiment_path: Path, model_names: list[str]) -> dict:
    """Load trained models from an experiment directory.

    Args:
        experiment_path (Path): Path to the experiment directory.
        model_names (list[str]): Names of model files to load.

    Returns:
        dict: Mapping of model name (stem) to loaded model object.
    """
    gnn_models_path = model_dir(experiment_path)
    models = {}

    for model_name in model_names:
        model_file = gnn_models_path / model_name
        model_name = model_file.stem
        termination = str(model_file).split(".")[-1]
        if termination == "pt":
            models[model_name] = load_gnn_model(model_file)
        else:
            models[model_name] = load_tml_model(model_file)

    return models


def load_scalers_from_experiment(experiment_path: Path, model_names: list[str]) -> dict:
    """Load trained scalers from an experiment directory.

    Args:
        experiment_path (Path): Path to the experiment directory.
        model_names (list[str]): Names of model files whose scalers to load.

    Returns:
        dict: Mapping of scaler name to loaded scaler object.
    """
    gnn_models_path = model_dir(experiment_path)
    scaler = {}

    for model_name in model_names:
        model_name = model_name.split(".")[0]
        scaler_name = model_name.split("-")[-1]
        scaler_file = gnn_models_path / f"{scaler_name}.pkl"
        scaler[scaler_name] = load_tml_model(scaler_file)

    return scaler
