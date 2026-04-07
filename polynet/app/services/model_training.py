from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
import torch
from torch import load, save

from polynet.app.options.file_paths import model_dir, representation_file
from polynet.config.enums import MolecularDescriptor
from polynet.config.schemas import DataConfig, RepresentationConfig
from polynet.data import sanitise_df

logger = logging.getLogger(__name__)

# Registry of descriptors whose features are inferred directly from the saved CSV
# (all non-target columns after sanitising), rather than specified by name in the config.
# Mirrors _COUNT_FP_REGISTRY in polynet.featurizer.descriptors.
# To support a new fingerprint type: add one entry here.
_AUTO_FEATURE_DESCRIPTORS: dict[MolecularDescriptor, str] = {
    MolecularDescriptor.PolyBERT: "polyBERT",
    MolecularDescriptor.Morgan: "morgan",
    MolecularDescriptor.RDKitFP: "rdkitfp",
}


# ---------------------------------------------------------------------------
# Model persistence helpers
# ---------------------------------------------------------------------------


def save_tml_model(model, path):
    joblib.dump(model, path)


def load_tml_model(path):
    return joblib.load(path)


def save_gnn_model(model, path):
    """
    Saves the GNN model to the specified path.

    Args:
        model: The GNN model to save.
        path (str): The path where the model will be saved.
    """
    save(model, path)


def load_gnn_model(path):
    """
    Loads the GNN model from the specified path.

    Args:
        path (str): The path from which to load the model.

    Returns:
        The loaded GNN model.
    """
    return load(path, weights_only=False, map_location=torch.device("cpu"))


def save_plot(fig, path, dpi=300):
    """
    Saves the plot to the specified path.

    Args:
        fig: The figure to save.
        path (str): The path where the plot will be saved.
    """
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    print(f"Plot saved to {path}")


# ---------------------------------------------------------------------------
# DataFrame loading
# ---------------------------------------------------------------------------


def load_dataframes(
    representation_options: RepresentationConfig, data_options: DataConfig, experiment_path: Path
) -> dict[MolecularDescriptor, pd.DataFrame]:
    """
    Load and validate the per-descriptor CSVs saved by the featuriser.

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
    """
    Decide which feature column names to validate for a given descriptor type.

    Returns ``None`` to signal that this descriptor should be skipped.

    Parameters
    ----------
    representation:
        The descriptor type being processed.
    features:
        The raw value from ``RepresentationConfig.molecular_descriptors``.
        This is a list of names for explicit-feature descriptors (RDKit, DataFrame),
        a dict for PolyMetriX, and an empty list for auto-feature fingerprints.
    sanitised_cols:
        Column names of the DataFrame after ``sanitise_df`` has been applied.
    target_col:
        Name of the target variable column.
    """
    # Auto-feature fingerprints (PolyBERT, Morgan, RDKitFP, …):
    # All non-target columns in the sanitised CSV are feature columns.
    if representation in _AUTO_FEATURE_DESCRIPTORS:
        return [c for c in sanitised_cols if c != target_col]

    # PolyMetriX: reconstruct expected column names from the config dict.
    if representation == MolecularDescriptor.PolyMetriX:
        return _pmx_feature_names(features)

    # All other descriptors (RDKit, DataFrame, …): use the explicit list.
    # Skip if empty — this descriptor was not fully configured.
    if not features:
        return None
    return list(features)


def _pmx_feature_names(features: dict) -> list[str]:
    """
    Reconstruct PolyMetriX column names from the features configuration dict.

    Parameters
    ----------
    features:
        Dict with keys ``"side_chain"``, ``"backbone"``, and ``"agg"``,
        as stored in ``RepresentationConfig.molecular_descriptors[PolyMetriX]``.
    """
    agg_method = features["agg"]
    side_feats = features["side_chain"]
    back_feats = features["backbone"]
    feats_side_chain = [
        f"{feat}_sidechainfeaturizer_{agg}" for agg in agg_method for feat in side_feats
    ]
    feats_backbone = [f"{feat}_sum_backbonefeaturizer" for feat in back_feats]
    return feats_side_chain + feats_backbone


# ---------------------------------------------------------------------------
# Experiment loading helpers
# ---------------------------------------------------------------------------


def load_models_from_experiment(experiment_path: Path, model_names: list[str]) -> dict:
    """
    Loads trained models from the specified experiment path.

    Args:
        experiment_path (Path): Path to the experiment directory.
        model_names (list[str]): Names of model files to load.

    Returns:
        dict: Dictionary containing model names as keys and loaded models as values.
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


def load_scalers_from_experiment(experiment_path: str, model_names: list[str]) -> dict:
    """
    Loads trained scalers from the specified experiment path.

    Args:
        experiment_path (str): Path to the experiment directory.
        model_names (list[str]): Names of model files whose scalers to load.

    Returns:
        dict: Dictionary containing scaler names as keys and loaded scalers as values.
    """
    gnn_models_path = model_dir(experiment_path)
    scaler = {}

    for model_name in model_names:
        model_name = model_name.split(".")[0]
        scaler_name = model_name.split("-")[-1]
        scaler_file = gnn_models_path / f"{scaler_name}.pkl"
        scaler[scaler_name] = load_tml_model(scaler_file)

    return scaler
