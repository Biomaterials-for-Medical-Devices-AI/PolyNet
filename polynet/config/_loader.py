"""
polynet.config._loader
=======================
Internal module — not part of the public API.

Provides the shared normalisation and parsing logic that both entry points
(``from_yaml`` and ``from_app``) use to convert a raw merged dictionary
into a validated ``ExperimentConfig``.

Why this module exists
----------------------
Both the YAML script runner and the Streamlit app produce a dict with the
same key structure, but that structure uses legacy field names and enum
values from the old ``options`` module. Rather than duplicating the
remapping logic in both entry points, it lives here and is called by both.

Remapping responsibilities
--------------------------
1. **Field name normalisation** — PascalCase / camelCase → snake_case,
   and old shorthand names (``node_feats`` → ``node_features``).
2. **Enum value normalisation** — old human-readable values
   (``"GlobalMaxPool"``, ``"Before Pooling"``) → new lowercase enum values
   (``"global_max_pool"``, ``"before_pooling"``). Supports both old and new
   values so existing JSON/YAML files are not immediately broken.
3. **Type coercion** — e.g. ``smiles_merge_approach`` was a single string,
   now it must be a list.
4. **Dropped fields** — fields that no longer exist in the new schemas
   (e.g. ``TrainGNNModel``) are silently removed to avoid Pydantic's
   ``extra="forbid"`` raising errors on legacy files.
"""

from __future__ import annotations

import warnings
from typing import Any

from polynet.config.experiment import ExperimentConfig


# ---------------------------------------------------------------------------
# Enum value compatibility maps
# Old value (as it appears in JSON/YAML) → new StrEnum value
# ---------------------------------------------------------------------------

_POOLING_COMPAT: dict[str, str] = {
    "GlobalMaxPool": "global_max_pool",
    "GlobalAddPool": "global_add_pool",
    "GlobalMeanPool": "global_mean_pool",
    "GlobalMeanMaxPool": "global_mean_max_pool",
}

_WEIGHTING_COMPAT: dict[str, str] = {
    "Before MPP": "before_mpp",
    "Before Pooling": "before_pooling",
    "No Weighting": "no_weighting",
}

_MERGING_COMPAT: dict[str, str] = {
    "Average": "average",
    "Weighted Average": "weighted_average",
    "Concatenate": "concatenate",
    "No Merging": "no_merging",
}

_PROBLEM_TYPE_COMPAT: dict[str, str] = {
    "classification": "classification",  # already correct
    "regression": "regression",  # already correct
}

_SPLIT_TYPE_COMPAT: dict[str, str] = {
    "train_val_test": "train_val_test",
    "train_test": "train_test",
    "cross_validation": "cross_validation",
    "nested_cross_validation": "nested_cross_validation",
    "leave_one_out": "leave_one_out",
}

_SPLIT_METHOD_COMPAT: dict[str, str] = {"Random": "random", "Stratified": "stratified"}

_TRANSFORM_COMPAT: dict[str, str] = {
    "NoTransformation": "no_transformation",
    "No Transformation": "no_transformation",
    "StandardScaler": "standard_scaler",
    "Standard Scaler": "standard_scaler",
    "MinMaxScaler": "min_max_scaler",
    "Min-Max Scaler": "min_max_scaler",
    "RobustScaler": "robust_scaler",
    "Robust Scaler": "robust_scaler",
    "PowerTransformer": "power_transformer",
    "Power Transformer": "power_transformer",
    "QuantileTransformer": "quantile_transformer",
    "Quantile Transformer": "quantile_transformer",
    "Normalizer": "normalizer",
}

_TML_MODEL_COMPAT: dict[str, str] = {
    "Linear Regression": "linear_regression",
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "Support Vector Machine": "support_vector_machine",
    "K-Neighbors Classifier": "k_neighbors_classifier",
    "Decision Tree Classifier": "decision_tree_classifier",
}

_NETWORK_COMPAT: dict[str, str] = {
    "GCN": "GCN",
    "TransformerConvGNN": "TransformerConvGNN",
    "GAT": "GAT",
    "GraphSAGE": "GraphSAGE",
    "MPNN": "MPNN",
    "CGGNN": "CGGNN",
}

_STRING_REPR_COMPAT: dict[str, str] = {
    "SMILES": "SMILES",
    "Psmiles": "psmiles",
    "psmiles": "psmiles",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compat(value: str, mapping: dict[str, str], field_name: str) -> str:
    """
    Normalise a single enum string value using a compatibility mapping.

    If the value is already in the new format (i.e. not in the mapping),
    it is returned as-is and Pydantic will validate it. If it's a legacy
    value, it's remapped with a deprecation warning.
    """
    if value in mapping:
        new_value = mapping[value]
        if new_value != value:
            warnings.warn(
                f"Field '{field_name}': value '{value}' is a legacy format. "
                f"It has been automatically remapped to '{new_value}'. "
                "Update your config file to use the new value to silence this warning.",
                DeprecationWarning,
                stacklevel=4,
            )
        return new_value
    return value  # pass through — Pydantic will validate


def _normalise_gnn_params(params: dict[str, Any]) -> dict[str, Any]:
    """
    Normalise a single GNN parameter dict (the value for one Network key).

    Handles:
    - Pooling value remapping
    - ApplyWeightingToGraph value remapping
    - Typo fix: assymetric_loss_strength → asymmetric_loss_strength
    """
    result = {}
    for k, v in params.items():
        # Fix typo in old configs
        if k == "assymetric_loss_strength":
            k = "asymmetric_loss_strength"

        if k == "pooling" and isinstance(v, str):
            v = _compat(v, _POOLING_COMPAT, "pooling")
        elif k == "apply_weighting_to_graph" and isinstance(v, str):
            v = _compat(v, _WEIGHTING_COMPAT, "apply_weighting_to_graph")

        result[k] = v
    return result


# ---------------------------------------------------------------------------
# Section normalisers
# Each function takes the raw dict for that section and returns a clean dict
# ready to be passed to the corresponding Pydantic schema.
# ---------------------------------------------------------------------------


def _normalise_data(raw: dict[str, Any]) -> dict[str, Any]:
    out = dict(raw)
    if "string_representation" in out and isinstance(out["string_representation"], str):
        out["string_representation"] = _compat(
            out["string_representation"], _STRING_REPR_COMPAT, "string_representation"
        )
    return out


def _normalise_general(raw: dict[str, Any]) -> dict[str, Any]:
    out = dict(raw)

    if "split_type" in out and isinstance(out["split_type"], str):
        out["split_type"] = _compat(out["split_type"], _SPLIT_TYPE_COMPAT, "split_type")

    if "split_method" in out and isinstance(out["split_method"], str):
        out["split_method"] = _compat(out["split_method"], _SPLIT_METHOD_COMPAT, "split_method")

    # Legacy: test_ratio / val_ratio stored as strings
    for ratio_field in ("test_ratio", "val_ratio"):
        if ratio_field in out and isinstance(out[ratio_field], str):
            try:
                out[ratio_field] = float(out[ratio_field])
            except ValueError:
                pass  # Let Pydantic raise a clear error

    return out


def _normalise_representation(raw: dict[str, Any]) -> dict[str, Any]:
    out = dict(raw)

    # Field renames
    if "node_feats" in out:
        out["node_features"] = out.pop("node_feats")
    if "edge_feats" in out:
        out["edge_features"] = out.pop("edge_feats")

    # smiles_merge_approach: string → list
    if "smiles_merge_approach" in out:
        val = out["smiles_merge_approach"]
        if isinstance(val, str):
            out["smiles_merge_approach"] = [_compat(val, _MERGING_COMPAT, "smiles_merge_approach")]
        elif isinstance(val, list):
            out["smiles_merge_approach"] = [
                _compat(v, _MERGING_COMPAT, "smiles_merge_approach") if isinstance(v, str) else v
                for v in val
            ]

    # molecular_descriptors: keys may be old string values
    if "molecular_descriptors" in out and isinstance(out["molecular_descriptors"], dict):
        _DESCRIPTOR_KEY_COMPAT = {
            "rdkit": "rdkit",
            "polybert": "polybert",
            "dataframe": "dataframe",
            "rdkit_dataframe": "rdkit_dataframe",
        }
        out["molecular_descriptors"] = {
            _DESCRIPTOR_KEY_COMPAT.get(k, k): v for k, v in out["molecular_descriptors"].items()
        }

    return out


def _normalise_train_gnn(raw: dict[str, Any]) -> dict[str, Any]:
    out = {}

    # Field renames: PascalCase → snake_case
    _FIELD_MAP = {
        "GNNConvolutionalLayers": "gnn_convolutional_layers",
        "TrainGNN": "train_gnn",
        "HyperparameterOptimisation": "hyperparameter_optimisation",
        "ShareGNNParameters": "share_gnn_parameters",
        # TrainGNNModel is dropped — redundant with train_gnn
    }
    _DROPPED = {"TrainGNNModel"}

    for k, v in raw.items():
        if k in _DROPPED:
            continue
        new_key = _FIELD_MAP.get(k, k)
        out[new_key] = v

    # Normalise GNN convolutional layers
    if "gnn_convolutional_layers" in out and isinstance(out["gnn_convolutional_layers"], dict):
        normalised_layers = {}
        for net_key, params in out["gnn_convolutional_layers"].items():
            # Normalise network key
            norm_net_key = _compat(net_key, _NETWORK_COMPAT, "gnn_convolutional_layers key")
            # Normalise params
            norm_params = _normalise_gnn_params(params) if isinstance(params, dict) else params
            normalised_layers[norm_net_key] = norm_params
        out["gnn_convolutional_layers"] = normalised_layers

    return out


def _normalise_train_tml(raw: dict[str, Any]) -> dict[str, Any]:
    out = {}

    # Field renames
    _FIELD_MAP = {
        "TrainTMLModels": "train_tml",
        "HyperparameterOptimization": "hyperparameter_optimisation",
        "TMLModelsParams": "model_params",
        "TransformFeatures": "transform_features",
    }

    # Old per-model boolean fields → selected_models list
    _MODEL_BOOL_FIELDS = {
        "TrainLinearRegression": "linear_regression",
        "TrainLogisticRegression": "logistic_regression",
        "TrainRandomForest": "random_forest",
        "TrainSupportVectorMachine": "support_vector_machine",
        "TrainXGBoost": "xgboost",
    }

    selected_models: list[str] = []

    for k, v in raw.items():
        if k in _MODEL_BOOL_FIELDS:
            if v is True:
                selected_models.append(_MODEL_BOOL_FIELDS[k])
        elif k in _FIELD_MAP:
            out[_FIELD_MAP[k]] = v
        else:
            out[k] = v

    # Only inject selected_models from booleans if not already present
    if selected_models and "selected_models" not in out:
        out["selected_models"] = selected_models

    # Normalise selected_models values
    if "selected_models" in out and isinstance(out["selected_models"], list):
        out["selected_models"] = [
            _compat(m, _TML_MODEL_COMPAT, "selected_models") if isinstance(m, str) else m
            for m in out["selected_models"]
        ]

    # Normalise transform_features
    if "transform_features" in out and isinstance(out["transform_features"], str):
        out["transform_features"] = _compat(
            out["transform_features"], _TRANSFORM_COMPAT, "transform_features"
        )

    return out


# ---------------------------------------------------------------------------
# Top-level section key map
# Maps the top-level keys that appear in JSON/YAML to the ExperimentConfig
# field names and the appropriate normaliser function.
# ---------------------------------------------------------------------------

_SECTION_NORMALISERS = {
    # New canonical keys
    "data": ("data", _normalise_data),
    "general": ("general", _normalise_general),
    "representation": ("representation", _normalise_representation),
    "plotting": ("plotting", lambda x: x),
    "train_gnn": ("train_gnn", _normalise_train_gnn),
    "train_tml": ("train_tml", _normalise_train_tml),
    # Legacy top-level key aliases (old dataclass names)
    "DataOptions": ("data", _normalise_data),
    "GeneralConfigOptions": ("general", _normalise_general),
    "RepresentationOptions": ("representation", _normalise_representation),
    "PlottingOptions": ("plotting", lambda x: x),
    "TrainGNNOptions": ("train_gnn", _normalise_train_gnn),
    "TrainTMLOptions": ("train_tml", _normalise_train_tml),
}


# ---------------------------------------------------------------------------
# Public entry point for this module
# ---------------------------------------------------------------------------


def build_experiment_config(merged: dict[str, Any]) -> ExperimentConfig:
    """
    Convert a merged raw dictionary into a validated ``ExperimentConfig``.

    This is the single internal function called by both ``from_yaml`` and
    ``from_app``. It handles all field name remapping, enum value
    normalisation, and type coercion before handing off to Pydantic.

    Parameters
    ----------
    merged:
        A dict whose top-level keys are either the new canonical section
        names (``"data"``, ``"general"``, etc.) or the legacy dataclass
        names (``"DataOptions"``, ``"GeneralConfigOptions"``, etc.).
        Values are the raw section dicts as loaded from JSON or YAML.

    Returns
    -------
    ExperimentConfig
        A fully validated experiment configuration.

    Raises
    ------
    pydantic.ValidationError
        If any required field is missing or any value fails validation.
    ValueError
        If an unrecognised top-level section key is encountered.
    """
    normalised: dict[str, Any] = {}
    unrecognised: list[str] = []

    for key, section_data in merged.items():
        if key not in _SECTION_NORMALISERS:
            unrecognised.append(key)
            continue

        target_field, normaliser = _SECTION_NORMALISERS[key]

        if target_field in normalised:
            raise ValueError(
                f"Duplicate section: both a legacy key and its canonical equivalent "
                f"are present for field '{target_field}'. Use only one."
            )

        normalised[target_field] = normaliser(section_data)

    if unrecognised:
        raise ValueError(
            f"Unrecognised top-level config section(s): {unrecognised}. "
            f"Expected one of: {list(_SECTION_NORMALISERS.keys())}"
        )

    return ExperimentConfig(**normalised)
