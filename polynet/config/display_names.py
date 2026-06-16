"""
polynet.config.display_names
============================
Human-readable display names for models and descriptors.

Internal identifiers used in ``predictions.csv`` columns and ``metrics.json``
keys are raw enum values joined as ``"{model}-{descriptor}"`` for TML
(e.g. ``"random_forest-polybert"``) or a bare network name for GNN
(e.g. ``"GCN"``). Those raw strings are lookup keys and must never change.

This module provides a **display-only** translation layer so figures and
tables can show clean labels such as ``"Random Forest polyBERT"`` (full) or
``"RF polyBERT"`` (abbreviated) without touching the underlying data.

Public API
----------
::

    from polynet.config.display_names import prettify_label

    prettify_label("random_forest-polybert")                 # "Random Forest polyBERT"
    prettify_label("random_forest-polybert", abbreviate=True)  # "RF polyBERT"
    prettify_label("GCN")                                     # "GCN"
"""

from __future__ import annotations

# Each entry maps a raw enum value to ``(full_name, abbreviation)``.
# Keys must match the enum ``.value`` strings in ``polynet.config.enums``.

# Traditional ML models (TraditionalMLModel) + GNN architectures (Network).
# Both appear as the "model" token of a label, so they share one table.
_MODEL_NAMES: dict[str, tuple[str, str]] = {
    # Traditional ML
    "linear_regression": ("Linear Regression", "LinReg"),
    "logistic_regression": ("Logistic Regression", "LogReg"),
    "random_forest": ("Random Forest", "RF"),
    "xgboost": ("XGBoost", "XGB"),
    "support_vector_machine": ("Support Vector Machine", "SVM"),
    "k_neighbors_classifier": ("K-Neighbors Classifier", "KNN"),
    "decision_tree_classifier": ("Decision Tree", "DT"),
    # GNN architectures
    "GCN": ("GCN", "GCN"),
    "TransformerConvGNN": ("Transformer GNN", "TF-GNN"),
    "GAT": ("GAT", "GAT"),
    "GraphSAGE": ("GraphSAGE", "SAGE"),
    "MPNN": ("MPNN", "MPNN"),
    "CGGNN": ("CGGNN", "CGGNN"),
}

# Molecular descriptors (MolecularDescriptor).
_DESCRIPTOR_NAMES: dict[str, tuple[str, str]] = {
    "rdkit": ("RDKit", "RDKit"),
    "polybert": ("polyBERT", "polyBERT"),
    "dataframe": ("DataFrame", "DF"),
    "rdkit_dataframe": ("RDKit + DataFrame", "RDKit+DF"),
    "polymetrix": ("PolyMetriX", "PMX"),
    "morgan": ("Morgan", "Morgan"),
    "rdkitfp": ("RDKit FP", "RDKitFP"),
}

# Evaluation metrics (EvaluationMetric). Mostly acronyms, so a single display
# form is enough — no separate abbreviation is needed.
_METRIC_NAMES: dict[str, str] = {
    # Classification
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1 Score",
    "auroc": "AUROC",
    "mcc": "MCC",
    "specificity": "Specificity",
    "g_score": "G-Score",
    # Regression
    "rmse": "RMSE",
    "mae": "MAE",
    "r2": "R²",
}


def _fallback(token: str) -> str:
    """Best-effort prettification for tokens with no predetermined mapping."""
    return token.replace("_", " ").title()


def _lookup(token: str, table: dict[str, tuple[str, str]], abbreviate: bool) -> str:
    """Return the full or abbreviated name for a token, falling back to title-case."""
    if token in table:
        full, abbr = table[token]
        return abbr if abbreviate else full
    return _fallback(token)


def prettify_label(raw: str | None, abbreviate: bool = False) -> str | None:
    """
    Convert a raw model/descriptor identifier into a clean display label.

    Handles three forms:

    - ``"{model}-{descriptor}"`` (TML) → ``"Random Forest polyBERT"`` /
      ``"RF polyBERT"``. Split on the first hyphen; the model and descriptor
      parts are translated independently.
    - ``"{network}"`` (GNN) → ``"GCN"`` / ``"Transformer GNN"``.
    - Any unrecognised token → title-cased with underscores replaced by spaces.

    Parameters
    ----------
    raw:
        The raw identifier. ``None`` is returned unchanged so callers can pass
        optional values through safely.
    abbreviate:
        When ``True`` use the short form (e.g. ``"RF"``); otherwise the full
        name (e.g. ``"Random Forest"``).

    Returns
    -------
    str | None
        The display label, or ``None`` if ``raw`` was ``None``.
    """
    if raw is None:
        return None

    raw = str(raw)

    if "-" in raw:
        model_part, descriptor_part = raw.split("-", 1)
        model_label = _lookup(model_part, _MODEL_NAMES, abbreviate)
        descriptor_label = _lookup(descriptor_part, _DESCRIPTOR_NAMES, abbreviate)
        return f"{model_label} {descriptor_label}"

    # Bare token: a model/network name, or a descriptor on its own.
    if raw in _MODEL_NAMES:
        return _lookup(raw, _MODEL_NAMES, abbreviate)
    if raw in _DESCRIPTOR_NAMES:
        return _lookup(raw, _DESCRIPTOR_NAMES, abbreviate)
    return _fallback(raw)


def prettify_metric(raw: str | None) -> str | None:
    """
    Convert a raw evaluation-metric key into a clean display label.

    Maps enum values such as ``"mae"`` → ``"MAE"`` and ``"r2"`` → ``"R²"``.
    Unrecognised keys fall back to title-case. ``None`` is returned unchanged.
    """
    if raw is None:
        return None
    raw = str(raw)
    return _METRIC_NAMES.get(raw, _fallback(raw))
