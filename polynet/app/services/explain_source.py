"""
polynet.app.services.explain_source
=====================================
Resolve *where* the Explain page should read data/representations from and
*where* it should write explanations, for either the experiment's own data or a
user-selected external ("unseen") dataset.

The explainability code historically overloaded a single ``experiment_path`` to
mean three different roots at once. For an external dataset these must be split:

- **models / feature-transformers** always live in the experiment;
- **representations + predictions** live in the dataset folder;
- **caches + saved explanations** are written to the dataset folder.

``resolve_explain_source`` returns those roots explicitly so callers don't have
to special-case external datasets throughout the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from polynet.config.paths import (
    gnn_raw_data_path,
    ml_results_file_path,
    representation_file_path,
    unseen_predictions_parent_path,
)


@dataclass
class ExplainSource:
    """Resolved roots/paths for one explainability run.

    Attributes
    ----------
    is_external:
        ``True`` when explaining a user-provided unseen dataset.
    dataset_name:
        Folder name of the external dataset, or ``None`` for experiment data.
    data_root:
        Root that holds ``representation/`` and predictions for this source.
    model_path:
        Root that holds trained models + feature-transformers — **always the
        experiment**, even for external datasets (they reuse the same models).
    cache_root:
        Root under which explanations/caches are written (``explanations/`` is
        created here). Equals ``data_root``.
    preds_csv:
        Path to the predictions CSV for this source.
    has_tml:
        Whether TML descriptor CSVs exist for this source.
    has_gnn:
        Whether a GNN raw graph dataset exists for this source.
    """

    is_external: bool
    dataset_name: str | None
    data_root: Path
    model_path: Path
    cache_root: Path
    preds_csv: Path
    has_tml: bool
    has_gnn: bool


def list_external_datasets(experiment_path: Path) -> list[str]:
    """Return the names of external prediction datasets for an experiment.

    These are the subdirectories of ``<experiment>/unseen_predictions/`` written
    by the Predict page / ``predict_external``.
    """
    parent = unseen_predictions_parent_path(experiment_path)
    if not parent.is_dir():
        return []
    return sorted(p.name for p in parent.iterdir() if p.is_dir())


def _has_tml(data_root: Path) -> bool:
    """True when descriptor CSVs exist under ``data_root``."""
    descriptors_dir = representation_file_path(data_root)
    return descriptors_dir.is_dir() and any(descriptors_dir.glob("*.csv"))


def _has_gnn(data_root: Path) -> bool:
    """True when a raw GNN graph dataset exists under ``data_root``."""
    raw_dir = gnn_raw_data_path(data_root)
    return raw_dir.is_dir() and any(raw_dir.glob("*.csv"))


def external_raw_csv(data_root: Path) -> Path | None:
    """Return the raw GNN CSV for an external dataset, or ``None`` if absent.

    ``predict_external`` writes the raw graph CSV under
    ``representation/GNN/raw/<original_name>.csv``; we glob for it since the
    original filename is not knowable from the dataset folder name alone.
    """
    raw_dir = gnn_raw_data_path(data_root)
    if not raw_dir.is_dir():
        return None
    csvs = sorted(raw_dir.glob("*.csv"))
    return csvs[0] if csvs else None


def resolve_explain_source(
    experiment_path: Path, external_dataset_name: str | None = None
) -> ExplainSource:
    """Resolve the data/model/cache roots for an explainability run.

    Parameters
    ----------
    experiment_path:
        The experiment root directory.
    external_dataset_name:
        When provided, resolve against
        ``<experiment>/unseen_predictions/<name>/``; otherwise against the
        experiment's own data.
    """
    if external_dataset_name:
        data_root = unseen_predictions_parent_path(experiment_path) / external_dataset_name
        return ExplainSource(
            is_external=True,
            dataset_name=external_dataset_name,
            data_root=data_root,
            model_path=experiment_path,
            cache_root=data_root,
            # predict_external saves predictions at <data_root>/predictions.csv
            preds_csv=data_root / "predictions.csv",
            has_tml=_has_tml(data_root),
            has_gnn=_has_gnn(data_root),
        )

    return ExplainSource(
        is_external=False,
        dataset_name=None,
        data_root=experiment_path,
        model_path=experiment_path,
        cache_root=experiment_path,
        preds_csv=ml_results_file_path(experiment_path),
        has_tml=_has_tml(experiment_path),
        has_gnn=_has_gnn(experiment_path),
    )
