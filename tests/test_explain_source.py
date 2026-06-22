"""
tests/test_explain_source.py
============================
Unit tests for the Explain-page data-source resolver.

Covers the three-root split (data / model / cache), predictions-path
selection, external-dataset discovery, and the TML/GNN availability flags —
all driven by directory layout, so no models or training are needed.
"""

from pathlib import Path

from polynet.app.services.explain_source import (
    external_raw_csv,
    list_external_datasets,
    resolve_explain_source,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")


def _make_repr(root: Path, *, tml: bool, gnn: bool) -> None:
    """Create representation dirs mirroring an experiment / unseen layout."""
    if tml:
        _touch(root / "representation" / "Descriptors" / "rdkit.csv")
    if gnn:
        _touch(root / "representation" / "GNN" / "raw" / "data.csv")


# ---------------------------------------------------------------------------
# Experiment data source
# ---------------------------------------------------------------------------


def test_experiment_source_roots(tmp_path):
    exp = tmp_path / "exp"
    _make_repr(exp, tml=True, gnn=True)

    src = resolve_explain_source(exp, None)

    assert src.is_external is False
    assert src.dataset_name is None
    assert src.data_root == exp
    assert src.model_path == exp
    assert src.cache_root == exp
    assert src.preds_csv == exp / "ml_results" / "predictions.csv"
    assert src.has_tml and src.has_gnn


def test_experiment_missing_representations(tmp_path):
    exp = tmp_path / "exp"
    exp.mkdir()
    src = resolve_explain_source(exp, None)
    assert not src.has_tml
    assert not src.has_gnn


# ---------------------------------------------------------------------------
# External data source — the three-root split
# ---------------------------------------------------------------------------


def test_external_source_splits_roots(tmp_path):
    exp = tmp_path / "exp"
    ds_dir = exp / "unseen_predictions" / "insilico_ALL"
    _make_repr(ds_dir, tml=True, gnn=True)

    src = resolve_explain_source(exp, "insilico_ALL")

    assert src.is_external is True
    assert src.dataset_name == "insilico_ALL"
    # data + cache live in the dataset folder; models stay in the experiment
    assert src.data_root == ds_dir
    assert src.cache_root == ds_dir
    assert src.model_path == exp
    # external predictions live at <dataset>/predictions.csv (not ml_results/)
    assert src.preds_csv == ds_dir / "predictions.csv"
    assert src.has_tml and src.has_gnn


def test_external_partial_representation_flags(tmp_path):
    exp = tmp_path / "exp"
    ds_dir = exp / "unseen_predictions" / "gnn_only"
    _make_repr(ds_dir, tml=False, gnn=True)

    src = resolve_explain_source(exp, "gnn_only")
    assert src.has_gnn is True
    assert src.has_tml is False


# ---------------------------------------------------------------------------
# Discovery + raw CSV lookup
# ---------------------------------------------------------------------------


def test_list_external_datasets(tmp_path):
    exp = tmp_path / "exp"
    for name in ("ds_b", "ds_a"):
        _make_repr(exp / "unseen_predictions" / name, tml=True, gnn=False)
    # a stray file should be ignored (only directories count)
    _touch(exp / "unseen_predictions" / "note.txt")

    assert list_external_datasets(exp) == ["ds_a", "ds_b"]  # sorted, dirs only


def test_list_external_datasets_none(tmp_path):
    exp = tmp_path / "exp"
    exp.mkdir()
    assert list_external_datasets(exp) == []


def test_external_raw_csv(tmp_path):
    ds_dir = tmp_path / "ds"
    _touch(ds_dir / "representation" / "GNN" / "raw" / "my_data.csv")
    found = external_raw_csv(ds_dir)
    assert found is not None and found.name == "my_data.csv"

    assert external_raw_csv(tmp_path / "empty") is None
