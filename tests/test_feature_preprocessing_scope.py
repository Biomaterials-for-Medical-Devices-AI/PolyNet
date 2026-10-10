"""
tests/test_feature_preprocessing_scope.py
=========================================
``feature_preprocessing`` is a pipeline-wide option: its scaler applies to TML
descriptors and to GNN polymer descriptors, and it is only meaningful when
there are tabular features to scale.
"""

import logging

import pytest

from polynet.config.enums import FeatureSelection, TransformDescriptor
from polynet.pipeline import runner


@pytest.fixture(scope="module")
def run_pipeline():
    return runner


_CFG = {
    "feature_preprocessing": {
        "scaler": "robust_scaler",
        "selectors": {"variance": {"threshold": 0.05}},
    }
}


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_absent_section_returns_none(run_pipeline, caplog):
    assert (
        run_pipeline._resolve_preprocessing_config(
            {}, train_tml=False, gnn_polymer_descriptors=True
        )
        is None
    )
    assert not _warnings(caplog)


def test_gnn_only_with_polymer_descriptors_uses_scaler(run_pipeline, caplog):
    cfg = run_pipeline._resolve_preprocessing_config(
        {"feature_preprocessing": {"scaler": "min_max_scaler"}},
        train_tml=False,
        gnn_polymer_descriptors=True,
    )
    assert cfg.scaler == TransformDescriptor.MinMaxScaler
    assert not _warnings(caplog)


def test_gnn_only_warns_that_selectors_are_ignored(run_pipeline, caplog):
    cfg = run_pipeline._resolve_preprocessing_config(
        _CFG, train_tml=False, gnn_polymer_descriptors=True
    )
    assert cfg.scaler == TransformDescriptor.RobustScaler
    assert any("selectors apply to TML models only" in w for w in _warnings(caplog))


def test_no_tabular_features_warns_no_effect(run_pipeline, caplog):
    run_pipeline._resolve_preprocessing_config(_CFG, train_tml=False, gnn_polymer_descriptors=False)
    assert any("has no effect" in w for w in _warnings(caplog))


def test_tml_uses_scaler_and_selectors_without_warning(run_pipeline, caplog):
    cfg = run_pipeline._resolve_preprocessing_config(
        _CFG, train_tml=True, gnn_polymer_descriptors=False
    )
    assert cfg.selectors == {FeatureSelection.Variance: {"threshold": 0.05}}
    assert not _warnings(caplog)
