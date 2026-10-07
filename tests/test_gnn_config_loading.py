"""
tests/test_gnn_config_loading.py
================================
The CLI builds the GNN training config from the ``gnn_training`` and
``training`` YAML sections. Misspelt keys must raise an error at load time
instead of being ignored (silently falling back to the defaults).
"""

import copy
import importlib.util
from pathlib import Path

from pydantic import ValidationError
import pytest

from polynet.config.enums import Network, TrainingParam

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_pipeline.py"


@pytest.fixture(scope="module")
def build_gnn_config():
    spec = importlib.util.spec_from_file_location("run_pipeline", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._build_gnn_config


_CFG = {
    "gnn_training": {
        "train_gnn": True,
        "gnn_convolutional_layers": {
            "GCN": {"LearningRate": 1e-3, "BatchSize": 8, "improved": False, "embedding_dim": 16},
            "GAT": {"LearningRate": 1e-3, "BatchSize": 8, "AsymmetricLossStrength": 0.5},
            "MPNN": {},
        },
        "hpo_split_strategy": "repeated_holdout",
        "hpo_val_fraction": 0.3,
        "hpo_n_repeats": 2,
        "hpo_num_samples": 7,
        "hpo_search_grid": {"GAT": {"LearningRate": [0.01]}},
    },
    "training": {"epochs": 12},
}


def _with(section, key, value, arch=None):
    cfg = copy.deepcopy(_CFG)
    target = cfg[section]
    if arch is not None:
        target = target["gnn_convolutional_layers"][arch]
    target[key] = value
    return cfg


def test_every_setting_is_read(build_gnn_config):
    gnn = build_gnn_config(copy.deepcopy(_CFG))
    assert gnn.epochs == 12
    assert gnn.hpo_val_fraction == 0.3 and gnn.hpo_n_repeats == 2
    assert gnn.hpo_num_samples == 7
    assert gnn.gnn_convolutional_layers[Network.GCN][TrainingParam.LearningRate] == 1e-3
    assert gnn.gnn_convolutional_layers[Network.GAT][TrainingParam.AsymmetricLossStrength] == 0.5
    assert gnn.hpo_search_grid["GAT"] == {TrainingParam.LearningRate: [0.01]}


@pytest.mark.parametrize(
    "cfg",
    [
        _with("gnn_training", "hpo_n_fold", 3),  # gnn_training key typo
        _with("gnn_training", "embeding_dim", 16, arch="GCN"),  # architecture parameter typo
        _with("gnn_training", "improved", True, arch="GAT"),  # GCN-only parameter
        _with("gnn_training", "seed", 1, arch="GCN"),  # set by PolyNet
        _with("training", "epoch", 3),  # training key typo
        _with("gnn_training", "epochs", 3),  # epochs belong to 'training'
    ],
)
def test_unknown_keys_are_rejected(build_gnn_config, cfg):
    with pytest.raises((ValidationError, ValueError)):
        build_gnn_config(cfg)
