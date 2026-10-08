"""
tests/test_hpo_settings.py
==========================
HPO settings shared by the CLI and the GUI: the default number of sampled
configurations, the ``hyperparameter_optimisation`` flag (filled in from the
hyperparameter blocks, warned about when it contradicts them) and the GUI's
search-grid editor helpers.
"""

import warnings

import pytest

from polynet.app.components.forms.search_grid import (
    changed_parameters,
    is_numeric_parameter,
    parse_extra_values,
)
from polynet.config.enums import Network, ProblemType, TraditionalMLModel, TrainingParam
from polynet.config.schemas import TrainGNNConfig, TrainTMLConfig
from polynet.config.search_grid import (
    default_gnn_architecture_grid,
    default_gnn_shared_grid,
    default_tml_grid,
    get_gnn_search_grid,
)

_EXPLICIT = {"embedding_dim": 16, "learning_rate": 0.01, "batch_size": 8}


def test_gnn_and_tml_sample_the_same_number_of_configurations_by_default():
    gnn = TrainGNNConfig(gnn_convolutional_layers={Network.GCN: {}})
    tml = TrainTMLConfig(train_tml=True, selected_models={TraditionalMLModel.RandomForest: {}})
    assert gnn.hpo_num_samples == tml.hpo_num_samples == 50


@pytest.mark.parametrize("block, expected", [({}, True), (_EXPLICIT, False)])
def test_unset_flag_is_filled_in_from_the_blocks(block, expected):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = TrainGNNConfig(gnn_convolutional_layers={Network.GCN: block})
    assert cfg.hyperparameter_optimisation is expected


@pytest.mark.parametrize(
    "flag, block, message",
    [
        (True, _EXPLICIT, "no hyperparameter optimisation will run"),
        (False, {}, "will run for them"),
    ],
)
def test_contradicting_flag_is_warned_about(flag, block, message):
    with pytest.warns(UserWarning, match=message):
        TrainGNNConfig(
            gnn_convolutional_layers={Network.GCN: block}, hyperparameter_optimisation=flag
        )
    with pytest.warns(UserWarning, match=message):
        TrainTMLConfig(
            train_tml=True,
            selected_models={
                TraditionalMLModel.RandomForest: {"n_estimators": 10} if block else {}
            },
            hyperparameter_optimisation=flag,
        )


def test_gui_default_grids_match_the_search_grids():
    for network in (Network.GCN, Network.GAT):
        full = get_gnn_search_grid(network, random_seed=0, problem_type=ProblemType.Classification)
        shown = {
            **default_gnn_shared_grid(ProblemType.Classification),
            **default_gnn_architecture_grid(network),
        }
        assert shown == {k: v for k, v in full.items() if k != TrainingParam.Seed}
    assert "random_state" not in default_tml_grid(
        TraditionalMLModel.RandomForest, ProblemType.Regression
    )


def test_extra_values_are_parsed_like_the_defaults():
    assert parse_extra_values("256, 512", [32, 64]) == [256, 512]
    assert parse_extra_values(" 0.2 ,0.3", [0.01, 0.1]) == [0.2, 0.3]
    assert parse_extra_values("", [1, 2]) == []
    with pytest.raises(ValueError):
        parse_extra_values("12.5", [100, 300])
    with pytest.raises(ValueError):
        parse_extra_values("abc", [0.1])


def test_numeric_parameters_and_changes():
    assert is_numeric_parameter([None, 3, 6]) and not is_numeric_parameter([True, False])
    assert not is_numeric_parameter(["rbf", "linear"])
    defaults = {"a": [1, 2], "b": ["x", "y"]}
    assert changed_parameters(defaults, {"a": [1, 2], "b": ["x"]}) == {"b": ["x"]}
    assert changed_parameters(defaults, {"a": [1, 2], "b": ["x", "y"]}) == {}
