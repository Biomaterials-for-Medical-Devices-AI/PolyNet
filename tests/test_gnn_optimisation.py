"""
tests/test_gnn_optimisation.py
==============================
GNN optimiser, learning-rate scheduler and regression loss are configurable
(``gnn_training.optimisation``), default to PolyNet's historical settings, and
are applied identically to final training and HPO trials.
"""

from unittest import mock
import warnings

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from pydantic import ValidationError  # noqa: E402
from torch import nn  # noqa: E402
from torch.optim import SGD, Adam  # noqa: E402
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, StepLR  # noqa: E402

from polynet.config.enums import (  # noqa: E402
    HpoSplitStrategy,
    Network,
    Optimizer,
    ProblemType,
    RegressionLoss,
    Scheduler,
    TrainingParam,
)
from polynet.config.schemas import GNNOptimisationConfig, TrainGNNConfig  # noqa: E402
from polynet.factories.loss import RMSELoss, create_loss  # noqa: E402
from polynet.factories.optimizer import step_scheduler  # noqa: E402
from polynet.training.gnn import _compute_loss, build_optimisation  # noqa: E402

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_defaults_reproduce_historical_settings():
    cfg = TrainGNNConfig(gnn_convolutional_layers={Network.GCN: {}}).optimisation
    assert cfg.optimizer == Optimizer.Adam
    assert cfg.scheduler == Scheduler.ReduceLROnPlateau
    assert (cfg.scheduler_factor, cfg.scheduler_patience, cfg.scheduler_min_lr) == (0.9, 15, 1e-8)
    assert cfg.regression_loss == RegressionLoss.RMSE


def test_yaml_style_dict_is_parsed():
    cfg = TrainGNNConfig(
        gnn_convolutional_layers={Network.GCN: {}},
        optimisation={
            "optimizer": "sgd",
            "scheduler": "step_lr",
            "scheduler_step_size": 20,
            "regression_loss": "mae",
        },
    ).optimisation
    assert (cfg.optimizer, cfg.scheduler, cfg.scheduler_step_size, cfg.regression_loss) == (
        Optimizer.SGD,
        Scheduler.StepLR,
        20,
        RegressionLoss.MAE,
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"optimizer": "lbfgs"},
        {"scheduler_factor": 1.5},
        {"scheduler_patience": -1},
        {"scheduler_milestones": [60, 30]},
        {"scheduler_milestones": []},
        {"regression_loss": "huber"},
        {"unknown_option": 1},
    ],
)
def test_invalid_settings_are_rejected(bad):
    with pytest.raises(ValidationError):
        GNNOptimisationConfig(**bad)


def test_warns_when_scheduler_param_is_unused():
    with pytest.warns(
        UserWarning, match="scheduler_patience=5 has no effect with scheduler='step_lr'"
    ):
        GNNOptimisationConfig(scheduler="step_lr", scheduler_patience=5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        GNNOptimisationConfig(scheduler="step_lr", scheduler_step_size=5)  # used → no warning


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def test_default_rmse_loss_matches_historical_sqrt_mse():
    pred, y = torch.tensor([[1.0], [2.0], [4.0]]), torch.tensor([1.5, 2.0, 1.0])
    new = _compute_loss(pred, y, create_loss(ProblemType.Regression), ProblemType.Regression)
    old = torch.sqrt(nn.MSELoss()(pred.squeeze(1), y))  # the pre-B2 formula
    assert torch.isclose(new, old)


@pytest.mark.parametrize("loss, cls", [("rmse", RMSELoss), ("mse", nn.MSELoss), ("mae", nn.L1Loss)])
def test_regression_loss_options(loss, cls):
    assert isinstance(create_loss(ProblemType.Regression, regression_loss=loss), cls)


def test_classification_ignores_regression_loss():
    loss = create_loss(ProblemType.Classification, regression_loss="mae")
    assert isinstance(loss, nn.CrossEntropyLoss)


# ---------------------------------------------------------------------------
# Scheduler stepping and the shared builder
# ---------------------------------------------------------------------------


def _params():
    return nn.Linear(2, 1)


def test_plateau_scheduler_receives_validation_loss():
    opt = Adam(_params().parameters(), lr=1.0)
    sched = ReduceLROnPlateau(opt, factor=0.5, patience=0)
    step_scheduler(sched, 1.0)
    step_scheduler(sched, 2.0)  # worse → decays
    assert opt.param_groups[0]["lr"] == 0.5


def test_epoch_schedulers_step_on_epochs_not_on_the_loss():
    opt = Adam(_params().parameters(), lr=1.0)
    sched = StepLR(opt, step_size=2, gamma=0.5)
    for _ in range(2):
        step_scheduler(sched, 123.0)  # a metric passed as epoch would break this
    assert opt.param_groups[0]["lr"] == 0.5


def test_builder_uses_the_settings():
    model = _params()
    cfg = GNNOptimisationConfig(
        optimizer="sgd",
        scheduler="multi_step_lr",
        scheduler_milestones=[5, 7],
        scheduler_factor=0.2,
        regression_loss="mae",
    )
    opt, sched, loss = build_optimisation(model, 0.01, ProblemType.Regression, cfg)
    assert isinstance(opt, SGD) and opt.param_groups[0]["lr"] == 0.01
    assert isinstance(sched, MultiStepLR) and sched.gamma == 0.2
    assert sorted(sched.milestones) == [5, 7]
    assert isinstance(loss, nn.L1Loss)


def test_builder_defaults():
    opt, sched, loss = build_optimisation(_params(), 0.01, ProblemType.Regression, None)
    assert isinstance(opt, Adam) and isinstance(sched, ReduceLROnPlateau)
    assert sched.factor == 0.9 and sched.patience == 15 and sched.min_lrs == [1e-8]
    assert isinstance(loss, RMSELoss)


# ---------------------------------------------------------------------------
# HPO trials use the same settings
# ---------------------------------------------------------------------------


def test_hpo_trial_uses_the_optimisation_settings():
    from test_polymer_descriptor_scaling import _dataset

    import polynet.training.hyperopt as hyperopt

    cfg = GNNOptimisationConfig(optimizer="sgd", scheduler="step_lr", regression_loss="mae")
    seen = []
    real = hyperopt.build_optimisation

    def spy(**kwargs):
        seen.append(kwargs["optimisation"])
        return real(**kwargs)

    data = _dataset(12)
    with (
        mock.patch.object(hyperopt.session, "report", lambda d: None),
        mock.patch.object(hyperopt, "build_optimisation", spy),
    ):
        hyperopt._gnn_target_function(
            config={
                TrainingParam.LearningRate: 0.01,
                TrainingParam.BatchSize: 4,
                "improved": False,
                "embedding_dim": 8,
                "n_convolutions": 1,
                "readout_layers": 2,
                "dropout": 0.0,
            },
            dataset=data,
            num_classes=1,
            splits=[(list(range(8)), list(range(8, 12)))],
            strategy=HpoSplitStrategy.Holdout,
            network=Network.GCN,
            problem_type=ProblemType.Regression,
            optimisation=cfg,
        )
    assert seen == [cfg]
