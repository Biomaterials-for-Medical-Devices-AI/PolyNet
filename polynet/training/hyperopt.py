"""
polynet.training.hyperopt
==========================
Hyperparameter optimisation for GNN architectures using Ray Tune.

Uses ASHA (Asynchronous Successive Halving) early stopping for
efficient search over large hyperparameter spaces. Results are cached
to disk so a completed HPO run is not repeated if the pipeline is
re-run with the same configuration.

Public API
----------
::

    from polynet.training.hyperopt import gnn_hyp_opt
"""

from __future__ import annotations

from copy import deepcopy
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import ray
from ray import tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import torch
from torch_geometric.loader import DataLoader

from polynet.config.enums import (
    HpoSplitStrategy,
    Network,
    Optimizer,
    ProblemType,
    Scheduler,
    TrainingParam,
)
from polynet.config.search_grid import get_gnn_search_grid
from polynet.factories.loss import create_loss
from polynet.factories.network import create_network
from polynet.factories.optimizer import create_optimizer, create_scheduler
from polynet.training.gnn import eval_network, train_network

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------


def _build_splits(
    dataset: list,
    problem_type: ProblemType,
    strategy: HpoSplitStrategy,
    random_seed: int,
    n_folds: int = 5,
    val_fraction: float = 0.2,
    n_repeats: int = 3,
) -> list[tuple[list[int], list[int]]]:
    """
    Build train/val index pairs for the HPO target function.

    Parameters
    ----------
    dataset:
        Full graph list passed to the HPO run.
    problem_type:
        Used to choose stratified vs plain splits for classification.
    strategy:
        Which split strategy to use.
    random_seed:
        Base random seed; repeated strategies offset this per repeat.
    n_folds:
        Number of folds for ``CrossValidation``.
    val_fraction:
        Fraction held out for validation in ``Holdout`` / ``RepeatedHoldout``.
    n_repeats:
        Number of independent splits for ``RepeatedHoldout``.

    Returns
    -------
    list[tuple[list[int], list[int]]]
        List of ``(train_indices, val_indices)`` pairs.
    """
    n = len(dataset)
    y = [data.y.item() for data in dataset]
    stratify = y if problem_type == ProblemType.Classification else None

    if strategy == HpoSplitStrategy.CrossValidation:
        cv = (
            StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
            if problem_type == ProblemType.Classification
            else KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        )
        return [(tr.tolist(), va.tolist()) for tr, va in cv.split(np.zeros(n), y)]

    if strategy == HpoSplitStrategy.Holdout:
        tr, va = train_test_split(
            range(n), test_size=val_fraction, random_state=random_seed, stratify=stratify
        )
        return [(list(tr), list(va))]

    # RepeatedHoldout
    splits = []
    for i in range(n_repeats):
        tr, va = train_test_split(
            range(n), test_size=val_fraction, random_state=random_seed + i, stratify=stratify
        )
        splits.append((list(tr), list(va)))
    return splits


# ---------------------------------------------------------------------------
# HPO entry point
# ---------------------------------------------------------------------------


def gnn_hyp_opt(
    exp_path: Path,
    gnn_arch: Network,
    dataset: list,
    num_classes: int,
    num_samples: int,
    iteration: int,
    problem_type: ProblemType | str,
    random_seed: int,
    hpo_split_strategy: HpoSplitStrategy = HpoSplitStrategy.CrossValidation,
    n_folds: int = 5,
    val_fraction: float = 0.2,
    n_repeats: int = 3,
) -> dict:
    """
    Run Ray Tune hyperparameter optimisation for a GNN architecture.

    If a results CSV already exists for this architecture and iteration,
    the best previously found configuration is loaded and returned
    without re-running the search.

    Parameters
    ----------
    exp_path:
        Root experiment directory. HPO results are stored under
        ``exp_path/gnn_hyp_opt/iteration_{iteration}/{arch}/``.
    gnn_arch:
        The GNN architecture to tune.
    dataset:
        Combined train+val graph list for this iteration.
    num_classes:
        Number of output classes (1 for regression).
    num_samples:
        Number of hyperparameter configurations to sample.
    iteration:
        Current bootstrap iteration (for result file naming).
    problem_type:
        Classification or regression.
    random_seed:
        Random seed for split generation.
    hpo_split_strategy:
        How to split data for HPO trials. ``CrossValidation`` (default) keeps
        the original K-fold behaviour with a single report per trial.
        ``Holdout`` and ``RepeatedHoldout`` report per epoch, enabling ASHA
        early stopping.
    n_folds:
        Number of folds. Used only when ``hpo_split_strategy`` is
        ``CrossValidation``.
    val_fraction:
        Fraction of data used for validation. Used by ``Holdout`` and
        ``RepeatedHoldout``.
    n_repeats:
        Number of independent random splits. Used only by ``RepeatedHoldout``.

    Returns
    -------
    dict
        Best hyperparameter configuration including ``LearningRate``,
        ``BatchSize``, and all architecture parameters.
    """
    problem_type = ProblemType(problem_type) if isinstance(problem_type, str) else problem_type

    hop_results_path = Path(exp_path) / "gnn_hyp_opt" / f"iteration_{iteration}"
    results_csv = hop_results_path / gnn_arch.value / f"{gnn_arch.value}.csv"

    config = get_gnn_search_grid(network=gnn_arch, random_seed=random_seed)

    if results_csv.exists():
        logger.info(f"Found existing HPO results at {results_csv}. Loading best config.")
        return _load_best_config(hop_results_path, gnn_arch, config)

    # --- Split indices ---
    splits = _build_splits(
        dataset=dataset,
        problem_type=problem_type,
        strategy=hpo_split_strategy,
        random_seed=random_seed,
        n_folds=n_folds,
        val_fraction=val_fraction,
        n_repeats=n_repeats,
    )

    # --- Ray Tune configuration ---
    tune_config = {k: tune.choice(v) if isinstance(v, list) else v for k, v in config.items()}

    # ASHA is meaningful only for strategies that report per epoch.
    use_asha = hpo_split_strategy != HpoSplitStrategy.CrossValidation
    asha = (
        ASHAScheduler(
            time_attr="epoch",
            metric="val_loss",
            mode="min",
            max_t=250,
            grace_period=50,
            reduction_factor=2,
        )
        if use_asha
        else None
    )

    reporter = CLIReporter(
        parameter_columns=[
            TrainingParam.AsymmetricLossStrength,
            TrainingParam.LearningRate,
            TrainingParam.BatchSize,
        ],
        metric_columns=["val_loss", "val_loss_std"],
    )

    ray.init(ignore_reinit_error=True, include_dashboard=False)

    results = tune.run(
        tune.with_parameters(
            _gnn_target_function,
            dataset=dataset,
            num_classes=num_classes,
            splits=splits,
            strategy=hpo_split_strategy,
            network=gnn_arch,
            problem_type=problem_type,
        ),
        config=tune_config,
        num_samples=num_samples,
        scheduler=asha,
        progress_reporter=reporter,
        storage_path=hop_results_path.resolve(),
        name=gnn_arch.value,
        resources_per_trial={"cpu": 0.5, "gpu": 0.5 if torch.cuda.is_available() else 0},
    )

    best_config = results.get_best_trial("val_loss", "min").config

    results_csv.parent.mkdir(parents=True, exist_ok=True)
    results.results_df.to_csv(results_csv, index=False)
    logger.info(f"HPO results saved to {results_csv}.")

    ray.shutdown()
    return best_config


# ---------------------------------------------------------------------------
# Ray Tune target function
# ---------------------------------------------------------------------------


def _gnn_target_function(
    config: dict,
    dataset: list,
    num_classes: int,
    splits: list[tuple[list[int], list[int]]],
    strategy: HpoSplitStrategy,
    network: Network,
    problem_type: ProblemType,
) -> None:
    """
    Ray Tune objective function — trains a GNN and reports validation loss.

    For ``CrossValidation``: trains each fold fully (250 epochs) and reports
    once at the end with the mean/std across folds. ASHA is not active.

    For ``Holdout`` / ``RepeatedHoldout``: trains all splits in epoch lockstep
    and reports after every epoch so ASHA can prune unpromising trials.

    This function is called by Ray Tune for each sampled configuration.
    It is not intended for direct use.
    """
    cfg = deepcopy(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr = cfg.pop(TrainingParam.LearningRate)
    batch_size = cfg.pop(TrainingParam.BatchSize)
    cfg.pop(TrainingParam.AsymmetricLossStrength, None)

    loss_fn = create_loss(problem_type)

    if strategy == HpoSplitStrategy.CrossValidation:
        # Original behaviour: train each fold fully, report once at end.
        fold_val_losses: list[float] = []

        for train_idx, val_idx in splits:
            train_set = [dataset[i] for i in train_idx]
            val_set = [dataset[i] for i in val_idx]

            train_loader = DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True,
                drop_last=len(train_set) % batch_size == 1,
            )
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

            model = create_network(
                network=network,
                problem_type=problem_type,
                n_node_features=dataset[0].num_node_features,
                n_edge_features=dataset[0].num_edge_features,
                n_classes=num_classes,
                **cfg,
            ).to(device)

            optimizer = create_optimizer(Optimizer.Adam, model, lr=lr)
            scheduler = create_scheduler(
                Scheduler.ReduceLROnPlateau, optimizer, patience=15, gamma=0.9, min_lr=1e-8
            )

            best_val_loss = float("inf")
            for _ in range(1, 251):
                train_network(model, train_loader, loss_fn, optimizer, device)
                val_loss = eval_network(model, val_loader, loss_fn, device)
                scheduler.step(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

            fold_val_losses.append(best_val_loss)

        session.report(
            {
                "val_loss": float(np.mean(fold_val_losses)),
                "val_loss_std": float(np.std(fold_val_losses)),
            }
        )

    else:
        # Holdout / RepeatedHoldout: epoch-lockstep with per-epoch reporting.
        split_data = []
        for train_idx, val_idx in splits:
            train_set = [dataset[i] for i in train_idx]
            val_set = [dataset[i] for i in val_idx]

            train_loader = DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True,
                drop_last=len(train_set) % batch_size == 1,
            )
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

            model = create_network(
                network=network,
                problem_type=problem_type,
                n_node_features=dataset[0].num_node_features,
                n_edge_features=dataset[0].num_edge_features,
                n_classes=num_classes,
                **cfg,
            ).to(device)

            optimizer = create_optimizer(Optimizer.Adam, model, lr=lr)
            scheduler = create_scheduler(
                Scheduler.ReduceLROnPlateau, optimizer, patience=15, gamma=0.9, min_lr=1e-8
            )
            split_data.append((model, train_loader, val_loader, optimizer, scheduler))

        best_val_losses = [float("inf")] * len(split_data)

        for epoch in range(1, 251):
            for k, (model, train_loader, val_loader, optimizer, scheduler) in enumerate(split_data):
                train_network(model, train_loader, loss_fn, optimizer, device)
                val_loss = eval_network(model, val_loader, loss_fn, device)
                scheduler.step(val_loss)
                if val_loss < best_val_losses[k]:
                    best_val_losses[k] = val_loss

            session.report(
                {
                    "val_loss": float(np.mean(best_val_losses)),
                    "val_loss_std": float(np.std(best_val_losses)),
                    "epoch": epoch,
                }
            )


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------


def _load_best_config(hop_results_path: Path, gnn_arch: Network, config_keys: dict) -> dict:
    """
    Load the best hyperparameter configuration from a saved CSV.

    Parameters
    ----------
    hop_results_path:
        Base path where HPO results are stored.
    gnn_arch:
        GNN architecture whose results to load.
    config_keys:
        Dict whose keys are the hyperparameter names to look up.
        Used to construct column names (``config/{param}``).

    Returns
    -------
    dict
        Best configuration as a plain Python dict with native types.

    Raises
    ------
    KeyError
        If ``val_loss`` column is missing from the results CSV.
    FileNotFoundError
        If the results CSV does not exist.
    """
    results_csv = hop_results_path / gnn_arch.value / f"{gnn_arch.value}.csv"

    if not results_csv.exists():
        raise FileNotFoundError(f"HPO results file not found: {results_csv}")

    results_df = pd.read_csv(results_csv)

    if "val_loss" not in results_df.columns:
        raise KeyError(f"'val_loss' column missing from {results_csv}.")

    best_row = results_df.loc[results_df["val_loss"].idxmin()]
    best_config: dict = {}

    for param in config_keys:
        col = f"config/{param}"
        if col not in results_df.columns:
            logger.warning(f"Column '{col}' not found in HPO results CSV — skipping.")
            continue

        value = best_row[col]
        if isinstance(value, np.generic):
            value = value.item()
        if pd.isna(value):
            value = None

        best_config[param] = value

    return best_config
