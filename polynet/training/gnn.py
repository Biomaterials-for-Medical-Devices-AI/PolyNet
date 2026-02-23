"""
polynet.training.gnn
=====================
GNN model training, evaluation, and ensemble orchestration.

Provides the core training loop (``train_model``), single-epoch
forward passes (``train_network``, ``eval_network``), and the
top-level ensemble trainer (``train_gnn_ensemble``) that iterates
over bootstrap splits and GNN architectures.

Hyperparameter optimisation is handled separately in
``polynet.training.hyperopt``.

Public API
----------
::

    from polynet.training.gnn import train_gnn_ensemble, train_model
"""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path

import torch
from torch.nn import Module
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from polynet.config.enums import Network, Optimizer, ProblemType, Scheduler, TrainingParam
from polynet.factories.loss import create_loss
from polynet.factories.network import create_network
from polynet.factories.optimizer import create_optimizer, create_scheduler
from polynet.training.metrics import compute_class_weights

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------


def filter_dataset_by_ids(dataset, ids: list) -> list:
    """Return all graph objects from ``dataset`` whose ``idx`` is in ``ids``."""
    id_set = set(ids)
    return [data for data in dataset if data.idx in id_set]


# ---------------------------------------------------------------------------
# Ensemble training
# ---------------------------------------------------------------------------


def train_gnn_ensemble(
    experiment_path: Path,
    dataset: Dataset,
    split_indexes: tuple,
    gnn_conv_params: dict[Network, dict | None],
    problem_type: ProblemType | str,
    num_classes: int,
    random_seed: int,
) -> tuple[dict, dict]:
    """
    Train a GNN ensemble across all bootstrap iterations and architectures.

    For each iteration, subsets the dataset by sample IDs, then trains
    each requested GNN architecture. If no hyperparameters are provided
    for an architecture, Ray Tune HPO is triggered automatically.

    Parameters
    ----------
    experiment_path:
        Root path for saving HPO results.
    dataset:
        Full PyG dataset containing all polymer graphs.
    split_indexes:
        Triple of ``(train_ids, val_ids, test_ids)`` as returned by
        ``get_data_split_indices``.
    gnn_conv_params:
        Mapping from ``Network`` enum to hyperparameter dict. Pass an
        empty dict or ``None`` to trigger HPO for that architecture.
    problem_type:
        Classification or regression.
    num_classes:
        Number of output classes (1 for regression).
    random_seed:
        Base random seed. Each iteration uses ``random_seed + i``.

    Returns
    -------
    tuple[dict, dict]
        ``(trained_models, loaders)`` where:
        - ``trained_models``: ``{"{arch}_{iteration}": fitted_model}``
        - ``loaders``: ``{"{iteration}": (train_loader, val_loader, test_loader)}``
    """
    # Deferred import to avoid circular dependency with hyperopt
    from polynet.training.hyperopt import gnn_hyp_opt

    problem_type = ProblemType(problem_type) if isinstance(problem_type, str) else problem_type

    train_ids, val_ids, test_ids = deepcopy(split_indexes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training device: {device}")

    trained_models: dict = {}
    loaders: dict = {}

    # Cache non-HPO training params per architecture across iterations
    arch_lr: dict[Network, float] = {}
    arch_batch_size: dict[Network, int] = {}
    arch_loss_strength: dict[Network, float | None] = {}

    for i, (train_idxs, val_idxs, test_idxs) in enumerate(zip(train_ids, val_ids, test_ids)):
        iteration = i + 1
        seed = random_seed + i

        train_set = filter_dataset_by_ids(dataset, train_idxs)
        val_set = filter_dataset_by_ids(dataset, val_idxs)
        test_set = filter_dataset_by_ids(dataset, test_idxs)

        # Prediction-only loaders (batch_size=1, no shuffle)
        loaders[str(iteration)] = (
            DataLoader(train_set, shuffle=False),
            DataLoader(val_set, shuffle=False),
            DataLoader(test_set, shuffle=False),
        )

        for gnn_arch, arch_params in gnn_conv_params.items():
            arch_params = arch_params or {}
            is_hpo = not arch_params

            if is_hpo:
                logger.info(
                    f"No hyperparameters for {gnn_arch.value} — "
                    "initialising hyperparameter optimisation."
                )
                arch_params = gnn_hyp_opt(
                    exp_path=experiment_path,
                    gnn_arch=gnn_arch,
                    dataset=train_set + val_set,
                    num_classes=int(num_classes),
                    num_samples=150,
                    iteration=iteration,
                    problem_type=problem_type,
                    random_seed=seed,
                )
                logger.info(f"HPO complete. Best params: {arch_params}")
                loss_strength = arch_params.pop(TrainingParam.AsymmetricLossStrength, None)
                lr = arch_params.pop(TrainingParam.LearningRate)
                batch_size = arch_params.pop(TrainingParam.BatchSize)
            else:
                if gnn_arch not in arch_lr:
                    arch_loss_strength[gnn_arch] = arch_params.pop(
                        TrainingParam.AsymmetricLossStrength, None
                    )
                    arch_lr[gnn_arch] = arch_params.pop(TrainingParam.LearningRate)
                    arch_batch_size[gnn_arch] = arch_params.pop(TrainingParam.BatchSize)

                loss_strength = arch_loss_strength[gnn_arch]
                lr = arch_lr[gnn_arch]
                batch_size = arch_batch_size[gnn_arch]

            model = create_network(
                network=gnn_arch,
                problem_type=problem_type,
                n_node_features=dataset[0].num_node_features,
                n_edge_features=dataset[0].num_edge_features,
                n_classes=int(num_classes),
                seed=seed,
                **arch_params,
            ).to(device)

            train_loader = DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True,
                drop_last=len(train_set) % batch_size == 1,
            )
            val_loader = DataLoader(val_set, shuffle=False)
            test_loader = DataLoader(test_set, shuffle=False)

            class_weights = None
            if problem_type == ProblemType.Classification and loss_strength is not None:
                all_labels = [data.y.item() for data in train_set]
                class_weights = compute_class_weights(
                    labels=all_labels,
                    num_classes=int(num_classes),
                    imbalance_strength=loss_strength,
                )

            optimizer = create_optimizer(Optimizer.Adam, model, lr=lr)
            scheduler = create_scheduler(
                Scheduler.ReduceLROnPlateau, optimizer, patience=15, gamma=0.9, min_lr=1e-8
            )
            loss_fn = create_loss(problem_type, class_weights=class_weights)

            model = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
            )
            trained_models[f"{gnn_arch.value}_{iteration}"] = model

    return trained_models, loaders


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def train_model(
    model: Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    loss_fn: Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str | torch.device,
    epochs: int = 250,
) -> Module:
    """
    Train a GNN model with early stopping based on validation loss.

    Saves the best model state (lowest validation loss) and restores
    it at the end of training. Loss curves are stored on ``model.losses``
    as ``(train_losses, val_losses, test_losses)`` for later plotting.

    Parameters
    ----------
    model:
        Initialised GNN model.
    train_loader, val_loader, test_loader:
        PyG DataLoaders for each split.
    loss_fn:
        Instantiated loss function.
    optimizer:
        Instantiated optimizer bound to ``model.parameters()``.
    scheduler:
        Learning rate scheduler. Expected to accept ``scheduler.step(val_loss)``.
    device:
        ``"cuda"`` or ``"cpu"``.
    epochs:
        Number of training epochs.

    Returns
    -------
    Module
        The trained model with best weights restored.
    """
    best_val_loss = float("inf")
    best_state = None
    train_list, val_list, test_list = [], [], []

    for epoch in range(1, epochs + 1):
        train_loss = train_network(model, train_loader, loss_fn, optimizer, device)
        val_loss = eval_network(model, val_loader, loss_fn, device)
        test_loss = eval_network(model, test_loader, loss_fn, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())

        scheduler.step(val_loss)

        logger.info(
            f"Epoch {epoch:03d} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"Test: {test_loss:.4f}"
        )

        train_list.append(train_loss)
        val_list.append(val_loss)
        test_list.append(test_loss)

    if best_state is not None:
        model.load_state_dict(best_state)

    model.losses = (train_list, val_list, test_list)
    return model


def train_network(
    model: Module,
    train_loader: DataLoader,
    loss_fn: Module,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
) -> float:
    """
    Run one training epoch and return the mean training loss.

    Parameters
    ----------
    model:
        GNN model in training mode.
    train_loader:
        DataLoader for the training set.
    loss_fn:
        Loss function.
    optimizer:
        Optimizer.
    device:
        Target device.

    Returns
    -------
    float
        Mean loss per sample across the epoch.
    """
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(
            x=batch.x,
            edge_index=batch.edge_index,
            batch_index=batch.batch,
            edge_attr=batch.edge_attr,
            monomer_weight=batch.weight_monomer,
        )

        loss = _compute_loss(out, batch.y, loss_fn, model.problem_type)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(train_loader.dataset)


def eval_network(
    model: Module, loader: DataLoader, loss_fn: Module, device: str | torch.device
) -> float:
    """
    Evaluate a model on a DataLoader and return the mean loss.

    Parameters
    ----------
    model:
        GNN model in evaluation mode.
    loader:
        DataLoader for the evaluation set.
    loss_fn:
        Loss function.
    device:
        Target device.

    Returns
    -------
    float
        Mean loss per sample.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(
                x=batch.x,
                edge_index=batch.edge_index,
                batch_index=batch.batch,
                edge_attr=batch.edge_attr,
                monomer_weight=batch.weight_monomer,
            )
            loss = _compute_loss(out, batch.y, loss_fn, model.problem_type)
            total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


def _compute_loss(
    out: torch.Tensor, y: torch.Tensor, loss_fn: Module, problem_type: ProblemType
) -> torch.Tensor:
    """Compute the appropriate loss based on problem type."""
    if problem_type == ProblemType.Regression:
        return torch.sqrt(loss_fn(out.squeeze(1), y.float()))
    return loss_fn(out, y.long())
