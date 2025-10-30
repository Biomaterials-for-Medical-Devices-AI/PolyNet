from copy import deepcopy
from pathlib import Path

import numpy as np
import ray
from ray import tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import KFold, StratifiedKFold
import torch
from torch.nn import Module
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from polynet.app.options.search_grids import get_grid_search
from polynet.call_methods import (
    compute_class_weights,
    create_network,
    make_loss,
    make_optimizer,
    make_scheduler,
)
from polynet.options.enums import NetworkParams, Networks, Optimizers, ProblemTypes, Schedulers


def filter_dataset_by_ids(dataset, ids):
    return [data for data in dataset if data.idx in ids]


def train_GNN_ensemble(
    experiment_path: Path,
    dataset: Dataset,
    split_indexes: tuple,
    gnn_conv_params: dict,
    problem_type: ProblemTypes,
    num_classes: int,
    random_seed: int,
):

    train_ids, val_ids, test_ids = deepcopy(split_indexes)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    trained_models = {}
    loaders = {}

    assymetric_losses = {}
    lrs = {}
    batch_sizes = {}

    for i, (train_idxs, val_idxs, test_idxs) in enumerate(zip(train_ids, val_ids, test_ids)):

        iteration = i + 1

        train_set = filter_dataset_by_ids(dataset, train_idxs)
        val_set = filter_dataset_by_ids(dataset, val_idxs)
        test_set = filter_dataset_by_ids(dataset, test_idxs)

        # === This loaders are only created to make predictions later, no for training
        train_loader = DataLoader(train_set, shuffle=False)
        val_loader = DataLoader(val_set, shuffle=False)
        test_loader = DataLoader(test_set, shuffle=False)

        loaders[str(iteration)] = (train_loader, val_loader, test_loader)

        for gnn_arch, arch_params in gnn_conv_params.items():

            is_hpo = not arch_params

            if is_hpo:
                print("No hyperparameters have been set. Initialising hyperparameter optimisation.")
                arch_params = gnn_hyp_opt(
                    exp_path=experiment_path,
                    gnn_arch=gnn_arch,
                    dataset=train_set + val_set,
                    num_classes=int(num_classes),
                    num_samples=50,
                    iteration=iteration,
                    problem_type=problem_type,
                    random_seed=random_seed + i,
                )
                print("Hyperparameter optimisation finalised.\nSelected hyperparameters:")
                for hyp, val in arch_params:
                    print(hyp + ": " + val)

                # take out non-model related params
                assymetric_loss_strength = arch_params.pop(
                    NetworkParams.AssymetricLossStrength, None
                )
                lr = arch_params.pop(NetworkParams.LearningRate, None)
                batch_size = arch_params.pop(NetworkParams.BatchSize, None)

            else:
                if gnn_arch not in assymetric_losses:
                    assymetric_losses[gnn_arch] = arch_params.pop(
                        NetworkParams.AssymetricLossStrength
                    )
                if gnn_arch not in lrs:
                    lrs[gnn_arch] = arch_params.pop(NetworkParams.LearningRate)
                if gnn_arch not in batch_sizes:
                    batch_sizes[gnn_arch] = arch_params.pop(NetworkParams.BatchSize)

                assymetric_loss_strength = assymetric_losses[gnn_arch]
                lr = lrs[gnn_arch]
                batch_size = batch_sizes[gnn_arch]

            # get model params together and create model
            model_kwargs = {
                # data related kwargs
                "n_node_features": dataset[0].num_node_features,
                "n_edge_features": dataset[0].num_edge_features,
                "n_classes": int(num_classes),
                # Experiment seed
                "seed": random_seed + i,
            }
            all_kwargs = {**model_kwargs, **arch_params}
            model = create_network(network=gnn_arch, problem_type=problem_type, **all_kwargs).to(
                device
            )

            # create loaders
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_set, shuffle=False)
            test_loader = DataLoader(test_set, shuffle=False)

            # Create optimizer, scheduler and loss function
            optimizer = make_optimizer(Optimizers.Adam, model, lr=lr)
            scheduler = make_scheduler(
                Schedulers.ReduceLROnPlateau, optimizer, step_size=15, gamma=0.9, min_lr=1e-8
            )
            # if (
            #     model.problem_type == ProblemTypes.Classification
            #     and assymetric_loss_strength is not None
            # ):
            #     weights = compute_class_weights(
            #         labels=data[data_options.target_variable_col].to_numpy(),
            #         num_classes=int(data_options.num_classes),
            #         imbalance_strength=assymetric_loss_strength,
            #     )
            # else:
            #     weights = None
            loss_fn = make_loss(model.problem_type, asymmetric_loss_weights=None)

            model = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                loss=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
            )
            model_log_name = f"{gnn_arch}_{iteration}"
            trained_models[model_log_name] = model

    return trained_models, loaders


def train_model(
    model: Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    loss: Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epochs: int = 250,
):
    best_val_loss = float("inf")
    train_list, val_list, test_list = [], [], []

    for epoch in range(1, epochs + 1):
        train_loss = train_network(model, train_loader, loss, optimizer, device)
        val_loss = eval_network(model, val_loader, loss, device)
        test_loss = eval_network(model, test_loader, loss, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = deepcopy(model.state_dict())

        scheduler.step(val_loss)

        print(
            f"Epoch: {epoch:03d}, LR: {scheduler.get_last_lr()[0]:3f}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}"
        )
        train_list.append(train_loss)
        val_list.append(val_loss)
        test_list.append(test_loss)

    model.load_state_dict(best_model_state)
    model.losses = (train_list, val_list, test_list)

    return model


def train_network(model, train_loader, loss_fn, optimizer, device):
    train_loss = 0
    model.train()

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

        if model.problem_type == ProblemTypes.Regression:
            loss = torch.sqrt(loss_fn(out.squeeze(1), batch.y.float()))
        elif model.problem_type == ProblemTypes.Classification:
            loss = loss_fn(out, batch.y.long())

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch.num_graphs

    return train_loss / len(train_loader.dataset)


def eval_network(model, test_loader, loss_fn, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(
                x=batch.x,
                edge_index=batch.edge_index,
                batch_index=batch.batch,
                edge_attr=batch.edge_attr,
                monomer_weight=batch.weight_monomer,
            )

            if model.problem_type == ProblemTypes.Regression:
                loss = torch.sqrt(loss_fn(out.squeeze(1), batch.y.float()))
            elif model.problem_type == ProblemTypes.Classification:
                loss = loss_fn(out, batch.y.long())

            test_loss += loss.item() * batch.num_graphs

    return test_loss / len(test_loader.dataset)


def gnn_target_function(
    config: dict,
    dataset: list,
    num_classes: int,
    train_idxs: list,
    val_idxs: list,
    network: Networks,
    problem_type: ProblemTypes,
):
    """
    Trains and evaluates a GNN model using K-fold cross-validation.
    Reports mean and std of validation loss to Ray Tune.
    """

    cfg = deepcopy(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr = cfg.pop(NetworkParams.LearningRate)
    batch_size = cfg.pop(NetworkParams.BatchSize)
    assymetric_loss_strength = cfg.pop(NetworkParams.AssymetricLossStrength)

    fold_val_losses = []

    for fold, (train_idx, val_idx) in enumerate(zip(train_idxs, val_idxs), 1):

        # Prepare datasets
        train_dataset = [dataset[i] for i in train_idx]
        val_dataset = [dataset[i] for i in val_idx]

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Prepare model input kwargs
        data_kwargs = {
            "n_node_features": dataset[0].num_node_features,
            "n_edge_features": dataset[0].num_edge_features,
            "n_classes": num_classes,
        }
        all_kwargs = {**data_kwargs, **cfg}

        # Create model and training tools
        model = create_network(network=network, problem_type=problem_type, **all_kwargs).to(device)
        optimizer = make_optimizer(Optimizers.Adam, model, lr=lr)
        scheduler = make_scheduler(
            Schedulers.ReduceLROnPlateau, optimizer, step_size=15, gamma=0.9, min_lr=1e-8
        )
        loss_fn = make_loss(model.problem_type, asymmetric_loss_weights=None)

        best_val_loss = float("inf")

        # === Training loop ===
        for epoch in range(1, 251):

            train_loss = train_network(
                model=model,
                train_loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
            )

            val_loss = eval_network(
                model=model, test_loader=val_loader, loss_fn=loss_fn, device=device
            )

            scheduler.step(val_loss)

            # Keep best validation loss for this fold
            if val_loss < best_val_loss:
                best_val_loss = val_loss

            # Optional: report intermediate results to Ray (helps ASHA)
            # session.report({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})

        fold_val_losses.append(best_val_loss)

    # Compute mean and std across folds
    mean_val_loss = np.mean(fold_val_losses)
    std_val_loss = np.std(fold_val_losses)

    # Final report to Ray
    session.report({"val_loss": mean_val_loss, "val_loss_std": std_val_loss})


def gnn_hyp_opt(
    exp_path: Path,
    gnn_arch: Networks,
    dataset: list,
    num_classes: int,
    num_samples: int,
    iteration: int,
    problem_type: ProblemTypes,
    random_seed: int,
    n_folds: int = 5,
):
    """
    Runs Ray Tune hyperparameter optimization with early stopping.
    """

    # --- Configure early stopping via ASHA ---
    asha_scheduler = ASHAScheduler(
        time_attr="epoch",
        metric="val_loss",
        mode="min",
        max_t=250,
        grace_period=50,  # allow warmup epochs before early stopping
        reduction_factor=2,  # controls aggressiveness
    )

    metric_cols = ["train_loss", "val_loss", "val_loss_std", "epoch"] + [
        f"val_loss_fold_{i+1}" for i in range(n_folds)
    ]

    reporter = CLIReporter(
        parameter_columns=[
            NetworkParams.AssymetricLossStrength,
            NetworkParams.LearningRate,
            NetworkParams.BatchSize,
            NetworkParams.NumConvolutions,
            NetworkParams.EmbeddingDim,
            NetworkParams.ReadoutLayers,
        ],
        metric_columns=metric_cols,
    )

    config = get_grid_search(
        model_name=gnn_arch, problem_type=problem_type, random_seed=random_seed
    )

    for key, value in config.items():
        if isinstance(value, list):
            config[key] = tune.choice(value)

    # --- Create indices for cross-validation ---
    y = [data.y.item() for data in dataset]
    if problem_type == ProblemTypes.Classification:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    else:
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    train_set_idxs, val_set_idxs = [], []
    for train_idx, val_idx in cv.split(np.zeros(len(y)), y):
        train_set_idxs.append(train_idx)
        val_set_idxs.append(val_idx)

    # --- Run Ray Tune ---
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    results = tune.run(
        tune.with_parameters(
            gnn_target_function,
            dataset=dataset,
            num_classes=num_classes,
            train_idxs=train_set_idxs,
            val_idxs=val_set_idxs,
            network=gnn_arch,
            problem_type=problem_type,
        ),
        config=config,
        num_samples=num_samples,
        scheduler=asha_scheduler,
        progress_reporter=reporter,
        storagstorage_path=Path(exp_path / "gnn_hyp_opt" / f"iteration_{iteration}").resolve(),
        name=gnn_arch,
        resources_per_trial={"cpu": 0.5, "gpu": 0.5 if torch.cuda.is_available() else 0},
    )

    best_trial = results.get_best_trial("val_loss", "min")
    best_config = best_trial.config

    all_runs_df = results.results_df
    all_runs_df.to_csv(exp_path / "gnn_hyp_opt" f"/{gnn_arch}.csv", index=False)

    ray.shutdown()

    return best_config
