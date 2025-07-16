import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from polynet.app.options.data import DataOptions
from polynet.app.options.file_paths import (
    gnn_model_dir,
    gnn_raw_data_file,
    gnn_raw_data_path,
    polynet_experiments_base_dir,
)
from polynet.app.options.general_experiment import GeneralConfigOptions
from polynet.app.options.representation import RepresentationOptions
from polynet.app.options.train_GNN import TrainGNNOptions
from polynet.app.services.model_training import split_data
from polynet.app.utils import (
    get_predicted_label_column_name,
    get_score_column_name,
    get_true_label_column_name,
)
from polynet.call_methods import (
    compute_class_weights,
    create_network,
    make_loss,
    make_optimizer,
    make_scheduler,
)
from polynet.featurizer.graph_representation.polymer import CustomPolymerGraph
from polynet.options.enums import (
    DataSets,
    Optimizers,
    ProblemTypes,
    Results,
    Schedulers,
    SplitMethods,
    SplitTypes,
)
from polynet.utils.model_training import predict_network, train_model


def train_network(
    train_gnn_options: TrainGNNOptions,
    general_experiment_options: GeneralConfigOptions,
    data_options: DataOptions,
    representation_options: RepresentationOptions,
    experiment_name: str,
    train_val_test_idxs: tuple,
):
    experiment_path = polynet_experiments_base_dir() / experiment_name

    # === Step 1: Load and prepare data
    data = pd.read_csv(
        gnn_raw_data_file(file_name=data_options.data_name, experiment_path=experiment_path),
        index_col=0,
    )

    # === Step 2: Create dataset and filter splits
    dataset = CustomPolymerGraph(
        filename=data_options.data_name,
        root=gnn_raw_data_path(experiment_path=experiment_path).parent,
        smiles_cols=data_options.smiles_cols,
        target_col=data_options.target_variable_col,
        id_col=data_options.id_col,
        weights_col=representation_options.weights_col,
        node_feats=representation_options.node_feats,
        edge_feats=representation_options.edge_feats,
    )

    train_ids, val_ids, test_ids = train_val_test_idxs

    def filter_dataset_by_ids(dataset, ids):
        return [data for data in dataset if data.idx in ids]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trained_models = {}

    # === Step 4: Train one model for each GNN architecture
    for i, (train_idxs, val_idxs, test_idxs) in enumerate(zip(train_ids, val_ids, test_ids)):

        iteration = i + 1

        trained_models[iteration] = {}
        trained_models[iteration][Results.Model.value] = {}

        train_set = filter_dataset_by_ids(dataset, train_idxs)
        val_set = filter_dataset_by_ids(dataset, val_idxs)
        test_set = filter_dataset_by_ids(dataset, test_idxs)

        # === Step 3: Prepare dataloaders
        batch_size = train_gnn_options.GNNBatchSize
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        trained_models[iteration][Results.Loaders.value] = (train_loader, val_loader, test_loader)

        for gnn_arch, arch_params in train_gnn_options.GNNConvolutionalLayers.items():
            st_msg = f"Training model with architecture: {gnn_arch}"

            model_kwargs = {
                "n_node_features": dataset[0].num_node_features,
                "n_edge_features": dataset[0].num_edge_features,
                "pooling": train_gnn_options.GNNPoolingMethod,
                "n_convolutions": train_gnn_options.GNNNumberOfLayers,
                "embedding_dim": train_gnn_options.GNNEmbeddingDimension,
                "readout_layers": train_gnn_options.GNNReadoutLayers,
                "n_classes": int(data_options.num_classes),
                "dropout": train_gnn_options.GNNDropoutRate,
                "apply_weighting_to_graph": train_gnn_options.ApplyMonomerWeighting,
                "seed": general_experiment_options.random_seed + iteration,
            }

            all_kwargs = {**model_kwargs, **arch_params}

            model = create_network(
                network=gnn_arch, problem_type=data_options.problem_type, **all_kwargs
            )

            optimizer = make_optimizer(Optimizers.Adam, model, lr=train_gnn_options.GNNLearningRate)
            scheduler = make_scheduler(
                Schedulers.ReduceLROnPlateau, optimizer, step_size=15, gamma=0.9, min_lr=1e-8
            )

            if (
                model.problem_type == ProblemTypes.Classification
                and train_gnn_options.AsymmetricLoss
            ):
                weights = compute_class_weights(
                    labels=data[data_options.target_variable_col].to_numpy(),
                    num_classes=int(data_options.num_classes),
                )
            else:
                weights = None

            loss_fn = make_loss(model.problem_type, asymmetric_loss_weights=weights)

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
            trained_models[iteration][Results.Model.value][gnn_arch] = model

    return trained_models


def predict_gnn_model(model, loaders, target_variable_name=None):

    train_loader, val_loader, test_loader = loaders

    idx, y_true, y_pred, y_score = predict_network(model, train_loader)
    predictions_train = create_results_dataframe(
        target_variable_name=target_variable_name,
        idx=idx,
        y_pred=y_pred,
        y_true=y_true,
        y_score=y_score,
        set_name=DataSets.Training.value,
        model_name=model._name,
    )

    idx, y_true, y_pred, y_score = predict_network(model, val_loader)
    predictions_val = create_results_dataframe(
        target_variable_name=target_variable_name,
        idx=idx,
        y_pred=y_pred,
        y_true=y_true,
        y_score=y_score,
        set_name=DataSets.Validation.value,
        model_name=model._name,
    )
    idx, y_true, y_pred, y_score = predict_network(model, test_loader)
    prediction_test = create_results_dataframe(
        target_variable_name=target_variable_name,
        idx=idx,
        y_pred=y_pred,
        y_true=y_true,
        y_score=y_score,
        set_name=DataSets.Test.value,
        model_name=model._name,
    )

    predictions = pd.concat(
        [predictions_train, predictions_val, prediction_test], ignore_index=True
    )

    return predictions


def create_results_dataframe(
    target_variable_name: str,
    idx: list,
    y_pred: list,
    y_true: list,
    y_score: list,
    set_name: str,
    model_name: str = None,
):

    true_label = get_true_label_column_name(target_variable_name=target_variable_name)
    predicted_label = get_predicted_label_column_name(
        target_variable_name=target_variable_name, model_name=model_name
    )

    results = pd.DataFrame(
        {
            Results.Index.value: idx,
            Results.Set.value: set_name,
            true_label: y_true,
            predicted_label: y_pred,
        }
    )

    if y_score is not None:
        probs = prepare_probs_df(
            probs=y_score, target_variable_name=target_variable_name, model_name=model_name
        )
        results = pd.concat([results, probs], axis=1)

    return results


def prepare_probs_df(probs: np.ndarray, target_variable_name: str = None, model_name: str = None):
    """
    Convert probability predictions into a DataFrame.

    - For binary classification (2 classes), include only the second class (index 1).
    - For multi-class classification (3+ classes), include a column per class.

    Args:
        probs (np.ndarray): Array of shape (n_samples, n_classes)
        target_variable_name (str): Name of the target variable
        model_name (str): Name of the model

    Returns:
        pd.DataFrame: A DataFrame with appropriately named probability columns
    """
    n_classes = probs.shape[1] if probs.ndim > 1 else 1
    probs_df = pd.DataFrame()

    if n_classes == 2:
        col_name = get_score_column_name(
            target_variable_name=target_variable_name, model_name=model_name
        )
        # Binary classification: only use the second class (probability of class 1)
        probs_df[f"{col_name}"] = probs[:, 1]
    else:
        # Multi-class classification: create one column per class
        for i in range(n_classes):
            col_name = get_score_column_name(
                target_variable_name=target_variable_name, model_name=model_name, class_num=i
            )
            probs_df[f"{col_name} {i}"] = probs[:, i]

    return probs_df
