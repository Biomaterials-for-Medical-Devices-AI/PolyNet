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
from polynet.call_methods import create_network, make_loss, make_optimizer, make_scheduler
from polynet.featurizer.graph_representation.polymer import CustomPolymerGraph
from polynet.options.enums import DataSets, Optimizers, Results, Schedulers, SplitMethods
from polynet.utils.data_preprocessing import class_balancer
from polynet.utils.model_training import predict_network, train_model


def train_network(
    train_gnn_options: TrainGNNOptions,
    general_experiment_options: GeneralConfigOptions,
    data_options: DataOptions,
    representation_options: RepresentationOptions,
    experiment_name: str,
):
    experiment_path = polynet_experiments_base_dir() / experiment_name

    # === Step 1: Load and prepare data
    data = pd.read_csv(
        gnn_raw_data_file(file_name=data_options.data_name, experiment_path=experiment_path),
        index_col=0,
    )

    train_ids, val_ids, test_ids = get_data_split_indices(
        data=data, data_options=data_options, general_experiment_options=general_experiment_options
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

    def filter_dataset_by_ids(dataset, ids):
        return [data for data in dataset if data.idx in ids]

    train_set = filter_dataset_by_ids(dataset, train_ids)
    val_set = filter_dataset_by_ids(dataset, val_ids)
    test_set = filter_dataset_by_ids(dataset, test_ids)

    # === Step 3: Prepare dataloaders
    batch_size = train_gnn_options.GNNBatchSize
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    loaders = (train_loader, val_loader, test_loader)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trained_models = {}

    # === Step 4: Train one model for each GNN architecture
    for gnn_arch, arch_params in train_gnn_options.GNNConvolutionalLayers.items():
        st_msg = f"Training model with architecture: {gnn_arch}"
        print(st_msg)

        model_kwargs = {
            "n_node_features": dataset[0].num_node_features,
            "n_edge_features": dataset[0].num_edge_features,
            "pooling": train_gnn_options.GNNPoolingMethod,
            "n_convolutions": train_gnn_options.GNNNumberOfLayers,
            "embedding_dim": train_gnn_options.GNNEmbeddingDimension,
            "readout_layers": train_gnn_options.GNNReadoutLayers,
            "n_classes": int(data_options.num_classes),
            "dropout": train_gnn_options.GNNDropoutRate,
            "seed": general_experiment_options.random_seed,
        }

        all_kwargs = {**model_kwargs, **arch_params}

        model = create_network(
            network=gnn_arch, problem_type=data_options.problem_type, **all_kwargs
        )

        optimizer = make_optimizer(Optimizers.Adam, model, lr=train_gnn_options.GNNLearningRate)
        scheduler = make_scheduler(
            Schedulers.ReduceLROnPlateau, optimizer, step_size=15, gamma=0.9, min_lr=1e-8
        )
        loss_fn = make_loss(model.problem_type)

        model = train_model(
            model, train_loader, val_loader, test_loader, loss_fn, optimizer, scheduler, device
        )

        trained_models[gnn_arch] = model

    return trained_models, loaders


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

    predictions = pd.concat([predictions_train, predictions_val, prediction_test])

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

    true_label = (
        f"{Results.Label.value} {target_variable_name}"
        if target_variable_name
        else Results.Label.value
    )
    predicted_label = (
        f"{model_name} {Results.Predicted.value} {target_variable_name}"
        if target_variable_name
        else f"{model_name} {Results.Predicted.value}"
    )
    return pd.DataFrame(
        {
            Results.Index.value: idx,
            Results.Set.value: set_name,
            true_label: y_true,
            predicted_label: y_pred,
        }
    )


def get_data_split_indices(
    data: pd.DataFrame, data_options: DataOptions, general_experiment_options: GeneralConfigOptions
):
    # Initial train-test split
    train_data, test_data = split_data(
        data=data,
        test_size=general_experiment_options.test_ratio,
        stratify=(
            data[data_options.target_variable_col]
            if general_experiment_options.split_method == SplitMethods.Stratified
            else None
        ),
        random_state=general_experiment_options.random_seed,
    )

    # Optional class balancing on training set
    if general_experiment_options.train_set_balance:
        train_data = class_balancer(
            data=train_data,
            target=data_options.target_variable_col,
            desired_class_proportion=general_experiment_options.train_set_balance,
        )

    # Further split train into train/validation
    train_data, val_data = split_data(
        data=train_data,
        test_size=general_experiment_options.val_ratio,
        stratify=(
            train_data[data_options.target_variable_col]
            if general_experiment_options.split_method == SplitMethods.Stratified
            else None
        ),
        random_state=general_experiment_options.random_seed,
    )

    return train_data.index, val_data.index, test_data.index
