from polynet.app.options.train_GNN import TrainGNNOptions
from polynet.app.options.data import DataOptions
from polynet.app.options.general_experiment import GeneralConfigOptions
from polynet.app.options.representation import RepresentationOptions
from polynet.app.options.file_paths import (
    train_gnn_model_options_path,
    gnn_raw_data_path,
    polynet_experiments_base_dir,
    gnn_raw_data_file,
)
from polynet.featurizer.graph_representation.polymer import CustomPolymerGraph
import pandas as pd
from polynet.app.services.model_training import split_data
from polynet.call_methods import create_network, make_optimizer, make_loss, make_scheduler
import torch
from polynet.options.enums import Networks, Pooling, Optimizers, Schedulers, SplitMethods, DataSets
from torch_geometric.loader import DataLoader
from polynet.utils.model_training import train_model
from polynet.utils.data_preprocessing import class_balancer, print_class_balance
from polynet.utils.model_training import predict_network
from polynet.app.utils import save_data


def train_network(
    train_gnn_options: TrainGNNOptions,
    general_experiment_options: GeneralConfigOptions,
    data_options: DataOptions,
    representation_options: RepresentationOptions,
    experiment_name: str,
):

    experiment_path = polynet_experiments_base_dir() / experiment_name

    weights_col = representation_options.weights_col
    node_feats = representation_options.node_feats
    edge_feats = representation_options.edge_feats

    data = pd.read_csv(
        gnn_raw_data_file(file_name=data_options.data_name, experiment_path=experiment_path),
        index_col=0,
    )

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

    train_data = class_balancer(
        data=train_data, target=data_options.target_variable_col, desired_class_proportion=0.6
    )

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

    train_ids = train_data.index
    val_ids = val_data.index
    test_ids = test_data.index

    dataset = CustomPolymerGraph(
        filename=data_options.data_name,
        root=gnn_raw_data_path(experiment_path=experiment_path).parent,
        smiles_cols=data_options.smiles_cols,
        target_col=data_options.target_variable_col,
        id_col=data_options.id_col,
        weights_col=weights_col,
        node_feats=node_feats,
        edge_feats=edge_feats,
    )

    train_set = [data for data in dataset if data.idx in train_ids]
    val_set = [data for data in dataset if data.idx in val_ids]
    test_set = [data for data in dataset if data.idx in test_ids]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_network(
        network=list(train_gnn_options.GNNConvolutionalLayers.keys())[0],
        improved=True,
        problem_type=data_options.problem_type,
        n_node_features=dataset[0].num_node_features,
        n_edge_features=dataset[0].num_edge_features,
        pooling=train_gnn_options.GNNPoolingMethod,
        n_convolutions=train_gnn_options.GNNNumberOfLayers,
        embedding_dim=train_gnn_options.GNNEmbeddingDimension,
        readout_layers=train_gnn_options.GNNReadoutLayers,
        n_classes=2,
        dropout=train_gnn_options.GNNDropoutRate,
        seed=42,
    )
    optimizer = make_optimizer(Optimizers.Adam, model, lr=0.01)
    loss = make_loss(model.problem_type)
    scheduler = make_scheduler(
        Schedulers.ReduceLROnPlateau, optimizer, step_size=15, gamma=0.9, min_lr=1e-8
    )

    batch_size = train_gnn_options.GNNBatchSize
    train_loader, val_loader, test_loader = (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size, shuffle=False),
        DataLoader(test_set, batch_size=batch_size, shuffle=False),
    )
    model = train_model(
        model, train_loader, val_loader, test_loader, loss, optimizer, scheduler, device
    )

    loaders = (train_loader, val_loader, test_loader)

    return model, loaders


def predict_gnn_model(model, loaders):

    train_loader, val_loader, test_loader = loaders

    predictions_train = predict_network(model, train_loader)
    predictions_train["Set"] = DataSets.Training.value
    predictions_val = predict_network(model, val_loader)
    predictions_val["Set"] = DataSets.Validation.value
    prediction_test = predict_network(model, test_loader)
    prediction_test["Set"] = DataSets.Test.value

    predictions = pd.concat([predictions_train, predictions_val, prediction_test])
    predictions["model"] = model.name

    return predictions
