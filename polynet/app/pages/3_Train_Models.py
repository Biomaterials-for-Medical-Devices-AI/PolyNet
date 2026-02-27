import json
from shutil import rmtree

import pandas as pd
import streamlit as st

from polynet.app.components.experiments import experiment_selector
from polynet.app.components.forms.train_models import (
    split_data_form,
    train_GNN_models_form,
    train_TML_models,
)
from polynet.app.components.plots import display_model_results
from polynet.app.options.file_paths import (
    data_options_path,
    general_options_path,
    gnn_raw_data_file,
    gnn_raw_data_path,
    ml_results_file_path,
    ml_results_parent_directory,
    model_dir,
    model_metrics_file_path,
    plots_directory,
    polynet_experiment_path,
    representation_file_path,
    representation_options_path,
    train_gnn_model_options_path,
    train_tml_model_options_path,
)
from polynet.app.options.state_keys import (
    GeneralConfigStateKeys,
    TrainGNNStateKeys,
    TrainTMLStateKeys,
)
from polynet.app.services.configurations import load_options, save_options
from polynet.app.services.experiments import get_experiments
from polynet.app.services.model_training import load_dataframes, save_gnn_model, save_tml_model
from polynet.app.utils import save_data
from polynet.config.column_names import get_iterator_name, get_true_label_column_name
from polynet.config.constants import ResultColumn
from polynet.config.schemas.data import DataConfig
from polynet.config.schemas.general import GeneralConfig
from polynet.config.schemas.representation import RepresentationConfig
from polynet.config.schemas.training import TrainGNNConfig, TrainTMLConfig
from polynet.factories.dataloader import get_data_split_indices
from polynet.featurizer.polymer_graph import CustomPolymerGraph
from polynet.inference.gnn import get_predictions_df_gnn
from polynet.inference.tml import get_predictions_df_tml
from polynet.training.evaluate import plot_learning_curves, plot_results
from polynet.training.gnn import train_gnn_ensemble
from polynet.training.metrics import get_metrics
from polynet.training.tml import train_tml_ensemble


def train_models(
    experiment_name: str,
    tml_models: dict,
    gnn_conv_params: dict,
    representation_options: RepresentationConfig,
    data_options: DataConfig,
):

    # paths for options and experiments
    experiment_path = polynet_experiment_path(experiment_name=experiment_name)
    tml_training_opts_path = train_tml_model_options_path(experiment_path=experiment_path)
    gnn_training_opts_path = train_gnn_model_options_path(experiment_path=experiment_path)
    ml_results_dir = ml_results_parent_directory(experiment_path=experiment_path)

    # delete old options if it exists
    if tml_training_opts_path.exists():
        tml_training_opts_path.unlink()
    if gnn_training_opts_path.exists():
        gnn_training_opts_path.unlink()
    if ml_results_dir.exists():
        rmtree(ml_results_dir)
    gen_options_path = general_options_path(experiment_path=experiment_path)
    if gen_options_path.exists():
        gen_options_path.unlink()

    # Get general experiment options
    general_experiment_options = GeneralConfig(
        split_type=st.session_state[GeneralConfigStateKeys.SplitType],
        split_method=st.session_state[GeneralConfigStateKeys.SplitMethod],
        train_set_balance=st.session_state.get(GeneralConfigStateKeys.DesiredProportion, None),
        test_ratio=st.session_state[GeneralConfigStateKeys.TestSize],
        val_ratio=st.session_state[GeneralConfigStateKeys.ValidationSize],
        random_seed=st.session_state[GeneralConfigStateKeys.RandomSeed],
        n_bootstrap_iterations=st.session_state.get(
            GeneralConfigStateKeys.BootstrapIterations, None
        ),
    )
    # save new options
    save_options(path=gen_options_path, options=general_experiment_options)

    # read the data
    data = pd.read_csv(
        gnn_raw_data_file(file_name=data_options.data_name, experiment_path=experiment_path),
        index_col=0,
    )
    # generate indices to split data
    train_val_test_idxs = get_data_split_indices(
        data=data,
        split_type=general_experiment_options.split_type,
        n_bootstrap_iterations=general_experiment_options.n_bootstrap_iterations,
        val_ratio=general_experiment_options.val_ratio,
        test_ratio=general_experiment_options.test_ratio,
        target_variable_col=data_options.target_variable_col,
        split_method=general_experiment_options.split_method,
        train_set_balance=general_experiment_options.train_set_balance,
        random_seed=general_experiment_options.random_seed,
    )

    # Create directory to save plots
    plots_dir = plots_directory(experiment_path=experiment_path)
    plots_dir.mkdir(parents=True)
    # directory for models
    models_dir = model_dir(experiment_path=experiment_path)
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = model_metrics_file_path(experiment_path=experiment_path)

    # check if user selected tml models to train
    if tml_models:
        # get and save options
        train_tml_options = TrainTMLConfig(
            train_tml=st.session_state[TrainTMLStateKeys.TrainTML],
            selected_models=list(tml_models.keys()),
            model_params=tml_models,
            transform_features=st.session_state[TrainTMLStateKeys.TrasformFeatures],
        )
        save_options(path=tml_training_opts_path, options=train_tml_options)

        # load dataframes
        dataframes = load_dataframes(
            representation_options=representation_options,
            experiment_path=experiment_path,
            target_variable_col=data_options.target_variable_col,
        )

        # train the models
        tml_models, dataframes, scalers = train_tml_ensemble(
            tml_models=tml_models,
            problem_type=data_options.problem_type,
            transform_type=train_tml_options.transform_features,
            dataframes=dataframes,
            random_seed=general_experiment_options.random_seed,
            train_val_test_idxs=train_val_test_idxs,
        )

        for model_name, model in tml_models.items():
            save_path = models_dir / f"{model_name}.joblib"
            save_tml_model(model, save_path)
        if scalers:
            for scaler_name, scaler in scalers.items():
                save_path = models_dir / f"{scaler_name}.pkl"
                save_tml_model(scaler, save_path)

        # generate predictions df
        tml_predictions_df = get_predictions_df_tml(
            models=tml_models,
            training_data=dataframes,
            split_type=general_experiment_options.split_type,
            target_variable_col=data_options.target_variable_col,
            problem_type=data_options.problem_type,
            target_variable_name=data_options.target_variable_name,
        )

        metrics_tml = get_metrics(
            predictions=tml_predictions_df,
            split_type=general_experiment_options.split_type,
            target_variable_name=data_options.target_variable_name,
            trained_models=tml_models.keys(),
            problem_type=data_options.problem_type,
        )

        plot_results(
            predictions=tml_predictions_df,
            split_type=general_experiment_options.split_type,
            target_variable_name=data_options.target_variable_name,
            ml_algorithms=tml_models.keys(),
            problem_type=data_options.problem_type,
            save_path=plots_dir,
            class_names=data_options.class_names,
        )

    if gnn_conv_params:
        train_gnn_options = TrainGNNConfig(
            train_gnn=st.session_state[TrainGNNStateKeys.TrainGNN],
            gnn_convolutional_layers=gnn_conv_params,
            # HyperparameterOptimisation=st.session_state[TrainGNNStateKeys.HypTunning],
            share_gnn_parameters=st.session_state.get(TrainGNNStateKeys.SharedGNNParams, False),
        )
        save_options(path=gnn_training_opts_path, options=train_gnn_options)

        dataset = CustomPolymerGraph(
            filename=data_options.data_name,
            root=gnn_raw_data_path(experiment_path).parent,
            smiles_cols=data_options.smiles_cols,
            target_col=data_options.target_variable_col,
            id_col=data_options.id_col,
            weights_col=representation_options.weights_col,
            node_feats=representation_options.node_features,
            edge_feats=representation_options.edge_features,
        )

        gnn_models, loaders = train_gnn_ensemble(
            experiment_path=experiment_path,
            dataset=dataset,
            split_indexes=train_val_test_idxs,
            gnn_conv_params=train_gnn_options.gnn_convolutional_layers,
            problem_type=data_options.problem_type,
            num_classes=data_options.num_classes,
            random_seed=general_experiment_options.random_seed,
        )

        for model_name, model in gnn_models.items():
            save_path = models_dir / f"{model_name}.pt"
            save_gnn_model(model, save_path)

        plot_learning_curves(gnn_models, save_path=plots_dir)

        gnn_predictions_df = get_predictions_df_gnn(
            models=gnn_models,
            loaders=loaders,
            problem_type=data_options.problem_type,
            split_type=general_experiment_options.split_type,
            target_variable_name=data_options.target_variable_name,
        )

        # metrics_gnn = get_metrics(
        #     predictions=gnn_predictions_df,
        #     split_type=general_experiment_options.split_type,
        #     target_variable_name=data_options.target_variable_name,
        #     trained_models=gnn_models.keys(),
        #     problem_type=data_options.problem_type,
        # )

        # plot_results(
        #     predictions=gnn_predictions_df,
        #     split_type=general_experiment_options.split_type,
        #     target_variable_name=data_options.target_variable_name,
        #     ml_algorithms=gnn_models.keys(),
        #     problem_type=data_options.problem_type,
        #     save_path=plots_dir,
        #     class_names=data_options.class_names,
        # )

    if tml_models and gnn_conv_params:

        iterator = get_iterator_name(general_experiment_options.split_type)
        label_col_name = get_true_label_column_name(
            target_variable_name=data_options.target_variable_name
        )
        gnn_predictions_df = gnn_predictions_df.drop(columns=[label_col_name])

        predictions = pd.merge(
            left=tml_predictions_df,
            right=gnn_predictions_df,
            on=[ResultColumn.INDEX, ResultColumn.SET, iterator],
        )

        metrics = {}

        for i in range(general_experiment_options.n_bootstrap_iterations):
            iteration = str(i + 1)
            metrics[iteration] = {
                # **metrics_gnn[iteration],
                **metrics_tml[iteration]
            }

    elif gnn_conv_params:
        predictions = gnn_predictions_df.copy()
        # metrics = metrics_gnn
    elif tml_models:
        predictions = tml_predictions_df.copy()
        metrics = metrics_tml

    save_data(data=predictions, data_path=ml_results_file_path(experiment_path=experiment_path))

    # with open(metrics_path, "w") as f:
    #     json.dump(metrics, f, indent=4)

    display_model_results(experiment_path=experiment_path, expanded=True)


st.header("Train Models")

st.markdown(
    """
    In this section, you can train machine learning models on your polymer dataset. The models will be trained using the representations you built in the previous section. You can choose from various machine learning algorithms, including Graph Neural Networks (GNNs), Random Forests, and Support Vector Machines (SVMs).
    If you have not yet built a representation of your polymers, please go back to the previous section and build a representation first. The models will be trained on the representations you built, so it is essential to have a representation before training models.
    """
)


choices = get_experiments()
experiment_name = experiment_selector(choices)


if experiment_name:

    experiment_path = polynet_experiment_path(experiment_name=experiment_name)

    path_to_data_opts = data_options_path(experiment_path=experiment_path)

    data_opts = load_options(path=path_to_data_opts, options_class=DataConfig)

    path_to_representation_opts = representation_options_path(experiment_path=experiment_path)
    representation_opts = load_options(
        path=path_to_representation_opts, options_class=RepresentationConfig
    )

    train_gnn_options = train_gnn_model_options_path(experiment_path=experiment_path)

    if train_gnn_options.exists():
        st.warning(
            "GNN model options already exist for this experiment. "
            "You can modify the settings below, but be aware that this will overwrite the existing results."
        )
        display_model_results(experiment_path=experiment_path, expanded=False)

    st.markdown("## Train Machine Learning Models (TMLs)")

    if representation_file_path(experiment_path=experiment_path).exists():

        tml_models = train_TML_models(problem_type=data_opts.problem_type)

    else:
        st.error("No descriptors representation found, TML models cannot be trained.")
        tml_models = {}

    st.markdown("## Graph Neural Networks (GNNs)")

    if gnn_raw_data_path(experiment_path=experiment_path).exists():

        gnn_conv_params = train_GNN_models_form(
            representation_opts=representation_opts, problem_type=data_opts.problem_type
        )

    else:
        st.error(
            "No graph representation found. Please build a graph representation of your polymers first."
        )
        gnn_conv_params = {}

    st.markdown("## Data Splitting Options")
    split_data_form(problem_type=data_opts.problem_type)

    if gnn_conv_params or tml_models:
        disabled = False
    else:
        disabled = True

    if st.button("Run Training", disabled=disabled):
        st.write("Training models...")

        train_models(
            experiment_name=experiment_name,
            tml_models=tml_models,
            gnn_conv_params=gnn_conv_params,
            representation_options=representation_opts,
            data_options=data_opts,
        )

        st.success("Models trained successfully!")
