from shutil import rmtree

import pandas as pd
import streamlit as st

from polynet.app.components.experiments import experiment_selector
from polynet.app.components.forms.train_models import (
    split_data_form,
    train_GNN_models_form,
    train_TML_models,
)
from polynet.app.components.plots import display_plots, display_predictions
from polynet.app.options.data import DataOptions
from polynet.app.options.file_paths import (
    data_options_path,
    general_options_path,
    gnn_model_dir,
    gnn_plots_directory,
    gnn_raw_data_file,
    gnn_raw_data_path,
    ml_gnn_results_file_path,
    ml_results_parent_directory,
    polynet_experiments_base_dir,
    representation_file,
    representation_file_path,
    representation_options_path,
    train_gnn_model_options_path,
)
from polynet.app.options.general_experiment import GeneralConfigOptions
from polynet.app.options.representation import RepresentationOptions
from polynet.app.options.state_keys import (
    GeneralConfigStateKeys,
    TrainGNNStateKeys,
    TrainTMLStateKeys,
)
from polynet.app.options.train_GNN import TrainGNNOptions
from polynet.app.services.configurations import load_options, save_options
from polynet.app.services.experiments import get_experiments
from polynet.app.services.model_training import save_gnn_model, save_plot
from polynet.app.services.train_gnn import predict_gnn_model, train_network
from polynet.app.utils import merge_model_predictions, save_data
from polynet.options.enums import DataSets, ProblemTypes, Results
from polynet.utils.model_training import predict_network
from polynet.utils.plot_utils import plot_confusion_matrix


def train_models(
    experiment_name: str,
    gnn_conv_params: dict,
    representation_options: RepresentationOptions,
    data_options: DataOptions,
):
    """
    Placeholder function for training models.
    This function is a placeholder for future implementation.
    """

    train_gnn_options = TrainGNNOptions(
        GNNConvolutionalLayers=gnn_conv_params,
        GNNNumberOfLayers=st.session_state[TrainGNNStateKeys.GNNNumberOfLayers],
        GNNEmbeddingDimension=st.session_state[TrainGNNStateKeys.GNNEmbeddingDimension],
        GNNPoolingMethod=st.session_state[TrainGNNStateKeys.GNNPoolingMethod],
        GNNReadoutLayers=st.session_state[TrainGNNStateKeys.GNNReadoutLayers],
        GNNDropoutRate=st.session_state[TrainGNNStateKeys.GNNDropoutRate],
        GNNLearningRate=st.session_state[TrainGNNStateKeys.GNNLearningRate],
        GNNBatchSize=st.session_state[TrainGNNStateKeys.GNNBatchSize],
    )

    general_experiment_options = GeneralConfigOptions(
        split_method=st.session_state[GeneralConfigStateKeys.SplitMethod],
        train_set_balance=st.session_state.get(GeneralConfigStateKeys.DesiredProportion, None),
        test_ratio=st.session_state[GeneralConfigStateKeys.TestSize],
        val_ratio=st.session_state[GeneralConfigStateKeys.ValidationSize],
        random_seed=st.session_state[GeneralConfigStateKeys.RandomSeed],
    )

    experiment_path = polynet_experiments_base_dir() / experiment_name

    gnn_training_opts_path = train_gnn_model_options_path(experiment_path=experiment_path)
    gen_options_path = general_options_path(experiment_path=experiment_path)
    ml_results_dir = ml_results_parent_directory(experiment_path=experiment_path)

    if gnn_training_opts_path.exists():
        gnn_training_opts_path.unlink()
    if gen_options_path.exists():
        gen_options_path.unlink()
    if ml_results_dir.exists():
        rmtree(ml_results_dir)

    save_options(path=gnn_training_opts_path, options=train_gnn_options)
    save_options(path=gen_options_path, options=general_experiment_options)

    model, loaders = train_network(
        train_gnn_options=train_gnn_options,
        general_experiment_options=general_experiment_options,
        experiment_name=experiment_name,
        data_options=data_options,
        representation_options=representation_options,
    )

    predictions_all = []

    gnn_models_dir = gnn_model_dir(experiment_path=experiment_path)
    gnn_models_dir.mkdir(parents=True, exist_ok=True)

    gnn_plots_dir = gnn_plots_directory(experiment_path=experiment_path)
    gnn_plots_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model in model.items():

        save_path = gnn_models_dir / f"{model_name}.pt"
        save_gnn_model(model=model, path=save_path)

        predictions = predict_gnn_model(model=model, loaders=loaders)
        predictions_all.append(predictions)

        if data_options.problem_type == ProblemTypes.Classification:
            predictions_test = predictions.loc[predictions[Results.Set.value] == DataSets.Test]
            fig = plot_confusion_matrix(
                y_true=predictions_test.iloc[:, 2],
                y_pred=predictions_test.iloc[:, 3],
                display_labels=(
                    list(data_options.class_names.values()) if data_options.class_names else None
                ),
                title=f"{data_options.target_variable_name}\nConfusion Matrix for\n {model_name}",
            )
            save_plot_path = gnn_plots_dir / f"{model_name}_confusion_matrix.png"
            save_plot(fig=fig, path=save_plot_path)
        else:
            pass

    predictions_all = merge_model_predictions(dfs=predictions_all)

    save_data(
        data=predictions_all,
        data_path=ml_gnn_results_file_path(
            experiment_path=experiment_path, file_name="predictions.csv"
        ),
    )
    display_predictions(predictions_df=predictions_all)
    display_plots(plots_path=gnn_plots_dir)


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

    experiment_path = polynet_experiments_base_dir() / experiment_name

    path_to_data_opts = data_options_path(
        experiment_path=polynet_experiments_base_dir() / experiment_name
    )

    data_opts = load_options(path=path_to_data_opts, options_class=DataOptions)

    path_to_representation_opts = representation_options_path(
        experiment_path=polynet_experiments_base_dir() / experiment_name
    )
    representation_opts = load_options(
        path=path_to_representation_opts, options_class=RepresentationOptions
    )

    train_gnn_options = train_gnn_model_options_path(
        experiment_path=polynet_experiments_base_dir() / experiment_name
    )

    if train_gnn_options.exists():
        st.error(
            "GNN model options already exist for this experiment. "
            "You can modify the settings below, but be aware that this will overwrite the existing results."
        )

        with st.expander("View Previous GNN Model Results", expanded=True):

            st.write("### Previous GNN Training Results")

            predictions = pd.read_csv(
                ml_gnn_results_file_path(
                    experiment_path=experiment_path, file_name="predictions.csv"
                )
            )
            plots_path = gnn_plots_directory(experiment_path=experiment_path)
            display_predictions(predictions_df=predictions)
            display_plots(plots_path=plots_path)

    st.markdown("## Train Machine Learning Models (TMLs)")

    if representation_file_path(experiment_path=experiment_path).exists():

        train_TML_models()

    else:
        st.error("No representation found. Please build a representation of your polymers first.")

    st.markdown("## Graph Neural Networks (GNNs)")

    if gnn_raw_data_path(experiment_path=experiment_path).exists():

        gnn_conv_params = train_GNN_models_form()
    else:
        st.error(
            "No graph representation found. Please build a graph representation of your polymers first."
        )
        gnn_conv_params = {}

    if gnn_conv_params:
        split_data_form(problem_type=data_opts.problem_type)

    if st.button("Run Training"):
        st.write("Training models...")

        train_models(
            experiment_name=experiment_name,
            gnn_conv_params=gnn_conv_params,
            representation_options=representation_opts,
            data_options=data_opts,
        )

        st.success("Models trained successfully!")
