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
from polynet.app.options.data import DataOptions
from polynet.app.options.file_paths import (
    data_options_path,
    general_options_path,
    gnn_model_dir,
    gnn_model_metrics_file_path,
    gnn_plots_directory,
    gnn_raw_data_path,
    ml_gnn_results_file_path,
    ml_results_parent_directory,
    polynet_experiments_base_dir,
    representation_file_path,
    representation_options_path,
    train_gnn_model_options_path,
)
from polynet.app.options.general_experiment import GeneralConfigOptions
from polynet.app.options.representation import RepresentationOptions
from polynet.app.options.state_keys import GeneralConfigStateKeys, TrainGNNStateKeys
from polynet.app.options.train_GNN import TrainGNNOptions
from polynet.app.services.configurations import load_options, save_options
from polynet.app.services.experiments import get_experiments
from polynet.app.services.model_training import calculate_metrics, save_gnn_model, save_plot
from polynet.app.services.train_gnn import predict_gnn_model, train_network
from polynet.app.utils import (
    get_iterator_name,
    get_predicted_label_column_name,
    get_score_column_name,
    get_true_label_column_name,
    merge_model_predictions,
    save_data,
    ensemble_predictions,
)
from polynet.options.enums import DataSets, ProblemTypes, Results
from polynet.utils.plot_utils import plot_auroc, plot_confusion_matrix, plot_parity


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
        ApplyMonomerWeighting=st.session_state[TrainGNNStateKeys.GNNMonomerWeighting],
    )

    general_experiment_options = GeneralConfigOptions(
        split_type=st.session_state[GeneralConfigStateKeys.SplitType],
        split_method=st.session_state[GeneralConfigStateKeys.SplitMethod],
        train_set_balance=st.session_state.get(GeneralConfigStateKeys.DesiredProportion, None),
        n_bootstrap_iterations=st.session_state.get(
            GeneralConfigStateKeys.BootstrapIterations, None
        ),
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

    model = train_network(
        train_gnn_options=train_gnn_options,
        general_experiment_options=general_experiment_options,
        experiment_name=experiment_name,
        data_options=data_options,
        representation_options=representation_options,
    )

    gnn_models_dir = gnn_model_dir(experiment_path=experiment_path)
    gnn_models_dir.mkdir(parents=True, exist_ok=True)

    gnn_plots_dir = gnn_plots_directory(experiment_path=experiment_path)
    gnn_plots_dir.mkdir(parents=True, exist_ok=True)

    metrics = {}

    predictions_iterations = pd.DataFrame()

    iterator = get_iterator_name(general_experiment_options.split_type)

    for iteration, model_params in model.items():

        loader = model_params[Results.Loaders.value]
        models = model_params[Results.Model.value]

        predictions_models = []
        ensemple_models = []
        metrics[iteration] = {}

        for model_name, model in models.items():

            save_path = gnn_models_dir / f"{model_name}_{iteration}.pt"
            save_gnn_model(model=model, path=save_path)

            label_col_name = get_true_label_column_name(
                target_variable_name=data_options.target_variable_name
            )
            predicted_col_name = get_predicted_label_column_name(
                target_variable_name=data_options.target_variable_name, model_name=model._name
            )

            predictions = predict_gnn_model(
                model=model, loaders=loader, target_variable_name=data_options.target_variable_name
            )

            predictions_train = predictions.loc[predictions[Results.Set.value] == DataSets.Training]
            predictions_val = predictions.loc[predictions[Results.Set.value] == DataSets.Validation]
            predictions_test = predictions.loc[predictions[Results.Set.value] == DataSets.Test]

            metrics[iteration][model_name] = {}

            if data_options.problem_type == ProblemTypes.Classification:

                fig = plot_confusion_matrix(
                    y_true=predictions_test[label_col_name],
                    y_pred=predictions_test[predicted_col_name],
                    display_labels=(
                        list(data_options.class_names.values())
                        if data_options.class_names
                        else None
                    ),
                    title=f"{data_options.target_variable_name}\nConfusion Matrix for\n {model_name} - {iteration}",
                )
                save_plot_path = gnn_plots_dir / f"{model_name}_{iteration}_confusion_matrix.png"
                save_plot(fig=fig, path=save_plot_path)

                for class_num in range(int(data_options.num_classes)):

                    if class_num == 0 and int(data_options.num_classes) == 2:
                        continue  # Skip class 0 if binary classification

                    probs_col_name = get_score_column_name(
                        target_variable_name=data_options.target_variable_name,
                        model_name=model._name,
                        class_num=class_num,
                    )

                    fig = plot_auroc(
                        y_true=predictions_test[label_col_name],
                        y_scores=predictions_test[probs_col_name],
                        title=f"{data_options.target_variable_name}\nROC Curve for\n {model_name} Class {class_num} - {iteration}",
                    )

                    save_plot_path = (
                        gnn_plots_dir / f"{model_name}_{iteration}_class_{class_num}_roc_curve.png"
                    )
                    save_plot(fig=fig, path=save_plot_path)

            else:
                fig = plot_parity(
                    y_true=predictions_test[label_col_name],
                    y_pred=predictions_test[predicted_col_name],
                    title=f"{data_options.target_variable_name}\nParity Plot for\n {model_name} - {iteration}",
                )

                save_plot_path = gnn_plots_dir / f"{model_name}_{iteration}_parity_plot.png"
                save_plot(fig=fig, path=save_plot_path)

            metrics[iteration][model_name][DataSets.Training.value] = calculate_metrics(
                y_true=predictions_train[label_col_name],
                y_pred=predictions_train[predicted_col_name],
                y_probs=(
                    predictions_train[probs_col_name]
                    if data_options.problem_type == ProblemTypes.Classification
                    else None
                ),
                problem_type=data_options.problem_type,
            )
            metrics[iteration][model_name][DataSets.Validation.value] = calculate_metrics(
                y_true=predictions_val[label_col_name],
                y_pred=predictions_val[predicted_col_name],
                y_probs=(
                    predictions_val[probs_col_name]
                    if data_options.problem_type == ProblemTypes.Classification
                    else None
                ),
                problem_type=data_options.problem_type,
            )
            metrics[iteration][model_name][DataSets.Test.value] = calculate_metrics(
                y_true=predictions_test[label_col_name],
                y_pred=predictions_test[predicted_col_name],
                y_probs=(
                    predictions_test[probs_col_name]
                    if data_options.problem_type == ProblemTypes.Classification
                    else None
                ),
                problem_type=data_options.problem_type,
            )

            predictions_models.append(predictions)
            ensemple_models.append(predictions[[Results.Index.value, predicted_col_name]])

        predictions_models = merge_model_predictions(dfs=predictions_models)
        ensemble_preds = ensemble_predictions(
            pred_dfs=ensemple_models, problem_type=data_options.problem_type
        )
        predictions_models = predictions_models.merge(
            ensemble_preds, on=Results.Index.value, how="left"
        )
        predictions_models[iterator] = iteration

        predictions_iterations = pd.concat(
            [predictions_iterations, predictions_models], axis=0, ignore_index=True
        )

    save_data(
        data=predictions_iterations,
        data_path=ml_gnn_results_file_path(
            experiment_path=experiment_path, file_name="predictions.csv"
        ),
    )

    metrics_path = gnn_model_metrics_file_path(experiment_path=experiment_path)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

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
        display_model_results(experiment_path=experiment_path, expanded=False)

    # st.markdown("## Train Machine Learning Models (TMLs)")

    # if representation_file_path(experiment_path=experiment_path).exists():

    #     train_TML_models()

    # else:
    #     st.error("No representation found. Please build a representation of your polymers first.")

    st.markdown("## Graph Neural Networks (GNNs)")

    if gnn_raw_data_path(experiment_path=experiment_path).exists():

        gnn_conv_params = train_GNN_models_form(representation_opts=representation_opts)
    else:
        st.error(
            "No graph representation found. Please build a graph representation of your polymers first."
        )
        gnn_conv_params = {}

    if gnn_conv_params:

        st.markdown("### Data Splitting Options")
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
