import json
from shutil import rmtree

import pandas as pd
from scipy.stats import mode
import streamlit as st

from polynet.app.components.experiments import experiment_selector
from polynet.app.components.forms.analyse_results import (
    confusion_matrix_plot_form,
    parity_plot_form,
)
from polynet.app.components.plots import display_mean_std_model_metrics, display_model_results
from polynet.app.options.data import DataOptions
from polynet.app.options.file_paths import (
    data_options_path,
    general_options_path,
    gnn_model_dir,
    gnn_predictions_file,
    gnn_predictions_file_path,
    gnn_predictions_metrics_file_path,
    gnn_predictions_plots_directory,
    gnn_raw_data_predict_file,
    gnn_raw_data_predict_path,
    ml_gnn_results_file_path,
    polynet_experiments_base_dir,
    representation_options_path,
    train_gnn_model_options_path,
)
from polynet.app.options.general_experiment import GeneralConfigOptions
from polynet.app.options.representation import RepresentationOptions
from polynet.app.options.state_keys import PredictPageStateKeys, ViewExperimentKeys
from polynet.app.options.train_GNN import TrainGNNOptions
from polynet.app.services.configurations import load_options
from polynet.app.services.experiments import get_experiments
from polynet.app.services.model_training import (
    calculate_metrics,
    load_models_from_experiment,
    predict_network,
)
from polynet.app.utils import create_directory
from polynet.featurizer.graph_representation.polymer import CustomPolymerGraph
from polynet.options.enums import ProblemTypes
from polynet.plotting.data_analysis import show_continuous_distribution, show_label_distribution
from polynet.utils.chem_utils import check_smiles_cols


def predict(
    experiment_path,
    df: pd.DataFrame,
    models: list[str],
    data_options: DataOptions,
    representation_options: RepresentationOptions,
):

    dataset_name = st.session_state[PredictPageStateKeys.PredictData].name

    path_to_data = gnn_raw_data_predict_path(
        experiment_path=experiment_path, file_name=dataset_name
    )

    if path_to_data.exists():
        rmtree(path_to_data)

    create_directory(path_to_data)

    file_path = gnn_raw_data_predict_file(experiment_path=experiment_path, file_name=dataset_name)

    predictions_dir = gnn_predictions_file_path(experiment_path=experiment_path)

    if predictions_dir.exists():
        rmtree(predictions_dir)
    create_directory(predictions_dir)

    if data_options.id_col in df.columns:
        df = df.set_index(data_options.id_col, drop=True)

    df.to_csv(file_path)

    dataset = CustomPolymerGraph(
        filename=dataset_name,
        root=path_to_data.parent,
        smiles_cols=data_options.smiles_cols,
        target_col=data_options.target_variable_col,
        id_col=data_options.id_col,
        weights_col=representation_options.weights_col,
        node_feats=representation_options.node_feats,
        edge_feats=representation_options.edge_feats,
    )

    models = load_models_from_experiment(experiment_path=experiment_path, model_names=models)

    predictions = predict_network(models=models, dataset=dataset)

    if len(models) > 1:

        if data_options.problem_type == ProblemTypes.Classification:

            # Calculate majority vote for classification tasks
            majority_vote, _ = mode(predictions.values, axis=1, keepdims=False)
            majority_vote = pd.Series(
                majority_vote, index=predictions.index, name="Ensemble Prediction"
            )
            predictions = predictions.merge(
                majority_vote, left_index=True, right_index=True, how="left"
            )

        elif data_options.problem_type == ProblemTypes.Regression:

            # Calculate mean prediction for regression tasks
            mean_prediction = predictions.mean(axis=1)
            mean_prediction = pd.Series(
                mean_prediction, index=predictions.index, name="Ensemble Prediction"
            )
            predictions = predictions.merge(
                mean_prediction, left_index=True, right_index=True, how="left"
            )

    if st.session_state.get(PredictPageStateKeys.CompareTarget, False):

        target_col = df[data_options.target_variable_col]

        predictions = predictions.merge(target_col, left_index=True, right_index=True, how="left")

        metrics = {}

        for col in predictions.columns[:-1]:

            if "_" in col:
                model, number = col.split("_")
            else:
                model = col
                number = "1"

            if number not in metrics:
                metrics[number] = {}
            if model not in metrics[number]:
                metrics[number][model] = {}

            metrics[number][model]["Test"] = calculate_metrics(
                y_true=predictions[data_options.target_variable_col],
                y_pred=predictions[col],
                y_probs=predictions[col],
                problem_type=data_options.problem_type,
            )

        metrics_path = gnn_predictions_metrics_file_path(experiment_path)

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        display_mean_std_model_metrics(metrics)

    st.write("Predictions:")
    st.dataframe(predictions)
    predictions.to_csv(gnn_predictions_file(experiment_path=experiment_path))


st.header("Predict with Trained GNN Models")

st.markdown(
    """
    In this section, you can make predictions using the trained GNN models. The predictions are based on the representation of the polymers that you have built in the previous sections. You can visualize the results using various plots, such as parity plots for regression tasks and confusion matrices for classification tasks.
    """
)


choices = get_experiments()
experiment_name = experiment_selector(choices)


if experiment_name:

    experiment_path = polynet_experiments_base_dir() / experiment_name

    path_to_data_opts = data_options_path(
        experiment_path=polynet_experiments_base_dir() / experiment_name
    )

    data_options = load_options(path=path_to_data_opts, options_class=DataOptions)
    smiles_cols = data_options.smiles_cols

    path_to_representation_opts = representation_options_path(
        experiment_path=polynet_experiments_base_dir() / experiment_name
    )
    representation_options = load_options(
        path=path_to_representation_opts, options_class=RepresentationOptions
    )
    weights_col = representation_options.weights_col

    path_to_train_gnn_options = train_gnn_model_options_path(
        experiment_path=polynet_experiments_base_dir() / experiment_name
    )

    train_gnn_options = load_options(path=path_to_train_gnn_options, options_class=TrainGNNOptions)

    path_to_general_opts = general_options_path(experiment_path=experiment_path)

    general_experiment_options = load_options(
        path=path_to_general_opts, options_class=GeneralConfigOptions
    )

    if not path_to_train_gnn_options.exists() or not path_to_general_opts.exists():
        st.error(
            "No models have been trained yet. Please train a model first in the 'Train GNN' section."
        )
        st.stop()

    display_model_results(experiment_path=experiment_path, expanded=False)

    predictions_path = predictions_path = ml_gnn_results_file_path(
        experiment_path=experiment_path, file_name="predictions.csv"
    )
    predictions = pd.read_csv(
        ml_gnn_results_file_path(experiment_path=experiment_path, file_name="predictions.csv"),
        index_col=0,
    )

    csv_file = st.file_uploader(
        "Upload a CSV file with SMILES strings for prediction",
        type="csv",
        key=PredictPageStateKeys.PredictData,
        help="Upload a CSV file containing SMILES strings for which you want to make predictions.",
    )

    if csv_file:

        path_to_data = gnn_raw_data_predict_path(
            experiment_path=experiment_path, file_name=csv_file.name
        )

        if path_to_data.exists():
            st.warning(
                "It seems that you have already uploaded this file. The previous prediction results will be overwritten."
            )

        df = pd.read_csv(csv_file)
        st.write("Preview of the uploaded data:")
        st.write(df)

        for smiles_col in smiles_cols:
            if smiles_col not in df.columns:
                st.error(f"Column '{smiles_col}' not found in the uploaded data.")
                st.stop()
            elif weights_col:
                col = weights_col[smiles_col]
                if col not in df.columns:
                    st.error(f"Column '{col}' for weights not found in the uploaded data.")
                    st.stop()

        invalid_smiles = check_smiles_cols(col_names=smiles_cols, df=df)
        if invalid_smiles:
            for col, smiles in invalid_smiles.values():
                st.error(f"Invalid SMILES found in column '{col}': {', '.join(smiles)}")
            st.stop()
        else:
            st.success("SMILES columns checked successfully.")

        if data_options.target_variable_col in df.columns:
            if st.checkbox(
                "Would you like to compare the predictions with the target variable?",
                key=PredictPageStateKeys.CompareTarget,
                value=True,
                help="If checked, the predictions will be compared with the target variable in the uploaded data.",
            ):
                if data_options.problem_type == ProblemTypes.Classification:

                    if pd.api.types.is_numeric_dtype(df[data_options.target_variable_col]):
                        unique_vals_target = df[data_options.target_variable_col].nunique()
                        if not unique_vals_target < 20:
                            st.error(
                                "The target variable seems to be a regression problem, but you have selected it for comparison. Please uncheck the comparison option or change the target variable."
                            )
                            st.stop()

                    st.markdown("**Label Distribution**")
                    st.pyplot(
                        show_label_distribution(
                            data=df,
                            target_variable=data_options.target_variable_col,
                            title=(
                                f"Label Distribution for {data_options.target_variable_name}"
                                if data_options.target_variable_name
                                else "Label Distribution"
                            ),
                            class_names=data_options.class_names,
                        )
                    )

                elif data_options.problem_type == ProblemTypes.Regression:

                    if not pd.api.types.is_numeric_dtype(df[data_options.target_variable_col]):
                        st.error(
                            "The target variable seems to be a classification problem, but you have selected it for comparison. Please uncheck the comparison option or change the target variable."
                        )
                        st.stop()

                    st.markdown("**Value Distribution**")
                    st.pyplot(
                        show_continuous_distribution(
                            data=df,
                            target_variable=data_options.target_variable_col,
                            title=(
                                f"Value Distribution for {data_options.target_variable_name}"
                                if data_options.target_variable_name
                                else "Value Distribution"
                            ),
                        )
                    )

        gnn_models_dir = gnn_model_dir(experiment_path=experiment_path)

        gnn_models = [
            model.name
            for model in gnn_models_dir.iterdir()
            if model.is_file() and model.suffix == ".pt"
        ]

        if st.toggle("Select all models", key=PredictPageStateKeys.SelectAllModels):
            default_models = gnn_models
        else:
            default_models = None

        models = st.multiselect(
            "Select a GNN Model to predict with",
            options=sorted(gnn_models),
            key="model",
            default=default_models,
        )

        if models:

            if st.button("Predict"):
                predict(
                    experiment_path=experiment_path,
                    df=df,
                    models=sorted(models),
                    data_options=data_options,
                    representation_options=representation_options,
                )
