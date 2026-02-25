import json
from shutil import rmtree

import pandas as pd
from scipy.stats import mode
import streamlit as st

from polynet.app.components.experiments import experiment_selector
from polynet.app.components.plots import (
    display_mean_std_model_metrics,
    display_model_results,
    display_unseen_predictions,
)
from polynet.app.options.file_paths import (
    data_options_path,
    general_options_path,
    ml_predictions_file_path,
    ml_predictions_metrics_file_path,
    ml_predictions_parent_path,
    ml_results_file_path,
    model_dir,
    polynet_experiments_base_dir,
    representation_options_path,
    train_gnn_model_options_path,
    train_tml_model_options_path,
    unseen_gnn_raw_data_file,
    unseen_gnn_raw_data_path,
    unseen_predictions_experiment_parent_path,
    unseen_predictions_ml_results_path,
)
from polynet.config.schemas import DataConfig
from polynet.config.schemas import GeneralConfig
from polynet.config.schemas import RepresentationConfig
from polynet.config.schemas import TrainGNNConfig
from polynet.config.schemas import TrainTMLConfig

from polynet.app.options.state_keys import PredictPageStateKeys
from polynet.app.services.configurations import load_options
from polynet.app.services.experiments import get_experiments
from polynet.app.services.model_training import (
    load_models_from_experiment,
    load_scalers_from_experiment,
)
from polynet.app.services.predict_model import predict_unseen_gnn, predict_unseen_tml
from polynet.app.utils import create_directory
from polynet.featurizer.descriptors import build_vector_representation
from polynet.featurizer.polymer_graph import CustomPolymerGraph
from polynet.config.enums import ProblemType, TransformDescriptor
from polynet.config.constants import ResultColumn
from polynet.config.column_names import get_true_label_column_name

from polynet.plotting.data_analysis import show_continuous_distribution, show_label_distribution
from polynet.training.metrics import calculate_metrics
from polynet.utils.chem_utils import check_smiles_cols, determine_string_representation


# TODO: refactor this page to spread functions in other modules
# TODO: prepare results plots when given target variable for comparison
def predict(
    experiment_path,
    df: pd.DataFrame,
    tml_models: list[str],
    gnn_models: list[str],
    data_options: DataConfig,
    representation_options: RepresentationConfig,
):

    dataset_name = st.session_state[PredictPageStateKeys.PredictData].name

    predictions_parent_dir = unseen_predictions_experiment_parent_path(
        file_name=dataset_name, experiment_path=experiment_path
    )

    if predictions_parent_dir.exists():
        rmtree(predictions_parent_dir)
    create_directory(predictions_parent_dir)

    # save data and remove old results
    path_to_data = unseen_gnn_raw_data_path(experiment_path=experiment_path, file_name=dataset_name)
    create_directory(path_to_data)

    file_path = unseen_gnn_raw_data_file(experiment_path=experiment_path, file_name=dataset_name)

    if data_options.id_col in df.columns:
        df = df.set_index(data_options.id_col, drop=True)
    df.to_csv(file_path)

    df = df.reset_index(drop=False)

    id_cols = (
        [data_options.id_col]
        + data_options.smiles_cols
        + list(representation_options.weights_col.values())
        + [data_options.target_variable_col]
    )
    id_cols = [col for col in df.columns if col in id_cols]
    id_df = df[id_cols].copy()
    true_label_name = get_true_label_column_name(data_options.target_variable_name)

    id_df = id_df.rename(
        columns={
            data_options.id_col: ResultColumn.INDEX,
            data_options.target_variable_col: true_label_name,
        }
    )

    if tml_models:
        # calculate descriptors
        descriptor_dfs = build_vector_representation(
            data=df,
            molecular_descriptors=representation_options.molecular_descriptors,
            smiles_cols=data_options.smiles_cols,
            id_col=data_options.id_col,
            target_col=data_options.target_variable_col,
            merging_approach=representation_options.smiles_merge_approach,
            weights_col=representation_options.weights_col,
            rdkit_independent=representation_options.rdkit_independent,
            df_descriptors_independent=representation_options.df_descriptors_independent,
            mix_rdkit_df_descriptors=representation_options.mix_rdkit_df_descriptors,
        )

        for df_name, df in descriptor_dfs.items():
            if df is None:
                continue
            new_cols = [col for col in df.columns if col not in id_cols]
            descriptor_dfs[df_name] = df[new_cols]

        # load the models and scalers
        models = load_models_from_experiment(
            experiment_path=experiment_path, model_names=tml_models
        )

        if train_tml_options.transform_features != TransformDescriptor.NoTransformation:
            scalers = load_scalers_from_experiment(
                experiment_path=experiment_path, model_names=tml_models
            )
        else:
            scalers = {}

        # get predictions
        predictions_tml = predict_unseen_tml(
            models=models, scalers=scalers, dfs=descriptor_dfs, data_options=data_options
        )

        predictions_tml = pd.concat([id_df, predictions_tml], axis=1, ignore_index=False)

    if gnn_models:
        # create graph representation
        dataset = CustomPolymerGraph(
            filename=dataset_name,
            root=path_to_data.parent,
            smiles_cols=data_options.smiles_cols,
            target_col=data_options.target_variable_col,
            id_col=data_options.id_col,
            weights_col=representation_options.weights_col,
            node_feats=representation_options.node_features,
            edge_feats=representation_options.edge_features,
        )

        # load the the selected gnn models
        models = load_models_from_experiment(
            experiment_path=experiment_path, model_names=gnn_models
        )

        # get predictions
        predictions_gnn = predict_unseen_gnn(
            models=models, dataset=dataset, data_options=data_options
        )

        cols = predictions_gnn.columns

        analysed_arch = []
        gnn_cols = []
        for col in cols:
            # get the column names that contain predicitons of GNN models (can be same architecture different seed/bootstraps)
            if ResultColumn.PREDICTED in col:
                gnn_cols.append(col)
            # check which architecture was used
            gnn_arch = col.split(" ")[0]
            if gnn_arch in analysed_arch or col == ResultColumn.INDEX:
                continue
            # stores each unique GNN architecture used (not repeats)
            else:
                analysed_arch.append(gnn_arch)

        ensemble_cols = {}
        for arch in analysed_arch:
            # get the columns with predictions of the same architecture
            arch_cols = [col for col in cols if arch in col and ResultColumn.PREDICTED in col]
            # store the columns in a dictionary where the key is the architecture used
            ensemble_cols[arch] = arch_cols
        # add a key for ALL GNN models
        ensemble_cols["GNN"] = gnn_cols

        ensemble_preds = []
        if data_options.problem_type == ProblemType.Classification:
            for gnn_arch, gnn_cols in ensemble_cols.items():
                if len(gnn_cols) < 2:
                    continue
                # get the predicitons for a given architecture
                gnn_df = predictions_gnn[gnn_cols].copy()
                # get majority vote
                ensemble_prediction, _ = mode(gnn_df.values, axis=1, keepdims=False)
                # append the voting prediction to list of ensmble predictions
                ensemble_preds.append(
                    pd.Series(
                        ensemble_prediction,
                        index=gnn_df.index,
                        name=f"{gnn_arch} Ensemble {ResultColumn.PREDICTED}",
                    )
                )

        elif data_options.problem_type == ProblemType.Regression:
            for gnn_arch, gnn_cols in ensemble_cols.items():
                if len(gnn_cols) < 2:
                    continue
                # get the predictions for a given architecture
                gnn_df = predictions_gnn[gnn_cols].copy()
                # get ensemble predictions
                ensemble_prediction = gnn_df.mean(axis=1)
                ensemble_preds.append(
                    pd.Series(
                        ensemble_prediction,
                        index=gnn_df.index,
                        name=f"{gnn_arch} Ensemble {ResultColumn.PREDICTED}",
                    )
                )

        if ensemble_preds:
            # concat all ensemble predictions into a single DF
            ensemble_preds = pd.concat(ensemble_preds, axis=1)
            # concat ensemble DF to the original predictions DF
            predictions_gnn = pd.concat([predictions_gnn, ensemble_preds], axis=1)

        predictions_gnn = pd.merge(left=id_df, right=predictions_gnn, on=[ResultColumn.INDEX])

    if gnn_models and tml_models:
        predictions = pd.merge(left=predictions_tml, right=predictions_gnn, on=list(id_df.columns))

    elif gnn_models:
        predictions = predictions_gnn.copy()

    else:
        predictions = predictions_tml.copy()

    if st.session_state.get(PredictPageStateKeys.CompareTarget, False):

        label_col_name = get_true_label_column_name(
            target_variable_name=data_options.target_variable_name
        )
        metrics = {}

        for col in predictions.columns:

            if ResultColumn.PREDICTED in col and "Ensemble" not in col:
                split_name = col.rsplit(" ", 3)
                model, number = split_name[0], split_name[1]
                model_name = f"{model} {number}"
                probs_cols = [
                    col_name
                    for col_name in predictions.columns
                    if ResultColumn.SCORE in col_name and model_name in col_name
                ]
            elif ResultColumn.PREDICTED in col:
                split_name = col.rsplit(" ", 1)
                model = split_name[0] + " " + split_name[1]
                probs_cols = None
            else:
                continue

            if number not in metrics:
                metrics[number] = {}
            if model not in metrics[number]:
                metrics[number][model] = {}

            print(probs_cols)

            metrics[number][model][dataset_name.split(".")[0]] = calculate_metrics(
                y_true=predictions[label_col_name],
                y_pred=predictions[col],
                y_probs=predictions[probs_cols] if probs_cols else None,
                problem_type=data_options.problem_type,
            )

        results_path = unseen_predictions_ml_results_path(
            file_name=dataset_name, experiment_path=experiment_path
        )
        results_path.mkdir()
        metrics_path = ml_predictions_metrics_file_path(dataset_name, experiment_path)

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        display_mean_std_model_metrics(metrics)

    st.subheader("Predictions")
    st.dataframe(predictions)
    predictions_parent_path = ml_predictions_parent_path(
        file_name=dataset_name, experiment_path=experiment_path
    )
    predictions_parent_path.mkdir(exist_ok=True)
    predictions.to_csv(
        ml_predictions_file_path(file_name=dataset_name, experiment_path=experiment_path)
    )


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

    # load data options
    path_to_data_opts = data_options_path(experiment_path=experiment_path)
    data_options = load_options(path=path_to_data_opts, options_class=DataConfig)
    smiles_cols = data_options.smiles_cols

    # load representation options
    path_to_representation_opts = representation_options_path(experiment_path=experiment_path)
    representation_options = load_options(
        path=path_to_representation_opts, options_class=RepresentationConfig
    )
    weights_col = representation_options.weights_col

    # load general options
    path_to_general_opts = general_options_path(experiment_path=experiment_path)
    general_experiment_options = load_options(
        path=path_to_general_opts, options_class=GeneralConfig
    )

    # load tml options
    path_to_train_tml_options = train_tml_model_options_path(experiment_path=experiment_path)
    if path_to_train_tml_options.exists:
        train_tml_options = load_options(
            path=path_to_train_tml_options, options_class=TrainTMLConfig
        )

    # load train gnn options
    path_to_train_gnn_options = train_gnn_model_options_path(experiment_path=experiment_path)
    if path_to_train_gnn_options.exists():
        train_gnn_options = load_options(
            path=path_to_train_gnn_options, options_class=TrainGNNConfig
        )

    if (
        not (path_to_train_gnn_options.exists() or path_to_train_tml_options.exists)
        or not path_to_general_opts.exists()
    ):
        st.error(
            "No models have been trained yet. Please train a model first in the 'Train GNN' section."
        )
        st.stop()

    display_model_results(experiment_path=experiment_path, expanded=False)
    display_unseen_predictions(experiment_path=experiment_path)

    predictions_path = ml_results_file_path(experiment_path=experiment_path)
    # predictions = pd.read_csv(predictions_path, index_col=0)

    csv_file = st.file_uploader(
        "Upload a CSV file with SMILES strings for prediction",
        type="csv",
        key=PredictPageStateKeys.PredictData,
        help="Upload a CSV file containing SMILES strings for which you want to make predictions.",
    )

    if csv_file:

        path_to_data = unseen_predictions_experiment_parent_path(
            experiment_path=experiment_path, file_name=csv_file.name
        )

        if path_to_data.exists():
            st.warning(
                "It seems that you have already analysed this file. The previous prediction results will be overwritten."
            )

        df = pd.read_csv(csv_file)

        if st.checkbox("Preview data"):
            st.write("Preview of the uploaded data:")
            st.dataframe(df)

        # check if the provided df has the smiles cols and weight cols required
        for col in smiles_cols:
            if col not in df.columns:
                st.error(f"Column '{col}' not found in the uploaded data.")
                st.stop()
            elif weights_col:
                col = weights_col[col]
                if col not in df.columns:
                    st.error(f"Column '{col}' for weights not found in the uploaded data.")
                    st.stop()

        # check if smiles are valid
        invalid_smiles = check_smiles_cols(col_names=smiles_cols, df=df)
        if invalid_smiles:
            for col, smiles in invalid_smiles.values():
                st.error(f"Invalid SMILES found in column '{col}': {', '.join(smiles)}")
            st.stop()

        str_representation = determine_string_representation(df=df, smiles_cols=smiles_cols)
        st.write(f"The `{str_representation}` representation has been identified.")
        st.success(f"`{str_representation}` columns checked successfully.")

        if str_representation != data_options.string_representation:
            st.warning(
                f"We found `{str_representation}` used in the provided data, however `{data_options.string_representation}` was used to train the models. You can continue with the predictions, however, it is recommended to use the same representation as used for training."
            )

        if data_options.target_variable_col in df.columns:
            if st.checkbox(
                "Would you like to compare the predictions with the target variable?",
                key=PredictPageStateKeys.CompareTarget,
                value=True,
                help="If checked, the predictions will be compared with the target variable in the uploaded data.",
            ):
                if data_options.problem_type == ProblemType.Classification:

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

                elif data_options.problem_type == ProblemType.Regression:

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

        gnn_models_dir = model_dir(experiment_path=experiment_path)

        if path_to_train_gnn_options.exists():
            gnn_models = [
                model.name
                for model in gnn_models_dir.iterdir()
                if model.is_file() and model.suffix == ".pt"
            ]

            if st.toggle("Select all models", key=PredictPageStateKeys.SelectAllModels):
                default_models = gnn_models
            else:
                default_models = None

            selected_gnn_models = st.multiselect(
                "Select a GNN Model to predict with",
                options=sorted(gnn_models),
                key="model",
                default=default_models,
            )
        else:
            selected_gnn_models = []

        if path_to_train_tml_options.exists():
            tml_models = [
                model.name
                for model in gnn_models_dir.iterdir()
                if model.is_file() and model.suffix == ".joblib"
            ]
            if st.toggle("Select all models"):
                default_models = tml_models
            else:
                default_models = None

            selected_tml_models = st.multiselect(
                "Select a TML model to predict with",
                options=sorted(tml_models),
                key="tml_model",
                default=default_models,
            )
        else:
            selected_tml_models = []

        if selected_gnn_models or selected_tml_models:
            disable = False
        else:
            disable = True

        if st.button("Predict", disabled=disable):
            predict(
                experiment_path=experiment_path,
                df=df,
                gnn_models=sorted(selected_gnn_models),
                tml_models=sorted(selected_tml_models),
                data_options=data_options,
                representation_options=representation_options,
            )
