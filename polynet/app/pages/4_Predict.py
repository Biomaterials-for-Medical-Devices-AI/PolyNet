from shutil import rmtree

import pandas as pd
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
    ml_results_file_path,
    polynet_experiments_base_dir,
    representation_options_path,
    train_gnn_model_options_path,
    train_tml_model_options_path,
    unseen_predictions_experiment_parent_path,
)
from polynet.app.options.state_keys import PredictPageStateKeys
from polynet.app.services.configurations import load_options
from polynet.app.services.experiments import get_experiments
from polynet.config.enums import ProblemType, TransformDescriptor
from polynet.config.schemas import (
    DataConfig,
    GeneralConfig,
    RepresentationConfig,
    TrainGNNConfig,
    TrainTMLConfig,
)
from polynet.pipeline import predict_external
from polynet.plotting.data_analysis import show_continuous_distribution, show_label_distribution
from polynet.utils.chem_utils import check_smiles_cols, determine_string_representation


st.header("Predict on New Data")

st.markdown(
    """
    Use trained models from an existing experiment to predict the target property for a new,
    unseen dataset. Upload a CSV file with the same column structure used during training
    (SMILES columns and, optionally, weight fraction columns). If the target property column
    is also present, per-model metrics will be computed automatically.
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
    if path_to_train_tml_options.exists():
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
        not (path_to_train_gnn_options.exists() or path_to_train_tml_options.exists())
        or not path_to_general_opts.exists()
    ):
        st.error(
            "No models have been trained yet. Please train a model first in the 'Train Models' section."
        )
        st.stop()

    display_model_results(experiment_path=experiment_path, expanded=False)
    display_unseen_predictions(experiment_path=experiment_path)

    csv_file = st.file_uploader(
        "Upload a CSV file for prediction",
        type="csv",
        key=PredictPageStateKeys.PredictData,
        help="Must contain the same SMILES column(s) used during training. "
        "The target property column is optional — include it to compute metrics.",
    )

    if csv_file:

        out_dir = unseen_predictions_experiment_parent_path(
            experiment_path=experiment_path, file_name=csv_file.name
        )

        if out_dir.exists():
            st.warning(
                "This file has been analysed before. The previous results will be overwritten."
            )

        df = pd.read_csv(csv_file)

        if st.checkbox("Preview data"):
            st.write("Preview of the uploaded data:")
            st.dataframe(df)

        # Validate SMILES and weight columns
        for col in smiles_cols:
            if col not in df.columns:
                st.error(f"Column '{col}' not found in the uploaded data.")
                st.stop()
            elif weights_col:
                weight_col_name = weights_col.get(col)
                if weight_col_name is not None and weight_col_name not in df.columns:
                    st.error(
                        f"Column '{weight_col_name}' for weights not found in the uploaded data."
                    )
                    st.stop()

        invalid_smiles = check_smiles_cols(col_names=smiles_cols, df=df)
        if invalid_smiles:
            for col, smiles in invalid_smiles.items():
                st.error(f"Invalid SMILES found in column '{col}': {', '.join(smiles)}")
            st.stop()

        str_representation = determine_string_representation(df=df, smiles_cols=smiles_cols)
        st.write(f"The `{str_representation}` representation has been identified.")
        st.success(f"`{str_representation}` columns checked successfully.")

        if str_representation != data_options.string_representation:
            st.warning(
                f"Found `{str_representation}` in the uploaded data, but `{data_options.string_representation}` "
                "was used during training. Predictions may be less reliable."
            )

        if data_options.target_variable_col in df.columns:
            if st.checkbox(
                "Compare predictions with the target variable?",
                key=PredictPageStateKeys.CompareTarget,
                value=True,
                help="When checked, per-model metrics are computed against the true labels.",
            ):
                if data_options.problem_type == ProblemType.Classification:
                    if pd.api.types.is_numeric_dtype(df[data_options.target_variable_col]):
                        if df[data_options.target_variable_col].nunique() >= 20:
                            st.error(
                                "The target column has too many unique values for a classification "
                                "task. Uncheck comparison or verify the problem type."
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
                            "The target column is not numeric. Uncheck comparison or verify "
                            "the problem type."
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

        if st.button("Predict"):
            if out_dir.exists():
                rmtree(out_dir)

            with st.spinner("Running predictions..."):
                predictions, metrics = predict_external(
                    data=df,
                    data_cfg=data_options,
                    repr_cfg=representation_options,
                    experiment_path=experiment_path,
                    out_dir=out_dir,
                    dataset_name=csv_file.name,
                )

            if metrics is not None and st.session_state.get(
                PredictPageStateKeys.CompareTarget, False
            ):
                display_mean_std_model_metrics(metrics)

            st.subheader("Predictions")
            st.dataframe(predictions)
            st.success(f"Predictions saved to `{out_dir / 'predictions.csv'}`")
