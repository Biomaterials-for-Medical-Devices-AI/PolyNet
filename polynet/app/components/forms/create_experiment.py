import pandas as pd
import streamlit as st

from polynet.app.options.file_paths import polynet_experiments_base_dir
from polynet.app.options.state_keys import CreateExperimentStateKeys
from polynet.options.enums import ProblemTypes, StringRepresentation
from polynet.plotting.data_analysis import show_continuous_distribution, show_label_distribution
from polynet.utils.chem_utils import (
    canonicalise_psmiles,
    canonicalise_smiles,
    check_smiles_cols,
    determine_string_representation,
)


def select_data_form():

    class_names = {}

    experiment_name = st.text_input(
        "Experiment name",
        placeholder="Enter a name for your experiment",
        key=CreateExperimentStateKeys.ExperimentName,
        help="This name will be used to identify your experiment in the app.",
    )

    exp_path = polynet_experiments_base_dir() / experiment_name

    if experiment_name and exp_path.exists():
        st.error(
            f"Experiment with name '{experiment_name}' already exists. Please choose a different name."
        )
    if not experiment_name:
        st.error("Please provide an experiment name.")

    csv_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        key=CreateExperimentStateKeys.DatasetName,
        help="Upload a CSV file containing SMILES strings and the target variable.",
    )

    if not csv_file:
        st.warning("Please upload a CSV file to proceed.")

    if csv_file:
        st.markdown("**Preview Data**")
        df = pd.read_csv(csv_file)
        if st.checkbox("Show data provided"):
            st.write(df)

        smiles_cols = st.multiselect(
            "Select SMILES columns",
            options=df.columns.tolist(),
            key=CreateExperimentStateKeys.SmilesCols,
        )

        if not smiles_cols:
            st.error("Please select at least one SMILES column.")
            st.stop()

        else:
            invalid_smiles = check_smiles_cols(col_names=smiles_cols, df=df)
            if invalid_smiles:
                for col, smiles in invalid_smiles.values():
                    st.error(f"Invalid SMILES found in column '{col}': {', '.join(smiles)}")

                st.stop()

            str_representation = determine_string_representation(df=df, smiles_cols=smiles_cols)
            st.write(f"The `{str_representation}` representation has been identified.")
            st.success(f"`{str_representation}` columns checked successfully.")
            st.session_state[CreateExperimentStateKeys.StringRepresentation] = str_representation

        if st.checkbox(
            f"Canonicalise `{str_representation}`",
            key=CreateExperimentStateKeys.CanonicaliseSMILES,
            help="Select this option to canonicalise the SMILES strings in the selected columns.",
            value=True,
        ):
            for col in smiles_cols:

                if str_representation == StringRepresentation.Smiles:
                    df[col] = df[col].apply(canonicalise_smiles)
                elif str_representation == StringRepresentation.PSmiles:
                    df[col] = df[col].apply(canonicalise_psmiles)
            st.success(f"`{str_representation}` columns canonicalized successfully.")

        st.selectbox(
            "Select column with the ID of each molecule",
            options=df.columns.tolist(),
            index=None,
            key=CreateExperimentStateKeys.IDCol,
        )

        target_col = st.selectbox(
            "Select target column",
            options=df.columns.tolist(),
            index=None,
            key=CreateExperimentStateKeys.TargetVariableCol,
            help="This column should contain the target variable you want to model.",
        )

        if target_col and pd.api.types.is_numeric_dtype(df[target_col]):
            st.write(f"Target variable '{target_col}' is numeric")
            unique_vals_target = df[target_col].nunique()
            if unique_vals_target < 20:
                st.warning("This looks like a classification problem is to be modeled.")
                suggested_problem_type = ProblemTypes.Classification
            else:
                st.warning("This looks like a regression problem is to be modeled.")
                suggested_problem_type = ProblemTypes.Regression
        elif target_col is None:
            st.error("Please select a target variable.")
            return False
        else:
            st.error(
                f"Target variable '{target_col}' is not numeric. Please select a numeric column."
            )
            suggested_problem_type = None
            return False

        if target_col:
            problems = [ProblemTypes.Classification, ProblemTypes.Regression]
            problem_type = st.selectbox(
                "Select the problem type",
                problems,
                index=(
                    problems.index(suggested_problem_type)
                    if suggested_problem_type in problems
                    else None
                ),
                key=CreateExperimentStateKeys.ProblemType,
            )

            if problem_type == ProblemTypes.Classification:
                n_classes = df[target_col].nunique()
            else:
                n_classes = 1

            st.text_input(
                "Number of classes",
                value=n_classes,
                key=CreateExperimentStateKeys.NumClasses,
                help="This will be used to create the plots and log information.",
                disabled=True,
            )

            with st.expander("Extra options"):
                target_name = st.text_input(
                    "Target variable name",
                    value=target_col,
                    key=CreateExperimentStateKeys.TargetVariableName,
                    help="This name will be used to create the plots and log information.",
                )

                st.text_input(
                    "Target variable units",
                    key=CreateExperimentStateKeys.TargetVariableUnits,
                    help="This will be used to create the plots and log information.",
                )

                if problem_type == ProblemTypes.Classification:

                    class_names = {}

                    for vals in sorted(df[target_col].unique()):
                        class_name = st.text_input(
                            f"Class {vals} name",
                            value=str(vals),
                            key=f"{CreateExperimentStateKeys.ClassNames}_{vals}",
                            help="This name will be used to create the plots and log information.",
                        )
                        class_names[str(vals)] = class_name
                    st.session_state[CreateExperimentStateKeys.ClassNames] = class_names

                    st.markdown("**Label Distribution**")

            if problem_type == ProblemTypes.Classification:

                fig = show_label_distribution(
                    data=df,
                    target_variable=target_col,
                    title=(
                        f"Label Distribution for {target_name}"
                        if target_name
                        else "Label Distribution"
                    ),
                    class_names=class_names,
                )
                st.pyplot(fig)

                return df

            elif problem_type == ProblemTypes.Regression:

                st.markdown("**Continuous Distribution**")
                st.pyplot(
                    show_continuous_distribution(
                        data=df,
                        target_variable=target_col,
                        title=(
                            f"Continuous Distribution for {target_name}"
                            if target_name
                            else "Continuous Distribution"
                        ),
                    )
                )
                return df
