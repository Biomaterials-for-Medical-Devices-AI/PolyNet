import pandas as pd
import streamlit as st

from polynet.app.options.state_keys import CreateExperimentStateKeys
from polynet.options.enums import ProblemTypes
from polynet.plotting.data_analysis import show_continuous_distribution, show_label_distribution
from polynet.utils.chem_utils import canonicalise_smiles, check_smiles
from polynet.app.components.forms.plot_customiser import get_plotting_options


def select_data_form():

    experiment_name = st.text_input(
        "Experiment name",
        placeholder="Enter a name for your experiment",
        key=CreateExperimentStateKeys.ExperimentName,
        help="This name will be used to identify your experiment in the app.",
    )

    csv_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        key=CreateExperimentStateKeys.DatasetName,
        help="Upload a CSV file containing SMILES strings and the target variable.",
    )

    smiles_cols = []

    if csv_file and experiment_name:
        st.markdown("**Preview Data**")
        df = pd.read_csv(csv_file)
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
            for col in smiles_cols:
                invalid_smiles_list = []
                for smiles in df[col]:
                    if not check_smiles(smiles):
                        invalid_smiles_list.append(str(smiles))
                if invalid_smiles_list:
                    st.error(
                        f"Invalid SMILES found in column '{col}': {', '.join(invalid_smiles_list)}"
                    )
                    st.stop()

            st.success("SMILES columns checked successfully.")

        if st.checkbox(
            "Canonicalise SMILES",
            key=CreateExperimentStateKeys.CanonicaliseSMILES,
            help="Select this option to canonicalise the SMILES strings in the selected columns.",
        ):
            for col in smiles_cols:
                df[col] = df[col].apply(canonicalise_smiles)
            st.success("SMILES columns canonicalized successfully.")

        # TODO: Add option to include graph-level features

        # graph_feats = {}

        # if st.checkbox("Include Graph-level features"):

        #     graph_level_features = st.multiselect(
        #         "Select graph-level features", options=df.columns.tolist()
        #     )

        #     st.write("Select the molecules which have the following features:")

        #     for feature in graph_level_features:
        #         graph_feats[feature] = st.multiselect(feature, options=smiles_cols)

        #     one_hot_encode_feats = st.multiselect(
        #         "Select features to one-hot encode", options=graph_level_features
        #     )
        #     ohe_pos_vals = {}

        #     for feature in one_hot_encode_feats:
        #         uni_vals = df[feature].unique().tolist()
        #         ohe_pos_vals[feature] = uni_vals

        #         st.write(f"Features to one-hot encode: {', '.join(one_hot_encode_feats)}")

        # else:
        #     st.write("No graph-level features selected")
        #     graph_level_features = None
        #     one_hot_encode_feats = None
        #     ohe_pos_vals = None

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
        else:
            st.error(
                f"Target variable '{target_col}' is not numeric. Please select a numeric column."
            )
            suggested_problem_type = None

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

            st.text_input(
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

            return True

            # with st.expander("Plotting Options", expanded=True):
            #     plot_opts = get_plotting_options()

            # if problem_type == ProblemTypes.Classification:
            #     st.header(f"Distribution of {target_name}")

            #     fig = show_label_distribution(
            #         data=df,
            #         target_variable=target_col,
            #         title=f"Distribution of {target_name}",
            #         return_fig=True,
            #     )
            #     st.plotly_chart(fig)

            # if problem_type == ProblemTypes.Regression:

            #     fig = show_continuous_distribution(
            #         data=df, target_variable=target_col, plot_opts=plot_opts
            #     )
            #     st.pyplot(fig)
