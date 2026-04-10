import pandas as pd
import streamlit as st

from polynet.app.components.forms.create_experiment import select_data_form
from polynet.app.options.file_paths import (
    data_file_path,
    general_options_path,
    polynet_experiments_base_dir,
)
from polynet.app.options.state_keys import CreateExperimentStateKeys, GeneralConfigStateKeys
from polynet.app.services.configurations import save_options
from polynet.app.services.experiments import create_experiment
from polynet.config.constants import ResultColumn
from polynet.config.schemas.data import DataConfig
from polynet.config.schemas.general import GeneralConfig


def save_experiment(df: pd.DataFrame):

    experiment_path = (
        polynet_experiments_base_dir() / st.session_state[CreateExperimentStateKeys.ExperimentName]
    )

    dataset_name = (
        st.session_state[CreateExperimentStateKeys.DatasetName].name
        if st.session_state[CreateExperimentStateKeys.DatasetName] is not None
        else st.session_state[CreateExperimentStateKeys.DatasetNameLoad] + ".csv"
    )

    path_to_data = data_file_path(file_name=dataset_name, experiment_path=experiment_path)

    id_col = (
        st.session_state[CreateExperimentStateKeys.IDCol]
        if st.session_state[CreateExperimentStateKeys.IDCol] is not None
        else ResultColumn.INDEX
    )

    data_options = DataConfig(
        data_name=dataset_name,
        data_path=str(path_to_data),
        smiles_cols=st.session_state[CreateExperimentStateKeys.SmilesCols],
        canonicalise_smiles=st.session_state[CreateExperimentStateKeys.CanonicaliseSMILES],
        target_variable_col=st.session_state[CreateExperimentStateKeys.TargetVariableCol],
        problem_type=st.session_state[CreateExperimentStateKeys.ProblemType],
        string_representation=st.session_state[CreateExperimentStateKeys.StringRepresentation],
        id_col=id_col,
        num_classes=st.session_state[CreateExperimentStateKeys.NumClasses],
        target_variable_name=st.session_state[CreateExperimentStateKeys.TargetVariableName],
        class_names=st.session_state.get(
            CreateExperimentStateKeys.ClassNames, None
        ),  # Optional, can be None
        target_variable_units=st.session_state[CreateExperimentStateKeys.TargetVariableUnits],
    )

    general_options = GeneralConfig(
        name=st.session_state[CreateExperimentStateKeys.ExperimentName],
        output_dir=str(experiment_path),
        random_seed=st.session_state[GeneralConfigStateKeys.RandomSeed],
    )

    create_experiment(experiment_path=experiment_path, data_options=data_options)

    save_options(
        path=general_options_path(experiment_path=experiment_path), options=general_options
    )

    df.to_csv(path_to_data, index=False)


st.write(
    """
    # Create Experiment

    In this section, you can create a new experiment by uploading a CSV file containing SMILES strings and the target variable you want to model. You can also select the columns that contain the SMILES strings and the target variable, as well as any additional features you want to include to your molecules.


    """
)

df = select_data_form()

st.number_input(
    "Random seed for reproducibility",
    min_value=0,
    max_value=100_000_000,
    value=1221,
    key=GeneralConfigStateKeys.RandomSeed,
    help="Ensures reproducible data splits and model training across runs.",
)

if st.button("Save Experiment", disabled=not isinstance(df, pd.DataFrame)):
    save_experiment(df)
    st.success("Experiment saved successfully!")
    st.balloons()
