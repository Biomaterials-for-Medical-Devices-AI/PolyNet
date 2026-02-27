import pandas as pd
import streamlit as st

from polynet.app.components.forms.create_experiment import select_data_form
from polynet.app.options.file_paths import data_file_path, polynet_experiments_base_dir
from polynet.app.options.state_keys import CreateExperimentStateKeys
from polynet.app.services.experiments import create_experiment
from polynet.config.schemas.data import DataConfig


def save_experiment(df):

    dataset_name = st.session_state[CreateExperimentStateKeys.DatasetName].name

    path_to_data = data_file_path(
        file_name=dataset_name,
        experiment_path=polynet_experiments_base_dir()
        / st.session_state[CreateExperimentStateKeys.ExperimentName],
    )

    data_options = DataConfig(
        data_name=dataset_name,
        data_path=str(path_to_data),
        smiles_cols=st.session_state[CreateExperimentStateKeys.SmilesCols],
        canonicalise_smiles=st.session_state[CreateExperimentStateKeys.CanonicaliseSMILES],
        target_variable_col=st.session_state[CreateExperimentStateKeys.TargetVariableCol],
        problem_type=st.session_state[CreateExperimentStateKeys.ProblemType],
        string_representation=st.session_state[CreateExperimentStateKeys.StringRepresentation],
        id_col=st.session_state[CreateExperimentStateKeys.IDCol],
        num_classes=st.session_state[CreateExperimentStateKeys.NumClasses],
        target_variable_name=st.session_state[CreateExperimentStateKeys.TargetVariableName],
        class_names=st.session_state.get(
            CreateExperimentStateKeys.ClassNames, None
        ),  # Optional, can be None
        target_variable_units=st.session_state[CreateExperimentStateKeys.TargetVariableUnits],
    )

    create_experiment(
        experiment_path=polynet_experiments_base_dir()
        / st.session_state[CreateExperimentStateKeys.ExperimentName],
        data_options=data_options,
    )

    df.to_csv(path_to_data, index=False)


st.write(
    """
    # Create Experiment

    In this section, you can create a new experiment by uploading a CSV file containing SMILES strings and the target variable you want to model. You can also select the columns that contain the SMILES strings and the target variable, as well as any additional features you want to include to your molecules.


    """
)

df = select_data_form()


if st.button("Save Experiment", disabled=not isinstance(df, pd.DataFrame)):
    save_experiment(df)
    st.success("Experiment saved successfully!")
    st.balloons()
