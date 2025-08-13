import pandas as pd
import streamlit as st

from polynet.app.components.forms.create_experiment import select_data_form
from polynet.app.options.data import DataOptions
from polynet.app.options.file_paths import data_file_path, polynet_experiments_base_dir
from polynet.app.options.state_keys import CreateExperimentStateKeys
from polynet.app.services.experiments import create_experiment


def save_experiment(class_names):

    dataset_name = st.session_state[CreateExperimentStateKeys.DatasetName].name

    path_to_data = data_file_path(
        file_name=dataset_name,
        experiment_path=polynet_experiments_base_dir()
        / st.session_state[CreateExperimentStateKeys.ExperimentName],
    )

    data_options = DataOptions(
        data_name=dataset_name,
        data_path=str(path_to_data),
        smiles_cols=st.session_state[CreateExperimentStateKeys.SmilesCols],
        canonicalise_smiles=st.session_state[CreateExperimentStateKeys.CanonicaliseSMILES],
        id_col=st.session_state[CreateExperimentStateKeys.IDCol],
        target_variable_col=st.session_state[CreateExperimentStateKeys.TargetVariableCol],
        problem_type=st.session_state[CreateExperimentStateKeys.ProblemType],
        num_classes=st.session_state[CreateExperimentStateKeys.NumClasses],
        target_variable_name=st.session_state[CreateExperimentStateKeys.TargetVariableName],
        target_variable_units=st.session_state[CreateExperimentStateKeys.TargetVariableUnits],
        string_representation=st.session_state[CreateExperimentStateKeys.StringRepresentation],
        class_names=class_names,  # Optional, can be None
    )

    create_experiment(
        save_dir=polynet_experiments_base_dir()
        / st.session_state[CreateExperimentStateKeys.ExperimentName],
        data_options=data_options,
    )

    data = st.session_state[CreateExperimentStateKeys.DatasetName]
    df = pd.read_csv(data)
    df.to_csv(path_to_data, index=False)


st.write(
    """
    # Create Experiment

    In this section, you can create a new experiment by uploading a CSV file containing SMILES strings and the target variable you want to model. You can also select the columns that contain the SMILES strings and the target variable, as well as any additional features you want to include to your molecules.


    """
)

class_names = select_data_form()


if st.button("Save Experiment", disabled=not class_names):
    save_experiment(class_names)
    st.success("Experiment saved successfully!")
    st.balloons()
