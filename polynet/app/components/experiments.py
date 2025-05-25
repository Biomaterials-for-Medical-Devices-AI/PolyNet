import streamlit as st
from polynet.app.options.state_keys import ViewExperimentKeys


def experiment_selector(options: list) -> str:
    """Select

    Args:
        options (list): The list of experiment names to choose from.

    Returns:
        str: The name of the experiment on disk.
    """

    return st.selectbox(
        "Select an experiment",
        options=options,
        index=None,
        placeholder="Experiment name",
        key=ViewExperimentKeys.ExperimentName,
    )
