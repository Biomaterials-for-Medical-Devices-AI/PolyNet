from pathlib import Path

import joblib
import pandas as pd
from torch import load, save

from polynet.app.options.file_paths import gnn_model_dir, representation_file
from polynet.app.options.representation import RepresentationOptions
from polynet.featurizer.preprocess import sanitise_df
from polynet.options.enums import MolecularDescriptors


def save_tml_model(model, path):
    joblib.dump(model, path)


def load_tml_model(path):
    return joblib.load(path)


def save_gnn_model(model, path):
    """
    Saves the GNN model to the specified path.

    Args:
        model: The GNN model to save.
        path (str): The path where the model will be saved.
    """

    save(model, path)


def load_gnn_model(path):
    """
    Loads the GNN model from the specified path.

    Args:
        path (str): The path from which to load the model.

    Returns:
        The loaded GNN model.
    """
    return load(path, weights_only=False)


def load_tml_model(path):

    return joblib.load(path)


def save_plot(fig, path, dpi=300):
    """
    Saves the plot to the specified path.

    Args:
        fig: The figure to save.
        path (str): The path where the plot will be saved.
    """
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    print(f"Plot saved to {path}")


def load_dataframes(
    representation_options: RepresentationOptions, experiment_path: Path, target_variable_col: str
):

    dataframe_dict = {}

    if representation_options.rdkit_independent:

        rdkit_file_path = representation_file(
            experiment_path=experiment_path, file_name=f"{MolecularDescriptors.RDKit}.csv"
        )

        rdkit_df = pd.read_csv(rdkit_file_path, index_col=0)

        rdkit_df = sanitise_df(
            df=rdkit_df,
            descriptors=representation_options.molecular_descriptors[MolecularDescriptors.RDKit],
            target_variable_col=target_variable_col,
        )

        dataframe_dict[MolecularDescriptors.RDKit] = rdkit_df

    if representation_options.df_descriptors_independent:
        df_file_path = representation_file(
            experiment_path=experiment_path, file_name=f"{MolecularDescriptors.DataFrame}.csv"
        )

        df_df = pd.read_csv(df_file_path, index_col=0)

        df_df = sanitise_df(
            df=df_df,
            descriptors=representation_options.molecular_descriptors[
                MolecularDescriptors.DataFrame
            ],
            target_variable_col=target_variable_col,
        )

        dataframe_dict[MolecularDescriptors.DataFrame] = df_df

    if representation_options.mix_rdkit_df_descriptors:

        mix_file_path = representation_file(
            experiment_path=experiment_path, file_name=f"{MolecularDescriptors.RDKit_DataFrame}.csv"
        )

        mix_df = pd.read_csv(mix_file_path, index_col=0)

        mix_df = sanitise_df(
            df=mix_df,
            descriptors=representation_options.molecular_descriptors[MolecularDescriptors.RDKit]
            + representation_options.molecular_descriptors[MolecularDescriptors.DataFrame],
            target_variable_col=target_variable_col,
        )

        dataframe_dict[MolecularDescriptors.RDKit_DataFrame] = mix_df

    if representation_options.polybert_fp:
        polybert_file_path = representation_file(
            experiment_path=experiment_path, file_name=f"{MolecularDescriptors.polyBERT}.csv"
        )
        polybert_df = pd.read_csv(polybert_file_path, index_col=0)
        polybert_df = sanitise_df(
            df=polybert_df,
            descriptors=[f"polyBERT_{i}" for i in range(600)],
            target_variable_col=target_variable_col,
        )
        dataframe_dict[MolecularDescriptors.polyBERT] = polybert_df

    return dataframe_dict


def load_models_from_experiment(experiment_path: str, model_names: list) -> dict:
    """
    Loads trained GNN models from the specified experiment path.

    Args:
        experiment_path (str): Path to the experiment directory.

    Returns:
        dict: Dictionary containing model names as keys and loaded models as values.
    """

    gnn_models_path = gnn_model_dir(experiment_path)
    models = {}

    for model_name in model_names:
        model_file = gnn_models_path / model_name
        model_name = model_file.stem
        termination = str(model_file).split(".")[-1]
        if termination == "pt":
            models[model_name] = load_gnn_model(model_file)
        else:
            models[model_name] = load_tml_model(model_file)

    return models


def load_scalers_from_experiment(experiment_path: str, model_names: list) -> dict:
    """
    Loads trained GNN models from the specified experiment path.

    Args:
        experiment_path (str): Path to the experiment directory.

    Returns:
        dict: Dictionary containing model names as keys and loaded models as values.
    """

    gnn_models_path = gnn_model_dir(experiment_path)
    scaler = {}

    for model_name in model_names:

        model_name = model_name.split(".")[0]
        scaler_name = model_name.split("-")[-1]

        scaler_file = gnn_models_path / f"{scaler_name}.pkl"

        scaler[scaler_name] = load_tml_model(scaler_file)

    return scaler
