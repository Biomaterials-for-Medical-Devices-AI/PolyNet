from pathlib import Path

import joblib
import pandas as pd
import torch
from torch import load, save

from polynet.app.options.file_paths import model_dir, representation_file
from polynet.config.column_names import get_fp_col_names
from polynet.config.enums import MolecularDescriptor
from polynet.config.schemas import RepresentationConfig
from polynet.featurizer.preprocess import sanitise_df


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
    return load(path, weights_only=False, map_location=torch.device("cpu"))


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
    representation_options: RepresentationConfig, experiment_path: Path, target_variable_col: str
) -> dict[MolecularDescriptor, pd.DataFrame]:

    dataframe_dict = {}

    for representation, features in representation_options.molecular_descriptors.items():

        file_path = representation_file(
            experiment_path=experiment_path, file_name=f"{representation}.csv"
        )
        df = pd.read_csv(file_path, index_col=0)

        if representation == MolecularDescriptor.PolyBERT and not features:
            features = get_fp_col_names()
        elif not features:
            continue
        df = sanitise_df(df=df, descriptors=features, target_variable_col=target_variable_col)
        dataframe_dict[representation] = df

    return dataframe_dict


def load_models_from_experiment(experiment_path: Path, model_names: list[str]) -> dict:
    """
    Loads trained GNN models from the specified experiment path.

    Args:
        experiment_path (str): Path to the experiment directory.

    Returns:
        dict: Dictionary containing model names as keys and loaded models as values.
    """

    gnn_models_path = model_dir(experiment_path)
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


def load_scalers_from_experiment(experiment_path: str, model_names: list[str]) -> dict:
    """
    Loads trained GNN models from the specified experiment path.

    Args:
        experiment_path (str): Path to the experiment directory.

    Returns:
        dict: Dictionary containing model names as keys and loaded models as values.
    """

    gnn_models_path = model_dir(experiment_path)
    scaler = {}

    for model_name in model_names:

        model_name = model_name.split(".")[0]
        scaler_name = model_name.split("-")[-1]

        scaler_file = gnn_models_path / f"{scaler_name}.pkl"

        scaler[scaler_name] = load_tml_model(scaler_file)

    return scaler
