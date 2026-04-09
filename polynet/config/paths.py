"""
polynet.config.paths
====================
Path helpers for PolyNet experiment directories and files.

All path construction for experiments, models, representations, plots, and
predictions is centralised here so both the CLI and GUI entry points resolve
paths consistently.
"""

from pathlib import Path


def polynet_experiments_base_dir() -> Path:
    """
    Return the path the base directory of all PolyNet experiments.

    This will be `/Users/<username>/PolyNetExperiments` on MacOS,
    `/home/<username>/PolyNetExperiments` on Linux, and
    `C:\\Users\\<username>\\PolyNetExperiments` on Windows.

    Returns:
        Path: The path to the PolyNet experiments base directory.
    """
    return Path.home() / "PolyNetExperiments"


def polynet_experiment_path(experiment_name: str) -> Path:
    """
    Returns the path to an experiment given the experiment name.

    Args:
        experiment_name: the name of the experiment

    Returns:
        Path: the path to the experiment.
    """
    return polynet_experiments_base_dir() / experiment_name


def data_file_path(file_name: str, experiment_path: Path) -> Path:
    """
    Return the path to the data file in the experiment directory.
    Args:
        file_name (str): The name of the data file.
        experiment_path (Path): The path to the experiment directory.
    Returns:
        Path: The path to the data file.
    """
    return experiment_path / file_name


def data_options_path(experiment_path: Path) -> Path:
    """Return the path to an experiment's data options JSON file.

    Args:
        experiment_path (Path): The path of the experiment.

    Returns:
        Path: The path to the experiment's data options.
    """
    return experiment_path / "data_options.json"


def representation_options_path(experiment_path: Path) -> Path:
    """Return the path to an experiment's representation options JSON file.

    Args:
        experiment_path (Path): The path of the experiment.

    Returns:
        Path: The path to the experiment's representation options.
    """
    return experiment_path / "representation_options.json"


def representation_parent_directory(experiment_path: Path) -> Path:
    """Return the path to the representation directory in the experiment directory.

    Args:
        experiment_path (Path): The path of the experiment.

    Returns:
        Path: The path to the representation directory.
    """
    return experiment_path / "representation"


def gnn_data_path(experiment_path: Path) -> Path:
    """Return the path to the GNN data directory in the experiment directory.

    Args:
        experiment_path (Path): The path of the experiment.

    Returns:
        Path: The path to the GNN data directory.
    """
    return representation_parent_directory(experiment_path) / "GNN"


def gnn_raw_data_path(experiment_path: Path) -> Path:
    """Return the path to the raw data directory for a GNN experiment.

    Args:
        experiment_path (Path): The path of the experiment.

    Returns:
        Path: The path to the raw data directory.
    """
    return gnn_data_path(experiment_path) / "raw"


def gnn_raw_data_file(file_name: str, experiment_path: Path) -> Path:
    """Return the path to a raw data file for a GNN experiment.

    Args:
        file_name (str): The name of the raw data file.
        experiment_path (Path): The path of the experiment.

    Returns:
        Path: The path to the raw data file.
    """
    return gnn_raw_data_path(experiment_path) / file_name


def unseen_predictions_parent_path(experiment_path: Path) -> Path:
    return experiment_path / "unseen_predictions"


def unseen_predictions_experiment_parent_path(file_name: str, experiment_path: Path) -> Path:
    file_name_no_ext = file_name.split(".")[0]
    return unseen_predictions_parent_path(experiment_path) / file_name_no_ext


def unseen_predictions_data_path(file_name: str, experiment_path: Path) -> Path:
    experiment_path = unseen_predictions_experiment_parent_path(
        file_name=file_name, experiment_path=experiment_path
    )
    return representation_parent_directory(experiment_path=experiment_path)


def unseen_gnn_raw_data_path(file_name: str, experiment_path: Path) -> Path:
    """Return the path to the raw GNN data directory for a prediction run.

    Args:
        file_name (str): The name of the raw data file.
        experiment_path (Path): The path of the experiment.

    Returns:
        Path: The path to the raw data directory for prediction.
    """
    experiment_path = unseen_predictions_experiment_parent_path(
        file_name=file_name, experiment_path=experiment_path
    )
    return gnn_raw_data_path(experiment_path=experiment_path)


def unseen_gnn_raw_data_file(file_name: str, experiment_path: Path) -> Path:
    """Return the path to the raw GNN data file for a prediction run.

    Args:
        file_name (str): The name of the raw data file.
        experiment_path (Path): The path of the experiment.

    Returns:
        Path: The path to the raw data file for prediction.
    """
    experiment_path = unseen_predictions_experiment_parent_path(
        file_name=file_name, experiment_path=experiment_path
    )
    return gnn_raw_data_file(file_name=file_name, experiment_path=experiment_path)


def unseen_predictions_ml_results_path(file_name: str, experiment_path: Path) -> Path:
    experiment_path = unseen_predictions_experiment_parent_path(
        file_name=file_name, experiment_path=experiment_path
    )
    return ml_results_parent_directory(experiment_path=experiment_path)


def ml_predictions_metrics_file_path(file_name: str, experiment_path: Path) -> Path:
    """Return the path to the predictions metrics file for a prediction run.

    Args:
        file_name (str): The name of the data file.
        experiment_path (Path): The path to the experiment directory.
    """
    experiment_path = unseen_predictions_experiment_parent_path(
        file_name=file_name, experiment_path=experiment_path
    )
    return model_metrics_file_path(experiment_path=experiment_path)


def ml_predictions_parent_path(file_name: str, experiment_path: Path) -> Path:
    experiment_path = unseen_predictions_experiment_parent_path(
        file_name=file_name, experiment_path=experiment_path
    )
    return ml_results_parent_directory(experiment_path=experiment_path)


def ml_predictions_file_path(file_name: str, experiment_path: Path) -> Path:
    """Return the path to the predictions CSV for a prediction run.

    Args:
        file_name (str): The name of the data file.
        experiment_path (Path): The path to the experiment directory.
    """
    experiment_path = unseen_predictions_experiment_parent_path(
        file_name=file_name, experiment_path=experiment_path
    )
    return ml_results_file_path(experiment_path=experiment_path)


def representation_file_path(experiment_path: Path) -> Path:
    """Return the path to the Descriptors directory in the experiment.

    Args:
        experiment_path (Path): The path to the experiment directory.
    """
    return representation_parent_directory(experiment_path) / "Descriptors"


def representation_file(file_name: str, experiment_path: Path) -> Path:
    """Return the path to a specific descriptor file in the experiment.

    Args:
        file_name (str): The name of the data file.
        experiment_path (Path): The path to the experiment directory.

    Returns:
        Path: The path to the data file.
    """
    return representation_parent_directory(experiment_path) / "Descriptors" / file_name


def train_tml_model_options_path(experiment_path: Path) -> Path:
    """Return the path to the TML model options JSON file.

    Args:
        experiment_path (Path): The path to the experiment directory.
    """
    return experiment_path / "train_tml_options.json"


def preprocessing_tml_model_options_path(experiment_path: Path) -> Path:
    """Return the path to the TML preprocessing options JSON file.

    Args:
        experiment_path (Path): The path to the experiment directory.
    """
    return experiment_path / "preprocessing_tml_options.json"


def data_spliting_options_path(experiment_path: Path) -> Path:
    """Return the path to the data splitting options JSON file.

    Args:
        experiment_path (Path): The path to the experiment directory.
    """
    return experiment_path / "split_options.json"


def train_gnn_model_options_path(experiment_path: Path) -> Path:
    """Return the path to the GNN model options JSON file.

    Args:
        experiment_path (Path): The path to the experiment directory.
    """
    return experiment_path / "train_gnn_options.json"


def general_options_path(experiment_path: Path) -> Path:
    """Return the path to the general options JSON file.

    Args:
        experiment_path (Path): The path to the experiment directory.
    """
    return experiment_path / "general_options.json"


def ml_results_parent_directory(experiment_path: Path) -> Path:
    """Return the path to the ml_results directory in the experiment.

    Args:
        experiment_path (Path): The path to the experiment directory.
    """
    return experiment_path / "ml_results"


def ml_results_file_path(experiment_path: Path) -> Path:
    """Return the path to the predictions CSV in the ml_results directory.

    Args:
        experiment_path (Path): The path to the experiment directory.
    """
    return ml_results_parent_directory(experiment_path) / "predictions.csv"


def model_dir(experiment_path: Path) -> Path:
    """Return the path to the models directory in the ml_results directory.

    Args:
        experiment_path (Path): The path to the experiment directory.
    """
    return ml_results_parent_directory(experiment_path) / "models"


def plots_directory(experiment_path: Path) -> Path:
    """Return the path to the plots directory in the ml_results directory.

    Args:
        experiment_path (Path): The path to the experiment directory.
    """
    return ml_results_parent_directory(experiment_path) / "plots"


def model_metrics_file_path(experiment_path: Path) -> Path:
    return ml_results_parent_directory(experiment_path) / "metrics.json"


def explanation_parent_directory(experiment_path: Path) -> Path:
    """Return the path to the explanations directory in the experiment.

    Args:
        experiment_path (Path): The path to the experiment directory.
    """
    return experiment_path / "explanations"


def explanation_json_file_path(experiment_path: Path) -> Path:
    """Return the path to the explanation JSON file.

    Args:
        experiment_path (Path): The path to the experiment directory.
    """
    return explanation_parent_directory(experiment_path) / "explanation.json"


def explanation_plots_path(experiment_path: Path, file_name: str) -> Path:
    """Return the path to an explanation plot file.

    Args:
        experiment_path (Path): The path to the experiment directory.
        file_name (str): The name of the explanation plots file.
    """
    return explanation_parent_directory(experiment_path) / "plots" / file_name
