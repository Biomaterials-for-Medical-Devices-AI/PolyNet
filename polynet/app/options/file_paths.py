from pathlib import Path


def polynet_experiments_base_dir() -> Path:
    """
    Return the path the base directory of all PolyNet experiments.

    This will be `/Users/<username>/HelixExperiments` on MacOS,
    `/home/<username>/PolyNetExperiments` on Linux, and
    `C:\\Users\\<username>\\PolyNetExperiments` on Windows.

    Returns:
        Path: The path to the PolyNet experiments base directory.
    """
    return Path.home() / "PolyNetExperiments"


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
    """Return the path to an experiment's data options.
    The path will be to a `json` file called `data_options.json`

    Args:
        experiment_path (str): The path of the experiment.

    Returns:
        Path: The path to the experiment's data options.

    Examples:
    ```python
    experiment_name = "test"
    experiment_path = helix_experiments_base_dir() / experiment_name
    data_options_file = data_options_path(experiment_path)
    ```
    """
    return experiment_path / "data_options.json"


def representation_options_path(experiment_path: Path) -> Path:
    """Return the path to an experiment's data options.
    The path will be to a `json` file called `data_options.json`

    Args:
        experiment_path (str): The path of the experiment.

    Returns:
        Path: The path to the experiment's data options.

    Examples:
    ```python
    experiment_name = "test"
    experiment_path = helix_experiments_base_dir() / experiment_name
    data_options_file = data_options_path(experiment_path)
    ```
    """
    return experiment_path / "representation_options.json"


def representation_parent_directory(experiment_path: Path) -> Path:
    """Return the path to the representation directory in the experiment directory.
    The path will be to a directory called `representation`.

    Args:
        experiment_path (Path): The path of the experiment.

    Returns:
    """
    return experiment_path / "representation"


def gnn_data_path(experiment_path: Path) -> Path:
    """Return the path to the raw data for a GNN experiment.
    The path will be to a `csv` file called `raw_data.csv`

    Args:
        experiment_path (Path): The path of the experiment.

    Returns:
        Path: The path to the raw data directory.
    """
    return representation_parent_directory(experiment_path) / "GNN"


def gnn_raw_data_path(experiment_path: Path) -> Path:
    """Return the path to the raw data for a GNN experiment.
    The path will be to a directory called `raw` inside the `training` directory.
    Args:
        experiment_path (Path): The path of the experiment.
    Returns:
        Path: The path to the raw data directory.
    """

    return gnn_data_path(experiment_path) / "training" / "raw"


def gnn_raw_data_file(file_name: str, experiment_path: Path):
    """Return the path to the raw data file for a GNN experiment.
    The path will be to a `csv` file called `raw_data.csv`

    Args:
        file_name (str): The name of the raw data file.
        experiment_path (Path): The path of the experiment.

    Returns:
        Path: The path to the raw data file.
    """
    return gnn_raw_data_path(experiment_path) / file_name


def gnn_raw_data_predict_path(file_name: str, experiment_path: Path) -> Path:
    """
    Return the path to the raw data file for a GNN experiment's prediction.
    The path will be to a `csv` file called `raw_data.csv`

    Args:
        filen_name (str): The name of the raw data file.
        experiment_path (Path): The path of the experiment.
    Returns:
        Path: The path to the raw data file for prediction.
    """

    file_name_no_ext = file_name.split(".")[0]

    return gnn_data_path(experiment_path) / "predict" / file_name_no_ext / "raw"


def gnn_raw_data_predict_file(file_name: str, experiment_path: Path) -> Path:
    """
    Return the path to the raw data file for a GNN experiment's prediction.
    The path will be to a `csv` file called `raw_data.csv`

    Args:
        file_name (str): The name of the raw data file.
        experiment_path (Path): The path of the experiment.
    Returns:
     Path: The path to the raw data file for prediction.
    """
    return (
        gnn_raw_data_predict_path(file_name=file_name, experiment_path=experiment_path) / file_name
    )


def representation_file_path(experiment_path: Path) -> Path:
    """
    Return the path to the representation file in the experiment directory.
    Args:
        file_name (str): The name of the representation file.
        experiment_path (Path): The path to the experiment directory.
    Returns:


    """
    return representation_parent_directory(experiment_path) / "Descriptors"


def representation_file(file_name: str, experiment_path: Path) -> Path:
    """
    Return the path to the data file in the experiment directory.
    Args:
        file_name (str): The name of the data file.
        experiment_path (Path): The path to the experiment directory.
    Returns:
        Path: The path to the data file.
    """
    return representation_parent_directory(experiment_path) / "Descriptors" / file_name


def train_gnn_model_options_path(experiment_path: Path) -> Path:
    """
    Return the path to the GNN model options file in the experiment directory.
    Args:
        experiment_path (Path): The path to the experiment directory.
    Returns:
    """
    return experiment_path / "train_gnn_options.json"


def general_options_path(experiment_path: Path) -> Path:
    """
    Return the path to the GNN model options file in the experiment directory.
    Args:
        experiment_path (Path): The path to the experiment directory.
    Returns:
    """
    return experiment_path / "general_options.json"


def ml_results_parent_directory(experiment_path: Path) -> Path:
    """
    Return the path to the machine learning results directory in the experiment directory.
    Args:
        experiment_path (Path): The path to the experiment directory.
    Returns:

    """
    return experiment_path / "ml_results"


def ml_gnn_results_directory(experiment_path: Path) -> Path:
    """
    Return the path to the machine learning GNN results directory in the experiment directory.
    Args:
        experiment_path (Path): The path to the experiment directory.
    Returns:
    """
    return ml_results_parent_directory(experiment_path) / "GNN"


def ml_gnn_results_file_path(experiment_path: Path, file_name: str) -> Path:
    """
    Return the path to the machine learning GNN results file in the experiment directory.
    Args:
        experiment_path (Path): The path to the experiment directory.
        file_name (str): The name of the results file.
    Returns:
    """
    return ml_gnn_results_directory(experiment_path) / file_name


def gnn_model_dir(experiment_path: Path) -> Path:
    """
    Return the path to the GNN model directory in the experiment directory.
    Args:
        experiment_path (Path): The path to the experiment directory.
    Returns:
    """
    return ml_gnn_results_directory(experiment_path) / "models"


def gnn_plots_directory(experiment_path: Path) -> Path:
    """
    Return the path to the GNN plots directory in the experiment directory.
    Args:
        experiment_path (Path): The path to the experiment directory.
    Returns:
    """
    return ml_gnn_results_directory(experiment_path) / "plots"


def gnn_model_metrics_file_path(experiment_path: Path) -> Path:

    return ml_gnn_results_directory(experiment_path) / "metrics.json"


def prediction_results_parent_path(experiment_path: Path) -> Path:
    """
    Return the path to the prediction results directory in the experiment directory.
    Args:
        experiment_path (Path): The path to the experiment directory.
    Returns:
    """
    return experiment_path / "ml_predictions"


def gnn_predictions_file_path(experiment_path: Path) -> Path:
    """
    Return the path to the machine learning GNN predictions file in the experiment directory.
    Args:
        experiment_path (Path): The path to the experiment directory.
        file_name (str): The name of the predictions file.
    Returns:
    """
    return prediction_results_parent_path(experiment_path) / "GNN"


def gnn_predictions_file(experiment_path: Path) -> Path:
    """
    Return the path to the machine learning GNN predictions file in the experiment directory.
    Args:
        experiment_path (Path): The path to the experiment directory.
        file_name (str): The name of the predictions file.
    Returns:
    """
    return prediction_results_parent_path(experiment_path) / "predictions.csv"


def gnn_predictions_plots_directory(experiment_path: Path) -> Path:
    """
    Return the path to the GNN predictions plots directory in the experiment directory.
    Args:
        experiment_path (Path): The path to the experiment directory.
    Returns:
    """
    return prediction_results_parent_path(experiment_path) / "plots"


def gnn_predictions_metrics_file_path(experiment_path: Path) -> Path:
    """
    Return the path to the GNN predictions metrics file in the experiment directory.
    Args:
        experiment_path (Path): The path to the experiment directory.
    Returns:
    """
    return prediction_results_parent_path(experiment_path) / "metrics.json"


def explanation_parent_directory(experiment_path: Path) -> Path:
    """
    Return the path to the explanation directory in the experiment directory.
    Args:
        experiment_path (Path): The path to the experiment directory.
    Returns:
    """
    return experiment_path / "explanations"


def explanation_json_file_path(experiment_path: Path) -> Path:
    """
    Return the path to the explanation file in the experiment directory.
    Args:
        experiment_path (Path): The path to the experiment directory.
        file_name (str): The name of the explanation file.
    Returns:
    """
    return explanation_parent_directory(experiment_path) / "explanation.json"


def explanation_plots_path(experiment_path: Path, file_name: str) -> Path:
    """
    Return the path to the explanation plots directory in the experiment directory.
    Args:
        experiment_path (Path): The path to the experiment directory.
        file_name (str): The name of the explanation plots file.
    Returns:
    """
    return explanation_parent_directory(experiment_path) / "plots" / file_name
