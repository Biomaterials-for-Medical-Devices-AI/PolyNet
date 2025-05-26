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


def gnn_raw_data_path(experiment_path: Path) -> Path:
    """Return the path to the raw data for a GNN experiment.
    The path will be to a `csv` file called `raw_data.csv`

    Args:
        experiment_path (Path): The path of the experiment.

    Returns:
        Path: The path to the raw data directory.
    """
    return experiment_path / "representation" / "GNN" / "raw"


def gnn_raw_data_file(file_name: str, experiment_path: Path):
    """Return the path to the raw data file for a GNN experiment.
    The path will be to a `csv` file called `raw_data.csv`

    Args:
        file_name (str): The name of the raw data file.
        experiment_path (Path): The path of the experiment.

    Returns:
        Path: The path to the raw data file.
    """
    return experiment_path / "representation" / "GNN" / "raw" / file_name


def representation_file_path(experiment_path: Path) -> Path:
    """
    Return the path to the representation file in the experiment directory.
    Args:
        file_name (str): The name of the representation file.
        experiment_path (Path): The path to the experiment directory.
    Returns:


    """
    return experiment_path / "representation" / "Descriptors"


def representation_file(file_name: str, experiment_path: Path) -> Path:
    """
    Return the path to the data file in the experiment directory.
    Args:
        file_name (str): The name of the data file.
        experiment_path (Path): The path to the experiment directory.
    Returns:
        Path: The path to the data file.
    """
    return experiment_path / "representation" / "Descriptors" / file_name
