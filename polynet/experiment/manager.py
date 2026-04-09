"""
polynet.experiment.manager
===========================
File-system helpers for creating and listing PolyNet experiments.
"""

import os
from pathlib import Path

from polynet.config.io import save_options
from polynet.config.paths import data_options_path, polynet_experiments_base_dir
from polynet.config.schemas.data import DataConfig
from polynet.utils import create_directory


def get_experiments(base_dir: Path | None = None) -> list[str]:
    """Get the list of experiments in the PolyNet experiment directory.

    If ``base_dir`` is not specified, the default from
    ``polynet_experiments_base_dir`` is used.

    Args:
        base_dir (Path | None): Specify a base directory for experiments.
            Defaults to None.

    Returns:
        list[str]: Sorted list of experiment names.
    """
    if base_dir is None:
        base_dir = polynet_experiments_base_dir()

    if not base_dir.exists():
        return []

    experiments = os.listdir(base_dir)
    experiments = filter(lambda x: not x.startswith("."), experiments)
    experiments = filter(lambda x: os.path.isdir(os.path.join(base_dir, x)), experiments)
    return sorted(list(experiments))


def create_experiment(experiment_path: Path, data_options: DataConfig) -> None:
    """Create an experiment directory and persist its data options.

    Args:
        experiment_path (Path): The path where the experiment will be created.
        data_options (DataConfig): The data configuration to save.
    """
    create_directory(experiment_path)
    path = data_options_path(experiment_path)
    save_options(path=path, options=data_options)
