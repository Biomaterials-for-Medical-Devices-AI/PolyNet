from pathlib import Path
from polynet.app.options.data import DataOptions
from polynet.app.services.configurations import save_options
from polynet.app.utils import create_directory
from polynet.app.options.file_paths import data_options_path
import os
from polynet.app.options.file_paths import polynet_experiments_base_dir


def get_experiments(base_dir: Path | None = None) -> list[str]:
    """Get the list of experiments in the PolyNet experiment directory.

    If `base_dir` is not specified, the default from `polynet_experiments_base_dir`
    is used

    Args:
        base_dir (Path | None, optional): Specify a base directory for experiments.
        Defaults to None.

    Returns:
        list[str]: The list of experiments.
    """
    # Get the base directory of all experiments
    if base_dir is None:
        base_dir = polynet_experiments_base_dir()

    if not base_dir.exists():
        # if no experiments directory, return empty list
        return []
    experiments = os.listdir(base_dir)
    # Filter out hidden files and directories
    experiments = filter(lambda x: not x.startswith("."), experiments)
    # Filter out files
    experiments = filter(lambda x: os.path.isdir(os.path.join(base_dir, x)), experiments)
    return list(experiments)


def create_experiment(save_dir: Path, data_options: DataOptions):
    """Create an experiment on disk with it's global plotting options,
    execution options and data options saved as `json` files.

    Args:
        save_dir (Path): The path to where the experiment will be created.
        plotting_options (PlottingOptions): The plotting options to save.
        execution_options (ExecutionOptions): The execution options to save.
        data_options (DataOptions): The data options to save.
    """
    create_directory(save_dir)
    # plot_file_path = plot_options_path(save_dir)
    # save_options(plot_file_path, plotting_options)
    data_file_path = data_options_path(save_dir)
    save_options(data_file_path, data_options)
