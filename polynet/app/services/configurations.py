from pathlib import Path
import dataclasses
import json
from typing import TypeVar

Options = TypeVar("Options")


def save_options(path: Path, options: Options):
    """Save options to a `json` file at the specified path.

    Args:
        path (Path): The path to the `json` file.
        options (Options): The options to save.
    """
    options_json = dataclasses.asdict(options)
    with open(path, "w") as json_file:
        json.dump(options_json, json_file, indent=4)


def load_options(path: Path, options_class: Options) -> Options:
    """Load options from a `json` file at the specified path.

    Args:
        path (Path): The path to the `json` file.
        options_class (Options): The class of the options to load.

    Returns:
        Options: The loaded options.
    """
    with open(path, "r") as json_file:
        options_json = json.load(json_file)
    return options_class(**options_json)
