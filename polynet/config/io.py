"""
polynet.config.io
=================
Serialisation helpers for experiment configuration objects.

Supports dataclasses, Pydantic v2 BaseModel instances, and plain dicts.
All configs are persisted as human-readable JSON files.
"""

import dataclasses
import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

Options = TypeVar("Options")


def save_options(path: Path, options: Any) -> None:
    """Save options/config to a JSON file at the specified path.

    Supports:
      - dataclass instances
      - Pydantic v2 models (BaseModel)
      - plain dicts
    """
    if dataclasses.is_dataclass(options):
        payload = dataclasses.asdict(options)
    elif BaseModel is not None and isinstance(options, BaseModel):
        payload = options.model_dump(mode="json")
    elif isinstance(options, dict):
        payload = options
    else:
        raise TypeError(
            f"Unsupported options type: {type(options)!r}. "
            "Expected dataclass, pydantic BaseModel, or dict."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)


def load_options(path: Path, options_class: Options) -> Options:
    """Load options from a JSON file at the specified path.

    Args:
        path (Path): The path to the JSON file.
        options_class (Options): The class of the options to load.

    Returns:
        Options: The loaded options.
    """
    with open(path, "r") as json_file:
        options_json = json.load(json_file)
    return options_class(**options_json)
