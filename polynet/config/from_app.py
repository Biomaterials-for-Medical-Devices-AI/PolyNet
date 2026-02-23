"""
polynet.config.from_app
========================
Streamlit app entry point for constructing an ``ExperimentConfig``.

Each page of the Streamlit app saves its configuration as a separate JSON
file. This module collects those files, merges them into a single dict, and
produces a validated ``ExperimentConfig`` — the same object that the
YAML entry point produces.

This ensures the app and script entry points are fully equivalent: both
produce an ``ExperimentConfig``, both go through the same validation, and
both hand off to the same ``PipelineRunner``.

Usage (from within the Streamlit app)
--------------------------------------
::

    from polynet.config.from_app import load_config_from_app_state

    cfg = load_config_from_app_state(config_dir="experiments/my_run/")

    # Or pass an explicit list of JSON file paths
    from polynet.config.from_app import load_config_from_json_files

    cfg = load_config_from_json_files([
        "experiments/my_run/data.json",
        "experiments/my_run/general.json",
        "experiments/my_run/representation.json",
        "experiments/my_run/train_gnn.json",
    ])

JSON file naming conventions
-----------------------------
Each JSON file should have a filename (without extension) that matches
either a canonical section name or a legacy dataclass name::

    data.json             → DataConfig       (or DataOptions.json)
    general.json          → GeneralConfig    (or GeneralConfigOptions.json)
    representation.json   → RepresentationConfig
    plotting.json         → PlottingConfig   (optional)
    train_gnn.json        → TrainGNNConfig   (or TrainGNNOptions.json)
    train_tml.json        → TrainTMLConfig   (or TrainTMLOptions.json)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from polynet.config._loader import _SECTION_NORMALISERS, build_experiment_config
from polynet.config.experiment import ExperimentConfig

# All recognised JSON filenames (stem only, case-sensitive)
_RECOGNISED_STEMS = set(_SECTION_NORMALISERS.keys())


def load_config_from_json_files(paths: list[str | Path]) -> ExperimentConfig:
    """
    Build an ``ExperimentConfig`` from a list of per-page JSON files.

    Parameters
    ----------
    paths:
        List of paths to JSON files. Each file must have a filename stem
        that matches a recognised section name (canonical or legacy).
        Unrecognised filenames raise a ``ValueError``.

    Returns
    -------
    ExperimentConfig
        A fully validated experiment configuration.

    Raises
    ------
    FileNotFoundError
        If any of the provided paths do not exist.
    json.JSONDecodeError
        If any file contains invalid JSON.
    ValueError
        If a filename stem is not a recognised section name, or if two
        files map to the same config section.
    pydantic.ValidationError
        If required fields are missing or values fail validation after
        merging all files.

    Examples
    --------
    >>> from polynet.config.from_app import load_config_from_json_files
    >>> cfg = load_config_from_json_files([
    ...     "runs/exp1/data.json",
    ...     "runs/exp1/general.json",
    ...     "runs/exp1/representation.json",
    ...     "runs/exp1/train_gnn.json",
    ... ])
    """
    merged: dict[str, Any] = {}

    for raw_path in paths:
        path = Path(raw_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found: '{path.resolve()}'. "
                "Check that all page configs have been saved before running the pipeline."
            )

        stem = path.stem  # filename without extension, e.g. "data" or "TrainGNNOptions"

        if stem not in _RECOGNISED_STEMS:
            raise ValueError(
                f"Unrecognised config file: '{path.name}'. "
                f"The filename stem must be one of: {sorted(_RECOGNISED_STEMS)}. "
                "Rename the file or check that it was saved by the correct app page."
            )

        with path.open("r", encoding="utf-8") as f:
            section_data: dict[str, Any] = json.load(f)

        if not isinstance(section_data, dict):
            raise ValueError(
                f"Expected '{path.name}' to contain a JSON object at the top level, "
                f"got {type(section_data).__name__}."
            )

        if stem in merged:
            raise ValueError(
                f"Duplicate section '{stem}': more than one file maps to the same "
                f"config section. Found duplicate at '{path}'."
            )

        merged[stem] = section_data

    return build_experiment_config(merged)


def load_config_from_app_state(
    config_dir: str | Path, *, required_sections: list[str] | None = None
) -> ExperimentConfig:
    """
    Build an ``ExperimentConfig`` by scanning a directory for JSON page configs.

    Scans ``config_dir`` for ``.json`` files whose stems match recognised
    section names, loads them all, and constructs a validated config.

    This is the most convenient function to call from the final Streamlit
    page (the "Run Experiment" page) since it does not require the app to
    track which files have been saved.

    Parameters
    ----------
    config_dir:
        Directory containing the per-page JSON config files.
    required_sections:
        Optional list of section stems that must be present. If any are
        missing a ``ValueError`` is raised with a clear message listing
        which pages still need to be completed. Defaults to requiring
        ``["data", "general", "representation"]`` — the minimum needed
        for any experiment.

    Returns
    -------
    ExperimentConfig
        A fully validated experiment configuration.

    Raises
    ------
    FileNotFoundError
        If ``config_dir`` does not exist.
    ValueError
        If required sections are missing, or if the directory contains no
        recognised JSON files.
    pydantic.ValidationError
        If required fields are missing or values fail validation.

    Examples
    --------
    >>> from polynet.config.from_app import load_config_from_app_state
    >>> cfg = load_config_from_app_state("experiments/current_run/")
    >>> cfg.runs_gnn
    True
    """
    if required_sections is None:
        required_sections = ["data", "general", "representation"]

    config_dir = Path(config_dir)

    if not config_dir.exists():
        raise FileNotFoundError(
            f"Config directory not found: '{config_dir.resolve()}'. "
            "Ensure the experiment directory has been created and pages have been saved."
        )

    if not config_dir.is_dir():
        raise ValueError(
            f"'{config_dir}' is not a directory. "
            "Provide the path to the folder containing the per-page JSON files."
        )

    # Collect all JSON files with recognised stems
    found_paths: list[Path] = []
    for json_file in sorted(config_dir.glob("*.json")):
        if json_file.stem in _RECOGNISED_STEMS:
            found_paths.append(json_file)

    if not found_paths:
        raise ValueError(
            f"No recognised config files found in '{config_dir}'. "
            f"Expected JSON files named after one of: {sorted(_RECOGNISED_STEMS)}."
        )

    # Check required sections are present
    found_stems = {p.stem for p in found_paths}
    missing = [s for s in required_sections if s not in found_stems]
    if missing:
        raise ValueError(
            f"The following required config sections are missing from '{config_dir}': "
            f"{missing}. Complete the corresponding app pages and save before running."
        )

    return load_config_from_json_files(found_paths)


def save_section_to_json(
    section_data: dict[str, Any], section_name: str, config_dir: str | Path
) -> Path:
    """
    Save a single page's config dict to a JSON file in ``config_dir``.

    This is a convenience helper for use within individual Streamlit pages.
    It ensures all pages save their configs in a consistent format and
    location, avoiding ad-hoc JSON serialisation scattered across the app.

    Parameters
    ----------
    section_data:
        The raw dict produced by the page (e.g. from ``session_state``).
    section_name:
        The canonical section name for this page (e.g. ``"data"``,
        ``"train_gnn"``). Used as the filename stem.
    config_dir:
        Directory to save into. Created if it does not exist.

    Returns
    -------
    Path
        The path to the saved JSON file.

    Raises
    ------
    ValueError
        If ``section_name`` is not a recognised section name.

    Examples
    --------
    >>> from polynet.config.from_app import save_section_to_json
    >>> save_section_to_json(
    ...     section_data={"data_name": "Tg dataset", ...},
    ...     section_name="data",
    ...     config_dir="experiments/current_run/",
    ... )
    PosixPath('experiments/current_run/data.json')
    """
    if section_name not in _RECOGNISED_STEMS:
        raise ValueError(
            f"Unrecognised section name: '{section_name}'. "
            f"Must be one of: {sorted(_RECOGNISED_STEMS)}."
        )

    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)

    output_path = config_dir / f"{section_name}.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(section_data, f, indent=2, default=str)

    return output_path
