"""
polynet.config.from_yaml
=========================
YAML-based entry point for constructing an ``ExperimentConfig``.

This is the entry point for script-based users who define their experiment
in a YAML file and run it via a ``main.py``::

    from polynet.config.from_yaml import load_config

    cfg = load_config("experiments/my_experiment.yaml")

YAML structure
--------------
The YAML file must have one top-level key per pipeline section.
Both the new canonical names and the legacy dataclass names are accepted
for backward compatibility::

    # Canonical form (recommended)
    data:
      data_name: My Polymer Dataset
      data_path: data/polymers.csv
      smiles_cols: [SMILES]
      target_variable_col: Tg
      problem_type: regression

    general:
      split_type: train_val_test
      split_method: random
      test_ratio: 0.2
      val_ratio: 0.1
      random_seed: 42
      n_bootstrap_iterations: 5

    representation:
      smiles_merge_approach: [no_merging]
      node_features:
        GetAtomicNum:
          allowable_vals: [6, 7, 8, 9]
          wildcard: true
      edge_features: {}
      molecular_descriptors: {}

    train_gnn:
      train_gnn: true
      gnn_convolutional_layers:
        GCN:
          improved: true
          n_convolutions: 2
          embedding_dim: 128
          pooling: global_max_pool
          readout_layers: 2
          dropout: 0.01
          learning_rate: 0.01
          batch_size: 32
          apply_weighting_to_graph: no_weighting
      share_gnn_parameters: true
      hyperparameter_optimisation: false
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from polynet.config._loader import build_experiment_config
from polynet.config.experiment import ExperimentConfig


def load_config(path: str | Path) -> ExperimentConfig:
    """
    Load and validate an experiment configuration from a YAML file.

    Parameters
    ----------
    path:
        Path to the YAML configuration file. Both absolute and relative
        paths are accepted.

    Returns
    -------
    ExperimentConfig
        A fully validated experiment configuration ready to be passed
        to the ``PipelineRunner``.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist at the given path.
    yaml.YAMLError
        If the file is not valid YAML.
    pydantic.ValidationError
        If required fields are missing or values fail validation.
    ValueError
        If the YAML structure contains unrecognised top-level sections.

    Examples
    --------
    >>> from polynet.config.from_yaml import load_config
    >>> cfg = load_config("experiments/tg_prediction.yaml")
    >>> cfg.data.target_variable_col
    'Tg'
    >>> cfg.runs_gnn
    True
    """
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "PyYAML is required to load YAML config files. " "Install it with: pip install pyyaml"
        ) from e

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: '{path.resolve()}'. " "Check the path and try again."
        )

    if path.suffix not in {".yaml", ".yml"}:
        raise ValueError(
            f"Expected a .yaml or .yml file, got '{path.suffix}'. " f"File path: '{path}'"
        )

    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(
            f"Expected the YAML file to contain a mapping at the top level, "
            f"got {type(raw).__name__}. Check the structure of '{path}'."
        )

    if not raw:
        raise ValueError(f"The YAML config file at '{path}' is empty or contains only comments.")

    return build_experiment_config(raw)
