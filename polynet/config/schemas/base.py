"""
polynet.config.schemas.base
============================
Shared Pydantic base models and mixins used across multiple config schemas.

Keep this module minimal — only add things here when two or more schemas
genuinely share the same fields and validation logic.
"""

from typing import Any

from pydantic import BaseModel, Field


class HyperparamOptimConfig(BaseModel):
    """
    Mixin for any training config that supports hyperparameter optimisation.

    Inherit from this alongside ``BaseModel`` for any model type that can
    run a randomised search or similar optimisation strategy.

    ``hpo_n_folds`` is shared by the GNN and TML pipelines: both score
    hyperparameter configurations by shuffled K-fold cross-validation
    (stratified for classification) on the training + validation samples of
    each split. Only ``k >= 2`` can be checked here; the data-dependent rules
    (``k`` vs. number of samples and smallest class) are checked once the data
    is split, by ``polynet.utils.validation.validate_hpo_folds``.

    ``hpo_num_samples`` (configurations sampled per search) and
    ``hpo_search_grid`` (user candidates replacing the defaults of
    ``polynet.config.search_grid``) are shared too; each training config
    validates its grid's keys (architectures / models) and parameters.
    """

    hyperparameter_optimisation: bool = Field(
        default=False,
        description=(
            "Whether to run hyperparameter optimisation before final training. "
            "When True, the search grid defined in ``config/search_grids.py`` "
            "is used for the selected model."
        ),
    )
    hpo_n_folds: int = Field(
        default=5,
        ge=2,
        description="Number of shuffled cross-validation folds used to score HPO configurations.",
    )
    hpo_num_samples: int = Field(
        default=150,
        ge=1,
        description="Number of hyperparameter configurations sampled per HPO run.",
    )
    hpo_search_grid: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "User search-grid candidates, keyed by architecture / model name; each "
            "parameter replaces the default candidates, the others keep their defaults."
        ),
    )


def ids_as_strings(value: Any) -> Any:
    """
    Turn a list of sample IDs into strings (``pydantic`` "before" validator helper).

    YAML reads unquoted numeric IDs (``[0, 12]``) as integers; sample IDs are
    compared as strings, so both ``[0, 12]`` and ``["0", "12"]`` are accepted.
    """
    if isinstance(value, list):
        return [str(v) for v in value]
    return value


def check_grid_parameters(
    where: str, params: dict[str, Any], allowed: set[str], reserved: frozenset[str]
) -> None:
    """
    Validate one entry of an ``hpo_search_grid``.

    Parameters
    ----------
    where:
        Config path of the entry, used in error messages
        (e.g. ``"gnn_training.hpo_search_grid.GCN"``).
    params:
        ``{parameter: candidates}`` supplied by the user.
    allowed:
        Parameters that may be searched for this entry.
    reserved:
        Parameters PolyNet sets itself.

    Raises
    ------
    ValueError
        If a parameter is reserved or unknown, or its candidates are not a
        non-empty list.
    """
    if not isinstance(params, dict) or not params:
        raise ValueError(f"{where} must be a non-empty mapping of parameter: [candidates].")
    for param, values in params.items():
        if param in reserved:
            raise ValueError(f"{where}.{param} is set by PolyNet and cannot be searched.")
        if param not in allowed:
            raise ValueError(
                f"{where}.{param} is not a searchable parameter here. "
                f"Allowed: {sorted(str(a) for a in allowed)}."
            )
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"{where}.{param} must be a non-empty list of candidates, got {values!r}."
            )


class PolynetBaseModel(BaseModel):
    """
    Base model for all polynet Pydantic schemas.

    Provides shared configuration for all schemas:
    - ``model_config``: forbids extra fields so that typos in YAML or the app
      are caught immediately rather than silently ignored.
    """

    model_config = {"frozen": False, "extra": "forbid"}
