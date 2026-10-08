"""
polynet.config.schemas.base
============================
Shared Pydantic base models and mixins used across multiple config schemas.

Keep this module minimal — only add things here when two or more schemas
genuinely share the same fields and validation logic.
"""

from typing import Any
import warnings

from pydantic import BaseModel, Field

# Default number of hyperparameter configurations sampled per HPO run (GNN Ray
# Tune trials and TML RandomizedSearchCV candidates alike).
DEFAULT_HPO_NUM_SAMPLES = 50


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
            "Declares whether hyperparameter optimisation is expected. HPO runs for every "
            "architecture / model whose hyperparameter block is empty ({}), whatever this "
            "flag says. When not set, it is filled in from the blocks; when it contradicts "
            "them, a warning is given (see ``resolve_hpo_flag``)."
        ),
    )
    hpo_n_folds: int = Field(
        default=5,
        ge=2,
        description="Number of shuffled cross-validation folds used to score HPO configurations.",
    )
    hpo_num_samples: int = Field(
        default=DEFAULT_HPO_NUM_SAMPLES,
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


def resolve_hpo_flag(cfg: BaseModel, blocks: dict | None, where: str) -> None:
    """
    Fill in ``hyperparameter_optimisation`` or warn when it contradicts the blocks.

    HPO runs for every architecture / model whose block is empty (``{}``); the
    flag does not switch it on or off. When the flag is not set, it is filled
    in from the blocks (true if any block is empty), so saved options record
    whether HPO ran. When it is set and contradicts the blocks, a warning
    explains what will actually happen.

    Parameters
    ----------
    cfg:
        A training config with ``hyperparameter_optimisation``.
    blocks:
        ``{architecture or model: hyperparameters}``.
    where:
        Config section for the message (e.g. ``"gnn_training"``).
    """
    empty = sorted(str(getattr(k, "value", k)) for k, v in (blocks or {}).items() if not v)
    if "hyperparameter_optimisation" not in cfg.model_fields_set:
        cfg.hyperparameter_optimisation = bool(empty)
        return
    if cfg.hyperparameter_optimisation and not empty:
        warnings.warn(
            f"{where}.hyperparameter_optimisation is true, but every model has explicit "
            "hyperparameters, so no hyperparameter optimisation will run. Leave a model's "
            "block empty ({}) to tune it.",
            UserWarning,
            stacklevel=3,
        )
    elif not cfg.hyperparameter_optimisation and empty:
        warnings.warn(
            f"{where}.hyperparameter_optimisation is false, but {empty} have empty "
            "hyperparameter blocks ({}), so hyperparameter optimisation will run for them. "
            "Give them hyperparameters to train them without tuning.",
            UserWarning,
            stacklevel=3,
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
