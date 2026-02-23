"""
polynet.config.schemas.base
============================
Shared Pydantic base models and mixins used across multiple config schemas.

Keep this module minimal — only add things here when two or more schemas
genuinely share the same fields and validation logic.
"""

from pydantic import BaseModel, Field


class HyperparamOptimConfig(BaseModel):
    """
    Mixin for any training config that supports hyperparameter optimisation.

    Inherit from this alongside ``BaseModel`` for any model type that can
    run a grid search or similar optimisation strategy.
    """

    hyperparameter_optimisation: bool = Field(
        default=False,
        description=(
            "Whether to run hyperparameter optimisation before final training. "
            "When True, the search grid defined in ``config/search_grids.py`` "
            "is used for the selected model."
        ),
    )


class PolynetBaseModel(BaseModel):
    """
    Base model for all polynet Pydantic schemas.

    Provides shared configuration for all schemas:
    - ``model_config``: forbids extra fields so that typos in YAML or the app
      are caught immediately rather than silently ignored.
    """

    model_config = {"frozen": False, "extra": "forbid"}
