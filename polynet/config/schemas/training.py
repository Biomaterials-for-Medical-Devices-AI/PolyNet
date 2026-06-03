"""
polynet.config.schemas.training
================================
Pydantic schemas for GNN and traditional ML training configuration.

Both schemas inherit from ``HyperparamOptimConfig`` for the shared
hyperparameter optimisation flag. All other fields are model-family specific.
"""

import warnings

from pydantic import Field, model_validator

from polynet.config.enums import HpoSplitStrategy, Network, TraditionalMLModel, TransformDescriptor
from polynet.config.schemas.base import HyperparamOptimConfig, PolynetBaseModel

# ---------------------------------------------------------------------------
# GNN training config
# ---------------------------------------------------------------------------


class TrainGNNConfig(PolynetBaseModel, HyperparamOptimConfig):
    """
    Configuration for training graph neural network models.

    Attributes
    ----------
    train_gnn:
        Master switch — set to False to skip GNN training entirely.
    gnn_convolutional_layers:
        Mapping from ``Network`` enum member to a dictionary of
        architecture hyperparameters (``ArchitectureParam`` keys → values).
        Each entry defines one GNN architecture to train.

        Example::

            {
                Network.GCN: {
                    ArchitectureParam.NumConvolutions: 3,
                    ArchitectureParam.EmbeddingDim: 64,
                    ArchitectureParam.Dropout: 0.05,
                    ArchitectureParam.PoolingMethod: Pooling.GlobalMeanPool,
                    TrainingParam.LearningRate: 0.001,
                    TrainingParam.BatchSize: 32,
                }
            }

    share_gnn_parameters:
        If True and a polymer is defined by multiple SMILES strings, all
        monomers share the same GNN weights during message passing.
        If False, each SMILES position gets its own GNN.
    hyperparameter_optimisation:
        Inherited from ``HyperparamOptimConfig``. When True, a grid search
        is run over the parameter space defined in ``config/search_grids.py``
        before final training.
    """

    train_gnn: bool = Field(
        default=True, description="Master switch to enable or disable GNN training."
    )
    gnn_convolutional_layers: dict[Network, dict] = Field(
        ..., description="GNN architectures to train, keyed by Network enum."
    )
    share_gnn_parameters: bool = Field(
        default=True, description="Share GNN weights across monomers in multi-SMILES polymers."
    )
    epochs: int = Field(default=250, ge=1, description="Number of training epochs per GNN model.")
    hpo_split_strategy: HpoSplitStrategy = Field(
        default=HpoSplitStrategy.CrossValidation,
        description="Split strategy used inside the HPO loop.",
    )
    hpo_n_folds: int = Field(default=5, ge=2, description="Folds for CrossValidation HPO strategy.")
    hpo_val_fraction: float = Field(
        default=0.2, gt=0.0, lt=1.0, description="Val fraction for Holdout / RepeatedHoldout HPO."
    )
    hpo_n_repeats: int = Field(
        default=3, ge=1, description="Number of random splits for RepeatedHoldout HPO."
    )

    @model_validator(mode="after")
    def layers_required_when_training(self) -> "TrainGNNConfig":
        if self.train_gnn and not self.gnn_convolutional_layers:
            raise ValueError(
                "train_gnn is True but gnn_convolutional_layers is empty. "
                "Define at least one GNN architecture to train."
            )
        return self

    @model_validator(mode="after")
    def warn_on_unused_hpo_params(self) -> "TrainGNNConfig":
        """
        Warn when HPO parameters are set to non-default values that have no
        effect under the chosen ``hpo_split_strategy``.

        Each strategy uses only a subset of the four HPO parameters:

        - ``cross_validation``  → only ``hpo_n_folds`` is used.
          ``hpo_val_fraction`` and ``hpo_n_repeats`` are ignored.

        - ``holdout``           → only ``hpo_val_fraction`` is used.
          ``hpo_n_folds`` and ``hpo_n_repeats`` are ignored.

        - ``repeated_holdout``  → ``hpo_val_fraction`` and ``hpo_n_repeats``
          are used. ``hpo_n_folds`` is ignored.

        Only non-default values trigger a warning, because a field left at its
        default was almost certainly not intentionally set by the user.
        """
        strategy = self.hpo_split_strategy

        _DEFAULTS = {"hpo_n_folds": 5, "hpo_val_fraction": 0.2, "hpo_n_repeats": 3}

        if strategy == HpoSplitStrategy.CrossValidation:
            if self.hpo_val_fraction != _DEFAULTS["hpo_val_fraction"]:
                warnings.warn(
                    f"hpo_val_fraction={self.hpo_val_fraction!r} has no effect when "
                    "hpo_split_strategy='cross_validation'. "
                    "Expected: set hpo_n_folds to control the number of CV folds instead.",
                    UserWarning,
                    stacklevel=2,
                )
            if self.hpo_n_repeats != _DEFAULTS["hpo_n_repeats"]:
                warnings.warn(
                    f"hpo_n_repeats={self.hpo_n_repeats!r} has no effect when "
                    "hpo_split_strategy='cross_validation'. "
                    "Expected: set hpo_n_folds to control the number of CV folds instead.",
                    UserWarning,
                    stacklevel=2,
                )

        elif strategy == HpoSplitStrategy.Holdout:
            if self.hpo_n_folds != _DEFAULTS["hpo_n_folds"]:
                warnings.warn(
                    f"hpo_n_folds={self.hpo_n_folds!r} has no effect when "
                    "hpo_split_strategy='holdout'. "
                    "Expected: set hpo_val_fraction to control the validation split size instead.",
                    UserWarning,
                    stacklevel=2,
                )
            if self.hpo_n_repeats != _DEFAULTS["hpo_n_repeats"]:
                warnings.warn(
                    f"hpo_n_repeats={self.hpo_n_repeats!r} has no effect when "
                    "hpo_split_strategy='holdout'. "
                    "Expected: use hpo_split_strategy='repeated_holdout' if you want "
                    "multiple independent splits.",
                    UserWarning,
                    stacklevel=2,
                )

        elif strategy == HpoSplitStrategy.RepeatedHoldout:
            if self.hpo_n_folds != _DEFAULTS["hpo_n_folds"]:
                warnings.warn(
                    f"hpo_n_folds={self.hpo_n_folds!r} has no effect when "
                    "hpo_split_strategy='repeated_holdout'. "
                    "Expected: set hpo_val_fraction and hpo_n_repeats to control the "
                    "validation fraction and number of random splits instead.",
                    UserWarning,
                    stacklevel=2,
                )

        return self


# ---------------------------------------------------------------------------
# Traditional ML training config
# ---------------------------------------------------------------------------


class TrainTMLConfig(PolynetBaseModel, HyperparamOptimConfig):
    """
    Configuration for training traditional (non-deep-learning) ML models.

    Attributes
    ----------
    train_tml:
        Master switch — set to False to skip traditional ML training entirely.
    selected_models:
        List of ``TraditionalMLModel`` members to train. At least one must
        be provided when ``train_tml`` is True.
    model_params:
        Optional mapping from ``TraditionalMLModel`` to a dictionary of
        fixed hyperparameters passed directly to the model constructor.
        Models not listed here receive their sklearn/XGBoost defaults.

        Example::

            {
                TraditionalMLModel.RandomForest: {
                    "n_estimators": 300,
                    "max_depth": 6,
                },
                TraditionalMLModel.XGBoost: {
                    "learning_rate": 0.05,
                }
            }

    transform_features:
        Feature scaling / transformation applied to descriptor inputs
        before training. Has no effect on raw graph inputs.
    hyperparameter_optimisation:
        Inherited from ``HyperparamOptimConfig``. When True, a grid search
        is run over the parameter space defined in ``config/search_grids.py``
        for each selected model.
    """

    train_tml: bool = Field(
        default=False, description="Master switch to enable or disable traditional ML training."
    )
    selected_models: dict[TraditionalMLModel, dict] | None = Field(
        default=None,
        description="Fixed hyperparameters per model. Overrides defaults, not the search grid.",
    )

    @model_validator(mode="after")
    def models_required_when_training(self) -> "TrainTMLConfig":
        if self.train_tml and not self.selected_models:
            raise ValueError(
                "train_tml is True but selected_models is empty. "
                "Provide at least one TraditionalMLModel to train."
            )
        return self
