"""
polynet.config.schemas.training
================================
Pydantic schemas for GNN and traditional ML training configuration.

Both schemas inherit from ``HyperparamOptimConfig`` for the shared
hyperparameter optimisation flag. All other fields are model-family specific.
"""

import warnings

from pydantic import Field, model_validator

from polynet.config.enums import (
    HpoSplitStrategy,
    Network,
    Optimizer,
    RegressionLoss,
    Scheduler,
    TraditionalMLModel,
    TrainingParam,
    TransformDescriptor,
)
from polynet.config.schemas.base import (
    HyperparamOptimConfig,
    PolynetBaseModel,
    check_grid_parameters,
)
from polynet.config.search_grid import (
    RESERVED_GNN_GRID_KEYS,
    RESERVED_TML_GRID_KEYS,
    SHARED_GNN_GRID_KEY,
    gnn_grid_parameters,
    shared_gnn_grid_parameters,
)

# ---------------------------------------------------------------------------
# GNN optimisation settings
# ---------------------------------------------------------------------------


class GNNOptimisationConfig(PolynetBaseModel):
    """
    Optimiser, learning-rate scheduler and loss used to train GNNs.

    Applied identically to final training and to every HPO trial. The
    defaults reproduce PolyNet's historical behaviour: Adam, ReduceLROnPlateau
    (factor 0.9, patience 15, min_lr 1e-8) and an RMSE loss for regression.

    Attributes
    ----------
    optimizer:
        Gradient-descent optimiser (``adam``, ``sgd``, ``rmsprop``,
        ``adadelta``, ``adagrad``). The learning rate comes from the
        architecture block (``LearningRate``) or from HPO.
    scheduler:
        Learning-rate scheduler (``reduce_lr_on_plateau``, ``step_lr``,
        ``multi_step_lr``, ``exponential_lr``). ``reduce_lr_on_plateau``
        monitors the validation loss; the others step once per epoch.
    scheduler_factor:
        Multiplicative learning-rate decay (``gamma``). Used by every scheduler.
    scheduler_patience:
        Epochs without validation improvement before decaying the learning
        rate. ``reduce_lr_on_plateau`` only.
    scheduler_min_lr:
        Lower bound on the learning rate. ``reduce_lr_on_plateau`` only.
    scheduler_step_size:
        Decay period in epochs. ``step_lr`` only.
    scheduler_milestones:
        Epochs at which to decay the learning rate. ``multi_step_lr`` only.
    regression_loss:
        Loss minimised for regression: ``rmse`` (default), ``mse`` or ``mae``.
        Classification always uses cross-entropy.
    """

    optimizer: Optimizer = Field(default=Optimizer.Adam, description="GNN optimiser.")
    scheduler: Scheduler = Field(
        default=Scheduler.ReduceLROnPlateau, description="Learning-rate scheduler."
    )
    scheduler_factor: float = Field(
        default=0.9, gt=0.0, lt=1.0, description="Learning-rate decay factor (gamma)."
    )
    scheduler_patience: int = Field(
        default=15, ge=0, description="Patience in epochs (reduce_lr_on_plateau)."
    )
    scheduler_min_lr: float = Field(
        default=1e-8, ge=0.0, description="Minimum learning rate (reduce_lr_on_plateau)."
    )
    scheduler_step_size: int = Field(default=10, ge=1, description="Decay period (step_lr).")
    scheduler_milestones: list[int] = Field(
        default_factory=lambda: [30, 60, 90], description="Decay epochs (multi_step_lr)."
    )
    regression_loss: RegressionLoss = Field(
        default=RegressionLoss.RMSE, description="Loss minimised for regression targets."
    )

    @model_validator(mode="after")
    def check_milestones(self) -> "GNNOptimisationConfig":
        m = self.scheduler_milestones
        if not m or any(e < 1 for e in m) or m != sorted(set(m)):
            raise ValueError(
                f"scheduler_milestones must be a non-empty, strictly increasing list of "
                f"positive epochs, got {m}."
            )
        return self

    @model_validator(mode="after")
    def warn_on_unused_scheduler_params(self) -> "GNNOptimisationConfig":
        """Warn when a scheduler parameter is changed but the chosen scheduler ignores it."""
        used_by = {
            "scheduler_patience": Scheduler.ReduceLROnPlateau,
            "scheduler_min_lr": Scheduler.ReduceLROnPlateau,
            "scheduler_step_size": Scheduler.StepLR,
            "scheduler_milestones": Scheduler.MultiStepLR,
        }
        for field, scheduler in used_by.items():
            if self.scheduler != scheduler and field in self.model_fields_set:
                warnings.warn(
                    f"{field}={getattr(self, field)!r} has no effect with "
                    f"scheduler='{self.scheduler.value}' (it is only used by "
                    f"'{scheduler.value}').",
                    UserWarning,
                    stacklevel=2,
                )
        return self


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
        If True, the shared GNN hyperparameter values (e.g. number of
        convolutions, embedding dim, pooling, dropout, learning rate, batch
        size) are entered once and applied to every selected architecture.
        If False, each architecture is configured with its own hyperparameter
        values. This controls hyperparameter sharing across architectures; it
        does not affect per-monomer message passing.
    optimisation:
        Optimiser, learning-rate scheduler and regression loss, applied to
        final training and to every HPO trial (``GNNOptimisationConfig``).
        Defaults reproduce Adam + ReduceLROnPlateau + RMSE.
    hyperparameter_optimisation:
        Inherited from ``HyperparamOptimConfig``. When True, Ray Tune samples
        random configurations from the search grid defined in
        ``config/search_grid.py`` before final training.
    """

    train_gnn: bool = Field(
        default=True, description="Master switch to enable or disable GNN training."
    )
    gnn_convolutional_layers: dict[Network, dict] = Field(
        ..., description="GNN architectures to train, keyed by Network enum."
    )
    share_gnn_parameters: bool = Field(
        default=True,
        description="Share GNN hyperparameter values across all selected architectures.",
    )
    epochs: int = Field(default=250, ge=1, description="Number of training epochs per GNN model.")
    hpo_split_strategy: HpoSplitStrategy = Field(
        default=HpoSplitStrategy.CrossValidation,
        description="Split strategy used inside the HPO loop.",
    )
    hpo_val_fraction: float = Field(
        default=0.2, gt=0.0, lt=1.0, description="Val fraction for Holdout / RepeatedHoldout HPO."
    )
    hpo_n_repeats: int = Field(
        default=3, ge=1, description="Number of random splits for RepeatedHoldout HPO."
    )
    optimisation: GNNOptimisationConfig = Field(
        default_factory=GNNOptimisationConfig,
        description="Optimiser, scheduler and loss (training and HPO).",
    )

    @model_validator(mode="after")
    def validate_hpo_search_grid(self) -> "TrainGNNConfig":
        """
        Check ``hpo_search_grid``: keys are selected architectures or ``shared``,
        parameters are in that architecture's default grid, candidates are
        non-empty lists. Warn when a grid can never be used.
        """
        selected = {net.value: net for net in self.gnn_convolutional_layers}
        for key, params in self.hpo_search_grid.items():
            where = f"gnn_training.hpo_search_grid.{key}"
            if key == SHARED_GNN_GRID_KEY:
                allowed = shared_gnn_grid_parameters()
            elif key in selected:
                allowed = gnn_grid_parameters(selected[key])
            else:
                raise ValueError(
                    f"gnn_training.hpo_search_grid has an entry for '{key}', which is not a "
                    f"selected architecture. Use one of {sorted(selected)} or "
                    f"'{SHARED_GNN_GRID_KEY}'."
                )
            check_grid_parameters(where, params, allowed, RESERVED_GNN_GRID_KEYS)
            for strength in params.get(TrainingParam.AsymmetricLossStrength, []):
                if strength is not None and not 0.0 <= strength <= 1.0:
                    raise ValueError(
                        f"{where}.{TrainingParam.AsymmetricLossStrength.value} candidates must "
                        f"be null or between 0 and 1, got {strength!r}."
                    )

            if key in selected and self.gnn_convolutional_layers[selected[key]]:
                warnings.warn(
                    f"{where} has no effect: '{key}' has explicit hyperparameters, so HPO "
                    "does not run for it. Leave its gnn_convolutional_layers block empty "
                    "({}) to search it.",
                    UserWarning,
                    stacklevel=2,
                )
        if SHARED_GNN_GRID_KEY in self.hpo_search_grid and all(
            self.gnn_convolutional_layers.values()
        ):
            warnings.warn(
                f"gnn_training.hpo_search_grid.{SHARED_GNN_GRID_KEY} has no effect: every "
                "architecture has explicit hyperparameters, so HPO does not run.",
                UserWarning,
                stacklevel=2,
            )
        return self

    @model_validator(mode="after")
    def check_architecture_parameters(self) -> "TrainGNNConfig":
        """Reject unknown parameters in ``gnn_convolutional_layers`` (e.g. typos)."""
        # Deferred import: the model registry imports torch.
        from polynet.factories.network import architecture_parameters

        # The seed is set by PolyNet for every split.
        training_params = {str(p) for p in TrainingParam} - {str(TrainingParam.Seed)}
        for network, params in self.gnn_convolutional_layers.items():
            allowed = architecture_parameters(network) | training_params
            unknown = sorted(str(p) for p in (params or {}) if str(p) not in allowed)
            if unknown:
                raise ValueError(
                    f"gnn_convolutional_layers.{network.value}: unknown parameter(s) "
                    f"{unknown}. Allowed: {sorted(allowed)}."
                )
        return self

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
        Inherited from ``HyperparamOptimConfig``. When True, a randomised
        search (``RandomizedSearchCV``, 30 configurations, ``hpo_n_folds``-fold
        shuffled CV) is run over the search grid defined in
        ``config/search_grid.py`` for each selected model.
    hpo_n_folds:
        Inherited from ``HyperparamOptimConfig`` (shared with GNN training).
        Number of cross-validation folds used to score each configuration
        (default 5, minimum 2).
    """

    train_tml: bool = Field(
        default=False, description="Master switch to enable or disable traditional ML training."
    )
    selected_models: dict[TraditionalMLModel, dict] | None = Field(
        default=None,
        description="Fixed hyperparameters per model. Overrides defaults, not the search grid.",
    )

    hpo_num_samples: int = Field(
        default=30,
        ge=1,
        description="Configurations sampled by RandomizedSearchCV (n_iter) per HPO run.",
    )

    @model_validator(mode="after")
    def validate_hpo_search_grid(self) -> "TrainTMLConfig":
        """
        Check ``hpo_search_grid``: keys are selected models, parameters are
        constructor parameters of that model's estimator(s), candidates are
        non-empty lists. Warn when a grid can never be used.
        """
        # Deferred import: polynet.training imports this schema module.
        from polynet.training.tml import _TML_REGISTRY

        selected = {m.value: m for m in (self.selected_models or {})}
        for key, params in self.hpo_search_grid.items():
            where = f"tml_models.hpo_search_grid.{key}"
            if key not in selected:
                raise ValueError(
                    f"tml_models.hpo_search_grid has an entry for '{key}', which is not a "
                    f"selected model. Use one of {sorted(selected)}."
                )
            model = selected[key]
            # Classifier and regressor may differ; accept parameters of either.
            allowed = set().union(
                *(cls().get_params() for (m, _), cls in _TML_REGISTRY.items() if m == model)
            )
            check_grid_parameters(where, params, allowed, RESERVED_TML_GRID_KEYS)

            if self.selected_models[model]:
                warnings.warn(
                    f"{where} has no effect: '{key}' has explicit hyperparameters, so HPO "
                    "does not run for it. Leave its selected_models block empty ({}) to "
                    "search it.",
                    UserWarning,
                    stacklevel=2,
                )
        return self

    @model_validator(mode="after")
    def models_required_when_training(self) -> "TrainTMLConfig":
        if self.train_tml and not self.selected_models:
            raise ValueError(
                "train_tml is True but selected_models is empty. "
                "Provide at least one TraditionalMLModel to train."
            )
        return self
