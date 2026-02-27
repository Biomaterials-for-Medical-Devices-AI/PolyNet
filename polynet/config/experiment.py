"""
polynet.config.experiment
==========================
Top-level experiment configuration that composes all pipeline sub-configs
into a single validated object.

Usage
-----
The ``ExperimentConfig`` is the primary entry point for the full pipeline.
Both the Streamlit app and the YAML-based script runner produce an
``ExperimentConfig`` before handing off to the ``PipelineRunner``::

    from polynet.config.experiment import ExperimentConfig

    # From a dict (e.g. parsed from YAML)
    cfg = ExperimentConfig(**config_dict)

    # Access sub-configs
    cfg.data.target_variable_col
    cfg.general.split_type
    cfg.train_gnn.gnn_convolutional_layers

Independent stage usage
-----------------------
Sub-configs can be used independently of ``ExperimentConfig`` when running
individual pipeline stages::

    from polynet.config.schemas import DataConfig, TrainGNNConfig

    data_cfg = DataConfig(
        data_name="my_dataset",
        data_path="data/polymers.csv",
        smiles_cols=["SMILES"],
        target_variable_col="Tg",
        problem_type=ProblemType.Regression,
    )

Cross-field validation
----------------------
``ExperimentConfig`` enforces consistency between sub-configs that
individual schemas cannot validate in isolation. For example:

- A graph representation requires a GNN training config
- Descriptor-only representations can use TML, GNN, or both
- At least one training config must be active
"""

from __future__ import annotations

from pydantic import Field, model_validator

from polynet.config.schemas.base import PolynetBaseModel
from polynet.config.schemas.data import DataConfig
from polynet.config.schemas.general import GeneralConfig
from polynet.config.schemas.plotting import PlottingConfig
from polynet.config.schemas.representation import RepresentationConfig
from polynet.config.schemas.training import TrainGNNConfig, TrainTMLConfig


class ExperimentConfig(PolynetBaseModel):
    """
    Full experiment configuration composed of all pipeline sub-configs.

    This is the single object that fully specifies a polynet experiment —
    from raw data loading through to model training and plotting. Both the
    Streamlit app and the YAML script runner produce an instance of this
    class before handing off to the ``PipelineRunner``.

    All sub-configs are individually validated by their own Pydantic schemas.
    ``ExperimentConfig`` then applies cross-config validation to catch
    inconsistencies between sub-configs that no individual schema can see.

    Attributes
    ----------
    data:
        Dataset loading and target variable configuration.
    general:
        Experiment-level settings: splitting strategy, random seed,
        and bootstrap / cross-validation settings.
    representation:
        Molecular representation configuration: graph features,
        molecular descriptors, and/or PolyBERT fingerprints.
    plotting:
        Global plot styling and output settings. Optional — if not
        provided, library defaults are used.
    train_gnn:
        GNN training configuration. Required if the representation
        includes graph-based features. Optional otherwise.
    train_tml:
        Traditional ML training configuration. Optional — can be
        combined with GNN training in the same experiment.
    """

    data: DataConfig = Field(..., description="Dataset and target variable configuration.")
    general: GeneralConfig = Field(..., description="Splitting, seed, and iteration settings.")
    representation: RepresentationConfig = Field(
        ..., description="Molecular representation configuration."
    )
    plotting: PlottingConfig = Field(
        default_factory=PlottingConfig,
        description="Plot styling and output settings. Uses library defaults if not provided.",
    )
    train_gnn: TrainGNNConfig | None = Field(
        default=None,
        description="GNN training configuration. Required when using graph-based representation.",
    )
    train_tml: TrainTMLConfig | None = Field(
        default=None, description="Traditional ML training configuration. Optional."
    )

    # ------------------------------------------------------------------
    # Cross-config validators
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def at_least_one_training_config(self) -> "ExperimentConfig":
        """
        Ensure the experiment has at least one active training stage.

        A config with neither GNN nor TML training is a pipeline that
        loads data and builds representations but never trains anything —
        almost certainly a misconfiguration.
        """
        gnn_active = self.train_gnn is not None and self.train_gnn.train_gnn
        tml_active = self.train_tml is not None and self.train_tml.train_tml

        if not gnn_active and not tml_active:
            raise ValueError(
                "At least one training stage must be active. "
                "Provide a TrainGNNConfig with train_gnn=True, a TrainTMLConfig "
                "with train_tml=True, or both."
            )
        return self

    @model_validator(mode="after")
    def graph_representation_requires_gnn(self) -> "ExperimentConfig":
        """
        Enforce that a graph-based representation is paired with a GNN.

        If the representation configures node features (i.e. it produces
        molecular graphs), a GNN training config must be provided and active.
        Training a traditional ML model directly on a graph representation
        is not supported — descriptors or fingerprints must be used instead.
        """
        has_graph_repr = bool(self.representation.node_features)
        gnn_active = self.train_gnn is not None and self.train_gnn.train_gnn

        if has_graph_repr and not gnn_active:
            raise ValueError(
                "representation.node_features is configured (graph-based representation) "
                "but no active GNN training config was provided. "
                "Either add a TrainGNNConfig with train_gnn=True, or remove "
                "node_features from the representation config and use descriptors instead."
            )
        return self

    @model_validator(mode="after")
    def tml_requires_non_graph_representation(self) -> "ExperimentConfig":
        """
        Enforce that TML training is paired with a descriptor or fingerprint
        representation, not a pure graph representation.

        Traditional ML models cannot consume molecular graphs directly —
        they require a fixed-length vector input. If TML is active, at
        least one of: molecular descriptors or PolyBERT fingerprints must
        be configured.
        """
        tml_active = self.train_tml is not None and self.train_tml.train_tml
        if not tml_active:
            return self

        has_descriptors = bool(self.representation.molecular_descriptors)
        has_polybert = self.representation.polybert_fp

        if not has_descriptors and not has_polybert:
            raise ValueError(
                "train_tml is active but the representation config provides no "
                "fixed-length vector inputs for traditional ML models. "
                "Add molecular_descriptors or set polybert_fp=True in the "
                "representation config, or disable train_tml."
            )
        return self

    @model_validator(mode="after")
    def polybert_requires_psmiles(self) -> "ExperimentConfig":
        """
        PolyBERT fingerprints are only meaningful for PSMILES strings.

        PolyBERT was trained on PSMILES notation. Using it with standard
        SMILES will produce unreliable embeddings. Raise a hard error to
        prevent silent quality issues.
        """
        from polynet.config.enums import StringRepresentation

        if (
            self.representation.polybert_fp
            and self.data.string_representation != StringRepresentation.PSMILES
        ):
            raise ValueError(
                "representation.polybert_fp is True but data.string_representation "
                f"is '{self.data.string_representation}'. PolyBERT requires PSMILES "
                "notation. Set data.string_representation to StringRepresentation.PSMILES "
                "or disable polybert_fp."
            )
        return self

    @model_validator(mode="after")
    def classification_num_classes_consistent(self) -> "ExperimentConfig":
        """
        For classification, warn if class_names is not provided.

        This is a soft check — class_names is optional, but its absence
        means plots and result tables will use raw class indices (0, 1, ...)
        rather than meaningful labels.
        """
        import warnings

        from polynet.config.enums import ProblemType

        if self.data.problem_type == ProblemType.Classification and self.data.class_names is None:
            warnings.warn(
                "data.problem_type is 'classification' but data.class_names is not set. "
                "Plots and result tables will use raw class indices. "
                "Consider providing class_names for more readable outputs.",
                UserWarning,
                stacklevel=2,
            )
        return self

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def is_classification(self) -> bool:
        """Shorthand to check if this is a classification experiment."""
        from polynet.config.enums import ProblemType

        return self.data.problem_type == ProblemType.Classification

    @property
    def is_regression(self) -> bool:
        """Shorthand to check if this is a regression experiment."""
        from polynet.config.enums import ProblemType

        return self.data.problem_type == ProblemType.Regression

    @property
    def runs_gnn(self) -> bool:
        """True if GNN training is active in this experiment."""
        return self.train_gnn is not None and self.train_gnn.train_gnn

    @property
    def runs_tml(self) -> bool:
        """True if traditional ML training is active in this experiment."""
        return self.train_tml is not None and self.train_tml.train_tml

    @property
    def uses_graph_representation(self) -> bool:
        """True if the representation includes graph-based (node/edge) features."""
        return bool(self.representation.node_features)

    @property
    def uses_descriptors(self) -> bool:
        """True if the representation includes molecular descriptors."""
        return bool(self.representation.molecular_descriptors)

    @property
    def target_name(self) -> str:
        """
        Human-readable target variable name for use in plots and results.

        Falls back to the raw column name if no display name was provided.
        """
        return self.data.target_variable_name or self.data.target_variable_col
