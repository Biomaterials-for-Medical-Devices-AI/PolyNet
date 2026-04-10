"""
polynet.config.schemas.representation
=======================================
Pydantic schema for molecular / polymer representation options.
"""

from pydantic import Field, model_validator

from polynet.config.enums import (
    AtomFeature,
    BondFeature,
    DescriptorMergingMethod,
    MolecularDescriptor,
)
from polynet.config.schemas.base import PolynetBaseModel


class RepresentationConfig(PolynetBaseModel):
    """
    Configuration for how polymers are represented as inputs to the model.

    Supports three representation families which can be combined:

    1. **Graph-based** — atoms as nodes, bonds as edges, with configurable
       node (``node_features``) and edge (``edge_features``) feature sets.
    2. **Descriptor-based** — RDKit molecular descriptors, descriptors loaded
       from an external DataFrame, or both.
    3. **Fingerprint-based** — PolyBERT latent-space fingerprints.

    When a polymer is represented by multiple SMILES strings (e.g. a blend),
    ``smiles_merge_approach`` controls how per-monomer representations are
    combined into a single polymer-level vector.

    Attributes
    ----------
    smiles_merge_approach:
        Ordered list of merging strategies applied when a polymer is defined
        by more than one SMILES string. Applied in the order provided.
    node_features:
        Mapping from ``AtomFeature`` enum members to their feature
        configuration dictionaries. Only used for graph representations.
    edge_features:
        Mapping from ``BondFeature`` enum members to their feature
        configuration dictionaries. Only used for graph representations.
    molecular_descriptors:
        Mapping from ``MolecularDescriptor`` to the descriptor configuration.
        Supported keys: ``rdkit`` (list of descriptor names), ``dataframe``
        (list of DataFrame column names), ``polybert`` (bool), etc.
        An empty dict disables descriptor-based representation.
    rdkit_independent:
        If True, RDKit descriptors are used as an independent representation
        (not merged with graph features). If False, they augment node features.
    df_descriptors_independent:
        Analogous to ``rdkit_independent`` for DataFrame descriptors.
    mix_rdkit_df_descriptors:
        If True, RDKit and DataFrame descriptors are concatenated into a
        single descriptor vector before passing to the model.
    weights_col:
        Optional mapping from SMILES column name to the column name that
        holds the monomer weight fraction, used for weighted merging.
    """

    smiles_merge_approach: DescriptorMergingMethod = Field(
        ..., min_length=1, description="Ordered merging strategies for multi-SMILES polymers."
    )
    node_features: dict[AtomFeature, dict] = Field(
        default_factory=dict, description="Atom feature configuration for graph construction."
    )
    edge_features: dict[BondFeature, dict] = Field(
        default_factory=dict, description="Bond feature configuration for graph construction."
    )
    molecular_descriptors: dict[MolecularDescriptor, list[str] | dict | bool | str] = Field(
        default_factory=dict, description="Descriptor types and their selected descriptor names."
    )
    rdkit_independent: bool = Field(
        default=True, description="Treat RDKit descriptors as an independent representation."
    )
    df_descriptors_independent: bool | None = Field(
        default=None, description="Treat DataFrame descriptors as an independent representation."
    )
    mix_rdkit_df_descriptors: bool = Field(
        default=False, description="Concatenate RDKit and DataFrame descriptors into one vector."
    )
    weights_col: dict[str, str] | None = Field(
        default=None, description="Mapping from SMILES column to weight-fraction column."
    )

    @model_validator(mode="after")
    def weighted_merge_requires_weights_col(self) -> "RepresentationConfig":
        if (
            DescriptorMergingMethod.WeightedAverage in self.smiles_merge_approach
            and self.weights_col is None
        ):
            raise ValueError(
                "smiles_merge_approach includes 'weighted_average' but "
                "weights_col is not set. Provide a mapping from SMILES column "
                "to weight-fraction column."
            )
        return self

    @model_validator(mode="after")
    def df_descriptors_require_columns(self) -> "RepresentationConfig":
        df_value = self.molecular_descriptors.get(MolecularDescriptor.DataFrame)
        if df_value is not None and (not isinstance(df_value, list) or len(df_value) == 0):
            raise ValueError(
                "molecular_descriptors includes 'dataframe' but no column names are provided. "
                "Set molecular_descriptors[dataframe] to a non-empty list of column names."
            )
        return self

    @model_validator(mode="after")
    def at_least_one_representation(self) -> "RepresentationConfig":
        has_graph = bool(self.node_features)
        has_descriptors = bool(self.molecular_descriptors)
        if not any([has_graph, has_descriptors]):
            raise ValueError(
                "At least one representation must be configured: set node_features "
                "for graph-based, or molecular_descriptors for descriptor-based representations."
            )
        return self
