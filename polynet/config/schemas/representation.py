"""
polynet.config.schemas.representation
=======================================
Pydantic schema for molecular / polymer representation options.
"""

from pydantic import Field, field_validator, model_validator

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
        Mapping from ``MolecularDescriptor`` to the list of descriptor names
        to compute. An empty dict disables descriptor-based representation.
    rdkit_descriptors:
        Explicit list of RDKit descriptor names to compute. If None, all
        descriptors available in the ``rdkit`` entry of ``molecular_descriptors``
        are used.
    df_descriptors:
        Column names from an external DataFrame to use as descriptors.
        Requires the DataFrame to be provided at featurization time.
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
    polybert_fp:
        Whether to compute PolyBERT fingerprints. Requires the PolyBERT
        model to be available.
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
    molecular_descriptors: dict[MolecularDescriptor, list[str]] = Field(
        default_factory=dict, description="Descriptor types and their selected descriptor names."
    )
    rdkit_descriptors: list[str] | None = Field(
        default=None, description="Explicit RDKit descriptor names. None uses all available."
    )
    df_descriptors: list[str] | None = Field(
        default=None, description="Column names from an external DataFrame to use as descriptors."
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
    polybert_fp: bool = Field(default=False, description="Compute PolyBERT fingerprints.")

    @field_validator("rdkit_descriptors", "df_descriptors", mode="before")
    @classmethod
    def empty_list_to_none(cls, v):
        """Treat an empty list the same as None — no descriptors selected."""
        if isinstance(v, list) and len(v) == 0:
            return None
        return v

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
        uses_df = MolecularDescriptor.DataFrame in self.molecular_descriptors
        if uses_df and self.df_descriptors is None:
            raise ValueError(
                "molecular_descriptors includes 'dataframe' but df_descriptors "
                "is not set. Provide the column names to use from the external DataFrame."
            )
        return self

    @model_validator(mode="after")
    def at_least_one_representation(self) -> "RepresentationConfig":
        has_graph = bool(self.node_features)
        has_descriptors = bool(self.molecular_descriptors)
        has_polybert = self.polybert_fp
        if not any([has_graph, has_descriptors, has_polybert]):
            raise ValueError(
                "At least one representation must be configured: set node_features "
                "for graph-based, molecular_descriptors for descriptor-based, "
                "or polybert_fp=True for PolyBERT fingerprints."
            )
        return self
