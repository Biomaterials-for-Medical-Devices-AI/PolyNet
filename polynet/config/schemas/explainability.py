"""
polynet.config.schemas.explainability
=======================================
Pydantic schema for the explainability stage configuration.

YAML layout::

    explainability:
      enabled: true
      algorithm: "chemistry_masking"

      # Which GNN architectures to explain.
      # Matches keys in gnn_training.gnn_convolutional_layers (e.g. GCN, GAT).
      # "all" explains every trained architecture.
      models: "all"

      # Which bootstrap iterations to explain (0-based indices matching training).
      # "all" explains every trained iteration.
      # [0, 2] explains only iterations 0 and 2.
      bootstraps: "all"

      # Molecular fragmentation strategy.
      fragmentation: "brics"           # brics | recap | murcko_scaffold | functional_groups

      # Data split to draw molecules from when explain_mol_ids is not set.
      explain_set: "test"              # train | validation | test | all

      # Normalisation applied before plotting.
      normalisation: "per_model"       # local | global | per_model | no_normalisation

      # For classification: which class to attribute (null for regression).
      target_class: null

      # Attribution distribution plot style.
      plot_type: "ridge"               # ridge | bar | strip

      # Show only the top-N and bottom-N fragments by mean attribution.
      # null shows all fragments.
      top_n: 10

      # Molecule IDs to generate per-molecule heatmaps and attribution tables for.
      # When null (default), the local explanation step is skipped entirely.
      # local_explain_mol_ids:
      #   - "poly_0001"
"""

from __future__ import annotations

from typing import Literal, Union

from pydantic import field_validator

from polynet.config.enums import (
    AttributionPlotType,
    ExplainAlgorithm,
    FragmentationMethod,
    ImportanceNormalisationMethod,
)
from polynet.config.schemas.base import PolynetBaseModel


class ExplainabilityConfig(PolynetBaseModel):
    """
    Configuration for the pipeline explainability stage.

    Attributes
    ----------
    enabled:
        Master switch — set to False to skip the stage entirely.
    algorithm:
        Attribution algorithm. Only ``chemistry_masking`` is supported.
    models:
        GNN architecture names to explain (must match
        ``gnn_training.gnn_convolutional_layers`` keys, e.g. ``GCN``).
        Use ``"all"`` to explain every trained architecture.
    bootstraps:
        Bootstrap iteration indices to explain (0-based, matching the
        training loop).  Use ``"all"`` for all iterations, or supply a
        list such as ``[0, 2]``.  Combined with ``models`` as a
        cross-product — e.g. ``models: [GCN, GAT]`` × ``bootstraps: [0, 2]``
        explains GCN_0, GCN_2, GAT_0, GAT_2.
    fragmentation:
        Fragmentation strategy used to decompose monomers into fragments.
    explain_set:
        Which data split to draw molecules from for the global distribution
        plot.  ``"all"`` takes the union of train, validation, and test sets
        across the selected bootstrap iterations.
    normalisation:
        How to normalise fragment attribution scores before plotting.
    target_class:
        For classification: the class index whose predicted probability is
        attributed. Pass ``null`` / ``None`` for regression.
    plot_type:
        Visual style for the global attribution distribution plot.
    top_n:
        Number of top and bottom fragments shown in the global plot.
        ``null`` shows every fragment.
    local_explain_mol_ids:
        Molecule IDs to generate per-molecule heatmaps and attribution tables
        for.  When ``None``, the local explanation step is skipped entirely.
    """

    enabled: bool = False
    algorithm: ExplainAlgorithm = ExplainAlgorithm.ChemistryMasking
    models: Union[list[str], Literal["all"]] = "all"
    bootstraps: Union[list[int], Literal["all"]] = "all"
    fragmentation: FragmentationMethod = FragmentationMethod.BRICS
    explain_set: Literal["train", "validation", "test", "all"] = "test"
    normalisation: ImportanceNormalisationMethod = ImportanceNormalisationMethod.PerModel
    target_class: int | None = None
    plot_type: AttributionPlotType = AttributionPlotType.Ridge
    top_n: int | None = 10
    local_explain_mol_ids: list[str] | None = None

    @field_validator("algorithm")
    @classmethod
    def _only_masking(cls, v: ExplainAlgorithm) -> ExplainAlgorithm:
        if v != ExplainAlgorithm.ChemistryMasking:
            raise ValueError(
                f"Only 'chemistry_masking' is supported as the explainability algorithm. "
                f"Got '{v}'."
            )
        return v
