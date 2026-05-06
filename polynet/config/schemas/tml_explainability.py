"""
polynet.config.schemas.tml_explainability
==========================================
Pydantic schema for the TML SHAP explainability stage configuration.

YAML layout::

    tml_explainability:
      enabled: false

      # TML model types to explain, e.g. ["RandomForest", "XGBoost"].
      # "all" explains every trained TML model type.
      models: "all"

      # Molecular descriptor representations to explain, e.g. ["morgan", "rdkit"].
      # "all" explains every trained descriptor.
      representations: "all"

      # Bootstrap iteration indices to explain (0-based).
      # "all" explains every trained iteration.
      bootstraps: "all"

      # Data split to draw samples from for the global distribution plot.
      explain_set: "test"              # train | validation | test | all

      # Normalisation applied to SHAP scores before plotting.
      normalisation: "per_model"       # local | global | per_model | no_normalisation

      # For classification: which class index to attribute (null for regression).
      target_class: null

      # Attribution distribution plot style.
      plot_type: "ridge"               # ridge | bar | strip

      # Show only the top-N features by mean |SHAP|.
      # null shows all features.
      top_n: 10

      # Sample IDs to generate per-instance waterfall/force plots for.
      # When null (default), the local explanation step is skipped entirely.
      local_explain_sample_ids: null
      # local_explain_sample_ids:
      #   - "poly_0001"

      # Plot style for local (per-instance) explanations.
      local_plot_type: "waterfall"     # waterfall | force | bar
"""

from __future__ import annotations

from typing import Literal, Union

from polynet.config.enums import AttributionPlotType, ImportanceNormalisationMethod
from polynet.config.schemas.base import PolynetBaseModel


class TMLExplainabilityConfig(PolynetBaseModel):
    """
    Configuration for the TML SHAP explainability pipeline stage.

    Attributes
    ----------
    enabled:
        Master switch — set to False to skip the stage entirely.
    models:
        TML model type names to explain (e.g. ``["RandomForest", "XGBoost"]``).
        Use ``"all"`` to explain every trained TML model type.
    representations:
        Descriptor names to explain (e.g. ``["morgan", "rdkit"]``).
        Use ``"all"`` to explain every trained descriptor.
    bootstraps:
        Bootstrap iteration indices to explain (0-based, matching training).
        Use ``"all"`` for all iterations, or supply a list such as ``[0, 2]``.
    explain_set:
        Which data split to draw samples from for the global distribution
        plot.  ``"all"`` takes the union of train, validation, and test sets
        across the selected bootstrap iterations.
    normalisation:
        How to normalise SHAP scores before plotting.
    target_class:
        For classification: the class index whose SHAP values are attributed.
        Pass ``null`` / ``None`` for regression.
    plot_type:
        Visual style for the global attribution distribution plot.
    top_n:
        Number of top features shown in the global plot by mean |SHAP|.
        ``null`` shows every feature.
    local_explain_sample_ids:
        Sample IDs to generate per-instance SHAP plots for.
        When ``None``, the local explanation step is skipped entirely.
    local_plot_type:
        Visual style for per-instance SHAP plots.
        ``"waterfall"`` shows a waterfall chart, ``"force"`` a force plot,
        ``"bar"`` a simple horizontal bar chart.
    """

    enabled: bool = False
    models: Union[list[str], Literal["all"]] = "all"
    representations: Union[list[str], Literal["all"]] = "all"
    bootstraps: Union[list[int], Literal["all"]] = "all"
    explain_set: Literal["train", "validation", "test", "all"] = "test"
    normalisation: ImportanceNormalisationMethod = ImportanceNormalisationMethod.PerModel
    target_class: int | None = None
    plot_type: AttributionPlotType = AttributionPlotType.Ridge
    top_n: int | None = 10
    local_explain_sample_ids: list[str] | None = None
    local_plot_type: Literal["waterfall", "force", "bar"] = "waterfall"
