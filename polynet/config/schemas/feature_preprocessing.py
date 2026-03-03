from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from polynet.config.enums import FeatureSelection, TransformDescriptor
from polynet.config.schemas.base import PolynetBaseModel


class FeatureTransformConfig(PolynetBaseModel):
    """
    Configuration for feature normalization and feature selection.

    Attributes
    ----------
    scaler:
        Feature normalization strategy. Use ``TransformDescriptor.NoTransformation``
        to disable scaling.
    selectors:
        Mapping of feature selection step → parameter dict. Steps are applied
        sequentially in the mapping order (insertion order is preserved in Python 3.7+).

        Supported steps and parameters:
        - ``FeatureSelection.Variance``:
            ``{"threshold": float}``  (variance <= threshold is removed)
        - ``FeatureSelection.Correlation``:
            ``{"threshold": float}``  (0 < threshold < 1, abs(corr) >= threshold removed)

        Example::
            selectors:
              Variance:
                threshold: 0.0
              Correlation:
                threshold: 0.95

    random_state:
        Random seed for reproducibility (reserved for future stochastic selectors).
    """

    scaler: TransformDescriptor = Field(
        default=TransformDescriptor.NoTransformation, description="Scaling strategy."
    )

    # Ordered mapping: FeatureSelection -> params dict
    selectors: dict[FeatureSelection, dict[str, Any]] = Field(
        default_factory=dict,
        description="Ordered selection steps applied after scaling, with per-step params.",
    )

    random_state: int = Field(default=42, description="Random seed for reproducibility.")

    @model_validator(mode="after")
    def validate_selectors(self) -> "FeatureTransformConfig":
        """
        Validate selector keys and required parameters.
        """
        for step, params in self.selectors.items():
            if params is None:
                params = {}
                self.selectors[step] = params  # type: ignore[index]

            if not isinstance(params, dict):
                raise ValueError(
                    f"selectors[{step}] must be a dict of parameters, got {type(params).__name__}."
                )

            # Only Variance and Correlation are supported (LASSO removed)
            if step == FeatureSelection.Variance:
                thr = params.get("threshold", 0.0)
                try:
                    thr_f = float(thr)
                except Exception as e:
                    raise ValueError(
                        f"selectors[{step}]['threshold'] must be a float, got {thr!r}."
                    ) from e
                if thr_f < 0.0:
                    raise ValueError(f"selectors[{step}]['threshold'] must be >= 0, got {thr_f}.")
                params["threshold"] = thr_f  # normalize

            elif step == FeatureSelection.Correlation:
                thr = params.get("threshold", 0.95)
                try:
                    thr_f = float(thr)
                except Exception as e:
                    raise ValueError(
                        f"selectors[{step}]['threshold'] must be a float, got {thr!r}."
                    ) from e
                if not (0.0 < thr_f < 1.0):
                    raise ValueError(
                        f"selectors[{step}]['threshold'] must be in (0, 1), got {thr_f}."
                    )
                params["threshold"] = thr_f  # normalize

            else:
                raise ValueError(
                    f"Unsupported feature selection step: {step!r}. "
                    f"Supported: {FeatureSelection.Variance}, {FeatureSelection.Correlation}."
                )

        return self
