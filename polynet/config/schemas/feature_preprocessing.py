from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from polynet.config.schemas.base import PolynetBaseModel
from polynet.config.enums import TransformDescriptor, FeatureSelection


class FeatureTransformConfig(PolynetBaseModel):
    """
    Configuration for feature normalization and feature selection.

    Attributes
    ----------
    scaler:
        Feature normalization strategy. Use "none" to disable scaling.
    selectors:
        Ordered list of feature selection steps to apply after scaling.
        Steps are applied sequentially, each reducing the feature set.
        Allowed values: "variance", "correlation", "lasso".

        Example::
            selectors: ["variance", "correlation", "lasso"]

    variance_threshold:
        Threshold for VarianceThreshold selection. Features with variance
        <= this value are removed.
    corr_threshold:
        Absolute correlation threshold for correlation-based selection.
        Features with abs(corr) >= threshold are removed (greedy upper-triangular rule).
    lasso_alpha:
        Regularization strength for LASSO regression selection (regression tasks).
    lasso_C:
        Inverse regularization strength for L1 logistic regression selection (classification tasks).
    lasso_max_iter:
        Max iterations for the L1 solver.
    random_state:
        Random seed used by stochastic selectors (e.g., some solvers).
    """

    scaler: TransformDescriptor = Field(
        default=TransformDescriptor.NoTransformation, description="Scaling strategy."
    )
    selectors: list[FeatureSelection] = Field(
        default_factory=list, description="Ordered selection steps applied after scaling."
    )

    variance_threshold: float = Field(default=0.0, ge=0.0, description="VarianceThreshold cutoff.")
    corr_threshold: float = Field(
        default=0.95, gt=0.0, lt=1.0, description="Correlation cutoff in (0, 1)."
    )

    # LASSO-related
    lasso_alpha: float = Field(default=1e-3, gt=0.0, description="Alpha for LASSO (regression).")
    lasso_C: float = Field(
        default=1.0, gt=0.0, description="C for L1 logistic regression (classification)."
    )
    lasso_max_iter: int = Field(
        default=10_000, ge=1, description="Max solver iterations for LASSO/L1 LR."
    )
    random_state: int = Field(default=42, description="Random seed for reproducibility.")

    @model_validator(mode="after")
    def selectors_unique_and_ordered(self) -> "FeatureTransformConfig":
        # Disallow duplicates because repeating the same selector is almost always accidental.
        if len(self.selectors) != len(set(self.selectors)):
            raise ValueError(
                f"selectors contains duplicates: {self.selectors}. "
                "Each selection step should appear at most once."
            )
        return self

    @model_validator(mode="after")
    def lasso_requires_scaling_recommended(self) -> "FeatureTransformConfig":
        # Not strictly required, but a helpful guardrail for users.
        if "lasso" in self.selectors and self.scaler == "none":
            # You could make this a warning instead if you prefer.
            raise ValueError(
                "selectors includes 'lasso' but scaler='none'. "
                "LASSO feature selection is usually unstable without scaling. "
                "Use scaler='standard' (recommended) or another scaler."
            )
        return self
