from __future__ import annotations

from pydantic import Field

from polynet.config.enums import TargetTransformDescriptor
from polynet.config.schemas.base import PolynetBaseModel


class TargetTransformConfig(PolynetBaseModel):
    """
    Configuration for target variable scaling in regression experiments.

    Attributes
    ----------
    strategy:
        Scaling strategy to apply to the target variable before training.
        Use ``TargetTransformDescriptor.NoTransformation`` (the default) to
        skip scaling entirely.

        The scaler is fitted exclusively on the training set and applied to
        validation and test sets to prevent data leakage.  Before metrics
        and plots are computed, predictions are inverse-transformed back to
        the original target range.

        Available strategies:

        - ``no_transformation`` — identity, no scaling applied
        - ``standard_scaler`` — zero mean, unit variance
        - ``min_max_scaler`` — scales to [0, 1]
        - ``robust_scaler`` — resistant to outliers (uses IQR)
        - ``log10`` — base-10 logarithm; requires all training targets > 0
        - ``log1p`` — natural log of (1 + y); requires all training targets > -1

        Example YAML::

            target_transform:
              strategy: standard_scaler
    """

    strategy: TargetTransformDescriptor = Field(
        default=TargetTransformDescriptor.NoTransformation,
        description="Scaling strategy for the regression target variable.",
    )
