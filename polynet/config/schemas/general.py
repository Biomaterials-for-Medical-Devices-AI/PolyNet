"""
polynet.config.schemas.general
================================
Pydantic schema for experiment-level settings: splitting, reproducibility,
and bootstrap configuration.
"""

from pydantic import Field, model_validator

from polynet.config.enums import SplitMethod, SplitType
from polynet.config.schemas.base import PolynetBaseModel


class GeneralConfig(PolynetBaseModel):
    """
    Experiment-level configuration covering data splitting strategy,
    random seed, and iteration settings.

    Attributes
    ----------
    split_type:
        The overall cross-validation or hold-out strategy to use.
    split_method:
        How samples are assigned to splits — randomly or stratified by
        the target variable (stratified is recommended for classification).
    train_set_balance:
        Fraction of the training set retained after any balancing step.
        Must be in (0, 1]. Only relevant when class imbalance correction
        is applied.
    test_ratio:
        Fraction of the full dataset reserved for the test set.
        Must be in (0, 1).
    val_ratio:
        Fraction of the full dataset reserved for the validation set.
        Must be in (0, 1). Only used when ``split_type`` includes a
        validation split (e.g. TrainValTest).
    random_seed:
        Global random seed for reproducibility across all stochastic steps.
    n_bootstrap_iterations:
        Number of bootstrap iterations. Only used when ``split_type`` is
        ``TrainValTest`` or ``TrainTest``.
    """

    split_type: SplitType = Field(..., description="Overall data splitting strategy.")
    split_method: SplitMethod = Field(
        default=SplitMethod.Random, description="Sample assignment method."
    )
    train_set_balance: float | None = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Ratio of labels in binary classification after balancing.",
    )
    test_ratio: float = Field(..., gt=0.0, lt=1.0, description="Fraction of data for the test set.")
    val_ratio: float = Field(
        default=0.1, gt=0.0, lt=1.0, description="Fraction of data for the validation set."
    )
    random_seed: int = Field(default=42, ge=0, description="Global random seed.")
    n_bootstrap_iterations: int = Field(
        default=1, ge=1, description="Number of bootstrap repetitions."
    )

    @model_validator(mode="after")
    def ratios_leave_room_for_training(self) -> "GeneralConfig":
        total_held_out = self.test_ratio + self.val_ratio
        if total_held_out >= 1.0:
            raise ValueError(
                f"test_ratio ({self.test_ratio}) + val_ratio ({self.val_ratio}) = "
                f"{total_held_out:.2f}, which leaves no data for training. "
                "Their sum must be less than 1.0."
            )
        return self

    @model_validator(mode="after")
    def val_ratio_only_relevant_for_train_val_test(self) -> "GeneralConfig":
        no_val_types = {
            SplitType.TrainTest,
            SplitType.CrossValidation,
            SplitType.NestedCrossValidation,
            SplitType.LeaveOneOut,
        }
        if self.split_type in no_val_types and self.val_ratio != 0.1:
            # Warn rather than error — user may have copy-pasted a full config
            import warnings

            warnings.warn(
                f"val_ratio is set to {self.val_ratio} but split_type is "
                f"'{self.split_type}', which does not use a validation set. "
                "val_ratio will be ignored.",
                UserWarning,
                stacklevel=2,
            )
        return self

    @model_validator(mode="after")
    def bootstrap_only_relevant_for_holdout_splits(self) -> "GeneralConfig":
        iterative_types = {
            SplitType.CrossValidation,
            SplitType.NestedCrossValidation,
            SplitType.LeaveOneOut,
        }
        if self.split_type in iterative_types and self.n_bootstrap_iterations > 1:
            import warnings

            warnings.warn(
                f"n_bootstrap_iterations is {self.n_bootstrap_iterations} but "
                f"split_type is '{self.split_type}', which does not use bootstrapping. "
                "n_bootstrap_iterations will be ignored.",
                UserWarning,
                stacklevel=2,
            )
        return self
