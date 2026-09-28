"""
polynet.config.schemas.split_data
================================
Pydantic schema for data splitting settings settings: splitting, reproducibility,
and bootstrap configuration.
"""

from pydantic import Field, model_validator

from polynet.config.enums import SplitMethod, SplitType
from polynet.config.schemas.base import PolynetBaseModel


class SplitConfig(PolynetBaseModel):
    """
    Data splitting configuration covering data splitting strategy,
    random seed, and iteration settings.

    Attributes
    ----------
    split_type:
        The overall splitting strategy. Only ``train_val_test`` (repeated
        random train/validation/test splits) is implemented; the other
        ``SplitType`` values are reserved for future work and rejected here.
    split_method:
        How samples are assigned to splits — randomly or stratified by
        the target variable (stratified is recommended for classification).
    train_set_balance:
        Desired proportion of the minority class after undersampling the
        majority class (e.g. ``0.5`` for 50/50). Must be in (0, 1]; ``None``
        or ``1.0`` disables balancing. Only relevant for binary
        classification. Applied to the non-test data before the validation
        split, so training and validation are balanced and the test set keeps
        the original class distribution (ACS Appl. Mater. Interfaces 2023, 15 (11), 14155–14163).
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
        description="Minority-class proportion of the training and validation sets after "
        "balancing (the test set keeps the original distribution).",
    )
    test_ratio: float = Field(..., gt=0.0, lt=1.0, description="Fraction of data for the test set.")
    val_ratio: float = Field(
        default=0.1, gt=0.0, lt=1.0, description="Fraction of data for the validation set."
    )
    n_bootstrap_iterations: int = Field(
        default=1, ge=1, description="Number of bootstrap repetitions."
    )

    @model_validator(mode="after")
    def only_train_val_test_is_implemented(self) -> "SplitConfig":
        if self.split_type != SplitType.TrainValTest:
            raise ValueError(
                f"split_type '{self.split_type.value}' is not implemented yet. Only "
                f"'{SplitType.TrainValTest.value}' is available: repeated random "
                "train/validation/test splits, controlled by test_ratio, val_ratio and "
                "n_bootstrap_iterations."
            )
        return self

    @model_validator(mode="after")
    def ratios_leave_room_for_training(self) -> "SplitConfig":
        total_held_out = self.test_ratio + self.val_ratio
        if total_held_out >= 1.0:
            raise ValueError(
                f"test_ratio ({self.test_ratio}) + val_ratio ({self.val_ratio}) = "
                f"{total_held_out:.2f}, which leaves no data for training. "
                "Their sum must be less than 1.0."
            )
        return self
