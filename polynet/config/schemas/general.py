"""
polynet.config.schemas.general
================================
Pydantic schema for experiment-level settings: name, path to store results, and random seed
"""

from pydantic import Field

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

    name: str = Field(..., description="Experiment name.")
    output_dir: str = Field(default="results/", description="Path to store the results.")
    random_seed: int = Field(
        default=42, description="Random seed for reproducibility of experiments."
    )
