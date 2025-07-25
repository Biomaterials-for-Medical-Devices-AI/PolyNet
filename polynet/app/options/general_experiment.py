from dataclasses import dataclass


@dataclass
class GeneralConfigOptions:
    """Data split options for the application."""

    split_type: str
    split_method: str
    train_set_balance: float
    test_ratio: str
    val_ratio: str
    random_seed: int
    n_bootstrap_iterations: int
