"""
polynet.config.constants
========================
Internal constants used throughout the polynet pipeline.

These are **not** user-facing options — they are fixed labels for dataframe
columns, result dictionaries, and other internal data structures. They are
kept separate from ``enums.py`` because they are never selected by a user
via the app or a YAML config.
"""

# ---------------------------------------------------------------------------
# Results dataframe column names
# ---------------------------------------------------------------------------


class ResultColumn:
    """
    Standard column names used in results DataFrames produced by the pipeline.

    Using a class of plain string constants (rather than an Enum) is
    intentional: these values are used as DataFrame column keys and
    dictionary keys throughout the codebase, and string constants are
    the most ergonomic for that purpose.
    """

    LABEL: str = "True"
    PREDICTED: str = "Predicted"
    INDEX: str = "Index"
    SET: str = "Set"
    SCORE: str = "Score"
    MODEL: str = "Model"
    LOADERS: str = "Loaders"


# ---------------------------------------------------------------------------
# Atom / bond feature descriptor dict keys
# ---------------------------------------------------------------------------


class FeatureKey:
    """
    Standard keys in atom and bond feature descriptor dictionaries used
    internally by the featurizer.
    """

    ALLOWABLE_VALS: str = "allowable_vals"
    WILDCARD: str = "wildcard"
    OPTIONS: str = "options"
    DEFAULT: str = "default"
    DESCRIPTION: str = "description"


class DataSet:
    """
    Standard dataset labels used in the 'Set' column of results DataFrames.
    """

    Training: str = "Training"
    Validation: str = "Validation"
    Test: str = "Test"
