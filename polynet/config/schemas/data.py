"""
polynet.config.schemas.data
============================
Pydantic schema for dataset loading and target variable configuration.
"""

from pydantic import Field, field_validator, model_validator

from polynet.config.enums import ProblemType, StringRepresentation
from polynet.config.schemas.base import PolynetBaseModel


class DataConfig(PolynetBaseModel):
    """
    Configuration for dataset loading and target variable definition.

    Attributes
    ----------
    data_name:
        A short human-readable name for the dataset. Used in plot titles
        and result filenames.
    data_path:
        Path to the CSV (or compatible) file containing the dataset.
    smiles_cols:
        One or more column names that contain SMILES strings. Multiple
        columns are used when modelling polymer blends or co-polymers.
    canonicalise_smiles:
        Whether to canonicalise SMILES strings using RDKit before
        featurization. Strongly recommended for reproducibility.
    target_variable_col:
        Column name containing the target property values.
    problem_type:
        Whether this is a regression or classification task.
    string_representation:
        The polymer string notation used in ``smiles_cols``.
    id_col:
        Optional column used as a sample identifier. If None, the
        DataFrame index is used.
    num_classes:
        Required when ``problem_type`` is Classification. Must match the
        number of distinct classes in ``target_variable_col``.
    target_variable_name:
        Human-readable name for the target variable, used in plot axes
        and result column headers. Defaults to ``target_variable_col``
        if not provided.
    class_names:
        Optional mapping from class index (as string) to a human-readable
        label, e.g. ``{"0": "inactive", "1": "active"}``.
    target_variable_units:
        Optional units string appended to axis labels, e.g. ``"kJ/mol"``.
    """

    data_name: str = Field(..., description="Short human-readable dataset name.")
    data_path: str = Field(..., description="Path to the dataset file.")
    smiles_cols: list[str] = Field(
        ..., min_length=1, description="Column(s) containing SMILES strings."
    )
    canonicalise_smiles: bool = Field(
        default=True, description="Canonicalise SMILES with RDKit before featurization."
    )
    target_variable_col: str = Field(..., description="Column name for the target property.")
    problem_type: ProblemType = Field(..., description="Regression or classification task.")
    string_representation: StringRepresentation = Field(
        default=StringRepresentation.SMILES,
        description="Polymer string notation used in smiles_cols.",
    )
    id_col: str | None = Field(default=None, description="Optional sample identifier column.")
    num_classes: int | None = Field(
        default=None, ge=2, description="Number of classes (classification only)."
    )
    target_variable_name: str | None = Field(
        default=None, description="Human-readable target variable name."
    )
    class_names: dict[str, str] | None = Field(
        default=None, description="Mapping from class index to label."
    )
    target_variable_units: str | None = Field(
        default=None, description="Units for the target variable."
    )

    @field_validator("data_name", "data_path", "target_variable_col")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field must not be an empty string.")
        return v

    @field_validator("smiles_cols")
    @classmethod
    def smiles_cols_not_empty_strings(cls, v: list[str]) -> list[str]:
        if any(not col.strip() for col in v):
            raise ValueError("smiles_cols must not contain empty strings.")
        return v

    @model_validator(mode="after")
    def classification_requires_num_classes(self) -> "DataConfig":
        if self.problem_type == ProblemType.Classification and self.num_classes is None:
            raise ValueError("num_classes is required when problem_type is 'classification'.")
        if self.problem_type == ProblemType.Regression and self.num_classes is not None:
            raise ValueError("num_classes should not be set when problem_type is 'regression'.")
        return self

    @model_validator(mode="after")
    def class_names_match_num_classes(self) -> "DataConfig":
        if self.class_names is not None and self.num_classes is not None:
            if len(self.class_names) != self.num_classes:
                raise ValueError(
                    f"class_names has {len(self.class_names)} entries but "
                    f"num_classes is {self.num_classes}. They must match."
                )
        return self
