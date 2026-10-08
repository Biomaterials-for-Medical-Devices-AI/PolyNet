"""
polynet.config.schemas.split_data
================================
Pydantic schema for data splitting settings settings: splitting, reproducibility,
and bootstrap configuration.
"""

import warnings

from pydantic import Field, model_validator

from polynet.config.enums import ProblemType, SplitMethod, SplitSampler, SplitType
from polynet.config.schemas.base import PolynetBaseModel
from polynet.config.schemas.fingerprints import SamplingFingerprintConfig

# Samplers that place polymers by their (sampling) fingerprint.
SAMPLERS_USING_FINGERPRINTS = frozenset(
    {SplitSampler.KennardStone, SplitSampler.SPXY, SplitSampler.KMeans, SplitSampler.OptiSim}
)
# Samplers that also (or only) use the target values.
SAMPLERS_USING_TARGET = frozenset({SplitSampler.SPXY, SplitSampler.TargetProperty})
# Samplers that ignore the random seed: every repetition gives the same split.
DETERMINISTIC_SAMPLERS = frozenset(
    {SplitSampler.KennardStone, SplitSampler.SPXY, SplitSampler.TargetProperty}
)


def deterministic_sampler_warning(sampler: SplitSampler, n_repetitions: int) -> str:
    """Warning shown (config load and GUI) for deterministic samplers with repetitions."""
    return (
        f"The '{SplitSampler(sampler).value}' sampler is deterministic: all {n_repetitions} "
        "repetitions use the same train/validation/test split. The models still differ "
        "between repetitions because each is trained with its own seed (random_seed + "
        "repetition − 1), but models without randomness (e.g. linear regression) will be "
        "identical."
    )


def available_samplers(
    problem_type: ProblemType | str, split_method: SplitMethod | str
) -> list[SplitSampler]:
    """
    Samplers that are valid (and sensible) for a problem type and split method.

    - ``stratified`` excludes the samplers that use the target (``spxy``,
      ``target_property``): within a class the target is constant.
    - Classification excludes ``target_property``: sorting by class label
      would put a single class in the test set.

    Parameters
    ----------
    problem_type:
        Classification or regression.
    split_method:
        ``random`` or ``stratified``.

    Returns
    -------
    list[SplitSampler]
        The valid samplers, in ``SplitSampler`` order (``random`` first).
    """
    excluded: set[SplitSampler] = set()
    if SplitMethod(split_method) == SplitMethod.Stratified:
        excluded |= SAMPLERS_USING_TARGET
    if ProblemType(problem_type) == ProblemType.Classification:
        excluded.add(SplitSampler.TargetProperty)
    return [s for s in SplitSampler if s not in excluded]


def stratified_target_sampler_error(sampler: SplitSampler) -> str:
    """Why a sampler that uses the target cannot be stratified."""
    return (
        f"The '{SplitSampler(sampler).value}' sampler cannot be used with split_method "
        "'stratified': it uses the target values, which are constant within a class. Use "
        "split_method 'random'"
        + (
            ", or the 'kennard_stone' sampler (SPXY without the target)."
            if sampler == SplitSampler.SPXY
            else "."
        )
    )


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
        Splits are drawn with astartes (``sampler="random"``; per class when
        stratified).
    train_set_balance:
        Desired proportion of the minority class after undersampling the
        majority class (e.g. ``0.5`` for 50/50). Must be in (0, 1]; ``None``
        or ``1.0`` disables balancing. Only relevant for binary
        classification. The training and validation sets are each balanced
        after the split, and the test set keeps the original class
        distribution (ACS Appl. Mater. Interfaces 2023, 15 (11), 14155–14163).
    test_ratio:
        Fraction of the full dataset reserved for the test set.
        Must be in (0, 1).
    val_ratio:
        Fraction of the full dataset reserved for the validation set, so
        ``test_ratio=0.1, val_ratio=0.1`` gives an 80/10/10 split. Must be in
        (0, 1), and ``test_ratio + val_ratio`` must be below 1. Only used when
        ``split_type`` includes a validation split (e.g. TrainValTest).
    random_seed:
        Global random seed for reproducibility across all stochastic steps.
    n_bootstrap_iterations:
        Number of bootstrap iterations. Only used when ``split_type`` is
        ``TrainValTest`` or ``TrainTest``.
    sampler:
        astartes sampler that draws the sets (``random`` by default). Samplers
        in ``SAMPLERS_USING_FINGERPRINTS`` place polymers by their
        ``sampling_fingerprint``; ``spxy`` and ``target_property`` also use
        the target values. astartes' default sampler hyperparameters are used.
    sampling_fingerprint:
        Fingerprint used only for sampling (never for the representations).
        Filled with the defaults (Morgan, 2048 bins, radius 3) when the
        sampler needs one; ``None`` otherwise.
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
    test_ratio: float = Field(
        ..., gt=0.0, lt=1.0, description="Fraction of the full dataset for the test set."
    )
    val_ratio: float = Field(
        default=0.1,
        gt=0.0,
        lt=1.0,
        description="Fraction of the full dataset for the validation set.",
    )
    n_bootstrap_iterations: int = Field(
        default=1, ge=1, description="Number of bootstrap repetitions."
    )
    sampler: SplitSampler = Field(
        default=SplitSampler.Random, description="astartes sampler drawing the sets."
    )
    sampling_fingerprint: SamplingFingerprintConfig | None = Field(
        default=None,
        description="Fingerprint used only for sampling (not for the representations).",
    )

    @model_validator(mode="after")
    def resolve_sampling_fingerprint(self) -> "SplitConfig":
        """Fill the default sampling fingerprint, or warn when it would not be used."""
        if self.sampler in SAMPLERS_USING_FINGERPRINTS:
            if self.sampling_fingerprint is None:
                self.sampling_fingerprint = SamplingFingerprintConfig()
        elif self.sampling_fingerprint is not None:
            warnings.warn(
                f"sampling_fingerprint has no effect with the '{self.sampler.value}' sampler, "
                "which does not use fingerprints; it is ignored.",
                UserWarning,
                stacklevel=2,
            )
            self.sampling_fingerprint = None
        return self

    @model_validator(mode="after")
    def target_samplers_are_not_stratified(self) -> "SplitConfig":
        if self.split_method == SplitMethod.Stratified and self.sampler in SAMPLERS_USING_TARGET:
            raise ValueError(stratified_target_sampler_error(self.sampler))
        return self

    @model_validator(mode="after")
    def warn_on_repeated_deterministic_splits(self) -> "SplitConfig":
        """Deterministic samplers give the same split in every repetition."""
        if self.sampler in DETERMINISTIC_SAMPLERS and self.n_bootstrap_iterations > 1:
            warnings.warn(
                deterministic_sampler_warning(self.sampler, self.n_bootstrap_iterations),
                UserWarning,
                stacklevel=2,
            )
        return self

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
