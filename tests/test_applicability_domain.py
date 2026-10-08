"""
tests/test_applicability_domain.py
==================================
Applicability domain of new polymers: k-nearest-neighbour domain (Tropsha et
al. 2003) with the Ruzicka or Euclidean distance, expected error from the
held-out test errors (Sheridan et al. 2004), and its use when predicting new
data.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from polynet.applicability import (
    DistanceErrorCalibration,
    KNNApplicabilityDomain,
    knn_similarity,
    ruzicka_similarity,
    summarise_domain,
)
from polynet.config.enums import ApplicabilityDomainMetric

# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------


def test_ruzicka_of_count_fingerprints():
    sim = ruzicka_similarity(np.array([[1, 2, 0]]), np.array([[1, 2, 0], [1, 1, 1], [0, 0, 3]]))
    # identical; Σmin = 2, Σmax = 4; no shared bin
    np.testing.assert_allclose(sim, [[1.0, 0.5, 0.0]])


def test_ruzicka_rejects_negative_counts():
    with pytest.raises(ValueError, match="non-negative"):
        ruzicka_similarity(np.array([[1.0, -1.0]]), np.array([[1.0, 1.0]]))


def test_knn_similarity_can_exclude_each_polymer_itself():
    fps = np.array([[2, 0], [1, 1], [0, 2]])
    # Without exclusion every polymer is its own nearest neighbour.
    np.testing.assert_allclose(knn_similarity(fps, fps, k=1), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(knn_similarity(fps, fps, k=1, exclude_self=True), [1 / 3] * 3)


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------

# Hand-worked example: nearest-neighbour similarities (k = 1) are 2/3, 2/3,
# 1/3 and 2/3, so the distances are 1/3, 1/3, 2/3, 1/3: <d> = 5/12 and
# σ = sqrt(1/48).
_REFERENCE = np.array([[2, 0], [1, 1], [0, 2], [2, 1]])


def test_cutoff_is_mean_plus_z_std_of_reference_distances():
    domain = KNNApplicabilityDomain(k=1, z=1.0).fit(_REFERENCE)
    assert domain.reference_distance_mean_ == pytest.approx(5 / 12)
    assert domain.reference_distance_std_ == pytest.approx(np.sqrt(1 / 48))
    assert domain.cutoff_distance_ == pytest.approx(5 / 12 + np.sqrt(1 / 48))


def test_reference_polymers_are_in_domain_and_unrelated_ones_are_not():
    domain = KNNApplicabilityDomain(k=1, z=0.5).fit(_REFERENCE)
    scored = domain.score(np.vstack([_REFERENCE[:1], [[0, 0]]]))
    assert scored["in_domain"].tolist() == [True, False]
    assert scored["distance"].tolist() == [0.0, 1.0]


def test_euclidean_domain_on_model_inputs():
    # Points 1 apart: every k = 1 distance is 1, so <d> = 1, σ = 0, cutoff 1.
    reference = np.array([[0.0], [1.0], [2.0], [3.0]])
    domain = KNNApplicabilityDomain(
        k=1, z=0.5, metric=ApplicabilityDomainMetric.EuclideanModelInputs
    ).fit(reference)
    scored = domain.score(np.array([[1.5], [10.0]]))
    assert scored["score"].tolist() == [0.5, 7.0]  # distance / cutoff
    assert scored["in_domain"].tolist() == [True, False]


def test_polymer_descriptors_outside_training_range_are_out_of_domain():
    descriptors = pd.DataFrame({"Mw": [10.0, 20.0, 30.0, 40.0]})
    domain = KNNApplicabilityDomain(k=1, z=0.5).fit(_REFERENCE, descriptors)
    # Same structure as a training polymer, but only the first Mw is in range.
    scored = domain.score(_REFERENCE[[0, 0]], pd.DataFrame({"Mw": [25.0, 100.0]}))
    assert scored["descriptors_in_range"].tolist() == [True, False]
    assert scored["in_domain"].tolist() == [True, False]


# ---------------------------------------------------------------------------
# Expected error
# ---------------------------------------------------------------------------


def test_expected_error_is_mean_test_error_of_the_score_bin():
    calibration = DistanceErrorCalibration(n_bins=2).fit(
        scores=[0.2, 0.3, 0.7, 0.8], errors=[0.0, 1.0, 2.0, 4.0]
    )
    # Bins [0.2, 0.5] and (0.5, 0.8]; closer than every test polymer → closest bin.
    np.testing.assert_allclose(calibration.predict([0.1, 0.25, 0.75]), [0.5, 0.5, 3.0])


def test_no_expected_error_beyond_every_test_polymer():
    calibration = DistanceErrorCalibration(n_bins=2).fit([0.2, 0.3, 0.7, 0.8], [0, 1, 2, 4])
    assert np.isnan(calibration.predict([0.95])).all()


# ---------------------------------------------------------------------------
# Prediction on new data
# ---------------------------------------------------------------------------

UNRELATED = ("FC(F)(F)C(F)(F)C(F)(F)S(=O)(=O)O", "C[Si](C)(C)O[Si](C)(C)C")


@pytest.fixture(scope="module")
def tml_experiment(tmp_path_factory):
    """A trained TML regression experiment saved like the CLI saves it."""
    sys.path.insert(0, str(Path(__file__).parent))
    import test_pipeline_regression as fx

    from polynet.config.enums import SplitType
    from polynet.config.schemas import SplitConfig
    from polynet.pipeline.stages import (
        compute_data_splits,
        compute_descriptors,
        run_tml_inference,
        train_tml,
    )

    path = tmp_path_factory.mktemp("experiment")
    df = fx._make_synthetic_df("regression", 40)
    data_cfg, repr_cfg = fx._make_data_cfg("regression", path), fx._make_repr_cfg()
    split_cfg = SplitConfig(
        split_type=SplitType.TrainValTest, test_ratio=0.2, val_ratio=0.2, n_bootstrap_iterations=2
    )
    df.to_csv(path / data_cfg.data_name)
    desc = compute_descriptors(df, data_cfg, repr_cfg, path)
    splits = compute_data_splits(df, data_cfg, split_cfg, 42, path, repr_cfg.weights_col)
    trained, data, _, target_scalers = train_tml(
        desc, splits, data_cfg, fx._make_tml_cfg(), fx._make_preprocessing_cfg(), 42, path
    )
    predictions = run_tml_inference(trained, data, data_cfg, split_cfg, target_scalers)
    predictions.to_csv(path / "ml_results" / "predictions.csv", index=False)
    return path, df, data_cfg, repr_cfg


def _new_polymers(df: pd.DataFrame) -> pd.DataFrame:
    """A copy of a training polymer and a polymer unrelated to the training set."""
    new = df.reset_index().iloc[:2].drop(columns="target").copy()
    new["id"] = ["copy", "unrelated"]
    new.loc[1, ["monomer1", "monomer2"]] = UNRELATED
    return new


def _predict(experiment, out_name, **kwargs):
    from polynet.pipeline.stages import predict_external

    path, df, data_cfg, repr_cfg = experiment
    out_dir = path / "unseen" / out_name
    predictions, _ = predict_external(
        _new_polymers(df), data_cfg, repr_cfg, path, out_dir, "new.csv", **kwargs
    )
    return predictions, out_dir


@pytest.mark.integration
def test_predictions_report_the_applicability_domain(tml_experiment):
    predictions, out_dir = _predict(tml_experiment, "default")

    # Default metric: Ruzicka on the count fingerprint only.
    assert predictions["TML AD Ruzicka In Domain"].tolist() == [True, False]
    score = predictions["TML AD Ruzicka Score"]
    assert score[0] <= 1 < score[1]
    # No test polymer is as far out as the unrelated one: no expected error.
    expected = predictions["random forest-rdkit AD Ruzicka Expected Abs Error target"]
    assert expected.notna().tolist() == [True, False]
    assert not any("Euclidean" in c for c in predictions.columns)

    saved = pd.read_csv(out_dir / "predictions.csv")
    assert "TML AD Ruzicka In Domain" in saved.columns
    report = json.loads((out_dir / "applicability_domain.json").read_text())
    assert "random forest-rdkit (ruzicka_morgan)" in report["expected_error"]


@pytest.mark.integration
def test_both_metrics_give_their_own_domains(tml_experiment):
    from polynet.config.schemas import ApplicabilityDomainConfig

    cfg = ApplicabilityDomainConfig(metric=["ruzicka_morgan", "euclidean_model_inputs"])
    predictions, out_dir = _predict(tml_experiment, "both", ad_cfg=cfg)

    # Euclidean domains are per representation of the TML models.
    assert predictions["TML rdkit AD Euclidean In Domain"].tolist() == [True, False]
    assert "random forest-rdkit AD Euclidean Expected Abs Error target" in predictions
    assert "random forest-rdkit AD Ruzicka Expected Abs Error target" in predictions
    assert list(summarise_domain(predictions).index) == ["TML · Ruzicka", "TML rdkit · Euclidean"]
    report = json.loads((out_dir / "applicability_domain.json").read_text())
    assert set(report["domains"]) == {"TML (ruzicka_morgan)", "TML rdkit (euclidean_model_inputs)"}


@pytest.mark.integration
def test_tml_reference_follows_include_validation_in_training(tml_experiment):
    from polynet.config.io import save_options
    from polynet.config.paths import split_indices_path, train_tml_model_options_path
    from polynet.config.schemas import TrainTMLConfig

    path = tml_experiment[0]
    split = json.loads(split_indices_path(path).read_text())
    n_train, n_val = len(split["train"][0]), len(split["val"][0])

    def n_reference(out_name):
        _, out_dir = _predict(tml_experiment, out_name)
        report = json.loads((out_dir / "applicability_domain.json").read_text())
        return report["domains"]["TML (ruzicka_morgan)"]["splits"]["1"]["n_reference"]

    options = train_tml_model_options_path(path)
    try:
        save_options(options, TrainTMLConfig(include_validation_in_training=True))
        assert n_reference("with_val") == n_train + n_val
        save_options(options, TrainTMLConfig(include_validation_in_training=False))
        assert n_reference("without_val") == n_train
    finally:
        options.unlink()


@pytest.mark.integration
def test_disabled_or_unassessable_domain_leaves_the_predictions_unchanged(tml_experiment):
    from polynet.config.schemas import ApplicabilityDomainConfig

    disabled, out_dir = _predict(
        tml_experiment, "disabled", ad_cfg=ApplicabilityDomainConfig(enabled=False)
    )
    assert not any(" AD " in c for c in disabled.columns)
    assert not (out_dir / "applicability_domain.json").exists()

    # Without the saved training dataset the predictions are still returned.
    path, _, data_cfg, _ = tml_experiment
    dataset = path / data_cfg.data_name
    moved = dataset.rename(path / "moved.csv")
    try:
        unassessed, _ = _predict(tml_experiment, "no_dataset")
    finally:
        moved.rename(dataset)
    assert unassessed.columns.tolist() == disabled.columns.tolist()
