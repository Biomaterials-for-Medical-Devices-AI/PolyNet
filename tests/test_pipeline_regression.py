"""
tests/test_pipeline_regression.py
==================================
End-to-end pipeline regression tests.

These tests run the full GNN and TML training + inference + metrics pipeline
on a small synthetic polymer dataset with a fixed random seed, then compare
the resulting metrics against pre-recorded reference values stored in
``tests/fixtures/pipeline/``.

Purpose
-------
After every major change (new feature, refactoring, dependency bump) re-running
these tests confirms that training logic, inference, and metric computation are
intact and produce bit-identical results.  They replace the manual "run the same
experiment and eyeball the numbers" workflow.

Running the tests
-----------------
    # Integration tests only
    pytest tests/test_pipeline_regression.py -v -m integration

    # Full suite (integration tests included)
    pytest tests/ -v

    # Skip integration tests for a fast unit-test pass
    pytest tests/ -m "not integration" -v

Regenerating reference fixtures
--------------------------------
Run the companion script **once** after an intentional change that affects
metric values, then commit the updated JSON files:

    python tests/generate_pipeline_fixtures.py
    git add tests/fixtures/pipeline/
    git commit -m "Update pipeline regression fixtures"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
N_SAMPLES = 30
EPOCHS = 3
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pipeline"

# Fixtures are generated on macOS, so use a tight tolerance there.
# Other platforms (CI / Linux) use different BLAS implementations that produce
# slightly different floating-point results even with the same seed.
_ON_MACOS = sys.platform == "darwin"
GNN_REL_TOL = 1e-2 if _ON_MACOS else 0.20
TML_REL_TOL = 1e-2

# Same 10 SMILES used in scripts/integration_test.py
_MONOMER_SMILES = [
    "c1ccccc1",   # benzene
    "C=C",        # ethylene
    "CC(=O)O",    # acetic acid
    "CCO",        # ethanol
    "c1ccncc1",   # pyridine
    "CC(C)=O",    # acetone
    "C1CCCCC1",   # cyclohexane
    "c1ccc(O)cc1",  # phenol
    "CC(N)=O",    # acetamide
    "c1ccc(N)cc1",  # aniline
]
_WEIGHT_PAIRS = [0.3, 0.5, 0.7]


# ---------------------------------------------------------------------------
# Synthetic data helper
# ---------------------------------------------------------------------------

def _make_synthetic_df(task: str, n_samples: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    """Build a minimal two-monomer polymer DataFrame.

    Mirrors ``make_synthetic_dataframe`` in ``scripts/integration_test.py``
    so both use the same data for manual cross-checks.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_samples):
        m1 = _MONOMER_SMILES[i % len(_MONOMER_SMILES)]
        m2 = _MONOMER_SMILES[(i + 3) % len(_MONOMER_SMILES)]
        weight = _WEIGHT_PAIRS[i % len(_WEIGHT_PAIRS)]
        target = (
            float(rng.normal(5.0, 1.5)) if task == "regression"
            else int(rng.integers(0, 2))
        )
        rows.append({
            "id": f"poly_{i:04d}",
            "monomer1": m1,
            "monomer2": m2,
            "weight_fraction_1": weight,
            "weight_fraction_2": 1 - weight,
            "target": target,
        })
    return pd.DataFrame(rows).set_index("id")


# ---------------------------------------------------------------------------
# Config factories
# ---------------------------------------------------------------------------

def _make_data_cfg(task: str, tmp_path: Path):
    from polynet.config.enums import ProblemType
    from polynet.config.schemas import DataConfig

    return DataConfig(
        data_name="test_data.csv",
        data_path=str(tmp_path / "test_data.csv"),
        smiles_cols=["monomer1", "monomer2"],
        target_variable_col="target",
        target_variable_name="target",
        problem_type=ProblemType(task),
        id_col="id",
        num_classes=1 if task == "regression" else 2,
    )


def _make_repr_cfg():
    from polynet.config.enums import DescriptorMergingMethod, MolecularDescriptor
    from polynet.config.schemas import RepresentationConfig

    # Use a small fixed set of fast RDKit descriptors so the test runs quickly
    # and is deterministic.  An empty list means "skip" per the pipeline design.
    _RDKIT_DESCRIPTORS = ["MolWt", "TPSA", "MolLogP", "NumHDonors", "NumHAcceptors"]
    return RepresentationConfig(
        smiles_merge_approach=DescriptorMergingMethod.WeightedAverage,
        molecular_descriptors={MolecularDescriptor.RDKit: _RDKIT_DESCRIPTORS},
        weights_col={"monomer1": "weight_fraction_1", "monomer2": "weight_fraction_2"},
    )


def _make_split_cfg():
    from polynet.config.enums import SplitMethod, SplitType
    from polynet.config.schemas import SplitConfig

    return SplitConfig(
        split_type=SplitType.TrainValTest,
        split_method=SplitMethod.Random,
        test_ratio=0.2,
        val_ratio=0.2,
        n_bootstrap_iterations=1,
    )


def _make_gnn_cfg():
    from polynet.config.enums import Network, TrainingParam
    from polynet.config.schemas import TrainGNNConfig

    return TrainGNNConfig(
        train_gnn=True,
        epochs=EPOCHS,
        gnn_convolutional_layers={
            Network.GCN: {
                TrainingParam.LearningRate: 1e-3,
                TrainingParam.BatchSize: 8,
                "embedding_dim": 16,
                "n_convolutions": 1,
                "readout_layers": 1,
                "dropout": 0.0,
                "improved": False,
            }
        },
    )


def _make_tml_cfg():
    from polynet.config.enums import TraditionalMLModel
    from polynet.config.schemas import TrainTMLConfig

    return TrainTMLConfig(
        train_tml=True,
        selected_models={TraditionalMLModel.RandomForest: {"n_estimators": 10, "max_depth": 3}},
    )


def _make_preprocessing_cfg():
    from polynet.config.schemas import FeatureTransformConfig

    return FeatureTransformConfig()  # NoTransformation, no selectors


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _metrics_to_dict(metrics: dict) -> dict:
    """Convert ``EvaluationMetric`` enum keys → plain string dict for JSON comparison."""
    import math

    result: dict = {}
    for iteration, iter_data in metrics.items():
        result[iteration] = {}
        for model, model_data in iter_data.items():
            result[iteration][model] = {}
            for split, split_data in model_data.items():
                result[iteration][model][split] = {
                    (k.value if hasattr(k, "value") else str(k)): v
                    for k, v in split_data.items()
                    # Exclude None and NaN — both indicate an undefined metric
                    # (e.g. AUROC when only one class present in a small split).
                    if v is not None and not (isinstance(v, float) and math.isnan(v))
                }
    return result


def _assert_metrics_match(actual_dict: dict, fixture_path: Path, rel: float = 1e-2) -> None:
    """Assert ``actual_dict`` matches the stored JSON fixture within ``rel`` tolerance.

    Always prints a comparison table to stdout — run ``pytest -s`` to see it
    for passing tests, or check captured output on failure.
    """
    if not fixture_path.exists():
        pytest.skip(
            f"Reference fixture not found: {fixture_path}. "
            "Run `python tests/generate_pipeline_fixtures.py` to generate it."
        )

    expected = json.loads(fixture_path.read_text())

    # Build and print comparison table before asserting so it's visible
    # even when all metrics pass (use `pytest -s` to see passing output).
    header = f"{'model':<30} {'split':<12} {'metric':<14} {'expected':>10} {'actual':>10} {'diff%':>8}"
    rows = []
    mismatches = []

    for iteration, iter_data in expected.items():
        for model, model_data in iter_data.items():
            for split, split_data in model_data.items():
                for metric, exp_val in split_data.items():
                    act_val = actual_dict[iteration][model][split][metric]
                    pct = abs(act_val - exp_val) / (abs(exp_val) + 1e-12) * 100
                    flag = " !" if pct > rel * 100 else ""
                    rows.append(
                        f"{model:<30} {split:<12} {metric:<14} {exp_val:>10.4f} {act_val:>10.4f} {pct:>7.2f}%{flag}"
                    )
                    if not act_val == pytest.approx(exp_val, rel=rel):
                        mismatches.append((model, split, metric, exp_val, act_val))

    print(f"\n--- {fixture_path.name} (tol={rel*100:.0f}%, platform={'macOS' if _ON_MACOS else sys.platform}) ---")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(row)

    for model, split, metric, exp_val, act_val in mismatches:
        assert act_val == pytest.approx(exp_val, rel=rel), (
            f"Metric mismatch [{model} / {split} / {metric}]: "
            f"expected {exp_val:.6f}, got {act_val:.6f}. "
            "If this is an intentional change, regenerate fixtures with "
            "`python tests/generate_pipeline_fixtures.py`."
        )


# ---------------------------------------------------------------------------
# Shared pipeline runners
# ---------------------------------------------------------------------------

def _run_gnn_pipeline(task: str, tmp_path: Path) -> dict:
    """Run graph dataset → splits → GNN train → inference → metrics."""
    from polynet.pipeline import (
        build_graph_dataset,
        compute_data_splits,
        compute_metrics,
        run_gnn_inference,
        train_gnn,
    )

    df = _make_synthetic_df(task)
    data_cfg = _make_data_cfg(task, tmp_path)
    repr_cfg = _make_repr_cfg()
    split_cfg = _make_split_cfg()
    gnn_cfg = _make_gnn_cfg()

    # Write CSV for build_graph_dataset
    df_reset = df.reset_index()
    df_reset.to_csv(tmp_path / "test_data.csv", index=False)

    dataset = build_graph_dataset(data=df_reset, data_cfg=data_cfg, repr_cfg=repr_cfg, out_dir=tmp_path)

    split_indexes = compute_data_splits(
        data=df, data_cfg=data_cfg, split_cfg=split_cfg, random_seed=SEED
    )

    gnn_trained, gnn_loaders, gnn_target_scalers = train_gnn(
        dataset=dataset,
        split_indexes=split_indexes,
        data_cfg=data_cfg,
        gnn_cfg=gnn_cfg,
        random_seed=SEED,
        out_dir=tmp_path,
    )

    predictions = run_gnn_inference(
        trained_models=gnn_trained,
        loaders=gnn_loaders,
        data_cfg=data_cfg,
        split_cfg=split_cfg,
        target_scalers=gnn_target_scalers,
    )

    return _metrics_to_dict(
        compute_metrics(
            predictions=predictions,
            trained_models=gnn_trained,
            data_cfg=data_cfg,
            split_cfg=split_cfg,
        )
    )


def _run_tml_pipeline(task: str, tmp_path: Path) -> dict:
    """Run descriptor computation → splits → TML train → inference → metrics."""
    from polynet.pipeline import (
        compute_data_splits,
        compute_descriptors,
        compute_metrics,
        run_tml_inference,
        train_tml,
    )

    df = _make_synthetic_df(task)
    data_cfg = _make_data_cfg(task, tmp_path)
    repr_cfg = _make_repr_cfg()
    split_cfg = _make_split_cfg()
    tml_cfg = _make_tml_cfg()
    preprocessing_cfg = _make_preprocessing_cfg()

    desc_dfs = compute_descriptors(data=df, data_cfg=data_cfg, repr_cfg=repr_cfg, out_dir=tmp_path)

    split_indexes = compute_data_splits(
        data=df, data_cfg=data_cfg, split_cfg=split_cfg, random_seed=SEED
    )

    tml_trained, tml_training_data, _, tml_target_scalers = train_tml(
        desc_dfs=desc_dfs,
        split_indexes=split_indexes,
        data_cfg=data_cfg,
        tml_cfg=tml_cfg,
        preprocessing_cfg=preprocessing_cfg,
        random_seed=SEED,
        out_dir=tmp_path,
    )

    predictions = run_tml_inference(
        trained=tml_trained,
        training_data=tml_training_data,
        data_cfg=data_cfg,
        split_cfg=split_cfg,
        target_scalers=tml_target_scalers,
    )

    return _metrics_to_dict(
        compute_metrics(
            predictions=predictions,
            trained_models=tml_trained,
            data_cfg=data_cfg,
            split_cfg=split_cfg,
        )
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestRegressionPipeline:
    """Pipeline regression tests for a continuous target variable."""

    def test_gnn(self, tmp_path):
        """GNN (GCN) on synthetic regression data produces expected metrics."""
        actual = _run_gnn_pipeline(task="regression", tmp_path=tmp_path)
        _assert_metrics_match(actual, FIXTURE_DIR / "regression_gnn_metrics.json", rel=GNN_REL_TOL)

    def test_tml(self, tmp_path):
        """TML (RandomForest + RDKit descriptors) on synthetic regression data produces expected metrics."""
        actual = _run_tml_pipeline(task="regression", tmp_path=tmp_path)
        _assert_metrics_match(actual, FIXTURE_DIR / "regression_tml_metrics.json", rel=TML_REL_TOL)


@pytest.mark.integration
class TestClassificationPipeline:
    """Pipeline regression tests for a binary classification target."""

    def test_gnn(self, tmp_path):
        """GNN (GCN) on synthetic classification data produces expected metrics."""
        actual = _run_gnn_pipeline(task="classification", tmp_path=tmp_path)
        _assert_metrics_match(actual, FIXTURE_DIR / "classification_gnn_metrics.json", rel=GNN_REL_TOL)

    def test_tml(self, tmp_path):
        """TML (RandomForest + RDKit descriptors) on synthetic classification data produces expected metrics."""
        actual = _run_tml_pipeline(task="classification", tmp_path=tmp_path)
        _assert_metrics_match(actual, FIXTURE_DIR / "classification_tml_metrics.json", rel=TML_REL_TOL)
