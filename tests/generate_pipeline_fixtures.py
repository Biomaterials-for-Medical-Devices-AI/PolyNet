"""
tests/generate_pipeline_fixtures.py
=====================================
Generate (or regenerate) the reference metric fixtures used by
``tests/test_pipeline_regression.py``.

Run this script **once** after intentional changes that affect metric values
(e.g. training logic, featurisation, inference), then commit the updated JSON
files alongside the code change:

    python tests/generate_pipeline_fixtures.py
    git add tests/fixtures/pipeline/
    git commit -m "Update pipeline regression fixtures"

The script uses the exact same config, seed, and data as the pytest tests so
the generated values are guaranteed to match a subsequent test run.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the repo root is on sys.path when run directly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pipeline"
SEED = 42
N_SAMPLES = 30
EPOCHS = 3

_MONOMER_SMILES = [
    "c1ccccc1",
    "C=C",
    "CC(=O)O",
    "CCO",
    "c1ccncc1",
    "CC(C)=O",
    "C1CCCCC1",
    "c1ccc(O)cc1",
    "CC(N)=O",
    "c1ccc(N)cc1",
]
_WEIGHT_PAIRS = [0.3, 0.5, 0.7]


def _make_synthetic_df(task: str) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for i in range(N_SAMPLES):
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


def _metrics_to_dict(metrics: dict) -> dict:
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


def _generate_gnn_fixture(task: str, tmp_path: Path) -> dict:
    from polynet.config.enums import (
        DescriptorMergingMethod,
        MolecularDescriptor,
        Network,
        ProblemType,
        SplitMethod,
        SplitType,
        TrainingParam,
    )
    from polynet.config.schemas import (
        DataConfig,
        RepresentationConfig,
        SplitConfig,
        TrainGNNConfig,
    )
    from polynet.pipeline import (
        build_graph_dataset,
        compute_data_splits,
        compute_metrics,
        run_gnn_inference,
        train_gnn,
    )

    df = _make_synthetic_df(task)

    data_cfg = DataConfig(
        data_name="test_data.csv",
        data_path=str(tmp_path / "test_data.csv"),
        smiles_cols=["monomer1", "monomer2"],
        target_variable_col="target",
        target_variable_name="target",
        problem_type=ProblemType(task),
        id_col="id",
        num_classes=1 if task == "regression" else 2,
    )
    repr_cfg = RepresentationConfig(
        smiles_merge_approach=DescriptorMergingMethod.WeightedAverage,
        molecular_descriptors={MolecularDescriptor.RDKit: ["MolWt", "TPSA", "MolLogP", "NumHDonors", "NumHAcceptors"]},
        weights_col={"monomer1": "weight_fraction_1", "monomer2": "weight_fraction_2"},
    )
    split_cfg = SplitConfig(
        split_type=SplitType.TrainValTest,
        split_method=SplitMethod.Random,
        test_ratio=0.2,
        val_ratio=0.2,
        n_bootstrap_iterations=1,
    )
    gnn_cfg = TrainGNNConfig(
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

    df_reset = df.reset_index()
    df_reset.to_csv(tmp_path / "test_data.csv", index=False)

    dataset = build_graph_dataset(data=df_reset, data_cfg=data_cfg, repr_cfg=repr_cfg, out_dir=tmp_path)
    split_indexes = compute_data_splits(data=df, data_cfg=data_cfg, split_cfg=split_cfg, random_seed=SEED)
    gnn_trained, gnn_loaders, gnn_target_scalers = train_gnn(
        dataset=dataset, split_indexes=split_indexes, data_cfg=data_cfg,
        gnn_cfg=gnn_cfg, random_seed=SEED, out_dir=tmp_path,
    )
    predictions = run_gnn_inference(
        trained_models=gnn_trained, loaders=gnn_loaders,
        data_cfg=data_cfg, split_cfg=split_cfg, target_scalers=gnn_target_scalers,
    )
    return _metrics_to_dict(
        compute_metrics(predictions=predictions, trained_models=gnn_trained,
                        data_cfg=data_cfg, split_cfg=split_cfg)
    )


def _generate_tml_fixture(task: str, tmp_path: Path) -> dict:
    from polynet.config.enums import (
        DescriptorMergingMethod,
        MolecularDescriptor,
        ProblemType,
        SplitMethod,
        SplitType,
        TraditionalMLModel,
    )
    from polynet.config.schemas import (
        DataConfig,
        FeatureTransformConfig,
        RepresentationConfig,
        SplitConfig,
        TrainTMLConfig,
    )
    from polynet.pipeline import (
        compute_data_splits,
        compute_descriptors,
        compute_metrics,
        run_tml_inference,
        train_tml,
    )

    df = _make_synthetic_df(task)

    data_cfg = DataConfig(
        data_name="test_data.csv",
        data_path=str(tmp_path / "test_data.csv"),
        smiles_cols=["monomer1", "monomer2"],
        target_variable_col="target",
        target_variable_name="target",
        problem_type=ProblemType(task),
        id_col="id",
        num_classes=1 if task == "regression" else 2,
    )
    repr_cfg = RepresentationConfig(
        smiles_merge_approach=DescriptorMergingMethod.WeightedAverage,
        molecular_descriptors={MolecularDescriptor.RDKit: ["MolWt", "TPSA", "MolLogP", "NumHDonors", "NumHAcceptors"]},
        weights_col={"monomer1": "weight_fraction_1", "monomer2": "weight_fraction_2"},
    )
    split_cfg = SplitConfig(
        split_type=SplitType.TrainValTest,
        split_method=SplitMethod.Random,
        test_ratio=0.2,
        val_ratio=0.2,
        n_bootstrap_iterations=1,
    )
    tml_cfg = TrainTMLConfig(
        train_tml=True,
        selected_models={TraditionalMLModel.RandomForest: {"n_estimators": 10, "max_depth": 3}},
    )
    preprocessing_cfg = FeatureTransformConfig()

    desc_dfs = compute_descriptors(data=df, data_cfg=data_cfg, repr_cfg=repr_cfg, out_dir=tmp_path)
    split_indexes = compute_data_splits(data=df, data_cfg=data_cfg, split_cfg=split_cfg, random_seed=SEED)
    tml_trained, tml_training_data, _, tml_target_scalers = train_tml(
        desc_dfs=desc_dfs, split_indexes=split_indexes, data_cfg=data_cfg,
        tml_cfg=tml_cfg, preprocessing_cfg=preprocessing_cfg,
        random_seed=SEED, out_dir=tmp_path,
    )
    predictions = run_tml_inference(
        trained=tml_trained, training_data=tml_training_data,
        data_cfg=data_cfg, split_cfg=split_cfg, target_scalers=tml_target_scalers,
    )
    return _metrics_to_dict(
        compute_metrics(predictions=predictions, trained_models=tml_trained,
                        data_cfg=data_cfg, split_cfg=split_cfg)
    )


def _save(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    print(f"  Saved → {path.relative_to(REPO_ROOT)}")


def main() -> None:
    print("Generating pipeline regression test fixtures …\n")
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = [
        ("regression",     "gnn", _generate_gnn_fixture),
        ("regression",     "tml", _generate_tml_fixture),
        ("classification", "gnn", _generate_gnn_fixture),
        ("classification", "tml", _generate_tml_fixture),
    ]

    for task, model_family, fn in scenarios:
        label = f"{task}/{model_family}"
        print(f"Running {label} …")
        with tempfile.TemporaryDirectory() as tmp:
            try:
                metrics = fn(task, Path(tmp))
                fixture_path = FIXTURE_DIR / f"{task}_{model_family}_metrics.json"
                _save(metrics, fixture_path)
                # Print a preview of the metrics
                for iteration, iter_data in metrics.items():
                    for model, model_data in iter_data.items():
                        for split, vals in model_data.items():
                            vals_str = "  ".join(f"{k}={v:.4f}" for k, v in vals.items())
                            print(f"    iter={iteration}  {model}  {split}: {vals_str}")
            except Exception as exc:
                print(f"  ✗ FAILED: {exc}")
                import traceback
                traceback.print_exc()
        print()

    print("Done. Commit the updated fixture files:")
    print("  git add tests/fixtures/pipeline/")
    print("  git commit -m 'Update pipeline regression fixtures'")


if __name__ == "__main__":
    main()
