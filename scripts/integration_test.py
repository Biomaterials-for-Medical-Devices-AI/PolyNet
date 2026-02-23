"""
PolyNet Integration Test
========================
Runs each pipeline stage independently using synthetic polymer data.
Each stage reports PASS / FAIL / SKIP and continues regardless of
prior failures, so you get a complete picture in one run.

Usage
-----
    python scripts/integration_test.py

Optional flags
--------------
    --epochs N      Training epochs per model (default: 5)
    --samples N     Synthetic dataset size (default: 40)
    --task          'regression' or 'classification' (default: regression)
    --gnn-only      Skip TML stages
    --tml-only      Skip GNN stages

Output
------
A summary table at the end lists every stage with its status and any
error message, so you can copy it directly into a bug report.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------


@dataclass
class StageResult:
    name: str
    status: str = "SKIP"  # PASS | FAIL | SKIP
    duration: float = 0.0
    error: str = ""
    notes: str = ""


RESULTS: list[StageResult] = []


def run_stage(name: str, fn, *args, **kwargs):
    """Execute a stage function, catch any exception, record result."""
    r = StageResult(name=name)
    print(f"\n{'='*60}")
    print(f"  STAGE: {name}")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        r.status = "PASS"
        r.duration = time.perf_counter() - t0
        print(f"  ✓ PASS ({r.duration:.1f}s)")
        RESULTS.append(r)
        return result
    except Exception as e:
        r.status = "FAIL"
        r.duration = time.perf_counter() - t0
        r.error = str(e)
        print(f"  ✗ FAIL ({r.duration:.1f}s)")
        print(f"  Error: {e}")
        traceback.print_exc()
        RESULTS.append(r)
        return None


def skip_stage(name: str, reason: str = ""):
    r = StageResult(name=name, status="SKIP", notes=reason)
    print(f"\n  -- SKIP: {name}" + (f" ({reason})" if reason else ""))
    RESULTS.append(r)
    return None


def print_summary():
    print(f"\n{'='*60}")
    print("  INTEGRATION TEST SUMMARY")
    print(f"{'='*60}")
    col_w = max(len(r.name) for r in RESULTS) + 2
    for r in RESULTS:
        status_fmt = {"PASS": "✓ PASS", "FAIL": "✗ FAIL", "SKIP": "-- SKIP"}[r.status]
        line = f"  {status_fmt:<8}  {r.name:<{col_w}}"
        if r.duration:
            line += f"  ({r.duration:.1f}s)"
        if r.error:
            short_err = r.error[:80] + ("..." if len(r.error) > 80 else "")
            line += f"\n           └─ {short_err}"
        print(line)

    n_pass = sum(1 for r in RESULTS if r.status == "PASS")
    n_fail = sum(1 for r in RESULTS if r.status == "FAIL")
    n_skip = sum(1 for r in RESULTS if r.status == "SKIP")
    print(f"\n  Total: {n_pass} passed, {n_fail} failed, {n_skip} skipped")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

# Representative polymer SMILES (pairs of monomers joined with *)
# Using simple RDKit-valid SMILES that won't need psmiles
_MONOMER_SMILES = [
    "c1ccccc1",  # benzene
    "C=C",  # ethylene
    "CC(=O)O",  # acetic acid
    "CCO",  # ethanol
    "c1ccncc1",  # pyridine
    "CC(C)=O",  # acetone
    "C1CCCCC1",  # cyclohexane
    "c1ccc(O)cc1",  # phenol
    "CC(N)=O",  # acetamide
    "c1ccc(N)cc1",  # aniline
]

_WEIGHT_PAIRS = [0.3, 0.5, 0.7]


def make_synthetic_dataframe(n_samples: int, task: str, seed: int = 42) -> pd.DataFrame:
    """
    Build a minimal DataFrame with two monomer SMILES columns, weight fractions,
    and a target property column.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_samples):
        m1 = _MONOMER_SMILES[i % len(_MONOMER_SMILES)]
        m2 = _MONOMER_SMILES[(i + 3) % len(_MONOMER_SMILES)]
        weight = _WEIGHT_PAIRS[i % len(_WEIGHT_PAIRS)]

        if task == "regression":
            target = float(rng.normal(5.0, 1.5))
        else:
            target = int(rng.integers(0, 2))

        rows.append(
            {
                "id": f"poly_{i:04d}",
                "monomer1": m1,
                "monomer2": m2,
                "weight_fraction_1": weight,
                "weight_fraction_2": 1 - weight,
                "target": target,
            }
        )

    return pd.DataFrame(rows).set_index("id")


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------


def stage_synthetic_data(n_samples: int, task: str):
    df = make_synthetic_dataframe(n_samples, task)
    print(f"  Created {len(df)} samples, task='{task}'")
    print(f"  Columns: {list(df.columns)}")
    print(df.head(3).to_string())
    return df


def stage_enums():
    """Verify that critical enums import and resolve correctly."""
    from polynet.config.enums import (
        Network,
        ProblemType,
        Pooling,
        SplitType,
        Optimizer,
        Scheduler,
        TraditionalMLModel,
        TransformDescriptor,
        EvaluationMetric,
    )

    checks = {
        "Network.GCN": Network.GCN,
        "Network.GAT": Network.GAT,
        "Network.CGGNN": Network.CGGNN,
        "Network.MPNN": Network.MPNN,
        "Network.GraphSAGE": Network.GraphSAGE,
        "Network.TransformerGNN": Network.TransformerGNN,
        "ProblemType.Regression": ProblemType.Regression,
        "ProblemType.Classification": ProblemType.Classification,
        "Pooling.GlobalMeanPool": Pooling.GlobalMeanPool,
        "SplitType.TrainValTest": SplitType.TrainValTest,
        "Optimizer.Adam": Optimizer.Adam,
        "Scheduler.ReduceLROnPlateau": Scheduler.ReduceLROnPlateau,
        "TraditionalMLModel.RandomForest": TraditionalMLModel.RandomForest,
        "TransformDescriptor.StandardScaler": TransformDescriptor.StandardScaler,
        "EvaluationMetric.RMSE": EvaluationMetric.RMSE,
        "EvaluationMetric.Accuracy": EvaluationMetric.Accuracy,
    }
    for name, val in checks.items():
        print(f"  {name} = {val!r}")
    return checks


def stage_graph_dataset(filename: str, tmp_path: Path):
    """Build a PyG polymer graph dataset from the synthetic DataFrame."""
    from polynet.featurizer.polymer_graph import CustomPolymerGraph  # adjust if path differs

    dataset = CustomPolymerGraph(
        root=str(tmp_path / "graph_dataset"),
        filename=filename,
        smiles_cols=["monomer1", "monomer2"],
        weights_col={"monomer1": "weight_fraction_1", "monomer2": "weight_fraction_2"},
        target_col="target",
        id_col="id",  # already in index
    )
    print(f"  Dataset: {len(dataset)} graphs")
    print(f"  Sample[0] node features: {dataset[0].num_node_features}")
    print(f"  Sample[0] edge features: {dataset[0].num_edge_features}")
    return dataset


def stage_data_split(dataset, n_samples: int):
    from polynet.factories.dataloader import get_data_split_indices
    from polynet.config.enums import SplitType, SplitMethod

    # Build a minimal DataFrame with index for splitting

    train_idxs, val_idxs, test_idxs = get_data_split_indices(
        data=dataset,
        split_type=SplitType.TrainValTest,
        n_bootstrap_iterations=2,
        val_ratio=0.15,
        test_ratio=0.15,
        target_variable_col=None,
        train_set_balance=None,
        split_method=SplitMethod.Random,
        random_seed=42,
    )
    print(f"  Iterations: {len(train_idxs)}")
    for i, (tr, va, te) in enumerate(zip(train_idxs, val_idxs, test_idxs)):
        print(f"  Iter {i+1}: train={len(tr)}, val={len(va)}, test={len(te)}")

    return train_idxs, val_idxs, test_idxs


def stage_network_factory(task: str):
    from polynet.factories.network import create_network, list_available_networks
    from polynet.config.enums import Network, ProblemType

    problem_type = ProblemType(task)
    available = list_available_networks()
    print(f"  Available networks: {available}")

    models = {}
    for net in [Network.GCN, Network.GAT, Network.CGGNN]:
        kwargs = dict(
            n_node_features=32,
            n_edge_features=5,
            embedding_dim=32,
            n_convolutions=2,
            readout_layers=2,
            n_classes=1 if task == "regression" else 2,
            dropout=0.1,
            seed=42,
        )
        if net == Network.GAT:
            kwargs["num_heads"] = 2
        if net == Network.GCN:
            kwargs["improved"] = False

        model = create_network(network=net, problem_type=problem_type, **kwargs)
        print(f"  {net.value}: {type(model).__name__}")
        models[net.value] = model

    return models


def stage_optimizer_factory(models: dict):
    from polynet.factories.optimizer import create_optimizer, create_scheduler
    from polynet.config.enums import Optimizer, Scheduler

    results = {}
    for name, model in models.items():
        opt = create_optimizer(Optimizer.Adam, model, lr=1e-3)
        sched = create_scheduler(Scheduler.ReduceLROnPlateau, opt, patience=5, min_lr=1e-6)
        print(f"  {name}: {type(opt).__name__} + {type(sched).__name__}")
        results[name] = (opt, sched)
    return results


def stage_loss_factory(task: str):
    from polynet.factories.loss import create_loss
    from polynet.config.enums import ProblemType

    loss = create_loss(ProblemType(task))
    print(f"  Loss function: {type(loss).__name__}")
    return loss


def stage_gnn_training(dataset, split_indexes, task: str, epochs: int, tmp_path: Path):
    from polynet.training.gnn import train_gnn_ensemble
    from polynet.config.enums import Network, ProblemType, TrainingParam

    problem_type = ProblemType(task)
    n_classes = 1 if task == "regression" else 2

    gnn_params = {
        Network.GCN: {
            TrainingParam.LearningRate: 1e-3,
            TrainingParam.BatchSize: 8,
            "improved": False,
            "embedding_dim": 32,
            "n_convolutions": 2,
        }
    }

    trained_models, loaders = train_gnn_ensemble(
        experiment_path=tmp_path,
        dataset=dataset,
        split_indexes=split_indexes,
        gnn_conv_params=gnn_params,
        problem_type=problem_type,
        num_classes=n_classes,
        random_seed=42,
    )

    print(f"  Trained models: {list(trained_models.keys())}")
    print(f"  Loaders: {list(loaders.keys())}")
    return trained_models, loaders


def stage_gnn_inference(trained_models, loaders, task: str):
    from polynet.inference.gnn import get_predictions_df_gnn
    from polynet.config.enums import ProblemType, SplitType

    predictions = get_predictions_df_gnn(
        models=trained_models,
        loaders=loaders,
        problem_type=ProblemType(task),
        split_type=SplitType.TrainValTest,
        target_variable_name="target",
    )
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Columns: {list(predictions.columns)}")
    print(predictions.head(3).to_string())
    return predictions


def stage_tml_training(df: pd.DataFrame, split_indexes, task: str):
    from polynet.training.tml import train_tml_ensemble
    from polynet.config.enums import ProblemType, TraditionalMLModel, TransformDescriptor

    # Build a descriptor DataFrame (simple numeric features from df)
    # In a real run this comes from featurizer.descriptors
    desc_df = df[["weight_fraction_1", "weight_fraction_2", "target"]].copy()
    desc_df["feat1"] = np.random.default_rng(0).normal(size=len(df))
    desc_df["feat2"] = np.random.default_rng(1).normal(size=len(df))

    # Reorder: features first, target last
    desc_df = desc_df[["weight_fraction_1", "weight_fraction_2", "feat1", "feat2", "target"]]

    tml_models_config = {TraditionalMLModel.RandomForest: {"n_estimators": 10, "max_depth": 3}}

    trained, training_data, scalers = train_tml_ensemble(
        tml_models=tml_models_config,
        problem_type=ProblemType(task),
        transform_type=TransformDescriptor.StandardScaler,
        dataframes={"descriptors": desc_df},
        random_seed=42,
        train_val_test_idxs=split_indexes,
    )

    print(f"  Trained TML models: {list(trained.keys())}")
    return trained, training_data, scalers, desc_df


def stage_tml_inference(trained, training_data, df: pd.DataFrame, task: str):
    from polynet.inference.tml import get_predictions_df_tml
    from polynet.config.enums import ProblemType, SplitType

    predictions = get_predictions_df_tml(
        models=trained,
        training_data=training_data,
        split_type=SplitType.TrainValTest,
        target_variable_col="target",
        problem_type=ProblemType(task),
        target_variable_name="target",
    )
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Columns: {list(predictions.columns)}")
    print(predictions.head(3).to_string())
    return predictions


def stage_metrics(predictions: pd.DataFrame, trained_models: dict, task: str):
    from polynet.training.metrics import get_metrics
    from polynet.config.enums import ProblemType, SplitType

    metrics = get_metrics(
        predictions=predictions,
        split_type=SplitType.TrainValTest,
        target_variable_name="target",
        trained_models=list(trained_models.keys()),
        problem_type=ProblemType(task),
    )
    for iteration, models in metrics.items():
        for model, sets in models.items():
            for set_name, m in sets.items():
                vals = {k.value if hasattr(k, "value") else k: f"{v:.4f}" for k, v in m.items()}
                print(f"  iter={iteration} model={model} set={set_name}: {vals}")
    return metrics


def stage_evaluate_plots(predictions, trained_models, task: str, tmp_path: Path):
    from polynet.training.evaluate import plot_learning_curves, plot_results
    from polynet.config.enums import ProblemType, SplitType

    plot_learning_curves(models=trained_models, save_path=tmp_path / "plots")
    print("  Learning curves saved.")

    plot_results(
        predictions=predictions,
        split_type=SplitType.TrainValTest,
        target_variable_name="target",
        ml_algorithms=list(trained_models.keys()),
        problem_type=ProblemType(task),
        save_path=tmp_path / "plots",
    )
    print("  Result plots saved.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="PolyNet integration test")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--samples", type=int, default=40)
    p.add_argument("--task", choices=["regression", "classification"], default="regression")
    p.add_argument("--gnn-only", action="store_true")
    p.add_argument("--tml-only", action="store_true")
    p.add_argument("--tmp-dir", default="./tmp/polynet_integration_test")
    return p.parse_args()


def main():
    args = parse_args()
    tmp_path = Path(args.tmp_dir)
    tmp_path.mkdir(parents=True, exist_ok=True)
    print(f"\nPolyNet Integration Test")
    print(f"  task={args.task}, samples={args.samples}, epochs={args.epochs}")
    print(f"  tmp_dir={tmp_path}")

    # --- Stage 1: Synthetic data ---
    df = run_stage("1. Synthetic data", stage_synthetic_data, args.samples, args.task)
    data_path = tmp_path / "graph_dataset" / "raw"
    data_path.mkdir(parents=True, exist_ok=True)
    filename = "synthetic_polymer_data.csv"
    df.to_csv(data_path / filename)

    # --- Stage 2: Enum imports ---
    run_stage("2. Enum imports", stage_enums)

    # --- Stage 3: Graph dataset ---
    if not args.tml_only:
        dataset = run_stage(
            "3. Graph dataset (featurizer)", stage_graph_dataset, filename, tmp_path
        )
    else:
        dataset = skip_stage("3. Graph dataset (featurizer)", "--tml-only")

    # --- Stage 4: Data splits ---
    split_result = run_stage("4. Data split indices", stage_data_split, df, args.samples)
    if split_result is None:
        print("\n  Cannot continue without split indices.")
        print_summary()
        sys.exit(1)
    train_idxs, val_idxs, test_idxs = split_result
    split_indexes = (train_idxs, val_idxs, test_idxs)

    # --- Stage 5: Network factory ---
    if not args.tml_only:
        factory_models = run_stage("5. Network factory", stage_network_factory, args.task)
    else:
        factory_models = skip_stage("5. Network factory", "--tml-only")

    # --- Stage 6: Optimizer/scheduler factory ---
    if factory_models and not args.tml_only:
        run_stage("6. Optimizer & scheduler factory", stage_optimizer_factory, factory_models)
    else:
        skip_stage("6. Optimizer & scheduler factory", "network factory failed or --tml-only")

    # --- Stage 7: Loss factory ---
    run_stage("7. Loss factory", stage_loss_factory, args.task)

    # --- Stage 8: GNN training ---
    gnn_trained, gnn_loaders = None, None
    if dataset is not None and not args.tml_only:
        gnn_result = run_stage(
            "8. GNN training",
            stage_gnn_training,
            dataset,
            split_indexes,
            args.task,
            args.epochs,
            tmp_path,
        )
        if gnn_result:
            gnn_trained, gnn_loaders = gnn_result
    else:
        skip_stage("8. GNN training", "dataset unavailable or --tml-only")

    # --- Stage 9: GNN inference ---
    gnn_predictions = None
    if gnn_trained and gnn_loaders:
        gnn_predictions = run_stage(
            "9. GNN inference", stage_gnn_inference, gnn_trained, gnn_loaders, args.task
        )
    else:
        skip_stage("9. GNN inference", "GNN training failed or --tml-only")

    # --- Stage 10: GNN metrics ---
    if gnn_predictions is not None and gnn_trained:
        run_stage("10. GNN metrics", stage_metrics, gnn_predictions, gnn_trained, args.task)
    else:
        skip_stage("10. GNN metrics", "GNN inference unavailable")

    # --- Stage 11: GNN result plots ---
    if gnn_predictions is not None and gnn_trained:
        run_stage(
            "11. GNN result plots",
            stage_evaluate_plots,
            gnn_predictions,
            gnn_trained,
            args.task,
            tmp_path,
        )
    else:
        skip_stage("11. GNN result plots", "GNN inference unavailable")

    # --- Stage 12: TML training ---
    tml_trained, tml_data, tml_scalers, desc_df = None, None, None, None
    if df is not None and not args.gnn_only:
        tml_result = run_stage("12. TML training", stage_tml_training, df, split_indexes, args.task)
        if tml_result:
            tml_trained, tml_data, tml_scalers, desc_df = tml_result
    else:
        skip_stage("12. TML training", "--gnn-only")

    # --- Stage 13: TML inference ---
    tml_predictions = None
    if tml_trained and tml_data:
        tml_predictions = run_stage(
            "13. TML inference", stage_tml_inference, tml_trained, tml_data, df, args.task
        )
    else:
        skip_stage("13. TML inference", "TML training failed or --gnn-only")

    # --- Stage 14: TML metrics ---
    if tml_predictions is not None and tml_trained:
        run_stage("14. TML metrics", stage_metrics, tml_predictions, tml_trained, args.task)
    else:
        skip_stage("14. TML metrics", "TML inference unavailable")

    # --- Done ---
    print_summary()
    n_fail = sum(1 for r in RESULTS if r.status == "FAIL")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
