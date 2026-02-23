"""
PolyNet Pipeline Runner
========================
Executes the full polymer property prediction pipeline from a YAML config.

Usage
-----
    python scripts/run_pipeline.py --config configs/experiment.yaml

    # Override specific settings from the command line
    python scripts/run_pipeline.py --config configs/experiment.yaml --epochs 100
    python scripts/run_pipeline.py --config configs/experiment.yaml --task classification
    python scripts/run_pipeline.py --config configs/experiment.yaml --no-gnn --no-explain

Stages
------
    1.  Load & validate data
    2.  Build graph dataset          (if gnn enabled)
    3.  Compute descriptors          (if descriptors enabled)
    4.  Compute data splits
    5.  Train GNN ensemble           (if gnn enabled)
    6.  Run GNN inference
    7.  Train TML ensemble           (if tml enabled)
    8.  Run TML inference
    9.  Compute metrics
    10. Plot results
    11. Run explainability           (if enabled)

All outputs are written under the directory specified by
``experiment.output_dir`` in the config file.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Logging setup — runs before any polynet imports so the root logger is
# configured before any module-level loggers are created.
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("polynet.pipeline")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Apply CLI flags over the loaded YAML config."""
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.task is not None:
        cfg["data"]["problem_type"] = args.task
    if args.no_gnn:
        cfg["gnn_models"]["enabled"] = False
    if args.no_tml:
        cfg["tml_models"]["enabled"] = False
    if args.no_explain:
        cfg["explainability"]["enabled"] = False
    return cfg


def resolve_path(path: str, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------


def announce(stage: str) -> float:
    bar = "=" * 60
    logger.info(f"\n{bar}\n  {stage}\n{bar}")
    return time.perf_counter()


def done(t0: float) -> None:
    logger.info(f"  Done ({time.perf_counter() - t0:.1f}s)")


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------


def stage_load_data(cfg: dict, root: Path) -> pd.DataFrame:
    announce("1. Load & validate data")
    from polynet.data.loader import load_dataset

    data_cfg = cfg["data"]
    data_path = resolve_path(data_cfg["path"], root)

    df = load_dataset(
        path=data_path,
        smiles_cols=cfg["structure"]["smiles_cols"],
        target_col=data_cfg["target_col"],
        id_col=data_cfg.get("id_col"),
        problem_type=data_cfg["problem_type"],
    )
    logger.info(f"  Loaded {len(df)} samples from {data_path}")
    logger.info(f"  Columns: {list(df.columns)}")
    return df


def stage_build_graph_dataset(cfg: dict, df: pd.DataFrame, out_dir: Path, root: Path):
    announce("2. Build graph dataset")
    from polynet.featurizer.polymer_graph import CustomPolymerGraph

    data_cfg = cfg["data"]
    struct_cfg = cfg["structure"]

    # Write the current (possibly filtered) DataFrame to the raw directory
    # so CustomPolymerGraph can read it via its standard file interface.
    raw_dir = out_dir / "graph_dataset" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    filename = "dataset.csv"
    df.to_csv(raw_dir / filename)

    dataset = CustomPolymerGraph(
        root=str(out_dir / "graph_dataset"),
        filename=filename,
        smiles_cols=struct_cfg["smiles_cols"],
        weights_col=struct_cfg.get("weights_cols"),
        target_col=data_cfg["target_col"],
        id_col=data_cfg.get("id_col"),
    )
    logger.info(f"  Graph dataset: {len(dataset)} graphs")
    logger.info(f"  Node features: {dataset[0].num_node_features}")
    logger.info(f"  Edge features: {dataset[0].num_edge_features}")
    return dataset


def stage_compute_descriptors(cfg: dict, df: pd.DataFrame):
    announce("3. Compute molecular descriptors")
    from polynet.featurizer.descriptors import build_vector_representation
    from polynet.config.enums import DescriptorMergingMethod
    from polynet.data.preprocessing import sanitise_df

    repr_cfg = cfg["representations"]["descriptors"]
    struct_cfg = cfg["structure"]
    data_cfg = cfg["data"]

    desc_dfs = build_vector_representation(
        data=df,
        molecular_descriptors=repr_cfg["molecular_descriptors"],
        smiles_cols=struct_cfg["smiles_cols"],
        id_col=data_cfg.get("id_col"),
        target_col=data_cfg["target_col"],
        merging_approach=DescriptorMergingMethod(
            repr_cfg.get("merging_method", "weighted_average")
        ),
        weights_col=struct_cfg.get("weights_cols"),
        rdkit_independent=repr_cfg.get("rdkit_independent", True),
        df_descriptors_independent=repr_cfg.get("df_descriptors_independent", False),
        mix_rdkit_df_descriptors=repr_cfg.get("mix_rdkit_df_descriptors", False),
    )
    for name, desc_df in desc_dfs.items():
        desc_dfs[name] = sanitise_df(
            df=desc_df,
            smiles_cols=struct_cfg["smiles_cols"],
            target_variable_col=data_cfg["target_col"],
            weights_cols=(
                list(struct_cfg.get("weights_cols").values())
                if struct_cfg.get("weights_cols")
                else None
            ),
        )
        logger.info(f"  Descriptor set '{name}': {desc_df.shape[1] - 1} features")
    return desc_dfs


def stage_data_split(cfg: dict, data: pd.DataFrame, out_dir: Path):
    announce("4. Compute data splits")
    from polynet.factories.dataloader import get_data_split_indices
    from polynet.config.enums import SplitMethod, SplitType

    split_cfg = cfg["splitting"]
    data_cfg = cfg["data"]

    train_idxs, val_idxs, test_idxs = get_data_split_indices(
        data=data,
        split_type=SplitType(split_cfg["split_type"]),
        split_method=SplitMethod(split_cfg.get("split_method", "Random")),
        n_bootstrap_iterations=split_cfg.get("n_bootstrap_iterations", 1),
        val_ratio=split_cfg.get("val_ratio", 0.15),
        test_ratio=split_cfg.get("test_ratio", 0.15),
        target_variable_col=data_cfg.get("target_col"),
        train_set_balance=split_cfg.get("train_set_balance"),
        random_seed=cfg["experiment"]["random_seed"],
    )

    for i, (tr, va, te) in enumerate(zip(train_idxs, val_idxs, test_idxs)):
        logger.info(f"  Iter {i+1}: train={len(tr)}, val={len(va)}, test={len(te)}")

    # Persist split indices for reproducibility
    splits_file = out_dir / "split_indices.json"
    with open(splits_file, "w") as f:
        json.dump(
            {
                "train": [list(map(str, s)) for s in train_idxs],
                "val": [list(map(str, s)) for s in val_idxs],
                "test": [list(map(str, s)) for s in test_idxs],
            },
            f,
            indent=2,
        )
    logger.info(f"  Split indices saved to {splits_file}")
    return train_idxs, val_idxs, test_idxs


def stage_train_gnn(cfg: dict, dataset, split_indexes, out_dir: Path):
    announce("5. Train GNN ensemble")
    from polynet.training.gnn import train_gnn_ensemble
    from polynet.config.enums import Network, ProblemType, TrainingParam

    data_cfg = cfg["data"]
    gnn_cfg = cfg["gnn_models"]
    problem_type = ProblemType(data_cfg["problem_type"])

    # Build the gnn_conv_params dict from config
    # Any architecture key that maps to an empty dict triggers HPO
    skip_keys = {"enabled"}
    gnn_conv_params = {}

    for arch_name, arch_params in gnn_cfg.items():
        if arch_name in skip_keys:
            continue
        net = Network(arch_name)
        params = dict(arch_params) if arch_params else {}

        # Rename YAML keys to TrainingParam enum keys where needed
        _KEY_MAP = {
            "LearningRate": TrainingParam.LearningRate,
            "BatchSize": TrainingParam.BatchSize,
        }
        remapped = {}
        for k, v in params.items():
            remapped[_KEY_MAP.get(k, k)] = v

        gnn_conv_params[net] = remapped

    trained_models, loaders = train_gnn_ensemble(
        experiment_path=out_dir,
        dataset=dataset,
        split_indexes=split_indexes,
        gnn_conv_params=gnn_conv_params,
        problem_type=problem_type,
        num_classes=data_cfg["num_classes"],
        random_seed=cfg["experiment"]["random_seed"],
    )

    logger.info(f"  Trained GNN models: {list(trained_models.keys())}")
    return trained_models, loaders


def stage_gnn_inference(cfg: dict, trained_models, loaders):
    announce("6. GNN inference")
    from polynet.inference.gnn import get_predictions_df_gnn
    from polynet.config.enums import ProblemType, SplitType

    predictions = get_predictions_df_gnn(
        models=trained_models,
        loaders=loaders,
        problem_type=ProblemType(cfg["data"]["problem_type"]),
        split_type=SplitType(cfg["splitting"]["split_type"]),
        target_variable_name=cfg["data"]["target_name"],
    )
    logger.info(f"  Predictions shape: {predictions.shape}")
    return predictions


def stage_train_tml(cfg: dict, desc_dfs, split_indexes):
    announce("7. Train TML ensemble")
    from polynet.training.tml import train_tml_ensemble
    from polynet.config.enums import ProblemType, TraditionalMLModel, TransformDescriptor

    data_cfg = cfg["data"]
    tml_cfg = cfg["tml_models"]
    problem_type = ProblemType(data_cfg["problem_type"])

    skip_keys = {"enabled", "descriptor_transform"}
    tml_models_config = {}
    for model_name, model_params in tml_cfg.items():
        if model_name in skip_keys:
            continue
        tml_models_config[TraditionalMLModel(model_name)] = model_params or {}

    trained, training_data, scalers = train_tml_ensemble(
        tml_models=tml_models_config,
        problem_type=problem_type,
        transform_type=TransformDescriptor(tml_cfg.get("descriptor_transform", "standard_scaler")),
        dataframes=desc_dfs,
        random_seed=cfg["experiment"]["random_seed"],
        train_val_test_idxs=split_indexes,
    )
    logger.info(f"  Trained TML models: {list(trained.keys())}")
    return trained, training_data, scalers


def stage_tml_inference(cfg: dict, trained, training_data):
    announce("8. TML inference")
    from polynet.inference.tml import get_predictions_df_tml
    from polynet.config.enums import ProblemType, SplitType

    predictions = get_predictions_df_tml(
        models=trained,
        training_data=training_data,
        split_type=SplitType(cfg["splitting"]["split_type"]),
        target_variable_col=cfg["data"]["target_col"],
        problem_type=ProblemType(cfg["data"]["problem_type"]),
        target_variable_name=cfg["data"]["target_name"],
    )
    logger.info(f"  Predictions shape: {predictions.shape}")
    return predictions


def stage_metrics(cfg: dict, predictions: pd.DataFrame, trained_models: dict, label: str):
    announce(f"9. Metrics ({label})")
    from polynet.training.metrics import get_metrics
    from polynet.config.enums import ProblemType, SplitType

    metrics = get_metrics(
        predictions=predictions,
        split_type=SplitType(cfg["splitting"]["split_type"]),
        target_variable_name=cfg["data"]["target_name"],
        trained_models=list(trained_models.keys()),
        problem_type=ProblemType(cfg["data"]["problem_type"]),
    )

    for iteration, models in metrics.items():
        for model, sets in models.items():
            for set_name, m in sets.items():
                vals = {
                    (k.value if hasattr(k, "value") else k): round(v, 4)
                    for k, v in m.items()
                    if v is not None
                }
                logger.info(f"  [{iteration}] {model} | {set_name}: {vals}")
    return metrics


def stage_plots(cfg: dict, predictions: pd.DataFrame, trained_models: dict, out_dir: Path):
    announce("10. Result plots")
    from polynet.training.evaluate import plot_learning_curves, plot_results
    from polynet.config.enums import ProblemType, SplitType

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_learning_curves(models=trained_models, save_path=plots_dir)

    class_names = cfg["data"].get("class_names")
    if isinstance(class_names, dict):
        class_names = {int(k): v for k, v in class_names.items()}

    plot_results(
        predictions=predictions,
        split_type=SplitType(cfg["splitting"]["split_type"]),
        target_variable_name=cfg["data"]["target_name"],
        ml_algorithms=list(trained_models.keys()),
        problem_type=ProblemType(cfg["data"]["problem_type"]),
        save_path=plots_dir,
        class_names=class_names,
    )
    logger.info(f"  Plots saved to {plots_dir}")


def stage_explain(cfg: dict, trained_models: dict, dataset, split_indexes, out_dir: Path):
    announce("11. Explainability")
    from polynet.config.enums import (
        ExplainAlgorithm,
        FragmentationMethod,
        ImportanceNormalisationMethod,
        ProblemType,
    )
    from polynet.explainability.pipeline import run_explanation
    from polynet.explainability.visualization import (
        plot_attribution_distribution,
        plot_mols_with_weights,
    )
    from polynet.visualization.utils import save_plot

    exp_cfg = cfg["explainability"]
    data_cfg = cfg["data"]
    explain_dir = out_dir / "explanations"
    explain_dir.mkdir(parents=True, exist_ok=True)

    # Determine which molecules to explain — default to first 5 from test set
    explain_mol_ids = exp_cfg.get("explain_mol_ids")
    if explain_mol_ids is None:
        _, _, test_idxs = split_indexes
        # Use test set from first iteration
        explain_mol_ids = [str(idx) for idx in test_idxs[0][:5]]
        logger.info(f"  explain_mol_ids not set — using first 5 test samples: {explain_mol_ids}")

    result = run_explanation(
        models=trained_models,
        dataset=dataset,
        explain_mol_ids=explain_mol_ids,
        plot_mol_ids=explain_mol_ids,
        algorithm=ExplainAlgorithm(exp_cfg["algorithm"]),
        problem_type=ProblemType(data_cfg["problem_type"]),
        experiment_path=out_dir,
        node_features=dataset.node_feats,  # pass your node feature config dict here if available
        normalisation=ImportanceNormalisationMethod(exp_cfg.get("normalisation", "Local")),
        cutoff=exp_cfg.get("cutoff", 0.05),
        fragmentation_method=FragmentationMethod(exp_cfg.get("fragmentation", "brics")),
    )

    # Fragment attribution distribution
    fig = plot_attribution_distribution(result.fragment_importances)
    save_plot(fig, explain_dir / "fragment_attributions.png")
    logger.info("  Saved fragment_attributions.png")

    # Per-molecule attribution heatmaps
    for mol_exp in result.mol_explanations:
        fig = plot_mols_with_weights(
            smiles_list=mol_exp.monomer_smiles,
            weights_list=mol_exp.per_monomer_weights,
            legend=mol_exp.monomer_smiles,
        )
        save_plot(fig, explain_dir / f"{mol_exp.mol_id}_heatmap.png")
        logger.info(f"  Saved {mol_exp.mol_id}_heatmap.png")


def save_metrics(metrics: dict, path: Path, label: str) -> None:
    """Serialise a metrics dict to JSON, converting enum keys to strings."""

    def _jsonify(obj):
        if isinstance(obj, dict):
            return {
                (k.value if hasattr(k, "value") else str(k)): _jsonify(v) for k, v in obj.items()
            }
        return obj

    out = path / f"metrics_{label}.json"
    with open(out, "w") as f:
        json.dump(_jsonify(metrics), f, indent=2)
    logger.info(f"  Metrics saved to {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the full PolyNet pipeline from a YAML config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", required=True, help="Path to the YAML experiment config file.")
    p.add_argument(
        "--epochs", type=int, default=None, help="Override training epochs from the config."
    )
    p.add_argument(
        "--task",
        choices=["regression", "classification"],
        default=None,
        help="Override problem_type from the config.",
    )
    p.add_argument("--no-gnn", action="store_true", help="Skip all GNN stages.")
    p.add_argument("--no-tml", action="store_true", help="Skip all TML stages.")
    p.add_argument("--no-explain", action="store_true", help="Skip the explainability stage.")
    p.add_argument(
        "--root",
        default=".",
        help="Project root directory. Relative paths in config are resolved from here. "
        "Defaults to the current working directory.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    exp_name = cfg["experiment"]["name"]
    out_dir = resolve_path(cfg["experiment"]["output_dir"], root)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Experiment : {exp_name}")
    logger.info(f"Output dir : {out_dir}")
    logger.info(f"Task       : {cfg['data']['problem_type']}")

    # Persist resolved config alongside outputs for reproducibility
    with open(out_dir / "config_used.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    gnn_enabled = cfg["gnn_models"].get("enabled", True)
    tml_enabled = cfg["tml_models"].get("enabled", False)
    explain_enabled = cfg["explainability"].get("enabled", False)
    desc_enabled = cfg["representations"]["descriptors"].get("enabled", False)

    t_total = time.perf_counter()

    # ------------------------------------------------------------------
    # Stage 1 — Load data
    # ------------------------------------------------------------------
    df = stage_load_data(cfg, root)

    # ------------------------------------------------------------------
    # Stage 2 — Graph dataset
    # ------------------------------------------------------------------
    dataset = None
    if gnn_enabled:
        try:
            dataset = stage_build_graph_dataset(cfg, df, out_dir, root)
        except Exception as e:
            logger.error(f"Graph dataset failed: {e}. GNN stages will be skipped.")
            gnn_enabled = False

    # ------------------------------------------------------------------
    # Stage 3 — Descriptors
    # ------------------------------------------------------------------
    desc_dfs = None
    if desc_enabled and tml_enabled:
        try:
            desc_dfs = stage_compute_descriptors(cfg, df)
        except Exception as e:
            logger.error(f"Descriptor computation failed: {e}. TML stages will be skipped.")
            tml_enabled = False

    # ------------------------------------------------------------------
    # Stage 4 — Data splits
    # ------------------------------------------------------------------
    # Split indices are computed on the graph dataset for GNN,
    # or on the DataFrame for TML-only runs.
    # split_source = dataset if dataset is not None else df
    train_idxs, val_idxs, test_idxs = stage_data_split(cfg, df, out_dir)
    split_indexes = (train_idxs, val_idxs, test_idxs)

    all_predictions = []
    all_trained_models = {}

    # ------------------------------------------------------------------
    # Stages 5–6 — GNN training + inference
    # ------------------------------------------------------------------
    gnn_predictions = None
    gnn_trained = {}
    if gnn_enabled and dataset is not None:
        try:
            gnn_trained, gnn_loaders = stage_train_gnn(cfg, dataset, split_indexes, out_dir)
            gnn_predictions = stage_gnn_inference(cfg, gnn_trained, gnn_loaders)
            all_predictions.append(gnn_predictions)
            all_trained_models.update(gnn_trained)
        except Exception as e:
            logger.error(f"GNN pipeline failed: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Stages 7–8 — TML training + inference
    # ------------------------------------------------------------------
    tml_predictions = None
    tml_trained = {}
    if tml_enabled and desc_dfs is not None:
        try:
            tml_trained, tml_training_data, _ = stage_train_tml(cfg, desc_dfs, split_indexes)
            tml_predictions = stage_tml_inference(cfg, tml_trained, tml_training_data)
            all_predictions.append(tml_predictions)
            all_trained_models.update(tml_trained)
        except Exception as e:
            logger.error(f"TML pipeline failed: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Stage 9 — Metrics
    # ------------------------------------------------------------------
    if gnn_predictions is not None and gnn_trained:
        gnn_metrics = stage_metrics(cfg, gnn_predictions, gnn_trained, "GNN")
        gnn_predictions.to_csv(out_dir / "predictions_gnn.csv", index=False)
        save_metrics(gnn_metrics, out_dir, "gnn")

    if tml_predictions is not None and tml_trained:
        tml_metrics = stage_metrics(cfg, tml_predictions, tml_trained, "TML")
        tml_predictions.to_csv(out_dir / "predictions_tml.csv", index=False)
        save_metrics(tml_metrics, out_dir, "tml")

    # ------------------------------------------------------------------
    # Stage 10 — Plots
    # ------------------------------------------------------------------
    if gnn_predictions is not None and gnn_trained:
        stage_plots(cfg, gnn_predictions, gnn_trained, out_dir / "gnn")

    if tml_predictions is not None and tml_trained:
        stage_plots(cfg, tml_predictions, tml_trained, out_dir / "tml")

    # ------------------------------------------------------------------
    # Stage 11 — Explainability
    # ------------------------------------------------------------------
    if explain_enabled and dataset is not None and gnn_trained:
        try:
            stage_explain(cfg, gnn_trained, dataset, split_indexes, out_dir)
        except Exception as e:
            logger.error(f"Explainability failed: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    total = time.perf_counter() - t_total
    logger.info(f"\nPipeline complete in {total:.1f}s. Results in {out_dir}")


if __name__ == "__main__":
    main()
