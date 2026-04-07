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
import dataclasses
import json
import logging
from pathlib import Path
import time
from typing import Any

import pandas as pd
from pydantic import BaseModel, ValidationError
import yaml

from polynet.config.schemas import (
    DataConfig,
    FeatureTransformConfig,
    GeneralConfig,
    RepresentationConfig,
    SplitConfig,
    TrainGNNConfig,
    TrainTMLConfig,
)
from polynet.config.schemas.base import PolynetBaseModel

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
        cfg["gnn_training"]["train_gnn"] = False
    if args.no_tml:
        cfg["tml_models"]["train_tml"] = False
    if args.no_explain:
        cfg["explainability"]["enabled"] = False
    return cfg


def resolve_path(path: str, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def validate_configs(config_type: PolynetBaseModel, cfg: dict):
    try:
        config_type.model_validate(cfg)
        logger.info("Valid config")
        return True
    except ValidationError as e:
        logger.warning(
            "Invalid config. While experiments might run, it is expected no compatibility with app."
        )
        logger.warning(e)


def save_options(path: Path, options: Any, config_type: PolynetBaseModel) -> None:
    """Save options/config to a JSON file at the specified path.

    Supports:
      - dataclass instances
      - Pydantic v2 models (BaseModel)
      - plain dicts
    """
    validate_configs(config_type, options)

    if dataclasses.is_dataclass(options):
        payload = dataclasses.asdict(options)
    elif BaseModel is not None and isinstance(options, BaseModel):
        # Pydantic v2
        payload = options.model_dump(mode="json")
    elif isinstance(options, dict):
        payload = options
    else:
        raise TypeError(
            f"Unsupported options type: {type(options)!r}. "
            "Expected dataclass, pydantic BaseModel, or dict."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)


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
# YAML → Pydantic config builders
# ---------------------------------------------------------------------------


def _build_data_config(cfg: dict) -> DataConfig:
    """Build DataConfig from the 'data' section of the YAML config."""
    return DataConfig.model_validate(cfg["data"])


def _build_repr_config(
    cfg: dict,
    node_feats: dict | None = None,
    edge_feats: dict | None = None,
) -> RepresentationConfig:
    """Build RepresentationConfig from the 'representations' section.

    ``node_feats`` and ``edge_feats`` override the YAML values when provided
    (used after the graph dataset is built to store the actual feature sets).
    """
    repr_dict = dict(cfg["representations"])
    if node_feats is not None:
        repr_dict["node_features"] = node_feats
    if edge_feats is not None:
        repr_dict["edge_features"] = edge_feats
    return RepresentationConfig.model_validate(repr_dict)


def _build_split_config(cfg: dict) -> SplitConfig:
    """Build SplitConfig from the 'splitting' section of the YAML config."""
    return SplitConfig.model_validate(cfg["splitting"])


def _build_gnn_config(cfg: dict) -> TrainGNNConfig:
    """Build TrainGNNConfig from the 'gnn_training' section.

    YAML string keys ``"LearningRate"`` and ``"BatchSize"`` are remapped to
    ``TrainingParam`` enum members, and architecture names to ``Network`` enum
    members, before constructing the Pydantic model.
    """
    from polynet.config.enums import Network, TrainingParam

    gnn_dict = cfg["gnn_training"]
    raw_layers = gnn_dict.get("gnn_convolutional_layers", {})
    _KEY_MAP = {
        "LearningRate": TrainingParam.LearningRate,
        "BatchSize": TrainingParam.BatchSize,
    }
    layers = {}
    for arch_name, arch_params in raw_layers.items():
        net = Network(arch_name)
        params = dict(arch_params) if arch_params else {}
        layers[net] = {_KEY_MAP.get(k, k): v for k, v in params.items()}

    return TrainGNNConfig(
        train_gnn=gnn_dict.get("train_gnn", True),
        gnn_convolutional_layers=layers,
        share_gnn_parameters=gnn_dict.get("share_gnn_parameters", True),
    )


def _build_tml_config(cfg: dict) -> TrainTMLConfig:
    """Build TrainTMLConfig from the 'tml_models' section."""
    return TrainTMLConfig.model_validate(cfg["tml_models"])


def _build_preprocessing_config(cfg: dict) -> FeatureTransformConfig:
    """Build FeatureTransformConfig from the 'feature_preprocessing' section."""
    return FeatureTransformConfig.model_validate(cfg["feature_preprocessing"])


# ---------------------------------------------------------------------------
# Data loading (script-specific)
# ---------------------------------------------------------------------------


def _load_data(cfg: dict, root: Path, out_dir: Path) -> pd.DataFrame:
    """Load and validate the dataset from disk. Script-specific stage."""
    from polynet.data.loader import load_dataset

    data_cfg = cfg["data"]
    data_path = resolve_path(data_cfg["data_path"], root)

    df = load_dataset(
        path=data_path,
        smiles_cols=data_cfg["smiles_cols"],
        target_col=data_cfg["target_variable_col"],
        id_col=data_cfg.get("id_col"),
        problem_type=data_cfg["problem_type"],
    )
    logger.info(f"  Loaded {len(df)} samples from {data_path}")
    logger.info(f"  Columns: {list(df.columns)}")
    save_options(path=out_dir / "data_options.json", options=data_cfg, config_type=DataConfig)
    return df


# ---------------------------------------------------------------------------
# Metrics serialisation
# ---------------------------------------------------------------------------


def save_metrics(metrics: dict, path: Path) -> None:
    """Serialise a metrics dict to JSON, converting enum keys to strings."""

    def _jsonify(obj):
        if isinstance(obj, dict):
            return {
                (k.value if hasattr(k, "value") else str(k)): _jsonify(v) for k, v in obj.items()
            }
        return obj

    out = path / "metrics.json"
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
    from polynet.pipeline import (
        build_graph_dataset,
        compute_data_splits,
        compute_descriptors,
        compute_metrics,
        plot_results_stage,
        run_explainability,
        run_gnn_inference,
        run_tml_inference,
        train_gnn,
        train_tml,
    )

    args = parse_args()
    root = Path(args.root).resolve()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    exp_cfg = cfg["experiment"]
    exp_name = exp_cfg["name"]
    out_dir = resolve_path(exp_cfg["output_dir"], root)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Experiment : {exp_name}")
    logger.info(f"Output dir : {out_dir}")
    logger.info(f"Task       : {cfg['data']['problem_type']}")

    save_options(out_dir / "general_options.json", options=exp_cfg, config_type=GeneralConfig)

    # Persist resolved config alongside outputs for reproducibility
    with open(out_dir / "config_used.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    gnn_enabled = cfg["gnn_training"].get("train_gnn", True)
    tml_enabled = cfg["tml_models"].get("train_tml", False)
    explain_enabled = cfg["explainability"].get("enabled", False)
    desc_enabled = bool(cfg["representations"].get("molecular_descriptors", False))

    random_seed = cfg["experiment"]["random_seed"]

    # Build Pydantic config objects shared across stages
    data_cfg = _build_data_config(cfg)
    split_cfg = _build_split_config(cfg)

    t_total = time.perf_counter()

    # ------------------------------------------------------------------
    # Stage 1 — Load data
    # ------------------------------------------------------------------
    t0 = announce("1. Load & validate data")
    df = _load_data(cfg, root, out_dir)
    df.to_csv(out_dir / cfg["data"]["data_name"])
    done(t0)

    # ------------------------------------------------------------------
    # Stage 2 — Graph dataset
    # ------------------------------------------------------------------
    dataset = None
    repr_cfg = _build_repr_config(cfg)

    if gnn_enabled:
        t0 = announce("2. Build graph dataset")
        try:
            dataset = build_graph_dataset(df, data_cfg, repr_cfg, out_dir)
            # Rebuild repr_cfg with the actual node/edge features from the dataset
            repr_cfg = _build_repr_config(cfg, dataset.node_feats, dataset.edge_feats)
            done(t0)
        except Exception as e:
            logger.error(f"Graph dataset failed: {e}. GNN stages will be skipped.")
            gnn_enabled = False

    # ------------------------------------------------------------------
    # Stage 3 — Descriptors
    # ------------------------------------------------------------------
    desc_dfs = None
    if desc_enabled and tml_enabled:
        t0 = announce("3. Compute molecular descriptors")
        try:
            desc_dfs = compute_descriptors(df, data_cfg, repr_cfg, out_dir)
            done(t0)
        except Exception as e:
            logger.error(f"Descriptor computation failed: {e}. TML stages will be skipped.")
            tml_enabled = False

    save_options(out_dir / "representation_options.json", repr_cfg, RepresentationConfig)

    # ------------------------------------------------------------------
    # Stage 4 — Data splits
    # ------------------------------------------------------------------
    t0 = announce("4. Compute data splits")
    train_idxs, val_idxs, test_idxs = compute_data_splits(
        data=df,
        data_cfg=data_cfg,
        split_cfg=split_cfg,
        random_seed=random_seed,
        out_dir=out_dir,
    )
    split_indexes = (train_idxs, val_idxs, test_idxs)
    save_options(out_dir / "split_options.json", split_cfg, SplitConfig)
    done(t0)

    all_predictions = []
    all_trained_models = {}

    # ------------------------------------------------------------------
    # Stages 5–6 — GNN training + inference
    # ------------------------------------------------------------------
    gnn_predictions = None
    gnn_trained: dict = {}
    if gnn_enabled and dataset is not None:
        t0 = announce("5. Train GNN ensemble")
        gnn_cfg = _build_gnn_config(cfg)
        save_options(out_dir / "train_gnn_options.json", gnn_cfg, TrainGNNConfig)
        try:
            gnn_trained, gnn_loaders = train_gnn(
                dataset, split_indexes, data_cfg, gnn_cfg, random_seed, out_dir
            )
            done(t0)

            t0 = announce("6. GNN inference")
            gnn_predictions = run_gnn_inference(gnn_trained, gnn_loaders, data_cfg, split_cfg)
            all_predictions.append(gnn_predictions)
            all_trained_models.update(gnn_trained)
            done(t0)
        except Exception as e:
            logger.error(f"GNN pipeline failed: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Stages 7–8 — TML training + inference
    # ------------------------------------------------------------------
    tml_predictions = None
    tml_trained: dict = {}
    if tml_enabled and desc_dfs is not None:
        t0 = announce("7. Train TML ensemble")
        tml_cfg = _build_tml_config(cfg)
        preprocessing_cfg = _build_preprocessing_config(cfg)
        save_options(out_dir / "train_tml_options.json", tml_cfg, TrainTMLConfig)
        save_options(
            out_dir / "preprocessing_tml_options.json", preprocessing_cfg, FeatureTransformConfig
        )
        try:
            tml_trained, tml_training_data, _ = train_tml(
                desc_dfs, split_indexes, data_cfg, tml_cfg, preprocessing_cfg, random_seed, out_dir
            )
            done(t0)

            t0 = announce("8. TML inference")
            tml_predictions = run_tml_inference(tml_trained, tml_training_data, data_cfg, split_cfg)
            all_predictions.append(tml_predictions)
            all_trained_models.update(tml_trained)
            done(t0)
        except Exception as e:
            logger.error(f"TML pipeline failed: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Stages 9 and 10 — Metrics and Plots
    # ------------------------------------------------------------------
    sources = []
    if gnn_predictions is not None and gnn_trained:
        sources.append((gnn_predictions, gnn_trained, "GNN"))
    if tml_predictions is not None and tml_trained:
        sources.append((tml_predictions, tml_trained, "TML"))

    if sources:
        metrics = {}
        plots_dir = out_dir / "ml_results" / "plots"

        for preds, trained, name in sources:
            t0 = announce(f"9. Metrics ({name})")
            source_metrics = compute_metrics(preds, trained, data_cfg, split_cfg)
            for iteration, models in source_metrics.items():
                for model, sets in models.items():
                    for set_name, m in sets.items():
                        vals = {
                            (k.value if hasattr(k, "value") else k): round(v, 4)
                            for k, v in m.items()
                            if v is not None
                        }
                        logger.info(f"  [{iteration}] {model} | {set_name}: {vals}")
            for iteration, iter_metrics in source_metrics.items():
                metrics.setdefault(iteration, {}).update(iter_metrics)
            done(t0)

            t0 = announce(f"10. Result plots ({name})")
            plot_results_stage(preds, trained, data_cfg, split_cfg, plots_dir)
            done(t0)

        if len(sources) == 2:
            from polynet.config.column_names import get_iterator_name, get_true_label_column_name
            from polynet.config.constants import ResultColumn

            iterator = get_iterator_name(split_cfg.split_type)
            label_col_name = get_true_label_column_name(
                target_variable_name=cfg["data"]["target_variable_name"]
            )
            gnn_predictions = gnn_predictions.drop(columns=[label_col_name])
            tml_predictions = tml_predictions.drop(columns=[ResultColumn.SET])
            predictions = pd.merge(
                left=gnn_predictions, right=tml_predictions, on=[ResultColumn.INDEX, iterator]
            )
        else:
            predictions = sources[0][0]

        save_metrics(metrics, out_dir / "ml_results")
        predictions.to_csv(out_dir / "ml_results" / "predictions.csv", index=False)

    # ------------------------------------------------------------------
    # Stage 11 — Explainability
    # ------------------------------------------------------------------
    if explain_enabled and dataset is not None and gnn_trained:
        t0 = announce("11. Explainability")
        try:
            run_explainability(
                gnn_trained, dataset, split_indexes, data_cfg, cfg["explainability"], out_dir
            )
            done(t0)
        except Exception as e:
            logger.error(f"Explainability failed: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    total = time.perf_counter() - t_total
    logger.info(f"\nPipeline complete in {total:.1f}s. Results in {out_dir}")


if __name__ == "__main__":
    main()
