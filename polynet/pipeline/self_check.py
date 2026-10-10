"""
polynet.pipeline.self_check
===========================
Check that a PolyNet installation works, end to end. Installed as the
``polynet check`` command.

Runs the real YAML pipeline (``polynet run``) on a small synthetic copolymer
dataset, once for regression and once for classification, in a temporary
folder, and prints a PASS / FAIL table that can be pasted into a bug report.
The regression run also explains the models and predicts on new data.

Usage
-----
    polynet check                 # about a minute on a laptop
    polynet check --epochs 20     # longer GNN training
    polynet check --keep          # keep the outputs (path is printed)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import logging
from pathlib import Path
import platform
import shutil
import tempfile
import time
import warnings

import numpy as np
import pandas as pd
import yaml

from polynet import __version__
from polynet.utils.optional_dependencies import (
    import_psmiles_canonicaliser,
    psmiles_canonicaliser_available,
)

# Packages whose import failures are the usual installation problems.
_CORE_PACKAGES = ["torch", "torch_geometric", "rdkit", "sklearn", "xgboost", "shap", "streamlit"]

_MONOMERS = [
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


@dataclass
class CheckResult:
    name: str
    status: str  # PASS | FAIL | INFO
    detail: str = ""
    duration: float = 0.0


class _ErrorCollector(logging.Handler):
    """Collects the errors the pipeline logs when one of its parts fails."""

    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def synthetic_copolymers(n_samples: int, problem_type: str, seed: int = 42) -> pd.DataFrame:
    """
    Build a two-monomer copolymer dataset written as plain SMILES.

    Args:
        n_samples (int): Number of polymers.
        problem_type (str): ``"regression"`` or ``"classification"`` (binary).
        seed (int): Seed for the random targets.

    Returns:
        pd.DataFrame: Columns ``id``, ``monomer_1``, ``monomer_2``, ``ratio_1``,
            ``ratio_2`` and ``target``.
    """
    rng = np.random.default_rng(seed)
    ratios = [0.3, 0.5, 0.7]
    rows = []
    for i in range(n_samples):
        ratio = ratios[i % len(ratios)]
        target = rng.normal(5.0, 1.5) if problem_type == "regression" else i % 2
        rows.append(
            {
                "id": f"poly_{i:04d}",
                "monomer_1": _MONOMERS[i % len(_MONOMERS)],
                "monomer_2": _MONOMERS[(i + 3) % len(_MONOMERS)],
                "ratio_1": ratio,
                "ratio_2": 1 - ratio,
                "target": target,
            }
        )
    return pd.DataFrame(rows)


def check_config(
    problem_type: str, data_path: Path, out_dir: Path, epochs: int, predict_path: Path | None
) -> dict:
    """
    Return a small but complete pipeline config: GCN and random forest on
    Morgan fingerprints, two repeated splits.

    Args:
        problem_type (str): ``"regression"`` or ``"classification"``.
        data_path (Path): Training CSV.
        out_dir (Path): Output directory of the run.
        epochs (int): GNN training epochs.
        predict_path (Path | None): CSV to predict after training; when given,
            the models are also explained.

    Returns:
        dict: The config, as ``polynet run`` reads it from YAML.
    """
    explain = predict_path is not None
    cfg = {
        "experiment": {
            "name": f"check_{problem_type}",
            "output_dir": str(out_dir),
            "random_seed": 42,
        },
        "data": {
            "data_name": data_path.name,
            "data_path": str(data_path),
            "smiles_cols": ["monomer_1", "monomer_2"],
            "canonicalise_smiles": True,
            "target_variable_col": "target",
            "target_variable_name": "target",
            "problem_type": problem_type,
            "string_representation": "smiles",
            "id_col": "id",
            "num_classes": 1 if problem_type == "regression" else 2,
        },
        "representations": {
            "smiles_merge_approach": "weighted_average",
            "weights_col": {"monomer_1": "ratio_1", "monomer_2": "ratio_2"},
            "molecular_descriptors": {"morgan": True},
        },
        "splitting": {
            "split_type": "train_val_test",
            "split_method": "random" if problem_type == "regression" else "stratified",
            "test_ratio": 0.2,
            "val_ratio": 0.2,
            "n_bootstrap_iterations": 2,
        },
        "gnn_training": {
            "train_gnn": True,
            "gnn_convolutional_layers": {
                "GCN": {
                    "LearningRate": 0.001,
                    "BatchSize": 8,
                    "improved": False,
                    "embedding_dim": 32,
                    "n_convolutions": 2,
                    "readout_layers": 1,
                    "dropout": 0.1,
                    "pooling": "global_mean_pool",
                }
            },
        },
        "training": {"epochs": epochs},
        "feature_preprocessing": {"scaler": "standard_scaler"},
        "tml_models": {
            "train_tml": True,
            "selected_models": {"random_forest": {"n_estimators": 20, "max_depth": 3}},
        },
        "explainability": {"enabled": explain, "fragmentation": "brics", "top_n": 5},
        "tml_explainability": {"enabled": explain, "top_n": 5},
    }
    if predict_path is not None:
        cfg["prediction"] = {"enabled": True, "data_path": str(predict_path)}
    return cfg


def _check_packages() -> CheckResult:
    missing = []
    for name in _CORE_PACKAGES:
        try:
            importlib.import_module(name)
        except Exception as e:  # a broken install can raise more than ImportError
            missing.append(f"{name} ({type(e).__name__}: {e})")
    detail = f"Python {platform.python_version()}, polynet {__version__}"
    if missing:
        return CheckResult("Python packages", "FAIL", "cannot import " + "; ".join(missing))
    return CheckResult("Python packages", "PASS", detail)


def _check_device() -> CheckResult:
    import torch

    if torch.cuda.is_available():
        device = f"CUDA GPU ({torch.cuda.get_device_name(0)})"
    else:
        device = "CPU (no CUDA GPU found; GNN training runs on the CPU)"
    return CheckResult("Compute device", "INFO", f"torch {torch.__version__}, {device}")


def _check_psmiles() -> CheckResult:
    if not psmiles_canonicaliser_available():
        return CheckResult(
            "PSMILES canonicaliser",
            "INFO",
            "not installed; only needed for PSMILES data (polynet install-psmiles)",
        )
    canonical = import_psmiles_canonicaliser()("[*]CCOCCO[*]")
    if canonical != "[*]COC[*]":
        return CheckResult(
            "PSMILES canonicaliser", "FAIL", f"[*]CCOCCO[*] gave {canonical}, not [*]COC[*]"
        )
    return CheckResult("PSMILES canonicaliser", "PASS", "installed and working")


def _check_pipeline(
    name: str, problem_type: str, work_dir: Path, epochs: int, n_samples: int, predict: bool
) -> CheckResult:
    from polynet.pipeline.runner import main as run_pipeline

    run_dir = work_dir / problem_type
    run_dir.mkdir(parents=True, exist_ok=True)
    data_path = run_dir / "train.csv"
    synthetic_copolymers(n_samples, problem_type).to_csv(data_path, index=False)

    predict_path = None
    if predict:
        predict_path = run_dir / "new_polymers.csv"
        new = synthetic_copolymers(10, problem_type, seed=7)
        new["id"] = [f"new_{i:02d}" for i in range(len(new))]
        new.to_csv(predict_path, index=False)

    out_dir = run_dir / "results"
    config_path = run_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(check_config(problem_type, data_path, out_dir, epochs, predict_path), f)

    collector = _ErrorCollector()
    polynet_logger = logging.getLogger("polynet")
    previous_level = polynet_logger.level
    # Errors must reach the collector even when the console only shows critical messages.
    polynet_logger.setLevel(min(polynet_logger.getEffectiveLevel(), logging.ERROR))
    polynet_logger.addHandler(collector)
    t0 = time.perf_counter()
    try:
        run_pipeline(["--config", str(config_path)])
    except Exception as e:
        collector.messages.append(f"{type(e).__name__}: {e}")
    finally:
        polynet_logger.removeHandler(collector)
        polynet_logger.setLevel(previous_level)
    duration = time.perf_counter() - t0

    if not collector.messages and not (out_dir / "ml_results" / "metrics.json").exists():
        collector.messages.append("the run finished without writing ml_results/metrics.json")
    if collector.messages:
        return CheckResult(name, "FAIL", " | ".join(collector.messages), duration)
    return CheckResult(name, "PASS", "", duration)


def print_report(results: list[CheckResult]) -> None:
    """Print the results as a table."""
    bar = "=" * 70
    width = max(len(r.name) for r in results) + 2
    print(f"\n{bar}\n  POLYNET CHECK\n{bar}")
    for r in results:
        line = f"  {r.status:<5} {r.name:<{width}}"
        if r.duration:
            line += f"({r.duration:.0f}s)"
        print(line)
        if r.detail:
            print(f"        └─ {r.detail}")
    n_fail = sum(r.status == "FAIL" for r in results)
    verdict = "PolyNet is working." if n_fail == 0 else f"{n_fail} check(s) failed."
    print(f"\n  {verdict}\n{bar}\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="polynet check",
        description="Check that PolyNet works by running the full pipeline on synthetic data.",
    )
    p.add_argument("--epochs", type=int, default=5, help="GNN training epochs (default: 5).")
    p.add_argument(
        "--samples", type=int, default=40, help="Polymers in the synthetic dataset (default: 40)."
    )
    p.add_argument("--keep", action="store_true", help="Keep the outputs instead of deleting them.")
    p.add_argument(
        "--verbose", action="store_true", help="Show the full pipeline log while running."
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Run all checks and print the report.

    Args:
        argv (list[str] | None): Command-line arguments, without the program
            name. Defaults to ``sys.argv[1:]``.

    Returns:
        int: 0 if every check passed, otherwise 1.
    """
    args = parse_args(argv)
    # Configured before the runner, whose own logging setup then has no effect.
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.CRITICAL,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    if not args.verbose:
        warnings.simplefilter("ignore")
    print("Checking PolyNet (about a minute)…")

    results = [_check_packages()]
    if results[0].status == "PASS":
        results += [_check_device(), _check_psmiles()]

        work_dir = Path(tempfile.mkdtemp(prefix="polynet_check_"))
        pipeline_checks = [
            ("Regression: GNN + TML, explanations, new data", "regression", True),
            ("Classification: GNN + TML", "classification", False),
        ]
        for name, problem_type, predict in pipeline_checks:
            print(f"  running {name.lower()}…")
            results.append(
                _check_pipeline(name, problem_type, work_dir, args.epochs, args.samples, predict)
            )

        failed = any(r.status == "FAIL" for r in results)
        if args.keep or failed:
            results.append(CheckResult("Outputs", "INFO", str(work_dir)))
        else:
            shutil.rmtree(work_dir, ignore_errors=True)

    print_report(results)
    return 1 if any(r.status == "FAIL" for r in results) else 0
