# PolyNet

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch%20Geometric-enabled-3C2179)](https://pytorch-geometric.readthedocs.io/)
[![RDKit](https://img.shields.io/badge/RDKit-supported-informational)](https://www.rdkit.org/)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Poetry](https://img.shields.io/badge/dependency%20management-poetry-60A5FA)](https://python-poetry.org/)
[![Status](https://img.shields.io/badge/status-active%20development-brightgreen)]()

<p align="center">
  <img src="static/polynet.png" alt="LipidLNPQuantification logo" width="400"/>
</p>

**PolyNet** is a Python library for polymer property prediction using graph neural networks (GNNs) and traditional machine learning. It provides a complete, configurable pipeline — from raw SMILES strings to trained models, evaluation metrics, result plots, and atom-level explainability — designed for polymer informatics research.

---

## Overview

Predicting the properties of polymers from their chemical structure is challenging because polymers are not single molecules but repeating units of one or more monomers combined in varying ratios. PolyNet addresses this by representing each polymer as a graph where atoms are nodes, bonds are edges, and monomer weight fractions are encoded as node attributes. This representation is fed into GNN architectures that learn structure–property relationships directly from the molecular graph.

PolyNet also supports traditional machine learning workflows using molecular descriptor vectors (RDKit descriptors and PolyBERT fingerprints), enabling direct comparison between GNN and TML approaches on the same dataset and splits.

---

## Features

- **Six GNN architectures** — GCN, GAT, CGGNN, MPNN, GraphSAGE, TransformerGNN — each with regression and classification variants
- **Traditional ML** — Random Forest, XGBoost, SVM, Logistic Regression, Linear Regression
- **Multi-monomer polymers** — handles copolymers with arbitrary numbers of monomers and weight fractions
- **Cross-monomer attention** — optional attention mechanism that models interactions between monomer subgraphs
- **Automatic HPO** — Ray Tune with ASHA early stopping when no hyperparameters are provided
- **Bootstrap ensemble training** — configurable number of train/val/test splits for robust uncertainty estimates
- **Fragment-level explainability** — node attribution via Captum (IntegratedGradients, Saliency, GuidedBackprop, etc.) and GNNExplainer, aggregated to functional group level using BRICS or RECAP fragmentation
- **Graph embedding visualisation** — PCA and t-SNE projections of latent representations
- **Publication-quality plots** — parity plots, ROC curves, confusion matrices, learning curves, and attribution heatmaps
- **Single-command pipeline** — YAML config file drives the entire workflow; no code changes between experiments

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/polynet.git
cd polynet

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

**Core dependencies:** PyTorch, PyTorch Geometric, RDKit, scikit-learn, XGBoost, imbalanced-learn, Ray Tune, Captum, matplotlib, seaborn, PyYAML.

> **PolyBERT fingerprints** require the optional `psmiles` package (`pip install psmiles`). All other features work without it.

---

## Quick Start

**1. Edit the config file** to point at your data and choose your models:

```yaml
# configs/experiment.yaml
data:
  path: "data/my_polymers.csv"
  id_col: "polymer_id"
  target_col: "Tg"
  target_name: "Glass Transition Temperature"
  problem_type: "regression"
  num_classes: 1

structure:
  smiles_cols:
    - "monomer1_smiles"
    - "monomer2_smiles"
  weights_cols:
    monomer1_smiles: "weight_fraction_1"
    monomer2_smiles: "weight_fraction_2"

gnn_models:
  enabled: true
  GCN:
    LearningRate: 0.001
    BatchSize: 32
    embedding_dim: 64
    n_convolutions: 3
```

**2. Run the pipeline:**

```bash
python scripts/run_pipeline.py --config configs/experiment.yaml
```

All outputs are written to the directory specified by `experiment.output_dir` in the config.

---

## Data Format

PolyNet expects a CSV file with one row per polymer sample. At minimum it needs:

| Column | Description |
|--------|-------------|
| `polymer_id` | Unique identifier (used as index) |
| `monomer1_smiles` | SMILES string for the first monomer |
| `monomer2_smiles` | SMILES string for the second monomer |
| `weight_fraction_1` | Weight fraction of monomer 1 (between 0 and 100) |
| `weight_fraction_2` | Weight fraction of monomer 2 (= 100 − weight_fraction_1) |
| `target` | Property to predict (numeric for regression, integer class for classification) |

Polymers with more than two monomers are supported by adding additional SMILES and weight fraction columns and listing them under `smiles_cols` and `weights_cols` in the config. Weight fraction columns are optional — if omitted, monomers are treated as equally weighted.

---

## Project Structure

```
polynet/
├── config/             # Enums, column name helpers, constants, schemas
├── data/               # CSV loading and validation, preprocessing (scaling, balancing)
├── factories/          # Model, optimizer, scheduler, loss, and dataloader factories
├── featurizer/
│   ├── descriptors.py  # RDKit descriptor computation and merging strategies
│   ├── selection.py    # Feature selection (correlation filter, entropy filter, SFS)
│   ├── graph.py        # Abstract base class for PyG polymer datasets
│   └── polymer_graph.py# Concrete polymer graph dataset builder
├── models/
│   ├── base.py         # BaseNetwork (regression) and BaseNetworkClassifier
│   └── gnn/            # GCN, GAT, CGGNN, MPNN, GraphSAGE, TransformerGNN
├── training/
│   ├── gnn.py          # GNN training loop and ensemble trainer
│   ├── tml.py          # TML model training and sklearn HPO
│   ├── metrics.py      # Metric calculation and class weight utilities
│   ├── evaluate.py     # Plot generation for training results
│   └── hyperopt.py     # Ray Tune HPO for GNN architectures
├── inference/
│   ├── gnn.py          # GNN predictions DataFrame assembly
│   ├── tml.py          # TML predictions DataFrame assembly
│   └── utils.py        # Shared inference helpers (probability formatting, iteration merging)
├── explainability/
│   ├── attributions.py # Explainer construction, attribution calculation, mask merging
│   ├── fragments.py    # Fragment-level importance aggregation
│   ├── embeddings.py   # Graph embedding extraction and dimensionality reduction
│   ├── visualization.py# Attribution heatmaps and embedding projection plots
│   └── pipeline.py     # run_explanation() — pure computation, no UI dependency
├── visualization/
│   ├── plots.py        # Learning curves, parity plots, ROC curves, confusion matrices
│   └── utils.py        # save_plot()
├── configs/
│   └── experiment.yaml # Template experiment configuration
└── scripts/
    ├── run_pipeline.py       # Main CLI entry point
    └── integration_test.py   # Staged integration test for debugging
```

---

## Configuration Reference

The YAML config file controls every aspect of the pipeline. Below is a full reference with all available options.

### `experiment`

```yaml
experiment:
  name: "my_experiment"        # Used in logging and output directory naming
  output_dir: "results/my_experiment"
  random_seed: 42
```

### `data`

```yaml
data:
  path: "data/polymers.csv"
  id_col: "polymer_id"          # Column used as the sample index
  target_col: "Tg"              # Raw column name in the CSV
  target_name: "Tg (°C)"       # Display name used in plot titles and metric keys
  problem_type: "regression"    # "regression" or "classification"
  num_classes: 1                # 1 for regression, N for N-class classification
  class_names: null             # Optional: {0: "inactive", 1: "active"}
```

### `structure`

```yaml
structure:
  smiles_cols:
    - "monomer1_smiles"
    - "monomer2_smiles"
  weights_cols:                 # Set to null to treat all monomers as equally weighted
    monomer1_smiles: "weight_fraction_1"
    monomer2_smiles: "weight_fraction_2"
```

### `representations`

```yaml
representations:
  graph:
    enabled: true               # Build PyG graph dataset for GNN training

  descriptors:
    enabled: false              # Compute vector descriptors for TML training
    rdkit: true                 # Include RDKit molecular descriptors
    polybert: false             # Include PolyBERT fingerprints (requires psmiles)
    merging_method: "weighted_average"  # weighted_average | average | concatenate
```

### `splitting`

```yaml
splitting:
  split_type: "TrainValTest"    # TrainValTest | LeaveOneOut
  split_method: "Random"        # Random | Scaffold
  n_bootstrap_iterations: 3    # Number of independent train/val/test splits
  val_ratio: 0.15
  test_ratio: 0.15
  train_set_balance: null       # Optional: balance training set (0.0–1.0)
```

### `gnn_models`

Each architecture block lists its hyperparameters. Leave the block empty (`{}`) to trigger automatic HPO via Ray Tune.

```yaml
gnn_models:
  enabled: true

  GCN:
    LearningRate: 0.001
    BatchSize: 32
    improved: false              # GCN-specific: improved normalisation
    embedding_dim: 64
    n_convolutions: 3
    readout_layers: 2
    dropout: 0.1
    pooling: "GlobalMeanPool"    # GlobalMeanPool | GlobalMaxPool | GlobalAddPool | GlobalMeanMaxPool

  GAT:
    LearningRate: 0.001
    BatchSize: 32
    num_heads: 4                 # GAT-specific: number of attention heads
    embedding_dim: 64
    n_convolutions: 3
    readout_layers: 2
    dropout: 0.1

  MPNN: {}                       # Empty → automatic HPO

training:
  epochs: 250
```

**Available architectures:** `GCN`, `GAT`, `CGGNN`, `MPNN`, `GraphSAGE`, `TransformerGNN`

**Architecture-specific required parameters:**

| Architecture | Extra parameter | Description |
|---|---|---|
| `GCN` | `improved: bool` | Improved normalisation from Kipf & Welling |
| `GAT` | `num_heads: int` | Number of multi-head attention heads |
| `TransformerGNN` | `num_heads: int` | Number of transformer attention heads |
| `GraphSAGE` | `bias: bool` | Whether to include bias terms |

**Shared optional parameters:**

| Parameter | Default | Description |
|---|---|---|
| `cross_att` | `false` | Enable cross-monomer attention |
| `apply_weighting_to_graph` | `"BeforePooling"` | `BeforePooling` or `BeforeMPP` |
| `seed` | `42` | Architecture-specific random seed |

### `tml_models`

```yaml
tml_models:
  enabled: false
  descriptor_transform: "StandardScaler"  # StandardScaler | MinMaxScaler | RobustScaler | NoTransformation

  RandomForest:
    n_estimators: 200
    max_depth: 10

  XGBoost: {}                   # Empty → GridSearchCV HPO

  # SupportVectorMachine:
  # LogisticRegression:
  # LinearRegression:
```

**Available models:** `RandomForest`, `XGBoost`, `SupportVectorMachine`, `LogisticRegression`, `LinearRegression`

### `explainability`

```yaml
explainability:
  enabled: false
  algorithm: "IntegratedGradients"    # See algorithms table below
  fragmentation: "brics"              # brics | recap
  normalisation: "Local"              # Local | Global
  cutoff: 0.05                        # Zero out attributions below this threshold
  explain_mol_ids: null               # null = first 5 test-set samples
  # explain_mol_ids:
  #   - "poly_0001"
  #   - "poly_0042"
```

**Available attribution algorithms:**

| Algorithm | Type | Notes |
|---|---|---|
| `IntegratedGradients` | Captum | Best for smooth attributions; recommended starting point |
| `Saliency` | Captum | Fast; raw gradients |
| `InputXGradient` | Captum | Input × gradient product |
| `GuidedBackprop` | Captum | Guided backpropagation |
| `Deconvolution` | Captum | Deconvolution-based |
| `ShapleyValueSampling` | Captum | SHAP approximation; slow but principled |
| `GNNExplainer` | PyG native | Graph-structure-aware |

---

## Running the Pipeline

### Basic usage

```bash
python scripts/run_pipeline.py --config configs/experiment.yaml
```

### CLI flags

| Flag | Description |
|---|---|
| `--config PATH` | Path to the YAML config file (required) |
| `--epochs N` | Override `training.epochs` from the config |
| `--task regression/classification` | Override `data.problem_type` |
| `--no-gnn` | Skip all GNN stages |
| `--no-tml` | Skip all TML stages |
| `--no-explain` | Skip the explainability stage |
| `--root PATH` | Project root for resolving relative paths (default: current directory) |

### Examples

```bash
# Quick smoke test — low epoch count to check a new dataset loads cleanly
python scripts/run_pipeline.py --config configs/experiment.yaml --epochs 5

# Train GNN only, no descriptors, no explainability
python scripts/run_pipeline.py --config configs/experiment.yaml --no-tml --no-explain

# Run from a different working directory
python scripts/run_pipeline.py --config configs/experiment.yaml --root /path/to/project

# Classification task, overriding the config
python scripts/run_pipeline.py --config configs/experiment.yaml --task classification
```

### Triggering automatic HPO

Leave any architecture's parameter block empty to trigger Ray Tune search:

```yaml
gnn_models:
  GCN: {}       # ← triggers HPO for this architecture
  GAT:
    LearningRate: 0.001   # ← uses provided hyperparameters
    BatchSize: 32
    num_heads: 4
    embedding_dim: 64
```

HPO results are cached under `{output_dir}/gnn_hyp_opt/` so that re-running the pipeline with the same config loads the best configuration without repeating the search.

---

## Outputs

Everything written under `experiment.output_dir`:

```
results/my_experiment/
├── config_used.yaml             # Exact configuration for reproducibility
├── split_indices.json           # Train/val/test sample IDs for each iteration
├── predictions_gnn.csv          # Full predictions DataFrame (GNN)
├── predictions_tml.csv          # Full predictions DataFrame (TML)
├── metrics_gnn.json             # All metrics by iteration, model, and split
├── metrics_tml.json
├── gnn/
│   └── plots/
│       ├── GCN_1_learning_curve.png
│       ├── GCN_1_parity_plot.png          # regression
│       ├── GCN_1_confusion_matrix.png     # classification
│       └── GCN_1_class_1_roc_curve.png   # classification
├── tml/
│   └── plots/
│       └── RandomForest-descriptors_1_parity_plot.png
├── explanations/
│   ├── explanations.json                  # Cached raw attribution masks
│   ├── fragment_attributions.png          # Attribution distribution by functional group
│   └── poly_0001_heatmap.png             # Per-atom heatmap for each explained molecule
└── gnn_hyp_opt/                           # Ray Tune HPO results (if HPO was triggered)
    └── iteration_1/
        └── GCN/
            └── GCN.csv
```

---

## Debugging

An integration test runs each pipeline stage independently using synthetic polymer data. Stages continue regardless of prior failures, giving a complete picture in one run:

```bash
# Full pipeline smoke test
python scripts/integration_test.py

# Classification task
python scripts/integration_test.py --task classification

# Isolate TML stages (no graph building, much faster)
python scripts/integration_test.py --tml-only

# More samples, longer training
python scripts/integration_test.py --samples 80 --epochs 20
```

The summary table at the end lists every stage with its status, duration, and error message:

```
============================================================
  INTEGRATION TEST SUMMARY
============================================================
  ✓ PASS    1. Synthetic data               (0.0s)
  ✓ PASS    2. Enum imports                 (0.1s)
  ✓ PASS    3. Graph dataset (featurizer)   (4.2s)
  ✓ PASS    4. Data split indices           (0.0s)
  ✓ PASS    5. Network factory              (0.2s)
  ✓ PASS    6. Optimizer & scheduler        (0.0s)
  ✓ PASS    7. Loss factory                 (0.0s)
  ✓ PASS    8. GNN training                 (12.1s)
  ✓ PASS    9. GNN inference                (1.3s)
  ✓ PASS    10. GNN metrics                 (0.1s)
  ✓ PASS    11. GNN result plots            (2.0s)
  ✓ PASS    12. TML training                (0.8s)
  ✓ PASS    13. TML inference               (0.1s)
  ✓ PASS    14. TML metrics                 (0.0s)

  Total: 14 passed, 0 failed, 0 skipped
```

---

## Extending PolyNet

### Adding a new GNN architecture

1. Create `polynet/models/gnn/myarch.py` following the pattern in `gcn.py` or `gat.py`
2. Add `MyArchBase`, `MyArchClassifier`, `MyArchRegressor`
3. Register in `polynet/models/gnn/__init__.py`
4. Add the `(Network.MyArch, ProblemType.Regression)` and `(Network.MyArch, ProblemType.Classification)` entries to `_NETWORK_REGISTRY` in `polynet/factories/network.py`
5. Add `MyArch` to the `Network` enum in `polynet/config/enums.py`

### Adding a new TML model

1. Add the model class to `_TML_REGISTRY` in `polynet/training/tml.py`
2. Add the identifier to the `TraditionalMLModel` enum in `polynet/config/enums.py`

### Adding a new attribution algorithm

1. Add the identifier to the `ExplainAlgorithm` enum in `polynet/config/enums.py`
2. Add a `ExplainAlgorithm.MyAlgo: captum.attr.MyMethod` entry to `_CAPTUM_REGISTRY` in `polynet/explainability/attributions.py`

---

## Developer team

Main developer: [Eduardo Aguilar-Bejarano](https://edaguilarb.github.io/)


## Citing PolyNet

Coming soon :)

<!-- ```bibtex
@software{polynet,
  author  = {},
  title   = {PolyNet: Graph Neural Networks for Polymer Property Prediction},
  year    = {},
  url     = {}
}
``` -->
