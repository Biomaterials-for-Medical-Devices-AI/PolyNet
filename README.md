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
  <img src="static/polynet.png" alt="PolyNet logo" width="400"/>
</p>

**PolyNet** is a Python library for polymer property prediction using graph neural networks (GNNs) and traditional machine learning (TML). It provides a complete, configurable pipeline — from raw SMILES strings to trained models, evaluation metrics, result plots, and atom-level explainability — designed for polymer informatics research.

The library is split into two layers:

- **Core package (`polynet`)** — pipeline stages, model training/inference, featurisers, config schemas, and path/IO helpers. No Streamlit dependency; importable from any Python script or notebook.
- **Optional GUI (`polynet.app`)** — a Streamlit web application that wraps the same core stages. Install only when you want an interactive interface.

Both entry points call the same underlying functions and produce identical results.

---

## Overview

Predicting polymer properties from chemical structure is challenging because polymers are repeating units of one or more monomers combined in varying ratios. PolyNet handles this by representing each polymer as a graph where atoms are nodes, bonds are edges, and monomer weight fractions are encoded as node attributes. This graph is fed into GNN architectures that learn structure–property relationships directly from the molecular graph.

PolyNet also supports traditional ML workflows using molecular descriptor vectors (RDKit descriptors, PolyBERT fingerprints, and others), enabling direct comparison between GNN and TML approaches on the same dataset and splits.

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
- **GUI-independent core** — all pipeline stages are importable from `polynet` without Streamlit

---

## Installation

PolyNet requires **Python 3.11** and uses [Poetry](https://python-poetry.org/) for dependency management.

### Step 1 — Clone the repository

```bash
git clone https://github.com/Biomaterials-for-Medical-Devices-AI/PolyNet.git
cd PolyNet
```

### Step 2 — Create a Python 3.11 environment

Choose either option.

**Option A — conda**

```bash
conda create -n polynet python=3.11
conda activate polynet
```

**Option B — venv**

```bash
python3.11 -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows
```

### Step 3 — Install Poetry

With the environment active, install Poetry via pip:

```bash
pip install poetry
```

Then tell Poetry to use the active environment rather than creating its own:

```bash
poetry config virtualenvs.create false --local
```

### Step 4 — Install the project

```bash
poetry install
```

Poetry reads `pyproject.toml` and `poetry.lock` and installs all dependencies into the active environment.

### Verifying the installation

```bash
python scripts/integration_test.py
```

### Common Poetry commands

| Command | Description |
|---|---|
| `poetry install` | Install all dependencies from `poetry.lock` |
| `poetry add <package>` | Add a new dependency |
| `poetry update` | Update all dependencies within the version constraints |
| `poetry run python <script>` | Run a script without manually activating the environment |

**Dependencies** (managed by Poetry): PyTorch, PyTorch Geometric, RDKit, scikit-learn, XGBoost, imbalanced-learn, Ray Tune, Captum, Streamlit, Pydantic, matplotlib, seaborn, scipy, statsmodels, and more — see `pyproject.toml` for the full list.

---

## Quick Start

### CLI

**1. Edit the config file** to point at your data and choose your models:

```yaml
# configs/experiment.yaml
experiment:
  name: "tg_prediction"
  output_dir: "results/tg_prediction"
  random_seed: 42

data:
  data_path: "data/polymers.csv"
  id_col: "polymer_id"
  target_variable_col: "Tg"
  target_variable_name: "Glass Transition Temperature"
  problem_type: "regression"
  smiles_cols:
    - "monomer1_smiles"
    - "monomer2_smiles"
```

**2. Run the pipeline:**

```bash
python scripts/run_pipeline.py --config configs/experiment.yaml
```

All outputs are written to the directory specified by `experiment.output_dir` in the config.

### Streamlit GUI

```bash
streamlit run polynet/app/Welcome_to_PolyNet.py
```

### Python API

The core stages can be called directly from any script or notebook:

```python
from polynet.pipeline import (
    build_graph_dataset,
    compute_descriptors,
    compute_data_splits,
    train_gnn,
    run_gnn_inference,
    train_tml,
    run_tml_inference,
    compute_metrics,
    predict_external,
)
from polynet.config.io import save_options, load_options
from polynet.config.paths import polynet_experiments_base_dir, model_dir
from polynet.experiment.manager import get_experiments, create_experiment
from polynet.models.persistence import load_models_from_experiment
from polynet.inference.predict import predict_unseen_tml, predict_unseen_gnn
```

---

## Data Format

PolyNet expects a CSV file with one row per polymer sample:

| Column | Required | Description |
|--------|----------|-------------|
| `polymer_id` | Recommended | Unique identifier (used as sample index) |
| `monomer1_smiles` | Yes | SMILES string for the first monomer |
| `monomer2_smiles` | Yes (copolymers) | SMILES string for the second monomer |
| `weight_fraction_1` | No | Weight fraction of monomer 1 (0–100). Omit to treat monomers as equally weighted |
| `weight_fraction_2` | No | Weight fraction of monomer 2 |
| `target` | Yes | Property to predict (numeric for regression, integer class for classification) |

Polymers with more than two monomers are supported by adding additional SMILES and weight fraction columns and listing them in the config. The target column is optional when predicting on unseen data.

---

## Project Structure

```
polynet/
├── config/                    # Schemas, enums, constants, IO, and path helpers
│   ├── enums.py               # ProblemType, MolecularDescriptor, Network, AtomFeature, etc.
│   ├── constants.py           # ResultColumn, shared string constants
│   ├── column_names.py        # Standardised prediction/score column name builders
│   ├── experiment.py          # ExperimentConfig (top-level Pydantic schema)
│   ├── io.py                  # save_options(), load_options()
│   ├── paths.py               # 32 path-helper functions for experiment directories
│   ├── from_app.py            # load_config_from_app_state(), save_section_to_json()
│   ├── from_yaml.py           # YAML → ExperimentConfig loader
│   ├── _loader.py             # Internal config merging and normalisation
│   ├── search_grid.py         # Default HPO search grids for TML models
│   └── schemas/               # Per-section Pydantic models
│       ├── data.py            # DataConfig
│       ├── general.py         # GeneralConfig
│       ├── representation.py  # RepresentationConfig
│       ├── training.py        # TrainGNNConfig, TrainTMLConfig
│       ├── split_data.py      # SplitConfig
│       ├── feature_preprocessing.py  # FeatureTransformConfig
│       ├── plotting.py        # PlottingConfig
│       └── base.py            # PolynetBaseModel (shared Pydantic base)
│
├── data/                      # Data loading and preprocessing
│   ├── __init__.py            # sanitise_df() — strip SMILES/weight cols, ensure target last
│   ├── loader.py              # load_dataset() — CSV loading with validation
│   ├── preprocessing.py       # sanitise_df(), scaling, SMILES-to-string conversion
│   ├── feature_transformer.py # FeatureTransformer (sklearn scaler wrapper)
│   └── creator.py             # Synthetic dataset generation (for testing)
│
├── experiment/                # Experiment filesystem management
│   ├── __init__.py
│   └── manager.py             # get_experiments(), create_experiment()
│
├── featurizer/                # Molecular feature generation
│   ├── allowable_sets.py      # atom_properties, bond_features (atom/bond encoder dicts)
│   ├── descriptors.py         # build_vector_representation() — RDKit, PolyBERT, Morgan, etc.
│   ├── graph.py               # PolymerGraphDataset (abstract PyG dataset base)
│   ├── polymer_graph.py       # CustomPolymerGraph (concrete dataset builder)
│   ├── pmx.py                 # PolyMetriX descriptor integration
│   └── selection.py           # Feature selection (correlation, entropy, SFS)
│
├── models/                    # Neural network models and persistence
│   ├── base.py                # BaseNetwork (regression) and BaseNetworkClassifier
│   ├── persistence.py         # save/load GNN and TML models; load_models_from_experiment(),
│   │                          # load_scalers_from_experiment(), load_dataframes()
│   └── gnn/                   # GNN architecture implementations
│       ├── gcn.py             # GCNBase, GCNRegressor, GCNClassifier
│       ├── gat.py             # GATBase, GATRegressor, GATClassifier
│       ├── cggnn.py           # CGGNNBase, CGGNNRegressor, CGGNNClassifier
│       ├── mpnn.py            # MPNNBase, MPNNRegressor, MPNNClassifier
│       ├── graphsage.py       # GraphSAGEBase, GraphSAGERegressor, GraphSAGEClassifier
│       └── transformer.py     # TransformerGNNBase, TransformerGNNRegressor, TransformerGNNClassifier
│
├── factories/                 # Object construction factories
│   ├── network.py             # get_network() — (Network, ProblemType) → model class
│   ├── dataloader.py          # get_data_split_indices(), build DataLoader splits
│   ├── loss.py                # get_loss_fn() — loss by problem type
│   └── optimizer.py          # get_optimizer(), get_scheduler()
│
├── training/                  # Model training loops
│   ├── gnn.py                 # train_gnn_ensemble() — GNN bootstrap ensemble trainer
│   ├── tml.py                 # train_tml_ensemble() — TML trainer with HPO and scaling
│   ├── metrics.py             # get_metrics(), calculate_metrics() — R², AUROC, F1, etc.
│   ├── evaluate.py            # plot_results(), plot_learning_curves()
│   └── hyperopt.py            # run_hpo() — Ray Tune + ASHA HPO for GNN
│
├── inference/                 # Prediction assembly and unseen-data prediction
│   ├── __init__.py
│   ├── gnn.py                 # get_predictions_df_gnn() — GNN predictions over splits
│   ├── tml.py                 # get_predictions_df_tml() — TML predictions over splits
│   ├── predict.py             # predict_unseen_tml(), predict_unseen_gnn()
│   └── utils.py               # prepare_probs_df(), assemble_predictions(),
│                              # merge_model_predictions(), ensemble_predictions()
│
├── pipeline/                  # Shared pipeline stage functions
│   ├── __init__.py            # Public API for all stages
│   └── stages.py              # build_graph_dataset(), compute_descriptors(),
│                              # compute_data_splits(), train_gnn(), run_gnn_inference(),
│                              # train_tml(), run_tml_inference(), compute_metrics(),
│                              # plot_results_stage(), predict_external(), run_explainability()
│
├── explainability/            # Attribution-based model explanation
│   ├── __init__.py
│   ├── attributions.py        # build_explainer(), calculate_attributions(), merge_mask_dicts()
│   ├── fragments.py           # Fragment-level importance aggregation (BRICS/RECAP)
│   ├── embeddings.py          # Graph embedding extraction (PCA, t-SNE)
│   ├── visualization.py       # plot_mols_with_weights(), plot_attribution_distribution()
│   └── pipeline.py            # run_explanation() — orchestrates full explainability run
│
├── plotting/                  # Data exploration plots
│   ├── data_analysis.py       # show_continuous_distribution(), show_label_distribution()
│   └── results.py             # Result-specific plotting helpers
│
├── visualization/             # Low-level plot rendering
│   ├── plots.py               # Learning curves, parity plots, ROC curves, confusion matrices
│   └── utils.py               # save_plot()
│
├── utils/                     # General utilities
│   ├── __init__.py            # create_directory(), save_data(), filter_dataset_by_ids(),
│   │                          # extract_number(), save_plot(), prepare_probs_df()
│   ├── chem_utils.py          # check_smiles_cols(), determine_string_representation()
│   ├── data_preprocessing.py  # keep_only_numerical_columns(), check_column_is_numeric(),
│   │                          # class_balancer(), print_class_balance()
│   ├── statistical_analysis.py # metrics_pvalue_matrix(), significance_marker(), mcnemar_*
│   └── plot_utils.py          # Low-level matplotlib helpers
│
└── app/                       # Streamlit GUI (optional, requires streamlit)
    ├── Welcome_to_PolyNet.py  # App entry point and landing page
    ├── pages/                 # One file per app page
    │   ├── 1_Create_Experiment.py
    │   ├── 2_Representation.py
    │   ├── 3_Train_Models.py
    │   ├── 4_Predict.py
    │   ├── 5_Explain_Models.py
    │   └── 6_Analyse_Results.py
    ├── components/            # Reusable Streamlit widgets and plots
    │   ├── experiments.py
    │   ├── plots.py
    │   └── forms/
    ├── services/              # Thin re-exports of core modules (backward-compatible shims)
    │   ├── configurations.py  → polynet.config.io
    │   ├── experiments.py     → polynet.experiment.manager
    │   ├── model_training.py  → polynet.models.persistence
    │   ├── predict_model.py   → polynet.inference.predict
    │   ├── explain_model.py   (Streamlit-coupled; stays in app)
    │   └── plot.py            (Streamlit-coupled; stays in app)
    ├── options/               # App-specific options (re-exports + Streamlit state keys)
    │   ├── file_paths.py      → polynet.config.paths
    │   ├── allowable_sets.py  → polynet.featurizer.allowable_sets
    │   ├── state_keys.py      (Streamlit session state key enums)
    │   └── plot.py            (App-specific plot configuration)
    └── utils/
        └── __init__.py        → polynet.utils + polynet.inference.utils + polynet.utils.*

scripts/
├── run_pipeline.py        # CLI entry point
└── integration_test.py    # Staged integration test for debugging

configs/
└── experiment.yaml        # Template experiment configuration
```

---

## Package API

### `polynet.pipeline`

The main entry point for running experiment stages. Both the CLI and the Streamlit app call these functions.

```python
from polynet.pipeline import (
    build_graph_dataset,    # Featurise data into a PyG graph dataset
    compute_descriptors,    # Compute molecular descriptor vectors and save CSVs
    compute_data_splits,    # Generate bootstrap train/val/test split indices
    train_gnn,              # Train a GNN ensemble, save .pt model files
    run_gnn_inference,      # Run GNN inference on all splits → predictions DataFrame
    train_tml,              # Train a TML ensemble, save .joblib + .pkl scaler files
    run_tml_inference,      # Run TML inference on all splits → predictions DataFrame
    compute_metrics,        # Compute evaluation metrics from predictions DataFrame
    plot_results_stage,     # Generate learning curves and result plots
    predict_external,       # Predict on unseen data using a trained experiment
    run_explainability,     # Run attribution-based explainability and save heatmaps
)
```

### `polynet.config.io`

Serialisation helpers for all config objects.

```python
from polynet.config.io import save_options, load_options

save_options(path, options)          # Save dataclass, Pydantic model, or dict → JSON
load_options(path, options_class)    # Load JSON → Pydantic model instance
```

### `polynet.config.paths`

Path helpers for all experiment directories and files. Every path used by the pipeline is constructed here.

```python
from polynet.config.paths import (
    polynet_experiments_base_dir,           # ~/PolyNetExperiments
    polynet_experiment_path,                # ~/PolyNetExperiments/<name>
    data_options_path,                      # <experiment>/data_options.json
    representation_options_path,            # <experiment>/representation_options.json
    general_options_path,                   # <experiment>/general_options.json
    train_gnn_model_options_path,           # <experiment>/train_gnn_options.json
    train_tml_model_options_path,           # <experiment>/train_tml_options.json
    model_dir,                              # <experiment>/ml_results/models/
    plots_directory,                        # <experiment>/ml_results/plots/
    ml_results_file_path,                   # <experiment>/ml_results/predictions.csv
    model_metrics_file_path,                # <experiment>/ml_results/metrics.json
    representation_file,                    # <experiment>/representation/Descriptors/<name>.csv
    gnn_raw_data_path,                      # <experiment>/representation/GNN/raw/
    unseen_predictions_experiment_parent_path,  # <experiment>/unseen_predictions/<file>/
    explanation_parent_directory,           # <experiment>/explanations/
    # ... and 18 more helpers
)
```

### `polynet.experiment.manager`

Filesystem operations for experiment lifecycle.

```python
from polynet.experiment.manager import get_experiments, create_experiment

experiments = get_experiments()                       # List all saved experiments
create_experiment(experiment_path, data_options)      # Create dir and save data config
```

### `polynet.models.persistence`

Model and scaler persistence helpers.

```python
from polynet.models.persistence import (
    save_gnn_model, load_gnn_model,                  # .pt files via torch.save/load
    save_tml_model, load_tml_model,                  # .joblib files via joblib
    load_models_from_experiment,                      # Load all .pt and .joblib from models/
    load_scalers_from_experiment,                     # Load all .pkl scalers from models/
    load_dataframes,                                  # Load and validate descriptor CSVs
)
```

### `polynet.inference.predict`

Run predictions on unseen (external) data using trained models.

```python
from polynet.inference.predict import predict_unseen_tml, predict_unseen_gnn

# TML: run all saved TML models on new descriptor DataFrames
preds_df = predict_unseen_tml(models, scalers, descriptor_dfs, data_cfg)

# GNN: run all saved GNN models on a new PolymerGraphDataset
preds_df = predict_unseen_gnn(models, dataset, data_cfg)
```

### `polynet.inference.utils`

Utilities for assembling and ensembling prediction DataFrames.

```python
from polynet.inference.utils import (
    prepare_probs_df,          # Convert probability arrays to named DataFrame columns
    assemble_predictions,      # Merge per-model per-iteration DataFrames into one wide table
    merge_model_predictions,   # Join multiple single-model prediction DataFrames by index
    ensemble_predictions,      # Majority vote (classification) or mean (regression) ensemble
)
```

### `polynet.featurizer.allowable_sets`

Allowable values and defaults for atom/bond features used by the GNN featuriser.

```python
from polynet.featurizer.allowable_sets import atom_properties, bond_features
```

### `polynet.utils`

General-purpose utilities.

```python
from polynet.utils import (
    create_directory,        # mkdir -p, no-op if already exists
    save_data,               # Save DataFrame to .csv or .xlsx
    filter_dataset_by_ids,   # Subset a PyG dataset to a list of sample IDs
    extract_number,          # Parse trailing integer from a model filename (e.g. "model_3.pt")
)
from polynet.utils.statistical_analysis import significance_marker   # p-value → "*", "**", "***"
from polynet.utils.data_preprocessing import (
    keep_only_numerical_columns,   # Drop non-numeric columns from a DataFrame
    check_column_is_numeric,       # Check whether a named column is numeric
)
```

---

## Configuration Reference

The YAML config controls every pipeline stage.

### `experiment`

```yaml
experiment:
  name: "my_experiment"
  output_dir: "results/my_experiment"
  random_seed: 42
```

### `data`

```yaml
data:
  data_path: "data/polymers.csv"
  id_col: "polymer_id"
  target_variable_col: "Tg"
  target_variable_name: "Glass Transition Temperature (°C)"
  problem_type: "regression"       # "regression" or "classification"
  num_classes: 1                   # 1 for regression, N for N-class classification
  class_names: null                # Optional: {0: "inactive", 1: "active"}
  smiles_cols:
    - "monomer1_smiles"
    - "monomer2_smiles"
```

### `representations`

```yaml
representations:
  weights_col:                     # null to treat all monomers as equally weighted
    monomer1_smiles: "weight_fraction_1"
    monomer2_smiles: "weight_fraction_2"
  molecular_descriptors:
    RDKit: []                      # Include RDKit descriptors
    Morgan: []                     # Morgan fingerprints
    PolyBERT: []                   # PolyBERT fingerprints (requires psmiles)
  smiles_merge_approach: "weighted_average"   # weighted_average | average | concatenate
```

### `splitting`

```yaml
splitting:
  split_type: "TrainValTest"       # TrainValTest | LeaveOneOut
  split_method: "Random"           # Random | Scaffold
  n_bootstrap_iterations: 3
  val_ratio: 0.15
  test_ratio: 0.15
  train_set_balance: null          # Optional: balance training set (0.0–1.0)
```

### `gnn_training`

Each architecture block lists its hyperparameters. Leave the block empty (`{}`) to trigger automatic HPO via Ray Tune.

```yaml
gnn_training:
  train_gnn: true
  share_gnn_parameters: true

  gnn_convolutional_layers:
    GCN:
      LearningRate: 0.001
      BatchSize: 32
      improved: false
      embedding_dim: 64
      n_convolutions: 3
      readout_layers: 2
      dropout: 0.1
      pooling: "GlobalMeanPool"

    GAT:
      LearningRate: 0.001
      BatchSize: 32
      num_heads: 4
      embedding_dim: 64
      n_convolutions: 3

    MPNN: {}                       # Empty → triggers automatic HPO

training:
  epochs: 250
```

**Available architectures:** `GCN`, `GAT`, `CGGNN`, `MPNN`, `GraphSAGE`, `TransformerGNN`

**Architecture-specific parameters:**

| Architecture | Parameter | Description |
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

### `tml_models`

```yaml
tml_models:
  train_tml: false
  selected_models:
    - RandomForest
    - XGBoost

feature_preprocessing:
  scaler: "StandardScaler"         # StandardScaler | MinMaxScaler | RobustScaler | NoTransformation
```

**Available models:** `RandomForest`, `XGBoost`, `SupportVectorMachine`, `LogisticRegression`, `LinearRegression`

### `explainability`

```yaml
explainability:
  enabled: false
  algorithm: "IntegratedGradients"
  fragmentation: "brics"           # brics | recap
  normalisation: "Local"           # Local | Global
  cutoff: 0.05
  explain_mol_ids: null            # null = first 5 test-set samples
```

**Available attribution algorithms:** `IntegratedGradients`, `Saliency`, `InputXGradient`, `GuidedBackprop`, `Deconvolution`, `ShapleyValueSampling`, `GNNExplainer`

### `prediction`

```yaml
prediction:
  enabled: true
  data_path: "data/new_polymers.csv"
```

---

## Running the Pipeline

### CLI flags

| Flag | Description |
|---|---|
| `--config PATH` | Path to the YAML config file (required) |
| `--epochs N` | Override `training.epochs` from the config |
| `--task regression/classification` | Override `data.problem_type` |
| `--no-gnn` | Skip all GNN stages |
| `--no-tml` | Skip all TML stages |
| `--no-explain` | Skip the explainability stage |
| `--predict-data PATH` | Path to a CSV of unseen samples to predict after training |
| `--root PATH` | Project root for resolving relative paths (default: current directory) |

### Examples

```bash
# Quick smoke test
python scripts/run_pipeline.py --config configs/experiment.yaml --epochs 5

# GNN only, no TML, no explainability
python scripts/run_pipeline.py --config configs/experiment.yaml --no-tml --no-explain

# Classification task
python scripts/run_pipeline.py --config configs/experiment.yaml --task classification

# Train and predict on external data in one command
python scripts/run_pipeline.py --config configs/experiment.yaml --predict-data data/test_set.csv

# Predict only on an already-trained experiment
python scripts/run_pipeline.py --config configs/experiment.yaml --no-gnn --no-tml --predict-data data/test_set.csv
```

### Predicting on external data

After training, PolyNet can predict the target property for new, unseen samples. The unseen CSV must contain the same SMILES column(s) used during training. The target column is optional — if present, per-model metrics are computed automatically.

**Via CLI:**

```bash
python scripts/run_pipeline.py --config configs/experiment.yaml --predict-data data/new_polymers.csv
```

**Via YAML config:**

```yaml
prediction:
  enabled: true
  data_path: "data/new_polymers.csv"
```

**Via the Streamlit app:** open the **Predict** page, select an experiment, and upload a CSV.

**Via Python API:**

```python
import pandas as pd
from polynet.pipeline import predict_external
from polynet.config.io import load_options
from polynet.config.paths import (
    data_options_path,
    representation_options_path,
    unseen_predictions_experiment_parent_path,
)
from polynet.config.schemas import DataConfig, RepresentationConfig

experiment_path = Path("results/my_experiment")
data_cfg = load_options(data_options_path(experiment_path), DataConfig)
repr_cfg = load_options(representation_options_path(experiment_path), RepresentationConfig)

df = pd.read_csv("data/new_polymers.csv")
out_dir = unseen_predictions_experiment_parent_path("new_polymers.csv", experiment_path)

predictions, metrics = predict_external(
    data=df,
    data_cfg=data_cfg,
    repr_cfg=repr_cfg,
    experiment_path=experiment_path,
    out_dir=out_dir,
    dataset_name="new_polymers.csv",
)
```

Outputs are written to `{output_dir}/unseen_predictions/{filename}/`:

```
results/my_experiment/unseen_predictions/new_polymers/
├── predictions.csv          # Per-model predictions + ensemble columns
├── metrics.json             # Per-model metrics (only when target column is present)
└── representation/
    └── GNN/
        └── raw/             # Raw graph data used by the GNN featuriser
```

The same `predict_external` function is used by both the CLI and the Streamlit app, guaranteeing identical results regardless of entry point.

---

## Outputs

Everything is written under `experiment.output_dir`:

```
results/my_experiment/
├── config_used.yaml             # Exact configuration used (for reproducibility)
├── split_indices.json           # Train/val/test sample IDs for each iteration
├── data_options.json            # Saved DataConfig
├── representation_options.json  # Saved RepresentationConfig
├── general_options.json         # Saved GeneralConfig
├── train_gnn_options.json       # Saved TrainGNNConfig  (if GNN was trained)
├── train_tml_options.json       # Saved TrainTMLConfig  (if TML was trained)
├── ml_results/
│   ├── predictions.csv          # Full predictions DataFrame
│   ├── metrics.json             # All metrics by iteration, model, and split
│   ├── models/                  # Saved model files
│   │   ├── GCN_1.pt             # GNN model (iteration 1)
│   │   ├── rf-Morgan_1.joblib   # TML model (iteration 1)
│   │   └── Morgan.pkl           # Feature scaler for Morgan descriptor
│   └── plots/
│       ├── GCN_1_learning_curve.png
│       ├── GCN_1_parity_plot.png
│       └── rf-Morgan_1_parity_plot.png
├── unseen_predictions/
│   └── new_polymers/
│       ├── predictions.csv
│       ├── metrics.json
│       └── representation/GNN/raw/
├── explanations/
│   ├── fragment_attributions.png
│   └── poly_0001_heatmap.png
└── gnn_hyp_opt/                 # Ray Tune HPO results (when HPO was triggered)
```

---

## Streamlit GUI

Launch with:

```bash
streamlit run polynet/app/Welcome_to_PolyNet.py
```

The app is organised into six sequential pages:

| Page | Purpose |
|---|---|
| **1 — Create Experiment** | Name the experiment, upload data, configure target and SMILES columns |
| **2 — Representation** | Select molecular descriptors (RDKit, Morgan, PolyBERT) and GNN featurisation options |
| **3 — Train Models** | Configure GNN architectures and TML models, set splits, start training |
| **4 — Predict** | Upload an unseen CSV and run prediction using a trained experiment |
| **5 — Explain Models** | Run atom attribution and visualise fragment importance heatmaps |
| **6 — Analyse Results** | Inspect training metrics, parity plots, ROC curves, and statistical comparisons |

The GUI is an optional extension of the core package. `polynet.app` can be omitted entirely — the core pipeline works without Streamlit.

---

## Architecture: Core vs GUI

All reusable logic lives in the core `polynet` package:

```
polynet/                  ← no Streamlit dependency
├── pipeline/stages.py    ← all pipeline stages
├── config/io.py          ← config serialisation
├── config/paths.py       ← path helpers
├── experiment/manager.py ← experiment filesystem
├── models/persistence.py ← model save/load
├── inference/predict.py  ← prediction on new data
└── ...

polynet/app/              ← Streamlit (optional)
├── services/             ← thin re-exports of core modules
│   ├── configurations.py → from polynet.config.io import ...
│   ├── model_training.py → from polynet.models.persistence import ...
│   └── ...
└── pages/                ← Streamlit page orchestration
```

The `polynet.app.services.*` modules are thin re-export shims — they import from core and re-export under the original names, so existing app pages continue to work unchanged. Any code that previously imported from `polynet.app.services.configurations` still works; the implementation now lives in `polynet.config.io`.

---

## Debugging

An integration test runs each pipeline stage independently using synthetic polymer data. All stages run regardless of prior failures, giving a complete picture in one pass:

```bash
# Full pipeline smoke test
python scripts/integration_test.py

# Classification task
python scripts/integration_test.py --task classification

# TML stages only (much faster — no graph building)
python scripts/integration_test.py --tml-only

# More samples, longer training
python scripts/integration_test.py --samples 80 --epochs 20
```

Example output:

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
  ✓ PASS   10. GNN metrics                 (0.1s)
  ✓ PASS   11. GNN result plots            (2.0s)
  ✓ PASS   12. TML training                (0.8s)
  ✓ PASS   13. TML inference               (0.1s)
  ✓ PASS   14. TML metrics                 (0.0s)

  Total: 14 passed, 0 failed, 0 skipped
```

---

## Extending PolyNet

### Adding a new GNN architecture

1. Create `polynet/models/gnn/myarch.py` following the pattern in `gcn.py`
2. Add `MyArchBase`, `MyArchClassifier`, `MyArchRegressor`
3. Register in `polynet/models/gnn/__init__.py`
4. Add `(Network.MyArch, ProblemType.Regression)` and `(Network.MyArch, ProblemType.Classification)` to `_NETWORK_REGISTRY` in `polynet/factories/network.py`
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
