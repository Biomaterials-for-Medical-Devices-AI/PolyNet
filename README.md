# PolyNet

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch%20Geometric-enabled-3C2179)](https://pytorch-geometric.readthedocs.io/)
[![RDKit](https://img.shields.io/badge/RDKit-supported-informational)](https://www.rdkit.org/)
[![Tests](https://github.com/Biomaterials-for-Medical-Devices-AI/PolyNet/actions/workflows/tests.yml/badge.svg)](https://github.com/Biomaterials-for-Medical-Devices-AI/PolyNet/actions/workflows/tests.yml)
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
- **Target variable scaling** — six scaling strategies for regression targets (StandardScaler, MinMaxScaler, RobustScaler, Log₁₀, Log(1+y), or none); scaler is fit on the training set only and automatically inverse-transformed before metrics and plots so all results are reported in the original target units
- **Polymer descriptor fusion** — user-supplied experimental or computed polymer-level features (e.g. molecular weight, chain length) can be concatenated to vectorial descriptor representations and, for GNN models, fused into the graph embedding after pooling so the FFN receives both learned and given features
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
│   ├── enums.py               # ProblemType, MolecularDescriptor, Network, AtomFeature, TargetTransformDescriptor, etc.
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
│       ├── target_preprocessing.py   # TargetTransformConfig
│       ├── plotting.py        # PlottingConfig
│       └── base.py            # PolynetBaseModel (shared Pydantic base)
│
├── data/                      # Data loading and preprocessing
│   ├── __init__.py            # sanitise_df() — strip SMILES/weight cols, ensure target last
│   ├── loader.py              # load_dataset() — CSV loading with validation
│   ├── preprocessing.py       # sanitise_df(), SMILES-to-string conversion, TargetScaler
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
│   └── utils.py               # prepare_probs_df(), assemble_predictions()
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
│   ├── plots.py               # Learning curves, parity plots, ROC curves, confusion matrices,
│   │                          # plot_pvalue_matrix(), plot_bootstrap_boxplots()
│   └── utils.py               # save_plot()
│
├── utils/                     # General utilities
│   ├── __init__.py            # create_directory(), save_data(), filter_dataset_by_ids(),
│   │                          # extract_number(), prepare_probs_df()
│   ├── chem_utils.py          # check_smiles_cols(), determine_string_representation()
│   ├── data_preprocessing.py  # keep_only_numerical_columns(), check_column_is_numeric(),
│   │                          # class_balancer(), print_class_balance()
│   └── statistical_analysis.py # metrics_pvalue_matrix(), significance_marker(), mcnemar_*
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

tests/                     # pytest suite (see Testing section)
├── conftest.py            # Shared fixtures
├── fixtures/              # Reference CSVs for regression tests
├── test_merging.py        # Unit tests for merging strategies
├── test_compute_rdkit.py  # Integration tests for compute_rdkit_descriptors
├── test_config_loader.py  # Tests for config normalisation pipeline
├── test_persistence.py    # Tests for load_dataframes column validation
└── test_descriptors_regression.py  # Numerical regression tests vs reference CSVs
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
    train_gnn,              # Train a GNN ensemble, save .pt model files → (trained, loaders, target_scalers)
    run_gnn_inference,      # Run GNN inference on all splits → predictions DataFrame
    train_tml,              # Train a TML ensemble, save .joblib + .pkl files → (trained, data, scalers, target_scalers)
    run_tml_inference,      # Run TML inference on all splits → predictions DataFrame
    compute_metrics,        # Compute evaluation metrics from predictions DataFrame
    plot_results_stage,     # Generate learning curves and result plots
    predict_external,       # Predict on unseen data using a trained experiment
    run_explainability,     # Run attribution-based explainability and save heatmaps
)
from polynet.config.schemas import TargetTransformConfig   # Target scaling config (regression)
from polynet.config.enums import TargetTransformDescriptor # Enum of scaling strategies
```

Key signatures for the training and inference stages:

```python
# train_gnn returns a 3-tuple; pass target_cfg to enable target scaling
gnn_trained, gnn_loaders, gnn_target_scalers = train_gnn(
    dataset, split_indexes, data_cfg, gnn_cfg,
    random_seed=42, out_dir=experiment_path,
    target_cfg=TargetTransformConfig(strategy=TargetTransformDescriptor.StandardScaler),
)

# run_gnn_inference — pass target_scalers to inverse-transform predictions
gnn_preds_df = run_gnn_inference(
    trained_models=gnn_trained, loaders=gnn_loaders,
    data_cfg=data_cfg, split_cfg=split_cfg,
    target_scalers=gnn_target_scalers,
)

# train_tml returns a 4-tuple
tml_trained, tml_training_data, tml_scalers, tml_target_scalers = train_tml(
    desc_dfs=dataframes, split_indexes=split_indexes,
    data_cfg=data_cfg, tml_cfg=tml_cfg, preprocessing_cfg=preprocessing_cfg,
    random_seed=42, out_dir=experiment_path,
    target_cfg=TargetTransformConfig(strategy=TargetTransformDescriptor.Log10),
)

# run_tml_inference — pass target_scalers to inverse-transform predictions
tml_preds_df = run_tml_inference(
    trained=tml_trained, training_data=tml_training_data,
    data_cfg=data_cfg, split_cfg=split_cfg,
    target_scalers=tml_target_scalers,
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
    target_transform_options_path,          # <experiment>/target_transform_options.json
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

Utilities for assembling prediction DataFrames.

```python
from polynet.inference.utils import (
    prepare_probs_df,          # Convert probability arrays to named DataFrame columns
    assemble_predictions,      # Merge per-model per-iteration DataFrames into one wide table
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
  smiles_merge_approach: "weighted_average"   # weighted_average | concatenate | no_merging
  polymer_descriptors:             # Optional: column names from your CSV to use as given features
    - "molecular_weight"
    - "degree_of_polymerisation"
```

---

## Polymer Descriptors

`polymer_descriptors` lets you inject experimental or pre-computed polymer-level features — things like measured molecular weight, degree of polymerisation, or any other numeric column in your CSV — directly into the modelling pipeline alongside the computed molecular representations.

### How it works

**Vectorial representations (TML)**

After all per-monomer molecular descriptors are computed and merged (weighted average, concatenation, or no-merging), the selected polymer descriptor columns are horizontally concatenated to every representation DataFrame. The final feature matrix seen by the TML model therefore contains both the structurally derived features and the given polymer-level features.

**Graph representations (GNN)**

Polymer descriptors cannot be part of node or edge features because they characterise the whole polymer, not individual atoms or bonds. Instead they are stored as a **graph-level tensor** on each PyG `Data` object during featurisation (`CustomPolymerGraph`). During the forward pass they are concatenated to the pooled graph embedding — **after pooling and after any monomer weight multiplication** — so the FFN readout receives both the learned graph representation and the experimental polymer context. The first readout layer is automatically widened to accommodate the extra dimensions.

```
GNN forward pass with polymer descriptors:

  node features → message passing → [optional weighting] → pooling
                                                               ↓
                                              [embedding, dim = embedding_dim]
                                                               ↓
                                     cat([embedding, polymer_descriptors], dim=1)
                                                               ↓
                                              FFN readout → prediction
```

### Configuration

Specify the column names to use in the `representations` section of your config:

```yaml
representations:
  polymer_descriptors:
    - "molecular_weight"
    - "degree_of_polymerisation"
```

The listed columns must be present as numeric columns in your input CSV. They are used by both the TML pipeline (if `molecular_descriptors` are configured) and the GNN pipeline (if `node_features` are configured) automatically — no other changes are required.

Omitting `polymer_descriptors` (or setting it to `null`) disables the feature entirely with no impact on existing experiments.

### Python API

```python
from polynet.config.schemas import RepresentationConfig

repr_cfg = RepresentationConfig(
    node_features={...},
    edge_features={...},
    polymer_descriptors=["molecular_weight", "degree_of_polymerisation"],
)

# build_graph_dataset picks up the columns automatically
dataset = build_graph_dataset(data=df, data_cfg=data_cfg, repr_cfg=repr_cfg, out_dir=out_dir)
```

---

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

### `target_transform`

Applies scaling to the **regression target variable**. The scaler is fit on the training set only; validation and test predictions are inverse-transformed before metrics and plots so all results remain in the original target units. This section is ignored for classification problems.

```yaml
target_transform:
  strategy: "standard_scaler"   # See available strategies below
```

**Available strategies:**

| Strategy | Description |
|---|---|
| `no_transformation` | No scaling (default) |
| `standard_scaler` | Standardise to zero mean and unit variance |
| `min_max_scaler` | Scale to the [0, 1] interval |
| `robust_scaler` | IQR-based scaling; outlier-resistant |
| `log10` | Apply log₁₀ transform — **requires all targets > 0** |
| `log1p` | Apply log(1 + y) transform — **requires all targets > −1** |

> **Note:** Using `log10` or `log1p` with non-positive target values will raise a `ValueError` at fit time with a descriptive message.

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

## Target Variable Scaling

For regression experiments, PolyNet can scale the target variable before training and automatically recover the original scale before computing metrics and generating plots. This can stabilise training (especially for targets spanning several orders of magnitude) without affecting how results are reported.

### How it works

1. A `TargetScaler` is **fit on the training set only** — no information from the validation or test sets leaks into the scaler.
2. The scaler transforms `y_train` (and `y_val` / `y_test` during training) so the model optimises on the scaled values.
3. At inference time, all predictions are **inverse-transformed back to the original target range** before metrics (R², RMSE, MAE) and plots (parity plots) are computed.
4. `y_true` values in the predictions DataFrame are always in the original scale.
5. Each bootstrap iteration has its own independently fitted `TargetScaler`.

### Available strategies

| Strategy | Enum value | Description |
|---|---|---|
| No scaling (default) | `no_transformation` | Identity — no change to y |
| Standardisation | `standard_scaler` | Subtract mean, divide by std (sklearn `StandardScaler`) |
| Min–Max | `min_max_scaler` | Scale to [0, 1] (sklearn `MinMaxScaler`) |
| Robust | `robust_scaler` | IQR-based — outlier-resistant (sklearn `RobustScaler`) |
| Log₁₀ | `log10` | `y → log₁₀(y)`; **all training targets must be > 0** |
| Log(1 + y) | `log1p` | `y → log(1 + y)`; **all training targets must be > −1** |

### Usage

**YAML config (CLI):**

```yaml
target_transform:
  strategy: "standard_scaler"
```

**Python API:**

```python
from polynet.config.schemas import TargetTransformConfig
from polynet.config.enums import TargetTransformDescriptor

target_cfg = TargetTransformConfig(strategy=TargetTransformDescriptor.Log10)
gnn_trained, gnn_loaders, gnn_target_scalers = train_gnn(..., target_cfg=target_cfg)
```

**Streamlit GUI:** on Page 3 (Train Models), a *Target Variable Scaling* section appears automatically for regression experiments. Select a strategy from the dropdown; a tooltip explains domain constraints for the log transforms.

### Saved files

Target scalers are serialised alongside the model files in `ml_results/models/`:

- **GNN**: `target_scaler_{iteration}.pkl` (e.g. `target_scaler_1.pkl`)
- **TML**: `target_{descriptor_name}_{iteration}.pkl` (e.g. `target_Morgan_1.pkl`)

When `predict_external` is called (CLI, GUI, or Python API), these files are loaded automatically and applied to new predictions — no manual wiring is needed.

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
│   │   ├── Morgan.pkl           # Feature scaler for Morgan descriptor
│   │   ├── target_scaler_1.pkl  # GNN target scaler (iteration 1; omitted when no_transformation)
│   │   └── target_Morgan_1.pkl  # TML target scaler for Morgan, iteration 1
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
| **3 — Train Models** | Configure GNN architectures and TML models, set data splits, optionally apply target variable scaling (regression), start training |
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

## Testing

PolyNet ships with a [pytest](https://docs.pytest.org/) suite under `tests/`. The suite is split into fast unit tests (no RDKit, no GPU) and slightly slower integration tests (RDKit only, ~4 s total). No network access, PolyBERT model, or PolyMetriX installation is required to run most tests.

### Running the tests

```bash
# Activate your environment first, then:
python -m pytest tests/ -v
```

To run individual modules:

```bash
python -m pytest tests/test_merging.py -v          # Merging logic only  (~0.5 s)
python -m pytest tests/test_compute_rdkit.py -v    # RDKit integration   (~2 s)
python -m pytest tests/test_config_loader.py -v    # Config normalisation (~0.5 s)
python -m pytest tests/test_persistence.py -v      # Persistence layer    (~0.5 s)
python -m pytest tests/test_descriptors_regression.py -v  # Regression vs reference CSVs (~2 s)
```

To list all collected tests without running them:

```bash
python -m pytest tests/ --co -q
```

### Test modules

| Module | What it covers | External deps |
|--------|----------------|--------------|
| `test_merging.py` | Unit tests for `merge_weighted`, `_merge_concatenate`, `_single_smiles`, and the `_merge` dispatch — no RDKit, uses manually constructed DataFrames | None |
| `test_compute_rdkit.py` | Output shape, column cleanliness, numeric sanity, and index preservation for `compute_rdkit_descriptors` across all merging strategies | RDKit |
| `test_config_loader.py` | Deprecated field migration (`rdkit_descriptors`, `df_descriptors`, `polybert_fp`), enum compatibility maps, field renames, and unrecognised-key handling in `build_experiment_config` | None |
| `test_persistence.py` | `_resolve_features` (pure function) and `load_dataframes` column validation — I/O is fully mocked | None |
| `test_descriptors_regression.py` | Numerical regression against user-supplied reference CSVs in `tests/fixtures/`; locks in exact pipeline outputs across all merging strategies | RDKit |

### Shared fixtures

Common fixtures are defined in `tests/conftest.py` and are automatically available to all test modules:

| Fixture | Type | Description |
|---------|------|-------------|
| `two_monomer_df` | `pd.DataFrame` | Two-monomer polymer dataset with percentage-scale weight columns |
| `single_monomer_df` | `pd.DataFrame` | Single-monomer dataset for `NoMerging` tests |
| `known_df_dict` | `dict[str, pd.DataFrame]` | Manually constructed df_dict with integer-friendly values for exact arithmetic assertions |
| `single_key_df_dict` | `dict[str, pd.DataFrame]` | Single-key df_dict |
| `empty_data_index` | `pd.DataFrame` | Empty-column DataFrame mirroring `compute_rdkit_descriptors`'s internal `data[[]]` pattern |

### Regression test fixtures

The regression tests in `test_descriptors_regression.py` compare live pipeline output against reference CSVs in `tests/fixtures/`:

| File | Contents |
|------|----------|
| `input_polymers.csv` | Input dataset — columns: `smiles_A`, `smiles_B`, `weight_A`, `weight_B`, target |
| `expected_weighted_average.csv` | Reference output for `WeightedAverage` merging |
| `expected_concatenate.csv` | Reference output for `Concatenate` merging |
| `expected_no_merging.csv` | Reference output for `NoMerging` on `smiles_A` only |

Weight columns are in **percentage scale (0–100)**, matching the pipeline formula `weight / 100`. To regenerate the reference files after an intentional change to descriptor logic, run:

```python
# scripts/generate_test_fixtures.py
import pandas as pd
from polynet.config.enums import DescriptorMergingMethod
from polynet.featurizer import compute_rdkit_descriptors

DESCRIPTORS = ["MolWt", "TPSA", "NumRotatableBonds"]
FIXTURES = "tests/fixtures"

df = pd.read_csv(f"{FIXTURES}/input_polymers.csv", index_col=0)

compute_rdkit_descriptors(df, ["smiles_A", "smiles_B"], DESCRIPTORS,
    DescriptorMergingMethod.WeightedAverage,
    weights_col={"smiles_A": "weight_A", "smiles_B": "weight_B"},
).to_csv(f"{FIXTURES}/expected_weighted_average.csv")

compute_rdkit_descriptors(df, ["smiles_A", "smiles_B"], DESCRIPTORS,
    DescriptorMergingMethod.Concatenate,
).to_csv(f"{FIXTURES}/expected_concatenate.csv")

compute_rdkit_descriptors(df, ["smiles_A"], DESCRIPTORS,
    DescriptorMergingMethod.NoMerging,
).to_csv(f"{FIXTURES}/expected_no_merging.csv")
```

### Writing new tests

- Place test files in `tests/` with the `test_` prefix.
- Add shared fixtures to `tests/conftest.py`.
- Tests that require RDKit but not GPU, PolyBERT, or PolyMetriX are fine in the main suite — they run in ~2 s.
- Tests that require PolyBERT or a GPU should be decorated with `@pytest.mark.slow` and skipped in CI by default.
- Always prefer testing public API (`compute_rdkit_descriptors`, `build_vector_representation`) over private helpers unless the private function has complex logic that warrants direct testing (e.g. `_merge_concatenate`).

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
