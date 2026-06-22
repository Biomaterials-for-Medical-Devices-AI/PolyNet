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

Predicting polymer properties from chemical structure is challenging because polymers are repeating units of one or more monomers combined in varying ratios. PolyNet handles this by representing each polymer as a graph where atoms are nodes, bonds are edges, and per-monomer molar ratios (normalised to sum to 1 across the participating monomers) are encoded as node attributes. This graph is fed into GNN architectures that learn structure–property relationships directly from the molecular graph.

PolyNet also supports traditional ML workflows using molecular descriptor vectors (RDKit descriptors, PolyBERT fingerprints, and others), enabling direct comparison between GNN and TML approaches on the same dataset and splits.

---

## Features

- **Six GNN architectures** — GCN, GAT, CGGNN, MPNN, GraphSAGE, TransformerGNN — each with regression and classification variants
- **Traditional ML** — Random Forest, XGBoost, SVM, Logistic Regression, Linear Regression
- **Multi-monomer polymers** — handles copolymers with any number of monomers and molar ratios; ratios are normalised per polymer to sum to 1 across the participating monomers (any scale works — fractions, percentages, or arbitrary ratios such as 1:1:2), and ratio-0 monomers are excluded from the graph entirely so homopolymers written as `(monomer, 100/0)` are invariant to the empty SMILES column
- **Per-monomer pooling** — `PerMonomerPooling` weighting mode pools each monomer's nodes separately and combines them as `Σ wᵢ·pool(monomerᵢ)`, removing the atom-count bias that arises when weighting before pooling on mixed-size copolymers (wD-MPNN-style weighted-mean pooling is also supported)
- **PSMILES attachment-point handling** — opt-in `IsAttachmentPoint` atom feature strips `*` (wildcard) atoms from the graph and flags the atoms that were attached to them, replacing dangling pseudo-element nodes with an explicit boolean marker
- **PolyMetriX descriptors** — polymer-aware chemical descriptors (molecular weight, ring counts, TPSA, etc.) computed on the full repeat unit, side chain, or backbone, with configurable aggregation; three modes — `side_chain`, `backbone`, and `polymer` — can be used in any combination
- **Automatic HPO** — Ray Tune with configurable split strategy (cross-validation, holdout, repeated holdout); ASHA early stopping active for holdout-based strategies
- **Bootstrap ensemble training** — configurable number of train/val/test splits for robust uncertainty estimates
- **Target variable scaling** — six scaling strategies for regression targets (StandardScaler, MinMaxScaler, RobustScaler, Log₁₀, Log(1+y), or none); scaler is fit on the training set only and automatically inverse-transformed before metrics and plots so all results are reported in the original target units
- **Polymer descriptor fusion** — user-supplied experimental or computed polymer-level features (e.g. molecular weight, chain length) can be concatenated to vectorial descriptor representations and, for GNN models, fused into the graph embedding after pooling so the FFN receives both learned and given features
- **GNN fragment-level explainability** — chemistry-masking attribution (Wellawatte et al., Nat. Commun. 2023): each fragment's importance is measured as the change in prediction when that fragment is masked from the graph pooling step; results are aggregated to functional-group level using BRICS or Murcko scaffold fragmentation; supports global distribution plots (ridge, bar, strip) and per-molecule attribution heatmaps
- **TML SHAP explainability** — SHAP-based attribution for all traditional ML models using auto-selected explainers (`TreeExplainer` for RF/XGBoost, `LinearExplainer` for linear models, `KernelExplainer` for SVM); global summaries and per-instance plots are rendered with the native `shap` package (beeswarm / bar / violin and waterfall / force / bar); SHAP values are cached to CSV and reused across runs
- **Configurable explanation display** — view local explanations averaged across the model ensemble or as one plot per model × molecule, with selectable attribution normalisation (local / per-model / global / none)
- **Statistical model comparison** — pairwise McNemar (classification) and Wilcoxon (regression) tests on the Analyse Results page, with multiple-comparison correction (Holm-Bonferroni, Bonferroni, Benjamini-Hochberg)
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
| `ratio_1` | No | Molar ratio of monomer 1. Omit to treat monomers as equally weighted |
| `ratio_2` | No | Molar ratio of monomer 2 |
| `target` | Yes | Property to predict (numeric for regression, integer class for classification) |

Copolymers with any number of monomers are supported by adding one SMILES column and one molar-ratio column per monomer and listing them in the config. Ratios are normalised per polymer to sum to 1 across the participating monomers, so they need not sum to 1 or 100; a ratio of 0 excludes that monomer. The target column is optional when predicting on unseen data.

---

## Documentation

Full documentation lives in [`docs/`](docs/README.md):

- **[Configuration reference](docs/configuration.md)** — every YAML section, including automatic HPO
- **[Descriptors](docs/descriptors.md)** — polymer descriptor fusion and the PolyMetriX integration
- **[Explainability](docs/explainability.md)** — GNN masking & TML SHAP attribution, normalisation, averaged vs per-model display, and statistical model comparison
- **[Running the pipeline & outputs](docs/running-and-outputs.md)** — CLI flags, predicting on external data, target scaling, and the output directory layout
- **[API reference](docs/api-reference.md)** — project structure and the public Python API
- **[GUI & architecture](docs/gui-and-architecture.md)** — the Streamlit application and the core-vs-GUI design
- **[Development](docs/development.md)** — running the test suite and extending PolyNet

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
