# Running the Pipeline & Outputs

- [CLI flags](#cli-flags)
- [Predicting on external data](#predicting-on-external-data)
- [Target-variable scaling](#target-variable-scaling)
- [Outputs](#outputs)
- [Debugging](#debugging)

---

## CLI flags

```bash
python scripts/run_pipeline.py --config configs/experiment.yaml
```

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

## Predicting on external data

After training, PolyNet can predict the target property for new, unseen samples. The
unseen CSV must contain the same SMILES column(s) used during training. The target
column is optional — if present, per-model metrics are computed automatically.

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

**Via the Streamlit app:** open the **Predict** page, select an experiment, and upload
a CSV.

**Via Python API:**

```python
import pandas as pd
from pathlib import Path
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

The same `predict_external` function is used by both the CLI and the Streamlit app,
guaranteeing identical results regardless of entry point.

## Target-variable scaling

For regression experiments, PolyNet can scale the target variable before training and
automatically recover the original scale before computing metrics and generating plots.
This can stabilise training (especially for targets spanning several orders of
magnitude) without affecting how results are reported. Configured via the
[`target_transform`](configuration.md#target_transform) section.

### How it works

1. A `TargetScaler` is **fit on the training set only** — no information from the
   validation or test sets leaks into the scaler.
2. The scaler transforms `y_train` (and `y_val` / `y_test` during training) so the
   model optimises on the scaled values.
3. At inference time, all predictions are **inverse-transformed back to the original
   target range** before metrics (R², RMSE, MAE) and plots (parity plots) are computed.
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

**Python API:**

```python
from polynet.config.schemas import TargetTransformConfig
from polynet.config.enums import TargetTransformDescriptor

target_cfg = TargetTransformConfig(strategy=TargetTransformDescriptor.Log10)
gnn_trained, gnn_loaders, gnn_target_scalers = train_gnn(..., target_cfg=target_cfg)
```

**Streamlit GUI:** on Page 3 (Train Models), a *Target Variable Scaling* section
appears automatically for regression experiments. Select a strategy from the dropdown;
a tooltip explains domain constraints for the log transforms.

### Saved files

Target scalers are serialised alongside the model files in `ml_results/models/`:

- **GNN**: `target_scaler_{iteration}.pkl` (e.g. `target_scaler_1.pkl`)
- **TML**: `target_{descriptor_name}_{iteration}.pkl` (e.g. `target_Morgan_1.pkl`)

When `predict_external` is called (CLI, GUI, or Python API), these files are loaded
automatically and applied to new predictions — no manual wiring is needed.

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
│   ├── fragment_attributions.png      # GNN global distribution plot
│   ├── poly_0001_heatmap.png          # GNN per-molecule attribution heatmap
│   ├── shap_morgan.csv                # TML SHAP value cache (one file per descriptor)
│   ├── morgan_shap_distribution.png   # TML global SHAP summary plot
│   └── ...                            # TML per-instance SHAP plots / CSVs
└── gnn_hyp_opt/                 # Ray Tune HPO results (when HPO was triggered)
```

The `predictions.csv` table holds one row per `(sample × bootstrap iteration)`, with a
`Set` column (train/val/test) and one predicted-value column per trained model.

## Debugging

An integration test runs each pipeline stage independently using synthetic polymer
data. All stages run regardless of prior failures, giving a complete picture in one
pass:

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
  ✓ PASS   10. GNN metrics                  (0.1s)
  ✓ PASS   11. GNN result plots             (2.0s)
  ✓ PASS   12. TML training                 (0.8s)
  ✓ PASS   13. TML inference                (0.1s)
  ✓ PASS   14. TML metrics                  (0.0s)

  Total: 14 passed, 0 failed, 0 skipped
```
