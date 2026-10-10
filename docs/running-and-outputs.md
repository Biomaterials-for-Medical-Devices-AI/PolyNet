# Running the Pipeline & Outputs

- [CLI flags](#cli-flags)
- [Predicting on external data](#predicting-on-external-data)
- [Target-variable scaling](#target-variable-scaling)
- [Outputs](#outputs)
- [Debugging](#debugging)

---

## CLI flags

```bash
polynet run --config configs/experiment.yaml
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

`python scripts/run_pipeline.py` accepts the same flags and is kept for existing workflows.

### Examples

```bash
# Quick smoke test
polynet run --config configs/experiment.yaml --epochs 5

# GNN only, no TML, no explainability
polynet run --config configs/experiment.yaml --no-tml --no-explain

# Classification task
polynet run --config configs/experiment.yaml --task classification

# Train and predict on external data in one command
polynet run --config configs/experiment.yaml --predict-data data/test_set.csv

# Predict only on an already-trained experiment
polynet run --config configs/experiment.yaml --no-gnn --no-tml --predict-data data/test_set.csv
```

## Predicting on external data

After training, PolyNet can predict the target property for new, unseen samples. The
unseen CSV must contain the same SMILES column(s) used during training. The target
column is optional — if present, per-model metrics are computed automatically.

**Via CLI:**

```bash
polynet run --config configs/experiment.yaml --predict-data data/new_polymers.csv
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
├── predictions.csv          # Per-model predictions + ensemble + applicability domain columns
├── metrics.json             # Per-model and ensemble metrics (only when target column is present)
├── applicability_domain.json  # Applicability domain settings, per-split cutoffs, expected-error bins
└── representation/
    └── GNN/
        └── raw/             # Raw graph data used by the GNN featuriser
```

### Ensemble predictions

An experiment trains one model per repeated random split for every GNN architecture
and every TML model × representation. When predicting new data, the models trained on
the different splits are combined into **ensembles**:

| Ensemble | Members |
|---|---|
| `{arch} Ensemble` (e.g. `GCN Ensemble`) | One GNN architecture, all splits |
| `GNN Ensemble` | All GNN architectures, all splits |
| `{model}-{representation} Ensemble` (e.g. `random forest-rdkit Ensemble`) | One TML model on one representation, all splits |

Columns added to `predictions.csv` for each ensemble:

| Task | Columns | Meaning |
|---|---|---|
| Regression | `{name} Ensemble Predicted {target}` | Mean of the member predictions |
| | `{name} Ensemble Std {target}` | Standard deviation of the member predictions (population, `ddof=0`) — the spread between members, not a calibrated uncertainty |
| Classification | `{name} Ensemble Predicted {target}` | Majority vote of the members (ties go to the smallest class label) |
| | `{name} Ensemble Vote Fraction {target}` | Share of members that voted for the ensemble class |

Ensembles need at least two members, i.e. `n_bootstrap_iterations ≥ 2` (with an even
number of splits, classification ties are possible). When the target column is present,
`metrics.json` holds the per-split metrics under `"1"`, `"2"`, … and the ensemble
metrics under `"ensemble"`, keyed by ensemble name. Classification ensembles only have
hard votes, so probability-based metrics (AUROC) are `null` for them. In the GUI the
ensembles appear as separate rows of the metrics tables.

### Applicability domain

Models are only reliable for polymers similar to the ones they were trained on. For
every prediction, PolyNet reports whether the new polymer lies in the **applicability
domain** of the models and the error to expect at its distance from the training set.

- **Distance.** A polymer's distance to the training set is its mean distance to its
  *k* nearest training polymers. Two distances are available (`metric`, one or both):
  - `ruzicka_morgan` (default): 1 − the min–max (Ruzicka) similarity, Σmin/Σmax, of
    ratio-weighted count fingerprints (Morgan by default: the monomer fingerprints
    averaged with their molar ratios, as in the representations). It is the
    generalisation of the Tanimoto coefficient to counts, and the same for every model.
    One domain per model family (`GNN`, `TML`).
  - `euclidean_model_inputs`: Euclidean distance in the inputs of the traditional
    models, i.e. their representation after the scaler and feature selection fitted on
    each split. One domain per representation (`TML rdkit`, `TML morgan`, …). GNNs have
    no such inputs and always use `ruzicka_morgan`.
- **Domain** (Tropsha, Gramatica & Gombar, *QSAR Comb. Sci.* 2003, 22, 69–77). The
  training polymers are placed relative to each other in the same way, giving a mean
  distance ⟨d⟩ and standard deviation σ. A new polymer is in the domain when its distance
  is at most ⟨d⟩ + Z·σ. Its **score** is distance / cutoff, so 1 is the boundary for
  either distance. With `representations.polymer_descriptors` (e.g. Mw), it must also lie
  within their training range.
- **Expected error** (Sheridan et al., *J. Chem. Inf. Comput. Sci.* 2004, 44,
  1912–1928). The held-out test polymers of every split are scored against that split's
  training polymers, and their errors are grouped into score bins (quantiles). A new
  polymer gets the mean test error of its bin: the absolute error (regression) or the
  accuracy (classification). Polymers farther out than every test polymer get no
  expected error, because no error was observed that far out.

The reference is what each model family was trained on: the training set for GNNs,
and training + validation for TML models when
`tml_models.include_validation_in_training` is true.

**Repeated splits.** Each split's models learned from a different training set, so every
split has its own domain (its own ⟨d⟩, σ and cutoff, listed per split in
`applicability_domain.json`). A new polymer is scored against each split, and the
results are combined like the ensembles: the mean score, the share of splits whose
domain contains it, and a majority vote for `In Domain`. For the expected error, the
test polymers of each split are scored against that split's training set only, so no
polymer is ever compared with a set it was trained on. Their (score, error) pairs from
all splits are pooled into one table of score bins per model, and a new polymer's
expected error is the mean of its per-split look-ups. A single reference merged from
all training sets is deliberately not used: each split's test polymers belong to the
training sets of other splits, so the calibration would be optimistic. Deterministic
samplers (e.g. `kennard_stone`) repeat the same split, so all domains are identical and
`In Domain Fraction` is 0 or 1.

`{scope}` is `GNN` / `TML` for `ruzicka_morgan` and `TML {representation}` for
`euclidean_model_inputs`; `{metric}` is `Ruzicka` or `Euclidean`.

| Column | Meaning |
|---|---|
| `{scope} AD {metric} Score` | Distance to the *k* nearest training polymers / domain cutoff (1 = boundary), averaged over the splits |
| `{scope} AD {metric} In Domain Fraction` | Share of the splits in whose domain the polymer lies |
| `{scope} AD {metric} In Domain` | In the domain of at least half of the splits |
| `{family} AD Descriptors In Range Fraction` | Share of the splits whose training range contains the polymer descriptors (only with `polymer_descriptors`) |
| `{name} AD {metric} Expected Abs Error {target}` / `… Expected Accuracy {target}` | Held-out test error of the split models of the ensemble `{name}` at a similar score |

On the showcase datasets, `ruzicka_morgan` separated new chemistry and ranked errors
clearly better than `euclidean_model_inputs` on chemically diverse polymers (Tg), and
both performed alike when only monomer ratios change (SNR copolymers).

The domain is computed at prediction time from files every experiment already saves:
the training dataset, `split_indices.json`, `ml_results/predictions.csv` and, for
`euclidean_model_inputs`, the training descriptors and per-split scalers. It therefore
also works for experiments trained before this feature. If one of these files is
missing, a warning is logged: without `split_indices.json` the whole dataset is the
reference and there is no expected error, and without the training dataset the
predictions are returned without the domain. Settings are under
[`prediction.applicability_domain`](configuration.md#prediction), and in the
**Applicability domain** expander of the Predict page.

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
├── hpo_search_spaces.json       # HPO grids actually searched (only when automatic HPO runs)
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
│   │   ├── polymer_descriptor_scaler_1.pkl  # GNN polymer descriptor scaler (iteration 1; only with polymer_descriptors)
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

### GNN learning curves

`plots/{model}_{iteration}_learning_curve.png` shows, for one GNN and one split, the
training, validation and test loss after every epoch.

- **Model selection uses the validation loss only.** Each GNN is trained for a fixed
  number of epochs (`training.epochs`; there is no early stopping). After every epoch
  the validation loss is computed, and at the end the weights from the epoch with the
  **lowest validation loss** are restored. The default `reduce_lr_on_plateau` scheduler
  also monitors the validation loss.
- **The test curve is for monitoring only.** The test loss is computed every epoch so
  the curves can be inspected, but it is never used for training, for choosing the
  learning rate or for choosing the final weights.
- **What the loss values are.** The curves show the training loss
  (`gnn_training.optimisation.regression_loss`, RMSE by default; cross-entropy for
  classification). With `target_transform` enabled they are in the *scaled* target
  units. The training curve is the mean of the per-batch losses during the epoch,
  computed with dropout active. The validation and test losses are computed **over the
  whole set** (all predictions first, then the loss once), so with the default RMSE
  loss they are the RMSE of the set and the best epoch is the one with the lowest
  validation RMSE; they do not depend on the batch size. With class weights
  (`AsymmetricLossStrength`) the validation curve is the weighted cross-entropy over
  the set. (Earlier versions averaged the loss of single samples, so with RMSE the
  validation/test curves — and the best-epoch choice — were in fact the MAE.)

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
