# Configuration Reference

The YAML config controls every pipeline stage. A complete template lives in
[`configs/experiment.yaml`](../configs/experiment.yaml).

- [`experiment`](#experiment)
- [`data`](#data)
- [`representations`](#representations)
- [`splitting`](#splitting)
- [`gnn_training`](#gnn_training)
- [Automatic HPO configuration](#automatic-hpo-configuration)
- [`tml_models`](#tml_models)
- [`target_transform`](#target_transform)
- [`explainability` (GNN)](#explainability-gnn)
- [`tml_explainability` (TML SHAP)](#tml_explainability-tml-shap)
- [`prediction`](#prediction)

See also: [Descriptors](descriptors.md) for `representations.molecular_descriptors`
and PolyMetriX, and [Explainability](explainability.md) for the meaning of the
explainability options.

---

## `experiment`

```yaml
experiment:
  name: "my_experiment"
  output_dir: "results/my_experiment"
  random_seed: 42
```

## `data`

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

## `representations`

```yaml
representations:
  weights_col:                     # null to treat all monomers as equally weighted
    monomer1_smiles: "weight_fraction_1"
    monomer2_smiles: "weight_fraction_2"
  molecular_descriptors:
    RDKit: []                      # Include RDKit descriptors
    Morgan: []                     # Morgan fingerprints
    PolyBERT: []                   # PolyBERT fingerprints (requires psmiles)
    PolyMetriX:                    # PolyMetriX polymer-aware descriptors (requires polymetrix)
      # Each of side_chain / backbone / polymer accepts either a list of
      # descriptor names OR the sentinel "all" to request every available
      # descriptor for that part. Any of the three keys may be omitted entirely
      # (the config requires only that *one* of them is provided).
      side_chain: "all"            # All chemical features + sidechain-topological features
      backbone:  ["num_rings", "num_atoms", "molecular_weight"]
      # polymer key omitted entirely — no full-repeat-unit descriptors computed
      agg: [sum, mean]             # Aggregation methods for side-chain features
  smiles_merge_approach: "weighted_average"   # weighted_average | concatenate | no_merging
  polymer_descriptors:             # Optional: column names from your CSV to use as given features
    - "molecular_weight"
    - "degree_of_polymerisation"
```

The full descriptor catalogue (RDKit, Morgan, PolyBERT, PolyMetriX), the polymer
descriptor fusion mechanism, and the PolyMetriX modes are documented in
[Descriptors](descriptors.md).

> **PSMILES note:** RDKit descriptors are computed on a capped molecule — the
> polymer attachment points (`*`, atomic number 0) are replaced with hydrogens and
> folded into the neighbouring atoms' implicit-H counts before descriptors run, so
> mass- and charge-based descriptors (e.g. Gasteiger partial charges) are no longer
> corrupted by the massless dummy atoms.

## `splitting`

```yaml
splitting:
  split_type: "TrainValTest"       # TrainValTest | LeaveOneOut
  split_method: "Random"           # Random | Scaffold
  n_bootstrap_iterations: 3
  val_ratio: 0.15
  test_ratio: 0.15
  train_set_balance: null          # Optional: balance training set (0.0–1.0)
```

## `gnn_training`

Each architecture block lists its hyperparameters. Leave the block empty (`{}`) to
trigger automatic HPO via Ray Tune.

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

  # HPO split strategy (optional — all fields below show their defaults)
  hpo_split_strategy: "cross_validation"  # cross_validation | holdout | repeated_holdout
  hpo_n_folds: 5                          # folds used by cross_validation
  hpo_val_fraction: 0.2                   # val fraction used by holdout / repeated_holdout
  hpo_n_repeats: 3                        # number of random splits for repeated_holdout

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
| `apply_weighting_to_graph` | `"PerMonomerPooling"` | One of: `PerMonomerPooling` (pools each monomer separately then sums `Σ wᵢ·pool(monomerᵢ)` — atom-count-bias-free), `BeforePooling` (wD-MPNN-style: weights node features then pools with weighted-mean normalisation `Σ wx / Σ w`), or `BeforeMPP` (multiplies node features by their monomer weight *before* message passing, so the convs see weighted inputs) |
| `AsymmetricLossStrength` | `null` | Classification only. When set to a float `s ∈ [0, 1]`, class loss weights are `(1 - s)·freq_weights + s·inverse_freq_weights` — `s = 0` upweights majority classes (no correction), `s = 1` is full inverse-frequency correction (rare classes get high weight). `null` disables class weighting entirely. Automatically explored by HPO over `[null, 0.25, 0.5, 0.75, 1.0]` for classification tasks. Ignored for regression. |

## Automatic HPO configuration

HPO is triggered automatically for any architecture whose parameter block is left
empty (`{}`). Ray Tune samples 150 random configurations from the search grid and
evaluates them using one of three **split strategies** that control how the
train+val data is partitioned inside each trial.

### Split strategies

| Strategy | `hpo_split_strategy` value | Speed | Reliability | ASHA pruning |
|---|---|---|---|---|
| **K-fold cross-validation** | `cross_validation` | Slowest (K × epochs per trial) | Highest — mean across all folds | ✗ (single report at end) |
| **Holdout** | `holdout` | Fastest (1 split, reports per epoch) | Lower — single random split | ✓ |
| **Repeated holdout** | `repeated_holdout` | Intermediate (N splits, reports per epoch) | Good — average across N repeats | ✓ |

- **`cross_validation`** (default) — the dataset is split into `hpo_n_folds` folds.
  Each trial trains one model per fold for the full number of epochs and reports a
  single aggregated val loss at the end. This is the most statistically reliable
  option and the right choice for small datasets (< ~500 samples) where a single
  random split would have high variance.

- **`holdout`** — a single stratified (classification) or random (regression)
  train/val split is created using `hpo_val_fraction`. Each trial reports val loss
  after every epoch, so Ray Tune's ASHA scheduler can prune underperforming trials
  early. This is 5× faster than 5-fold CV and is recommended for large datasets
  where a single well-sized validation set is sufficient.

- **`repeated_holdout`** — `hpo_n_repeats` independent random splits are created.
  Trials train one model per split in epoch lockstep and report the mean val loss
  across all repeats after each epoch, enabling ASHA pruning. A good balance between
  speed and reliability: 3 repeats are ~3× cheaper than 5-fold CV while averaging out
  the noise of a single holdout.

### HPO parameters

| Parameter | Default | Applies to | Description |
|---|---|---|---|
| `hpo_split_strategy` | `cross_validation` | all | Split strategy for HPO trials |
| `hpo_n_folds` | `5` | `cross_validation` | Number of CV folds |
| `hpo_val_fraction` | `0.2` | `holdout`, `repeated_holdout` | Fraction of data held out for validation |
| `hpo_n_repeats` | `3` | `repeated_holdout` | Number of independent random splits |

Setting a parameter that has no effect for the chosen strategy (e.g.
`hpo_val_fraction` under `cross_validation`) emits a warning at config-load time.

### Example configurations

**Fast HPO for a large dataset (recommended starting point for > ~1 000 samples):**
```yaml
gnn_training:
  gnn_convolutional_layers:
    GCN: {}
  hpo_split_strategy: "holdout"
  hpo_val_fraction: 0.15
```

**Balanced speed/reliability with repeated holdout (300–1 000 samples):**
```yaml
gnn_training:
  gnn_convolutional_layers:
    GCN: {}
  hpo_split_strategy: "repeated_holdout"
  hpo_val_fraction: 0.2
  hpo_n_repeats: 3
```

**Thorough cross-validation for small datasets (< ~300 samples, default):**
```yaml
gnn_training:
  gnn_convolutional_layers:
    GCN: {}
  hpo_split_strategy: "cross_validation"
  hpo_n_folds: 5
```

> **Note:** HPO results are cached to `{output_dir}/gnn_hyp_opt/iteration_{n}/{arch}/{arch}.csv`.
> If this file already exists when the pipeline is re-run, the cached best
> configuration is reloaded without repeating the search — delete the file to force a
> fresh run.

## `tml_models`

```yaml
tml_models:
  train_tml: false
  selected_models:
    - RandomForest
    - XGBoost

feature_preprocessing:
  scaler: "StandardScaler"         # StandardScaler | MinMaxScaler | RobustScaler | NoTransformation
```

**Available models:** `RandomForest`, `XGBoost`, `SupportVectorMachine`,
`LogisticRegression`, `LinearRegression`

> **Robust feature preprocessing:** when the feature transformer is fit, any
> descriptor column containing `NaN` or `±inf` in the training data is dropped (with a
> logged warning naming the columns), so a single undefined descriptor cannot abort
> training. At transform time, values that are `NaN`/`±inf` only on new data are
> imputed with the per-column training mean, so prediction on unseen molecules can
> still proceed.

## `target_transform`

Applies scaling to the **regression target variable**. The scaler is fit on the
training set only; validation and test predictions are inverse-transformed before
metrics and plots so all results remain in the original target units. This section is
ignored for classification problems. See
[Target-variable scaling](running-and-outputs.md#target-variable-scaling) for the full
behaviour and saved-file layout.

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

> **Note:** Using `log10` or `log1p` with non-positive target values will raise a
> `ValueError` at fit time with a descriptive message.

## `explainability` (GNN)

```yaml
explainability:
  enabled: false
  algorithm: "chemistry_masking"

  # Which GNN architectures to explain (must match gnn_training keys).
  models: "all"             # or: [GCN, GAT]

  # Which bootstrap iterations to explain (1-based, matching model filenames).
  bootstraps: "all"         # or: [1, 2]

  fragmentation: "brics"    # brics | murcko_scaffold
  explain_set: "test"       # train | validation | test | all  — controls the global distribution plot
  normalisation: "per_model" # local | global | per_model | no_normalisation
  target_class: null        # null for regression; integer for classification
  plot_type: "ridge"        # ridge | bar | strip
  top_n: 10                 # top-N and bottom-N fragments shown; null = all

  # Molecule IDs to generate per-molecule heatmaps and attribution CSVs for.
  # When null (default), the local explanation step is skipped entirely.
  # explain_set still controls which molecules go into the distribution plot.
  local_explain_mol_ids: null
  # local_explain_mol_ids:
  #   - "poly_0001"
  #   - "poly_0042"
```

The attribution method, fragmentation strategies, and normalisation semantics are
described in [Explainability](explainability.md).

## `tml_explainability` (TML SHAP)

```yaml
tml_explainability:
  enabled: false

  # Which TML model types to explain (must match tml_models.selected_models).
  models: "all"             # or: [RandomForest, XGBoost]

  # Which descriptor representations to explain (must match representations.molecular_descriptors keys).
  representations: "all"    # or: [Morgan, RDKit]

  # Which bootstrap iterations to explain (1-based, matching model filenames).
  bootstraps: "all"         # or: [1, 2]

  explain_set: "test"       # train | validation | test | all  — controls the global summary plot
  normalisation: "per_model" # local | global | per_model | no_normalisation
  target_class: null        # null for regression; integer class index for classification
  plot_type: "beeswarm"     # beeswarm | bar | violin  (native shap.summary_plot styles)
  top_n: 10                 # Features shown; null = all

  # Sample IDs to generate per-instance SHAP plots for.
  # When null (default), the local explanation step is skipped entirely.
  local_explain_sample_ids: null
  # local_explain_sample_ids:
  #   - "poly_0001"
  #   - "poly_0042"

  local_plot_type: "waterfall"  # waterfall | force | bar  (native shap plots)
```

The global TML attribution view is rendered with the native `shap` package
(`shap.summary_plot` — beeswarm / bar / violin), and per-instance plots use native
`shap.plots.waterfall` / `force` / `bar`. Explainer selection, caching, and the
GUI-only options (averaged vs per-model display, top-N features, custom colours) are
covered in [Explainability](explainability.md).

## `prediction`

```yaml
prediction:
  enabled: true
  data_path: "data/new_polymers.csv"
```

See [Predicting on external data](running-and-outputs.md#predicting-on-external-data).
