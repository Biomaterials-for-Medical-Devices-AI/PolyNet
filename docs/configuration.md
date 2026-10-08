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
  string_representation: "smiles"  # smiles | psmiles
  canonicalise_smiles: true        # canonicalise structures before featurisation (default true)
```

### Built-in benchmark dataset (`benchmark_dataset`)

Instead of `data_path`, the CLI can load a built-in benchmark dataset — the same ones the
GUI offers under *Load benchmarking dataset*. Give exactly one of `data_path` and
`benchmark_dataset`. The template `configs/experiment.yaml` uses it by default:

```yaml
data:
  data_name: "curated_tg.csv"
  benchmark_dataset: "curated_tg"  # instead of data_path
  smiles_cols: ["PSMILES"]
  target_variable_col: "Tg(K)"
  id_col: "ID"
  problem_type: "regression"
  string_representation: "psmiles"
  num_classes: 1
```

| `benchmark_dataset` | Content |
|---|---|
| `curated_tg` | 7,367 homopolymers (`PSMILES`) with experimental glass transition temperatures `Tg(K)` from the curated polymetrix dataset (Zenodo record 15210035); IDs in `ID`. |
| `fluorine_nmr_snr` | 418 fluorinated copolymers with their ¹⁹F NMR signal-to-noise ratio `SNR`. Every polymer combines the same six acrylate monomers (`smiles_1` … `smiles_6`, PSMILES) in different molar ratios (`ratio_1` … `ratio_6`, each row sums to 1); IDs in `ID`. `MolWt` and `Dispersity` are measured for 159 polymers only (empty otherwise). From Reis et al., *J. Am. Chem. Soc.* 2021, 143, 17677–17689, as used by Tao et al., *STAR Protocols* 2022, 3. |

`curated_tg` is downloaded on first use (internet access needed once) and cached by
polymetrix; `fluorine_nmr_snr` ships with PolyNet. Both are validated like a CSV file
(columns, structures, unique IDs, target), and a copy is saved as `data_name` in the
experiment's output directory. `curated_tg` holds homopolymers: use
`smiles_merge_approach: "no_merging"` and `weights_col: null`. For `fluorine_nmr_snr`,
weight each monomer by its ratio:

```yaml
data:
  data_name: "fluorine_nmr_snr.csv"
  benchmark_dataset: "fluorine_nmr_snr"
  smiles_cols: ["smiles_1", "smiles_2", "smiles_3", "smiles_4", "smiles_5", "smiles_6"]
  target_variable_col: "SNR"
  id_col: "ID"
  problem_type: "regression"
  string_representation: "psmiles"
  num_classes: 1
representations:
  smiles_merge_approach: "weighted_average"
  weights_col: {smiles_1: ratio_1, smiles_2: ratio_2, smiles_3: ratio_3,
                smiles_4: ratio_4, smiles_5: ratio_5, smiles_6: ratio_6}
```

`MolWt` and `Dispersity` are mostly missing, so they cannot be used as
`polymer_descriptors` on the full dataset.

### Structure validation and canonicalisation

The GUI, the CLI and external prediction (`predict_external`) prepare the structure
columns with the same function (`polynet.data.structures.prepare_structures`):

1. **Detect** the representation (PSMILES if every value has at least two `*`
   attachment points, otherwise SMILES). If it differs from `string_representation`, a
   warning is logged and the configured value is used.
2. **Validate** every structure with RDKit. Invalid structures stop the run with an
   error listing examples per column, e.g.
   `column 'monomer1_smiles': 2 invalid, e.g. C1CC, xyz`.
3. **Canonicalise** the structures when `canonicalise_smiles` is `true` (RDKit for
   SMILES, `psmiles` for PSMILES), so the same molecule is always written the same way.
   New data passed to `predict_external` is canonicalised with the training settings.

Missing structures (empty cells) are treated as invalid everywhere: the run stops and
the error shows them as `<missing>`. For a homopolymer in a multi-monomer dataset, repeat
its SMILES in the other structure column and give it a weight of 0.

> Before this was shared, the CLI accepted `canonicalise_smiles` but did not apply it.
> Canonicalisation does not change the molecules or the descriptors, but it can reorder
> the atoms of non-canonical inputs, which changes the random dropout masks during GNN
> training. GNN results on such datasets can therefore differ slightly from earlier CLI
> runs (as they would with a different seed); set `canonicalise_smiles: false` to
> reproduce them exactly.

### Sample IDs (`id_col`)

IDs must be unique. The CLI stops with an error listing example duplicates. The GUI
shows a warning and numbers the samples by row order instead (the ID column is kept as
a regular column). Without `id_col`, samples are numbered by row order.

## `representations`

```yaml
representations:
  weights_col:                     # null to treat all monomers as equally weighted
    monomer1_smiles: "weight_fraction_1"
    monomer2_smiles: "weight_fraction_2"
  molecular_descriptors:
    RDKit: []                      # Include RDKit descriptors
    Morgan: []                     # Morgan count fingerprints, defaults (2048 bins, radius 3)
    # Morgan: {fp_size: 1024, radius: 2}   # …or with custom settings (radius 2 ≈ ECFP4)
    # RDKitFP: {fp_size: 2048}             # RDKit count fingerprints (fp_size only)
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
  split_type: "train_val_test"     # the only implemented split type
  split_method: "random"           # random | stratified (by target; recommended for classification)
  n_bootstrap_iterations: 3        # number of repeated random splits
  val_ratio: 0.15                  # fraction of the full dataset (70/15/15 here)
  test_ratio: 0.15                 # fraction of the full dataset
  train_set_balance: null          # Optional: balance training set (0.0–1.0)
  sampler: "random"                # random | kennard_stone | spxy | kmeans | optisim | target_property
  sampling_fingerprint:            # fingerprint samplers only; used for splitting only
    fingerprint: "morgan"          # morgan | rdkitfp | polybert
    fp_size: 2048                  # morgan / rdkitfp only
    radius: 3                      # morgan only
```

**Terminology.** Throughout PolyNet (configs, GUI, outputs such as the
`bootstrap_iteration` column), the repeated train/validation/test splits are called
*bootstraps* (`n_bootstrap_iterations`, `bootstraps` in the explainability settings). They
are repeated random (or sampler-based) splits drawn without replacement, not bootstrap
resamples drawn with replacement.

**Split type.** Only `train_val_test` is implemented: for each of the
`n_bootstrap_iterations` repeated random splits (seed `random_seed + i`), the full
dataset is split into training, validation and test sets in one step with
[astartes](https://github.com/JacksonBurns/astartes) (`train_val_test_split`,
`sampler="random"`).

**Split ratios.** `test_ratio` and `val_ratio` are both fractions of the **full
dataset**, and the training set gets the rest: `test_ratio: 0.1` and `val_ratio: 0.1`
give an 80/10/10 split. Their sum must be below 1. Following astartes, the training and
validation sizes are rounded **down** to whole samples and the test set takes the
remaining samples (e.g. 37 samples at 70/15/15 → 25/5/7).

**Split method.** `random` splits the whole dataset with the chosen sampler. `stratified`
splits each class separately with the same sampler and merges the results, so the
training, validation and test sets keep the class proportions (up to rounding within
each class). Every class needs enough samples to appear in all three sets; otherwise the
split stops with an error naming the class.

**Sampler (`sampler`).** The sets are drawn with an [astartes](https://github.com/JacksonBurns/astartes)
sampler, always with astartes' default hyperparameters:

| `sampler` | Uses | Behaviour |
|---|---|---|
| `random` (default) | — | Random split. |
| `kennard_stone` | sampling fingerprint | Training set chosen to span the fingerprint space; validation and test are the most "interpolated" polymers. Deterministic. |
| `spxy` | sampling fingerprint + target | Kennard–Stone on joint fingerprint and target distances. Deterministic. |
| `kmeans` | sampling fingerprint | k-means clusters (n/10 + 1); each cluster stays in one set (extrapolation). |
| `optisim` | sampling fingerprint | OptiSim diverse clusters; each cluster stays in one set (extrapolation). |
| `target_property` | target | Polymers sorted by target value; the extremes go to the test set. Deterministic. |

- **Sampling fingerprint (`sampling_fingerprint`).** Fingerprint samplers place each polymer
  by the ratio-weighted average of its monomers' fingerprints — the same computation as the
  fingerprint representations, but with these settings:
  - `morgan` / `rdkitfp`: count fingerprints with `fp_size` (and, for Morgan, `radius`);
  - `polybert`: the 600-dimensional polyBERT embedding (`xushijie/polyBERT`, downloaded on
    first use; no settings). polyBERT expects PSMILES; plain SMILES that cannot be
    canonicalised as PSMILES are embedded as given, with a warning.

  It is used **only for splitting** and never changes the representations. Without
  `sampling_fingerprint`, Morgan with 2048 bins and radius 3 is used; with a sampler that
  needs no fingerprint, it is ignored with a warning.
- **Deterministic samplers** (`kennard_stone`, `spxy`, `target_property`) ignore the seed, so
  all `n_bootstrap_iterations` repetitions use the same split. This is allowed, with a warning:
  the models still differ between repetitions because each is trained with its own seed,
  but models without randomness (e.g. linear regression) will be identical.
- **Cluster samplers** (`kmeans`, `optisim`) keep each cluster in one set, so validation and
  test can be smaller or larger than requested (astartes' message is logged); they can fail
  on small datasets or small classes when a cluster is larger than the validation set.
- `spxy` and `target_property` cannot be combined with `split_method: stratified` (the target
  is constant within a class). `target_property` is not meaningful for classification (it
  would put a single class in the test set). The GUI only offers the samplers that are valid
  for the problem type and split method, and reports a sampler that cannot split the data
  (e.g. clusters larger than the validation set) as a message instead of an error.
- `dbscan` and `sphere_exclusion` are not offered: their default distance thresholds are far
  below the typical distance between polymer count fingerprints, so they cannot fill the
  sets.
- **Provenance.** The sampler and sampling fingerprint are saved in `split_options.json`, and
  `split_indices.json` has a `sampling` record (sampler, astartes version, fingerprint
  settings, ratio columns).

> **Changed behaviour.** Earlier versions drew the splits with scikit-learn, held out
> the test set first and applied `val_ratio` to the remaining data. Experiments run with
> those versions are not reproduced exactly.
The other values of the `SplitType` enum (`train_test`, `cross_validation`,
`nested_cross_validation`, `leave_one_out`) are reserved for future work: a config that
uses them is rejected at load time with an error explaining that only `train_val_test`
is available. The GUI offers only `train_val_test`.

**Class balancing (`train_set_balance`, binary classification).** For each split:

1. the full dataset is split into training, validation and test sets as above;
2. the training set and the validation set are **each** balanced by randomly
   undersampling their majority class until the minority class makes up
   `train_set_balance` of the set (e.g. `0.5` → 50/50).

Training and validation sets are therefore **both balanced**, and only the **test set
keeps the original class distribution**. This follows the protocol of
*ACS Appl. Mater. Interfaces 2023, 15 (11), 14155–14163*. Undersampling uses the split's
seed, so splits are reproducible. Undersampling removes samples, so the training and
validation sets are smaller than their ratios of the full dataset would suggest. Use
`split_method: stratified` with balancing: both sets then contain the same share of
the minority class, so the training:validation proportion is kept after balancing (with
`random`, the number of minority samples landing in validation varies between splits).

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
      pooling: "global_mean_pool"

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
  hpo_num_samples: 50                     # configurations sampled per HPO run
  hpo_search_grid: {}                     # optional custom candidates, see below

  # Optimiser, scheduler and loss (optional — all fields below show their defaults)
  optimisation:
    optimizer: "adam"                     # adam | sgd | rmsprop | adadelta | adagrad
    scheduler: "reduce_lr_on_plateau"     # reduce_lr_on_plateau | step_lr | multi_step_lr | exponential_lr
    scheduler_factor: 0.9                 # learning-rate decay factor (gamma), all schedulers
    scheduler_patience: 15                # reduce_lr_on_plateau only
    scheduler_min_lr: 1.0e-8              # reduce_lr_on_plateau only
    scheduler_step_size: 10               # step_lr only
    scheduler_milestones: [30, 60, 90]    # multi_step_lr only
    regression_loss: "rmse"               # rmse | mse | mae (regression only)

training:
  epochs: 250
```

### Optimiser, scheduler and loss (`optimisation`)

The `optimisation` block controls how every GNN is trained — the final models **and**
every HPO trial use the same settings. All fields are optional; the defaults reproduce
PolyNet's standard settings (Adam, ReduceLROnPlateau with factor 0.9 / patience 15 /
min_lr 1e-8, RMSE loss). The learning rate itself comes from each architecture block
(`LearningRate`) or from HPO.

| Field | Default | Used by | Description |
|---|---|---|---|
| `optimizer` | `adam` | all | Gradient-descent optimiser |
| `scheduler` | `reduce_lr_on_plateau` | all | `reduce_lr_on_plateau` lowers the learning rate when the validation loss stops improving; `step_lr`, `multi_step_lr` and `exponential_lr` decay it on a fixed epoch schedule |
| `scheduler_factor` | `0.9` | all schedulers | Multiplicative decay (`gamma`), in (0, 1) |
| `scheduler_patience` | `15` | `reduce_lr_on_plateau` | Epochs without validation improvement before decaying |
| `scheduler_min_lr` | `1e-8` | `reduce_lr_on_plateau` | Lower bound on the learning rate |
| `scheduler_step_size` | `10` | `step_lr` | Decay every N epochs |
| `scheduler_milestones` | `[30, 60, 90]` | `multi_step_lr` | Epochs at which to decay (strictly increasing) |
| `regression_loss` | `rmse` | regression | `rmse` (root mean squared error; per batch during training, over the whole set for validation/test), `mse` or `mae` (mean absolute error, less sensitive to outliers). Classification always uses cross-entropy (with optional `AsymmetricLossStrength` class weights). |

Setting a scheduler parameter that the chosen scheduler does not use emits a warning at
config-load time. In the GUI these options are under **Advanced training options** in
the GNN section of the Train Models page.

**Available architectures:** `GCN`, `GAT`, `CGGNN`, `MPNN`, `GraphSAGE`, `TransformerConvGNN`

Every key is checked when the config is loaded: a misspelt `gnn_training` key, an
unknown parameter in an architecture block (including one that belongs to another
architecture, e.g. `improved` outside `GCN`) or a key other than `epochs` in `training`
stops the run with an error listing the allowed names. The number of epochs is set in
`training.epochs` only (in the GUI: *Number of training epochs* in the GNN section of the
Train Models page); HPO trials train for the same number of epochs.

**Architecture-specific parameters:**

| Architecture | Parameter | Description |
|---|---|---|
| `GCN` | `improved: bool` | Improved normalisation from Kipf & Welling |
| `GAT` | `num_heads: int` | Number of multi-head attention heads |
| `TransformerConvGNN` | `num_heads: int` | Number of transformer attention heads |
| `GraphSAGE` | `bias: bool` | Whether to include bias terms |

**Shared optional parameters:**

| Parameter | Default | Description |
|---|---|---|
| `apply_weighting_to_graph` | `"before_pooling"` | One of: `per_monomer_pooling` (pools each monomer separately then sums `Σ wᵢ·pool(monomerᵢ)` — atom-count-bias-free), `before_pooling` (wD-MPNN-style: weights node features then pools with weighted-mean normalisation `Σ wx / Σ w`), `before_mpp` (multiplies node features by their monomer weight *before* message passing, so the convs see weighted inputs), or `no_weighting` |
| `AsymmetricLossStrength` | `null` | Classification only. When set to a float `s ∈ [0, 1]`, class loss weights are `(1 - s)·freq_weights + s·inverse_freq_weights` — `s = 0` upweights majority classes (no correction), `s = 1` is full inverse-frequency correction (rare classes get high weight). `null` disables class weighting entirely. Ignored for regression. With automatic HPO (classification) it is tuned over `[null, 0.25, 0.5, 0.75, 1.0]`; set `AsymmetricLossStrength: [s]` in `hpo_search_grid` to fix it (or to search other values) — see [Class weights in HPO](#class-weights-in-hpo). |

## Automatic HPO configuration

HPO is triggered automatically for any architecture whose parameter block is left
empty (`{}`). Ray Tune samples `hpo_num_samples` (default 50) random configurations
from the search grid and evaluates them using one of three **split strategies** that
control how the train+val data is partitioned inside each trial. The search grid is the
default grid of `polynet/config/search_grid.py`, optionally customised with
`hpo_search_grid` — see [Custom search grids](#custom-search-grids-hpo_search_grid).

Every HPO trial trains for `training.epochs` epochs — the same number as the final
models — so the selected hyperparameters are tuned for the training length actually
used. With `holdout` / `repeated_holdout`, ASHA stops a trial at
`training.epochs` at the latest and lets every trial run at least one fifth of them
(50 of the default 250) before pruning it.

### Split strategies

| Strategy | `hpo_split_strategy` value | Speed | Reliability | ASHA pruning |
|---|---|---|---|---|
| **K-fold cross-validation** | `cross_validation` | Slowest (K × epochs per trial) | Highest — mean across all folds | ✗ (single report at end) |
| **Holdout** | `holdout` | Fastest (1 split, reports per epoch) | Lower — single random split | ✓ |
| **Repeated holdout** | `repeated_holdout` | Intermediate (N splits, reports per epoch) | Good — average across N repeats | ✓ |

- **`cross_validation`** (default) — the dataset is split into `hpo_n_folds` folds.
  Each trial trains one model per fold for `training.epochs` epochs and reports a
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
| `hpo_n_folds` | `5` | `cross_validation` | Number of CV folds `k` (shuffled; stratified for classification). See [Choosing the number of folds](#choosing-the-number-of-folds-hpo_n_folds). |
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

> **Note:** HPO results are cached to
> `{output_dir}/gnn_hyp_opt/iteration_{n}/{arch}_{hash}/{arch}.csv`, next to a
> `search_space.json` that records the searched grid and settings. The `{hash}` is
> computed from everything that defines the search — the grid actually searched (the
> default candidates, with any parameter set in `hpo_search_grid` replacing its defaults,
> plus the seed), `hpo_num_samples`, the HPO split settings, the `optimisation` settings and the
> polymer-descriptor scaler. Re-running the same search reloads the cached best
> configuration; changing any of these starts a new search in a new directory, so
> stale results are never reused. Delete the directory to force a fresh run. (Caches
> written before this scheme are not reused.)

### Custom search grids (`hpo_search_grid`)

Both `gnn_training` and `tml_models` accept `hpo_search_grid` to replace the default
candidates of individual parameters, and `hpo_num_samples` to set how many
configurations are sampled:

```yaml
gnn_training:
  gnn_convolutional_layers:
    GCN: {}                              # empty block → HPO
    GAT: {}
  hpo_num_samples: 30                    # default 50
  hpo_search_grid:
    shared:                              # applies to every architecture
      embedding_dim: [64, 128]
      LearningRate: [0.001, 0.01]
    GAT:                                 # architecture-specific (wins over shared)
      num_heads: [2, 4, 8]

tml_models:
  selected_models:
    random_forest: {}                    # empty block → HPO
  hpo_num_samples: 20                    # default 50 (RandomizedSearchCV n_iter)
  hpo_search_grid:
    random_forest:
      n_estimators: [200, 500, 1000]
      max_depth: [null, 10, 20]
```

- **Merge.** A parameter you list *replaces* the default candidates for that parameter;
  every parameter you do not list keeps its default candidates. For GNNs the order is
  default ← `shared` ← architecture entry. The random seed (`seed`, `random_state`,
  and `probability` for classification SVMs) is always set by PolyNet.
- **Validation at config load.** GNN keys must be selected architectures (or `shared`)
  and their parameters must be in that architecture's default grid; TML keys must be
  selected models and their parameters must be constructor parameters of the model's
  estimator. Every value must be a non-empty list. Unknown names, unknown parameters,
  reserved parameters and empty lists are errors listing the allowed values. A grid for
  a model that has explicit hyperparameters (so HPO does not run for it) triggers a
  warning.
- **Class weights in HPO.** <a id="class-weights-in-hpo"></a> For classification,
  GNN HPO also tunes `AsymmetricLossStrength` (class weights in the cross-entropy
  loss) over `[null, 0.25, 0.5, 0.75, 1.0]`, and the final models use the chosen
  value. Each trial trains with its own class weights (computed from the training part
  of each HPO split), but every trial is **scored with the unweighted** cross-entropy
  on its validation data, so trials with different weights are compared on the same
  scale. Validation losses of HPO trials are computed over the whole validation set, so
  they do not depend on the trial's batch size (with per-batch RMSE, smaller batches
  used to give lower values and were favoured). To fix the strength while HPO tunes everything else, give a single value,
  e.g. `shared: {AsymmetricLossStrength: [0.5]}` (or `[null]` for no weighting).
  Candidates must be `null` or between 0 and 1. For regression the setting is ignored
  with a warning.
- **`hyperparameter_optimisation`.** HPO runs for every architecture / model whose block is
  empty (`{}`); this flag does not switch it on or off. If you leave it out, it is filled in
  from the blocks (true when any block is empty) and saved that way with the experiment. If
  you set it and it contradicts the blocks — `true` with no empty block (no HPO will run),
  or `false` with an empty block (HPO will run anyway) — a warning says what will happen.
- **In the GUI.** Ticking *Perform hyperparameter tuning* (GNN or TML section of the Train
  Models page) shows the number of configurations to sample (default 50) and, for each
  grid, the default candidates of every tunable parameter: deselect candidates or type
  extra numeric values. Only the parameters you change are saved as `hpo_search_grid`, so
  untouched grids give exactly the default search.
- **Sample counts.** `hpo_num_samples` (default 50 for both GNN and TML) must be ≥ 1. For TML it is capped, with a
  warning, at the number of distinct grid combinations (sampling more would only
  repeat configurations).
- **Provenance.** The settings, as written by the user, are saved in `config_used.yaml`
  and in `train_gnn_options.json` / `train_tml_options.json`. Before training starts,
  `hpo_search_spaces.json` records the grid **actually searched** for every
  architecture / model that runs HPO — the default candidates, with any parameter set in
  `hpo_search_grid` replacing its defaults (the per-split seed is left out) — with the
  sample counts (TML `n_iter` after capping) and HPO split settings. The GNN grid
  actually searched is also written
  to `gnn_hyp_opt/iteration_{n}/{arch}_{hash}/search_space.json`; for TML, the grid
  actually searched, the number of samples used, the folds and the best parameters of each tuned
  model are written to `tml_hyp_opt/{model}-{representation}_{iteration}.json`.

## `tml_models`

```yaml
tml_models:
  train_tml: false
  selected_models:
    - RandomForest
    - XGBoost
  hpo_n_folds: 5            # CV folds for automatic HPO (optional, default 5, min 2)
  include_validation_in_training: true   # false: train on the training samples only (like GNNs)
```

**Training data (`include_validation_in_training`).** TML models have no epochs to
select, so by default (`true`) they are trained on the **training + validation** samples
of each split, and their feature transformer, target scaler and hyperparameter search are
fitted on those samples too; their predictions and metrics then report the validation
samples as part of the training set. With `false`, everything is fitted on the
**training samples only** — the same data the GNNs train on — and the validation samples
are predicted and scored as a separate `Validation` set. Use `false` for a like-for-like
comparison between TML models and GNNs. In the GUI this is the *Include the validation
set in TML training* switch.

**Available models:** `RandomForest`, `XGBoost`, `SupportVectorMachine`,
`LogisticRegression`, `LinearRegression`

**Automatic HPO:** leave a model's block empty (`{}`) to tune it automatically.
`RandomizedSearchCV` samples `hpo_num_samples` (default 50) configurations from the model's search grid
(`polynet/config/search_grid.py`) and scores them by `hpo_n_folds`-fold
cross-validation on the samples the model is trained on (training + validation by default,
training only with `include_validation_in_training: false`). Folds are always
shuffled (stratified for classification) with the split's seed, so a dataset sorted by
target cannot produce biased folds.

| Parameter | Default | Description |
|---|---|---|
| `hpo_n_folds` | `5` | Number of CV folds `k`. See [Choosing the number of folds](#choosing-the-number-of-folds-hpo_n_folds). |
| `hpo_num_samples` | `50` | Configurations sampled per search (`n_iter`), capped at the number of grid combinations. |
| `hpo_search_grid` | `{}` | Custom candidates per model. See [Custom search grids](#custom-search-grids-hpo_search_grid). |

In the GUI, the fold count appears under *Perform hyperparameter tuning* on the Train
Models page.

### Choosing the number of folds (`hpo_n_folds`)

`gnn_training.hpo_n_folds` and `tml_models.hpo_n_folds` are the same setting, shared by
both pipelines: each scores hyperparameter configurations by shuffled `k`-fold
cross-validation (stratified for classification) on the **training + validation**
samples of every split, using the same fold splitter. `k` is checked in two places:

1. **At config load** — `k ≥ 2`.
2. **After the data is split, before any model is trained** — for every split, `k` may
   not exceed the number of training + validation samples, and for classification it
   may not exceed the size of the smallest class (stratified folds need every class in
   every fold).

The second check only runs for a pipeline that will actually cross-validate: TML when a
model block is empty (`{}`), GNN when an architecture block is empty and
`hpo_split_strategy` is `cross_validation`. An invalid `k` stops the run (CLI) or shows
an error (GUI) naming the setting, the split and the allowed range, e.g.:

```
gnn_training.hpo_n_folds=40 is invalid for split 1: 40 folds exceed the size of the
smallest class (class 0 has 29 samples). ... choose between 2 and 29 folds.
```

TML models require a [`feature_preprocessing`](#feature_preprocessing) section.

## `feature_preprocessing`

A **pipeline-wide** option: the scaler applies to every tabular feature the pipeline
uses, regardless of which model families are trained.

```yaml
feature_preprocessing:
  scaler: "standard_scaler"   # no_transformation | standard_scaler | min_max_scaler | robust_scaler
                              # | power_transformer | quantile_transformer | normalizer
  selectors:                  # TML models only
    variance:
      threshold: 0.05
    correlation:
      threshold: 0.95
```

| Field | Applies to | Description |
|---|---|---|
| `scaler` | TML descriptors **and** GNN polymer descriptors | Fitted on the training set of each split, then applied to validation, test and external data |
| `selectors` | TML descriptors only | Variance / correlation feature selection, applied after scaling |

The section is only meaningful when there are tabular features to scale: when TML
models are trained, or when GNNs are trained with
`representations.polymer_descriptors`. Otherwise it has no effect and a warning is
logged; `selectors` given for a GNN-only run are ignored with a warning. In the GUI,
the *Feature Preprocessing* section on the Train Models page appears under the same
conditions (feature selection is only offered when TML models are selected).

> **Polymer descriptors in GNNs:** the scaler is fitted on the polymer descriptors of
> the training graphs of each split. If a GNN-only experiment with polymer descriptors
> has no `feature_preprocessing` section, `standard_scaler` is used. Note that TML
> transformers are fitted on training + validation samples while the GNN descriptor
> scaler is fitted on training samples only, so the scaled values differ slightly between
> the two model families; we are working on aligning them. See
> [Polymer descriptor fusion](descriptors.md#polymer-descriptor-fusion).

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

  # Which splits to explain, numbered from 1 like the model files (e.g. GCN_1),
  # metrics.json and the bootstrap_iteration column.
  bootstraps: "all"         # or: [1, 2]

  fragmentation: "brics"    # brics | murcko_scaffold
  explain_set: "test"       # train | validation | test | all  — global plot; each model uses its own split's set
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

  # Which splits to explain, numbered from 1 like the model files (e.g. GCN_1),
  # metrics.json and the bootstrap_iteration column.
  bootstraps: "all"         # or: [1, 2]

  # train | validation | test | all  — global plot; each model uses its own split's set.
  # With tml_models.include_validation_in_training: true (default) TML models are trained
  # on train + validation, so "validation" (and "all") explain training data; use "test".
  explain_set: "test"
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

  # Applicability domain of the new polymers (optional; these are the defaults).
  applicability_domain:
    enabled: true          # false skips it
    metric: ruzicka_morgan # ruzicka_morgan | euclidean_model_inputs | a list of both
    k_neighbours: 5        # nearest training polymers
    z: 0.5                 # in domain if distance <= <d> + z·sigma; larger z widens the domain
    n_bins: 5              # score bins for the expected error
    fingerprint:           # count fingerprint of ruzicka_morgan (morgan | rdkitfp)
      fingerprint: morgan
      fp_size: 2048
      radius: 3            # morgan only
```

See [Predicting on external data](running-and-outputs.md#predicting-on-external-data)
and [Applicability domain](running-and-outputs.md#applicability-domain).
