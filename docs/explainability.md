# Explainability

PolyNet provides two attribution pipelines that share the same configuration shape and
the same options on the Streamlit **Explain Models** page:

- **GNN** — chemistry-masking fragment attribution (substructure masking; Wu et al., *Nat. Commun.* 14,
  2585 (2023), [doi:10.1038/s41467-023-38192-3](https://doi.org/10.1038/s41467-023-38192-3)).
- **TML** — SHAP-based feature attribution, rendered with the native `shap` package.

Both have a **global** view (population-level summary) and a **local** view (per
molecule / per sample). The YAML config (`explainability`, `tml_explainability`) drives
the pipeline; some options are only exposed in the GUI and are noted as such below.

- [GNN chemistry-masking attribution](#gnn-chemistry-masking-attribution)
- [TML SHAP attribution](#tml-shap-attribution)
- [Normalisation strategies](#normalisation-strategies)
- [Averaged vs per-model display (local)](#averaged-vs-per-model-display-local)
- [Prediction breakdown (local panels)](#prediction-breakdown-local-panels)
- [Caching](#caching)
- [Statistical model comparison (Analyse Results)](#statistical-model-comparison-analyse-results)

See [Configuration reference](configuration.md#explainability-gnn) for the exact YAML
fields.

---

## GNN chemistry-masking attribution

Each fragment's importance is the change in prediction when that fragment is masked
from the graph pooling step: attribution = `Y_pred_full − Y_pred_masked` for each
fragment occurrence. Fragments come from BRICS or Murcko-scaffold fragmentation, and
attributions are aggregated per fragment.

- **Global** — a fragment-attribution distribution plot. `plot_type` selects
  `ridge`, `bar`, or `strip`.
- **Local** — one per-molecule atom heatmap (the per-fragment scores mapped back to
  atom weights) plus a per-fragment attribution table.

`explain_set` selects which molecules go into the global distribution;
`local_explain_mol_ids` selects which molecules get per-molecule heatmaps/CSVs (the
local step is skipped entirely when `null`).

**Each model explains only its own split.** Every model was trained on one repeated
split, so in the global view a model only explains the `explain_set` molecules *of its
own split* (e.g. its own test molecules) — never molecules that were in its training
data. With several splits the distribution therefore pools each model's own held-out
molecules. `explain_set: all` uses every molecule of the model's split. The same rule
applies to the TML SHAP summary below, and to the GUI when a set (Training /
Validation / Test) is selected; picking "All" or external data applies no restriction.
Local (per-molecule) explanations are not restricted: they explain the molecules you
list with every selected model, and the prediction breakdown shows which set the
molecule was in for each model.

Implementation notes:

- Masking deletes (rather than zeros) the fragment's nodes from pooling, so the mean
  denominator / max candidate set reflect only the remaining nodes.
- The remaining nodes are pooled with the model's own pooling step, including its
  monomer weighting (`apply_weighting_to_graph`): the weighted mean of
  `before_pooling` and the per-monomer pooling of `per_monomer_pooling`. Masking
  nothing therefore gives exactly the model's prediction, so the attribution only
  reflects the removed fragment. (Before this was fixed, masked copolymers were
  pooled with a plain unweighted pool, which added an offset unrelated to the
  fragment.)
- Atom indices are mapped through the graph's per-node `monomer_id`, so attribution
  works correctly even when PSMILES wildcard (`*`) atoms were stripped during
  featurisation and the graph has fewer nodes than the RDKit molecule.
- A fragment that spans the entire graph (e.g. a tiny monomer with no breakable bonds)
  is skipped — masking it would leave nothing to pool.

## TML SHAP attribution

SHAP values are computed on the **transformed (normalised + feature-selected)**
descriptors — i.e. the exact space the model was trained on — using the same fitted
`FeatureTransformer` saved during training. PolyNet auto-selects the explainer by model
type: `TreeExplainer` for Random Forest / XGBoost, `LinearExplainer` for
linear/logistic regression, and `KernelExplainer` for SVM.

Both views use the native `shap` package:

- **Global** — `shap.summary_plot`. `plot_type` selects `beeswarm` (one dot per
  sample per feature, coloured by feature value), `bar` (mean |SHAP|), or `violin`.
  In the beeswarm/violin, the x-axis is the SHAP value (model space) while point
  **colour** reflects the raw feature value (standard SHAP convention).
- **Local** — `shap.plots.waterfall` / `force` / `bar`, selected by `local_plot_type`.
  Plots honour the user's positive/negative colours, and (GUI) a "top N features"
  selector controls `max_display` for the waterfall and bar plots (the force plot has
  no such limit, so it is truncated to the top-N by |SHAP| before drawing).

`explain_set` selects the sample population for the global plot;
`local_explain_sample_ids` selects which samples get per-instance plots.

> **Warning — for TML, the validation set is training data by default.** With
> `tml_models.include_validation_in_training: true` (the default), traditional ML
> models are fitted on the training and validation samples together. Explaining
> `explain_set: validation` (or `all`, or the GUI "Validation" / "All" pills) then
> explains samples the TML model was trained on, not held-out ones; use
> `explain_set: test` to explain predictions on unseen samples. The CLI logs and the
> GUI shows this warning when such a set is selected. With
> `include_validation_in_training: false` the validation samples are held out (as for
> GNNs) and no warning is shown.

## Normalisation strategies

The `normalisation` option (`local`, `global`, `per_model`, `no_normalisation`)
controls how attribution magnitudes are scaled before plotting. It only affects the
displayed scale — raw attributions are always what is cached.

| Strategy | Meaning |
|---|---|
| `no_normalisation` | Raw attribution values straight from the pipeline. |
| `local` | Each instance is divided by its own largest \|attribution\|, so every instance has at least one value of magnitude 1. |
| `per_model` | Each trained model — a specific (representation × algorithm × bootstrap) — is divided by the largest \|attribution\| found across all of that model's instances. |
| `global` | Everything is divided by the single largest \|attribution\| found across all models and instances. |

Order of operations for the **global** views:

- **TML** normalises *before* averaging across bootstraps, at the
  `(representation × model × bootstrap)` grain. This is what makes `per_model` distinct
  from `global` — without it, averaging away the per-model dimension would collapse the
  two. (`global` and `no_normalisation` are unaffected by the ordering, since dividing
  by a constant commutes with averaging.)
- **GNN** keeps every `(model × molecule × occurrence)` score as a separate point in
  the distribution.

When a single model is selected, `per_model` and `global` are mathematically identical
(the model's max *is* the global max).

## Averaged vs per-model display (local)

> **GUI option** (Explain Models → local tab). The non-GUI pipeline always uses
> `Average`.

When several models are selected for the same molecule, the **local** view offers two
display strategies:

- **Average across models** (default) — the selected models are merged into a single
  explanation, giving one plot per molecule. For TML the per-sample SHAP vectors are
  averaged; for GNN the per-fragment scores are averaged.
- **Show each model separately** — one plot per `model × molecule`, each labelled with
  the model and bootstrap (e.g. `Random Forest polyBERT — bootstrap 1`,
  `GCN — bootstrap 2`).

Normalisation behaves identically in both modes; the only difference is whether the
models are aggregated into one plot or shown individually.

## Prediction breakdown (local panels)

> **GUI feature.**

Each local explanation panel shows the molecule's true label plus a table with one row
per selected **model × bootstrap**: the model name, the bootstrap, which **set**
(train / validation / test) the molecule belonged to in that bootstrap, and that
bootstrap's **predicted value**.

Because each bootstrap stores predictions only for the molecules in its own split, a
molecule may be absent from some bootstraps. Those rows are still shown, with `Set` and
`Predicted` displayed as `N/A`, so the table always reflects every selected model ×
bootstrap.

Model, descriptor, and metric names are shown with clean display labels (e.g.
`random_forest-polybert` → `Random Forest polyBERT`, `mae` → `MAE`), with an optional
abbreviated mode (`RF polyBERT`) for compact figures.

## Caching

Attributions are computed once and reused:

- **GNN** — raw masking attributions are written to a JSON cache under
  `explanations/`. Re-running reuses cached `(model, molecule, fragmentation, class)`
  entries. The cache records the version of the masking computation; a cache written
  by an older version is discarded (with a warning) and recomputed.
- **TML** — SHAP values are written to `explanations/shap_{descriptor}.csv` (one file
  per descriptor) and reused on subsequent runs.

Delete the cache files to force recomputation — for example after retraining or
changing the target class. Normalisation is **never** written to the cache, so changing
the normalisation strategy does not require recomputation.

## Statistical model comparison (Analyse Results)

> **GUI feature** (Analyse Results page). Works for TML-only, GNN-only, and hybrid
> experiments — model lists are derived from the predictions table, so any trained
> model can be compared.

The Analyse Results page can statistically compare model predictions and metrics:

- **Predictions** — pairwise McNemar tests (classification) or paired Wilcoxon
  signed-rank tests on the per-sample **absolute errors** `|y_true − y_pred|`
  (regression). Comparing absolute errors tests for a difference in accuracy;
  signed residuals would only test for a difference in bias.
- **Metrics** — pairwise tests on a metric measured over the repeated random splits,
  plus box plots of the per-split metric distributions. Two tests are available:

  | Test | When to use |
  |---|---|
  | **Wilcoxon signed-rank** (default) | Non-parametric, paired by split. Needs **at least 6 splits**: with `k` splits the smallest two-sided p-value it can return is `2 / 2^k` (0.0625 for 5 splits), so with 5 or fewer no difference can be significant at 0.05. The page warns when there are fewer than 6 splits. |
  | **Nadeau–Bengio corrected t-test** | Recommended for repeated splits. The splits share training data, so the per-split metrics are not independent and ordinary paired tests are too optimistic. The corrected resampled t-test (Nadeau & Bengio, *Machine Learning* 52, 239–281, 2003) replaces the variance term `σ²/k` of the paired t-test by `(1/k + n_test/n_train)·σ²`, with `k − 1` degrees of freedom. |

  For the corrected test, `n_test / n_train` is computed from `split_indices.json` as
  the mean size of the analysed set (validation or test) over the mean size of the
  training set **excluding** the validation set (the conservative choice, giving a
  larger correction). It needs at least 2 splits and is not meaningful for
  training-set metrics.

Because comparing *k* models pairwise produces `k·(k−1)/2` simultaneous tests, a
**multiple-comparison correction** is applied to the p-value matrices, selectable in
the UI:

| Option | Controls |
|---|---|
| **Holm-Bonferroni** (default) | Family-wise error rate |
| **Bonferroni** | Family-wise error rate |
| **Benjamini-Hochberg (FDR)** | False discovery rate |
| **None (raw p-values)** | No correction |

The significance markers (`*`, `**`, `***`) and the displayed p-values reflect the
corrected values.
