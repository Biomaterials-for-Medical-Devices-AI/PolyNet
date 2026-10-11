# Explainability

PolyNet provides two attribution pipelines that share the same configuration shape and
a Streamlit workflow: **Explain Models** calculates importances and shows default
previews; **Analyse Results** loads saved analyses for full plot customisation:

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

### SME fragment labels and publication structure key (GUI)

On **Analyse Results → Feature importance plots → Population**, GNN SME ridge,
bar and strip plots offer **Fragment labels: Fragment SMILES / F tags**. The default
uses the original atom-letter fragment labels (for example `CO` or `c1ccccc1`),
rather than atomic-number SMARTS such as `[#6]-[#8]`. This is presentation
only: raw attribution values, normalisation, ranking and top/bottom selection are
unchanged. Page 5 previews and CLI defaults are unchanged.

**Tag colour mode** applies to both original fragment labels and F tags. Choose
**Custom colour** (default black, `#000000`) to use a colour picker or enter a hex
code, or **Match distribution** to use each fragment's ridge/bar colour. For strip
plots, which colour individual observations separately, tags match the mean diamond.
The separate structure figure uses the same tag colours; atom labels and bonds
remain black. Colours update with the selected data and plot palette without
changing attribution values or fixed F assignments.

**Observation counts** can be hidden (default), placed **Next to tag** as
`F1 (n=…)`, or on the **Right side** as `n=…`, for ridge, bar and strip plots.
The count is the number of attribution values in that fragment's displayed
distribution. Repeated occurrences and selected model/bootstrap combinations
contribute separate observations; this is not a count of unique molecules.
Changing models, sets or molecule selection updates the counts. Top-N filtering
only changes which fragments are displayed; it does not renumber F tags.

### Exporting explanation plots (GUI)

Every population plot and every individual molecule/model plot on page 6 has its
own **Export format** selector and **Export plot** button. **SVG** is the default;
**PDF** also preserves vector lines, points, distributions and text. Font outlines
in SVG and embedded fonts in PDF avoid relying on fonts installed on the reader's
computer. **PNG** is lossless and records the selected **Export DPI** (72–1200,
default 300). The controls cover all SME and SHAP plot styles and each descriptor.

Local GNN heatmaps are redrawn by RDKit at the requested resolution for their
three-inch molecule panels, rather than enlarging a fixed 600-pixel bitmap.
These heatmaps and SHAP force-plot gradients remain embedded raster images in
SVG/PDF; their source detail limits enlargement. Other plot artists, including
colourbars, retain vector detail. Exports include the current colours, labels,
observation counts, normalisation and selection, with a tight boundary to include
right-side counts. Screen previews use 150 DPI to keep rendering responsive.
Downloading does not rerun inference or write attribution caches. PNGs exceeding
40 million pixels prompt a lower DPI or vector export.

The separate fragment structure key retains its own vector SVG and publication
PNG export controls, described below.

### Separate fragment structure key (GUI)

The masking cache stores canonical fragment **SMILES**, rather than SMARTS queries.
The CSV's additional SMARTS column is generated with RDKit `MolToSmarts`; it is
not used as the distribution plot label.
With F tags, `F1`, `F2`, … are assigned by sorted original fragment identifiers across
the **entire saved analysis**, not just the selected models or molecules. Changing
models, bootstraps, set, normalisation or top-N preserves the fragment's tag. Tags
are scoped to one analysis; different saved analyses can have different mappings.

**Download F-to-SMARTS CSV** exports `Tag`, `SMARTS` and `Fragment` (the original
cache identifier). It contains exactly the displayed fragments, in descending
plot order. Gaps in the F numbering after filtering are expected.

**Create separate fragment structure figure** draws each fragment with its F tag
beneath it. It is a separate structure key, not an overlay on the distribution.
Controls include molecule panel width/height, horizontal/vertical spacing, structures
per row, atom/tag fonts, bond length and line width. Physical sizes are millimetres;
fonts and line widths are points. Molecule height excludes the tag strip below it.
RDKit may shorten bonds to fit a large fragment inside the chosen panel.

The **SVG** export contains vector bonds and glyphs and remains sharp at any size.
Structures are drawn with [RDKit's MolDraw2D SVG/Cairo drawers](https://www.rdkit.org/docs/source/rdkit.Chem.Draw.rdMolDraw2D.html).
**Prepare publication PNG** redraws the molecules at 300, 600 (default) or 1200 DPI
and records DPI metadata, rather than enlarging the preview. Choose an appropriate
physical size for the publication. To avoid excessive raster memory use, PNG canvases
above 40 million pixels are rejected; SVG remains available. Structure exports retain
stereochemistry and charges, use black atom labels/bonds, and introduce no new
dependency or attribution calculation. Invalid structures produce an explicit error.

### Selecting models and samples

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

> **GUI option** (Analyse Results → Feature importance plots → loaded local analysis). The non-GUI pipeline always uses
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
  per descriptor) and reused on subsequent runs. A cache written by an older version is
  discarded (with a warning) and recomputed. Version 3 also stores each explainer
  baseline for additive waterfall and force plots; run page 5 again for older SHAP caches.

Models without a tree or linear explainer (e.g. SVMs) use SHAP's `KernelExplainer`, which
samples randomly. PolyNet seeds its background sample and each sample's computation, so
their SHAP values are reproducible: they do not depend on the run or on which other
samples are explained.

Delete the cache files to force recomputation — for example after retraining or
changing the target class. Normalisation is **never** written to the cache, so changing
the normalisation strategy does not require recomputation.

### Completed GUI analysis snapshots

Page 5 also saves `explanations/analyses/{family}_{selection_hash}.json`
under the selected data source (experiment or external dataset). A versioned snapshot
contains raw selected scores, model/sample/class/fragmentation selection, per-model
sample restrictions, descriptor feature values or molecule SMILES, and prediction
breakdowns. It contains no fitted model objects and uses atomic JSON writes. Re-running
the same selection replaces its snapshot; different selections have separate entries.
Normalisation, colours and plot styles remain presentation settings.

Page 6 automatically loads the selected snapshot. Population and individual-molecule
views use the same raw attributions; there are no separate Global/Local entries or run
dates in the selector. Individual molecules can be chosen from any completed analysis.
In both views, choose architectures/models, bootstrap iterations and the explained set,
or refine the selection to individual model/bootstrap combinations. Selecting one GNN
architecture produces its own population plot. Set membership is read from the saved
prediction breakdown for each model/bootstrap/sample pair; it is not a global molecule
label. Only saved pairs in the requested set are displayed. External data and snapshots
without set metadata do not offer train/validation/test filters. The snapshot's class and
fragmentation remain fixed; select another saved analysis to view different ones.
Select the cached model/bootstrap combinations to show, then choose an ensemble average
or separate plots. Each molecule lists its available combinations. Missing pairs are
excluded from averages and never computed on page 6; run page 5 to add them. The older
GUI/CLI raw-cache import option has been removed: plotting uses completed GUI snapshots.
Rendering calls the core `render_cached_*_attribution` APIs and never writes raw caches.
Local renderers accept per-model sample restrictions, also applied to prediction panels;
PerModel scaling still uses each selected model's complete saved population as reference.

GUI retraining or representation rebuilding removes old explanations and snapshots,
including those for external predictions, because they refer to the old models or inputs.
A GUI explanation rerun warns before replacing only the chosen analysis and its cached
model/bootstrap/sample/class/fragmentation scores; other saved analyses retain their own
raw scores. This does not change the CLI's cache reuse. For CLI retraining, remove outdated
raw caches manually before explaining newly trained models.

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

## Local normalisation and SHAP baselines

`PerModel` divides by each fitted model/bootstrap's largest **absolute** attribution
across its whole saved analysis, before averaging. Choosing a single molecule does not
change that reference maximum. This applies to TML and GNN local plots; individual models' raw scores remain
unchanged. A zero maximum uses a divisor of one. `Global` uses one maximum across all
selected models in the saved population. `Local` scales the displayed local vector.

SHAP caches retain the expected output for each model/class as `base_value`, outside the
feature columns. Local plots average baselines across exactly the cached combinations
used for the contributions. A feature absent from a model's selected inputs contributes
zero to that model's ensemble average. Baselines and contributions use the same scaling,
so unnormalised waterfall/force endpoints reconstruct the explained output and normalised
plots are labelled as scaled model output. Force plots include every contribution even
when a smaller feature limit is selected; waterfall/bar still honour that limit.

Classification output depends on the explainer: probability explainers (e.g. random
forests) use class probabilities, while binary linear models and some tree explainers
use margins/log-odds. For class zero,
both a single-output linear baseline and its contributions are negated. Separate plots
help interpret outputs from models with different scales. A snapshot missing a baseline
cannot render a local SHAP plot; run the analysis again on page 5.
