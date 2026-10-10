# Changelog

All notable changes to PolyNet are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/).

## [1.0.0] — Unreleased

First tagged release. Earlier development versions were labelled `0.1.0` but
never released; experiments run with them still load, but some results are not
reproduced exactly (see **Results that change**).

### Features

- **Pipeline** from (P)SMILES to trained models, metrics, plots and
  explanations, run from a YAML config (`polynet run --config experiment.yaml`)
  or the Streamlit GUI (`polynet`). Both call the same pipeline stages.
- **Polymers:** homopolymers and copolymers with any number of monomers and
  molar ratios; ratio weighting before message passing, before pooling or per
  monomer.
- **Representations:** molecular graphs for GNNs (GCN, GAT, GraphSAGE, MPNN,
  CGGNN, TransformerConv); RDKit descriptors, Morgan / RDKit fingerprints (size and
  radius configurable), polyBERT embeddings and PolyMetriX polymer descriptors
  for traditional ML (random forest, XGBoost, SVM, linear / logistic models).
- **Splits:** repeated train/validation/test splits drawn with astartes —
  random (default), Kennard–Stone, SPXY, k-means, OptiSim or target-property,
  optionally stratified; `val_ratio` and `test_ratio` are fractions of the full
  dataset.
- **Hyperparameter optimisation:** GNN (Ray Tune with ASHA) and TML
  (randomised search with shuffled k-fold CV); number of configurations and
  search grids configurable, searched grids saved.
- **Statistics:** Wilcoxon signed-rank test on absolute errors (regression),
  McNemar (classification) and the Nadeau–Bengio corrected resampled t-test on
  metrics, with multiple-comparison correction.
- **Explainability:** GNN chemistry masking (Wu et al., *Nat. Commun.* 2023)
  with BRICS / Murcko fragments, and SHAP for TML models.
- **Predictions on new data:** per-architecture, all-GNN and per TML model ×
  representation ensembles (mean ± standard deviation, or majority vote + vote
  fraction), with the applicability domain of each polymer (kNN Ruzicka or
  Euclidean distance) and the expected error from held-out test errors.
- **Built-in benchmarks:** `curated_tg` (downloaded on first use) and
  `fluorine_nmr_snr` (bundled; 418 fluorinated copolymers with ¹⁹F NMR SNR).

### Results that change

Compared with unreleased development versions before October 2026
(PR #74, #75):

- splits, now drawn with astartes, with `val_ratio` a fraction of the full
  dataset (old splits: `val_ratio = old × (1 − test_ratio)`);
- GNN best epoch, now chosen on the validation RMSE over the whole set (was
  averaged per batch);
- classification GNN HPO, which now tunes class weights;
- TML regression HPO (shuffled folds);
- default number of HPO configurations (GNN 150 → 50, TML 30 → 50);
- copolymer masking attributions (masked graphs now use the model's own
  monomer weighting);
- CLI canonicalisation, now applied as configured;
- SVM SHAP values, now reproducible.

### Packaging

- `polynet run` installs the YAML pipeline as a command (replaces
  `python scripts/run_pipeline.py`).
- `polynet check` runs the full pipeline on synthetic data and reports what
  works.
- `polynet --version` and `polynet.__version__`.
- `black`, `isort` and `ipykernel` moved to the development dependencies;
  unused `gbigsmiles` removed.
- canonicalize-psmiles is no longer a dependency of the package (it is not on
  PyPI and has a non-commercial licence). `poetry install` still installs it;
  otherwise the GUI offers a one-click install when PSMILES data is loaded, or
  run `polynet install-psmiles`. SMILES data does not need it.
- Package metadata moved to the standard `[project]` table.

[1.0.0]: https://github.com/Biomaterials-for-Medical-Devices-AI/PolyNet/releases/tag/v1.0.0
