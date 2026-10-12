# Changelog

All notable changes to PolyNet are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/).

## [1.0.2] — 2026-10-12

### Added

- Every GUI page shows the PolyNet version in the bottom-right corner ("You are
  using PolyNet vX.Y.Z").

### Fixed

- `polynet --help` (also `-h` and `polynet help`) lists PolyNet's commands
  instead of Streamlit's options; `polynet install-psmiles --help` shows its
  options instead of starting the installation.
- The GUI welcome page shows how to cite PolyNet (reference and BibTeX with the
  Zenodo DOI).
- README: installation from PyPI (`pip install polynet`), with Poetry for
  development; PyPI badge; the Zenodo DOI badge loads on the PyPI page.

## [1.0.1] — 2026-10-12

### Added

- GUI, Create Experiment: selecting a built-in benchmark sets its SMILES, ID
  and target columns and problem type, and locks them (`curated_tg`:
  `PSMILES`, `ID`, `Tg(K)`; `fluorine_mri_snr`: `smiles_1`–`smiles_6`, `ID`,
  `SNR`; both regression). The target name and units used in plots get
  defaults and stay editable. Switching back to uploading a file clears them.
- GUI, Create Experiment: choose how many polymers of a built-in benchmark to
  model. `curated_tg` (7,367 polymers) explains how long training on all of
  them can take and defaults to a random subset of 1,000; smaller benchmarks
  (`fluorine_mri_snr`, 418) default to all polymers but 50. Any size up to the
  full dataset can be chosen. The subset is reproducible (the same polymers for
  the same size, and a smaller subset is contained in a larger one), and the
  experiment saves the selected polymers only. Without an ID column, polymers
  are numbered by their row in the full dataset, so a polymer keeps its number
  whatever the subset size.
- When polymers are left out, any number of them (100 by default, or all of
  them when fewer are left), with their measured values, can be downloaded as
  a CSV, with an explanation of how to use it: upload it on the Predict page to
  compare predictions with measurements on polymers the models have never seen.
  Numbers outside the valid range are reported: the download is disabled for an
  invalid download size, and the experiment cannot be saved with an invalid
  number of polymers. When every polymer is used, the page says that none
  are left and how to keep some aside.

### Changed

- The `fluorine_nmr_snr` benchmark is renamed `fluorine_mri_snr`: its `SNR` is
  the ¹⁹F MRI signal-to-noise ratio (Reis et al. 2021), not NMR. Configs that
  use `fluorine_nmr_snr` still load.

### Fixed

- The GUI (`polynet`, `polynet gui`) starts with Streamlit's file watcher off.
  With the dependency versions a fresh `pip install` resolves, the watcher's
  module scan logged a `No module named 'torchvision'` traceback for every
  `transformers` module on each rerun once polyBERT was used. Pass
  `--server.fileWatcherType auto` to reload the app on source changes when
  developing (see `docs/development.md`).
- README logo and links use full GitHub URLs, so they work on the PyPI page.

## [1.0.0] — 2026-10-12

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
  CGCNN, TransformerConv); RDKit descriptors, Morgan / RDKit fingerprints (size and
  radius configurable), polyBERT embeddings and PolyMetriX polymer descriptors
  for traditional ML (random forest, XGBoost, SVM, linear / logistic models).
- **Splits:** repeated train/validation/test splits drawn with astartes —
  random (default), Kennard–Stone, SPXY, k-means or target-property,
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

### Renamed and removed

- The `CGGNN` architecture is now called `CGCNN` (crystal graph convolutional
  neural network, Xie & Grossman 2018), its published name. Configs and
  experiments that use `CGGNN` still load and are mapped to `CGCNN`.
- The `optisim` sampler is disabled: astartes 1.3.3 assigns polymers to
  OptiSim clusters at random, so the split is not extrapolative. Use `kmeans`
  for a cluster-based split. Experiments saved with `optisim` still load.

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

[1.0.2]: https://github.com/Biomaterials-for-Medical-Devices-AI/PolyNet/releases/tag/v1.0.2
[1.0.1]: https://github.com/Biomaterials-for-Medical-Devices-AI/PolyNet/releases/tag/v1.0.1
[1.0.0]: https://github.com/Biomaterials-for-Medical-Devices-AI/PolyNet/releases/tag/v1.0.0
