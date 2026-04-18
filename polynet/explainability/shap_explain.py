"""
polynet.explainability.shap_explain
=====================================
SHAP-based explainability pipeline for traditional ML (TML) models.

Mirrors the GNN chemistry-masking pipeline in structure but uses SHAP values as
attributions and a flat CSV cache (one per descriptor) instead of nested JSON.
All three distribution plots are shared with the GNN pipeline via
``polynet.explainability.visualization`` — they only require a
``dict[str, list[float]]`` input.

Cache format
------------
``{experiment_path}/explanations/shap_{descriptor}.csv``::

    model_type,iteration,sample_id,class_idx,feature_A,feature_B,...
    RandomForest,0,poly_001,regression,0.12,-0.05,...
    RandomForest,1,poly_001,regression,0.10,-0.03,...
    XGBoost,0,poly_001,0,0.08,0.02,...

- ``model_type``: the sklearn model class name (e.g. ``"RandomForest"``)
- ``iteration``: integer bootstrap index
- ``sample_id``: string sample identifier (matches DataFrame index)
- ``class_idx``: ``"regression"`` for regression, integer string for classification
- Remaining columns: one SHAP value per selected feature

High-level API::

    from polynet.explainability.shap_explain import (
        compute_global_shap_attribution,
        compute_local_shap_attribution,
        InstanceAttributionResult,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from polynet.config.enums import AttributionPlotType, ImportanceNormalisationMethod, ProblemType
from polynet.config.paths import explanation_parent_directory, model_dir
from polynet.explainability.explain import GlobalAttributionResult
from polynet.explainability.visualization import (
    get_cmap,
    plot_attribution_bar,
    plot_attribution_distribution,
    plot_attribution_strip,
)

logger = logging.getLogger(__name__)

_REGRESSION_KEY = "regression"
_META_COLS = ["model_type", "iteration", "sample_id", "class_idx"]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class InstanceAttributionResult:
    """
    Return value for a single instance from :func:`compute_local_shap_attribution`.

    Attributes
    ----------
    sample_idx:
        The sample identifier (matches the DataFrame index).
    info_msg:
        Human-readable summary of the SHAP run for this instance.
    true_label:
        Ground-truth label as a string, or ``"N/A"``.
    predicted_label:
        Model predicted label as a string, or ``"N/A"``.
    attribution_df:
        Per-feature SHAP table with columns
        ``[Feature, SHAP Value, Normalised SHAP]``.
        Empty when ``warning`` is set.
    figure:
        Waterfall / force / bar figure for this instance.  ``None`` when
        ``warning`` is set.
    warning:
        Non-empty string when no SHAP values could be computed; ``None`` otherwise.
    """

    sample_idx: str | int
    info_msg: str
    true_label: str
    predicted_label: str
    attribution_df: pd.DataFrame
    figure: plt.Figure | None
    warning: str | None = None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _class_key(problem_type: ProblemType, target_class: int | None) -> str:
    """Return the string used to identify the class in the cache."""
    if problem_type == ProblemType.Regression:
        return _REGRESSION_KEY
    return str(target_class) if target_class is not None else "0"


def _parse_model_log_name(model_log_name: str) -> tuple[str, str, int]:
    """
    Parse ``"RandomForest-morgan_0"`` into ``("RandomForest", "morgan", 0)``.

    Model log names follow the convention built in
    ``polynet.training.tml.train_tml_ensemble``:
    ``f"{model_id}-{descriptor}_{iteration}"``.
    """
    model_type, log_name = model_log_name.split("-", 1)
    descriptor, iteration_str = log_name.rsplit("_", 1)
    return model_type, descriptor, int(iteration_str)


def _shap_cache_path(experiment_path: Path, descriptor: str) -> Path:
    """Return the path for the per-descriptor SHAP cache CSV."""
    return explanation_parent_directory(experiment_path) / f"shap_{descriptor}.csv"


def _load_shap_cache(experiment_path: Path, descriptor: str) -> pd.DataFrame:
    """Load the SHAP cache for a descriptor, returning an empty DataFrame if absent."""
    path = _shap_cache_path(experiment_path, descriptor)
    if path.exists():
        df = pd.read_csv(path, dtype={col: str for col in _META_COLS})
        feat_cols = [c for c in df.columns if c not in _META_COLS]
        if feat_cols:
            df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce")
        return df
    return pd.DataFrame(columns=_META_COLS)


def _save_shap_cache(cache_df: pd.DataFrame, experiment_path: Path, descriptor: str) -> None:
    """Persist the SHAP cache for a descriptor to disk."""
    path = _shap_cache_path(experiment_path, descriptor)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache_df.to_csv(path, index=False)


def _load_feature_transformer(experiment_path: Path, descriptor: str, iteration: int):
    """Load the fitted FeatureTransformer for a given descriptor and iteration."""
    path = model_dir(experiment_path) / f"{descriptor}_{iteration}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"FeatureTransformer not found at {path}. "
            "Ensure TML training has completed before running SHAP explanation."
        )
    return joblib.load(path)


def _select_shap_explainer(model, X_background: np.ndarray):
    """
    Auto-select a SHAP explainer based on the model type.

    - Tree-based models (RF, XGBoost, ExtraTrees) → ``shap.TreeExplainer``
    - Linear models (LinearRegression, LogisticRegression) → ``shap.LinearExplainer``
    - SVM and others → ``shap.KernelExplainer`` with a small background sample
    """
    import shap

    model_name = type(model).__name__
    tree_types = (
        "RandomForest",
        "ExtraTrees",
        "GradientBoosting",
        "XGB",
        "LGBM",
        "CatBoost",
        "DecisionTree",
    )
    linear_types = ("LinearRegression", "LogisticRegression", "Ridge", "Lasso", "ElasticNet")

    if any(model_name.startswith(t) for t in tree_types):
        return shap.TreeExplainer(model)

    if any(model_name.startswith(t) for t in linear_types):
        background = shap.maskers.Independent(X_background, max_samples=min(100, len(X_background)))
        return shap.LinearExplainer(model, background)

    # Fallback: KernelExplainer (slow, model-agnostic)
    n_bg = min(50, len(X_background))
    bg_sample = X_background[np.random.choice(len(X_background), n_bg, replace=False)]
    logger.warning(
        f"Using KernelExplainer for {model_name} — this may be slow. "
        "Consider using a tree-based or linear model for faster SHAP computation."
    )
    predict_fn = getattr(model, "predict_proba", model.predict)
    return shap.KernelExplainer(predict_fn, bg_sample)


def _compute_shap_values_for_row(
    explainer, x_row: np.ndarray, problem_type: ProblemType, target_class: int | None
) -> np.ndarray:
    """
    Compute SHAP values for a single instance, returning a 1-D array of shape
    ``(n_features,)``.

    Handles all SHAP output formats:
    - list of arrays (older SHAP, binary/multiclass): ``[class0_vals, class1_vals]``
    - 2-D array (regression or newer SHAP binary): ``(1, n_features)``
    - 3-D array (newer SHAP multiclass): ``(1, n_features, n_classes)``
    """
    x_2d = x_row.reshape(1, -1)
    values = explainer.shap_values(x_2d)

    # List of arrays: older SHAP TreeExplainer for binary/multiclass classification.
    # Each element has shape (1, n_features).
    if isinstance(values, list):
        idx = target_class if target_class is not None else 1
        idx = min(idx, len(values) - 1)
        values = values[idx]

    arr = np.asarray(values, dtype=float)

    if arr.ndim == 3:
        # (1, n_features, n_classes) — newer SHAP TreeExplainer for classification.
        class_idx = target_class if target_class is not None else 1
        class_idx = min(class_idx, arr.shape[2] - 1)
        arr = arr[0, :, class_idx]  # → (n_features,)
    elif arr.ndim == 2:
        arr = arr[0]  # (1, n_features) → (n_features,)

    return arr.ravel()  # guarantee 1-D


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


def compute_and_cache_shap(
    models: dict,
    descriptor_dfs: dict,
    experiment_path: Path,
    problem_type: ProblemType,
    explain_sample_ids: list[str],
    target_class: int | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Compute SHAP values for each TML model × sample pair and cache results to CSV.

    Parameters
    ----------
    models:
        Dict of ``{model_log_name: fitted_sklearn_model}`` as returned by
        ``train_tml``.  Keys follow ``"{ModelType}-{descriptor}_{iteration}"``.
    descriptor_dfs:
        Dict of ``{MolecularDescriptor | str: pd.DataFrame}`` as returned by
        ``load_dataframes``.  Each DataFrame has feature columns followed by the
        target column as the last column.
    experiment_path:
        Experiment root directory.  Cache CSVs are written to
        ``{experiment_path}/explanations/``.
    problem_type:
        Regression or Classification.
    explain_sample_ids:
        String sample identifiers to explain.
    target_class:
        Class index for classification.  Ignored for regression.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from descriptor name to the full (existing + new) cache DataFrame
        filtered to the requested models and sample IDs.
    """
    import shap  # defer import so SHAP is optional

    cls_key = _class_key(problem_type, target_class)
    explain_ids_set = set(str(s) for s in explain_sample_ids)

    # Normalise descriptor_dfs keys to strings
    str_dfs: dict[str, pd.DataFrame] = {str(k): v for k, v in descriptor_dfs.items()}

    # Group model keys by descriptor so we load each cache once
    descriptor_to_model_keys: dict[str, list[str]] = {}
    for model_log_name in models:
        _, descriptor, _ = _parse_model_log_name(model_log_name)
        descriptor_to_model_keys.setdefault(descriptor, []).append(model_log_name)

    results: dict[str, pd.DataFrame] = {}

    for descriptor, model_keys in descriptor_to_model_keys.items():
        if descriptor not in str_dfs:
            logger.warning(
                f"Descriptor '{descriptor}' not found in descriptor_dfs "
                f"(available: {list(str_dfs.keys())}). Skipping."
            )
            continue

        df = str_dfs[descriptor]
        X_all_df = df.iloc[:, :-1]  # all features, excluding target
        # Index aligned sample lookup
        df_index_str = X_all_df.index.astype(str)

        cache_df = _load_shap_cache(experiment_path, descriptor)
        new_rows: list[dict] = []

        for model_log_name in model_keys:
            model_type, _, iteration = _parse_model_log_name(model_log_name)
            model = models[model_log_name]

            # Load feature transformer for this descriptor × iteration
            try:
                transformer = _load_feature_transformer(experiment_path, descriptor, iteration)
            except FileNotFoundError as exc:
                logger.warning(str(exc))
                continue

            feature_names = transformer.get_feature_names_out()
            X_all = transformer.transform(X_all_df)  # (n_all, n_selected_features)

            # Build explainer using full transformed data as background
            try:
                explainer = _select_shap_explainer(model, X_all)
            except Exception as exc:
                logger.warning(f"Could not build SHAP explainer for {model_log_name}: {exc}")
                continue

            # Determine which sample IDs are missing from the cache
            cached_mask = (
                (cache_df["model_type"] == model_type)
                & (cache_df["iteration"] == str(iteration))
                & (cache_df["class_idx"] == cls_key)
            )
            cached_ids = set(cache_df.loc[cached_mask, "sample_id"].tolist())
            missing_ids = explain_ids_set - cached_ids

            if not missing_ids:
                logger.info(
                    f"[SHAP] All {len(explain_ids_set)} sample(s) already cached "
                    f"for {model_log_name}."
                )
                continue

            logger.info(
                f"[SHAP] Computing {len(missing_ids)} missing sample(s) " f"for {model_log_name}…"
            )
            for sample_id in missing_ids:
                # Find row in X_all matching this sample_id
                mask = df_index_str == sample_id
                if not mask.any():
                    logger.warning(
                        f"Sample '{sample_id}' not found in descriptor '{descriptor}'. " "Skipping."
                    )
                    continue

                row_idx = int(np.flatnonzero(mask)[0])
                x_row = X_all[row_idx]

                try:
                    shap_vals = _compute_shap_values_for_row(
                        explainer, x_row, problem_type, target_class
                    )
                except Exception as exc:
                    logger.warning(
                        f"SHAP computation failed for sample '{sample_id}' / "
                        f"{model_log_name}: {exc}"
                    )
                    continue

                row = {
                    "model_type": model_type,
                    "iteration": str(iteration),
                    "sample_id": sample_id,
                    "class_idx": cls_key,
                }
                row.update(dict(zip(feature_names, shap_vals.tolist())))
                new_rows.append(row)

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            cache_df = pd.concat([cache_df, new_df], ignore_index=True)
            _save_shap_cache(cache_df, experiment_path, descriptor)
            logger.info(f"[SHAP] Cached {len(new_rows)} new row(s) for descriptor '{descriptor}'.")

        # Return only the subset relevant to requested models + sample IDs
        model_types = {_parse_model_log_name(k)[0] for k in model_keys}
        iterations = {str(_parse_model_log_name(k)[2]) for k in model_keys}
        mask = (
            cache_df["model_type"].isin(model_types)
            & cache_df["iteration"].isin(iterations)
            & cache_df["sample_id"].isin(explain_ids_set)
            & (cache_df["class_idx"] == cls_key)
        )
        results[descriptor] = cache_df.loc[mask].reset_index(drop=True)

    return results


# ---------------------------------------------------------------------------
# Distribution & normalisation
# ---------------------------------------------------------------------------


def shap_cache_to_distribution(
    cache_df: pd.DataFrame,
    model_log_names: list[str],
    sample_ids: list[str],
    normalisation_type: ImportanceNormalisationMethod,
) -> dict[str, list[float]]:
    """
    Flatten a SHAP cache DataFrame into ``{feature_name: [scores]}`` for plotting.

    Applies the same four normalisation strategies as the GNN masking pipeline:

    - ``Global``: all scores divided by the global max-absolute value
    - ``PerModel``: each model type's scores divided by that model's max-absolute value
    - ``Local``: each (model × sample) unit divided by its own max-absolute value
    - ``NoNormalisation``: raw scores

    Parameters
    ----------
    cache_df:
        Filtered cache DataFrame (output of :func:`compute_and_cache_shap`).
    model_log_names:
        Model log names to include (parsed for model_type + iteration).
    sample_ids:
        Sample IDs to include.
    normalisation_type:
        Normalisation strategy.

    Returns
    -------
    dict[str, list[float]]
        ``{feature_name: [score_1, score_2, ...]}``.
        All individual scores are preserved (no averaging across models).
    """
    if cache_df.empty:
        return {}

    feature_cols = [c for c in cache_df.columns if c not in _META_COLS]
    sample_ids_set = set(str(s) for s in sample_ids)

    # Build lookup sets for filtering
    model_specs: list[tuple[str, str]] = []
    for key in model_log_names:
        mt, _, it = _parse_model_log_name(key)
        model_specs.append((mt, str(it)))

    distribution: dict[str, list[float]] = {f: [] for f in feature_cols}

    # ---- Global divisor pre-computation ----
    global_divisor = 1.0
    if normalisation_type == ImportanceNormalisationMethod.Global:
        all_vals = cache_df[feature_cols].to_numpy(dtype=float)
        max_abs = np.nanmax(np.abs(all_vals))
        global_divisor = float(max_abs) if max_abs > 0 else 1.0

    # ---- Per-model divisors ----
    per_model_divisors: dict[str, float] = {}
    if normalisation_type == ImportanceNormalisationMethod.PerModel:
        for mt, it in model_specs:
            key = f"{mt}_{it}"
            mask = (cache_df["model_type"] == mt) & (cache_df["iteration"] == it)
            vals = cache_df.loc[mask, feature_cols].to_numpy(dtype=float)
            max_abs = np.nanmax(np.abs(vals)) if vals.size > 0 else 1.0
            per_model_divisors[key] = float(max_abs) if max_abs > 0 else 1.0

    # ---- Collect scores ----
    for mt, it in model_specs:
        mask = (
            (cache_df["model_type"] == mt)
            & (cache_df["iteration"] == it)
            & cache_df["sample_id"].isin(sample_ids_set)
        )
        subset = cache_df.loc[mask]

        for _, row in subset.iterrows():
            unit_vals = row[feature_cols].to_numpy(dtype=float)

            if normalisation_type == ImportanceNormalisationMethod.Local:
                max_abs = np.nanmax(np.abs(unit_vals))
                divisor = float(max_abs) if max_abs > 0 else 1.0
            elif normalisation_type == ImportanceNormalisationMethod.PerModel:
                divisor = per_model_divisors.get(f"{mt}_{it}", 1.0)
            elif normalisation_type == ImportanceNormalisationMethod.Global:
                divisor = global_divisor
            else:
                divisor = 1.0

            for feat, val in zip(feature_cols, unit_vals / divisor):
                distribution[feat].append(float(val))

    # Drop features with no scores
    return {f: v for f, v in distribution.items() if v}


def merge_shap_attributions(
    cache_df: pd.DataFrame, model_log_names: list[str], sample_ids: list[str]
) -> pd.DataFrame:
    """
    Average SHAP values across ensemble models for each sample.

    Parameters
    ----------
    cache_df:
        Filtered SHAP cache (output of :func:`compute_and_cache_shap`).
    model_log_names:
        Model log names defining the ensemble.
    sample_ids:
        Sample IDs to include.

    Returns
    -------
    pd.DataFrame
        Indexed by ``sample_id``, one column per feature, values are mean SHAP
        values across models.
    """
    if cache_df.empty:
        return pd.DataFrame()

    feature_cols = [c for c in cache_df.columns if c not in _META_COLS]
    if not feature_cols:
        return pd.DataFrame()

    sample_ids_set = set(str(s) for s in sample_ids)

    # Use vectorized filtering — avoids slow row-wise apply and list & Series issues.
    model_types_set = {_parse_model_log_name(k)[0] for k in model_log_names}
    iterations_set = {str(_parse_model_log_name(k)[2]) for k in model_log_names}

    mask = (
        cache_df["model_type"].isin(model_types_set)
        & cache_df["iteration"].isin(iterations_set)
        & cache_df["sample_id"].isin(sample_ids_set)
    )
    subset = cache_df.loc[mask]

    if subset.empty:
        return pd.DataFrame()

    merged = (
        subset.groupby("sample_id")[feature_cols]
        .mean()
        .reset_index()
        .set_index("sample_id")
    )
    return merged


# ---------------------------------------------------------------------------
# Local visualisation
# ---------------------------------------------------------------------------


def plot_shap_waterfall(
    shap_values: dict[str, float],
    base_value: float = 0.0,
    plot_type: str = "waterfall",
    neg_color: str = "#40bcde",
    pos_color: str = "#e64747",
    top_n: int = 15,
) -> plt.Figure:
    """
    Render a per-instance SHAP explanation as a waterfall, force, or bar plot.

    Parameters
    ----------
    shap_values:
        ``{feature_name: shap_value}`` for one sample (ensemble-averaged).
    base_value:
        Model expected value (base line).  Not used for the simple bar variant.
    plot_type:
        ``"waterfall"``, ``"force"``, or ``"bar"``.
    neg_color, pos_color:
        Hex colours for negative / positive attributions.
    top_n:
        Maximum features to show (sorted by |SHAP|).
    """
    if not shap_values:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No SHAP values available", ha="center", va="center")
        ax.axis("off")
        return fig

    # Sort by |shap| descending, keep top_n
    items = sorted(shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    features = [k for k, _ in items]
    values = np.array([v for _, v in items])

    if plot_type == "bar":
        # Reuse shared distribution bar — wrap single instance as {feat: [val]}
        return plot_attribution_bar(
            {f: [v] for f, v in zip(features, values)}, neg_color=neg_color, pos_color=pos_color
        )

    # Waterfall / force: custom matplotlib implementation
    colors = [pos_color if v >= 0 else neg_color for v in values]

    if plot_type == "waterfall":
        # Features sorted: smallest |shap| at top, largest at bottom (conventional)
        features = features[::-1]
        values = values[::-1]
        colors = colors[::-1]

        running = base_value
        lefts = []
        for v in values:
            lefts.append(min(running, running + v))
            running += v

        fig, ax = plt.subplots(figsize=(9, max(4, len(features) * 0.4 + 1)))
        bars = ax.barh(
            range(len(features)),
            np.abs(values),
            left=lefts,
            color=[pos_color if v >= 0 else neg_color for v in values],
            edgecolor="white",
            linewidth=0.5,
            height=0.6,
        )
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=9)
        ax.axvline(base_value, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xlabel("Prediction value")
        ax.set_title(f"SHAP waterfall (base = {base_value:.3f})")

    else:  # force-style: stacked horizontal bars from base
        pos_feats = [(f, v) for f, v in zip(features, values) if v > 0]
        neg_feats = [(f, v) for f, v in zip(features, values) if v <= 0]

        fig, ax = plt.subplots(figsize=(10, 3))
        cursor = base_value
        for f, v in sorted(pos_feats, key=lambda x: -abs(x[1])):
            ax.barh(0, v, left=cursor, color=pos_color, edgecolor="white", height=0.6)
            if abs(v) > 0.01:
                ax.text(
                    cursor + v / 2,
                    0,
                    f"{f}\n{v:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                    weight="bold",
                )
            cursor += v
        cursor = base_value
        for f, v in sorted(neg_feats, key=lambda x: abs(x[1])):
            ax.barh(0, v, left=cursor + v, color=neg_color, edgecolor="white", height=0.6)
            if abs(v) > 0.01:
                ax.text(
                    cursor + v / 2,
                    0,
                    f"{f}\n{v:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                    weight="bold",
                )
            cursor += v
        ax.axvline(base_value, color="black", linewidth=1.2)
        ax.set_yticks([])
        ax.set_xlabel("Prediction value")
        ax.set_title("SHAP force plot")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# High-level public API
# ---------------------------------------------------------------------------


def compute_global_shap_attribution(
    models: dict,
    descriptor_dfs: dict,
    explain_sample_ids: list[str],
    experiment_path: Path,
    problem_type: ProblemType,
    neg_color: str = "#40bcde",
    pos_color: str = "#e64747",
    normalisation_type: ImportanceNormalisationMethod = ImportanceNormalisationMethod.PerModel,
    target_class: int | None = None,
    top_n: int | None = 10,
    plot_type: AttributionPlotType = AttributionPlotType.Ridge,
) -> dict[str, GlobalAttributionResult]:
    """
    Compute global SHAP feature attribution across the selected population.

    One :class:`~polynet.explainability.explain.GlobalAttributionResult` is
    returned per descriptor (e.g. one for Morgan, one for RDKit).

    Parameters
    ----------
    models:
        ``{model_log_name: model}`` from ``train_tml``.
    descriptor_dfs:
        ``{descriptor: pd.DataFrame}`` from ``load_dataframes``.
    explain_sample_ids:
        Sample IDs to include in the distribution.
    experiment_path:
        Experiment root — cache CSVs written here.
    problem_type:
        Regression or Classification.
    neg_color, pos_color:
        Attribution colours.
    normalisation_type:
        Normalisation strategy.
    target_class:
        Class index for classification.
    top_n:
        Show only top-N and bottom-N features by mean SHAP.  ``None`` shows all.
    plot_type:
        ``Ridge``, ``Bar``, or ``Strip``.

    Returns
    -------
    dict[str, GlobalAttributionResult]
        Keyed by descriptor name.
    """
    cache_by_descriptor = compute_and_cache_shap(
        models=models,
        descriptor_dfs=descriptor_dfs,
        experiment_path=experiment_path,
        problem_type=problem_type,
        explain_sample_ids=explain_sample_ids,
        target_class=target_class,
    )

    results: dict[str, GlobalAttributionResult] = {}

    for descriptor, cache_df in cache_by_descriptor.items():
        model_log_names = [k for k in models if _parse_model_log_name(k)[1] == descriptor]
        n_models = len({_parse_model_log_name(k)[0] for k in model_log_names})
        n_samples = cache_df["sample_id"].nunique()

        attribution_dict = shap_cache_to_distribution(
            cache_df=cache_df,
            model_log_names=model_log_names,
            sample_ids=explain_sample_ids,
            normalisation_type=normalisation_type,
        )

        if not attribution_dict:
            results[descriptor] = GlobalAttributionResult(
                figure=None,
                attribution_dict={},
                n_mols=0,
                n_models=n_models,
                n_frags_total=0,
                n_shown=0,
                target_class=target_class,
                normalisation_type=normalisation_type,
                warning=f"No SHAP values found for descriptor '{descriptor}'.",
            )
            continue

        n_frags_total = len(attribution_dict)

        # Apply top_n filtering by mean attribution
        if top_n is not None and n_frags_total > top_n * 2:
            means = {f: float(np.mean(v)) for f, v in attribution_dict.items()}
            sorted_feats = sorted(means, key=lambda f: means[f])
            top_feats = set(sorted_feats[:top_n] + sorted_feats[-top_n:])
            attribution_dict = {f: v for f, v in attribution_dict.items() if f in top_feats}
        n_shown = len(attribution_dict)

        if plot_type == AttributionPlotType.Bar:
            fig = plot_attribution_bar(attribution_dict, neg_color=neg_color, pos_color=pos_color)
        elif plot_type == AttributionPlotType.Strip:
            fig = plot_attribution_strip(attribution_dict, neg_color=neg_color, pos_color=pos_color)
        else:
            fig = plot_attribution_distribution(
                attribution_dict, neg_color=neg_color, pos_color=pos_color
            )

        results[descriptor] = GlobalAttributionResult(
            figure=fig,
            attribution_dict=attribution_dict,
            n_mols=n_samples,
            n_models=n_models,
            n_frags_total=n_frags_total,
            n_shown=n_shown,
            target_class=target_class,
            normalisation_type=normalisation_type,
        )

    return results


def compute_local_shap_attribution(
    models: dict,
    descriptor_dfs: dict,
    explain_sample_ids: list[str],
    experiment_path: Path,
    problem_type: ProblemType,
    neg_color: str = "#40bcde",
    pos_color: str = "#e64747",
    normalisation_type: ImportanceNormalisationMethod = ImportanceNormalisationMethod.PerModel,
    target_class: int | None = None,
    local_plot_type: str = "waterfall",
    predictions: dict | None = None,
) -> dict[str, list[InstanceAttributionResult]]:
    """
    Compute per-instance SHAP attribution with waterfall / force / bar plots.

    Parameters
    ----------
    models:
        ``{model_log_name: model}`` from ``train_tml``.
    descriptor_dfs:
        ``{descriptor: pd.DataFrame}`` from ``load_dataframes``.
    explain_sample_ids:
        Sample IDs to generate local explanations for.
    experiment_path:
        Experiment root.
    problem_type:
        Regression or Classification.
    neg_color, pos_color:
        Attribution colours.
    normalisation_type:
        Normalisation strategy applied to SHAP values before plotting.
    target_class:
        Class index for classification.
    local_plot_type:
        ``"waterfall"``, ``"force"``, or ``"bar"``.
    predictions:
        Optional ``{sample_id: {"true": ..., "predicted": ...}}`` for label
        display in the result.

    Returns
    -------
    dict[str, list[InstanceAttributionResult]]
        Keyed by descriptor name; each value is a list with one result per sample.
    """
    cache_by_descriptor = compute_and_cache_shap(
        models=models,
        descriptor_dfs=descriptor_dfs,
        experiment_path=experiment_path,
        problem_type=problem_type,
        explain_sample_ids=explain_sample_ids,
        target_class=target_class,
    )

    results: dict[str, list[InstanceAttributionResult]] = {}

    for descriptor, cache_df in cache_by_descriptor.items():
        model_log_names = [k for k in models if _parse_model_log_name(k)[1] == descriptor]

        # Merge ensemble
        merged_df = merge_shap_attributions(
            cache_df=cache_df, model_log_names=model_log_names, sample_ids=explain_sample_ids
        )

        # Global divisor (if needed)
        global_divisor = 1.0
        if normalisation_type == ImportanceNormalisationMethod.Global and not merged_df.empty:
            feature_cols = [c for c in merged_df.columns if c not in _META_COLS]
            max_abs = np.nanmax(np.abs(merged_df[feature_cols].to_numpy(dtype=float)))
            global_divisor = float(max_abs) if max_abs > 0 else 1.0

        descriptor_results: list[InstanceAttributionResult] = []

        for sample_id in explain_sample_ids:
            sid = str(sample_id)

            if merged_df.empty or sid not in merged_df.index:
                descriptor_results.append(
                    InstanceAttributionResult(
                        sample_idx=sid,
                        info_msg="",
                        true_label="N/A",
                        predicted_label="N/A",
                        attribution_df=pd.DataFrame(),
                        figure=None,
                        warning=f"No SHAP values found for sample '{sid}' (descriptor '{descriptor}').",
                    )
                )
                continue

            shap_row = merged_df.loc[sid]
            feature_cols = list(shap_row.index)
            raw_vals = shap_row.to_numpy(dtype=float)

            # Normalise
            if normalisation_type == ImportanceNormalisationMethod.Local:
                max_abs = np.nanmax(np.abs(raw_vals))
                divisor = float(max_abs) if max_abs > 0 else 1.0
            elif normalisation_type == ImportanceNormalisationMethod.Global:
                divisor = global_divisor
            else:
                divisor = 1.0

            normed_vals = raw_vals / divisor if divisor != 1.0 else raw_vals.copy()

            # Attribution DataFrame
            attr_df = (
                pd.DataFrame(
                    {
                        "Feature": feature_cols,
                        "SHAP Value": raw_vals,
                        "Normalised SHAP": normed_vals,
                    }
                )
                .sort_values("SHAP Value", key=abs, ascending=False)
                .reset_index(drop=True)
            )

            # Labels
            pred_info = (predictions or {}).get(sid, {})
            true_label = str(pred_info.get("true", "N/A"))
            predicted_label = str(pred_info.get("predicted", "N/A"))

            # Figure
            shap_dict = dict(zip(feature_cols, normed_vals.tolist()))
            fig = plot_shap_waterfall(
                shap_values=shap_dict,
                plot_type=local_plot_type,
                neg_color=neg_color,
                pos_color=pos_color,
            )
            plt.close("all")

            cls_str = "" if problem_type == ProblemType.Regression else f", class {target_class}"
            info_msg = (
                f"SHAP attribution for `{sid}` | {descriptor}"
                f"{cls_str} | normalisation: `{normalisation_type}`"
            )

            descriptor_results.append(
                InstanceAttributionResult(
                    sample_idx=sid,
                    info_msg=info_msg,
                    true_label=true_label,
                    predicted_label=predicted_label,
                    attribution_df=attr_df,
                    figure=fig,
                )
            )

        results[descriptor] = descriptor_results

    return results
