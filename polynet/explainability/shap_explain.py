"""
polynet.explainability.shap_explain
=====================================
SHAP-based explainability pipeline for traditional ML (TML) models.

Mirrors the GNN chemistry-masking pipeline in structure but uses SHAP values as
attributions and a flat CSV cache (one per descriptor) instead of nested JSON.

Global attribution plots are rendered with the native ``shap`` package
(``shap.summary_plot`` — beeswarm / bar / violin) so they match the look and
conventions users expect from SHAP. Per-instance (local) plots remain custom
arrow-based waterfall / force charts.

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

from contextlib import contextmanager
from dataclasses import dataclass
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from polynet.config.display_names import prettify_label
from polynet.config.enums import (
    ExplanationAggregation,
    ImportanceNormalisationMethod,
    ProblemType,
    ShapGlobalPlotType,
)
from polynet.config.paths import explanation_parent_directory, model_dir
from polynet.explainability.explain import GlobalAttributionResult
from polynet.explainability.visualization import get_cmap

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
    model_label:
        Display label of the model this result belongs to (e.g.
        ``"Random Forest polyBERT — bootstrap 1"``) when explanations are shown
        per model (``Separate`` aggregation). ``None`` for the averaged ensemble.
    """

    sample_idx: str | int
    info_msg: str
    true_label: str
    predicted_label: str
    attribution_df: pd.DataFrame
    figure: plt.Figure | None
    warning: str | None = None
    model_label: str | None = None


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

    # Track whether the per-class slice has already been chosen, so the 2-D
    # branch below doesn't re-apply class handling to an already-selected array.
    class_already_selected = False

    # List of arrays: older SHAP TreeExplainer for binary/multiclass classification.
    # Each element has shape (1, n_features) and already corresponds to one class.
    if isinstance(values, list):
        idx = target_class if target_class is not None else 1
        idx = min(idx, len(values) - 1)
        values = values[idx]
        class_already_selected = True

    arr = np.asarray(values, dtype=float)

    if arr.ndim == 3:
        # (1, n_features, n_classes) — newer SHAP TreeExplainer for classification.
        class_idx = target_class if target_class is not None else 1
        class_idx = min(class_idx, arr.shape[2] - 1)
        arr = arr[0, :, class_idx]  # → (n_features,)
    elif arr.ndim == 2:
        # (1, n_features) — regression, or binary classification where SHAP
        # returns a single set of contributions for the POSITIVE class (class 1).
        # LinearExplainer (LogisticRegression) and binary TreeExplainer (XGBoost)
        # both do this. For binary classification the negative class (class 0) is
        # the exact mirror of the positive class, so negate when class 0 is
        # requested — this keeps the 2-D path consistent with the 3-D and list
        # paths (which return class0 = -class1) instead of silently ignoring
        # ``target_class``. Skip this when the class was already selected from a
        # list output, and never negate regression (target_class is None).
        row = arr[0]  # (1, n_features) → (n_features,)
        if (
            not class_already_selected
            and problem_type != ProblemType.Regression
            and target_class is not None
            and int(target_class) == 0
        ):
            row = -row
        arr = row

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
    cache_root: Path | None = None,
    target_col: str | None = None,
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
        ``load_dataframes``.  For training data, the target column is last; for
        external (label-free) datasets it is absent.  Pass ``target_col`` so
        the pipeline drops the right column rather than blindly using ``iloc[:, :-1]``.
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

    # Models / feature-transformers always load from ``experiment_path``; SHAP
    # caches are read/written under ``cache_root`` (the unseen-dataset folder for
    # external datasets). Defaults to the experiment.
    cache_root = cache_root if cache_root is not None else experiment_path

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
        X_all_df = df.drop(columns=[target_col], errors="ignore")
        # Index aligned sample lookup
        df_index_str = X_all_df.index.astype(str)

        cache_df = _load_shap_cache(cache_root, descriptor)
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
            _save_shap_cache(cache_df, cache_root, descriptor)
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
# Global SHAP matrix assembly & normalisation
# ---------------------------------------------------------------------------


def _normalise_cache_attributions(
    cache_df: pd.DataFrame, normalisation_type: ImportanceNormalisationMethod
) -> pd.DataFrame:
    """
    Normalise per-row SHAP values in a cache **before** ensemble averaging.

    Each row is one ``(model_type, iteration, sample)`` SHAP vector. Because the
    cache is per-descriptor, ``(model_type, iteration)`` identifies one trained
    model instance (representation × model × bootstrap). Normalisation is applied
    at the grain implied by the strategy; the caller then averages across
    bootstraps/models per sample for display:

    - ``NoNormalisation``: rows unchanged.
    - ``Local``: each row divided by its own max-absolute value, so every
      (model × bootstrap × instance) row has a ±1.
    - ``PerModel``: each row divided by the max-absolute value across all rows of
      its ``(model_type, iteration)`` — one (instance, feature) within that
      trained model reaches ±1.
    - ``Global``: every row divided by the single global max-absolute value.

    Returns a copy with the feature columns scaled; meta columns are untouched.
    A zero divisor is replaced with 1.0 so all-zero blocks pass through unchanged.

    Dividing by a constant commutes with averaging, so ``Global`` and
    ``NoNormalisation`` are unaffected by doing this before vs. after the merge;
    only ``Local`` and ``PerModel`` depend on the pre-merge grain.
    """
    if normalisation_type == ImportanceNormalisationMethod.NoNormalisation:
        return cache_df

    feature_cols = [c for c in cache_df.columns if c not in _META_COLS]
    if not feature_cols:
        return cache_df

    out = cache_df.copy()
    vals = out[feature_cols].to_numpy(dtype=float)

    if normalisation_type == ImportanceNormalisationMethod.Local:
        row_max = np.nanmax(np.abs(vals), axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0
        out[feature_cols] = vals / row_max

    elif normalisation_type == ImportanceNormalisationMethod.PerModel:
        for _, idx in out.groupby(["model_type", "iteration"]).groups.items():
            block = out.loc[idx, feature_cols].to_numpy(dtype=float)
            max_abs = np.nanmax(np.abs(block)) if block.size else 0.0
            divisor = float(max_abs) if max_abs > 0 else 1.0
            out.loc[idx, feature_cols] = block / divisor

    else:  # Global
        max_abs = np.nanmax(np.abs(vals)) if vals.size else 0.0
        divisor = float(max_abs) if max_abs > 0 else 1.0
        out[feature_cols] = vals / divisor

    return out


def _aligned_feature_matrix(
    sample_index: pd.Index,
    feature_cols: list[str],
    descriptor_df: pd.DataFrame | None,
    target_col: str | None,
) -> np.ndarray | None:
    """
    Build a raw feature-value matrix aligned to the SHAP matrix for colouring.

    Returns an array of shape ``(len(sample_index), len(feature_cols))`` whose
    rows match ``sample_index`` (sample IDs) and whose columns match
    ``feature_cols``. Features absent from ``descriptor_df`` (or missing samples)
    are filled with ``NaN``, which SHAP renders as grey points.

    Returns ``None`` when ``descriptor_df`` is not supplied.
    """
    if descriptor_df is None:
        return None

    desc = descriptor_df.copy()
    desc.index = desc.index.astype(str)
    desc = desc.drop(columns=[target_col], errors="ignore")

    # reindex to the exact rows/cols we need; missing entries become NaN
    aligned = desc.reindex(index=sample_index, columns=feature_cols)
    return aligned.to_numpy(dtype=float)


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

    merged = subset.groupby("sample_id")[feature_cols].mean().reset_index().set_index("sample_id")
    return merged


# ---------------------------------------------------------------------------
# Local SHAP plotting (native shap package)
# ---------------------------------------------------------------------------


def _lighten(color: str, amount: float = 0.6) -> tuple[float, float, float]:
    """Blend a colour towards white by ``amount`` (0 = unchanged, 1 = white)."""
    import matplotlib.colors as mcolors

    r, g, b = mcolors.to_rgb(color)
    return (r + (1.0 - r) * amount, g + (1.0 - g) * amount, b + (1.0 - b) * amount)


@contextmanager
def _shap_local_colours(neg_color: str, pos_color: str):
    """
    Temporarily set shap's style colours so waterfall / bar honour user colours.

    shap >= 0.46 styles its plots via ``shap.plots._style``. Waterfall and bar
    have no per-call colour argument, but the style colours can be overridden in
    a context. Falls back to a no-op on shap builds without the style system.
    """
    try:
        from shap.plots import _style as shap_style
    except Exception:
        yield
        return

    with shap_style.style_context(
        primary_color_positive=pos_color,
        primary_color_negative=neg_color,
        secondary_color_positive=_lighten(pos_color),
        secondary_color_negative=_lighten(neg_color),
    ):
        yield


def _recolour_force_figure(fig: plt.Figure, neg_color: str, pos_color: str) -> None:
    """
    Recolour a shap matplotlib force plot to the user's colours.

    ``shap.plots.force(..., matplotlib=True)`` hardcodes its colours and ignores
    ``plot_cmap``, so the only way to honour user colours is to remap the known
    constants after drawing. Unmatched colours are left untouched, so this is a
    safe best-effort that degrades gracefully if shap changes its palette.
    """
    import matplotlib.colors as mcolors

    # shap force constants (positive primary/secondary, negative primary/secondary).
    remap = {
        (1.0, 0.051, 0.341): mcolors.to_rgb(pos_color),  # #FF0D57
        (1.0, 0.765, 0.835): _lighten(pos_color),  # #FFC3D5
        (0.118, 0.533, 0.898): mcolors.to_rgb(neg_color),  # #1E88E5
        (0.82, 0.902, 0.98): _lighten(neg_color),  # #D1E6FA
    }

    def _mapped(color) -> tuple | None:
        try:
            key = tuple(round(c, 3) for c in mcolors.to_rgb(color))
        except (ValueError, TypeError):
            return None
        return remap.get(key)

    for ax in fig.axes:
        for patch in ax.patches:
            new_fc = _mapped(patch.get_facecolor())
            if new_fc is not None:
                patch.set_facecolor(new_fc)
            new_ec = _mapped(patch.get_edgecolor())
            if new_ec is not None:
                patch.set_edgecolor(new_ec)
        for line in ax.get_lines():
            new_c = _mapped(line.get_color())
            if new_c is not None:
                line.set_color(new_c)
        for text in ax.texts:
            new_c = _mapped(text.get_color())
            if new_c is not None:
                text.set_color(new_c)


def plot_local_shap(
    values: np.ndarray,
    feature_names: list[str],
    base_value: float = 0.0,
    data: np.ndarray | None = None,
    plot_type: str = "waterfall",
    max_display: int = 15,
    neg_color: str = "#40bcde",
    pos_color: str = "#e64747",
) -> plt.Figure:
    """
    Render a single-instance SHAP explanation with the native ``shap`` package.

    Parameters
    ----------
    values:
        1-D array of SHAP values for one sample (ensemble-merged, possibly
        normalised), one entry per feature in ``feature_names``.
    feature_names:
        Feature names, parallel to ``values``.
    base_value:
        Anchor for the waterfall / force plot (``E[f(x)]``). The cache stores
        only SHAP values, so this defaults to 0.0; with normalisation enabled
        the additive reconstruction is not meaningful anyway.
    data:
        Optional raw feature values for the instance, used to label each row as
        ``value = feature``. ``None`` shows feature names only.
    plot_type:
        ``"waterfall"`` (default), ``"force"``, or ``"bar"`` — native
        ``shap.plots`` styles.
    max_display:
        Maximum number of features (by ``|SHAP|``) to show. Waterfall and bar
        use shap's own ``max_display``; the force plot has none, so it is
        truncated to the top ``max_display`` features here.
    neg_color, pos_color:
        Hex colours for negative / positive contributions. Waterfall and bar
        honour them exactly via shap's style context; the force plot is recoloured
        post-hoc since shap hardcodes its force palette.

    Returns
    -------
    plt.Figure
    """
    import shap

    if values is None or len(values) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No SHAP values available", ha="center", va="center")
        ax.axis("off")
        return fig

    explanation = shap.Explanation(
        values=np.asarray(values, dtype=float),
        base_values=float(base_value),
        data=(np.asarray(data, dtype=float) if data is not None else None),
        feature_names=list(feature_names),
    )

    plt.figure()
    if plot_type == "force":
        # shap's force plot has no max_display and shows every feature, so
        # truncate to the top-N by |SHAP| ourselves before drawing.
        force_expl = explanation
        vals = np.asarray(explanation.values, dtype=float)
        if max_display is not None and len(vals) > max_display:
            top = np.sort(np.argsort(np.abs(vals))[::-1][:max_display])
            force_expl = shap.Explanation(
                values=vals[top],
                base_values=explanation.base_values,
                data=(explanation.data[top] if explanation.data is not None else None),
                feature_names=[explanation.feature_names[i] for i in top],
            )
        shap.plots.force(force_expl, matplotlib=True, show=False)
        fig = plt.gcf()
        _recolour_force_figure(fig, neg_color, pos_color)
    else:
        with _shap_local_colours(neg_color, pos_color):
            if plot_type == "bar":
                shap.plots.bar(explanation, max_display=max_display, show=False)
            else:  # waterfall
                shap.plots.waterfall(explanation, max_display=max_display, show=False)
        fig = plt.gcf()

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Global SHAP plotting (native shap package)
# ---------------------------------------------------------------------------


def plot_global_shap(
    shap_matrix: np.ndarray,
    feature_matrix: np.ndarray | None,
    feature_names: list[str],
    plot_type: ShapGlobalPlotType = ShapGlobalPlotType.Beeswarm,
    max_display: int | None = 10,
    neg_color: str = "#40bcde",
    pos_color: str = "#e64747",
) -> plt.Figure:
    """
    Render a global SHAP summary using the native ``shap`` package.

    Parameters
    ----------
    shap_matrix:
        Ensemble-averaged SHAP values, shape ``(n_samples, n_features)``.
    feature_matrix:
        Raw feature values aligned to ``shap_matrix`` (same shape), used to
        colour beeswarm / violin points. ``None`` falls back to no colouring.
    feature_names:
        Column names, length ``n_features``.
    plot_type:
        ``Beeswarm`` (dot), ``Bar`` (mean |SHAP|), or ``Violin``.
    max_display:
        Maximum number of features to display, ranked by mean |SHAP|.
        ``None`` shows all features.
    neg_color, pos_color:
        Hex colours for the low→high feature-value colourmap (beeswarm/violin)
        and the bar colour.

    Returns
    -------
    plt.Figure
    """
    import shap

    n_features = len(feature_names)
    display = n_features if max_display is None else int(min(max_display, n_features))

    # Fresh figure so summary_plot (which draws on the current axes) doesn't
    # overwrite an unrelated figure.
    fig = plt.figure()

    if plot_type == ShapGlobalPlotType.Bar:
        # Bar summary needs no feature values — it's mean |SHAP| per feature.
        shap.summary_plot(
            shap_matrix,
            features=feature_matrix,
            feature_names=feature_names,
            plot_type="bar",
            max_display=display,
            color=pos_color,
            show=False,
        )
    else:
        shap_plot_type = "violin" if plot_type == ShapGlobalPlotType.Violin else "dot"
        shap.summary_plot(
            shap_matrix,
            features=feature_matrix,
            feature_names=feature_names,
            plot_type=shap_plot_type,
            max_display=display,
            cmap=get_cmap(neg_color, pos_color),
            color_bar=feature_matrix is not None,
            show=False,
        )

    fig = plt.gcf()
    fig.tight_layout()
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
    plot_type: ShapGlobalPlotType = ShapGlobalPlotType.Beeswarm,
    cache_root: Path | None = None,
    target_col: str | None = None,
) -> dict[str, GlobalAttributionResult]:
    """
    Compute global SHAP feature attribution across the selected population.

    One :class:`~polynet.explainability.explain.GlobalAttributionResult` is
    returned per descriptor (e.g. one for Morgan, one for RDKit). Plots are
    rendered with the native ``shap`` package (beeswarm / bar / violin).

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
        Maximum number of features to display, ranked by mean |SHAP|.
        ``None`` shows all features.
    plot_type:
        ``Beeswarm``, ``Bar``, or ``Violin`` (native ``shap`` summary styles).

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
        cache_root=cache_root,
        target_col=target_col,
    )

    # Normalise descriptor_dfs keys to strings for lookup
    str_dfs: dict[str, pd.DataFrame] = {str(k): v for k, v in descriptor_dfs.items()}

    results: dict[str, GlobalAttributionResult] = {}

    for descriptor, cache_df in cache_by_descriptor.items():
        model_log_names = [k for k in models if _parse_model_log_name(k)[1] == descriptor]
        n_models = len({_parse_model_log_name(k)[0] for k in model_log_names})

        # Normalise per trained instance (representation × model × bootstrap)
        # FIRST, then average across the ensemble → one row per sample. Doing the
        # normalisation before the merge is what makes PerModel/Local meaningful
        # (see _normalise_cache_attributions).
        normalised_cache = _normalise_cache_attributions(cache_df, normalisation_type)
        merged_df = merge_shap_attributions(
            cache_df=normalised_cache,
            model_log_names=model_log_names,
            sample_ids=explain_sample_ids,
        )

        if merged_df.empty:
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

        feature_names = list(merged_df.columns)
        # Already normalised at the per-instance grain before merging.
        shap_matrix = merged_df.to_numpy(dtype=float)
        feature_matrix = _aligned_feature_matrix(
            sample_index=merged_df.index,
            feature_cols=feature_names,
            descriptor_df=str_dfs.get(descriptor),
            target_col=target_col,
        )

        n_samples = merged_df.shape[0]
        n_frags_total = len(feature_names)
        n_shown = n_frags_total if top_n is None else min(top_n, n_frags_total)

        fig = plot_global_shap(
            shap_matrix=shap_matrix,
            feature_matrix=feature_matrix,
            feature_names=feature_names,
            plot_type=plot_type,
            max_display=top_n,
            neg_color=neg_color,
            pos_color=pos_color,
        )

        # attribution_dict kept for back-compat: {feature: [per-sample SHAP]}.
        attribution_dict = {
            feat: shap_matrix[:, j].tolist() for j, feat in enumerate(feature_names)
        }

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


def _separate_local_results(
    cache_df: pd.DataFrame,
    descriptor: str,
    model_log_names: list[str],
    explain_sample_ids: list[str],
    normalisation_type: ImportanceNormalisationMethod,
    problem_type: ProblemType,
    target_class: int | None,
    descriptor_df: pd.DataFrame | None,
    target_col: str | None,
    local_plot_type: str,
    max_display: int,
    neg_color: str,
    pos_color: str,
) -> list[InstanceAttributionResult]:
    """
    Build one :class:`InstanceAttributionResult` per model × sample (Separate).

    Mirrors the per-sample normalisation of the Average path (``Local`` → each
    row by its own max-abs, ``Global`` → a single divisor over the displayed
    rows, otherwise raw), but keeps each model's explanation distinct and tags it
    with ``model_label``. Results are ordered by sample, then by model.
    """
    feature_cols = [c for c in cache_df.columns if c not in _META_COLS]
    results: list[InstanceAttributionResult] = []

    if not feature_cols or cache_df.empty:
        return results

    sample_ids_set = {str(s) for s in explain_sample_ids}

    # Global divisor over all displayed (model × sample) rows for this descriptor.
    global_divisor = 1.0
    if normalisation_type == ImportanceNormalisationMethod.Global:
        sel = cache_df[cache_df["sample_id"].isin(sample_ids_set)]
        if not sel.empty:
            max_abs = np.nanmax(np.abs(sel[feature_cols].to_numpy(dtype=float)))
            global_divisor = float(max_abs) if max_abs > 0 else 1.0

    cls_str = "" if problem_type == ProblemType.Regression else f", class {target_class}"

    for sample_id in explain_sample_ids:
        sid = str(sample_id)
        for key in sorted(model_log_names):
            model_type, _, iteration = _parse_model_log_name(key)
            model_label = f"{prettify_label(f'{model_type}-{descriptor}')} — bootstrap {iteration}"

            mask = (
                (cache_df["model_type"] == model_type)
                & (cache_df["iteration"] == str(iteration))
                & (cache_df["sample_id"] == sid)
            )
            row = cache_df.loc[mask]
            if row.empty:
                results.append(
                    InstanceAttributionResult(
                        sample_idx=sid,
                        info_msg="",
                        true_label="N/A",
                        predicted_label="N/A",
                        attribution_df=pd.DataFrame(),
                        figure=None,
                        warning=f"No SHAP values found for sample '{sid}' / {model_label}.",
                        model_label=model_label,
                    )
                )
                continue

            raw_vals = row.iloc[0][feature_cols].to_numpy(dtype=float)

            if normalisation_type == ImportanceNormalisationMethod.Local:
                max_abs = np.nanmax(np.abs(raw_vals))
                divisor = float(max_abs) if max_abs > 0 else 1.0
            elif normalisation_type == ImportanceNormalisationMethod.Global:
                divisor = global_divisor
            else:
                divisor = 1.0

            normed_vals = raw_vals / divisor if divisor != 1.0 else raw_vals.copy()

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

            feature_matrix = _aligned_feature_matrix(
                sample_index=pd.Index([sid]),
                feature_cols=feature_cols,
                descriptor_df=descriptor_df,
                target_col=target_col,
            )
            instance_data = feature_matrix[0] if feature_matrix is not None else None
            fig = plot_local_shap(
                values=normed_vals,
                feature_names=feature_cols,
                base_value=0.0,
                data=instance_data,
                plot_type=local_plot_type,
                max_display=max_display,
                neg_color=neg_color,
                pos_color=pos_color,
            )
            plt.close("all")

            info_msg = (
                f"SHAP attribution for `{sid}` | {descriptor}"
                f"{cls_str} | normalisation: `{normalisation_type}`"
            )
            results.append(
                InstanceAttributionResult(
                    sample_idx=sid,
                    info_msg=info_msg,
                    true_label="N/A",
                    predicted_label="N/A",
                    attribution_df=attr_df,
                    figure=fig,
                    model_label=model_label,
                )
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
    max_display: int = 15,
    predictions: dict | None = None,
    cache_root: Path | None = None,
    target_col: str | None = None,
    aggregation: ExplanationAggregation = ExplanationAggregation.Average,
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
        Hex colours for negative / positive contributions, applied to the native
        ``shap`` local plots (see :func:`plot_local_shap`).
    normalisation_type:
        Normalisation strategy applied to SHAP values before plotting.
    target_class:
        Class index for classification.
    local_plot_type:
        ``"waterfall"``, ``"force"``, or ``"bar"`` — native ``shap.plots`` styles.
    max_display:
        Maximum number of features (by ``|SHAP|``) shown in the waterfall and
        bar plots. The force plot displays all features and ignores this.
    predictions:
        Optional ``{sample_id: {"true": ..., "predicted": ...}}`` for label
        display in the result.
    aggregation:
        ``Average`` (default) merges the selected models into a single
        explanation per sample. ``Separate`` keeps one explanation per
        model × sample, each tagged with ``model_label``.

    Returns
    -------
    dict[str, list[InstanceAttributionResult]]
        Keyed by descriptor name. For ``Average`` there is one result per
        sample; for ``Separate`` there is one result per model × sample.
    """
    cache_by_descriptor = compute_and_cache_shap(
        models=models,
        descriptor_dfs=descriptor_dfs,
        experiment_path=experiment_path,
        problem_type=problem_type,
        explain_sample_ids=explain_sample_ids,
        target_class=target_class,
        cache_root=cache_root,
        target_col=target_col,
    )

    # Raw descriptor values (per instance) are used to label native SHAP plots.
    str_dfs: dict[str, pd.DataFrame] = {str(k): v for k, v in descriptor_dfs.items()}

    results: dict[str, list[InstanceAttributionResult]] = {}

    for descriptor, cache_df in cache_by_descriptor.items():
        model_log_names = [k for k in models if _parse_model_log_name(k)[1] == descriptor]
        descriptor_df = str_dfs.get(descriptor)

        # Separate aggregation: one explanation per model × sample (no merge).
        if aggregation == ExplanationAggregation.Separate:
            results[descriptor] = _separate_local_results(
                cache_df=cache_df,
                descriptor=descriptor,
                model_log_names=model_log_names,
                explain_sample_ids=explain_sample_ids,
                normalisation_type=normalisation_type,
                problem_type=problem_type,
                target_class=target_class,
                descriptor_df=descriptor_df,
                target_col=target_col,
                local_plot_type=local_plot_type,
                max_display=max_display,
                neg_color=neg_color,
                pos_color=pos_color,
            )
            continue

        # ----- Average aggregation (default, historical behaviour) -----
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

            # Figure — native shap local plot. Raw feature values (when available)
            # label each row; SHAP magnitudes are the merged/normalised values.
            feature_matrix = _aligned_feature_matrix(
                sample_index=pd.Index([sid]),
                feature_cols=feature_cols,
                descriptor_df=descriptor_df,
                target_col=target_col,
            )
            instance_data = feature_matrix[0] if feature_matrix is not None else None
            fig = plot_local_shap(
                values=normed_vals,
                feature_names=feature_cols,
                base_value=0.0,
                data=instance_data,
                plot_type=local_plot_type,
                max_display=max_display,
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
