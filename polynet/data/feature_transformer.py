from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import (
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from polynet.config.enums import FeatureSelection, ProblemType, TransformDescriptor
from polynet.config.schemas.feature_preprocessing import FeatureTransformConfig

logger = logging.getLogger(__name__)


class FeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Feature normalization + multiple feature selection steps (no LASSO).

    Scaling is fit once on training features and reused for transform.
    Feature selection steps are applied sequentially in the order provided
    by the `selectors` dict.

    Parameters
    ----------
    scaler:
        Feature scaling strategy (TransformDescriptor).
    selectors:
        Ordered mapping of FeatureSelection -> parameters dict. Each step
        is applied sequentially to the (optionally) scaled features.

        Expected selector params:
          - FeatureSelection.Variance: {"threshold": float}
          - FeatureSelection.Correlation: {"threshold": float}  (0 < thr < 1)

        Example
        -------
        selectors = {
            FeatureSelection.Variance: {"threshold": 0.0},
            FeatureSelection.Correlation: {"threshold": 0.95},
        }
    """

    def __init__(
        self,
        scaler: TransformDescriptor = TransformDescriptor.NoTransformation,
        selectors: dict[FeatureSelection, dict] | None = None,
        *,
        random_state: int = 42,
        problem_type: ProblemType = ProblemType.Classification,
    ):
        self.scaler = scaler
        self.selectors = selectors or {}
        self.random_state = random_state
        self.problem_type = problem_type

        # You can still build a validated config from them
        self.config = FeatureTransformConfig(
            scaler=self.scaler, selectors=self.selectors, random_state=self.random_state
        )

        # learned attributes
        self.scaler_ = None
        self.selector_steps_: list[tuple[FeatureSelection, object]] = []
        self.selected_mask_: Optional[np.ndarray] = None
        self.selected_features_: Optional[list[str]] = None
        self.feature_names_in_: Optional[list[str]] = None
        # Per-column training means (raw space, post-sanitisation) used to
        # impute NaN / ±inf values that appear in inference data for columns
        # that were clean during training.
        self.fill_values_: Optional[dict[str, float]] = None
        self._is_fit = False

    # -----------------------------
    # Helpers
    # -----------------------------

    def _as_frame(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])

    def _make_scaler(self):
        s = self.config.scaler
        if s == TransformDescriptor.NoTransformation:
            return None
        if s == TransformDescriptor.StandardScaler:
            return StandardScaler()
        if s == TransformDescriptor.MinMaxScaler:
            return MinMaxScaler()
        if s == TransformDescriptor.RobustScaler:
            return RobustScaler()
        if s == TransformDescriptor.PowerTransformer:
            return PowerTransformer()
        if s == TransformDescriptor.QuantileTransformer:
            return QuantileTransformer()
        if s == TransformDescriptor.Normalizer:
            return Normalizer()
        raise ValueError(f"Unknown scaler: {s!r}")

    @staticmethod
    def _drop_problematic_columns(Xdf: pd.DataFrame) -> pd.DataFrame:
        """
        Identify and drop columns that contain NaN or ±infinity values.

        Called during ``fit()`` only. These values cannot be handled by any
        sklearn scaler and would cause a ``ValueError`` at fit time. The
        expected input is a fully numeric DataFrame where every column
        represents a molecular descriptor.

        Expected behaviour
        ------------------
        - Columns with at least one ``NaN``, ``+inf``, or ``-inf`` value are
          dropped unconditionally and a ``WARNING`` is emitted listing every
          affected column name.
        - If *all* columns are problematic the resulting DataFrame will be
          empty; downstream code will then raise a ``ValueError`` with a
          descriptive message rather than a cryptic sklearn traceback.
        - Columns that are entirely finite are returned unchanged.

        Parameters
        ----------
        Xdf:
            Numeric feature DataFrame as produced by ``_as_frame()``.

        Returns
        -------
        pd.DataFrame
            Copy of ``Xdf`` with problematic columns removed.
        """
        bad_cols = [
            col
            for col in Xdf.columns
            if Xdf[col].isnull().any()
            or np.isinf(Xdf[col].to_numpy(dtype=float, na_value=np.nan)).any()
        ]

        if bad_cols:
            logger.warning(
                "%d feature column(s) contain NaN or ±infinity and will be dropped before "
                "fitting: %s",
                len(bad_cols),
                bad_cols,
            )
            Xdf = Xdf.drop(columns=bad_cols)

        return Xdf

    def _impute_for_transform(self, Xdf: pd.DataFrame) -> pd.DataFrame:
        """
        Impute NaN and ±infinity values in inference data with training column means.

        Called during ``transform()`` after the input has been aligned to
        ``feature_names_in_``. A descriptor may be well-defined for all
        training molecules yet undefined (NaN) or numerically unstable (±inf)
        for molecules in an external prediction or SHAP explanation set — for
        example, Gasteiger-charge-based RDKit descriptors such as
        ``MaxPartialCharge`` and ``BCUT2D_*`` return ``NaN`` for atoms RDKit
        cannot parametrise (PSMILES dummy atoms, unusual elements). Filling
        with the training mean keeps the prediction path alive and is
        equivalent to predicting "average behaviour" for the missing feature.

        Expected behaviour
        ------------------
        - Any column containing at least one ``NaN``, ``+inf``, or ``-inf``
          is replaced with the mean of that column computed on the **training**
          data (stored in ``fill_values_`` during ``fit()``).
        - A ``WARNING`` is emitted listing every affected column so the user
          can investigate the descriptor computation for the affected molecules.
        - Columns with entirely finite values are returned unchanged.

        Parameters
        ----------
        Xdf:
            Numeric feature DataFrame already aligned to ``feature_names_in_``.

        Returns
        -------
        pd.DataFrame
            Copy of ``Xdf`` with NaN / ±inf replaced by training column means.
        """
        numeric_arr = Xdf.to_numpy(dtype=float, na_value=np.nan)
        bad_mask = np.isnan(numeric_arr) | np.isinf(numeric_arr)

        if not bad_mask.any():
            return Xdf

        Xdf = Xdf.copy()
        bad_col_names = []

        for j, col in enumerate(Xdf.columns):
            if bad_mask[:, j].any():
                bad_col_names.append(col)
                fill = self.fill_values_.get(col, 0.0) if self.fill_values_ else 0.0
                Xdf[col] = Xdf[col].replace([np.inf, -np.inf], np.nan).fillna(fill)

        logger.warning(
            "%d feature column(s) contain NaN or ±infinity in the inference data and "
            "were imputed with the training mean: %s",
            len(bad_col_names),
            bad_col_names,
        )

        return Xdf

    @staticmethod
    def _get_threshold(params: dict, *, default: float, name: str = "threshold") -> float:
        """Read a float threshold from params with a default."""
        val = params.get(name, default)
        try:
            return float(val)
        except Exception as e:
            raise ValueError(f"{name} must be a float, got {val!r}.") from e

    @staticmethod
    def _fit_corr_mask(X: pd.DataFrame, thr: float) -> np.ndarray:
        if not (0.0 < thr < 1.0):
            raise ValueError(f"Correlation threshold must be in (0, 1), got {thr}.")

        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if (upper[col] >= thr).any()]
        keep_mask = ~X.columns.isin(to_drop)
        return np.asarray(keep_mask, dtype=bool)

    # -----------------------------
    # API
    # -----------------------------

    def fit(self, X: pd.DataFrame | np.ndarray, y=None):  # y unused (kept for sklearn API)
        Xdf = self._as_frame(X)

        # Drop columns containing NaN / ±inf before any scaler or selector sees the data.
        # feature_names_in_ is set *after* sanitisation so that transform() only ever
        # looks for the columns that were actually used during fitting.
        Xdf = self._drop_problematic_columns(Xdf)
        if Xdf.empty or Xdf.shape[1] == 0:
            raise ValueError(
                "All feature columns were dropped due to NaN or ±infinity values. "
                "Check your descriptor computation for the affected representation."
            )

        self.feature_names_in_ = list(Xdf.columns)

        # Store per-column training means so transform() can impute NaN / ±inf
        # values that appear in inference data for columns that were clean here.
        self.fill_values_ = Xdf.mean().to_dict()

        # 1) fit scaler
        self.scaler_ = self._make_scaler()
        if self.scaler_ is None:
            X_scaled = Xdf.to_numpy()
        else:
            self.scaler_.fit(Xdf.to_numpy())
            X_scaled = self.scaler_.transform(Xdf.to_numpy())

        # mask is maintained in ORIGINAL feature space
        current_mask = np.ones(X_scaled.shape[1], dtype=bool)

        self.selector_steps_.clear()

        # 2) sequential selectors in dict order
        for step, params in (self.config.selectors or {}).items():
            X_step = X_scaled[:, current_mask]

            if step == FeatureSelection.Variance:
                thr = self._get_threshold(params, default=0.0)
                vt = VarianceThreshold(threshold=thr)
                vt.fit(X_step)
                step_mask = vt.get_support()
                self.selector_steps_.append((FeatureSelection.Variance, vt))

            elif step == FeatureSelection.Correlation:
                thr = self._get_threshold(params, default=0.95)
                current_cols = [c for c, keep in zip(self.feature_names_in_, current_mask) if keep]
                X_step_df = pd.DataFrame(X_step, columns=current_cols)
                step_mask = self._fit_corr_mask(X_step_df, thr)
                self.selector_steps_.append((FeatureSelection.Correlation, {"threshold": thr}))

            else:
                raise ValueError(
                    f"Unsupported selector step: {step!r}. "
                    f"Supported: {FeatureSelection.Variance}, {FeatureSelection.Correlation}."
                )

            # Compose step_mask back into original-space mask
            new_mask = np.zeros_like(current_mask)
            kept_positions = np.flatnonzero(current_mask)
            new_mask[kept_positions[step_mask]] = True
            current_mask = new_mask

        self.selected_mask_ = current_mask
        self.selected_features_ = [
            c for c, keep in zip(self.feature_names_in_, current_mask) if keep
        ]
        self._is_fit = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("FeatureTransformer must be fit() before calling transform().")

        if self.feature_names_in_ is None or self.selected_mask_ is None:
            raise RuntimeError("FeatureTransformer is in an invalid state (missing fitted attrs).")

        Xdf = self._as_frame(X)

        missing = set(self.feature_names_in_) - set(Xdf.columns)
        if missing:
            raise ValueError(f"Input is missing required columns: {sorted(missing)}")

        # Select only the columns that were present and clean during fit().
        # Columns that were dropped at fit time (NaN/±inf in training data) are
        # silently ignored here — extra columns in X are simply not selected.
        Xdf = Xdf.loc[:, self.feature_names_in_]

        # Impute any NaN / ±inf that appear in this data but were not present
        # during training (e.g. a descriptor undefined for a new molecule).
        # Uses per-column training means stored in fill_values_ during fit().
        Xdf = self._impute_for_transform(Xdf)

        if self.scaler_ is None:
            X_scaled = Xdf.to_numpy()
        else:
            X_scaled = self.scaler_.transform(Xdf.to_numpy())

        return X_scaled[:, self.selected_mask_]

    def fit_transform(self, X: pd.DataFrame | np.ndarray, y=None) -> np.ndarray:
        return self.fit(X, y=y).transform(X)

    def get_feature_names_out(self) -> list[str]:
        if self.selected_features_ is None:
            raise RuntimeError("FeatureTransformer must be fit() before get_feature_names_out().")
        return list(self.selected_features_)
