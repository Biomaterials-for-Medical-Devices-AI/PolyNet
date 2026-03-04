from __future__ import annotations

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
        # Keep FeatureTransformConfig if you want a single place to validate defaults,
        # but selectors are now per-step param dicts.
        self.config = FeatureTransformConfig(
            scaler=scaler, selectors=selectors or {}, random_state=random_state
        )
        self.problem_type = problem_type

        # learned attributes
        self.scaler_ = None
        self.selector_steps_: list[tuple[FeatureSelection, object]] = []
        self.selected_mask_: Optional[np.ndarray] = None
        self.selected_features_: Optional[list[str]] = None
        self.feature_names_in_: Optional[list[str]] = None
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
        self.feature_names_in_ = list(Xdf.columns)

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

        Xdf = Xdf.loc[:, self.feature_names_in_]

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
