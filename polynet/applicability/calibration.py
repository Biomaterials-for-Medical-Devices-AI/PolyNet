"""
polynet.applicability.calibration
=================================
Expected error as a function of the distance to the training set (Sheridan
et al., J. Chem. Inf. Comput. Sci. 2004, 44, 1912–1928).

The held-out test polymers of every split are placed relative to that split's
training polymers, and their errors are grouped into bins of domain score
(quantiles of the test scores; the score is the distance divided by the
domain cutoff). A new polymer's expected error is the
mean test error of its bin: the absolute error for regression, the accuracy
(share of correct predictions) for classification.

::

    from polynet.applicability.calibration import DistanceErrorCalibration, prediction_errors
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from polynet.config.enums import ProblemType


class DistanceErrorCalibration:
    """
    Binned relation between the domain score and the test error.

    Parameters
    ----------
    n_bins:
        Number of score bins (quantiles). Tied scores can give fewer bins.

    Attributes
    ----------
    edges_:
        Bin edges, increasing; the outer edges are the lowest and highest
        test score.
    values_:
        Mean test error per bin.
    counts_:
        Test predictions per bin.
    """

    def __init__(self, n_bins: int = 5) -> None:
        self.n_bins = n_bins

    def fit(self, scores: np.ndarray, errors: np.ndarray) -> "DistanceErrorCalibration":
        """
        Bin the test errors by domain score.

        Raises
        ------
        ValueError
            Without test predictions, or with mismatched lengths.
        """
        scores = np.asarray(scores, dtype=float)
        errors = np.asarray(errors, dtype=float)
        if len(scores) == 0 or len(scores) != len(errors):
            raise ValueError(
                "Calibration needs one error per test score, got "
                f"{len(scores)} scores and {len(errors)} errors."
            )
        self.edges_ = np.unique(np.quantile(scores, np.linspace(0, 1, self.n_bins + 1)))
        bins = self._bin(scores)
        n = max(1, len(self.edges_) - 1)
        self.counts_ = np.bincount(bins, minlength=n)
        sums = np.bincount(bins, weights=errors, minlength=n)
        self.values_ = np.divide(sums, self.counts_, out=np.full(n, np.nan), where=self.counts_ > 0)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """
        Expected error (mean test error of the bin) at each domain score.

        Polymers farther from the training set than every test polymer get
        ``NaN``: no test error was observed that far out, so none is
        extrapolated. Polymers closer than every test polymer get the error
        of the closest bin.
        """
        scores = np.asarray(scores, dtype=float)
        expected = self.values_[self._bin(scores)]
        return np.where(scores > self.edges_[-1], np.nan, expected)

    def _bin(self, scores: np.ndarray) -> np.ndarray:
        # Inner edges only, so values below/above the test range go to the outer bins.
        return np.digitize(scores, self.edges_[1:-1], right=True)

    def summary(self) -> dict:
        """Fitted bins, for the saved report."""
        return {
            "bin_edges": self.edges_.tolist(),
            "expected_error": [None if np.isnan(v) else float(v) for v in self.values_],
            "n_test_predictions": self.counts_.tolist(),
        }


def prediction_errors(
    y_true: pd.Series, y_pred: pd.Series, problem_type: ProblemType | str
) -> pd.Series:
    """
    Per-sample error used for the calibration.

    Returns the absolute error for regression and ``1.0`` / ``0.0`` for a
    correct / wrong class for classification (its mean is the accuracy).
    """
    if ProblemType(problem_type) == ProblemType.Classification:
        return (y_true == y_pred).astype(float)
    return (y_true - y_pred).abs()
