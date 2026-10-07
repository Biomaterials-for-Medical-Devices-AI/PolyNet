"""
tests/test_ridge_plot.py
========================
The global ridge plot scales each fragment's row to its own density, so a
fragment whose scores are nearly all identical (a very tall, narrow KDE) does
not flatten every other row into an invisible line.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

from polynet.explainability.visualization import plot_attribution_distribution  # noqa: E402


def test_each_row_is_scaled_to_its_own_density():
    rng = np.random.default_rng(0)
    scores = {
        "narrow": list(rng.normal(0, 1e-5, 500)),  # e.g. masking changes nothing
        "wide": list(rng.normal(0.3, 0.2, 500)),
    }
    fig = plot_attribution_distribution(scores)
    for ax in fig.axes:
        peak = max(np.max(line.get_ydata()) for line in ax.get_lines() if len(line.get_ydata()))
        # The row's own curve reaches most of its axis height (visible).
        assert peak > 0.5 * ax.get_ylim()[1]
