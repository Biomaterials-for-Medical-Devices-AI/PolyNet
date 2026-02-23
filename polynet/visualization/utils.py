"""
polynet.visualization.utils
============================
Utilities for saving matplotlib figures to disk.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def save_plot(fig: plt.Figure, path: str | Path, dpi: int = 300) -> None:
    """
    Save a matplotlib figure to disk and close it.

    Parameters
    ----------
    fig:
        The figure to save.
    path:
        Destination file path. The parent directory must already exist.
    dpi:
        Resolution in dots per inch.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    logger.info(f"Plot saved to {path}")
