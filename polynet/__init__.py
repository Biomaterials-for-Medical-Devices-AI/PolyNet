"""PolyNet: polymer property prediction with GNNs and traditional ML."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("polynet")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "unknown"
