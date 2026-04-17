"""
polynet.explainability.attributions
=====================================
Cache utilities for GNN explainability.

Public API
----------
::

    from polynet.explainability.attributions import deep_update
"""

from __future__ import annotations


def deep_update(original: dict, updates: dict) -> dict:
    """
    Recursively merge ``updates`` into ``original`` in-place.

    Parameters
    ----------
    original:
        Base dictionary to update.
    updates:
        Dictionary whose values override or extend ``original``.

    Returns
    -------
    dict
        The updated ``original`` dict (modified in-place).
    """
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(original.get(key), dict):
            deep_update(original[key], value)
        else:
            original[key] = value
    return original
