"""
polynet.experiment
==================
File-system operations for PolyNet experiment lifecycle management.

Use ``polynet.experiment.manager`` to create and list experiments.
The Pydantic schema for experiment configuration lives separately in
``polynet.config.experiment``.
"""

from polynet.experiment.manager import create_experiment, get_experiments

__all__ = ["get_experiments", "create_experiment"]
