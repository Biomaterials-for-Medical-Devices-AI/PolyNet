"""
polynet.factories.loss
=======================
Factory function for constructing PyTorch loss functions.

Public API
----------
::

    from polynet.factories.loss import create_loss
    from polynet.config.enums import ProblemType

    loss_fn = create_loss(ProblemType.Regression)

    # For classification with class weighting:
    loss_fn = create_loss(
        ProblemType.Classification,
        class_weights=torch.tensor([0.3, 0.7]),
    )
"""

from __future__ import annotations

import torch
import torch.nn as nn

from polynet.config.enums import ProblemType


def create_loss(
    problem_type: ProblemType | str, class_weights: torch.Tensor | None = None
) -> nn.Module:
    """
    Construct and return a PyTorch loss function for the given task type.

    Parameters
    ----------
    problem_type:
        The supervised task type. Accepts a ``ProblemType`` enum member
        or its string value (e.g. ``"regression"``).
    class_weights:
        Optional class weight tensor for ``CrossEntropyLoss``. Used to
        correct for class imbalance in classification tasks. If provided,
        must have shape ``(num_classes,)``.

        Compute weights using ``polynet.training.metrics.compute_class_weights``
        before passing here.

        Ignored for regression tasks.

    Returns
    -------
    nn.Module
        An instantiated loss function:
        - Classification → ``nn.CrossEntropyLoss``
        - Regression → ``nn.MSELoss``

    Raises
    ------
    ValueError
        If ``problem_type`` is not recognised.

    Examples
    --------
    >>> from polynet.factories.loss import create_loss
    >>> from polynet.config.enums import ProblemType
    >>> loss_fn = create_loss(ProblemType.Regression)
    >>> loss_fn = create_loss(ProblemType.Classification)
    """
    problem_type = ProblemType(problem_type) if isinstance(problem_type, str) else problem_type

    _LOSS_REGISTRY: dict[ProblemType, nn.Module] = {
        ProblemType.Classification: nn.CrossEntropyLoss(weight=class_weights),
        ProblemType.Regression: nn.MSELoss(),
    }

    if problem_type not in _LOSS_REGISTRY:
        raise ValueError(
            f"Problem type '{problem_type.value}' is not supported. "
            f"Available: {[p.value for p in _LOSS_REGISTRY]}."
        )

    return _LOSS_REGISTRY[problem_type]
