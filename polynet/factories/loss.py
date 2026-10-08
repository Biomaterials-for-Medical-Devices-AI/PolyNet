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

from polynet.config.enums import ProblemType, RegressionLoss


class RMSELoss(nn.Module):
    """Root mean squared error: ``sqrt(mean((y_pred - y_true) ** 2))`` over the batch."""

    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(y_pred, y_true))


def create_loss(
    problem_type: ProblemType | str,
    class_weights: torch.Tensor | None = None,
    regression_loss: RegressionLoss | str = RegressionLoss.RMSE,
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
    regression_loss:
        Loss for regression tasks: ``rmse`` (default, ``RMSELoss``), ``mse``
        (``nn.MSELoss``) or ``mae`` (``nn.L1Loss``). Ignored for
        classification.

    Returns
    -------
    nn.Module
        An instantiated loss function:
        - Classification → ``nn.CrossEntropyLoss``
        - Regression → ``RMSELoss`` / ``nn.MSELoss`` / ``nn.L1Loss``

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

    if problem_type == ProblemType.Classification:
        return nn.CrossEntropyLoss(weight=class_weights)
    if problem_type == ProblemType.Regression:
        _REGRESSION_LOSSES = {
            RegressionLoss.RMSE: RMSELoss,
            RegressionLoss.MSE: nn.MSELoss,
            RegressionLoss.MAE: nn.L1Loss,
        }
        return _REGRESSION_LOSSES[RegressionLoss(regression_loss)]()

    raise ValueError(
        f"Problem type '{problem_type.value}' is not supported. "
        f"Available: {[p.value for p in ProblemType]}."
    )
