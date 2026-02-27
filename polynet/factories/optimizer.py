"""
polynet.factories.optimizer
============================
Factory functions for constructing PyTorch optimizers and learning rate
schedulers.

Design
------
Both factories use dispatch dicts rather than if/elif chains, and accept
either enum members or their string values. The scheduler factory uses an
explicit ``SchedulerConfig`` dataclass to avoid the ambiguous ``step_size``
argument that meant different things for different schedulers in the
previous implementation.

Public API
----------
::

    from polynet.factories.optimizer import create_optimizer, create_scheduler
    from polynet.config.enums import Optimizer, Scheduler

    optimizer = create_optimizer(
        optimizer=Optimizer.Adam,
        model=model,
        lr=0.001,
    )

    scheduler = create_scheduler(
        scheduler=Scheduler.ReduceLROnPlateau,
        optimizer=optimizer,
        patience=10,
        gamma=0.5,
        min_lr=1e-6,
    )
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from torch.optim import SGD, Adadelta, Adagrad, Adam, RMSprop
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, ReduceLROnPlateau, StepLR

from polynet.config.enums import Optimizer, Scheduler

# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------


def create_optimizer(optimizer: Optimizer | str, model: nn.Module, lr: float, **kwargs: Any) -> Any:
    """
    Construct and return a PyTorch optimizer bound to ``model.parameters()``.

    Parameters
    ----------
    optimizer:
        The optimizer type. Accepts an ``Optimizer`` enum member or its
        string value (e.g. ``"adam"``).
    model:
        The model whose parameters the optimizer will update.
    lr:
        Initial learning rate.
    **kwargs:
        Additional keyword arguments forwarded to the optimizer constructor.
        For example, ``weight_decay=1e-4`` for Adam.

    Returns
    -------
    torch.optim.Optimizer
        An instantiated optimizer.

    Raises
    ------
    ValueError
        If the optimizer type is not recognised.

    Examples
    --------
    >>> from polynet.factories.optimizer import create_optimizer
    >>> from polynet.config.enums import Optimizer
    >>> opt = create_optimizer(Optimizer.Adam, model, lr=0.001)
    """
    optimizer = Optimizer(optimizer) if isinstance(optimizer, str) else optimizer

    _OPTIMIZER_REGISTRY = {
        Optimizer.Adam: lambda: Adam(model.parameters(), lr=lr, eps=1e-9, **kwargs),
        Optimizer.SGD: lambda: SGD(model.parameters(), lr=lr, **kwargs),
        Optimizer.RMSprop: lambda: RMSprop(model.parameters(), lr=lr, **kwargs),
        Optimizer.Adadelta: lambda: Adadelta(model.parameters(), lr=lr, **kwargs),
        Optimizer.Adagrad: lambda: Adagrad(model.parameters(), lr=lr, **kwargs),
    }

    if optimizer not in _OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Optimizer '{optimizer.value}' is not registered. "
            f"Available: {[o.value for o in _OPTIMIZER_REGISTRY]}."
        )

    return _OPTIMIZER_REGISTRY[optimizer]()


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------


def create_scheduler(
    scheduler: Scheduler | str,
    optimizer: Any,
    *,
    gamma: float = 0.1,
    step_size: int = 10,
    milestones: list[int] | None = None,
    patience: int = 10,
    min_lr: float = 1e-8,
    **kwargs: Any,
) -> Any:
    """
    Construct and return a PyTorch learning rate scheduler.

    Parameters are named explicitly per scheduler type to avoid the
    previous ambiguity where ``step_size`` meant different things
    depending on which scheduler was selected.

    Parameters
    ----------
    scheduler:
        The scheduler type. Accepts a ``Scheduler`` enum member or its
        string value (e.g. ``"reduce_lr_on_plateau"``).
    optimizer:
        The optimizer whose learning rate the scheduler will modify.
    gamma:
        Multiplicative factor of learning rate decay. Used by
        ``StepLR``, ``MultiStepLR``, ``ExponentialLR``, and
        ``ReduceLROnPlateau``.
    step_size:
        Period of learning rate decay (in epochs). Used by ``StepLR``
        only.
    milestones:
        List of epoch indices at which to decay the learning rate.
        Used by ``MultiStepLR`` only. Defaults to ``[30, 60, 90]``
        if not provided.
    patience:
        Number of epochs with no improvement before reducing the
        learning rate. Used by ``ReduceLROnPlateau`` only.
    min_lr:
        Minimum learning rate. Used by ``ReduceLROnPlateau`` only.
    **kwargs:
        Additional keyword arguments forwarded to the scheduler
        constructor.

    Returns
    -------
    torch.optim.lr_scheduler.LRScheduler
        An instantiated learning rate scheduler.

    Raises
    ------
    ValueError
        If the scheduler type is not recognised.

    Examples
    --------
    >>> from polynet.factories.optimizer import create_scheduler
    >>> from polynet.config.enums import Scheduler
    >>> sched = create_scheduler(
    ...     Scheduler.ReduceLROnPlateau,
    ...     optimizer,
    ...     patience=10,
    ...     gamma=0.5,
    ...     min_lr=1e-6,
    ... )
    """
    scheduler = Scheduler(scheduler) if isinstance(scheduler, str) else scheduler

    _milestones = milestones if milestones is not None else [30, 60, 90]

    _SCHEDULER_REGISTRY = {
        Scheduler.StepLR: lambda: StepLR(optimizer, step_size=step_size, gamma=gamma, **kwargs),
        Scheduler.MultiStepLR: lambda: MultiStepLR(
            optimizer, milestones=_milestones, gamma=gamma, **kwargs
        ),
        Scheduler.ExponentialLR: lambda: ExponentialLR(optimizer, gamma=gamma, **kwargs),
        Scheduler.ReduceLROnPlateau: lambda: ReduceLROnPlateau(
            optimizer, mode="min", factor=gamma, patience=patience, min_lr=min_lr, **kwargs
        ),
    }

    if scheduler not in _SCHEDULER_REGISTRY:
        raise ValueError(
            f"Scheduler '{scheduler.value}' is not registered. "
            f"Available: {[s.value for s in _SCHEDULER_REGISTRY]}."
        )

    return _SCHEDULER_REGISTRY[scheduler]()
