"""
polynet.factories
==================
Factory functions and classes for constructing all pipeline objects.

All public factory functions accept individual arguments rather than
requiring a config object, so they can be used independently by library
users who do not want to go through the full config system::

    from polynet.factories import create_network, create_optimizer, create_loss
    from polynet.config.enums import Network, ProblemType, Optimizer

    model = create_network(Network.GCN, ProblemType.Regression, embedding_dim=64, ...)
    optimizer = create_optimizer(Optimizer.Adam, model, lr=0.001)
    loss_fn = create_loss(ProblemType.Regression)

    # Compute split indices from a DataFrame
    from polynet.factories import get_data_split_indices, SplitGenerator
    from polynet.config.enums import SplitType, SplitMethod

    train_idxs, val_idxs, test_idxs = get_data_split_indices(
        data=df, split_type=SplitType.TrainValTest, ...
    )
    generator = SplitGenerator(SplitType.TrainValTest, batch_size=32)
    for train_loader, val_loader, test_loader in generator.split(
        dataset,
        train_indices=train_idxs,
        val_indices=val_idxs,
        test_indices=test_idxs,
    ):
        ...
"""

from polynet.factories.dataloader import SplitGenerator, get_data_split_indices
from polynet.factories.loss import create_loss
from polynet.factories.network import create_network, list_available_networks
from polynet.factories.optimizer import create_optimizer, create_scheduler

__all__ = [
    # Network
    "create_network",
    "list_available_networks",
    # Optimisation
    "create_optimizer",
    "create_scheduler",
    # Loss
    "create_loss",
    # Data splitting
    "get_data_split_indices",
    "SplitGenerator",
]
