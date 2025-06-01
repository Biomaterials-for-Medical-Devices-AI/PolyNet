import torch.nn as nn
from torch.optim import SGD, Adam, RMSprop
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, ReduceLROnPlateau, StepLR
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from polynet.models.GAT import GATClassifier, GATRegressor
from polynet.models.GCN import GCNClassifier, GCNRegressor
from polynet.models.TransfomerGNN import TransformerGNNClassifier, TransformerGNNRegressor
from polynet.models.graphsage import GraphSAGE
from polynet.options.enums import Networks, Optimizers, ProblemTypes, Schedulers, SplitTypes


def create_network(network: str, problem_type: ProblemTypes, **kwargs):

    # Create a network
    if network == Networks.GCN:
        if problem_type == ProblemTypes.Classification:
            network = GCNClassifier(**kwargs)
        elif problem_type == ProblemTypes.Regression:
            network = GCNRegressor(**kwargs)

    elif network == Networks.TransformerGNN:
        if problem_type == ProblemTypes.Classification:
            network = TransformerGNNClassifier(**kwargs)
        elif problem_type == ProblemTypes.Regression:
            network = TransformerGNNRegressor(**kwargs)

    elif network == Networks.GAT:
        if problem_type == ProblemTypes.Classification:
            network = GATClassifier(**kwargs)
        elif problem_type == ProblemTypes.Regression:
            network = GATRegressor(**kwargs)

    elif network == Networks.GraphSAGE:
        network = GraphSAGE(**kwargs)

    return network


def make_optimizer(optimizer, model, lr):
    if optimizer == Optimizers.Adam:
        optimizer = Adam(model.parameters(), lr=lr, eps=1e-9)
    elif optimizer == Optimizers.SGD:
        optimizer = SGD(model.parameters(), lr=lr)
    elif optimizer == Optimizers.RMSprop:
        optimizer = RMSprop(model.parameters(), lr=lr)
    else:
        raise NotImplementedError(f"Optimizer type {optimizer} not implemented")

    return optimizer


def make_scheduler(scheduler, optimizer, step_size, gamma, min_lr):
    if scheduler == Schedulers.StepLR:
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler == Schedulers.MultiStepLR:
        scheduler = MultiStepLR(optimizer, milestones=step_size, gamma=gamma)
    elif scheduler == Schedulers.ExponentialLR:
        scheduler = ExponentialLR(optimizer, gamma=gamma)
    elif scheduler == Schedulers.ReduceLROnPlateau:
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=gamma, patience=step_size, min_lr=min_lr
        )
    else:
        raise NotImplementedError(f"Scheduler type {scheduler} not implemented")

    return scheduler


def make_loss(problem_type, mae=None):
    if problem_type == "classification":
        loss = nn.CrossEntropyLoss()
    elif problem_type == "regression" and mae is None:
        loss = nn.MSELoss()
    else:
        raise ValueError(f"Problem type {problem_type} not supported")
    return loss


class Split_Generator:
    def __init__(self, split_type, batch_size, train_size=None, val_size=None, test_size=None):
        self.split_type = split_type
        self.batch_size = batch_size
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

    def split_data(
        self, dataset, train_split_indices=None, test_split_indices=None, val_split_indices=None
    ):
        length = len(dataset)
        if self.split_type in [
            SplitTypes.TrainValTest,
            SplitTypes.TrainTest,
            SplitTypes.CrossValidation,
            SplitTypes.NestedCrossValidation,
        ]:
            raise NotImplementedError(f"Split type {self.split_type} is not yet implemented!")

        elif self.split_type == SplitTypes.LeaveOneOut:
            # Deduce the indices for the train set based on test and validation indices
            if train_split_indices is None:
                train_split_indices = []
                for i in range(length):
                    test_idx = test_split_indices[i]
                    if val_split_indices is not None:
                        val_idx = val_split_indices[i]
                        train_idxs = [x for x in range(length) if x not in [test_idx, val_idx]]
                    else:
                        train_idxs = [x for x in range(length) if x != test_idx]
                    train_split_indices.append(train_idxs)

            for i in range(length):
                train_idxs = train_split_indices[i]
                train_subset = Subset(dataset, train_idxs)
                test_idx = test_split_indices[i]
                test_subset = Subset(dataset, [test_idx])

                train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
                test_loader = DataLoader(test_subset)
                if val_split_indices is not None:
                    val_idx = val_split_indices[i]
                    val_subset = Subset(dataset, [val_idx])
                    val_loader = DataLoader(val_subset)
                else:
                    val_loader = None
                yield train_loader, val_loader, test_loader
