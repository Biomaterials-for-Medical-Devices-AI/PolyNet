"""
tests/test_eval_loss.py
=======================
Validation and test losses are computed over the whole set, so they do not
depend on the batch size: with the RMSE loss, a batch size of 1 used to give
the MAE, and HPO trials with different batch sizes were scored on different
measures.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from torch_geometric.data import Data  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402
from torch_geometric.nn import global_mean_pool  # noqa: E402

from polynet.config.enums import ProblemType, RegressionLoss  # noqa: E402
from polynet.factories.loss import create_loss  # noqa: E402
from polynet.training.gnn import eval_network, evaluate_losses  # noqa: E402


class _MeanOfFeatures(torch.nn.Module):
    """Predicts the mean node feature of each graph (no parameters)."""

    def __init__(self, problem_type, n_outputs=1):
        super().__init__()
        self.problem_type = problem_type
        self.n_outputs = n_outputs

    def forward(self, x, batch_index, **kwargs):
        return global_mean_pool(x, batch_index).repeat(1, self.n_outputs)


def _graphs(predictions, targets):
    return [
        Data(x=torch.tensor([[p]]), edge_index=torch.empty(2, 0, dtype=torch.long), y=torch.tensor([t]))
        for p, t in zip(predictions, targets)
    ]


_PRED = [1.0, 2.0, 3.0, 4.0, 10.0]
_TRUE = [1.5, 2.0, 2.0, 5.0, 4.0]


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_rmse_is_over_the_whole_set_for_any_batch_size(batch_size):
    loader = DataLoader(_graphs(_PRED, _TRUE), batch_size=batch_size)
    errors = torch.tensor(_PRED) - torch.tensor(_TRUE)
    loss = eval_network(
        _MeanOfFeatures(ProblemType.Regression), loader, create_loss("regression"), "cpu"
    )

    assert loss == pytest.approx(errors.pow(2).mean().sqrt().item(), rel=1e-6)
    assert loss != pytest.approx(errors.abs().mean().item())  # not the MAE


def test_mae_and_mse_are_set_means():
    loader = DataLoader(_graphs(_PRED, _TRUE), batch_size=2)
    errors = torch.tensor(_PRED) - torch.tensor(_TRUE)
    model = _MeanOfFeatures(ProblemType.Regression)
    mae, mse = evaluate_losses(
        model,
        loader,
        [
            create_loss("regression", regression_loss=RegressionLoss.MAE),
            create_loss("regression", regression_loss=RegressionLoss.MSE),
        ],
        "cpu",
    )
    assert mae == pytest.approx(errors.abs().mean().item(), rel=1e-6)
    assert mse == pytest.approx(errors.pow(2).mean().item(), rel=1e-6)


def test_classification_cross_entropy_does_not_depend_on_batch_size():
    graphs = _graphs([0.2, -1.0, 0.7, 0.1], [0, 1, 1, 0])
    model = _MeanOfFeatures(ProblemType.Classification, n_outputs=2)
    loss_fn = create_loss("classification", class_weights=torch.tensor([0.3, 0.7]))
    losses = [
        eval_network(model, DataLoader(graphs, batch_size=b), loss_fn, "cpu") for b in (1, 2, 4)
    ]
    assert losses == pytest.approx([losses[2]] * 3, rel=1e-6)
