"""
tests/test_polymer_descriptor_scaling.py
========================================
Polymer-level descriptors fed to the GNN readout are scaled with the same
``FeatureTransformer`` (and ``feature_preprocessing.scaler`` strategy) as the
tabular features, fitted on the training split only, and the fitted scaler
travels inside the saved model.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.data import Data  # noqa: E402

from polynet.config.enums import (  # noqa: E402
    Network,
    ProblemType,
    TrainingParam,
    TransformDescriptor,
)
from polynet.data.feature_transformer import FeatureTransformer  # noqa: E402
from polynet.models.gnn.gcn import GCNRegressor  # noqa: E402
from polynet.models.persistence import load_gnn_model, save_gnn_model  # noqa: E402
from polynet.training.gnn import (  # noqa: E402
    fit_polymer_descriptor_scaler,
    train_gnn_ensemble,
)

N_NODE, N_EDGE, N_DESC = 8, 3, 2


def _graph(idx: int, desc: list[float], y: float = 0.0) -> Data:
    gen = torch.Generator().manual_seed(idx)
    return Data(
        x=torch.randn(4, N_NODE, generator=gen),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
        edge_attr=torch.randn(4, N_EDGE, generator=gen),
        y=torch.tensor(y, dtype=torch.float32),
        polymer_descriptors=torch.tensor([desc], dtype=torch.float32),
        idx=idx,
    )


def _dataset(n: int = 24) -> list[Data]:
    rng = np.random.default_rng(0)
    # Molecular-weight-like first column (thousands) and a small second column.
    return [
        _graph(i, [float(rng.uniform(5_000, 50_000)), float(rng.uniform(0, 1))], y=float(i))
        for i in range(n)
    ]


def _predict(model, g: Data) -> np.ndarray:
    return model.predict(
        x=g.x,
        edge_index=g.edge_index,
        edge_attr=g.edge_attr,
        batch_index=torch.zeros(g.num_nodes, dtype=torch.long),
        polymer_descriptors=g.polymer_descriptors,
    )


def _model() -> GCNRegressor:
    model = GCNRegressor(
        improved=False,
        n_node_features=N_NODE,
        n_edge_features=N_EDGE,
        n_polymer_descriptors=N_DESC,
        seed=0,
    )
    model.eval()
    return model


def test_scaler_is_fitted_on_training_graphs_only():
    train = _dataset()[:16]

    scaler = fit_polymer_descriptor_scaler(train, TransformDescriptor.StandardScaler)

    train_x = torch.cat([d.polymer_descriptors for d in train]).numpy()
    np.testing.assert_allclose(scaler.scaler_.mean_, train_x.mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(scaler.scaler_.var_, train_x.var(axis=0), rtol=1e-5)
    assert isinstance(scaler, FeatureTransformer)
    assert scaler.selectors == {}  # scaling only, no feature selection


@pytest.mark.parametrize(
    "strategy", [s for s in TransformDescriptor if s != TransformDescriptor.NoTransformation]
)
def test_model_applies_exactly_the_tabular_transformation(strategy):
    """Every tabular scaler option is supported, and the model feeds the readout
    exactly ``FeatureTransformer.transform`` of the raw descriptors."""
    data = _dataset()
    scaler = fit_polymer_descriptor_scaler(data[:16], strategy)
    g = data[20]

    scaled_model = _model()
    scaled_model.set_polymer_descriptor_scaler(scaler)
    with_scaler = _predict(scaled_model, g)

    raw_model = _model()  # same seed → same weights, no scaler
    pre_scaled = g.clone()
    pre_scaled.polymer_descriptors = torch.as_tensor(
        scaler.transform(g.polymer_descriptors.numpy()), dtype=torch.float32
    )
    np.testing.assert_allclose(with_scaler, _predict(raw_model, pre_scaled), rtol=1e-5)


def test_no_transformation_returns_no_scaler():
    assert fit_polymer_descriptor_scaler(_dataset(), TransformDescriptor.NoTransformation) is None


def test_graphs_without_descriptors_return_no_scaler():
    g = _graph(0, [1.0, 2.0])
    del g.polymer_descriptors
    assert fit_polymer_descriptor_scaler([g], TransformDescriptor.StandardScaler) is None


def test_nan_descriptor_in_training_raises():
    data = _dataset()
    data[0].polymer_descriptors[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or infinite"):
        fit_polymer_descriptor_scaler(data, TransformDescriptor.StandardScaler)


def test_save_load_predict_roundtrip_keeps_scaling(tmp_path):
    data = _dataset()
    model = _model()
    unscaled = _predict(model, data[20])
    model.set_polymer_descriptor_scaler(
        fit_polymer_descriptor_scaler(data[:16], TransformDescriptor.StandardScaler)
    )
    before = _predict(model, data[20])
    assert not np.allclose(before, unscaled)  # scaling actually changes the input

    path = tmp_path / "model.pt"
    save_gnn_model(model, path)
    reloaded = load_gnn_model(path)
    reloaded.eval()

    np.testing.assert_array_equal(_predict(reloaded, data[20]), before)


def test_legacy_model_without_scaler_attribute_is_unchanged():
    """Models pickled before descriptor scaling existed have no scaler attribute
    and must keep using the raw descriptors."""
    data = _dataset()
    model = _model()
    expected = _predict(model, data[0])
    del model.polymer_descriptor_scaler
    np.testing.assert_array_equal(_predict(model, data[0]), expected)


def test_ensemble_attaches_per_split_scaler_fitted_on_train_ids():
    data = _dataset()
    train_ids = [[d.idx for d in data[:16]], [d.idx for d in data[8:24]]]
    val_ids = [[d.idx for d in data[16:20]], [d.idx for d in data[:4]]]
    test_ids = [[d.idx for d in data[20:]], [d.idx for d in data[4:8]]]

    trained, _, _ = train_gnn_ensemble(
        experiment_path=None,
        dataset=data,
        split_indexes=(train_ids, val_ids, test_ids),
        gnn_conv_params={
            Network.GCN: {
                TrainingParam.LearningRate: 0.01,
                TrainingParam.BatchSize: 8,
                "improved": False,
                "embedding_dim": 8,
                "n_convolutions": 1,
                "readout_layers": 2,
                "dropout": 0.0,
            }
        },
        problem_type=ProblemType.Regression,
        num_classes=1,
        random_seed=0,
        epochs=1,
        polymer_descriptor_scaler=TransformDescriptor.MinMaxScaler,
    )

    by_idx = {d.idx: d for d in data}
    for i, ids in enumerate(train_ids, start=1):
        scaler = trained[f"{Network.GCN.value}_{i}"].polymer_descriptor_scaler
        train_x = torch.cat([by_idx[j].polymer_descriptors for j in ids]).numpy()
        np.testing.assert_allclose(scaler.scaler_.data_min_, train_x.min(axis=0))
        np.testing.assert_allclose(scaler.scaler_.data_max_, train_x.max(axis=0))
