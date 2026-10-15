import numpy as np
import torch

from sentinelfed.data import partition
from sentinelfed.model import IDSNet
from sentinelfed.server import fedavg
from sentinelfed.simulate import run


def test_dirichlet_partition_covers_every_sample_once():
    y = np.repeat(np.arange(4), 250)
    shards = partition(y, num_clients=8, scheme="dirichlet", alpha=0.3, seed=0)
    joined = np.concatenate(shards)
    assert len(shards) == 8
    assert all(len(s) > 0 for s in shards)
    assert sorted(joined.tolist()) == list(range(len(y)))


def test_iid_partition_is_balanced():
    shards = partition(np.zeros(100, dtype=int), 4, "iid", 1.0, seed=0)
    assert [len(s) for s in shards] == [25, 25, 25, 25]


def test_fedavg_weights_by_sample_count():
    a = {"w": torch.zeros(2), "n": torch.tensor(3)}
    b = {"w": torch.ones(2), "n": torch.tensor(7)}
    out = fedavg([a, b], [1, 3])
    assert torch.allclose(out["w"], torch.full((2,), 0.75))
    assert out["n"].item() == 3


def test_model_output_shape():
    model = IDSNet(10, 5, [16, 8], dropout=0.1)
    assert model(torch.randn(4, 10)).shape == (4, 5)


def test_short_simulation_learns():
    cfg = {
        "seed": 0,
        "data": {"source": "synthetic", "n_samples": 3000, "n_features": 20,
                 "n_classes": 4, "test_size": 0.2, "binary": False},
        "federation": {"num_clients": 4, "partition": "dirichlet",
                       "dirichlet_alpha": 1.0, "rounds": 8, "clients_per_round": 1.0},
        "client": {"local_epochs": 2, "batch_size": 64, "lr": 0.003},
        "model": {"hidden_dims": [32], "dropout": 0.0},
    }
    history = run(cfg)
    assert len(history) == 8
    # Always predicting benign scores 0.6 accuracy and ~0.19 macro-F1.
    assert history[-1]["accuracy"] > 0.7
    assert history[-1]["macro_f1"] > history[0]["macro_f1"]
