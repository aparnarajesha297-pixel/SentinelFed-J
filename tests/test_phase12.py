import numpy as np
import torch

from sentinelfed.federated import Site, aggregate, poison


def _setup():
    g = {"a": torch.zeros(3), "n": torch.tensor(1)}
    honest = [{"a": torch.tensor([1.0, 1.0, 1.0]), "n": torch.tensor(1)} for _ in range(5)]
    rogue = {"a": torch.tensor([100.0, 100.0, 100.0]), "n": torch.tensor(1)}
    sites = [Site(k, np.arange(1), np.arange(1), np.arange(1), n_train=10) for k in range(6)]
    return g, honest + [rogue], sites


def test_fedavg_is_pulled_by_one_rogue_update():
    g, states, sites = _setup()
    assert torch.allclose(aggregate("fedavg", sites, states, g)["a"], torch.full((3,), 17.5))


def test_median_and_krum_ignore_one_rogue_update():
    g, states, sites = _setup()
    assert torch.allclose(aggregate("median", sites, states, g)["a"], torch.ones(3))
    assert torch.allclose(aggregate("krum", sites, states, g)["a"], torch.ones(3))


def test_reputation_drops_for_an_update_opposite_to_the_median():
    g, states, sites = _setup()
    states[-1] = {"a": torch.tensor([-5.0, -5.0, -5.0]), "n": torch.tensor(1)}
    for _ in range(3):
        aggregate("reputation", sites, states, g)
    assert sites[-1].reputation < min(s.reputation for s in sites[:-1])
    assert sites[-1].reputation >= 0.05


def test_poison_flips_and_scales_the_update():
    g = {"a": torch.ones(2), "n": torch.tensor(3)}
    out = poison({"a": torch.tensor([2.0, 0.0]), "n": torch.tensor(3)}, g, 10.0)
    assert torch.allclose(out["a"], torch.tensor([-9.0, 11.0]))
    assert out["n"].item() == 3
