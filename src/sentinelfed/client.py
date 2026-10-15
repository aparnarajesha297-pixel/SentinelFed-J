import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class Client:
    """One IIoT site holding a private shard of traffic data."""

    def __init__(self, cid: int, X: np.ndarray, y: np.ndarray, cfg: dict):
        self.cid = cid
        self.num_samples = len(y)
        self.cfg = cfg
        self.data = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

    def fit(self, model: nn.Module, global_state: dict, seed: int) -> dict:
        """Run local epochs starting from the global weights; return new weights."""
        model.load_state_dict(global_state)
        model.train()
        gen = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            self.data,
            batch_size=self.cfg["batch_size"],
            shuffle=True,
            generator=gen,
        )
        opt = torch.optim.Adam(model.parameters(), lr=self.cfg["lr"])
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(self.cfg["local_epochs"]):
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(model(xb), yb).backward()
                opt.step()
        return {k: v.detach().clone() for k, v in model.state_dict().items()}
