"""Unsupervised branch, risk calibration and the Trust/Verify/Reject policy (Phase 5)."""

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from torch import nn


class GRUAutoencoder(nn.Module):
    """Sequence autoencoder over a device's last L window feature vectors."""

    def __init__(self, in_dim: int, hidden: int = 64, latent: int = 16):
        super().__init__()
        self.encoder = nn.GRU(in_dim, hidden, batch_first=True)
        self.to_z = nn.Linear(hidden, latent)
        self.from_z = nn.Linear(latent, hidden)
        self.decoder = nn.GRU(in_dim, hidden, batch_first=True)
        self.out = nn.Linear(hidden, in_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.encoder(x)
        return self.to_z(h[-1])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        h0 = torch.tanh(self.from_z(z))[None]
        # teacher forcing: decoder sees the sequence shifted right by one step
        shifted = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        dec, _ = self.decoder(shifted, h0)
        return self.out(dec), z


def sequences(center_feat: np.ndarray, seq_idx: np.ndarray, ids: np.ndarray) -> tuple:
    idx = seq_idx[ids]
    pad = idx < 0
    x = center_feat[np.maximum(idx, 0)]
    x[pad] = 0.0
    return torch.from_numpy(x), torch.from_numpy(pad)


def train_autoencoder(center_feat, seq_idx, train_ids, val_ids, epochs=5, batch=1024,
                      lr=1e-3, seed=0, log=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = GRUAutoencoder(center_feat.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def loss_on(ids):
        x, pad = sequences(center_feat, seq_idx, ids)
        rec, _ = model(x)
        err = ((rec - x) ** 2).mean(-1)
        return err[~pad].mean()

    best, best_state = np.inf, None
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(train_ids)
        for s in range(0, len(perm), batch):
            loss = loss_on(perm[s:s + batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = np.mean([loss_on(val_ids[s:s + 8192]).item() for s in range(0, len(val_ids), 8192)])
        log(f"  GRU-AE epoch {ep + 1}: U_val reconstruction MSE {val:.4f}")
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.eval(), best


@torch.no_grad()
def latent(model, center_feat, seq_idx, ids, batch=8192) -> np.ndarray:
    return np.concatenate([model.encode(sequences(center_feat, seq_idx, ids[s:s + batch])[0]).numpy()
                           for s in range(0, len(ids), batch)])


@dataclass
class Mahalanobis:
    """Distance of z from the normal-behaviour map fitted on U_train."""
    mean: np.ndarray
    inv_cov: np.ndarray
    ridge: float

    @classmethod
    def fit(cls, z: np.ndarray, ridge: float = 1e-3) -> "Mahalanobis":
        cov = np.cov(z, rowvar=False) + ridge * np.eye(z.shape[1])   # small fixed value
        return cls(z.mean(0), np.linalg.inv(cov), ridge)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        d = z - self.mean
        return np.sqrt(np.einsum("ij,jk,ik->i", d, self.inv_cov, d))


class EmpiricalScaler:
    """Maps a distance to 0-1 as its percentile among U_val distances."""

    def __init__(self, reference: np.ndarray):
        self.ref = np.sort(reference)

    def __call__(self, d: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.ref, d, side="right") / len(self.ref)


@dataclass
class RiskModel:
    """Mixes the two scores and calibrates the mixture (Platt scaling)."""
    alpha: float
    platt: LogisticRegression

    @classmethod
    def fit(cls, p_sup, d_norm, y, grid=np.linspace(0, 1, 21)):
        # pick the mixing weight by validation PR-AUC, then calibrate on validation
        scores = [average_precision_score(y, a * p_sup + (1 - a) * d_norm) for a in grid]
        alpha = float(grid[int(np.argmax(scores))])
        mix = alpha * p_sup + (1 - alpha) * d_norm
        platt = LogisticRegression(C=1e6).fit(mix[:, None], y)
        return cls(alpha, platt)

    def __call__(self, p_sup, d_norm):
        mix = self.alpha * p_sup + (1 - self.alpha) * d_norm
        return self.platt.predict_proba(mix[:, None])[:, 1]


def best_f1_threshold(y, score, grid=None) -> tuple[float, float]:
    grid = np.unique(np.quantile(score, np.linspace(0, 1, 501))) if grid is None else grid
    f1s = [f1_score(y, score >= t, zero_division=0) for t in grid]
    i = int(np.argmax(f1s))
    return float(grid[i]), float(f1s[i])


TRUST, VERIFY, REJECT = "Trust", "Verify", "Reject"


def decide(p_sup, d_norm, tau_sup, tau_unsup, labels_available=True) -> np.ndarray:
    """Trust / Verify / Reject, as specified in the guide."""
    u = d_norm >= tau_unsup
    if not labels_available:
        return np.where(u, REJECT, TRUST)
    s = p_sup >= tau_sup
    return np.where(s & u, REJECT, np.where(~s & ~u, TRUST, VERIFY))
