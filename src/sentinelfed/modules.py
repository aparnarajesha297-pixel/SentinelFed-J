"""LC, CE, GC and Weighted Attention Fusion (Phase 4).

Shapes, with B windows in a batch and H the embedding size:
  LC     ego graph of window t (+ the device's previous window)  -> h_L  (B, H)
  CE     h_L of the device's last L windows                      -> h_C  (B, H)
  GC     own activity pattern vs. up to M other devices          -> h_G  (B, H)
  Fusion h_L, h_C, h_G                                           -> h (B, H), weights (B, 3)
"""

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_dense_batch


def ego_batch(ego: dict, ids: np.ndarray) -> Batch:
    """Collate the stored ego graphs of windows ``ids`` into one PyG batch."""
    graphs = []
    for i in ids:
        n0, n1 = ego["node_ptr"][i], ego["node_ptr"][i + 1]
        e0, e1 = ego["edge_ptr"][i], ego["edge_ptr"][i + 1]
        graphs.append(Data(x=torch.from_numpy(np.asarray(ego["node_feat"][n0:n1], dtype=np.float32)),
                           edge_index=torch.from_numpy(ego["edge_index"][:, e0:e1]),
                           edge_attr=torch.from_numpy(np.asarray(ego["edge_attr"][e0:e1], dtype=np.float32))))
    return Batch.from_data_list(graphs)


class LCModule(nn.Module):
    """Local Correlation: GNN over the ego graph, then cross-attention from the
    current window's centre device to the nodes of its previous window.

    The guide compares a window with the *next* one; comparing with the
    previous one catches the same quiet-to-active escalation without reading
    the future (the later window is the one being scored).
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 64, heads: int = 2):
        super().__init__()
        self.inp = nn.Linear(node_dim, hidden)
        self.conv1 = GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=edge_dim,
                               add_self_loops=True, fill_value="mean")
        self.conv2 = GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=edge_dim,
                               add_self_loops=True, fill_value="mean")
        self.norm1, self.norm2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.cross = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.out = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

    def encode(self, g: Batch) -> torch.Tensor:
        h = torch.relu(self.inp(g.x))
        h = self.norm1(h + torch.relu(self.conv1(h, g.edge_index, g.edge_attr)))
        h = self.norm2(h + torch.relu(self.conv2(h, g.edge_index, g.edge_attr)))
        return h

    def forward(self, cur: Batch, prev: Batch) -> torch.Tensor:
        centre = self.encode(cur)[cur.ptr[:-1]]          # node 0 of every graph
        hp = self.encode(prev)
        keys, mask = to_dense_batch(hp, prev.batch)
        ctx, _ = self.cross(centre[:, None], keys, keys, key_padding_mask=~mask)
        ctx = ctx[:, 0]
        return self.out(torch.cat([centre, ctx, centre - ctx], dim=-1))


class CEModule(nn.Module):
    """Contextual Enrichment: bidirectional transformer over the device's
    last L window embeddings (all at or before the scored window), trained by
    reconstructing randomly hidden positions."""

    def __init__(self, dim: int, seq_len: int, layers: int = 2, heads: int = 4):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(seq_len, dim))
        self.mask_token = nn.Parameter(torch.zeros(dim))
        layer = nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.recon = nn.Linear(dim, dim)
        nn.init.normal_(self.pos, std=0.02)

    def encode(self, seq: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        return self.encoder(seq + self.pos, src_key_padding_mask=pad)

    def forward(self, seq: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        return self.encode(seq, pad)[:, -1]               # the scored window is last

    def masked_loss(self, seq: torch.Tensor, pad: torch.Tensor, p: float = 0.15) -> torch.Tensor:
        target = seq.detach()
        hide = (torch.rand(pad.shape, device=seq.device) < p) & ~pad
        if not hide.any():
            hide[:, -1] = True
        x = torch.where(hide[..., None], self.mask_token.expand_as(seq), seq)
        rec = self.recon(self.encode(x, pad))
        return ((rec - target) ** 2)[hide].mean()


class GCModule(nn.Module):
    """Global Correlation: cross-attention from a device's activity pattern
    to the patterns of the other devices active in the same period."""

    def __init__(self, pattern_len: int, hidden: int = 64, heads: int = 2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(pattern_len + 1, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.empty = nn.Parameter(torch.zeros(1, 1, hidden))   # "no other device" key
        self.cross = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.out = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

    def _embed(self, pat: torch.Tensor) -> torch.Tensor:
        pat = pat.float()
        return self.enc(torch.cat([pat, pat.mean(-1, keepdim=True)], dim=-1))

    def forward(self, own: torch.Tensor, others: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q = self._embed(own)                                  # (B, H)
        k = self._embed(others)                               # (B, M, H)
        k = torch.cat([self.empty.expand(len(q), 1, -1), k], dim=1)
        pad = torch.cat([torch.zeros(len(q), 1, dtype=torch.bool, device=q.device), ~mask], dim=1)
        ctx, _ = self.cross(q[:, None], k, k, key_padding_mask=pad)
        ctx = ctx[:, 0]
        return self.out(torch.cat([q, ctx, q - ctx], dim=-1))


class WeightedFusion(nn.Module):
    """Learned per-window weights over the views (Weighted Attention Fusion)."""

    def __init__(self, hidden: int = 64, att: int = 32):
        super().__init__()
        self.proj = nn.Linear(hidden, att)
        self.score = nn.Linear(att, 1, bias=False)

    def forward(self, views: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        v = torch.stack(views, dim=1)                         # (B, V, H)
        w = torch.softmax(self.score(torch.tanh(self.proj(v))).squeeze(-1), dim=1)
        return (w[..., None] * v).sum(1), w
