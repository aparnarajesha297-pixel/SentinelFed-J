"""Training for the supervised branch (Phase 5).

The supervised branch is trained in two stages, because training LC end to
end through CE would mean running the GNN on 8 history graphs per window:

  stage A  LC + a linear classifier on the ego graphs (class-weighted loss)
  stage B  LC frozen; CE (initialised by masked pretraining), GC, fusion and
           the classifier are trained on top of the stored h_L

Both stages use S_train labels only and pick their best epoch on S_val.
"""

from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from .modules import CEModule, GCModule, LCModule, WeightedFusion

VIEWS = ("LC", "CE", "GC")


def collate(ego: dict, ids: np.ndarray) -> SimpleNamespace:
    """Vectorised batch of stored ego graphs (much faster than PyG Data lists)."""
    ids = np.asarray(ids)
    n0, n1 = ego["node_ptr"][ids], ego["node_ptr"][ids + 1]
    e0, e1 = ego["edge_ptr"][ids], ego["edge_ptr"][ids + 1]
    nc, ec = n1 - n0, e1 - e0
    noff = np.concatenate([[0], np.cumsum(nc)[:-1]])
    eoff = np.concatenate([[0], np.cumsum(ec)[:-1]])
    nidx = np.arange(nc.sum()) - np.repeat(noff, nc) + np.repeat(n0, nc)
    eidx = np.arange(ec.sum()) - np.repeat(eoff, ec) + np.repeat(e0, ec)
    edge_index = ego["edge_index"][:, eidx] + np.repeat(noff, ec)
    return SimpleNamespace(
        x=torch.from_numpy(ego["node_feat"][nidx]),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(ego["edge_attr"][eidx]),
        batch=torch.from_numpy(np.repeat(np.arange(len(ids)), nc)),
        ptr=torch.from_numpy(np.concatenate([[0], np.cumsum(nc)])),
    )


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights, square-rooted so rare classes are boosted
    without one tiny class dominating the loss."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = np.sqrt(counts.sum() / (n_classes * np.maximum(counts, 1)))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


class LCClassifier(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden, n_classes):
        super().__init__()
        self.lc = LCModule(node_dim, edge_dim, hidden)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, cur, prev):
        return self.head(self.lc(cur, prev))


class SupervisedHead(nn.Module):
    """CE + GC + Weighted Fusion + classifier over stored h_L.

    ``views`` selects which embeddings are fused (used by the Phase 7
    ablation); fusion weights are learned only over the selected views.
    """

    def __init__(self, hidden, seq_len, pattern_len, n_classes, views=VIEWS):
        super().__init__()
        self.views = tuple(views)
        self.ce = CEModule(hidden, seq_len)
        self.gc = GCModule(pattern_len, hidden)
        self.fusion = WeightedFusion(hidden)
        self.classifier = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                        nn.Dropout(0.1), nn.Linear(hidden, n_classes))

    def embed(self, h_l, seq, pad, own, others, mask):
        out = {}
        if "LC" in self.views:
            out["LC"] = h_l
        if "CE" in self.views:
            out["CE"] = self.ce(seq, pad)
        if "GC" in self.views:
            out["GC"] = self.gc(own, others, mask)
        h, w = self.fusion([out[v] for v in self.views])
        return h, w, out

    def forward(self, *inputs):
        h, w, _ = self.embed(*inputs)
        return self.classifier(h), w


def lc_batches(windows, ids):
    prev = windows["prev_window"].to_numpy()[ids]
    return np.where(prev >= 0, prev, ids)


def train_lc(ego, windows, y, train_ids, val_ids, n_classes, hidden=64, epochs=3,
             batch=1024, lr=1e-3, seed=0, log=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = LCClassifier(ego["node_feat"].shape[1], ego["edge_attr"].shape[1], hidden, n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y[train_ids], n_classes))
    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(train_ids)
        for s in range(0, len(perm), batch):
            b = perm[s:s + batch]
            logits = model(collate(ego, b), collate(ego, lc_batches(windows, b)))
            loss = loss_fn(logits, torch.from_numpy(y[b]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        pred = predict_lc(model, ego, windows, val_ids).argmax(1)
        f1 = f1_score(y[val_ids], pred, average="macro")
        log(f"  LC epoch {ep + 1}: val macro-F1 {f1:.4f}")
        if f1 > best:
            best, best_state = f1, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.eval(), best


@torch.no_grad()
def predict_lc(model, ego, windows, ids, batch=4096, embed=False):
    model.eval()
    out = []
    for s in range(0, len(ids), batch):
        b = ids[s:s + batch]
        cur, prev = collate(ego, b), collate(ego, lc_batches(windows, b))
        out.append(model.lc(cur, prev) if embed else torch.softmax(model(cur, prev), 1))
    return torch.cat(out).numpy()


def head_inputs(h_L, seq_idx, act, ids):
    idx = torch.from_numpy(seq_idx[ids])
    pad = idx < 0
    seq = h_L[idx.clamp(min=0)]
    seq = seq.masked_fill(pad[..., None], 0.0)
    return (h_L[torch.from_numpy(ids)], seq, pad,
            torch.from_numpy(act["own"][ids]), torch.from_numpy(act["others"][ids]),
            torch.from_numpy(act["others_mask"][ids]))


def train_head(h_L, seq_idx, act, y, train_ids, val_ids, n_classes, seq_len, pattern_len,
               views=VIEWS, epochs=6, batch=1024, lr=1e-3, pretrain_epochs=2, seed=0, log=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SupervisedHead(h_L.shape[1], seq_len, pattern_len, n_classes, views)

    if "CE" in views and pretrain_epochs:
        opt = torch.optim.Adam(model.ce.parameters(), lr=lr)
        for ep in range(pretrain_epochs):
            model.train()
            perm = rng.permutation(train_ids)
            losses = []
            for s in range(0, len(perm), batch):
                _, seq, pad, *_ = head_inputs(h_L, seq_idx, act, perm[s:s + batch])
                loss = model.ce.masked_loss(seq, pad)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            log(f"  CE masked pretraining epoch {ep + 1}: MSE {np.mean(losses):.4f}")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y[train_ids], n_classes))
    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(train_ids)
        for s in range(0, len(perm), batch):
            b = perm[s:s + batch]
            logits, _ = model(*head_inputs(h_L, seq_idx, act, b))
            loss = loss_fn(logits, torch.from_numpy(y[b]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        prob, _ = predict_head(model, h_L, seq_idx, act, val_ids)
        f1 = f1_score(y[val_ids], prob.argmax(1), average="macro")
        log(f"  head epoch {ep + 1}: val macro-F1 {f1:.4f}")
        if f1 > best:
            best, best_state = f1, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.eval(), best


@torch.no_grad()
def predict_head(model, h_L, seq_idx, act, ids, batch=8192, embeddings=False):
    model.eval()
    probs, weights, embs = [], [], {v: [] for v in model.views + ("fused",)}
    for s in range(0, len(ids), batch):
        b = ids[s:s + batch]
        h, w, views = model.embed(*head_inputs(h_L, seq_idx, act, b))
        probs.append(torch.softmax(model.classifier(h), 1))
        weights.append(w)
        if embeddings:
            for k, v in views.items():
                embs[k].append(v)
            embs["fused"].append(h)
    out = (torch.cat(probs).numpy(), torch.cat(weights).numpy())
    if embeddings:
        return out + ({k: torch.cat(v).numpy() for k, v in embs.items()},)
    return out
