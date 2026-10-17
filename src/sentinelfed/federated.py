"""Federated training of the supervised branch (Phase 11).

Sites share only LC and CE (guide, Phase 11). GC, the fusion layer and the
classifier stay local, so every site ends up with its own personalised
model built on the shared LC/CE. Training mirrors the two centralised
stages in train.py:

  stage A  rounds of local LC + local linear head training; LC is aggregated
  stage B  CE masked pretraining rounds, then supervised rounds of local
           CE + GC + fusion + classifier; CE is aggregated

Aggregators: isolated (no sharing), FedAvg, FedProx (proximal term on the
shared weights during local training) and the guide's reputation-weighted
rule.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .server import fedavg
from .train import (LCClassifier, SupervisedHead, class_weights, collate, head_inputs,
                    lc_batches, predict_head, predict_lc)

METHODS = ("isolated", "fedavg", "fedprox", "reputation")
ROBUST = ("median", "krum")


# ---------------------------------------------------------------- site split
def assign_sites(data, n_sites=6):
    """Deterministic, non-IID assignment of windows to sites.

    Attack recordings (whole segment_ids) are sorted by (attack type,
    segment_id) and dealt round-robin, so each site holds 2-3 attack types.
    Normal windows are cut, within each split, into n_sites contiguous time
    blocks (by recording, then end slot); site j gets block j.
    """
    w = data.windows
    seg = w["segment_id"].to_numpy()
    typ = w["attack_type"].to_numpy()
    attack_segs = sorted({(t, s) for t, s in zip(typ, seg) if t != "Normal"})
    seg_site = {s: i % n_sites for i, (_, s) in enumerate(attack_segs)}
    site = np.full(len(w), -1)
    for s, k in seg_site.items():
        site[seg == s] = k
    split = w["split"].to_numpy()
    for sp in ("train", "val", "test"):
        idx = np.flatnonzero((typ == "Normal") & (split == sp))
        order = np.lexsort((w["end_slot"].to_numpy()[idx], seg[idx]))
        for k, block in enumerate(np.array_split(idx[order], n_sites)):
            site[block] = k
    assert (site >= 0).all()
    return site, seg_site


@dataclass
class Site:
    k: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    lc_model: nn.Module = None
    head: nn.Module = None
    n_train: int = 0
    reputation: float = 1.0
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------- aggregation
def _shared(state, prefix):
    return {k: v for k, v in state.items() if k.startswith(prefix)}


def _flat(state, keys):
    return torch.cat([state[k].flatten().float() for k in keys])


def _unflat(vec, like, keys):
    out, pos = {}, 0
    for k in keys:
        n = like[k].numel()
        out[k] = vec[pos:pos + n].view_as(like[k]).to(like[k].dtype)
        pos += n
    return out


def aggregate(method, sites, shared_states, global_state, rep_beta=0.8, rep_floor=0.05, krum_f=2):
    """Return the new shared state (or None for isolated training)."""
    if method == "isolated":
        return None
    sizes = [s.n_train for s in sites]
    if method in ("fedavg", "fedprox"):
        return fedavg(shared_states, sizes)
    keys = [k for k in global_state if torch.is_floating_point(global_state[k])]
    if method == "median":                      # coordinate-wise median of the submitted models
        med = torch.stack([_flat(st, keys) for st in shared_states]).median(0).values
        out = _unflat(med, global_state, keys)
        out.update({k: v.clone() for k, v in global_state.items() if k not in out})
        return out
    if method == "krum":                        # Blanchard et al. 2017: keep the single update
        deltas = torch.stack([_flat(st, keys) - _flat(global_state, keys) for st in shared_states])
        n = len(deltas)
        m = max(1, n - krum_f - 2)
        d2 = torch.cdist(deltas, deltas) ** 2
        scores = torch.stack([d2[i][torch.arange(n) != i].sort().values[:m].sum() for i in range(n)])
        best = int(scores.argmin())
        out = {k: v.clone() for k, v in shared_states[best].items()}
        return out
    # reputation-weighted: similarity of each site's update to the
    # coordinate-wise median update, smoothed over rounds; weight =
    # reputation x data size; a site's reputation never drops below the floor.
    keys = list(global_state)
    deltas = [torch.cat([(st[k] - global_state[k]).flatten().float() for k in keys]) for st in shared_states]
    median = torch.stack(deltas).median(0).values
    for s, d in zip(sites, deltas):
        sim = torch.nn.functional.cosine_similarity(d, median, dim=0).item()
        s.reputation = max(rep_floor, rep_beta * s.reputation + (1 - rep_beta) * max(sim, 0.0))
    return fedavg(shared_states, [s.reputation * n for s, n in zip(sites, sizes)])


def _prox(model, prefix, ref, mu):
    if ref is None or mu == 0:
        return 0.0
    return 0.5 * mu * sum(((p - ref[n]) ** 2).sum() for n, p in model.named_parameters()
                          if n.startswith(prefix) and n in ref)


def n_params(state):
    return int(sum(v.numel() for v in state.values() if torch.is_floating_point(v)))


# ---------------------------------------------------------------- training
def poison(state, global_state, scale):
    """Model poisoning: submit the local update flipped and scaled by -scale."""
    return {k: (global_state[k] - scale * (v - global_state[k])) if torch.is_floating_point(v) else v
            for k, v in state.items()}


def run_federated(data, method, seed, n_sites=6, rounds_a=3, pre_rounds=2, rounds_b=6,
                  local_epochs=1, mu=0.01, batch=1024, lr=1e-3, log=print,
                  malicious=(), attack_scale=0.0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_cls = len(data.classes)
    site_of, seg_site = assign_sites(data, n_sites)
    ids = data.ids
    sites = []
    for k in range(n_sites):
        m = site_of == k
        sites.append(Site(k, np.intersect1d(ids["train"], np.flatnonzero(m)),
                          np.intersect1d(ids["val"], np.flatnonzero(m)),
                          np.intersect1d(ids["test"], np.flatnonzero(m))))
        sites[-1].n_train = len(sites[-1].train)

    node_dim, edge_dim = data.ego["node_feat"].shape[1], data.ego["edge_attr"].shape[1]
    comm = {"lc_params": 0, "ce_params": 0}

    # ---------------- stage A: LC ------------------------------------------
    torch.manual_seed(seed)
    init = LCClassifier(node_dim, edge_dim, 64, n_cls).state_dict()
    for s in sites:
        s.lc_model = LCClassifier(node_dim, edge_dim, 64, n_cls)
        s.lc_model.load_state_dict(init)
    global_lc = _shared(init, "lc.")
    comm["lc_params"] = n_params(global_lc)
    for r in range(rounds_a):
        states = []
        for s in sites:
            model = s.lc_model
            if method != "isolated":
                model.load_state_dict(global_lc, strict=False)
            ref = {n: p.detach().clone() for n, p in model.named_parameters()} if method == "fedprox" else None
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.CrossEntropyLoss(weight=class_weights(data.y_multi[s.train], n_cls))
            model.train()
            for _ in range(local_epochs):
                perm = rng.permutation(s.train)
                for i in range(0, len(perm), batch):
                    b = perm[i:i + batch]
                    out = model(collate(data.ego, b), collate(data.ego, lc_batches(data.windows, b)))
                    loss = loss_fn(out, torch.from_numpy(data.y_multi[b])) + _prox(model, "lc.", ref, mu)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            st = _shared(model.state_dict(), "lc.")
            states.append(poison(st, global_lc, attack_scale) if s.k in malicious else st)
        new = aggregate(method, sites, states, global_lc)
        if new is not None:
            global_lc = new
        log(f"  stage A round {r + 1}/{rounds_a}")
    if method != "isolated":
        for s in sites:
            s.lc_model.load_state_dict(global_lc, strict=False)

    # h_L for every window, from each site's LC (identical across sites unless isolated)
    all_ids = np.arange(len(data.windows))
    if method == "isolated":
        h_by_site = {s.k: torch.from_numpy(predict_lc(s.lc_model, data.ego, data.windows, all_ids, embed=True))
                     for s in sites}
    else:
        h = torch.from_numpy(predict_lc(sites[0].lc_model, data.ego, data.windows, all_ids, embed=True))
        h_by_site = {s.k: h for s in sites}

    # ---------------- stage B: CE shared, GC/fusion/classifier local ----------
    torch.manual_seed(seed + 1)
    init = SupervisedHead(64, data.cfg["seq_len"], data.cfg["pattern_len"], n_cls).state_dict()
    for s in sites:
        s.head = SupervisedHead(64, data.cfg["seq_len"], data.cfg["pattern_len"], n_cls)
        s.head.load_state_dict(init)
    global_ce = _shared(init, "ce.")
    comm["ce_params"] = n_params(global_ce)

    def local_round(s, supervised):
        model = s.head
        if method != "isolated":
            model.load_state_dict(global_ce, strict=False)
        ref = {n: p.detach().clone() for n, p in model.named_parameters()} if method == "fedprox" else None
        params = model.ce.parameters() if not supervised else model.parameters()
        opt = torch.optim.Adam(params, lr=lr)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights(data.y_multi[s.train], n_cls))
        h_L = h_by_site[s.k]
        model.train()
        for _ in range(local_epochs):
            perm = rng.permutation(s.train)
            for i in range(0, len(perm), batch):
                b = perm[i:i + batch]
                inp = head_inputs(h_L, data.seq_idx, data.act, b)
                if supervised:
                    logits, _ = model(*inp)
                    loss = loss_fn(logits, torch.from_numpy(data.y_multi[b]))
                else:
                    loss = model.ce.masked_loss(inp[1], inp[2])
                loss = loss + _prox(model, "ce.", ref, mu)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return _shared(model.state_dict(), "ce.")

    for phase, n_rounds, supervised in (("CE pretraining", pre_rounds, False),
                                        ("supervised", rounds_b, True)):
        for r in range(n_rounds):
            states = [local_round(s, supervised) for s in sites]
            states = [poison(st, global_ce, attack_scale) if s.k in malicious else st
                      for s, st in zip(sites, states)]
            new = aggregate(method, sites, states, global_ce)
            if new is not None:
                global_ce = new
            log(f"  stage B {phase} round {r + 1}/{n_rounds}")
    if method != "isolated":
        for s in sites:
            s.head.load_state_dict(global_ce, strict=False)

    rounds_total = rounds_a + pre_rounds + rounds_b
    bytes_per_round = {
        "stage_A": 0 if method == "isolated" else 2 * n_sites * comm["lc_params"] * 4,
        "stage_B": 0 if method == "isolated" else 2 * n_sites * comm["ce_params"] * 4,
    }
    mean_bytes = (rounds_a * bytes_per_round["stage_A"]
                  + (pre_rounds + rounds_b) * bytes_per_round["stage_B"]) / rounds_total
    return {"sites": sites, "h_by_site": h_by_site, "seg_site": seg_site, "site_of": site_of,
            "comm": comm, "bytes_per_round": bytes_per_round, "mean_bytes_per_round": mean_bytes,
            "rounds": rounds_total, "reputations": [s.reputation for s in sites]}


def site_scores(data, run, s, ids):
    prob, _ = predict_head(s.head, run["h_by_site"][s.k], data.seq_idx, data.act, ids)
    return prob
