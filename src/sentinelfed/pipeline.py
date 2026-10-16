"""Reusable training and scoring for the experiments (Phases 6-10).

``load_phase4`` reads the Phase 4 artefacts once; ``fit_dual`` trains the
full model for one seed (the same procedure as scripts/phase5_train.py) and
returns an object that scores any set of windows.
"""

import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

from .detector import (EmpiricalScaler, Mahalanobis, RiskModel, best_f1_threshold, latent,
                       train_autoencoder)
from .edges import OUTCOMES
from .train import VIEWS, predict_head, predict_lc, train_head, train_lc

# Everything that names the protocol: protocol labels, ports and the
# application-layer fields that only exist for one protocol.
PROTOCOL_IDENTITY = re.compile(r"^(protocol_|tcp\.srcport|tcp\.dstport|udp\.port|mqtt\.|mbtcp\.|"
                               r"http\.|dns\.)")


def record_feature_names(preprocess_path="artifacts/preprocess_supervised.joblib") -> list[str]:
    pre = joblib.load(preprocess_path)
    base = [n.split("__", 1)[1] for n in pre["preprocessor"].get_feature_names_out()]
    return base + ["log_time_gap", "log_byte_ratio", "first_seen"] + [f"outcome_{o}" for o in OUTCOMES]


@dataclass
class Data:
    windows: object
    ego: dict
    act: dict
    seq_idx: np.ndarray
    center: np.ndarray
    cfg: dict
    classes: list
    y: np.ndarray
    y_multi: np.ndarray
    ids: dict = field(default_factory=dict)
    dropped: list = field(default_factory=list)


def load_phase4(path="data/phase4", drop=None) -> Data:
    """Load windows, graphs and activity. ``drop`` is a regex over record
    feature names; matching features are zeroed in every input."""
    p = Path(path)
    W = pickle.load(open(p / "windows.pkl", "rb"))
    if (p / "ego_graphs").is_dir():     # low-memory format: one .npy per array, memory-mapped
        ego = {k: np.load(p / "ego_graphs" / f"{k}.npy", mmap_mode="r")
               for k in ("node_feat", "edge_index", "edge_attr", "node_ptr", "edge_ptr")}
    else:
        ego = pickle.load(open(p / "ego_graphs.pkl", "rb"))
    act = pickle.load(open(p / "activity_flags.pkl", "rb"))
    windows, center = W["windows"], W["center_feat"]
    dropped = []
    if drop is not None:
        names = record_feature_names()
        E = len(names)
        cols = [i for i, n in enumerate(names) if drop.search(n)]
        dropped = [names[i] for i in cols]
        node_cols = cols + [E + i for i in cols]
        center = center.copy()
        center[:, node_cols] = 0.0
        ego = dict(ego)
        ego["node_feat"] = np.array(ego["node_feat"])
        ego["node_feat"][:, node_cols] = 0.0
        ego["edge_attr"] = np.array(ego["edge_attr"])
        ego["edge_attr"][:, cols] = 0.0
    classes = ["Normal"] + sorted(c for c in windows["attack_type"].unique() if c != "Normal")
    y_multi = windows["attack_type"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    y = windows["label"].to_numpy().astype(int)
    split = windows["split"].to_numpy()
    ids = {s: np.flatnonzero(split == s) for s in ("train", "val", "test")}
    ids["u_train"] = ids["train"][y[ids["train"]] == 0]
    ids["u_val"] = ids["val"][y[ids["val"]] == 0]
    return Data(windows, ego, act, W["seq_idx"], center, W["config"], classes, y, y_multi, ids, dropped)


@dataclass
class DualModel:
    data: Data
    lc: object
    head: object
    ae: object
    maha: Mahalanobis
    scaler: EmpiricalScaler
    h_L: torch.Tensor
    tau_sup: float = 0.5
    tau_unsup: float = 0.99
    risk: RiskModel = None

    def supervised(self, ids):
        prob, w = predict_head(self.head, self.h_L, self.data.seq_idx, self.data.act, ids)
        return prob, w

    def unsupervised(self, ids):
        z = latent(self.ae, self.data.center, self.data.seq_idx, ids)
        return self.scaler(self.maha(z))

    def score(self, ids):
        prob, w = self.supervised(ids)
        p_sup = 1.0 - prob[:, 0]
        d_norm = self.unsupervised(ids)
        out = {"prob": prob, "p_sup": p_sup, "d_norm": d_norm, "fusion_weights": w}
        if self.risk is not None:
            out["risk"] = self.risk(p_sup, d_norm)
        return out


def fit_dual(data: Data, seed=0, views=VIEWS, lc_epochs=3, head_epochs=6, ae_epochs=5,
             log=print, fit_unsupervised=True) -> DualModel:
    torch.manual_seed(seed)
    np.random.seed(seed)
    ids, n_cls = data.ids, len(data.classes)
    ae = maha = scaler = None
    if fit_unsupervised:
        ae, _ = train_autoencoder(data.center, data.seq_idx, ids["u_train"], ids["u_val"],
                                  epochs=ae_epochs, seed=seed, log=log)
        maha = Mahalanobis.fit(latent(ae, data.center, data.seq_idx, ids["u_train"]))
        scaler = EmpiricalScaler(maha(latent(ae, data.center, data.seq_idx, ids["u_val"])))
    lc, _ = train_lc(data.ego, data.windows, data.y_multi, ids["train"], ids["val"], n_cls,
                     epochs=lc_epochs, seed=seed, log=log)
    h_L = torch.from_numpy(predict_lc(lc, data.ego, data.windows,
                                      np.arange(len(data.windows)), embed=True))
    head, _ = train_head(h_L, data.seq_idx, data.act, data.y_multi, ids["train"], ids["val"],
                         n_cls, data.cfg["seq_len"], data.cfg["pattern_len"], views=views,
                         epochs=head_epochs, seed=seed, log=log)
    model = DualModel(data, lc, head, ae, maha, scaler, h_L)
    va = ids["val"]
    prob, _ = model.supervised(va)
    model.tau_sup, _ = best_f1_threshold(data.y[va], 1.0 - prob[:, 0])
    if fit_unsupervised:
        model.risk = RiskModel.fit(1.0 - prob[:, 0], model.unsupervised(va), data.y[va])
    return model


def binary_metrics(y, score, threshold) -> dict:
    pred = (score >= threshold).astype(int)
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return {"Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0),
            "Specificity": tn / max(tn + fp, 1), "F1": f1_score(y, pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y, score), "PR_AUC": average_precision_score(y, score)}
