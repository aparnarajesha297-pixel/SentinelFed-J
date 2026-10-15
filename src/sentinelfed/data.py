"""Loading, preprocessing and client partitioning of IIoT traffic data."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    class_names: list[str]

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    @property
    def n_classes(self) -> int:
        return len(self.class_names)


def load_dataset(cfg: dict, seed: int) -> Dataset:
    if cfg["source"] == "synthetic":
        X, y, names = _synthetic(cfg, seed)
    elif cfg["source"] == "csv":
        X, y, names = _from_csv(cfg)
    else:
        raise ValueError(f"unknown data source {cfg['source']!r}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["test_size"], stratify=y, random_state=seed
    )
    # Fitting the scaler on the pooled training set is a simulation shortcut;
    # real clients would need agreed-upon or federated normalisation stats.
    scaler = StandardScaler().fit(X_train)
    return Dataset(
        scaler.transform(X_train).astype(np.float32),
        y_train.astype(np.int64),
        scaler.transform(X_test).astype(np.float32),
        y_test.astype(np.int64),
        names,
    )


def _synthetic(cfg: dict, seed: int):
    """Imbalanced stand-in for IIoT traffic: class 0 is benign and dominant."""
    n_classes = cfg["n_classes"]
    attack_share = 0.4 / (n_classes - 1)
    weights = [0.6] + [attack_share] * (n_classes - 1)
    X, y = make_classification(
        n_samples=cfg["n_samples"],
        n_features=cfg["n_features"],
        n_informative=cfg["n_features"] // 2,
        n_redundant=cfg["n_features"] // 4,
        n_classes=n_classes,
        n_clusters_per_class=2,
        weights=weights,
        class_sep=1.2,
        random_state=seed,
    )
    names = ["Normal"] + [f"Attack_{i}" for i in range(1, n_classes)]
    if cfg.get("binary"):
        y = (y != 0).astype(int)
        names = ["Normal", "Attack"]
    return X, y, names


def _from_csv(cfg: dict):
    """Load a tabular IDS dataset (Edge-IIoTset, X-IIoTID, TON_IoT, ...)."""
    df = pd.read_csv(cfg["csv_path"], low_memory=False)
    label_col = cfg["label_column"]
    labels = df.pop(label_col).astype(str)
    # Drop columns that are identifiers rather than traffic features.
    df = df.drop(columns=cfg.get("drop_columns", []), errors="ignore")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all").fillna(0.0)

    if cfg.get("binary"):
        y = (labels != cfg["benign_label"]).astype(int).to_numpy()
        names = [cfg["benign_label"], "Attack"]
    else:
        enc = LabelEncoder().fit(labels)
        y = enc.transform(labels)
        names = list(enc.classes_)
    return df.to_numpy(dtype=np.float64), y, names


def partition(y: np.ndarray, num_clients: int, scheme: str, alpha: float,
              seed: int) -> list[np.ndarray]:
    """Split sample indices across clients.

    ``dirichlet`` draws each class's share per client from Dir(alpha), the
    usual way to simulate non-IID devices that each see a few attack types.
    """
    rng = np.random.default_rng(seed)
    if scheme == "iid":
        return [np.sort(p) for p in np.array_split(rng.permutation(len(y)), num_clients)]
    if scheme != "dirichlet":
        raise ValueError(f"unknown partition scheme {scheme!r}")

    shards: list[list[int]] = [[] for _ in range(num_clients)]
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        props = rng.dirichlet([alpha] * num_clients)
        cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
        for shard, part in zip(shards, np.split(idx, cuts)):
            shard.extend(part.tolist())

    # A client with no data would break training; give it a few samples.
    for i, shard in enumerate(shards):
        if not shard:
            donor = max(range(num_clients), key=lambda j: len(shards[j]))
            shards[i] = shards[donor][-10:]
            shards[donor] = shards[donor][:-10]
    return [np.sort(np.array(s, dtype=np.int64)) for s in shards]
