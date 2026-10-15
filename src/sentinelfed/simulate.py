"""Run a federated training simulation from a YAML config.

    python -m sentinelfed.simulate --config configs/default.yaml federation.rounds=5
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .client import Client
from .config import apply_overrides, load_config
from .data import load_dataset, partition
from .metrics import evaluate
from .model import IDSNet
from .server import fedavg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run(cfg: dict) -> list[dict]:
    seed = cfg["seed"]
    set_seed(seed)
    ds = load_dataset(cfg["data"], seed)
    fed, ccfg = cfg["federation"], cfg["client"]

    shards = partition(ds.y_train, fed["num_clients"], fed["partition"],
                       fed["dirichlet_alpha"], seed)
    clients = [Client(i, ds.X_train[s], ds.y_train[s], ccfg) for i, s in enumerate(shards)]
    benign = ds.class_names.index("Normal") if "Normal" in ds.class_names else 0

    def new_model():
        return IDSNet(ds.n_features, ds.n_classes, cfg["model"]["hidden_dims"],
                      cfg["model"]["dropout"])

    global_model, worker = new_model(), new_model()
    rng = np.random.default_rng(seed)
    n_pick = max(1, round(fed["clients_per_round"] * len(clients)))
    history = []

    print(f"{len(clients)} clients, {ds.n_classes} classes, {ds.n_features} features, "
          f"shard sizes {[c.num_samples for c in clients]}")
    for rnd in range(1, fed["rounds"] + 1):
        start = time.time()
        picked = rng.choice(len(clients), size=n_pick, replace=False)
        global_state = global_model.state_dict()
        updates, sizes = [], []
        for cid in picked:
            c = clients[cid]
            updates.append(c.fit(worker, global_state, seed=seed * 1000 + rnd * 100 + int(cid)))
            sizes.append(c.num_samples)
        global_model.load_state_dict(fedavg(updates, sizes))

        m = evaluate(global_model, ds.X_test, ds.y_test, benign)
        m.update(round=rnd, seconds=round(time.time() - start, 2))
        history.append(m)
        print(f"round {rnd:3d}  acc {m['accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}  "
              f"DR {m['detection_rate']:.4f}  FAR {m['false_alarm_rate']:.4f}")
    return history


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("overrides", nargs="*", help="dotted key=value overrides")
    args = ap.parse_args(argv)

    cfg = apply_overrides(load_config(args.config), args.overrides)
    history = run(cfg)

    out = Path(cfg.get("output_dir", "runs")) / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
