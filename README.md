# SentinelFed-J

Federated intrusion detection for IIoT networks, in PyTorch.

Each IIoT site trains an intrusion classifier on its own traffic and sends
only model weights to an aggregation server, so raw packets never leave the
site. The server combines the weights with FedAvg. This repo simulates the
whole thing on one machine: it splits a dataset across virtual clients
(optionally non-IID, so each site sees a different mix of attacks), runs the
training rounds, and reports detection metrics after every round.

## Quick start

```bash
pip install -e ".[dev]"
python -m sentinelfed.simulate                      # 20 rounds on synthetic data
python -m sentinelfed.simulate federation.rounds=5 federation.dirichlet_alpha=0.1
pytest
```

You can override any config key from the command line with `section.key=value`.
Each run writes `config.json` and `history.json` under `runs/<timestamp>/`.

The default config uses synthetic, imbalanced traffic (60% benign, five attack
classes), so it runs without downloading anything. With the default seed, 20
rounds across 10 clients reach about 0.88 accuracy and 0.80 macro-F1.

## Using a real dataset

Put a CSV in `data/` and switch the source:

```bash
python -m sentinelfed.simulate data.source=csv data.csv_path=data/Edge-IIoTset.csv \
    data.label_column=Attack_type "data.drop_columns=[Attack_label, ip.src_host, ip.dst_host]"
```

`data/README.md` lists datasets that work and which columns to drop.

## Layout

```
configs/default.yaml     experiment settings
src/sentinelfed/
  data.py                loading, scaling, IID / Dirichlet client partitioning
  model.py               MLP classifier (LayerNorm, see below)
  client.py              local training on one site's shard
  server.py              FedAvg aggregation
  metrics.py             accuracy, macro P/R/F1, detection rate, false alarm rate
  simulate.py            round loop and CLI
tests/                   unit tests plus a short end-to-end run
```

## Metrics

Besides the usual multi-class scores, every round reports two numbers from
the IDS point of view. Detection rate (DR) is the share of attack samples
flagged as any attack. False alarm rate (FAR) is the share of benign samples
flagged. A low FAR matters a lot in industrial settings, because operators
stop trusting alerts quickly.

## Design notes

- The model uses LayerNorm, not BatchNorm. In the first version, BatchNorm's
  running statistics were averaged across clients with very different class
  mixes, and the global model learned much more slowly.
- The feature scaler is fit on the pooled training split. That's fine for a
  simulation, but a real deployment needs shared or federated normalisation
  statistics.
- `dirichlet_alpha` sets how skewed each site's data is. At 0.1 most clients
  see only one or two attack types. At 100 the split is close to IID.

## Not built yet

Ideas for next steps: secure aggregation or differential privacy on client
updates, robust aggregation (Krum, trimmed mean) against poisoned clients,
FedProx for heterogeneous clients, and per-round communication cost logging.
