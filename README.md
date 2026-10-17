# SentinelFed-J

Federated, leakage-aware intrusion detection for IIoT networks, built on
Edge-IIoTset (Ferrag et al., IEEE Access 2022). It follows the supervisor's
student guide, Phases 0-14. Every number reported below traces back to a file
in `results/`, and `results/run_environment.json` names the commit that
produced each file.

**Start with [`HAND_IN.md`](HAND_IN.md).** It maps every item of the guide's
final checklist and output-file list to the file that satisfies it, and lists
every deviation from the guide and every limitation the write-up has to state.

## What the system does

A device's traffic is cut into short windows (3 one-second slots). Three views
describe each window:

- **LC (Local Correlation):** a graph network over the device and its peers in
  the window, with cross-attention to the device's previous window.
- **CE (Contextual Enrichment):** a transformer over the device's last 8 window
  embeddings, pretrained by reconstructing hidden ones.
- **GC (Global Correlation):** cross-attention between the device's activity
  pattern and other devices' patterns in the same period.

A learned fusion combines the three views for a 13-class classifier. A GRU
autoencoder trained only on normal traffic gives a second, unsupervised score.
Both feed a calibrated risk and a Trust / Verify / Reject decision, and each
decision is written to a SHA-256 hash-chained ledger. Sites train together by
sharing only LC and CE.

## Main results (S_test, protocol-blind model)

| | Result | File |
|---|---|---|
| Attack-vs-normal F1 | 0.9989 ± 0.0001 (5 seeds) | `results/experiment1_results_protocol_blind.csv` |
| Attack-type macro-F1 | 0.952 ± 0.004, against 0.910 for a static GNN and 0.862 for logistic regression | same |
| Significance | full model beats every baseline on the same windows in all 5 seeds (McNemar; bootstrap CIs exclude 0) | `results/paired_*.csv` |
| Leakage gap | a random split would inflate macro-F1 by 3.4 points | `results/leakage_results.csv` |
| Federated, 6 non-IID sites | FedAvg global F1 0.970; isolated training 0.804 | `results/federated_results.csv` |
| Poisoning | Krum unaffected; the guide's reputation rule does **not** protect | `results/poisoning_results.csv` |
| Scalability | 500 devices processed in 73 ms per second of traffic on one core | `results/scalability_results.csv` |
| Ledger | tampering detected at the altered record; +16% latency vs a plain log | `results/ledger_results.csv` |

Attack-vs-normal detection is close to trivial on this dataset: one feature
already reaches F1 0.99 (`results/sanity_shortcuts.csv`). Attack-type
identification is where the methods differ. See `HAND_IN.md` before quoting any
of these numbers.

## Reproducing

Needs Python 3.10+, about 15 GB of RAM and a Kaggle API token (for the dataset
download). Everything runs on CPU.

```bash
pip install -e ".[dev]" torch-geometric xgboost lightgbm matplotlib
# put DNN-EdgeIIoT-dataset.csv in data/ (Kaggle: mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot)
python scripts/phase1_audit.py
python scripts/phase2_clean.py
python scripts/phase3_split.py
python scripts/phase4_build.py
python scripts/phase5_train.py
python scripts/sanity_shortcuts.py
python scripts/phase6_main.py --protocol-blind     # main table (also: no flag = all features)
python scripts/paired_tests.py
python scripts/phase7_ablation.py
python scripts/phase6_main.py --stress --seeds 0 1 2   # scores reused by Phases 8 and 9
python scripts/phase8_calibration.py
python scripts/phase9_per_attack.py
./scripts/phase10_run_all.sh                       # resumable; several hours
python scripts/phase11_federated.py
python scripts/phase12_poisoning.py                # resumable; about 5 hours
python scripts/phase14_ledger.py                   # run 13 and 14 on an idle machine
python scripts/phase13_scalability.py
python scripts/figures.py && python scripts/provenance.py
pytest
```

Large intermediate files (`data/`, `artifacts/`) are git-ignored; the scripts
rebuild them. Long scripts append finished runs to a CSV and skip them on
rerun, so they can be restarted after an interruption.

## Layout

```
scripts/phaseN_*.py      one script per guide phase (plus sanity, paired tests, figures, provenance)
src/sentinelfed/
  edges.py               causal per-record edge features
  graphs.py              device-windows, ego graphs, activity patterns, CE history
  modules.py             LC, CE, GC, weighted fusion
  train.py               two-stage supervised training, static GNN baseline
  detector.py            GRU autoencoder, Mahalanobis score, calibration, Trust/Verify/Reject
  pipeline.py            loading Phase 4 artefacts, one-seed training, metrics
  federated.py           sites, FedAvg / FedProx / median / Krum / reputation, poisoning
  runlog.py              code, data, hardware and software versions for run logs
results/                 every table, figure and run log the guide asks for
tests/                   20 tests: causality, split boundaries, graph offsets, policy, aggregators
```

The original synthetic FedAvg scaffold (`simulate.py`, `data.py`, `model.py`,
`client.py`, `metrics.py`) still runs with `python -m sentinelfed.simulate` and
is covered by `tests/test_core.py`; the research pipeline does not use it.
