# SentinelFed-J: hand-in map

Each section follows the guide. "File" means a path in this repository unless
marked *(rebuilt)*: those are large intermediate files that are git-ignored and
recreated by the script named. `results/run_environment.json` records the commit
that produced every results file, plus the data version, hardware and software.

## Final checklist (guide, Part C)

| Item | Status | Where |
|---|---|---|
| Dataset audit, with the exact version recorded | Done | `results/dataset_audit.csv` (file name, size, SHA-256, download date) |
| Split cut-point timestamps and row counts verified | Done | `results/split_cutpoints.csv`, `results/split_audit.csv`, `results/split_checks.csv` |
| Confirmation of no preprocessing leakage | Done | `results/split_checks.csv`: all four mandatory checks pass; `tests/test_phase4.py` tests causality and split boundaries |
| Main comparison over multiple seeds | Done | `results/experiment1_results_protocol_blind.csv` (5 seeds, all baselines on the same inputs); `results/experiment1_results.csv` (all-features model, 5 seeds) |
| Ablation study | Done | `results/ablation_results.csv`, `results/ablation_increments.csv` (5 seeds) |
| ECE / Brier / reliability analysis | Done | `results/calibration_results.csv`, `results/reliability_diagram.png` |
| Per-attack results | Done | `results/per_attack_results.csv` (classes under 500 test windows flagged) |
| Sequential-vs-random leakage experiment | Done | `results/leakage_results.csv`, `results/leakage_comparison.png` |
| Class-imbalance and sensitivity studies | Done | `results/imbalance_results.csv`, `results/sparsity_results.csv`, `results/sensitivity_results.csv` |
| Federated comparison | Done | `results/federated_results.csv`, `results/federated_sites.csv` |
| Poisoning study, more than one attacker setup | Done | `results/poisoning_results.csv` (1 and 2 attackers, mild and strong) |
| Scalability study | Done | `results/scalability_results.csv` |
| Ledger tamper test | Done | `results/ledger_results.csv`, `results/ledger_tamper_test.json` |
| Every reported number traces to a saved file | Done | per-seed files saved for every multi-seed result; `results/run_environment.json` |

## Statistical reporting checklist

| Item | Status |
|---|---|
| Fixed, documented seeds | Seeds 0-4 (main table, ablation) or 0-2 (later phases), stated in every run log |
| Mean ± std for every metric | Yes; per-seed values saved alongside |
| Paired comparisons on the same test records | `results/paired_mcnemar.csv`, `results/paired_bootstrap_macro_f1.csv`, `results/paired_wilcoxon_seeds.csv` |
| Split cut points, versions, hardware, hyperparameters recorded | `results/run_log_*.json`, `results/run_environment.json`, `results/split_cutpoints.csv` |
| Raw per-seed metrics saved | `*_seedwise.csv`, `phase10_runs.csv`, `phase11_runs.csv`, `phase12_runs.csv` |
| Split code and configuration kept | `scripts/phase3_split.py`, `results/run_log_phase3.json` |
| Final test set kept untouched until the very end | **Not met.** See limitation L1 |

## Output files (guide's list)

All present in `results/`: `dataset_audit.csv`, `split_audit.csv`,
`experiment1_results.csv`, `experiment1_seedwise.csv`, `ablation_results.csv`,
`calibration_results.csv`, `per_attack_results.csv`, `leakage_results.csv`,
`sparsity_results.csv`, `imbalance_results.csv`, `sensitivity_results.csv`,
`federated_results.csv`, `poisoning_results.csv`, `scalability_results.csv`,
`ledger_results.csv`. The protocol-blind main table is
`experiment1_results_protocol_blind.csv` (see D5).

Per-phase files *(rebuilt)*: `processed_data.csv` (Phase 2); the six split
files (Phase 3); `features.csv`, `windows.pkl`, `ego_graphs.pkl`,
`activity_flags.pkl` and the four embedding files (Phase 4); trained model files
(Phase 5, `artifacts/phase5/`).

## Figures (guide's list)

| # | Figure | File |
|---|---|---|
| 1 | Overall architecture | `results/fig1_architecture.png` |
| 2 | LATS split, supervised and unsupervised families | `results/fig2_lats_split.png` (drawn from the real cut points) |
| 3 | LC / CE / GC pipeline | `results/fig3_pipeline.png` (with the real Phase 4 shapes) |
| 4 | Main comparison, mean ± std | `results/fig4_main_comparison.png` |
| 5 | Ablation | `results/ablation.png` |
| 6 | Reliability diagram | `results/reliability_diagram.png` |
| 7 | Per-attack F1 / recall | `results/per_attack_f1_recall.png` |
| 8 | Sequential vs random | `results/leakage_comparison.png` |
| 9 | Federated performance and communication | `results/federated_comparison.png` |
| 10 | Poisoning robustness | `results/poisoning_robustness.png` |
| 11 | Latency vs devices | `results/latency_vs_devices.png` |
| 12 | Hash-chain tampering and ledger overhead | `results/ledger_overhead_tamper.png` |

Also: `risk_distribution.png`, `disagreement_distribution.png`,
`decision_regions.png` (Phase 5) and `sensitivity_window/hop/classifier.png`
(Phase 10).

## Deviations from the guide (state these in the methods section)

- **D1. Two attack classes excluded.** DDoS_UDP and MITM rows have their fields
  shifted between columns in every CSV release of the dataset, so they were
  dropped (Phase 2). 12 attack types remain. Re-extracting them from the `.pcap`
  files would restore them.
- **D2. LATS within each recording.** The CSV timestamps have no month or day,
  and each class was recorded separately, so a single global time cut would put
  whole classes in one split. Each of the 20 recordings is split 70/15/15 on its
  own clock; all four mandatory checks hold (Phase 3).
- **D3. Unit of analysis.** A device-window of 3 one-second slots, not a single
  record.
- **D4. LC compares a window with the previous one,** not the next, so no
  future data is used.
- **D5. Protocol-blind main model.** Protocol labels, ports and all MQTT,
  Modbus, HTTP and DNS fields are zeroed in every input. This was decided after
  seeing Phase 6 test results (see L1). The all-features model is reported
  alongside.
- **D6. Two-stage supervised training** (LC first, then CE / GC / fusion /
  classifier) instead of fully end-to-end, for CPU cost.
- **D7. Extra headline metric.** Attack-type macro-F1 is reported because
  attack-vs-normal detection is near ceiling on this data (sanity check).
- **D8. Federated sites** are defined by recordings and time blocks, since the
  data has only four real LAN hosts. The rule is in
  `src/sentinelfed/federated.py:assign_sites` and the result in
  `results/federated_sites.csv`. Classifiers stay local (the guide shares only
  LC and CE), so a site cannot name attack types it never saw.
- **D9. Scalability on replayed traffic.** Networks of 10-500 devices are built
  by replaying real normal sensor/broker traffic, one broker per ten devices.
- **D10. Hop study stored in float16** to fit in memory; a k = 1 float16 control
  matches float32 (0.955 vs 0.954).

## Limitations (state these in the paper)

- **L1. The test set influenced decisions.** The protocol-blind model became
  the main model after its Phase 6 test scores were seen; on validation the
  all-features model scored slightly higher (0.990 vs about 0.98 macro-F1).
  S_test was then used in Phases 6-14, and no unused data is left for a final
  confirmatory check.
- **L2. Detection is near trivial on this dataset.** Normal traffic is MQTT
  sensor telemetry and attacks come from a separate attacker host. One feature
  (packet count, or "contains MQTT") reaches F1 0.98-0.99. Claims should rest on
  attack-type identification, not detection F1.
- **L3. Test periods are minutes after training,** within the same recording,
  so the split cannot measure drift over days.
- **L4. Small classes.** Six attack types have under 500 test windows
  (Uploading: 52), flagged in `per_attack_results.csv`.
- **L5. The dual head adds nothing.** The unsupervised branch never changed a
  prediction (risk mix weight 0.95-1.0). The guide's Verify rule sends 18% of
  test windows to a person, almost all of them attacks the supervised branch
  already identifies correctly.
- **L6. GC contributes almost nothing** (1-3% fusion weight; 0.40 macro-F1
  alone), because most windows involve only two devices.
- **L7. The guide's reputation rule does not resist poisoning** here. It does
  about as badly as FedAvg because reputation decays slowly over 3 LC rounds and
  is multiplied by the attackers' large data share. Krum is unaffected. A
  stronger rule should be specified in advance and tested on a new attack setup,
  not tuned on these results.
- **L8. Federated training time** comes from a sequential simulation on one
  machine, not real networked sites.
- **L9. Phase 10 latency** was timed on normal-traffic windows only and is
  noise-level; build time is the fair graph-construction cost.
- **L10. "Phishing" framing.** Edge-IIoTset has no phishing class. The paper
  must explain how the attack types relate to phishing-style device behaviour.
