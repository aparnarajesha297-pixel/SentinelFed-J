"""Phase 12: poisoning robustness of the federated setup.

    python scripts/phase12_poisoning.py            # resumable
    python scripts/phase12_poisoning.py --report

Attack: malicious sites submit their LC/CE update flipped and scaled
(model poisoning) in every round: mild = x(-1), strong = x(-10).
Attackers are fixed, worst case for size-weighted averaging: site 3 (the
largest) for one attacker, sites 3 and 4 (together over half the training
data) for two. Aggregators: FedAvg, coordinate-wise median, Krum (f=2),
reputation-weighted. Metrics are computed on honest sites only:
global attack-vs-normal F1 clean vs poisoned, and "attacks newly missed" =
share of test attacks the clean run detected that the poisoned run misses
(same aggregator, seed and site).
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sentinelfed.detector import best_f1_threshold  # noqa: E402
from sentinelfed.federated import run_federated, site_scores  # noqa: E402
from sentinelfed.pipeline import PROTOCOL_IDENTITY, binary_metrics, load_phase4  # noqa: E402
from sentinelfed.runlog import environment  # noqa: E402

RUNS = "phase12_runs.csv"
AGGREGATORS = {"fedavg": "FedAvg", "median": "Coordinate-wise median", "krum": "Krum",
               "reputation": "Proposed reputation-weighted"}
CONFIGS = [("clean", (), 0.0), ("1 malicious, mild", (3,), 1.0), ("1 malicious, strong", (3,), 10.0),
           ("2 malicious, mild", (3, 4), 1.0), ("2 malicious, strong", (3, 4), 10.0)]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_runs(res):
    p = res / RUNS
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["aggregator", "config", "seed", "site"])


def run_all(args, res, store):
    data = load_phase4(args.phase4, drop=PROTOCOL_IDENTITY)
    te = data.ids["test"]
    attack = data.y[te] == 1
    for seed in args.seeds:
        for agg in args.aggregators:
            for config, bad, scale in CONFIGS:
                r = load_runs(res)
                if ((r.aggregator == agg) & (r.config == config) & (r.seed == seed)).any():
                    continue
                t0 = time.time()
                run = run_federated(data, agg, seed, malicious=bad, attack_scale=scale, log=lambda m: None)
                secs = time.time() - t0
                rows = []
                for s in run["sites"]:
                    pv = site_scores(data, run, s, s.val)
                    thr, _ = best_f1_threshold(data.y[s.val], 1 - pv[:, 0])
                    pg = 1 - site_scores(data, run, s, te)[:, 0]
                    m = binary_metrics(data.y[te], pg, thr)
                    detected = pg >= thr
                    clean_file = store / f"clean_{agg}_seed{seed}_site{s.k}.npy"
                    if config == "clean":
                        np.save(clean_file, detected)
                        newly = 0.0
                    else:
                        cd = np.load(clean_file)
                        base = cd & attack
                        newly = float((base & ~detected).sum() / max(base.sum(), 1))
                    rows.append({"aggregator": agg, "config": config, "seed": seed, "site": s.k,
                                 "malicious": s.k in bad, "F1": m["F1"], "Recall": m["Recall"],
                                 "Precision": m["Precision"], "newly_missed": newly,
                                 "reputation": s.reputation, "train_seconds": secs})
                pd.concat([load_runs(res), pd.DataFrame(rows)], ignore_index=True).to_csv(res / RUNS, index=False)
                d = pd.DataFrame(rows)
                h = d[~d.malicious]
                print(f"{agg:10s} {config:20s} seed {seed}: honest-site F1 {h.F1.mean():.4f}  "
                      f"newly missed {100 * h.newly_missed.mean():.1f}%  ({secs:.0f}s)", flush=True)


def report(res):
    r = load_runs(res)
    rows = []
    for config, bad, scale in CONFIGS[1:]:
        for agg, label in AGGREGATORS.items():
            honest = ~r.site.isin(bad)
            clean = r[(r.aggregator == agg) & (r.config == "clean") & honest]
            pois = r[(r.aggregator == agg) & (r.config == config) & honest]
            if not len(pois):
                continue
            cs = clean.groupby("seed").F1.mean()
            ps = pois.groupby("seed").F1.mean()
            common = cs.index.intersection(ps.index)
            delta = ps[common] - cs[common]
            nm = pois.groupby("seed").newly_missed.mean()
            rep_bad = r[(r.aggregator == agg) & (r.config == config) & r.site.isin(bad)].reputation.mean()
            rows.append({"Attacker configuration": config, "Aggregator": label,
                         "F1 clean": f"{cs[common].mean():.4f} ± {cs[common].std():.4f}",
                         "F1 poisoned": f"{ps[common].mean():.4f} ± {ps[common].std():.4f}",
                         "ΔF1": f"{delta.mean():+.4f} ± {delta.std():.4f}",
                         "Attacks newly missed (%)": f"{100 * nm.mean():.1f} ± {100 * nm.std():.1f}",
                         "Final reputation of attackers": f"{rep_bad:.3f}" if agg == "reputation" else "n/a",
                         "seeds": len(common), "_d": delta.mean(), "_ds": delta.std(),
                         "_nm": 100 * nm.mean(), "_nms": 100 * nm.std()})
    t = pd.DataFrame(rows)
    t.drop(columns=[c for c in t if c.startswith("_")]).to_csv(res / "poisoning_results.csv", index=False)

    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.0), dpi=150, sharey=True)
    aggs = list(AGGREGATORS.values())
    y = np.arange(len(aggs))[::-1]
    offs = np.linspace(-0.27, 0.27, 4)
    for (config, _, _), off, c in zip(CONFIGS[1:], offs, SERIES):
        d = t[t["Attacker configuration"] == config].set_index("Aggregator").reindex(aggs)
        a.errorbar(d["_d"], y + off, xerr=d["_ds"], fmt="o", color=c, markersize=5, capsize=2,
                   markeredgecolor="white", label=config)
        b.errorbar(d["_nm"], y + off, xerr=d["_nms"], fmt="o", color=c, markersize=5, capsize=2,
                   markeredgecolor="white")
    a.axvline(0, color=MUTED, linewidth=1)
    a.set_yticks(y, aggs, fontsize=8.5)
    a.set_xlabel("ΔF1 on honest sites (poisoned − clean)")
    b.set_xlabel("Attacks newly missed (%)")
    for ax in (a, b):
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED)
    a.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(1.05, -0.17), ncol=4)
    fig.suptitle("Poisoning robustness: sign-flipped updates from 1 or 2 malicious sites", fontsize=10,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(res / "poisoning_robustness.png")
    plt.close(fig)
    (res / "run_log_phase12.json").write_text(json.dumps({
        **environment(), "attack": "model poisoning: submitted LC/CE update = global - scale x (local - global)",
        "scales": {"mild": 1.0, "strong": 10.0}, "malicious_sites": {"1": [3], "2": [3, 4]},
        "krum_f": 2, "reputation": {"beta": 0.8, "floor": 0.05},
        "metrics_on": "honest sites only; threshold per site from its own validation windows",
        "seeds": sorted(r.seed.unique().tolist())}, indent=2, default=str))
    with pd.option_context("display.width", 250):
        print(t.drop(columns=[c for c in t if c.startswith("_")]).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase4", default="data/phase4")
    ap.add_argument("--results", default="results")
    ap.add_argument("--store", default="data/phase12")
    ap.add_argument("--aggregators", nargs="+", default=list(AGGREGATORS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    res, store = Path(args.results), Path(args.store)
    store.mkdir(parents=True, exist_ok=True)
    if not args.report:
        run_all(args, res, store)
    report(res)


if __name__ == "__main__":
    main()
