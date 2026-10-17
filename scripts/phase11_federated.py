"""Phase 11: federated learning across non-IID sites (protocol-blind model).

    python scripts/phase11_federated.py              # all methods, 3 seeds (resumable)
    python scripts/phase11_federated.py --report

Each site's personalised model (shared LC/CE + local GC/fusion/classifier)
is scored on
  * the global test set (every site's test windows, including attack types
    the site never saw) -> "Global" columns, averaged over sites;
  * its own test windows -> "Local F1".
Each site picks its decision threshold on its own validation windows.
The centralised upper bound is the Phase 6 protocol-blind supervised branch.
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
from sklearn.metrics import f1_score  # noqa: E402

from sentinelfed.detector import best_f1_threshold  # noqa: E402
from sentinelfed.federated import METHODS, assign_sites, run_federated, site_scores  # noqa: E402
from sentinelfed.pipeline import PROTOCOL_IDENTITY, binary_metrics, load_phase4  # noqa: E402
from sentinelfed.runlog import environment  # noqa: E402

RUNS = "phase11_runs.csv"
LABELS = {"isolated": "Isolated local training", "fedavg": "FedAvg", "fedprox": "FedProx",
          "reputation": "Proposed (reputation-weighted)"}
C1, C2, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"


def load_runs(res):
    p = res / RUNS
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["method", "seed", "site"])


def site_table(data, res):
    site_of, seg_site = assign_sites(data)
    w = data.windows
    rows = []
    for k in sorted(set(site_of)):
        m = site_of == k
        recs = sorted(s for s, v in seg_site.items() if v == k)
        types = sorted(set(w.loc[m & (w["attack_type"] != "Normal").to_numpy(), "attack_type"]))
        rows.append({"site": k, "attack recordings": " ".join(recs), "attack types": ", ".join(types),
                     **{f"{sp} windows": int((m & (w["split"] == sp).to_numpy()).sum())
                        for sp in ("train", "val", "test")},
                     "normal share of train": float((w.loc[m & (w["split"] == "train").to_numpy(),
                                                          "attack_type"] == "Normal").mean())})
    t = pd.DataFrame(rows)
    t.to_csv(res / "federated_sites.csv", index=False)
    return t


def run_all(args, res):
    data = load_phase4(args.phase4, drop=PROTOCOL_IDENTITY)
    print(site_table(data, res).to_string(index=False), flush=True)
    te = data.ids["test"]
    for method in args.methods:
        for seed in args.seeds:
            r = load_runs(res)
            if ((r["method"] == method) & (r["seed"] == seed)).any():
                continue
            t0 = time.time()
            run = run_federated(data, method, seed, log=lambda m: None)
            secs = time.time() - t0
            rows = []
            for s in run["sites"]:
                pv = site_scores(data, run, s, s.val)
                thr, _ = best_f1_threshold(data.y[s.val], 1 - pv[:, 0])
                pg = site_scores(data, run, s, te)
                g = binary_metrics(data.y[te], 1 - pg[:, 0], thr)
                pl = site_scores(data, run, s, s.test)
                loc = binary_metrics(data.y[s.test], 1 - pl[:, 0], thr)
                seen = sorted(set(data.y_multi[s.train]))
                unseen_attack = np.isin(data.y_multi[te], seen, invert=True)
                rows.append({
                    "method": method, "seed": seed, "site": s.k,
                    "global_F1": g["F1"], "global_Recall": g["Recall"], "global_Precision": g["Precision"],
                    "global_macro_F1": f1_score(data.y_multi[te], pg.argmax(1), average="macro"),
                    "recall_on_unseen_attack_types": float(((1 - pg[:, 0]) >= thr)[unseen_attack].mean())
                    if unseen_attack.any() else np.nan,
                    "local_F1": loc["F1"], "classes_seen": len(seen),
                    "reputation": s.reputation, "comm_MB_per_round": run["mean_bytes_per_round"] / 1e6,
                    "comm_MB_total": (run["rounds"] * run["mean_bytes_per_round"]) / 1e6,
                    "train_seconds": secs, "lc_params": run["comm"]["lc_params"],
                    "ce_params": run["comm"]["ce_params"]})
            pd.concat([load_runs(res), pd.DataFrame(rows)], ignore_index=True).to_csv(res / RUNS, index=False)
            d = pd.DataFrame(rows)
            print(f"{method} seed {seed}: global F1 {d.global_F1.mean():.4f}  global recall "
                  f"{d.global_Recall.mean():.4f}  unseen-type recall {d.recall_on_unseen_attack_types.mean():.4f}  "
                  f"local F1 {d.local_F1.mean():.4f}  ({secs:.0f}s)", flush=True)


def report(res):
    r = load_runs(res)
    per_seed = r.groupby(["method", "seed"]).mean(numeric_only=True).reset_index()
    rows = []
    for m in METHODS:
        d = per_seed[per_seed.method == m]
        if not len(d):
            continue
        fmt = lambda c: f"{d[c].mean():.4f} ± {d[c].std(ddof=1) if len(d) > 1 else 0:.4f}"  # noqa: E731
        rows.append({"Method": LABELS[m], "Global F1": fmt("global_F1"), "Global Recall": fmt("global_Recall"),
                     "Global attack-type macro-F1": fmt("global_macro_F1"),
                     "Recall on attack types unseen at the site": fmt("recall_on_unseen_attack_types"),
                     "Local F1": fmt("local_F1"),
                     "Comm. / round (MB)": f"{d.comm_MB_per_round.mean():.2f}",
                     "Training time (s)": f"{d.train_seconds.mean():.0f}", "seeds": len(d),
                     "_f1": d.global_F1.mean(), "_f1s": d.global_F1.std(ddof=1), "_mb": d.comm_MB_per_round.mean(),
                     "_unseen": d.recall_on_unseen_attack_types.mean(),
                     "_unseen_s": d.recall_on_unseen_attack_types.std(ddof=1)})
    # centralised upper bound: Phase 6 protocol-blind supervised branch, Phase 10 baseline time
    sw = pd.read_csv(res / "experiment1_seedwise_protocol_blind.csv")
    c = sw[sw.Model == "Supervised branch alone (LC+CE+GC+Fusion)"]
    p10 = pd.read_csv(res / "phase10_runs.csv")
    t_c = p10[p10.study == "baseline"]["train_seconds"].mean()
    rows.append({"Method": "Centralized (upper bound)",
                 "Global F1": f"{c.F1.mean():.4f} ± {c.F1.std():.4f}",
                 "Global Recall": f"{c.Recall.mean():.4f} ± {c.Recall.std():.4f}",
                 "Global attack-type macro-F1": f"{c.macro_F1_attack_type.mean():.4f} ± {c.macro_F1_attack_type.std():.4f}",
                 "Recall on attack types unseen at the site": "n/a (one site sees everything)",
                 "Local F1": "n/a", "Comm. / round (MB)": "n/a", "Training time (s)": f"{t_c:.0f}",
                 "seeds": len(c), "_f1": c.F1.mean(), "_f1s": c.F1.std(), "_mb": np.nan,
                 "_unseen": np.nan, "_unseen_s": np.nan})
    t = pd.DataFrame(rows)
    t.drop(columns=[x for x in t if x.startswith("_")]).to_csv(res / "federated_results.csv", index=False)

    fed = t[t.Method != "Centralized (upper bound)"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    y = np.arange(len(fed))[::-1]
    a.errorbar(fed["_unseen"], y, xerr=fed["_unseen_s"], fmt="o", color=C1, capsize=3,
               markeredgecolor="white", label="Recall on attack types unseen at the site")
    a.errorbar(fed["_f1"], y, xerr=fed["_f1s"].fillna(0), fmt="s", color=C2, capsize=3,
               markeredgecolor="white", label="Global attack-vs-normal F1")
    cen = t[t.Method == "Centralized (upper bound)"]["_f1"].iloc[0]
    a.axvline(cen, color=MUTED, linestyle="--", linewidth=1)
    a.text(cen, 1.01, "centralized F1 ", fontsize=7.5, color=MUTED, va="bottom", ha="right",
           transform=a.get_xaxis_transform())
    a.set_yticks(y, fed["Method"], fontsize=8.5)
    a.set_xlabel("Score on the global test set (mean ± std over seeds)")
    a.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.45, -0.18), ncol=2)
    b.barh(y, fed["_mb"], color=C1, height=0.5)
    for yi, v in zip(y, fed["_mb"]):
        b.text(v, yi, f" {v:.2f} MB", va="center", fontsize=8, color=INK)
    b.set_yticks(y, [""] * len(fed))
    b.set_xlabel("Communication per round, all sites (MB)")
    b.set_xlim(0, max(fed["_mb"].max() * 1.35, 0.1))
    for ax in (a, b):
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED)
    fig.suptitle("Federated learning across 6 non-IID sites (protocol-blind model)", fontsize=10,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(res / "federated_comparison.png")
    plt.close(fig)
    (res / "run_log_phase11.json").write_text(json.dumps({
        **environment(), "sites": 6, "shared": "LC and CE", "local": "GC, fusion, classifier",
        "rounds": {"stage_A_LC": 3, "stage_B_CE_pretraining": 2, "stage_B_supervised": 6},
        "local_epochs": 1, "fedprox_mu": 0.01, "reputation": {"beta": 0.8, "floor": 0.05},
        "threshold_rule": "each site: best F1 on its own validation windows",
        "seeds": sorted(r["seed"].unique().tolist())}, indent=2, default=str))
    with pd.option_context("display.width", 250, "display.max_colwidth", 40):
        print(t.drop(columns=[x for x in t if x.startswith("_")]).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase4", default="data/phase4")
    ap.add_argument("--results", default="results")
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    res = Path(args.results)
    if not args.report:
        run_all(args, res)
    report(res)


if __name__ == "__main__":
    main()
