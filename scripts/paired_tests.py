"""Paired comparisons on the same S_test windows (Part C of the guide).

    python scripts/paired_tests.py

Reference model: Full SentinelFed-J, protocol-blind. Every comparison uses
the same test windows, per seed:
  * attack-vs-normal: McNemar's exact test on which windows each model gets
    right, at each model's own validation-chosen threshold;
  * attack type: paired bootstrap 95% CI of the macro-F1 difference
    (1,000 resamples of test windows), where both models predict types;
  * across seeds: Wilcoxon signed-rank on per-seed scores (with 5 seeds the
    smallest attainable two-sided p is 0.0625, so it cannot reach 0.05).
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import f1_score

from sentinelfed.runlog import environment

REF = "Full SentinelFed-J (Dual Head, proposed)"
N_BOOT = 1000


def mcnemar(correct_a, correct_b):
    b = int((correct_a & ~correct_b).sum())     # A right, B wrong
    c = int((~correct_a & correct_b).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p


def boot_macro_diff(y, pa, pb, seed):
    rng = np.random.default_rng(seed)
    n = len(y)
    labels = np.unique(y)
    obs = f1_score(y, pa, labels=labels, average="macro") - f1_score(y, pb, labels=labels, average="macro")
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        diffs[i] = (f1_score(y[idx], pa[idx], labels=labels, average="macro", zero_division=0)
                    - f1_score(y[idx], pb[idx], labels=labels, average="macro", zero_division=0))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return obs, lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="data/phase6_protocol_blind")
    ap.add_argument("--all-features", default="data/phase6")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    res = Path(args.results)
    files = sorted(glob.glob(f"{args.scores}/test_scores_seed*.npz"))
    trivial = np.load(f"{args.scores}/trivial_baselines.npz")
    rows, boot_rows = [], []

    for f in files:
        z = np.load(f)
        seed = int(Path(f).stem.split("seed")[1])
        y, ym = z["y"], z["y_multi"]
        thr = json.loads(str(z["thresholds"]))
        tree_name = next(k for k in thr if k.startswith("Depth-1"))
        binary = {
            REF: z["risk"] >= thr[REF],
            "Supervised branch alone (LC+CE+GC+Fusion)": z["p_sup"] >= thr["Supervised branch alone (LC+CE+GC+Fusion)"],
            "Unsupervised branch alone (GRU-AE + Mahalanobis)": z["d_norm"] >= thr["Unsupervised branch alone (GRU-AE + Mahalanobis)"],
            "Static graph baseline (GINE)": z["static_gnn"] >= thr["Static graph baseline (GINE)"],
            "Isolation Forest": z["isolation_forest"] >= thr["Isolation Forest"],
            "One-Class SVM": z["one_class_svm"] >= thr["One-Class SVM"],
            "Logistic regression (window features)": (1 - trivial["lr_prob"][:, 0]) >= thr["Logistic regression (window features)"],
            tree_name: trivial["tree_score"] >= thr[tree_name],
        }
        ref_ok = binary[REF] == (y == 1)
        for name, pred in binary.items():
            if name == REF:
                continue
            b, c, p = mcnemar(ref_ok, pred == (y == 1))
            rows.append({"seed": seed, "comparison": f"{REF} vs {name}",
                         "windows_only_reference_right": b, "windows_only_other_right": c,
                         "mcnemar_p": p})

        types = {"Static graph baseline (GINE)": z["static_gnn_prob"].argmax(1),
                 "Logistic regression (window features)": trivial["lr_prob"].argmax(1)}
        af = Path(args.all_features) / Path(f).name
        if af.exists():
            za = np.load(af)
            assert (za["window_id"] == z["window_id"]).all()
            types["Full model, all features"] = za["prob"].argmax(1)
        ref_types = z["prob"].argmax(1)
        for name, pred in types.items():
            obs, lo, hi = boot_macro_diff(ym, ref_types, pred, seed)
            boot_rows.append({"seed": seed, "comparison": f"{REF} vs {name}",
                              "macro_F1_difference": obs, "ci95_low": lo, "ci95_high": hi,
                              "ci_excludes_zero": bool(lo > 0 or hi < 0)})
        print(f"seed {seed} done", flush=True)

    mc = pd.DataFrame(rows)
    bt = pd.DataFrame(boot_rows)
    summary_mc = mc.groupby("comparison").agg(
        seeds=("seed", "size"),
        mean_only_reference_right=("windows_only_reference_right", "mean"),
        mean_only_other_right=("windows_only_other_right", "mean"),
        max_mcnemar_p=("mcnemar_p", "max"),
        seeds_p_below_0_05=("mcnemar_p", lambda p: int((p < 0.05).sum()))).reset_index()
    summary_bt = bt.groupby("comparison").agg(
        seeds=("seed", "size"), mean_difference=("macro_F1_difference", "mean"),
        min_ci95_low=("ci95_low", "min"), max_ci95_high=("ci95_high", "max"),
        seeds_ci_excludes_zero=("ci_excludes_zero", "sum")).reset_index()

    # across seeds: Wilcoxon on per-seed macro-F1 from the Phase 6 seedwise table
    sw = pd.read_csv(res / "experiment1_seedwise_protocol_blind.csv")
    piv = sw.pivot_table(index="seed", columns="Model", values="macro_F1_attack_type")
    wil = []
    for name in ["Static graph baseline (GINE)"]:
        if name in piv and REF in piv:
            d = piv[REF] - piv[name]
            p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
            wil.append({"comparison": f"{REF} vs {name}", "seeds": len(d),
                        "mean_macro_F1_difference": d.mean(), "wilcoxon_p": p,
                        "note": "with 5 seeds the smallest attainable p is 0.0625"})
    pd.DataFrame(wil).to_csv(res / "paired_wilcoxon_seeds.csv", index=False)

    mc.to_csv(res / "paired_mcnemar_seedwise.csv", index=False)
    bt.to_csv(res / "paired_bootstrap_seedwise.csv", index=False)
    summary_mc.to_csv(res / "paired_mcnemar.csv", index=False)
    summary_bt.to_csv(res / "paired_bootstrap_macro_f1.csv", index=False)
    (res / "run_log_paired_tests.json").write_text(json.dumps({
        **environment(), "scores": args.scores, "bootstrap_resamples": N_BOOT,
        "seeds": sorted(mc["seed"].unique().tolist())}, indent=2, default=str))
    with pd.option_context("display.width", 250, "display.max_colwidth", 95):
        print(summary_mc.round(4).to_string(index=False))
        print(summary_bt.round(4).to_string(index=False))
        print(pd.DataFrame(wil).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
