"""Phase 6: main detection experiment (and the protocol-blind stress test).

    python scripts/phase6_main.py                       # main table, 5 seeds
    python scripts/phase6_main.py --stress --seeds 0 1 2

Every model's decision threshold is the one that maximises F1 on S_val; the
test windows (S_test) are scored once with that threshold. Reported per
seed and as mean +/- std.

--stress zeroes every feature that names the protocol (protocol labels,
ports, MQTT / Modbus / HTTP / DNS fields) in all inputs and retrains, to
measure how much detection survives without the protocol shortcut.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeClassifier

from sentinelfed.detector import best_f1_threshold
from sentinelfed.pipeline import (PROTOCOL_IDENTITY, binary_metrics, fit_dual, load_phase4,
                                  record_feature_names)
from sentinelfed.runlog import environment
from sentinelfed.train import StaticGNNClassifier, predict_lc, train_lc

OCSVM_SAMPLE = 10_000


def evaluate(rows, name, seed, data, score_val, score_test, multi_pred_test=None, threshold=None,
             note=""):
    va, te = data.ids["val"], data.ids["test"]
    if threshold is None:
        threshold, _ = best_f1_threshold(data.y[va], score_val)
    m = binary_metrics(data.y[te], score_test, threshold)
    m["macro_F1_attack_type"] = (f1_score(data.y_multi[te], multi_pred_test, average="macro")
                                 if multi_pred_test is not None else np.nan)
    m["val_F1"] = binary_metrics(data.y[va], score_val, threshold)["F1"]
    rows.append({"Model": name, "seed": seed, "threshold": threshold, **m, "note": note})
    print(f"  {name:48s} seed {seed}: test F1 {m['F1']:.4f}  PR-AUC {m['PR_AUC']:.4f}  "
          f"attack-type macro-F1 {m['macro_F1_attack_type']:.4f}", flush=True)


def window_feature_names():
    base = record_feature_names()
    return [f"sent:{n}" for n in base] + [f"recv:{n}" for n in base] + \
        ["log_n_sent", "log_n_received", "log_n_peers"]


def trivial_baselines(rows, data, seed):
    """Deterministic references from the sanity check (same for every seed)."""
    X, ids, y = data.center, data.ids, data.y
    tr, va, te = ids["train"], ids["val"], ids["test"]
    names = window_feature_names()
    best = (-1, None, None)
    for i in range(X.shape[1]):
        if np.ptp(X[tr, i]) == 0:
            continue
        t = DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=0)
        t.fit(X[tr][:, [i]], y[tr])
        f1 = f1_score(y[va], t.predict(X[va][:, [i]]))
        if f1 > best[0]:
            best = (f1, i, t)
    _, i, t = best
    tree_te = t.predict_proba(X[te][:, [i]])[:, 1]
    evaluate(rows, f"Depth-1 tree, best single feature ({names[i]})", seed, data,
             t.predict_proba(X[va][:, [i]])[:, 1], tree_te, note="deterministic")
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[tr], data.y_multi[tr])
    lr_prob = lr.predict_proba(X[te])
    evaluate(rows, "Logistic regression (window features)", seed, data,
             1 - lr.predict_proba(X[va])[:, 0], 1 - lr_prob[:, 0],
             lr.predict(X[te]), note="deterministic")
    return {"tree_score": tree_te, "lr_prob": lr_prob}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase4", default="data/phase4")
    ap.add_argument("--results", default="results")
    ap.add_argument("--scores", default="data/phase6")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--stress", action="store_true",
                    help="protocol-blind, trivial baselines + dual model only (original stress test)")
    ap.add_argument("--protocol-blind", action="store_true",
                    help="protocol-blind features for every model, all baselines")
    args = ap.parse_args()
    blind = args.stress or args.protocol_blind
    tag = "_stress" if args.stress else ("_protocol_blind" if args.protocol_blind else "")
    res, score_dir = Path(args.results), Path(args.scores + tag)
    score_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    data = load_phase4(args.phase4, drop=PROTOCOL_IDENTITY if blind else None)
    if blind:
        print(f"protocol-blind: zeroed {len(data.dropped)} record features: {data.dropped}")
    ids = data.ids
    va, te = ids["val"], ids["test"]
    rows = []
    seedwise_name = {"_stress": "stress_test_seedwise.csv",
                     "_protocol_blind": "experiment1_seedwise_protocol_blind.csv",
                     "": "experiment1_seedwise.csv"}[tag]

    trivial = trivial_baselines(rows, data, seed=args.seeds[0])
    np.savez_compressed(score_dir / "trivial_baselines.npz", **trivial)
    for seed in args.seeds:
        print(f"=== seed {seed} ({time.time() - t0:.0f}s)", flush=True)
        saved = {"window_id": data.windows["window_id"].to_numpy()[te], "y": data.y[te],
                 "y_multi": data.y_multi[te]}

        if not args.stress:
            iso = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1)
            iso.fit(data.center[ids["u_train"]])
            s_va, s_te = -iso.score_samples(data.center[va]), -iso.score_samples(data.center[te])
            evaluate(rows, "Isolation Forest", seed, data, s_va, s_te)
            saved["isolation_forest"] = s_te

            rng = np.random.default_rng(seed)
            sample = rng.choice(ids["u_train"], min(OCSVM_SAMPLE, len(ids["u_train"])), replace=False)
            oc = OneClassSVM(nu=0.01, gamma="scale").fit(data.center[sample])
            s_va, s_te = -oc.decision_function(data.center[va]), -oc.decision_function(data.center[te])
            evaluate(rows, "One-Class SVM", seed, data, s_va, s_te,
                     note=f"fit on {len(sample):,} sampled U_train windows")
            saved["one_class_svm"] = s_te

            gnn, _ = train_lc(data.ego, data.windows, data.y_multi, ids["train"], va,
                              len(data.classes), epochs=3, seed=seed, model_cls=StaticGNNClassifier)
            p_va = predict_lc(gnn, data.ego, data.windows, va)
            p_te = predict_lc(gnn, data.ego, data.windows, te)
            evaluate(rows, "Static graph baseline (GINE)", seed, data, 1 - p_va[:, 0], 1 - p_te[:, 0],
                     p_te.argmax(1))
            saved["static_gnn"] = 1 - p_te[:, 0]
            saved["static_gnn_prob"] = p_te

        model = fit_dual(data, seed=seed, log=lambda m: None)
        sv, st = model.score(va), model.score(te)
        evaluate(rows, "Unsupervised branch alone (GRU-AE + Mahalanobis)", seed, data,
                 sv["d_norm"], st["d_norm"])
        evaluate(rows, "Supervised branch alone (LC+CE+GC+Fusion)", seed, data,
                 sv["p_sup"], st["p_sup"], st["prob"].argmax(1))
        evaluate(rows, "Full SentinelFed-J (Dual Head, proposed)", seed, data,
                 sv["risk"], st["risk"], st["prob"].argmax(1),
                 note=f"risk mix weight on supervised = {model.risk.alpha:.2f}")
        saved.update({"p_sup": st["p_sup"], "d_norm": st["d_norm"], "risk": st["risk"],
                      "prob": st["prob"], "fusion_weights": st["fusion_weights"],
                      "tau_sup": model.tau_sup, "tau_unsup": model.tau_unsup,
                      "val_p_sup": sv["p_sup"], "val_d_norm": sv["d_norm"], "val_risk": sv["risk"],
                      "val_y": data.y[va], "val_y_multi": data.y_multi[va]})
        saved["thresholds"] = json.dumps({r["Model"]: r["threshold"] for r in rows if r["seed"] == seed})
        np.savez_compressed(score_dir / f"test_scores_seed{seed}.npz", **saved)
        pd.DataFrame(rows).to_csv(res / seedwise_name, index=False)

    seedwise = pd.DataFrame(rows)
    metrics = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC_AUC", "PR_AUC",
               "macro_F1_attack_type"]
    order = list(dict.fromkeys(seedwise["Model"]))
    agg = seedwise.groupby("Model", sort=False)[metrics].agg(["mean", "std"]).reindex(order)
    table = pd.DataFrame({"Model": order, "seeds": seedwise.groupby("Model").size().reindex(order).values})
    for m in metrics:
        mean, std = agg[(m, "mean")], agg[(m, "std")].fillna(0.0)
        table[m] = [("n/a" if np.isnan(a) else f"{a:.4f} ± {b:.4f}") for a, b in zip(mean, std)]
    name = {"_stress": "stress_test_results.csv", "_protocol_blind": "experiment1_results_protocol_blind.csv",
            "": "experiment1_results.csv"}[tag]
    table.to_csv(res / name, index=False)
    seedwise.to_csv(res / seedwise_name, index=False)
    (res / f"run_log_phase6{tag}.json").write_text(json.dumps({
        **environment(), "seeds": args.seeds, "mode": tag or "all features",
        "dropped_features": data.dropped,
        "threshold_rule": "best F1 on S_val per model and seed", "seconds": round(time.time() - t0)},
        indent=2))
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print(table.to_string(index=False))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
