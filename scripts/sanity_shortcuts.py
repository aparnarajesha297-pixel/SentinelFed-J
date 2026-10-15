"""Sanity check: how much of the Phase 5 score do trivial models reach?

    python scripts/sanity_shortcuts.py

Same windows, same splits, scored on S_val only (S_test is not touched).
If a depth-1 tree on one feature already scores ~0.99, the full model's
score says more about how separable the dataset is than about LC/CE/GC.
"""

import argparse
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier, export_text

from sentinelfed.edges import OUTCOMES


def feature_names(pre) -> list[str]:
    base = list(pre["preprocessor"].get_feature_names_out())
    base = [n.replace("num__", "").replace("cat__", "") for n in base]
    base += ["log_time_gap", "log_byte_ratio", "first_seen"] + [f"outcome_{o}" for o in OUTCOMES]
    return [f"sent:{n}" for n in base] + [f"recv:{n}" for n in base] + \
        ["log_n_sent", "log_n_received", "log_n_peers"]


def scores(model, X, y_bin, y_multi=None):
    p = model.predict_proba(X)
    if y_multi is None:
        prob = p[:, 1]
        pred = (prob >= 0.5).astype(int)
        return {"F1": f1_score(y_bin, pred), "PR_AUC": average_precision_score(y_bin, prob),
                "macro_F1_multiclass": np.nan}
    prob = 1.0 - p[:, list(model.classes_).index(0)]
    return {"F1": f1_score(y_bin, (prob >= 0.5).astype(int)),
            "PR_AUC": average_precision_score(y_bin, prob),
            "macro_F1_multiclass": f1_score(y_multi, model.predict(X), average="macro")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase4", default="data/phase4")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    res = Path(args.results)

    W = pickle.load(open(Path(args.phase4) / "windows.pkl", "rb"))
    w, X = W["windows"], W["center_feat"]
    names = feature_names(joblib.load("artifacts/preprocess_supervised.joblib"))
    assert len(names) == X.shape[1], (len(names), X.shape)
    classes = ["Normal"] + sorted(c for c in w["attack_type"].unique() if c != "Normal")
    y_multi = w["attack_type"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    y = w["label"].to_numpy().astype(int)
    tr = (w["split"] == "train").to_numpy()
    va = (w["split"] == "val").to_numpy()
    rows = []

    # 1. protocol mix only
    proto = [i for i, n in enumerate(names) if ":protocol_" in n]
    t = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0)
    t.fit(X[tr][:, proto], y_multi[tr])
    rows.append({"model": "Decision tree, depth 3, protocol mix only",
                 "features": len(proto), **scores(t, X[va][:, proto], y[va], y_multi[va])})
    proto_rules = export_text(t, feature_names=[names[i] for i in proto], max_depth=3)

    # 2. every single feature on its own, depth-1 and depth-2 trees (binary)
    single = []
    for i in range(X.shape[1]):
        if np.ptp(X[tr, i]) == 0:
            continue
        for depth in (1, 2):
            s = DecisionTreeClassifier(max_depth=depth, class_weight="balanced", random_state=0)
            s.fit(X[tr][:, [i]], y[tr])
            r = scores(s, X[va][:, [i]], y[va])
            single.append({"feature": names[i], "depth": depth, **r})
    single = pd.DataFrame(single).sort_values("F1", ascending=False)
    single.to_csv(res / "sanity_single_feature.csv", index=False)
    for depth in (1, 2):
        best = single[single.depth == depth].iloc[0]
        rows.append({"model": f"Decision tree, depth {depth}, best single feature "
                              f"({best['feature']})", "features": 1,
                     "F1": best["F1"], "PR_AUC": best["PR_AUC"], "macro_F1_multiclass": np.nan})

    # 3. shallow tree and logistic regression on all window features
    for depth in (3, 6):
        t = DecisionTreeClassifier(max_depth=depth, class_weight="balanced", random_state=0)
        t.fit(X[tr], y_multi[tr])
        rows.append({"model": f"Decision tree, depth {depth}, all window features",
                     "features": X.shape[1], **scores(t, X[va], y[va], y_multi[va])})
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(X[tr], y_multi[tr])
    rows.append({"model": "Logistic regression, all window features",
                 "features": X.shape[1], **scores(lr, X[va], y[va], y_multi[va])})

    # reference: the full model from Phase 5
    p5 = json.loads((res / "phase5_thresholds.json").read_text())["metrics_val"]["supervised"]
    rows.append({"model": "Phase 5 supervised branch (LC+CE+GC+fusion)", "features": X.shape[1],
                 "F1": p5["F1"], "PR_AUC": p5["PR_AUC"],
                 "macro_F1_multiclass": p5["macro_F1_multiclass"]})

    table = pd.DataFrame(rows)
    table.to_csv(res / "sanity_shortcuts.csv", index=False)
    (res / "sanity_protocol_tree.txt").write_text(proto_rules)
    with pd.option_context("display.width", 200, "display.max_colwidth", 90):
        print(table.round(4).to_string(index=False))
        print("\nTop single features (depth 1):")
        print(single[single.depth == 1].head(10).round(4).to_string(index=False))
        print("\nProtocol-only tree:\n" + proto_rules)


if __name__ == "__main__":
    main()
