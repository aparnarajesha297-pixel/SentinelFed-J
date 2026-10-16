"""Phase 10: training history, leakage, imbalance and sensitivity studies.

    python scripts/phase10_sensitivity.py --study baseline
    python scripts/phase10_sensitivity.py --study history|split|imbalance|classifier
    python scripts/phase10_sensitivity.py --study window --phase4 data/phase4_w4 --setting w4
    python scripts/phase10_sensitivity.py --study hop --phase4 data/phase4_k2 --setting k2
    python scripts/phase10_sensitivity.py --report

All runs use the protocol-blind model and the supervised branch (Phases 6-8
showed the unsupervised branch does not change predictions), 3 seeds, with
thresholds chosen on S_val. Every finished (study, setting, seed) is
appended to results/phase10_runs.csv and skipped on rerun, so the long
sweep can be resumed after the 2-hour background limit.

The baseline (w=3, k=1, full training history, LATS split) is the reference
row of the history, split, window, hop and classifier studies.
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import average_precision_score, f1_score  # noqa: E402

from sentinelfed.detector import best_f1_threshold  # noqa: E402
from sentinelfed.pipeline import PROTOCOL_IDENTITY, binary_metrics, fit_dual, load_phase4  # noqa: E402
from sentinelfed.train import predict_head  # noqa: E402

SEEDS = [int(s) for s in __import__("os").environ.get("P10_SEEDS", "0 1 2").split()]
RUNS = "phase10_runs.csv"
BASE = "w3_k1"
C1, C2, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"


# ---------------------------------------------------------------- bookkeeping
def load_runs(res):
    p = res / RUNS
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["study", "setting", "seed"])


def is_done(res, study, setting, seed):
    r = load_runs(res)
    return ((r["study"] == study) & (r["setting"].astype(str) == str(setting)) & (r["seed"] == seed)).any()


def append(res, rows):
    r = pd.concat([load_runs(res), pd.DataFrame(rows)], ignore_index=True)
    r.to_csv(res / RUNS, index=False)


# ---------------------------------------------------------------- core run
def evaluate(model, data, test_ids, val_ids=None):
    """Binary + attack-type metrics with the threshold chosen on validation."""
    val_ids = data.ids["val"] if val_ids is None else val_ids
    pv, _ = model.supervised(val_ids)
    t0 = time.perf_counter()
    pt, _ = model.supervised(test_ids)
    infer = time.perf_counter() - t0
    thr, _ = best_f1_threshold(data.y[val_ids], 1 - pv[:, 0])
    m = binary_metrics(data.y[test_ids], 1 - pt[:, 0], thr)
    return {"F1": m["F1"], "Precision": m["Precision"], "Recall": m["Recall"], "PR_AUC": m["PR_AUC"],
            "macro_F1": f1_score(data.y_multi[test_ids], pt.argmax(1), average="macro"),
            "val_macro_F1": f1_score(data.y_multi[val_ids], pv.argmax(1), average="macro"),
            "head_infer_ms_per_window": 1000 * infer / len(test_ids)}


def train(data, seed):
    t0 = time.time()
    model = fit_dual(data, seed=seed, fit_unsupervised=False, log=lambda m: None)
    return model, time.time() - t0


def lc_latency_ms(model, data, ids, n=4096):
    """End-to-end scoring latency per window: LC on the ego graphs + head."""
    from sentinelfed.train import predict_lc
    ids = ids[:n]
    t0 = time.perf_counter()
    h = predict_lc(model.lc, data.ego, data.windows, ids, embed=True)
    t1 = time.perf_counter()
    model.supervised(ids)
    t2 = time.perf_counter()
    del h
    return 1000 * (t1 - t0) / len(ids), 1000 * (t2 - t1) / len(ids)


def with_ids(data, **ids):
    new = dict(data.ids)
    new.update(ids)
    return dataclasses.replace(data, ids=new)


# ---------------------------------------------------------------- studies
def study_baseline(args, res, data, setting=BASE, study="baseline", save_emb=True):
    rows = []
    for seed in SEEDS:
        if is_done(res, study, setting, seed):
            continue
        model, secs = train(data, seed)
        m = evaluate(model, data, data.ids["test"])
        lc_ms, head_ms = lc_latency_ms(model, data, data.ids["test"])
        w = data.windows
        tr = data.ids["train"]
        L = data.seq_idx.shape[1]
        row = {"study": study, "setting": setting, "seed": seed, **m, "train_seconds": secs,
               "train_windows": len(tr),
               "near_empty_share": float(((data.seq_idx[tr] >= 0).sum(1) < L / 2).mean()),
               "latency_ms_per_window": lc_ms + head_ms, "lc_ms_per_window": lc_ms,
               "windows": len(w), "mean_nodes": w["n_nodes"].mean(), "mean_edges": w["n_edges"].mean(),
               "build_seconds": args.build_seconds}
        if save_emb:
            _, _, emb = predict_head(model.head, model.h_L, data.seq_idx, data.act,
                                     np.arange(len(w)), embeddings=True)
            Path("data/phase10").mkdir(parents=True, exist_ok=True)
            np.save(f"data/phase10/fused_seed{seed}.npy", emb["fused"].astype(np.float32))
        rows.append(row)
        append(res, [row])
        print(f"{study} {setting} seed {seed}: F1 {m['F1']:.4f} macro-F1 {m['macro_F1']:.4f} "
              f"({secs:.0f}s)", flush=True)


def study_history(args, res, data):
    """Keep only the most recent fraction of each recording's training period."""
    w = data.windows
    tr = data.ids["train"]
    seg = w["segment_id"].to_numpy()
    slot = w["end_slot"].to_numpy()
    L = data.seq_idx.shape[1]
    for frac in (0.25, 0.5):
        setting = f"latest {int(frac * 100)}% of training period"
        keep = np.zeros(len(w), bool)
        for s in np.unique(seg[tr]):
            rows = tr[seg[tr] == s]
            lo, hi = slot[rows].min(), slot[rows].max()
            keep[rows[slot[rows] >= hi - frac * (hi - lo)]] = True
        new_tr = np.flatnonzero(keep)
        dropped = np.zeros(len(w), bool)
        dropped[tr] = True
        dropped[new_tr] = False
        # history that reaches into the dropped period is no longer available
        seq = data.seq_idx.copy()
        seq[np.isin(seq, np.flatnonzero(dropped))] = -1
        win = w.copy()
        prev = win["prev_window"].to_numpy().copy()
        prev[(prev >= 0) & dropped[np.maximum(prev, 0)]] = -1
        win["prev_window"] = prev
        d2 = dataclasses.replace(with_ids(data, train=new_tr), seq_idx=seq, windows=win)
        near_empty = float(((seq[new_tr] >= 0).sum(1) < L / 2).mean())
        for seed in SEEDS:
            if is_done(res, "history", setting, seed):
                continue
            model, secs = train(d2, seed)
            m = evaluate(model, d2, data.ids["test"])
            row = {"study": "history", "setting": setting, "seed": seed, **m, "train_seconds": secs,
                   "train_windows": len(new_tr), "near_empty_share": near_empty}
            append(res, [row])
            print(f"history {setting} seed {seed}: F1 {m['F1']:.4f} macro-F1 {m['macro_F1']:.4f}", flush=True)


def study_split(args, res, data):
    """Random split with the same per-class split sizes as LATS."""
    for seed in SEEDS:
        if is_done(res, "split", "random", seed):
            continue
        rng = np.random.default_rng(seed)
        new = {"train": [], "val": [], "test": []}
        split = data.windows["split"].to_numpy()
        for c in np.unique(data.y_multi):
            idx = rng.permutation(np.flatnonzero(data.y_multi == c))
            a = (split[data.y_multi == c] == "train").sum()
            b = a + (split[data.y_multi == c] == "val").sum()
            new["train"].append(idx[:a])
            new["val"].append(idx[a:b])
            new["test"].append(idx[b:])
        ids = {k: np.sort(np.concatenate(v)) for k, v in new.items()}
        d2 = with_ids(data, **ids)
        model, secs = train(d2, seed)
        m = evaluate(model, d2, ids["test"], ids["val"])
        row = {"study": "split", "setting": "random", "seed": seed, **m, "train_seconds": secs}
        append(res, [row])
        print(f"split random seed {seed}: F1 {m['F1']:.4f} macro-F1 {m['macro_F1']:.4f}", flush=True)


def stratified_sample(rng, ids, y_multi, n):
    """Sample n of ids keeping each class's share (n is capped at len(ids))."""
    n = min(n, len(ids))
    cls, counts = np.unique(y_multi[ids], return_counts=True)
    take = np.floor(counts / counts.sum() * n).astype(int)
    take[np.argmax(counts)] += n - take.sum()
    return np.sort(np.concatenate([rng.choice(ids[y_multi[ids] == c], k, replace=False)
                                   for c, k in zip(cls, take)]))


def study_imbalance(args, res, data):
    """Train at 1:1 attack-to-normal; test at 1:1, 1:3, 1:5, 1:7."""
    y, ym = data.y, data.y_multi
    tr, te = data.ids["train"], data.ids["test"]
    for seed in SEEDS:
        if is_done(res, "imbalance", "train 1:1, test 1:1", seed):
            continue
        rng = np.random.default_rng(seed)
        normal_tr, attack_tr = tr[y[tr] == 0], tr[y[tr] == 1]
        n = min(len(normal_tr), len(attack_tr))
        new_tr = np.sort(np.concatenate([rng.choice(normal_tr, n, replace=False) if len(normal_tr) > n else normal_tr,
                                         stratified_sample(rng, attack_tr, ym, n) if len(attack_tr) > n else attack_tr]))
        d2 = with_ids(data, train=new_tr)
        model, secs = train(d2, seed)
        normal_te, attack_te = te[y[te] == 0], te[y[te] == 1]
        pv, _ = model.supervised(data.ids["val"])
        thr, _ = best_f1_threshold(y[data.ids["val"]], 1 - pv[:, 0])
        p_all, _ = model.supervised(te)
        score = dict(zip(te, 1 - p_all[:, 0]))
        rows = []
        for r in (1, 3, 5, 7):
            vals = []
            n_att = min(len(attack_te), len(normal_te) // r)
            n_norm = min(len(normal_te), n_att * r)
            for draw in range(5):
                rr = np.random.default_rng(1000 * seed + 10 * r + draw)
                att = stratified_sample(rr, attack_te, ym, n_att)
                nor = normal_te if n_norm == len(normal_te) else np.sort(rr.choice(normal_te, n_norm, replace=False))
                ids = np.concatenate([nor, att])
                s = np.array([score[i] for i in ids])
                m = binary_metrics(y[ids], s, thr)
                vals.append([m["Precision"], m["Recall"], m["F1"], average_precision_score(y[ids], s)])
            v = np.mean(vals, 0)
            rows.append({"study": "imbalance", "setting": f"train 1:1, test 1:{r}", "seed": seed,
                         "Precision": v[0], "Recall": v[1], "F1": v[2], "PR_AUC": v[3],
                         "test_attack": n_att, "test_normal": n_norm,
                         "train_windows": len(new_tr), "train_seconds": secs})
        append(res, rows)
        print(f"imbalance seed {seed}: " + ", ".join(f"1:{r} F1 {x['F1']:.4f}" for r, x in zip((1, 3, 5, 7), rows)),
              flush=True)


def study_classifier(args, res, data):
    """Swap the fusion head's classifier for standard models on the same fused embeddings."""
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from xgboost import XGBClassifier

    tr, va, te = data.ids["train"], data.ids["val"], data.ids["test"]
    y, ym = data.y, data.y_multi
    makers = {
        "MLP": lambda s: MLPClassifier((64,), max_iter=60, early_stopping=True, random_state=s),
        "SVM (RBF, 20k sample)": lambda s: SVC(kernel="rbf", probability=True, random_state=s),
        "Random Forest": lambda s: RandomForestClassifier(200, n_jobs=-1, random_state=s,
                                                          class_weight="balanced_subsample"),
        "XGBoost": lambda s: XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                           tree_method="hist", n_jobs=4, random_state=s),
        "LightGBM": lambda s: LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                                             class_weight="balanced", random_state=s, verbose=-1),
    }
    for seed in SEEDS:
        X = np.load(f"data/phase10/fused_seed{seed}.npy")
        for name, make in makers.items():
            if is_done(res, "classifier", name, seed):
                continue
            fit_ids = tr
            if name.startswith("SVM"):
                fit_ids = stratified_sample(np.random.default_rng(seed), tr, ym, 20_000)
            clf = make(seed)
            t0 = time.time()
            clf.fit(X[fit_ids], ym[fit_ids])
            fit_s = time.time() - t0
            t0 = time.time()
            pv, pt = clf.predict_proba(X[va]), clf.predict_proba(X[te])
            pred_s = time.time() - t0
            normal = list(clf.classes_).index(0)
            thr, _ = best_f1_threshold(y[va], 1 - pv[:, normal])
            m = binary_metrics(y[te], 1 - pt[:, normal], thr)
            macro = f1_score(ym[te], clf.classes_[pt.argmax(1)], average="macro")
            row = {"study": "classifier", "setting": name, "seed": seed, "F1": m["F1"],
                   "Precision": m["Precision"], "Recall": m["Recall"], "PR_AUC": m["PR_AUC"],
                   "macro_F1": macro, "fit_seconds": fit_s, "predict_seconds": pred_s,
                   "fit_windows": len(fit_ids)}
            append(res, [row])
            print(f"classifier {name} seed {seed}: F1 {m['F1']:.4f} macro-F1 {macro:.4f} "
                  f"fit {fit_s:.0f}s", flush=True)


# ---------------------------------------------------------------- report
def ms(df, col):
    g = df.groupby("setting", sort=False)[col]
    return g.mean(), g.std().fillna(0)


def fmt(mean, std):
    return [f"{a:.4f} ± {b:.4f}" for a, b in zip(mean, std)]


def two_panel(path, title, x_labels, left, left_err, left_label, right, right_label, x_title):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.0, 3.6), dpi=150)
    x = np.arange(len(x_labels))
    a.errorbar(x, left, yerr=left_err, fmt="o-", color=C1, markersize=6, linewidth=1.8,
               markeredgecolor="white", capsize=3)
    b.plot(x, right, "o-", color=C2, markersize=6, linewidth=1.8, markeredgecolor="white")
    for ax, lab in [(a, left_label), (b, right_label)]:
        ax.set_xticks(x, x_labels, fontsize=9)
        ax.set_xlabel(x_title)
        ax.set_ylabel(lab)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED)
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def report(res):
    r = load_runs(res)
    base = r[r.study == "baseline"].assign(setting=BASE)
    out = {}

    # training history
    h = pd.concat([base.assign(setting="full training period"), r[r.study == "history"]])
    if len(h):
        mf, sf = ms(h, "F1")
        mm, sm = ms(h, "macro_F1")
        t = pd.DataFrame({"Training history": mf.index, "F1": fmt(mf, sf), "Attack-type macro-F1": fmt(mm, sm),
                          "Train windows": h.groupby("setting", sort=False)["train_windows"].mean().reindex(mf.index).values,
                          "Near-empty window share": h.groupby("setting", sort=False)["near_empty_share"].mean().reindex(mf.index).values,
                          "seeds": h.groupby("setting", sort=False).size().reindex(mf.index).values})
        t.to_csv(res / "sparsity_results.csv", index=False)
        out["sparsity"] = t

    # leakage
    s = pd.concat([base.assign(setting="Sequential (LATS)"), r[r.study == "split"].assign(setting="Random")])
    if len(s):
        mf, sf = ms(s, "F1")
        mm, sm = ms(s, "macro_F1")
        t = pd.DataFrame({"Split protocol": mf.index, "F1": fmt(mf, sf), "Attack-type macro-F1": fmt(mm, sm),
                          "seeds": s.groupby("setting", sort=False).size().reindex(mf.index).values})
        if {"Sequential (LATS)", "Random"} <= set(mf.index):
            gap = mf["Random"] - mf["Sequential (LATS)"]
            gap_m = mm["Random"] - mm["Sequential (LATS)"]
            t["F1 gap (random - LATS)"] = ["", f"{gap:+.4f}"]
            t["macro-F1 gap (random - LATS)"] = ["", f"{gap_m:+.4f}"]
            fig, ax = plt.subplots(figsize=(5.6, 3.4), dpi=150)
            x = np.arange(2)
            ax.bar(x - 0.18, mf.values, 0.34, yerr=sf.values, color=C1, label="Attack-vs-normal F1", capsize=3)
            ax.bar(x + 0.18, mm.values, 0.34, yerr=sm.values, color=C2, label="Attack-type macro-F1", capsize=3)
            for xi, (a, b) in enumerate(zip(mf.values, mm.values)):
                ax.text(xi - 0.18, a + 0.01, f"{a:.3f}", ha="center", fontsize=8, color=INK)
                ax.text(xi + 0.18, b + 0.01, f"{b:.3f}", ha="center", fontsize=8, color=INK)
            ax.set_xticks(x, list(mf.index))
            ax.set_ylim(0, 1.1)
            ax.set_ylabel("Score on test split")
            ax.set_title("Sequential (LATS) vs random split", fontsize=10, color=INK, loc="left")
            ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=2)
            ax.grid(axis="y", color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            fig.tight_layout()
            fig.savefig(res / "leakage_comparison.png")
            plt.close(fig)
        t.to_csv(res / "leakage_results.csv", index=False)
        out["leakage"] = t

    # imbalance
    im = r[r.study == "imbalance"]
    if len(im):
        cols = {}
        for c in ["Precision", "Recall", "F1", "PR_AUC"]:
            m_, s_ = ms(im, c)
            cols[c] = fmt(m_, s_)
        t = pd.DataFrame({"Setting": m_.index, **cols,
                          "seeds": im.groupby("setting", sort=False).size().reindex(m_.index).values})
        t.to_csv(res / "imbalance_results.csv", index=False)
        out["imbalance"] = t

    # sensitivity: window, hop, classifier
    sens = []
    win = pd.concat([base.assign(setting="w3"), r[r.study == "window"]])
    hop = pd.concat([base.assign(setting="k1"), r[r.study == "hop"]])
    for study, df, key in [("window", win, "w"), ("hop", hop, "k")]:
        if df["setting"].nunique() < 1:
            continue
        df = df.assign(v=df["setting"].str.extract(r"(\d+)")[0].astype(int),
                       f16=df["setting"].str.endswith("f16")).sort_values(["v", "f16"])
        order = list(dict.fromkeys(df["setting"]))
        g = df.groupby("setting")
        for st in order:
            d = g.get_group(st)
            sens.append({"study": study, "setting": st, "F1": f"{d.F1.mean():.4f} ± {d.F1.std():.4f}",
                         "Attack-type macro-F1": f"{d.macro_F1.mean():.4f} ± {d.macro_F1.std():.4f}",
                         "val macro-F1": d.val_macro_F1.mean(),
                         "latency_ms_per_window": d.latency_ms_per_window.mean(),
                         "build_seconds": d.build_seconds.mean(), "mean_nodes": d.mean_nodes.mean(),
                         "mean_edges": d.mean_edges.mean(), "train_seconds": d.train_seconds.mean(),
                         "seeds": len(d)})
        mm = g["macro_F1"].mean().reindex(order)
        sm = g["macro_F1"].std().reindex(order).fillna(0)
        if study == "window":
            two_panel(res / "sensitivity_window.png", "Window size w: accuracy and latency",
                      [s[1:] for s in order], mm.values, sm.values, "Attack-type macro-F1",
                      g["latency_ms_per_window"].mean().reindex(order).values,
                      "Scoring latency (ms per window)", "Window size w (one-second slots)")
        else:
            two_panel(res / "sensitivity_hop.png", "Hop length k: accuracy and graph-building cost",
                      [s[1:].replace("f16", "\n(fp16)") for s in order], mm.values, sm.values, "Attack-type macro-F1",
                      g["build_seconds"].mean().reindex(order).values,
                      "Phase 4 build time (s, all windows)", "Hop length k")
    cl = r[r.study == "classifier"]
    if len(cl):
        default = base.assign(setting="Default (fusion head)", fit_seconds=base["train_seconds"],
                              predict_seconds=np.nan)
        cl = pd.concat([default, cl])
        order = list(dict.fromkeys(cl["setting"]))
        g = cl.groupby("setting")
        for st in order:
            d = g.get_group(st)
            sens.append({"study": "classifier", "setting": st,
                         "F1": f"{d.F1.mean():.4f} ± {d.F1.std():.4f}",
                         "Attack-type macro-F1": f"{d.macro_F1.mean():.4f} ± {d.macro_F1.std():.4f}",
                         "fit_seconds": d.fit_seconds.mean(), "predict_seconds": d.predict_seconds.mean(),
                         "seeds": len(d)})
        two_panel(res / "sensitivity_classifier.png", "Classifier on the fused embedding: accuracy and runtime",
                  [o.replace(" (RBF, 20k sample)", "\n(20k)").replace("Default (fusion head)", "Default")
                   .replace("Random Forest", "RF") for o in order],
                  g["macro_F1"].mean().reindex(order).values, g["macro_F1"].std().reindex(order).fillna(0).values,
                  "Attack-type macro-F1", g["fit_seconds"].mean().reindex(order).values,
                  "Training time (s)", "")
    if sens:
        t = pd.DataFrame(sens)
        t.to_csv(res / "sensitivity_results.csv", index=False)
        out["sensitivity"] = t
        choice = {}
        for study in ("window", "hop"):
            d = t[t.study == study]
            if len(d):
                best = d["val macro-F1"].max()
                # smallest setting within 0.002 of the best validation score
                pick = d[d["val macro-F1"] >= best - 0.002].iloc[0]["setting"]
                choice[study] = pick
        from sentinelfed.runlog import environment
        (res / "run_log_phase10.json").write_text(json.dumps({
            **environment(), "seeds": sorted(r["seed"].unique().tolist()), "model": "protocol-blind, supervised branch",
            "threshold_rule": "best F1 on validation", "runs_file": RUNS}, indent=2, default=str))
        (res / "phase10_choice.json").write_text(json.dumps({
            "rule": "smallest setting within 0.002 of the best mean validation macro-F1", **choice}, indent=2))
        out["choice"] = choice
    with pd.option_context("display.width", 250, "display.max_colwidth", 40):
        for k, v in out.items():
            print(f"--- {k}")
            print(v if isinstance(v, dict) else v.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=["baseline", "history", "split", "imbalance", "classifier",
                                        "window", "hop"])
    ap.add_argument("--setting", default=None)
    ap.add_argument("--phase4", default="data/phase4")
    ap.add_argument("--results", default="results")
    ap.add_argument("--build-seconds", type=float, default=np.nan)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    res = Path(args.results)
    if args.report:
        report(res)
        return
    if args.study == "classifier":
        study_classifier(args, res, load_phase4(args.phase4, drop=PROTOCOL_IDENTITY))
        return
    data = load_phase4(args.phase4, drop=PROTOCOL_IDENTITY)
    if args.study == "baseline":
        study_baseline(args, res, data)
    elif args.study in ("window", "hop"):
        study_baseline(args, res, data, setting=args.setting, study=args.study, save_emb=False)
    elif args.study == "history":
        study_history(args, res, data)
    elif args.study == "split":
        study_split(args, res, data)
    elif args.study == "imbalance":
        study_imbalance(args, res, data)


if __name__ == "__main__":
    main()
