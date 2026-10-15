"""Phase 5: dual detection head and the Trust / Verify / Reject policy.

    python scripts/phase5_train.py                    # full data (data/phase4)
    python scripts/phase5_train.py --phase4 data/phase4_dev --tag _dev

Order follows the guide's build order: the unsupervised branch first, then
the supervised branch, then the combined risk score and decision policy.
Every choice here (epochs, mixing weight, calibration, thresholds) uses
training and validation windows only. Test windows are not scored.
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score  # noqa: E402

from sentinelfed.detector import (REJECT, TRUST, VERIFY, EmpiricalScaler, Mahalanobis,  # noqa: E402
                                  RiskModel, best_f1_threshold, decide, latent,
                                  train_autoencoder)
from sentinelfed.train import predict_head, predict_lc, train_head, train_lc  # noqa: E402

NORMAL_C, ATTACK_C = "#2a78d6", "#eb6834"
REGION_C = {TRUST: "#0ca30c", VERIFY: "#fab219", REJECT: "#d03b3b"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
UNSUP_FPR = 0.01      # unsupervised threshold: 1% false alarms on U_val


def style(ax):
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def plot_risk(risk, y, path, title):
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    bins = np.linspace(0, 1, 41)
    for mask, name, c in [(y == 0, "Normal", NORMAL_C), (y == 1, "Attack", ATTACK_C)]:
        ax.hist(risk[mask], bins=bins, color=c, alpha=0.75, label=f"{name} (n={mask.sum():,})",
                edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("Calibrated risk")
    ax.set_ylabel("Windows (log scale)")
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_disagreement(p_sup, d_norm, y, tau_s, tau_u, path, title):
    s, u = p_sup >= tau_s, d_norm >= tau_u
    groups = {"Both normal": ~s & ~u, "Supervised only": s & ~u,
              "Unsupervised only": ~s & u, "Both attack": s & u}
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    x = np.arange(len(groups))
    n_norm = [int((m & (y == 0)).sum()) for m in groups.values()]
    n_att = [int((m & (y == 1)).sum()) for m in groups.values()]
    ax.bar(x - 0.2, n_norm, 0.38, color=NORMAL_C, label="Normal")
    ax.bar(x + 0.2, n_att, 0.38, color=ATTACK_C, label="Attack")
    for xi, a, b in zip(x, n_norm, n_att):
        ax.text(xi - 0.2, max(a, 1) * 1.15, f"{a:,}", ha="center", fontsize=8, color=MUTED)
        ax.text(xi + 0.2, max(b, 1) * 1.15, f"{b:,}", ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(x, list(groups), fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("Windows (log scale)")
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_regions(p_sup, d_norm, y, tau_s, tau_u, path, title, max_points=6000, seed=0):
    fig, ax = plt.subplots(figsize=(5.6, 5.0), dpi=150)
    regions = [((0, tau_s), (0, tau_u), TRUST), ((tau_s, 1), (tau_u, 1), REJECT),
               ((tau_s, 1), (0, tau_u), VERIFY), ((0, tau_s), (tau_u, 1), VERIFY)]
    for (x0, x1), (y0, y1), name in regions:
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color=REGION_C[name],
                                   alpha=0.12, linewidth=0))
        if (x1 - x0) > 0.08 and (y1 - y0) > 0.06:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, name, ha="center", va="center",
                    fontsize=11, color=INK, alpha=0.8, fontweight="bold")
    rng = np.random.default_rng(seed)
    for lab, name, c in [(0, "Normal", NORMAL_C), (1, "Attack", ATTACK_C)]:
        idx = np.flatnonzero(y == lab)
        idx = rng.choice(idx, min(len(idx), max_points // 2), replace=False)
        ax.scatter(p_sup[idx], d_norm[idx], s=9, color=c, alpha=0.55, linewidths=0.4,
                   edgecolors="white", label=f"{name} (sample of {len(idx):,})")
    ax.axvline(tau_s, color=MUTED, linewidth=1, linestyle="--")
    ax.axhline(tau_u, color=MUTED, linewidth=1, linestyle="--")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(f"Supervised attack score (threshold {tau_s:.3f})")
    ax.set_ylabel(f"Unsupervised distance, scaled (threshold {tau_u:.2f})")
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase4", default="data/phase4")
    ap.add_argument("--out", default="data/phase5")
    ap.add_argument("--artifacts", default="artifacts/phase5")
    ap.add_argument("--results", default="results")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lc-epochs", type=int, default=3)
    ap.add_argument("--head-epochs", type=int, default=6)
    ap.add_argument("--ae-epochs", type=int, default=5)
    args = ap.parse_args()
    p4, out = Path(args.phase4), Path(args.out + args.tag)
    art, res = Path(args.artifacts + args.tag), Path(args.results)
    for p in (out, art, res):
        p.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    torch.manual_seed(args.seed)
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    W = pickle.load(open(p4 / "windows.pkl", "rb"))
    ego = pickle.load(open(p4 / "ego_graphs.pkl", "rb"))
    act = pickle.load(open(p4 / "activity_flags.pkl", "rb"))
    windows, seq_idx, center = W["windows"], W["seq_idx"], W["center_feat"]
    cfg = W["config"]

    classes = ["Normal"] + sorted(c for c in windows["attack_type"].unique() if c != "Normal")
    y_multi = windows["attack_type"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    y = windows["label"].to_numpy().astype(int)
    split = windows["split"].to_numpy()
    tr, va = np.flatnonzero(split == "train"), np.flatnonzero(split == "val")
    u_tr, u_va = tr[y[tr] == 0], va[y[va] == 0]
    log(f"windows: train {len(tr):,}, val {len(va):,}; U_train {len(u_tr):,}, U_val {len(u_va):,}; "
        f"{len(classes)} classes")

    # ---- unsupervised branch -------------------------------------------------
    log("unsupervised branch (GRU autoencoder on U_train)")
    ae, ae_val = train_autoencoder(center, seq_idx, u_tr, u_va, epochs=args.ae_epochs,
                                   seed=args.seed, log=log)
    fit_val = np.concatenate([tr, va])
    z_tr = latent(ae, center, seq_idx, u_tr)
    maha = Mahalanobis.fit(z_tr, ridge=1e-3)
    scaler = EmpiricalScaler(maha(latent(ae, center, seq_idx, u_va)))
    z_tv = latent(ae, center, seq_idx, fit_val)
    d_norm = np.full(len(windows), np.nan)
    d_norm[fit_val] = scaler(maha(z_tv))
    tau_u = 1.0 - UNSUP_FPR
    log(f"  val PR-AUC (unsupervised alone) {average_precision_score(y[va], d_norm[va]):.4f}")

    # ---- supervised branch -----------------------------------------------------
    log("supervised branch, stage A: LC")
    lc_model, lc_f1 = train_lc(ego, windows, y_multi, tr, va, len(classes),
                               epochs=args.lc_epochs, seed=args.seed, log=log)
    all_ids = np.arange(len(windows))
    h_L = torch.from_numpy(predict_lc(lc_model, ego, windows, all_ids, embed=True))
    log("supervised branch, stage B: CE + GC + fusion + classifier")
    head, head_f1 = train_head(h_L, seq_idx, act, y_multi, tr, va, len(classes),
                               cfg["seq_len"], cfg["pattern_len"], epochs=args.head_epochs,
                               seed=args.seed, log=log)
    prob_va, w_va = predict_head(head, h_L, seq_idx, act, va)
    p_sup = np.full(len(windows), np.nan)
    p_sup[va] = 1.0 - prob_va[:, 0]
    tau_s, f1_s = best_f1_threshold(y[va], p_sup[va])

    # ---- risk score and decision policy (validation only) ------------------
    risk_model = RiskModel.fit(p_sup[va], d_norm[va], y[va])
    risk_va = risk_model(p_sup[va], d_norm[va])
    dec = decide(p_sup[va], d_norm[va], tau_s, tau_u)
    dec_nolab = decide(p_sup[va], d_norm[va], tau_s, tau_u, labels_available=False)
    base_rate = y[va].mean()

    rows = []
    for name, d in [("labels available", dec), ("no labels", dec_nolab)]:
        for k in (TRUST, VERIFY, REJECT):
            m = d == k
            rows.append({"policy": name, "decision": k, "windows": int(m.sum()),
                         "share": m.mean(), "attack_rate": float(y[va][m].mean()) if m.any() else np.nan})
    policy = pd.DataFrame(rows)

    def f1_at(score, t):
        return f1_score(y[va], score >= t, zero_division=0)

    metrics = {
        "val_windows": int(len(va)), "val_attack_rate": float(base_rate),
        "supervised": {"tau": tau_s, "F1": f1_s, "PR_AUC": average_precision_score(y[va], p_sup[va]),
                       "ROC_AUC": roc_auc_score(y[va], p_sup[va]),
                       "macro_F1_multiclass": f1_score(y_multi[va], prob_va.argmax(1), average="macro")},
        "unsupervised": {"tau": tau_u, "F1": f1_at(d_norm[va], tau_u),
                         "PR_AUC": average_precision_score(y[va], d_norm[va]),
                         "ROC_AUC": roc_auc_score(y[va], d_norm[va]),
                         "false_alarm_rate_U_val": float((d_norm[u_va] >= tau_u).mean())},
        "combined_risk": {"alpha_supervised": risk_model.alpha,
                          "PR_AUC": average_precision_score(y[va], risk_va),
                          "ROC_AUC": roc_auc_score(y[va], risk_va),
                          "F1_at_0.5": f1_at(risk_va, 0.5)},
        "fusion_mean_weights_val": dict(zip(head.views, w_va.mean(0).round(4).tolist())),
        "fusion_weights_val_by_class": {
            c: dict(zip(head.views, w_va[y_multi[va] == i].mean(0).round(3).tolist()))
            for i, c in enumerate(classes) if (y_multi[va] == i).any()},
    }

    # ---- plots (validation data) ---------------------------------------------
    sub = f"S_val, {len(va):,} windows"
    plot_risk(risk_va, y[va], res / f"risk_distribution{args.tag}.png",
              f"Calibrated risk by true class ({sub})")
    plot_disagreement(p_sup[va], d_norm[va], y[va], tau_s, tau_u,
                      res / f"disagreement_distribution{args.tag}.png",
                      f"Where the two branches agree and disagree ({sub})")
    plot_regions(p_sup[va], d_norm[va], y[va], tau_s, tau_u, res / f"decision_regions{args.tag}.png",
                 f"Trust / Verify / Reject regions ({sub})", seed=args.seed)

    # ---- save models and embeddings ------------------------------------------
    torch.save({"state": lc_model.state_dict(), "node_dim": ego["node_feat"].shape[1],
                "edge_dim": ego["edge_attr"].shape[1], "classes": classes}, art / "lc.pt")
    torch.save({"state": head.state_dict(), "views": head.views, "classes": classes,
                "seq_len": cfg["seq_len"], "pattern_len": cfg["pattern_len"]}, art / "head.pt")
    torch.save({"state": ae.state_dict(), "in_dim": center.shape[1]}, art / "gru_ae.pt")
    joblib.dump({"mahalanobis": maha, "u_val_scaler": scaler, "risk_model": risk_model,
                 "tau_supervised": tau_s, "tau_unsupervised": tau_u, "classes": classes},
                art / "decision.joblib")
    _, _, emb = predict_head(head, h_L, seq_idx, act, all_ids, embeddings=True)
    wid = windows["window_id"].to_numpy()
    for name, key in [("lc", "LC"), ("ce", "CE"), ("gc", "GC"), ("fused", "fused")]:
        with open(out / f"{name}_embeddings.pkl", "wb") as f:
            pickle.dump({"window_id": wid, "embedding": emb[key]}, f)

    thresholds = {"tau_supervised": tau_s, "tau_unsupervised": tau_u,
                  "tau_supervised_rule": "maximises F1 of the supervised score on S_val",
                  "tau_unsupervised_rule": f"{UNSUP_FPR:.0%} false alarms on U_val "
                                           "(usable at sites with no labels)",
                  "seed": args.seed, "seconds": round(time.time() - t0, 1),
                  "metrics_val": metrics}
    (res / f"phase5_thresholds{args.tag}.json").write_text(json.dumps(thresholds, indent=2, default=float))
    policy.to_csv(res / f"phase5_policy_val{args.tag}.csv", index=False)
    (res / f"phase5_train{args.tag}.log.txt").write_text("\n".join(log_lines))
    log(json.dumps(metrics, indent=2, default=float))
    log(policy.to_string(index=False))
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
