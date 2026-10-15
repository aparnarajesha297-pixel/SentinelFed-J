"""Phase 7: ablation study on the protocol-blind model.

    python scripts/phase7_ablation.py

Per seed, LC is trained once; the fusion head is then retrained for every
subset of views, so variants differ only in which views are fused. "CE only"
still reads the LC embeddings, because CE's input is the sequence of h_L
(guide, Phase 4); it drops LC as a separate view, not as CE's input.

The headline metric is attack-type macro-F1 (Phase 6 showed attack-vs-normal
detection is near ceiling on this data). Thresholds come from S_val.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from sentinelfed.detector import (EmpiricalScaler, Mahalanobis, RiskModel,  # noqa: E402
                                  best_f1_threshold, latent, train_autoencoder)
from sentinelfed.pipeline import PROTOCOL_IDENTITY, binary_metrics, load_phase4  # noqa: E402
from sentinelfed.train import predict_head, predict_lc, train_head, train_lc  # noqa: E402

VARIANTS = [
    ("LC only", ("LC",), "Local neighbourhood contribution"),
    ("CE only", ("CE",), "Temporal-context contribution"),
    ("GC only", ("GC",), "Global-context contribution"),
    ("LC + CE", ("LC", "CE"), "Local + temporal"),
    ("LC + GC", ("LC", "GC"), "Local + global"),
    ("LC + CE + GC", ("LC", "CE", "GC"), "Multi-view supervised representation"),
    ("Full + Dual Head", ("LC", "CE", "GC"), "Complete model"),
]
# (variant, the variant it adds one piece to, the piece)
STEPS = [("LC + CE", "LC only", "+ CE"), ("LC + GC", "LC only", "+ GC"),
         ("LC + CE + GC", "LC + CE", "+ GC"), ("LC + CE + GC", "LC + GC", "+ CE"),
         ("Full + Dual Head", "LC + CE + GC", "+ unsupervised branch")]
BAR_C, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"


def plot(table, path):
    t = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.6, 3.9), dpi=150)
    y = np.arange(len(t))
    ax.barh(y, t["macro_F1_mean"], xerr=t["macro_F1_std"], color=BAR_C, height=0.6,
            error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 3})
    for yi, (m, s) in enumerate(zip(t["macro_F1_mean"], t["macro_F1_std"])):
        ax.text(m + s + 0.004, yi, f"{m:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, t["Variant"], fontsize=9)
    lo = max(0.0, t["macro_F1_mean"].min() - t["macro_F1_std"].max() - 0.05)
    ax.set_xlim(lo, 1.0)
    ax.set_xlabel("Attack-type macro-F1 on S_test (mean ± std over seeds)")
    ax.set_title("Ablation: which views help (protocol-blind model)", fontsize=10, color=INK, loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase4", default="data/phase4")
    ap.add_argument("--results", default="results")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--all-features", action="store_true", help="skip the protocol-blind mask")
    args = ap.parse_args()
    res = Path(args.results)
    t0 = time.time()
    data = load_phase4(args.phase4, drop=None if args.all_features else PROTOCOL_IDENTITY)
    ids = data.ids
    va, te = ids["val"], ids["test"]
    n_cls = len(data.classes)
    rows = []

    for seed in args.seeds:
        print(f"=== seed {seed} ({time.time() - t0:.0f}s)", flush=True)
        torch.manual_seed(seed)
        ae, _ = train_autoencoder(data.center, data.seq_idx, ids["u_train"], ids["u_val"],
                                  seed=seed, log=lambda m: None)
        maha = Mahalanobis.fit(latent(ae, data.center, data.seq_idx, ids["u_train"]))
        scaler = EmpiricalScaler(maha(latent(ae, data.center, data.seq_idx, ids["u_val"])))
        d_va = scaler(maha(latent(ae, data.center, data.seq_idx, va)))
        d_te = scaler(maha(latent(ae, data.center, data.seq_idx, te)))

        lc, _ = train_lc(data.ego, data.windows, data.y_multi, ids["train"], va, n_cls,
                         seed=seed, log=lambda m: None)
        h_L = torch.from_numpy(predict_lc(lc, data.ego, data.windows,
                                          np.arange(len(data.windows)), embed=True))
        heads = {}
        for name, views, purpose in VARIANTS:
            if views not in heads:
                heads[views], _ = train_head(h_L, data.seq_idx, data.act, data.y_multi, ids["train"],
                                             va, n_cls, data.cfg["seq_len"], data.cfg["pattern_len"],
                                             views=views, seed=seed, log=lambda m: None)
            head = heads[views]
            p_va, _ = predict_head(head, h_L, data.seq_idx, data.act, va)
            p_te, w_te = predict_head(head, h_L, data.seq_idx, data.act, te)
            s_va, s_te = 1 - p_va[:, 0], 1 - p_te[:, 0]
            if name == "Full + Dual Head":
                risk = RiskModel.fit(s_va, d_va, data.y[va])
                s_va, s_te = risk(s_va, d_va), risk(s_te, d_te)
            thr, _ = best_f1_threshold(data.y[va], s_va)
            m = binary_metrics(data.y[te], s_te, thr)
            macro = f1_score(data.y_multi[te], p_te.argmax(1), average="macro")
            row = {"Variant": name, "seed": seed, "Precision": m["Precision"], "Recall": m["Recall"],
                   "F1": m["F1"], "PR_AUC": m["PR_AUC"], "macro_F1_attack_type": macro,
                   "Purpose": purpose}
            if len(views) > 1:
                row["fusion_weights"] = json.dumps(dict(zip(views, w_te.mean(0).round(3).tolist())))
            rows.append(row)
            print(f"  {name:18s} seed {seed}: F1 {m['F1']:.4f}  attack-type macro-F1 {macro:.4f}",
                  flush=True)
        pd.DataFrame(rows).to_csv(res / "ablation_seedwise.csv", index=False)

    sw = pd.DataFrame(rows)
    order = [v[0] for v in VARIANTS]
    g = sw.groupby("Variant")
    table = pd.DataFrame({"Variant": order})
    for m in ["Precision", "Recall", "F1", "macro_F1_attack_type"]:
        mean, std = g[m].mean().reindex(order), g[m].std().reindex(order).fillna(0)
        table[m] = [f"{a:.4f} ± {b:.4f}" for a, b in zip(mean, std)]
    table["macro_F1_mean"] = g["macro_F1_attack_type"].mean().reindex(order).values
    table["macro_F1_std"] = g["macro_F1_attack_type"].std().reindex(order).fillna(0).values
    table["Purpose"] = [v[2] for v in VARIANTS]
    table["seeds"] = g.size().reindex(order).values
    table.to_csv(res / "ablation_results.csv", index=False)

    # incremental change from adding one piece, paired by seed
    inc = []
    piv = sw.pivot(index="seed", columns="Variant", values="macro_F1_attack_type")
    piv_f1 = sw.pivot(index="seed", columns="Variant", values="F1")
    for to, frm, piece in STEPS:
        d, d1 = piv[to] - piv[frm], piv_f1[to] - piv_f1[frm]
        inc.append({"from": frm, "to": to, "added": piece,
                    "delta_macro_F1_mean": d.mean(), "delta_macro_F1_std": d.std(),
                    "seeds_improved": int((d > 0).sum()), "seeds": len(d),
                    "delta_binary_F1_mean": d1.mean()})
    inc = pd.DataFrame(inc)
    inc.to_csv(res / "ablation_increments.csv", index=False)
    plot(table, res / "ablation.png")
    (res / "run_log_phase7.json").write_text(json.dumps({
        "seeds": args.seeds, "protocol_blind": not args.all_features,
        "dropped_features": data.dropped, "seconds": round(time.time() - t0)}, indent=2))
    with pd.option_context("display.width", 250):
        print(table.drop(columns=["macro_F1_mean", "macro_F1_std"]).to_string(index=False))
        print(inc.round(4).to_string(index=False))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
