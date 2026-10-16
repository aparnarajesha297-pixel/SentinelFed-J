"""Paper figures that are not produced by a phase script.

    python scripts/figures.py                 # all available
    python scripts/figures.py --only lats architecture pipeline comparison

  fig1_architecture.png   overall system
  fig2_lats_split.png     LATS split per recording, from results/split_cutpoints.csv
  fig3_pipeline.png       LC / CE / GC pipeline with the Phase 4 output shapes
  fig4_main_comparison.png  Phase 6 (protocol-blind) mean ± std per model
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BOX = "#eef3fb"      # light tint for boxes
BOX2 = "#fdf0e9"
BOX3 = "#e8f6f0"


def box(ax, x, y, w, h, text, fc=BOX, ec=BLUE, size=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x0, y0, x1, y1, color=MUTED, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=10,
                                 color=color, lw=1.2, linestyle=ls))


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def fig_architecture(res):
    fig, ax = canvas(10, 5.6)
    ax.text(0.0, 0.97, "SentinelFed-J: system overview", fontsize=11, color=INK, fontweight="bold")
    box(ax, 0.00, 0.62, 0.13, 0.16, "IIoT traffic\n(one site)", fc="#f3f2ef", ec=MUTED)
    box(ax, 0.17, 0.62, 0.15, 0.16, "Edge features\n+ windows\n(w one-second slots)")
    box(ax, 0.36, 0.78, 0.15, 0.11, "LC: ego graph GNN\n+ cross-attention")
    box(ax, 0.36, 0.62, 0.15, 0.11, "CE: masked\ntransformer")
    box(ax, 0.36, 0.46, 0.15, 0.11, "GC: activity\ncross-attention")
    box(ax, 0.55, 0.62, 0.13, 0.16, "Weighted\nattention\nfusion")
    box(ax, 0.72, 0.62, 0.13, 0.16, "Supervised\nclassifier\n(13 classes)")
    box(ax, 0.36, 0.22, 0.15, 0.13, "GRU autoencoder\n(normal only)", fc=BOX2, ec=ORANGE)
    box(ax, 0.55, 0.22, 0.13, 0.13, "Mahalanobis\ndistance", fc=BOX2, ec=ORANGE)
    box(ax, 0.88, 0.40, 0.12, 0.20, "Calibrated risk\n→ Trust /\nVerify /\nReject", fc=BOX3, ec=AQUA, bold=True)
    box(ax, 0.88, 0.10, 0.12, 0.14, "Hash-chained\nledger", fc="#f3f2ef", ec=MUTED)
    arrow(ax, 0.13, 0.70, 0.17, 0.70)
    for y in (0.835, 0.675, 0.515):
        arrow(ax, 0.32, 0.70, 0.36, y)
        arrow(ax, 0.51, y, 0.55, 0.70)
    arrow(ax, 0.435, 0.78, 0.435, 0.73)        # LC output feeds CE
    arrow(ax, 0.68, 0.70, 0.72, 0.70)
    arrow(ax, 0.85, 0.70, 0.90, 0.60)
    arrow(ax, 0.32, 0.66, 0.36, 0.30, color=ORANGE)
    arrow(ax, 0.51, 0.285, 0.55, 0.285, color=ORANGE)
    arrow(ax, 0.68, 0.285, 0.88, 0.45, color=ORANGE)
    arrow(ax, 0.94, 0.40, 0.94, 0.24)
    ax.text(0.36, 0.93, "Shared with other sites through reputation-weighted federated learning: LC and CE",
            fontsize=8, color=BLUE)
    ax.text(0.375, 0.395, "Kept local at each site: GC and the normal-behaviour map", fontsize=8, color=MUTED)
    ax.text(0.0, 0.03, "Blue: supervised branch.  Orange: unsupervised branch.  Green: decision.  "
            "Gray: input and audit trail.", fontsize=7.5, color=MUTED)
    fig.savefig(res / "fig1_architecture.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_lats(res):
    c = pd.read_csv(res / "split_cutpoints.csv")
    c = c.sort_values(["Attack_type", "segment_id"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.6, 6.4), dpi=150)
    for i, r in c.iterrows():
        span = r["end_s"] - r["start_s"]
        a = (r["val_from_s"] - r["start_s"]) / span
        b = (r["test_from_s"] - r["start_s"]) / span
        y = len(c) - 1 - i
        ax.barh(y, a, left=0, color=BLUE, height=0.62)
        ax.barh(y, b - a, left=a, color=ORANGE, height=0.62)
        ax.barh(y, 1 - b, left=b, color=AQUA, height=0.62)
        mins = span / 60
        ax.text(1.01, y, f"{mins:,.0f} min" if mins >= 1 else f"{span:.0f} s", va="center",
                fontsize=7.5, color=MUTED)
    ax.set_yticks(range(len(c)), [f"{t} ({s})" for t, s in zip(c["Attack_type"], c["segment_id"])][::-1],
                  fontsize=8)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["start", "25%", "50%", "75%", "end"])
    ax.set_xlabel("Position within each recording's own time range")
    ax.set_title("LATS: every recording is split on its own clock (train 70%, val 15%, test 15% of records)",
                 fontsize=9.5, color=INK, loc="left")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="S_train (U_train = its normal windows)"),
                       Patch(color=ORANGE, label="S_val (U_val = its normal windows)"),
                       Patch(color=AQUA, label="S_test (U_test = its attack windows)")],
              frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    fig.savefig(res / "fig2_lats_split.png", facecolor="white")
    plt.close(fig)


def fig_pipeline(res):
    raw = pd.read_csv(res / "phase4_shapes.csv").set_index("component")["shape"]
    sh = raw.map(lambda v: "x".join(f"{int(p):,}" for p in str(v).split("x")))
    fig, ax = canvas(10, 3.4)
    ax.text(0.0, 0.94, "LC / CE / GC pipeline (shapes from the full Phase 4 build)", fontsize=10.5,
            color=INK, fontweight="bold")
    steps = [
        ("Records", f"edge features\n{sh['features.csv'].replace('x', ' × ')}"),
        ("Device-windows", f"w = 3 slots\n{sh['windows'].split('x')[0]} windows"),
        ("Ego graphs", f"k = 1\n{sh['ego graphs: node features'].split('x')[0]} nodes"),
        ("LC", f"h_L\n{sh['LC h_L'].replace('x', ' × ')}"),
        ("CE", f"h_C, last 8 h_L\n{sh['CE h_C'].replace('x', ' × ')}"),
        ("Fusion", f"h\n{sh['Fusion h'].replace('x', ' × ')}"),
    ]
    w, gap, y = 0.135, 0.032, 0.42
    for i, (title, body) in enumerate(steps):
        x = i * (w + gap)
        box(ax, x, y, w, 0.34, f"{title}\n{body}", size=8)
        if i:
            arrow(ax, x - gap, y + 0.17, x, y + 0.17)
    gx = 3 * (w + gap)
    box(ax, gx, 0.02, w, 0.28, f"GC\nh_G\n{sh['GC h_G'].replace('x', ' × ')}", size=8)
    box(ax, 2 * (w + gap), 0.02, w, 0.28,
        f"Activity matrix\n{sh['activity: other devices'].split('x')[0]} windows\n× 32 others × 16 slots", size=8)
    arrow(ax, 2 * (w + gap) + w, 0.16, gx, 0.16)
    arrow(ax, gx + w, 0.16, 5 * (w + gap) + 0.02, y, color=MUTED)
    fig.savefig(res / "fig3_pipeline.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_comparison(res):
    p = res / "experiment1_seedwise_protocol_blind.csv"
    sw = pd.read_csv(p)
    order = list(dict.fromkeys(sw["Model"]))
    g = sw.groupby("Model")
    short = {m: (m.replace(" (window features)", "").replace(" (GRU-AE + Mahalanobis)", "")
                 .replace(" (LC+CE+GC+Fusion)", "").replace("Full SentinelFed-J (Dual Head, proposed)",
                                                             "Full SentinelFed-J (proposed)")
                 .replace("Depth-1 tree, best single feature", "Depth-1 tree"))
             for m in order}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), dpi=150, sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, col, title, c in [(axes[0], "F1", "Attack-vs-normal F1", BLUE),
                              (axes[1], "macro_F1_attack_type", "Attack-type macro-F1", ORANGE)]:
        mean = g[col].mean().reindex(order)
        std = g[col].std().reindex(order).fillna(0)
        ok = mean.notna().to_numpy()
        ax.errorbar(mean[ok], y[ok], xerr=std[ok], fmt="o", color=c, markersize=6,
                    markeredgecolor="white", ecolor=c, capsize=3)
        for yi, (m, s) in zip(y[ok], zip(mean[ok], std[ok])):
            ax.text(m, yi + 0.28, f"{m:.3f}", ha="center", fontsize=7.5, color=INK)
        for yi in y[~ok]:
            ax.text(0.5, yi, "n/a (no attack types)", fontsize=7.5, color=MUTED, va="center",
                    transform=ax.get_yaxis_transform())
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED)
        lo = np.nanmin(mean - std)
        ax.set_xlim(max(0, lo - 0.03), 1.005)
    axes[0].set_yticks(y, [short[m] for m in order], fontsize=8.5)
    n = int(g.size().max())
    fig.suptitle(f"Main comparison on S_test, protocol-blind features (mean ± std over up to {n} seeds)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(res / "fig4_main_comparison.png", facecolor="white")
    plt.close(fig)


FIGS = {"architecture": fig_architecture, "lats": fig_lats, "pipeline": fig_pipeline,
        "comparison": fig_comparison}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--only", nargs="*", default=list(FIGS))
    args = ap.parse_args()
    for name in args.only:
        try:
            FIGS[name](Path(args.results))
            print(f"{name}: ok")
        except FileNotFoundError as e:
            print(f"{name}: skipped ({e.filename} not found yet)")


if __name__ == "__main__":
    main()
